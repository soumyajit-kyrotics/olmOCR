"""
generate_html.py — Convert PDF pages to faithful HTML using a Vision-Language Model.

Supports:
  - OpenRouter API (default): uses Qwen 2.5 VL via cloud
  - Local vLLM server: for running quantized models locally

This is the first step of OLMoCR v2's synthetic data pipeline.
The generated HTML becomes ground truth for extracting unit tests.
"""

import argparse
import base64
import glob
import json
import os
import random
import re
import sys
import time
import threading
import traceback as _traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests
from dotenv import load_dotenv

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.render import render_pdf_to_base64png


# ---------------------------------------------------------------------------
# API defaults (overridable via CLI)
# ---------------------------------------------------------------------------

DEFAULT_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL   = "qwen/qwen3-vl-235b-a22b-thinking"

# Alibaba's free tier for Qwen models is typically free up to ~32k total tokens
# (prompt + completion). Above this threshold, dynamic pricing kicks in and the
# request is no longer free even if the model is listed as $0/M tokens.
# We warn at 28k to give a safety buffer.
FREE_TOKEN_THRESHOLD = 28_000  # tokens; warn when total context exceeds this


# ---------------------------------------------------------------------------
# Progress / Error Logger
# ---------------------------------------------------------------------------

class PipelineLogger:
    """
    Thread-safe logger that writes to two files in <output_dir>/logs/:

      progress.log  — one line per event: page skipped/done/failed, PDF summary,
                      session start/end. Easy to tail -f during a long run.
      errors.log    — full traceback for every failed page so you can diagnose
                      what went wrong without digging through stdout.

    All lines are timestamped (ISO-8601, local time).  The logger is safe to
    call from multiple threads simultaneously.
    """

    def __init__(self, output_dir: str):
        log_dir = os.path.join(output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        self._progress_path = os.path.join(log_dir, "progress.log")
        self._error_path    = os.path.join(log_dir, "errors.log")
        self._output_dir    = output_dir          # needed for inventory scan
        self._lock = threading.Lock()
        # Session-level counters (updated by process_pdf)
        self.pdfs_processed  = 0
        self.pdfs_skipped    = 0   # PDFs where every page was already done
        self.pages_done      = 0   # newly generated this session
        self.pages_skipped   = 0   # already existed on disk
        self.pages_failed    = 0   # VLM returned None or raised

    # ── Public API ────────────────────────────────────────────────────

    def inventory(self) -> tuple[int, int]:
        """Scan the html/ output dir and return (pages_done, pdfs_done).

        Counts .html files (= pages) and unique PDF base names (= PDFs).
        Safe to call before processing starts; returns (0, 0) if dir is empty.
        """
        html_dir = os.path.join(self._output_dir, "html")
        if not os.path.exists(html_dir):
            return 0, 0
        html_files = [f for f in os.listdir(html_dir) if f.endswith(".html")]
        pages = len(html_files)
        # Strip the "_page_N.html" suffix to get unique PDF names
        pdf_names = {re.sub(r"_page_\d+\.html$", "", f) for f in html_files}
        return pages, len(pdf_names)

    def session_start(self, num_pdfs: int, output_dir: str, model: str, workers: int):
        pages_done, pdfs_done = self.inventory()
        resume_line = (
            f"  Resume   : {pages_done} pages already done "
            f"across {pdfs_done} PDF(s) from previous session(s)\n"
            if pages_done > 0
            else "  Resume   : fresh start (no previous output found)\n"
        )
        msg = (
            f"=== SESSION START ===\n"
            f"  Time     : {self._now()}\n"
            f"  PDFs     : {num_pdfs} total in queue\n"
            f"{resume_line}"
            f"  Output   : {output_dir}\n"
            f"  Model    : {model}\n"
            f"  Workers  : {workers} per PDF\n"
        )
        self._write_progress(msg)
        # Also print resume status to stdout so it's visible immediately
        if pages_done > 0:
            print(
                f"\n[RESUME] {pages_done} pages already done across "
                f"{pdfs_done} PDF(s) from previous session(s). "
                f"Skipping completed pages automatically.\n"
            )
        else:
            print("[START] Fresh run — no previous output found.\n")

    def session_end(self, elapsed: float):
        msg = (
            f"=== SESSION END ===\n"
            f"  Time          : {self._now()}\n"
            f"  Elapsed       : {elapsed:.1f}s ({elapsed/60:.1f} min)\n"
            f"  PDFs processed: {self.pdfs_processed}  "
               f"(+{self.pdfs_skipped} fully skipped)\n"
            f"  Pages done    : {self.pages_done}\n"
            f"  Pages skipped : {self.pages_skipped}\n"
            f"  Pages failed  : {self.pages_failed}\n"
        )
        self._write_progress(msg)
        print(msg)

    def pdf_start(self, pdf_path: str, num_pages: int, workers: int):
        self._write_progress(
            f"[PDF START] {os.path.basename(pdf_path)}  "
            f"({num_pages} pages, {workers} workers)"
        )

    def pdf_done(self, pdf_path: str, done: int, skipped: int, failed: int, elapsed: float):
        with self._lock:
            self.pdfs_processed += 1
            if done == 0 and failed == 0:
                self.pdfs_skipped += 1
        self._write_progress(
            f"[PDF DONE ] {os.path.basename(pdf_path)}  "
            f"done={done} skipped={skipped} failed={failed}  "
            f"elapsed={elapsed:.1f}s"
        )

    def page_skipped(self, pdf_name: str, page_num: int):
        with self._lock:
            self.pages_skipped += 1
        self._write_progress(f"[SKIP ] {pdf_name} page {page_num}")

    def page_done(self, pdf_name: str, page_num: int, html_len: int, elapsed: float,
                  usage: dict | None = None):
        with self._lock:
            self.pages_done += 1
        usage_str = ""
        if usage:
            usage_str = (
                f"  prompt={usage.get('prompt_tokens',0)} "
                f"completion={usage.get('completion_tokens',0)} "
                f"total={usage.get('total_tokens',0)} "
                f"provider={usage.get('provider','?')}"
            )
            # Extra warning in log if over threshold
            if usage.get('total_tokens', 0) >= FREE_TOKEN_THRESHOLD:
                usage_str += f"  ⚠ OVER FREE THRESHOLD ({FREE_TOKEN_THRESHOLD})"
        self._write_progress(
            f"[DONE ] {pdf_name} page {page_num}  "
            f"{html_len} chars  {elapsed:.1f}s"
            + usage_str
        )

    def page_failed(self, pdf_name: str, page_num: int, reason: str, exc: Exception = None):
        with self._lock:
            self.pages_failed += 1
        self._write_progress(f"[FAIL ] {pdf_name} page {page_num}  {reason}")
        # Full traceback → errors.log
        tb = _traceback.format_exc() if exc else ""
        self._write_error(
            f"--- {self._now()} ---\n"
            f"PDF : {pdf_name}  Page: {page_num}\n"
            f"Reason: {reason}\n"
            f"{tb}\n"
        )
        # Dead-letter queue → failed_pages.txt (tab-separated, easy to grep/retry)
        # Format: timestamp <TAB> pdf_name <TAB> page_num <TAB> reason
        dlq_path = os.path.join(os.path.dirname(self._progress_path), "failed_pages.txt")
        with self._lock:
            with open(dlq_path, "a", encoding="utf-8") as f:
                f.write(f"{self._now()}\t{pdf_name}\t{page_num}\t{reason}\n")

    def running_totals(self) -> str:
        """Human-readable one-liner for printing to stdout."""
        return (
            f"Progress → PDFs: {self.pdfs_processed} processed, "
            f"Pages: {self.pages_done} done / "
            f"{self.pages_skipped} skipped / "
            f"{self.pages_failed} failed"
        )

    # ── Internals ─────────────────────────────────────────────────────

    @staticmethod
    def _now() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _write_progress(self, msg: str):
        line = f"[{self._now()}] {msg}\n"
        with self._lock:
            with open(self._progress_path, "a", encoding="utf-8") as f:
                f.write(line)

    def _write_error(self, msg: str):
        with self._lock:
            with open(self._error_path, "a", encoding="utf-8") as f:
                f.write(msg)



class PaidModelError(RuntimeError):
    """Raised when OpenRouter reports a non-zero cost for a supposedly free model.

    This is a safety guard: if the model switches from free to paid, the pipeline
    stops immediately before accumulating any significant credit charges.
    """


def _call_api(
    api_key: str,
    messages: list,
    api_url: str = DEFAULT_API_URL,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 16000,
    temperature: float = 0.2,
) -> tuple[str, dict]:
    """Send a request to the VLM API and return (response_text, usage_dict).

    usage_dict contains: prompt_tokens, completion_tokens, total_tokens, cost.
    Prints token stats to the terminal and warns if nearing the free tier limit.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        # Force OpenRouter to only use the free provider (Alibaba for Qwen models).
        # If the free provider is down/overloaded, this returns an error instead of
        # silently falling back to a paid provider and charging credits.
        "provider": {"allow_fallbacks": False},
    }
    WALL_CLOCK_TIMEOUT = 300  # hard cap per attempt regardless of byte-trickle

    max_retries = 3
    for attempt in range(max_retries):
        # Run the HTTP call in a daemon thread so we can enforce a hard wall-clock
        # deadline.  requests.post(timeout=60) only guards against silence between
        # bytes; a slow-thinking model can trickle tokens every ~30s indefinitely,
        # preventing the socket timeout from ever firing.  The thread join with
        # WALL_CLOCK_TIMEOUT kills the wait after 5 minutes total.
        _result_holder = [None]
        _exc_holder    = [None]

        def _do_request():
            try:
                _result_holder[0] = requests.post(
                    api_url, headers=headers, json=payload, timeout=60,
                )
            except Exception as _e:
                _exc_holder[0] = _e

        _t = threading.Thread(target=_do_request, daemon=True)
        _t.start()
        _t.join(timeout=WALL_CLOCK_TIMEOUT)

        if _t.is_alive():
            # Thread still running — hard wall-clock exceeded
            print(f"  API wall-clock timeout after {WALL_CLOCK_TIMEOUT}s "
                  f"(attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                wait = 5.0 * (attempt + 1)
                print(f"  Retrying in {wait:.0f}s...")
                time.sleep(wait)
                continue
            raise TimeoutError(
                f"API call exceeded {WALL_CLOCK_TIMEOUT}s wall-clock limit "
                f"after {max_retries} attempts"
            )

        if _exc_holder[0] is not None:
            exc = _exc_holder[0]
            if attempt < max_retries - 1 and isinstance(
                exc, (requests.exceptions.Timeout,
                       requests.exceptions.ConnectionError)
            ):
                base_wait = 2 ** attempt * 5
                jitter    = random.uniform(-base_wait * 0.5, base_wait * 0.5)
                wait      = max(1.0, base_wait + jitter)
                print(f"  Request error {exc}, retrying in {wait:.1f}s...")
                time.sleep(wait)
                continue
            raise exc

        response = _result_holder[0]

        if response.status_code == 200:
            result = response.json()
            # ── Token usage ───────────────────────────────────────────────
            raw_usage = result.get("usage") or {}
            usage = {
                "prompt_tokens":     int(raw_usage.get("prompt_tokens", 0)),
                "completion_tokens": int(raw_usage.get("completion_tokens", 0)),
                "total_tokens":      int(raw_usage.get("total_tokens", 0)),
                "cost":              float(raw_usage.get("cost", 0) or 0),
                "provider":          result.get("provider", "Unknown"),
            }
            total_tok = usage["total_tokens"]
            # Print token stats to terminal
            tok_line = (
                f"    tokens: prompt={usage['prompt_tokens']} "
                f"completion={usage['completion_tokens']} "
                f"total={total_tok}  "
                f"provider={usage['provider']}"
            )
            print(tok_line)
            # Warn if approaching Alibaba's free threshold
            if total_tok >= FREE_TOKEN_THRESHOLD:
                print(
                    f"  ⚠ WARNING: {total_tok} tokens used — "
                    f"approaching/exceeding the free tier threshold ({FREE_TOKEN_THRESHOLD}). "
                    f"Alibaba may start charging for this context size!"
                )
            # ── Cost guard ───────────────────────────────────────────────
            if usage["cost"] > 0:
                raise PaidModelError(
                    f"Model '{model}' is no longer free! "
                    f"OpenRouter routed to paid provider '{usage['provider']}' "
                    f"and charged ${usage['cost']:.6f} for this request. "
                    f"The free provider (Alibaba) was likely down or overloaded. "
                    f"Pipeline stopped to protect your credits. "
                    f"Retry later when the free provider is available again."
                )
            return result["choices"][0]["message"]["content"], usage
        # Retry on transient server errors with exponential backoff + jitter.
        # Jitter (±50% of base wait) prevents all parallel workers from
        # retrying simultaneously (thundering-herd) after Alibaba rate-limits.
        if response.status_code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
            base_wait = 2 ** attempt * 5  # 5s, 10s, 20s
            jitter    = random.uniform(-base_wait * 0.5, base_wait * 0.5)
            wait      = max(1.0, base_wait + jitter)
            print(f"  API Error {response.status_code}, retrying in {wait:.1f}s "
                  f"(attempt {attempt + 1}/{max_retries}, jitter={jitter:+.1f}s)...")
            time.sleep(wait)
            continue
        print(f"API Error {response.status_code}: {response.text[:500]}")
        response.raise_for_status()
    raise RuntimeError("Max retries exceeded")


# ---------------------------------------------------------------------------
# HTML Generation (layout analysis + HTML in one call)
# ---------------------------------------------------------------------------

def _generate_html(
    api_key: str, image_base64: str, width: int, height: int,
    api_url: str = DEFAULT_API_URL, model: str = DEFAULT_MODEL,
) -> str | None:
    """
    Analyze the page layout and produce semantic HTML that reproduces the page
    visually — all in a single API call.

    The thinking model reasons about layout in its chain-of-thought before
    generating HTML, so no separate layout-analysis call is needed.

    Returns the extracted HTML string, or None if extraction fails.
    """
    a4_height = round(width * 1.414)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert document-to-HTML converter. You produce clean, "
                "semantic HTML that visually reproduces PDF pages with pixel-level fidelity. "
                "You handle legal documents, academic papers, math-heavy content, tables, "
                "and any other document type."
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                },
                {
                    "type": "text",
                    "text": (
                        "First, carefully study this PDF page and analyze its layout:\n"
                        "  - How many columns does the text have?\n"
                        "  - Are there margin notes or side annotations?\n"
                        "  - Are there tables? How many rows/columns?\n"
                        "  - Are there headers, footers, or page numbers?\n"
                        "  - Are there math equations or special formatting?\n\n"
                        "Then, create HTML that **exactly reproduces** this PDF page. "
                        "The rendered HTML must be VISUALLY INDISTINGUISHABLE from the original.\n\n"
                        "=== CSS FOUNDATION ===\n\n"
                        "Use this CSS as the base. You MAY add more CSS classes in the <style> block "
                        "as needed for the specific content, but do NOT use inline style= attributes.\n\n"
                        "```css\n"
                        "body {\n"
                        f"    width: {width}px;\n"
                        f"    min-height: {a4_height}px;\n"
                        "    overflow: visible;\n"
                        "    box-sizing: border-box;\n"
                        "    font-family: 'Times New Roman', Georgia, serif;\n"
                        "    font-size: 18pt;\n"
                        "    line-height: 1.5;\n"
                        "    padding: 80px 100px;\n"
                        "    margin: 0;\n"
                        "}\n"
                        "```\n\n"
                        "=== MANDATORY RULES ===\n\n"
                        "1. STRUCTURE — CRITICAL:\n"
                        "   - ANY margin notes, side notes, case citations, names, dates, or annotations "
                        "that appear in the LEFT or RIGHT margin of the page MUST be wrapped in an "
                        "<aside> tag. Do NOT use <div> with custom class names for side content. "
                        "Using <div> instead of <aside> will BREAK downstream processing.\n"
                        "   - Use flexbox for the page layout: <aside> + <div class=\"main-col\">\n"
                        "   - For single-column text with no margin notes: use <div class=\"main-col\"> directly\n"
                        "   - For multi-column layouts: use CSS columns or flexbox\n"
                        "   - Headers/footers/page numbers: wrap in <header>/<footer> tags\n\n"
                        "2. SEMANTIC TAGS:\n"
                        "   - Headings: <h1> through <h6> for document headings\n"
                        "   - Paragraphs: <p> for text blocks\n"
                        "   - Tables: <table> with <thead>/<tbody>, <tr>, <th>, <td>\n"
                        "   - Lists: <ul>/<ol> with <li>\n"
                        "   - Emphasis: <em> for italic, <strong> for bold\n"
                        "   - blockquote for quoted text\n\n"
                        "3. MATH EQUATIONS:\n"
                        "   - Include MathJax in <head>:\n"
                        "     <script src=\"https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js\"></script>\n"
                        "   - Inline math: \\\\( LaTeX \\\\)\n"
                        "   - Display math: \\\\[ LaTeX \\\\]\n"
                        "   - Only include MathJax if the page actually contains equations.\n\n"
                        "4. STRICT BANS — these will BREAK rendering:\n"
                        "   - NO inline style= attributes (define CSS classes instead)\n"
                        "   - NO position: fixed or position: absolute (use normal flow)\n"
                        "   - NO JavaScript (except MathJax CDN if needed)\n\n"
                        "5. TEXT ACCURACY: Copy ALL text from the image exactly as written. "
                        "Do not paraphrase, summarize, or skip any content. Every word must be present.\n\n"
                        "6. VISUAL FIDELITY: Match the original's font sizes, spacing, alignment, "
                        "indentation, and overall proportions as closely as possible.\n\n"
                        "Enclose your final HTML in a ```html code block."
                    ),
                },
            ],
        }
    ]
    response_text, usage = _call_api(api_key, messages, api_url=api_url, model=model, max_tokens=8000)
    html = _extract_html_block(response_text)
    if html:
        html = _inject_viewport_css(html, width, height)
    return html, usage


def _extract_html_block(text: str) -> str | None:
    """Extract HTML from a ```html ... ``` code block."""
    match = re.search(r"```html\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: try without language specifier
    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        content = match.group(1).strip()
        if "<" in content and ">" in content:
            return content

    return None


def _inject_viewport_css(html: str, width: int, height: int) -> str:
    """
    Post-process HTML to enforce correct viewport dimensions.

    Injects minimal overrides: page sizing and font defaults only.
    Does NOT override the VLM's layout decisions (width %, centering, etc.).
    """
    a4_height = round(width * 1.414)
    viewport_css = (
        f"\n<style>\n"
        f"  /* Injected viewport constraints */\n"
        f"  html {{ margin: 0 !important; padding: 0 !important; }}\n"
        f"  body {{\n"
        f"    width: {width}px !important;\n"
        f"    min-height: {a4_height}px !important;\n"
        f"    overflow: visible !important;\n"
        f"    box-sizing: border-box !important;\n"
        f"    margin: 0 !important;\n"
        f"  }}\n"
        f"</style>\n"
    )

    # Inject BEFORE </head> so it takes precedence over LLM styles
    if "</head>" in html:
        html = html.replace("</head>", f"{viewport_css}</head>", 1)
    elif "<body" in html:
        body_idx = html.index("<body")
        html = html[:body_idx] + viewport_css + html[body_idx:]
    else:
        html = f"<html><head>{viewport_css}</head>\n" + html

    return html


# ---------------------------------------------------------------------------
# Per-page Processing
# ---------------------------------------------------------------------------

def process_page(
    pdf_path: str, page_num: int, api_key: str, max_dim: int = 2048,
    api_url: str = DEFAULT_API_URL, model: str = DEFAULT_MODEL,
) -> dict | None:
    """
    Process a single PDF page: render to PNG, then call the VLM once to
    analyze layout and generate HTML in the same request.

    Returns a dict with:
      - html: the generated HTML string
      - page: page number
      - source: source PDF path
      - width/height: dimensions of the rendered page
    """
    t0 = time.time()
    print(f"  [Page {page_num}] Rendering to PNG...")
    image_base64 = render_pdf_to_base64png(pdf_path, page_num, max_dim)

    # Get image dimensions from the base64 PNG
    png_bytes = base64.b64decode(image_base64)
    width, height = _get_png_dimensions(png_bytes)
    print(f"  [Page {page_num}] Image size: {width}x{height}")

    print(f"  [Page {page_num}] Calling VLM (layout + HTML in one call)...")
    result_html, usage = _generate_html(api_key, image_base64, width, height, api_url=api_url, model=model)

    if result_html is None:
        print(f"  [Page {page_num}] WARNING: Failed to extract HTML from response")
        return None

    elapsed = time.time() - t0
    print(f"  [Page {page_num}] HTML generated ({len(result_html)} chars) in {elapsed:.1f}s")
    return {
        "html":   result_html,
        "page":   page_num,
        "source": pdf_path,
        "width":  width,
        "height": height,
        "usage":  usage,        # prompt_tokens, completion_tokens, total_tokens, cost, provider
        "elapsed": elapsed,
    }


def _get_png_dimensions(png_bytes: bytes) -> tuple:
    """Extract width and height from PNG header bytes."""
    # PNG width is at bytes 16-19, height at 20-23 (big-endian)
    if png_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        width = int.from_bytes(png_bytes[16:20], "big")
        height = int.from_bytes(png_bytes[20:24], "big")
        return width, height
    return 800, 1000  # sensible fallback


# ---------------------------------------------------------------------------
# Batch Processing
# ---------------------------------------------------------------------------

def process_pdf(
    pdf_path: str, api_key: str, output_dir: str, max_dim: int = 2048,
    api_url: str = DEFAULT_API_URL, model: str = DEFAULT_MODEL,
    workers: int = 1,
    logger: "PipelineLogger | None" = None,
) -> list:
    """Process all pages of a PDF and save generated HTML files.

    Pages are processed with up to `workers` threads in parallel.
    Already-completed pages (both .html and .json exist) are skipped.
    Progress and errors are logged to <output_dir>/logs/ if a logger is provided.
    """
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)
    pdf_name  = os.path.splitext(os.path.basename(pdf_path))[0]

    html_dir = os.path.join(output_dir, "html")
    os.makedirs(html_dir, exist_ok=True)

    effective_workers = min(workers, num_pages)
    if logger:
        logger.pdf_start(pdf_path, num_pages, effective_workers)

    pdf_t0 = time.time()
    # Per-PDF counters (thread-safe via a local lock)
    _lock        = threading.Lock()
    _done_count  = [0]
    _skip_count  = [0]
    _fail_count  = [0]

    def _process_one(page_num: int) -> dict | None:
        """Process a single page, with resume support and startup stagger."""
        html_path = os.path.join(html_dir, f"{pdf_name}_page_{page_num}.html")
        meta_path = os.path.join(html_dir, f"{pdf_name}_page_{page_num}.json")

        # Resume support: skip pages already successfully processed
        if os.path.exists(html_path) and os.path.exists(meta_path):
            with _lock:
                _skip_count[0] += 1
            if logger:
                logger.page_skipped(pdf_name, page_num)
            # (no stagger needed for skipped pages)
            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["html"] = html_content
            return meta

        # Stagger: each page waits (index × 2s + small jitter) before making its
        # first API call.  This spreads the burst across workers so Alibaba doesn’t
        # see them all arrive at once and fire a 429.
        page_index = page_nums.index(page_num)   # 0-based position in queue
        stagger    = min(page_index * 2.0 + random.uniform(0, 1.0), 30.0)
        if stagger > 0:
            time.sleep(stagger)

        try:
            result = process_page(pdf_path, page_num, api_key, max_dim, api_url=api_url, model=model)
        except PaidModelError:
            raise  # must propagate immediately — do NOT catch as a normal page failure
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            print(f"  [Page {page_num}] ERROR: {reason}")
            with _lock:
                _fail_count[0] += 1
            if logger:
                logger.page_failed(pdf_name, page_num, reason, exc)
            return None

        if result is None:
            reason = "VLM returned None (could not extract HTML)"
            print(f"  [Page {page_num}] FAILED: {reason}")
            with _lock:
                _fail_count[0] += 1
            if logger:
                logger.page_failed(pdf_name, page_num, reason)
            return None

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(result["html"])

        meta = {k: v for k, v in result.items() if k != "html"}
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        with _lock:
            _done_count[0] += 1
        if logger:
            logger.page_done(pdf_name, page_num, len(result["html"]),
                             result.get("elapsed", 0), usage=result.get("usage"))
        print(f"  [Page {page_num}] Saved → {html_path}")
        return result

    page_nums = list(range(1, num_pages + 1))
    results   = []

    if effective_workers <= 1:
        for page_num in page_nums:
            result = _process_one(page_num)
            if result is not None:
                results.append(result)
    else:
        print(f"  Processing {num_pages} pages with {effective_workers} parallel workers")
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {executor.submit(_process_one, p): p for p in page_nums}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)

    pdf_elapsed = time.time() - pdf_t0
    if logger:
        logger.pdf_done(
            pdf_path,
            done=_done_count[0],
            skipped=_skip_count[0],
            failed=_fail_count[0],
            elapsed=pdf_elapsed,
        )
        print(f"  {logger.running_totals()}")

    results.sort(key=lambda r: r.get("page", 0))
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def check_model_is_free(model: str, api_key: str) -> bool:
    """Query the OpenRouter models catalogue and return True if the model is free.

    Checks both prompt and completion pricing — if either is > 0, returns False.
    Returns True (assume free) on any network/parse error so a transient failure
    doesn't block the pipeline; the response-level cost guard is the fallback.
    """
    try:
        resp = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        resp.raise_for_status()
        for m in resp.json().get("data", []):
            if m["id"] == model:
                pricing = m.get("pricing") or {}
                prompt_cost     = float(pricing.get("prompt", 0) or 0)
                completion_cost = float(pricing.get("completion", 0) or 0)
                return prompt_cost == 0.0 and completion_cost == 0.0
        # Model not found in catalogue — treat as unknown/paid to be safe
        print(f"WARNING: Model '{model}' not found in OpenRouter catalogue. "
              "Cannot verify pricing. Proceeding with caution.")
        return False
    except Exception as exc:
        print(f"WARNING: Could not verify model pricing ({exc}). "
              "Proceeding — response-level cost guard is still active.")
        return True  # don't block on network errors


def main():
    parser = argparse.ArgumentParser(
        description="Generate faithful HTML from PDF pages using a Vision-Language Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Via OpenRouter (default):\n"
            "  python -m synth.generate_html --pdf-dir data/\n\n"
            "  # Via local vLLM server:\n"
            "  python -m synth.generate_html --pdf-dir data/ \\\ \n"
            "    --api-url http://localhost:8000/v1/chat/completions \\\ \n"
            "    --model Qwen/Qwen3-VL-235B-A22B-Thinking\n"
        ),
    )
    parser.add_argument("--pdf-dir", required=True, help="Directory containing PDF files")
    parser.add_argument("--output-dir", default="synth_output", help="Output directory")
    parser.add_argument("--max-pages", type=int, default=999999, help="Max pages to process across all PDFs")
    parser.add_argument("--max-dim", type=int, default=2048, help="Max image dimension in pixels")
    parser.add_argument("--api-key", help="API key (or set OPENROUTER_API_KEY in .env). Use 'none' for local vLLM.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL,
                        help=f"API endpoint URL (default: {DEFAULT_API_URL})")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers per PDF (default: 1). Each worker makes one API call.")
    parser.add_argument("--pdf-workers", type=int, default=1,
                        help="Parallel PDFs to process concurrently (default: 1). Total API calls = pdf-workers × workers.")
    args = parser.parse_args()

    load_dotenv()
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY") or ""

    # For local vLLM, no API key is needed
    is_local = "localhost" in args.api_url or "127.0.0.1" in args.api_url
    if not api_key and not is_local:
        print("Error: No API key. Use --api-key or set OPENROUTER_API_KEY in .env")
        sys.exit(1)
    if is_local and not api_key:
        api_key = "none"  # vLLM doesn't require auth

    # ── Model pricing check (OpenRouter only) ────────────────────────
    is_openrouter = "openrouter.ai" in args.api_url
    if is_openrouter:
        print(f"Checking model pricing for '{args.model}'...")
        if not check_model_is_free(args.model, api_key):
            print(
                f"\nERROR: '{args.model}' is NOT free on OpenRouter.\n"
                f"Pipeline stopped to protect your credits.\n"
                f"Verify at: https://openrouter.ai/models\n"
                f"If you intended to use a paid model, remove this check "
                f"or use --api-url with a local endpoint."
            )
            sys.exit(1)
        print(f"  ✓ Confirmed free ($0 prompt / $0 completion)")

    # Collect PDF files (recursive, matches run_pipeline.py behaviour)
    pdf_files = sorted(glob.glob(
        os.path.join(args.pdf_dir, "**", "*.pdf"), recursive=True
    ))

    if not pdf_files:
        print(f"No PDFs found under {args.pdf_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    logger = PipelineLogger(args.output_dir)
    workers     = getattr(args, "workers", 1)
    pdf_workers = getattr(args, "pdf_workers", 1)

    print(f"Found {len(pdf_files)} PDF(s) under {args.pdf_dir}")
    print(f"  API         : {args.api_url}")
    print(f"  Model       : {args.model}")
    print(f"  PDF workers : {pdf_workers} concurrent PDFs")
    print(f"  Page workers: {workers} per PDF")
    print(f"  Total API   : up to {pdf_workers * workers} concurrent calls")
    print(f"  Logs        : {args.output_dir}/logs/")

    logger.session_start(
        num_pdfs=len(pdf_files),
        output_dir=args.output_dir,
        model=args.model,
        workers=workers,
    )

    session_t0  = time.time()
    total_pages = 0

    if pdf_workers <= 1:
        # Sequential (original behaviour)
        for pdf_path in pdf_files:
            if total_pages >= args.max_pages:
                break
            print(f"\nProcessing: {pdf_path}")
            results = process_pdf(
                pdf_path, api_key, args.output_dir, args.max_dim,
                api_url=args.api_url, model=args.model,
                workers=workers,
                logger=logger,
            )
            total_pages += len(results)
            if total_pages >= args.max_pages:
                print(f"Reached max pages limit ({args.max_pages})")
                break
    else:
        # Inter-PDF parallelism: process pdf_workers PDFs concurrently
        _page_lock  = threading.Lock()
        _page_total = [0]
        _stop       = [False]

        def _process_one_pdf(pdf_path: str) -> int:
            with _page_lock:
                if _page_total[0] >= args.max_pages or _stop[0]:
                    return 0
            print(f"\nProcessing: {pdf_path}")
            try:
                results = process_pdf(
                    pdf_path, api_key, args.output_dir, args.max_dim,
                    api_url=args.api_url, model=args.model,
                    workers=workers,
                    logger=logger,
                )
            except PaidModelError:
                raise
            except Exception as exc:
                print(f"  ERROR processing {pdf_path}: {exc}")
                return 0
            with _page_lock:
                _page_total[0] += len(results)
                if _page_total[0] >= args.max_pages:
                    _stop[0] = True
                    print(f"Reached max pages limit ({args.max_pages})")
            return len(results)

        with ThreadPoolExecutor(max_workers=pdf_workers) as pdf_executor:
            futures = {pdf_executor.submit(_process_one_pdf, p): p for p in pdf_files}
            for future in as_completed(futures):
                try:
                    future.result()
                except PaidModelError:
                    pdf_executor.shutdown(wait=False, cancel_futures=True)
                    raise
                except Exception as exc:
                    print(f"  PDF-level error: {exc}")

        total_pages = _page_total[0]

    logger.session_end(elapsed=time.time() - session_t0)
    print(f"\nDone! Generated HTML for {total_pages} pages → {args.output_dir}/html/")
    print(f"Logs written to {args.output_dir}/logs/")


if __name__ == "__main__":
    main()