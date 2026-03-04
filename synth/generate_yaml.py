"""
generate_yaml.py — Generate YAML OCR output directly from PDF pages using a VLM.

This is the fast-path alternative to generate_html.py for bulk SFT data generation.
Instead of generating rich HTML (used for GRPO unit tests), it asks the VLM to output
YAML directly — matching the final model's inference format.

Key differences vs generate_html.py:
  - ~4-5x faster: YAML completions are 300–1000 tokens vs 3000–9000 for HTML
  - No rendering step needed (no Playwright)
  - No prepare_sft_data.py conversion needed
  - Skips pages already covered by generate_html (have .html files)
  - Output goes to synth_output/yaml/ folder

Use this for bulk SFT data generation AFTER generate_html.py has covered the
representative subset needed for GRPO unit tests.
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
from src.anchor import get_anchor_text


# ---------------------------------------------------------------------------
# API defaults
# ---------------------------------------------------------------------------

DEFAULT_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL   = "qwen/qwen3-vl-235b-a22b-thinking"

# Alibaba free tier: warn when total tokens approach threshold
FREE_TOKEN_THRESHOLD = 28_000


# ---------------------------------------------------------------------------
# Prompt template (same as prepare_sft_data.py — the model's target format)
# ---------------------------------------------------------------------------

# Two prompt variants depending on whether native PDF text is available:
#
# YAML_PROMPT_WITH_ANCHOR — digital PDFs: native text extracted via pypdf is
#     appended as <|anchor_start|>...<|anchor_end|> after the image.
#
# YAML_PROMPT_IMAGE_ONLY  — scanned PDFs (sc folder): no embedded text,
#     prompt makes no mention of "raw textual content".

_YAML_COMMON = (
    "/no_think\n"  # Disable thinking mode in Qwen3 — output YAML directly
    "Turn equations into a LaTeX representation, and tables into markdown format.\n"
    "IMPORTANT: Reflow the text to remove visual line-wrap artifacts from the PDF layout. "
    "Join lines that belong to the SAME paragraph into a single flowing sentence. "
    "Preserve paragraph boundaries — each distinct paragraph in the original text should remain a separate paragraph, separated by a blank line. "
    "If a word is split across lines with a hyphen (e.g., 'argu-\\nment'), rejoin it as one word ('argument').\n"
    "IMPORTANT: Extract ONLY the main body text of the page. You MUST exclude ALL of the following:\n"
    "  - Headers and footers (any repeating text at the top or bottom of the page)\n"
    "  - Page numbers (whether at the top, bottom, or margins of the page)\n"
    "  - Margin notes and side annotations\n"
    "  - Single letters (A, B, C, D...) or single numbers that appear in the margins or at the very start of paragraphs as layout/section markers — these are print annotations, not body text\n"
    "  - Any text that appears outside the main text column/body area\n"
    "Keep references, footnotes, and citations that are part of the main body text.\n"
    "Read any natural handwriting.\n"
    "This is likely one page out of several in the document, so be sure to preserve "
    "any sentences that come from the previous page, or continue onto the next page, exactly as they are.\n"
    "If there is no text at all that you think you should read, output null for natural_text.\n"
    "Do not hallucinate.\n\n"
    "Respond in the following YAML format. You MUST wrap your entire output in a single ```yaml code block:\n"
    "primary_language: <two-letter language code, or null if no readable text>\n"
    "is_rotation_valid: <true if page is correctly oriented for reading, false otherwise>\n"
    "rotation_correction: <0, 90, 180, or 270 — degrees of clockwise rotation needed>\n"
    "is_table: <true if majority of page content is tabular>\n"
    "is_diagram: <true if majority of page content is a visual diagram>\n"
    "natural_text: |\n"
    "  <the extracted plain text, indented by 2 spaces>"
)

# For digital PDFs — tells model that anchor text follows the image
YAML_PROMPT_WITH_ANCHOR = (
    "Below is the image of one page of a PDF document, as well as some raw textual content "
    "that was previously extracted for it that includes position information for each image "
    "and block of text (The origin [0x0] of the coordinates is in the lower left corner of the image). "
    "Just return the plain text representation of this document as if you were reading it naturally.\n"
    + _YAML_COMMON
)

# For scanned PDFs — image only, no mention of raw text (there is none)
YAML_PROMPT_IMAGE_ONLY = (
    "Below is the image of one page of a scanned PDF document. "
    "There is no embedded text — read the content directly from the scanned image.\n"
    "Just return the plain text representation of this document as if you were reading it naturally.\n"
    + _YAML_COMMON
)


# ---------------------------------------------------------------------------
# Safety: paid-model guard
# ---------------------------------------------------------------------------

class PaidModelError(RuntimeError):
    """Raised when OpenRouter reports a non-zero cost for a supposedly free model.

    This is a safety guard: if the model switches from free to paid, the pipeline
    stops immediately before accumulating any significant credit charges.
    """


# ---------------------------------------------------------------------------
# Thread-safe logger (mirrors generate_html.py's PipelineLogger)
# ---------------------------------------------------------------------------

class PipelineLogger:
    def __init__(self, output_dir: str):
        log_dir = os.path.join(output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        self._progress_path = os.path.join(log_dir, "yaml_progress.log")
        self._error_path    = os.path.join(log_dir, "yaml_errors.log")
        self._dlq_path      = os.path.join(log_dir, "yaml_failed_pages.txt")
        self._lock = threading.Lock()
        self._output_dir = output_dir
        self.pdfs_processed = 0
        self.pdfs_skipped   = 0
        self.pages_done     = 0
        self.pages_skipped  = 0
        self.pages_failed   = 0

    def _now(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _write_progress(self, msg: str):
        with self._lock:
            with open(self._progress_path, "a", encoding="utf-8") as f:
                f.write(f"[{self._now()}] {msg}\n")

    def _write_error(self, msg: str):
        with self._lock:
            with open(self._error_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

    def inventory(self) -> tuple[int, int]:
        yaml_dir = os.path.join(self._output_dir, "yaml")
        if not os.path.exists(yaml_dir):
            return 0, 0
        yaml_files = [f for f in os.listdir(yaml_dir) if f.endswith(".yaml")]
        pages = len(yaml_files)
        pdf_names = {re.sub(r"_page_\d+\.yaml$", "", f) for f in yaml_files}
        return pages, len(pdf_names)

    def session_start(self, num_pdfs: int, output_dir: str, model: str, workers: int):
        pages_done, pdfs_done = self.inventory()
        resume_line = (
            f"  Resume : {pages_done} YAML pages already done across {pdfs_done} PDF(s)\n"
            if pages_done > 0 else "  Resume : fresh start\n"
        )
        msg = (
            f"=== YAML SESSION START ===\n"
            f"  Time   : {self._now()}\n"
            f"  PDFs   : {num_pdfs}\n"
            f"{resume_line}"
            f"  Output : {output_dir}\n"
            f"  Model  : {model}\n"
            f"  Workers: {workers} per PDF\n"
        )
        self._write_progress(msg)
        if pages_done > 0:
            print(f"\n[RESUME] {pages_done} YAML pages already done. Skipping automatically.\n")
        else:
            print("[START] Fresh YAML generation run.\n")

    def session_end(self, elapsed: float):
        msg = (
            f"=== YAML SESSION END ===\n"
            f"  Time         : {self._now()}\n"
            f"  Elapsed      : {elapsed:.1f}s ({elapsed/60:.1f} min)\n"
            f"  PDFs done    : {self.pdfs_processed} (+{self.pdfs_skipped} fully skipped)\n"
            f"  Pages done   : {self.pages_done}\n"
            f"  Pages skipped: {self.pages_skipped}\n"
            f"  Pages failed : {self.pages_failed}\n"
        )
        self._write_progress(msg)
        print(msg)

    def pdf_start(self, pdf_path: str, num_pages: int, workers: int):
        self._write_progress(
            f"[PDF START] {os.path.basename(pdf_path)}  ({num_pages} pages, {workers} workers)"
        )

    def pdf_done(self, pdf_path: str, done: int, skipped: int, failed: int, elapsed: float):
        with self._lock:
            self.pdfs_processed += 1
            if done == 0 and failed == 0:
                self.pdfs_skipped += 1
        self._write_progress(
            f"[PDF DONE ] {os.path.basename(pdf_path)}  "
            f"done={done} skipped={skipped} failed={failed}  elapsed={elapsed:.1f}s"
        )

    def page_skipped(self, pdf_name: str, page_num: int):
        with self._lock:
            self.pages_skipped += 1
        self._write_progress(f"[SKIP ] {pdf_name} page {page_num}")

    def page_done(self, pdf_name: str, page_num: int, yaml_len: int, elapsed: float,
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
            if usage.get('total_tokens', 0) >= FREE_TOKEN_THRESHOLD:
                usage_str += f"  ⚠ OVER FREE THRESHOLD ({FREE_TOKEN_THRESHOLD})"
        self._write_progress(
            f"[DONE ] {pdf_name} page {page_num}  {yaml_len} chars  {elapsed:.1f}s" + usage_str
        )

    def page_failed(self, pdf_name: str, page_num: int, reason: str, exc: Exception = None):
        with self._lock:
            self.pages_failed += 1
        self._write_progress(f"[FAIL ] {pdf_name} page {page_num}  {reason}")
        tb = _traceback.format_exc() if exc else ""
        self._write_error(
            f"--- {self._now()} ---\n"
            f"PDF : {pdf_name}  Page: {page_num}\n"
            f"Reason: {reason}\n{tb}\n"
        )
        with self._lock:
            with open(self._dlq_path, "a", encoding="utf-8") as f:
                f.write(f"{self._now()}\t{pdf_name}\t{page_num}\t{reason}\n")

    def running_totals(self) -> str:
        return (
            f"Progress → PDFs: {self.pdfs_processed} processed, "
            f"Pages: {self.pages_done} done / "
            f"{self.pages_skipped} skipped / "
            f"{self.pages_failed} failed"
        )


# ---------------------------------------------------------------------------
# API call (same as generate_html.py — cost guard + jitter + provider lock)
# ---------------------------------------------------------------------------

def _call_api(
    api_key: str,
    messages: list,
    api_url: str = DEFAULT_API_URL,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 8192,
    temperature: float = 0.1,
) -> tuple[str, dict]:
    """Call the VLM and return (response_text, usage_dict)."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        # Cap Qwen3 chain-of-thought to 500 thinking tokens.
        # Without this, dense pages burn 6000-9000 reasoning tokens before
        # writing a single word of YAML, causing 200-350s per page.
        # budget_tokens lets the model reason briefly but not excessively.
        "thinking": {"budget_tokens": 2046},
        # Lock to free provider only — no paid fallback
        "provider": {"allow_fallbacks": False},
    }
    WALL_CLOCK_TIMEOUT = 240  # hard cap per attempt — 240s allows slow pages, cuts truly stuck calls

    max_retries = 3
    for attempt in range(max_retries):
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
            # OpenRouter sometimes returns HTTP 200 with an error body (no 'choices')
            # when the backend (Alibaba) is overloaded. Treat this as a retryable error.
            if "choices" not in result:
                err_msg = result.get("error", {}).get("message", str(result)[:200])
                print(f"  OpenRouter returned 200 but no 'choices': {err_msg}")
                if attempt < max_retries - 1:
                    wait = 5.0 * (attempt + 1)
                    print(f"  Retrying in {wait:.0f}s...")
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"API returned 200 with no 'choices' after {max_retries} attempts: {err_msg}")
            raw_usage = result.get("usage") or {}
            usage = {
                "prompt_tokens":     int(raw_usage.get("prompt_tokens", 0)),
                "completion_tokens": int(raw_usage.get("completion_tokens", 0)),
                "total_tokens":      int(raw_usage.get("total_tokens", 0)),
                "cost":              float(raw_usage.get("cost", 0) or 0),
                "provider":          result.get("provider", "Unknown"),
            }
            total_tok = usage["total_tokens"]
            print(
                f"    tokens: prompt={usage['prompt_tokens']} "
                f"completion={usage['completion_tokens']} "
                f"total={total_tok}  provider={usage['provider']}"
            )
            if total_tok >= FREE_TOKEN_THRESHOLD:
                print(
                    f"  ⚠ WARNING: {total_tok} tokens — approaching free tier threshold "
                    f"({FREE_TOKEN_THRESHOLD}). Alibaba may charge!"
                )
            if usage["cost"] > 0:
                raise PaidModelError(
                    f"Model '{model}' is no longer free! "
                    f"OpenRouter routed to paid provider '{usage['provider']}' "
                    f"and charged ${usage['cost']:.6f}. "
                    f"Pipeline stopped to protect your credits."
                )
            _msg = result["choices"][0]["message"]
            content = _msg.get("content") or ""
            reasoning = _msg.get("reasoning") or ""
            
            # Combine both fields just in case OpenRouter or the VLM 
            # accidentally dumped the YAML inside the thought block
            full_response = f"{reasoning}\n{content}"
            
            return full_response, usage


        # Retry with exponential backoff + jitter
        if response.status_code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
            base_wait = 2 ** attempt * 5
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
# YAML generation (single VLM call)
# ---------------------------------------------------------------------------

def _generate_yaml(
    api_key: str,
    image_base64: str,
    anchor_text: str = "",
    api_url: str = DEFAULT_API_URL,
    model: str = DEFAULT_MODEL,
) -> tuple[str | None, dict]:
    """Send page image (+ optional anchor text) to VLM, return (yaml_string, usage_dict)."""
    if anchor_text and anchor_text.strip():
        # Digital PDF: include anchor text to help alignment
        prompt = (
            YAML_PROMPT_WITH_ANCHOR
            + f"\n\n<|anchor_start|>\n{anchor_text.strip()}\n<|anchor_end|>"
        )
    else:
        # Scanned PDF: image only — use the correct prompt with no misleading "raw text" mention
        prompt = YAML_PROMPT_IMAGE_ONLY

    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict, automated OCR data-extraction API. "
                "You DO NOT converse, you DO NOT explain your reasoning, and you DO NOT think out loud. "
                "You must immediately output the extracted text in the requested YAML format. "
                "You must start your response directly with the ```yaml code fence. "
                "Any conversational text (e.g., 'Okay', 'Here is the text', or 'Let's tackle this') will cause a fatal system crash."
            )
        },
        {
            "role": "user",
            "content": [
                {"type": "text",      "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
            ],
        }
    ]
    response_text, usage = _call_api(
        api_key, messages, api_url=api_url, model=model, max_tokens=8192
    )
    yaml_text = _extract_yaml_from_response(response_text) if response_text else None
    if yaml_text:
        return yaml_text, usage
    return None, usage


def _extract_yaml_from_response(raw: str) -> str | None:
    """Ultimate fallback extractor to catch stubborn VLM outputs."""
    text = raw.strip()
    
    # 1. Strip out <think> blocks if the API left them in the raw text
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    
    # 2. Try to find the strict markdown fences first
    code_match = re.search(r"```(?:yaml)?\s*\n(.*?)```", text, re.DOTALL)
    if code_match:
        extracted = code_match.group(1).strip()
        if "natural_text:" in extracted:
            return extracted

    # 3. FALLBACK: The model completely ignored the markdown fences.
    # Hunt for the exact start of the olmOCR schema.
    schema_start = text.find("primary_language:")
    if schema_start != -1 and "natural_text:" in text:
        # Slice the string from 'primary_language:' to the very end
        extracted = text[schema_start:].strip()
        
        # Clean up any trailing markdown backticks the model might have left
        extracted = re.sub(r"```$", "", extracted).strip()
        return extracted

    # 4. DEBUGGING: If it STILL fails, print exactly what the model did 
    # so we aren't flying blind.
    print(f"\n--- FATAL EXTRACTION ERROR ---")
    print(f"The VLM output did not contain 'primary_language:' or 'natural_text:'.")
    print(f"RAW VLM OUTPUT:\n{raw[:1000]}...\n------------------------------\n")
    
    return None


# ---------------------------------------------------------------------------
# Per-page + per-PDF processing
# ---------------------------------------------------------------------------

def _get_png_dimensions(png_bytes: bytes) -> tuple:
    import struct
    return struct.unpack(">II", png_bytes[16:24])


def process_page_yaml(
    pdf_path: str, page_num: int, api_key: str, max_dim: int = 2048,
    api_url: str = DEFAULT_API_URL, model: str = DEFAULT_MODEL,
    prefix: str = "",
) -> dict | None:
    t0 = time.time()
    tag = f"{prefix}[Page {page_num}]" if prefix else f"  [Page {page_num}]"
    print(f"{tag} Rendering to PNG...")
    image_base64 = render_pdf_to_base64png(pdf_path, page_num, max_dim)
    png_bytes = base64.b64decode(image_base64)
    width, height = _get_png_dimensions(png_bytes)
    print(f"{tag} Image size: {width}x{height}")

    # Try to extract native PDF text (works for digital PDFs, returns "" for scanned)
    anchor_text = ""
    try:
        anchor_text = get_anchor_text(pdf_path, page_num) or ""
    except Exception:
        pass
    # Heuristic: <20 non-whitespace chars → almost certainly a scanned image
    is_scanned = len(anchor_text.strip()) < 20
    print(
        f"{tag} Type: "
        f"{'scanned (image-only prompt)' if is_scanned else 'digital (anchor text prompt)'}"
    )

    print(f"{tag} Calling VLM (YAML)...")
    yaml_text, usage = _generate_yaml(
        api_key, image_base64,
        anchor_text=anchor_text,
        api_url=api_url, model=model,
    )

    if yaml_text is None:
        print(f"{tag} WARNING: VLM did not return valid YAML")
        return None

    elapsed = time.time() - t0
    print(f"{tag} YAML generated ({len(yaml_text)} chars) in {elapsed:.1f}s")
    return {
        "yaml":       yaml_text,
        "page":       page_num,
        "source":     pdf_path,
        "width":      width,
        "height":     height,
        "is_scanned": is_scanned,
        "usage":      usage,
        "elapsed":    elapsed,
    }


def process_pdf_yaml(
    pdf_path: str,
    api_key: str,
    output_dir: str,
    html_dir: str,          # pages with .html are skipped (covered by generate_html)
    max_dim: int = 2048,
    api_url: str = DEFAULT_API_URL,
    model: str = DEFAULT_MODEL,
    workers: int = 4,
    logger: "PipelineLogger | None" = None,
) -> list:
    try:
        from pypdf import PdfReader
        num_pages = len(PdfReader(pdf_path).pages)
    except Exception as e:
        print(f"  Could not read {pdf_path}: {e}")
        return []

    if num_pages == 0:
        return []

    pdf_name  = re.sub(r"\.pdf$", "", os.path.basename(pdf_path), flags=re.IGNORECASE)
    # Short tag for terminal log lines: last part after the last "~" or full name if no "~"
    _short    = pdf_name.split("~")[-1] if "~" in pdf_name else pdf_name
    pdf_tag   = f"  [{_short}]"  # e.g. [scr_1973_2_437_451_e]
    yaml_dir  = os.path.join(output_dir, "yaml")
    os.makedirs(yaml_dir, exist_ok=True)

    effective_workers = min(workers, num_pages)
    if logger:
        logger.pdf_start(pdf_path, num_pages, effective_workers)

    pdf_t0       = time.time()
    _lock        = threading.Lock()
    _done_count  = [0]
    _skip_count  = [0]
    _fail_count  = [0]
    page_nums    = list(range(1, num_pages + 1))
    results      = []

    def _process_one(page_num: int) -> dict | None:
        yaml_path = os.path.join(yaml_dir, f"{pdf_name}_page_{page_num}.yaml")
        meta_path = os.path.join(yaml_dir, f"{pdf_name}_page_{page_num}.json")
        html_path = os.path.join(html_dir, f"{pdf_name}_page_{page_num}.html")

        # Skip if HTML already exists (covered by generate_html pipeline)
        if os.path.exists(html_path):
            with _lock:
                _skip_count[0] += 1
            if logger:
                logger.page_skipped(pdf_name, page_num)
            return None  # html pages don't contribute to yaml results

        # Resume: skip if YAML already exists (from VLM or html_to_yaml.py)
        if os.path.exists(yaml_path):
            with _lock:
                _skip_count[0] += 1
            if logger:
                logger.page_skipped(pdf_name, page_num)
            # Return metadata if available, otherwise just signal skip
            if os.path.exists(meta_path):
                with open(yaml_path, "r", encoding="utf-8") as f:
                    yaml_content = f.read()
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta["yaml"] = yaml_content
                return meta
            return None

        # Stagger to avoid thundering-herd on Alibaba
        page_index = page_nums.index(page_num)
        stagger    = min(page_index * 2.0 + random.uniform(0, 1.0), 30.0)
        if stagger > 0:
            time.sleep(stagger)

        try:
            result = process_page_yaml(pdf_path, page_num, api_key, max_dim,
                                       api_url=api_url, model=model, prefix=f"{pdf_tag} ")
        except PaidModelError:
            raise
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            print(f"{pdf_tag} [Page {page_num}] ERROR: {reason}")
            with _lock:
                _fail_count[0] += 1
            if logger:
                logger.page_failed(pdf_name, page_num, reason, exc)
            return None

        if result is None:
            reason = "VLM did not return valid YAML"
            print(f"{pdf_tag} [Page {page_num}] FAILED: {reason}")
            with _lock:
                _fail_count[0] += 1
            if logger:
                logger.page_failed(pdf_name, page_num, reason)
            return None

        # Save YAML and metadata — prepend document_id for stable tracking
        doc_id     = f"{pdf_name}_page_{page_num}"
        yaml_with_id = f"document_id: {doc_id}\n{result['yaml']}"
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(yaml_with_id)
        meta = {k: v for k, v in result.items() if k != "yaml"}
        meta["document_id"] = doc_id
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        with _lock:
            _done_count[0] += 1
        if logger:
            logger.page_done(pdf_name, page_num, len(result["yaml"]),
                             result.get("elapsed", 0), usage=result.get("usage"))
        print(f"{pdf_tag} [Page {page_num}] Saved → {yaml_path}")
        return result

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
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except PaidModelError:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
                except Exception as exc:
                    page_num = futures[future]
                    print(f"  [Page {page_num}] Unhandled: {exc}")

    elapsed = time.time() - pdf_t0
    if logger:
        logger.pdf_done(pdf_path, _done_count[0], _skip_count[0], _fail_count[0], elapsed)
    print(logger.running_totals() if logger else
          f"  Done: {_done_count[0]} | Skipped: {_skip_count[0]} | Failed: {_fail_count[0]}")
    results.sort(key=lambda r: r.get("page", 0))
    return results


# ---------------------------------------------------------------------------
# Startup: verify model is free
# ---------------------------------------------------------------------------

def check_model_is_free(model: str, api_key: str) -> bool:
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
                return (float(pricing.get("prompt", 0) or 0) == 0.0 and
                        float(pricing.get("completion", 0) or 0) == 0.0)
        print(f"WARNING: Model '{model}' not found in catalogue.")
        return False
    except Exception as exc:
        print(f"WARNING: Could not verify model pricing ({exc}). Proceeding cautiously.")
        return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate YAML OCR output directly from PDF pages (fast-path SFT data)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # All PDFs, skip pages already in HTML:\n"
            "  python -m synth.generate_yaml --pdf-dir data/ --output-dir synth_output/\n\n"
            "  # Only sc folder:\n"
            "  python -m synth.generate_yaml --pdf-dir data/sc/ --output-dir synth_output/\n"
        ),
    )
    parser.add_argument("--pdf-dir",    required=True,  help="Directory containing PDF files")
    parser.add_argument("--output-dir", default="synth_output", help="Output directory")
    parser.add_argument("--html-dir",   default=None,
                        help="HTML dir to check for already-covered pages "
                             "(default: <output-dir>/html)")
    parser.add_argument("--max-pages",  type=int, default=999999,
                        help="Max pages to process (default: all)")
    parser.add_argument("--max-dim",    type=int, default=2048)
    parser.add_argument("--api-key",    help="API key (or set OPENROUTER_API_KEY in .env)")
    parser.add_argument("--api-url",    default=DEFAULT_API_URL)
    parser.add_argument("--model",      default=DEFAULT_MODEL)
    parser.add_argument("--workers",    type=int, default=4,
                        help="Parallel workers per PDF (default: 4). Each worker makes one API call.")
    parser.add_argument("--pdf-workers", type=int, default=1,
                        help="Parallel PDFs to process concurrently (default: 1). Total API calls = pdf-workers × workers.")
    args = parser.parse_args()

    load_dotenv()
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY") or ""

    is_local = "localhost" in args.api_url or "127.0.0.1" in args.api_url
    if not api_key and not is_local:
        print("Error: No API key. Use --api-key or set OPENROUTER_API_KEY in .env")
        sys.exit(1)
    if is_local and not api_key:
        api_key = "none"

    # Model pricing check
    is_openrouter = "openrouter.ai" in args.api_url
    if is_openrouter:
        print(f"Checking model pricing for '{args.model}'...")
        if not check_model_is_free(args.model, api_key):
            print(
                f"\nERROR: '{args.model}' is NOT free on OpenRouter.\n"
                f"Pipeline stopped to protect your credits.\n"
            )
            sys.exit(1)
        print(f"  ✓ Confirmed free ($0 prompt / $0 completion)")

    pdf_files = sorted(glob.glob(
        os.path.join(args.pdf_dir, "**", "*.pdf"), recursive=True
    ), reverse=True)
    if not pdf_files:
        print(f"No PDFs found under {args.pdf_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    html_dir = args.html_dir or os.path.join(args.output_dir, "html")

    logger = PipelineLogger(args.output_dir)
    workers     = args.workers
    pdf_workers = args.pdf_workers

    print(f"Found {len(pdf_files)} PDF(s) under {args.pdf_dir}")
    print(f"  API          : {args.api_url}")
    print(f"  Model        : {args.model}")
    print(f"  PDF workers  : {pdf_workers} concurrent PDFs")
    print(f"  Page workers : {workers} per PDF")
    print(f"  Total API    : up to {pdf_workers * workers} concurrent calls")
    print(f"  HTML skip-dir: {html_dir}")
    print(f"  YAML output  : {args.output_dir}/yaml/")
    print(f"  Logs         : {args.output_dir}/logs/yaml_progress.log\n")

    logger.session_start(len(pdf_files), args.output_dir, args.model, workers)

    t0 = time.time()
    pages_processed = 0

    try:
        if pdf_workers <= 1:
            # Sequential (original behaviour)
            for pdf_path in pdf_files:
                if pages_processed >= args.max_pages:
                    print(f"\nReached --max-pages limit ({args.max_pages}). Stopping.")
                    break
                print(f"\nProcessing: {pdf_path}")
                results = process_pdf_yaml(
                    pdf_path=pdf_path,
                    api_key=api_key,
                    output_dir=args.output_dir,
                    html_dir=html_dir,
                    max_dim=args.max_dim,
                    api_url=args.api_url,
                    model=args.model,
                    workers=workers,
                    logger=logger,
                )
                pages_processed += len(results)
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
                    results = process_pdf_yaml(
                        pdf_path=pdf_path,
                        api_key=api_key,
                        output_dir=args.output_dir,
                        html_dir=html_dir,
                        max_dim=args.max_dim,
                        api_url=args.api_url,
                        model=args.model,
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

            pages_processed = _page_total[0]

    except PaidModelError as e:
        print(f"\n{'='*60}")
        print(f"STOPPED: {e}")
        print(f"{'='*60}")
        logger.session_end(time.time() - t0)
        sys.exit(2)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        logger.session_end(time.time() - t0)


if __name__ == "__main__":
    main()
