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
import json
import os
import re
import sys

import requests
from dotenv import load_dotenv

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.render import render_pdf_to_base64png


# ---------------------------------------------------------------------------
# API defaults (overridable via CLI)
# ---------------------------------------------------------------------------

DEFAULT_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "qwen/qwen-2.5-vl-7b-instruct"


def _call_api(
    api_key: str,
    messages: list,
    api_url: str = DEFAULT_API_URL,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 16000,
    temperature: float = 0.2,
) -> str:
    """Send a request to the VLM API and return the text response."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    max_retries = 3
    for attempt in range(max_retries):
        response = requests.post(api_url, headers=headers, json=payload, timeout=300)
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        # Retry on transient server errors
        if response.status_code in (500, 502, 503, 504) and attempt < max_retries - 1:
            wait = 2 ** attempt * 5  # 5s, 10s, 20s
            print(f"  API Error {response.status_code}, retrying in {wait}s (attempt {attempt + 1}/{max_retries})...")
            import time
            time.sleep(wait)
            continue
        print(f"API Error {response.status_code}: {response.text[:500]}")
        response.raise_for_status()
    raise RuntimeError("Max retries exceeded")


# ---------------------------------------------------------------------------
# Step 1: Layout Analysis
# ---------------------------------------------------------------------------

def _analyze_layout(api_key: str, image_base64: str, api_url: str = DEFAULT_API_URL, model: str = DEFAULT_MODEL) -> str:
    """
    Analyze the page layout before generating HTML.
    Returns a text description of the layout structure.
    """
    messages = [
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
                        "Analyze the layout of this document page. Describe:\n"
                        "1. How many columns does the text have? How are they arranged?\n"
                        "2. Are there any tables? If so, how many rows and columns?\n"
                        "3. Are there any images or figures? Where are they positioned?\n"
                        "4. Are there headers, footers, or page numbers?\n"
                        "5. Are there any math equations or special formatting?\n"
                        "6. Is there any complex formatting that would be challenging to reproduce in HTML?\n\n"
                        "Please be very precise about the number of columns and how they're arranged."
                    ),
                },
            ],
        }
    ]
    return _call_api(api_key, messages, api_url=api_url, model=model, max_tokens=2000)


# ---------------------------------------------------------------------------
# Step 2: HTML Generation
# ---------------------------------------------------------------------------

def _generate_html(
    api_key: str, image_base64: str, analysis_text: str, width: int, height: int,
    api_url: str = DEFAULT_API_URL, model: str = DEFAULT_MODEL,
) -> str | None:
    """
    Produce semantic HTML that reproduces the page.
    Returns the extracted HTML string, or None if extraction fails.
    """
    a4_height = round(width * 1.414)

    html_template = (
        "<!DOCTYPE html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "    <meta charset=\"UTF-8\">\n"
        "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
        "    <title>PAGE TITLE HERE</title>\n"
        "    <style>\n"
        f"        body {{\n"
        f"            width: {width}px;\n"
        f"            min-height: {a4_height}px;\n"
        "            overflow: visible;\n"
        "            box-sizing: border-box;\n"
        "            font-family: 'Times New Roman', Georgia, serif;\n"
        "            font-size: 18pt;\n"
        "            line-height: 1.5;\n"
        "            padding: 80px 100px;\n"
        "            margin: 0;\n"
        "        }\n"
        "        header { text-align: center; margin-bottom: 2em; position: relative; }\n"
        "        .page-number { position: absolute; left: 0; top: 0; font-size: 20pt; font-weight: bold; }\n"
        "        .header-text { font-size: 18pt; font-weight: bold; letter-spacing: 0.1em; }\n"
        "        .page-layout { display: flex; gap: 30px; }\n"
        "        aside { flex: 0 0 180px; font-size: 16pt; line-height: 1.4; padding-right: 15px; }\n"
        "        .main-col { flex: 1; }\n"
        "        .case-name { font-style: italic; margin-bottom: 1em; }\n"
        "        .judge-name { font-style: italic; margin-top: 1.5em; }\n"
        "        p { margin: 1.3em 0; text-align: justify; text-indent: 2em; }\n"
        "        p.no-indent { text-indent: 0; }\n"
        "        .italic { font-style: italic; }\n"
        "        .appeal-heading { text-align: center; margin: 1.5em 0; font-size: 18pt; }\n"
        "        .counsel { margin: 1.2em 0; text-indent: 0; }\n"
        "        .date { margin: 1.2em 0; font-weight: bold; }\n"
        "        .conclusion { text-align: center; font-style: italic; margin: 2em 0; }\n"
        "    </style>\n"
        "</head>\n"
        "<body>\n"
        "    <header>\n"
        "        <div class=\"page-number\">PAGE_NUM</div>\n"
        "        <div class=\"header-text\">HEADER TEXT</div>\n"
        "    </header>\n"
        "    <div class=\"page-layout\">\n"
        "        <aside>\n"
        "            <div style=\"margin-bottom: 1.5em;\">YEAR</div>\n"
        "            <div class=\"case-name\">Party1<br>v.<br>Party2</div>\n"
        "            <div class=\"judge-name\">Judge Name J.</div>\n"
        "        </aside>\n"
        "        <div class=\"main-col\">\n"
        "            <!-- MAIN CONTENT PARAGRAPHS HERE -->\n"
        "        </div>\n"
        "    </div>\n"
        "</body>\n"
        "</html>"
    )

    messages = [
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
                        "Create HTML that **exactly reproduces** this PDF page. "
                        "The rendered HTML must be VISUALLY INDISTINGUISHABLE from the original.\n\n"
                        f"Layout analysis:\n{analysis_text}\n\n"
                        "=== MANDATORY RULES ===\n\n"
                        "1. You MUST follow the EXACT HTML template below. "
                        "Do NOT deviate from this structure. Fill in the content from the image.\n\n"
                        f"```html\n{html_template}\n```\n\n"
                        "2. STRICT BANS — the following will BREAK rendering:\n"
                        "   - NO inline style= attributes (use CSS classes only)\n"
                        "   - NO position: fixed or position: absolute\n"
                        "   - NO <h1> tags for headers (use <div class=\"header-text\"> inside <header>)\n\n"
                        "3. HEADER FORMAT: Use the <header> structure exactly as shown:\n"
                        "   - <div class=\"page-number\"> for the page number (left-aligned)\n"
                        "   - <div class=\"header-text\"> for the running header text (centered)\n\n"
                        "4. ASIDE (margin notes): The aside column is for case citations that "
                        "appear in the margin of the original page. Structure the aside as:\n"
                        "   - Year in its own <div>\n"
                        "   - Case name in <div class=\"case-name\"> with <br> between party names\n"
                        "   - Judge name in <div class=\"judge-name\">\n"
                        "   The aside MUST use flex: 0 0 180px (fixed 180px width). "
                        "If no margin notes exist on this page, omit the <aside> entirely and "
                        "remove the .page-layout wrapper.\n\n"
                        "5. CSS CLASSES to use for different content types:\n"
                        "   - Normal paragraphs: <p> (auto text-indent: 2em)\n"
                        "   - First paragraph or non-indented: <p class=\"no-indent\">\n"
                        "   - Italic text like \"Held\": <span class=\"italic\">Held</span>\n"
                        "   - Appeal headings: <p class=\"appeal-heading no-indent\">\n"
                        "   - Counsel names: <p class=\"counsel no-indent\">\n"
                        "   - Dates: <p class=\"date no-indent\">\n"
                        "   - \"Appeal dismissed\" type conclusions: <p class=\"conclusion\">\n\n"
                        "6. TEXT ACCURACY: Copy ALL text from the image exactly. "
                        "Do not summarize or skip any paragraphs. Every word must be present.\n\n"
                        "Enclose your final HTML in a ```html code block."
                    ),
                },
            ],
        }
    ]
    response_text = _call_api(api_key, messages, api_url=api_url, model=model, max_tokens=16000)
    html = _extract_html_block(response_text)
    if html:
        html = _inject_viewport_css(html, width, height)
    return html


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
    Post-process HTML to ensure it has correct portrait viewport constraints.
    Injects a <style> block BEFORE </head> so it overrides the LLM's CSS.
    Uses !important to enforce body constraints.
    """
    # Force A4 aspect ratio
    a4_height = round(width * 1.414)
    viewport_css = (
        f"\n<style>\n"
        f"  /* Injected viewport constraints — override LLM CSS */\n"
        f"  html {{ margin: 0 !important; padding: 0 !important; }}\n"
        f"  body {{\n"
        f"    /* --- THE FIX: AGGRESSIVE SQUEEZE --- */\n"
        f"    /* 1. Constrain content to 55% width to match the book's narrow column */\n"
        f"    width: 55% !important;\n"
        f"    /* 2. Center the column in the middle of the page */\n"
        f"    margin: 0 auto !important;\n"
        f"    \n"
        f"    min-height: {a4_height}px !important;\n"
        f"    overflow: visible !important;\n"
        f"    box-sizing: border-box !important;\n"
        f"    \n"
        f"    /* 3. Vertical padding only (horizontal handled by width) */\n"
        f"    padding: 60px 0px !important;\n"
        f"    \n"
        f"    font-family: 'Times New Roman', Georgia, serif !important;\n"
        f"    /* 4. Increase font size to 18pt to fill vertical space */\n"
        f"    font-size: 18pt !important;\n"
        f"    line-height: 1.5 !important;\n"
        f"    text-align: justify !important;\n"
        f"  }}\n"
        f"</style>\n"
    )

    # Inject BEFORE </head> so it takes precedence over LLM styles
    if "</head>" in html:
        html = html.replace("</head>", f"{viewport_css}</head>", 1)
    elif "<body" in html:
        # No </head> — inject before <body>
        body_idx = html.index("<body")
        html = html[:body_idx] + viewport_css + html[body_idx:]
    else:
        # Fallback — prepend
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
    Process a single PDF page through the synthetic data pipeline.

    Returns a dict with:
      - html: the generated HTML string
      - page: page number
      - source: source PDF path
      - width/height: dimensions of the rendered page
    """
    print(f"  [Page {page_num}] Rendering to PNG...")
    image_base64 = render_pdf_to_base64png(pdf_path, page_num, max_dim)

    # Get image dimensions from the base64 PNG
    png_bytes = base64.b64decode(image_base64)
    width, height = _get_png_dimensions(png_bytes)
    print(f"  [Page {page_num}] Image size: {width}x{height}")

    print(f"  [Page {page_num}] Analyzing layout...")
    analysis = _analyze_layout(api_key, image_base64, api_url=api_url, model=model)
    print(f"  [Page {page_num}] Layout analysis done ({len(analysis)} chars)")

    print(f"  [Page {page_num}] Generating HTML...")
    html = _generate_html(api_key, image_base64, analysis, width, height, api_url=api_url, model=model)

    if html is None:
        print(f"  [Page {page_num}] WARNING: Failed to extract HTML from response")
        return None

    print(f"  [Page {page_num}] HTML generated ({len(html)} chars)")
    return {
        "html": html,
        "page": page_num,
        "source": pdf_path,
        "width": width,
        "height": height,
        "analysis": analysis,
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
) -> list:
    """Process all pages of a PDF and save generated HTML files."""
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    html_dir = os.path.join(output_dir, "html")
    os.makedirs(html_dir, exist_ok=True)

    results = []
    for page_num in range(1, num_pages + 1):
        result = process_page(pdf_path, page_num, api_key, max_dim, api_url=api_url, model=model)
        if result is None:
            continue

        # Save HTML file
        html_path = os.path.join(html_dir, f"{pdf_name}_page_{page_num}.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(result["html"])

        # Save metadata
        meta_path = os.path.join(html_dir, f"{pdf_name}_page_{page_num}.json")
        meta = {k: v for k, v in result.items() if k != "html"}
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        results.append(result)
        print(f"  [Page {page_num}] Saved → {html_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
            "    --model Qwen/Qwen2.5-VL-32B-Instruct-AWQ\n"
        ),
    )
    parser.add_argument("--pdf-dir", required=True, help="Directory containing PDF files")
    parser.add_argument("--output-dir", default="synth_output", help="Output directory")
    parser.add_argument("--max-pages", type=int, default=50, help="Max pages to process across all PDFs")
    parser.add_argument("--max-dim", type=int, default=2048, help="Max image dimension in pixels")
    parser.add_argument("--api-key", help="API key (or set OPENROUTER_API_KEY in .env). Use 'none' for local vLLM.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL,
                        help=f"API endpoint URL (default: {DEFAULT_API_URL})")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Model name (default: {DEFAULT_MODEL})")
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

    # Collect PDF files
    pdf_files = sorted([
        os.path.join(args.pdf_dir, f)
        for f in os.listdir(args.pdf_dir)
        if f.lower().endswith(".pdf")
    ])

    if not pdf_files:
        print(f"No PDFs found in {args.pdf_dir}")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF(s) in {args.pdf_dir}")
    print(f"  API: {args.api_url}")
    print(f"  Model: {args.model}")
    os.makedirs(args.output_dir, exist_ok=True)

    total_pages = 0
    for pdf_path in pdf_files:
        if total_pages >= args.max_pages:
            break

        print(f"\nProcessing: {pdf_path}")
        results = process_pdf(
            pdf_path, api_key, args.output_dir, args.max_dim,
            api_url=args.api_url, model=args.model,
        )
        total_pages += len(results)

        if total_pages >= args.max_pages:
            print(f"Reached max pages limit ({args.max_pages})")
            break

    print(f"\nDone! Generated HTML for {total_pages} pages → {args.output_dir}/html/")


if __name__ == "__main__":
    main()
