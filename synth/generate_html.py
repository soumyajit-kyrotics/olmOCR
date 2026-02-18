"""
generate_html.py — Convert PDF pages to faithful HTML using Claude Sonnet.

Uses Claude Sonnet via OpenRouter to:
  1. Analyze the layout of a PDF page (columns, tables, math, images)
  2. Generate clean semantic HTML that faithfully reproduces the page

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
# Claude Sonnet via OpenRouter
# ---------------------------------------------------------------------------

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "anthropic/claude-sonnet-4.5"


def _call_claude(api_key: str, messages: list, max_tokens: int = 16000, temperature: float = 0.2) -> str:
    """Send a request to Claude Sonnet via OpenRouter and return the text response."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=180)
    if response.status_code != 200:
        print(f"API Error {response.status_code}: {response.text[:500]}")
        response.raise_for_status()
    result = response.json()
    return result["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Step 1: Layout Analysis
# ---------------------------------------------------------------------------

def _analyze_layout(api_key: str, image_base64: str) -> str:
    """
    Ask Claude to analyze the page layout before generating HTML.
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
    return _call_claude(api_key, messages, max_tokens=2000)


# ---------------------------------------------------------------------------
# Step 2: HTML Generation
# ---------------------------------------------------------------------------

def _generate_html(api_key: str, image_base64: str, analysis_text: str, width: int, height: int) -> str | None:
    """
    Ask Claude to produce semantic HTML that reproduces the page.
    Returns the extracted HTML string, or None if extraction fails.
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
                        "Your goal is to create HTML that, when rendered at the exact page dimensions, "
                        "is VISUALLY INDISTINGUISHABLE from the original PDF page. "
                        "The rendered HTML screenshot should look like a photocopy of the original.\n\n"
                        f"Here's my analysis of the document structure:\n\n{analysis_text}\n\n"
                        "Requirements:\n\n"
                        "--- PAGE DIMENSIONS ---\n"
                        f"1. This page is exactly {width}px wide. The viewport has A4 aspect ratio "
                        f"(height = {round(width * 1.414)}px). Your HTML body MUST use: "
                        f"width: {width}px; min-height: {round(width * 1.414)}px; "
                        "overflow: hidden; box-sizing: border-box; "
                        "font-family: Times New Roman, Georgia, serif; font-size: 14pt; line-height: 1.5;\n\n"
                        "--- VISUAL FIDELITY (HIGHEST PRIORITY) ---\n"
                        "2. MARGINS: Carefully estimate the margins of the original page. "
                        "Set body padding to replicate the exact same whitespace around the text block. "
                        "Typical book pages have ~60-80px padding on each side. The text block width "
                        "and position must match the original so line breaks occur at the same points.\n"
                        "3. TYPOGRAPHY: Match the original font sizes. This document appears to use LARGE print relative to the page size. "
                        "Use font-size: 18pt and line-height: 1.5. "
                        "The goal is matching the text density — same number of lines per paragraph.\n"
                        "Headings should be proportionally larger. The goal is matching the text density — same number of lines per paragraph.\n"
                        "4. SPACING AND PAGE FILLING: The content MUST fill the page vertically similar to "
                        "the original. If the original has text near the bottom of the page, your HTML must too. "
                        "Use generous line-height (1.5-1.8), paragraph margins (1.2em-1.5em), and padding "
                        "to spread content vertically. A paragraph at the bottom of the original page should "
                        "appear at roughly the same vertical position in the HTML.\n\n"
                        "--- SEMANTIC STRUCTURE ---\n"
                        "5. Use appropriate HTML tags: h1-h6 for headings, p for paragraphs, etc.\n"
                        "6. Page numbers and running headers/footers MUST be inside <header> or <footer> tags.\n"
                        "7. Images: use <div class='image'> with grey background and black outline, preserving "
                        "original size and position. Include data-description, data-x, data-y, data-width, "
                        "data-height attributes.\n"
                        "8. Math: use \\[ \\] or \\( \\) delimiters for LaTeX.\n\n"
                        "--- LAYOUT ---\n"
                        "9. COLUMNS: If the document has multi-column layout, preserve the exact same "
                        "number of columns using CSS flexbox or grid.\n"
                        "10. MARGIN NOTES: Any margin notes, side notes, or marginal annotations MUST be "
                        "wrapped in <aside> tags. Use a CSS flexbox layout: main content takes ~75-80%% width, "
                        "aside column takes ~20-25%% width. Do NOT use position:absolute — it causes overlap. "
                        "Example: .page-layout { display: flex; } .main-col { flex: 3; } aside { flex: 1; }\n\n"
                        "Enclose your HTML in a ```html code block."
                    ),
                },
            ],
        }
    ]
    response_text = _call_claude(api_key, messages, max_tokens=16000)
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
        f"    overflow: hidden !important;\n"
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

def process_page(pdf_path: str, page_num: int, api_key: str, max_dim: int = 2048) -> dict | None:
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

    print(f"  [Page {page_num}] Analyzing layout with Claude Sonnet...")
    analysis = _analyze_layout(api_key, image_base64)
    print(f"  [Page {page_num}] Layout analysis done ({len(analysis)} chars)")

    print(f"  [Page {page_num}] Generating HTML with Claude Sonnet...")
    html = _generate_html(api_key, image_base64, analysis, width, height)

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

def process_pdf(pdf_path: str, api_key: str, output_dir: str, max_dim: int = 2048) -> list:
    """Process all pages of a PDF and save generated HTML files."""
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    html_dir = os.path.join(output_dir, "html")
    os.makedirs(html_dir, exist_ok=True)

    results = []
    for page_num in range(1, num_pages + 1):
        result = process_page(pdf_path, page_num, api_key, max_dim)
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
        description="Generate faithful HTML from PDF pages using Claude Sonnet (via OpenRouter)"
    )
    parser.add_argument("--pdf-dir", required=True, help="Directory containing PDF files")
    parser.add_argument("--output-dir", default="synth_output", help="Output directory")
    parser.add_argument("--max-pages", type=int, default=50, help="Max pages to process across all PDFs")
    parser.add_argument("--max-dim", type=int, default=2048, help="Max image dimension in pixels")
    parser.add_argument("--api-key", help="OpenRouter API key (or set OPENROUTER_API_KEY)")
    args = parser.parse_args()

    load_dotenv()
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: No API key. Use --api-key or set OPENROUTER_API_KEY in .env")
        sys.exit(1)

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
    os.makedirs(args.output_dir, exist_ok=True)

    total_pages = 0
    for pdf_path in pdf_files:
        if total_pages >= args.max_pages:
            break

        print(f"\nProcessing: {pdf_path}")
        results = process_pdf(pdf_path, api_key, args.output_dir, args.max_dim)
        total_pages += len(results)

        if total_pages >= args.max_pages:
            print(f"Reached max pages limit ({args.max_pages})")
            break

    print(f"\nDone! Generated HTML for {total_pages} pages → {args.output_dir}/html/")


if __name__ == "__main__":
    main()
