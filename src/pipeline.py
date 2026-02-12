"""
pipeline.py — Pipeline orchestration for processing PDFs page by page.

Ties together all modules (anchor, render, prompt, api) to process
entire PDF documents through the OLMoCR-style pipeline.

Pages are processed concurrently using a thread pool for faster throughput.

For each page:
  1. Extract anchor text (document anchoring)  →  anchor.py
  2. Render page to high-res PNG               →  render.py
  3. Build the OLMoCR prompt                    →  prompt.py
  4. Send to GPT-4o and parse response          →  api.py
  5. Save per-page JSON + combined plain text
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from pypdf import PdfReader

from .anchor import get_anchor_text
from .render import render_pdf_to_base64png
from .prompt import build_prompt
from .api import call_gpt4o


def process_page(pdf_path: str, page_num: int, api_key: str, max_dim: int = 2048) -> dict:
    """
    Process a single PDF page through the full OLMoCR-style pipeline.

    Args:
        pdf_path: path to the PDF file
        page_num: 1-indexed page number
        api_key: OpenRouter API key
        max_dim: max pixel dimension for rendered page image

    Returns:
        Parsed response dict with natural_text and metadata
    """
    print(f"  [Page {page_num}] Extracting anchor text...")
    anchor_text = get_anchor_text(pdf_path, page_num)

    print(f"  [Page {page_num}] Rendering page to PNG ({max_dim}px max)...")
    image_b64 = render_pdf_to_base64png(pdf_path, page_num, max_dim)

    print(f"  [Page {page_num}] Building prompt ({len(anchor_text)} chars anchor text)...")
    prompt = build_prompt(anchor_text)

    print(f"  [Page {page_num}] Calling GPT-4o...")
    result = call_gpt4o(prompt, image_b64, api_key)

    print(f"  [Page {page_num}] Done — language={result.get('primary_language')}, "
          f"table={result.get('is_table')}, diagram={result.get('is_diagram')}")
    return result


def _process_page_safe(pdf_path: str, page_num: int, api_key: str, max_dim: int) -> dict:
    """Wrapper that catches exceptions so one page failure doesn't kill the pool."""
    try:
        result = process_page(pdf_path, page_num, api_key, max_dim)
        return {"page": page_num, **result}
    except Exception as e:
        print(f"  [Page {page_num}] ERROR: {e}")
        return {"page": page_num, "error": str(e)}


def process_pdf(
    pdf_path: str,
    api_key: str,
    output_dir: str = "output",
    max_dim: int = 2048,
    max_workers: int = 4,
) -> list:
    """
    Process an entire PDF document through the OLMoCR-style pipeline.

    Pages are processed concurrently using a thread pool for faster throughput.
    Results are sorted back into page order for correct output.

    Args:
        pdf_path: path to the PDF file
        api_key: OpenRouter API key
        output_dir: directory to save output files (default: "output")
        max_dim: max pixel dimension for rendered page images
        max_workers: number of concurrent page processing threads (default: 4)

    Returns:
        List of response dicts, one per page (in page order)
    """
    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    print(f"Processing: {pdf_path} ({num_pages} pages, {max_workers} workers)")
    print(f"Output dir: {output_dir}")
    print("-" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Process pages concurrently
    all_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_page_safe, pdf_path, page_num, api_key, max_dim): page_num
            for page_num in range(1, num_pages + 1)
        }

        for future in as_completed(futures):
            result = future.result()
            all_results.append(result)

            # Save per-page result as it completes
            page_num = result["page"]
            page_json_path = os.path.join(output_dir, f"{pdf_name}_page_{page_num}.json")
            with open(page_json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

    # Sort results back into page order
    all_results.sort(key=lambda r: r["page"])

    # Collect text in page order
    all_texts = [r["natural_text"] for r in all_results if r.get("natural_text")]

    # Save combined results (all pages as JSON array)
    combined_json_path = os.path.join(output_dir, f"{pdf_name}_all_pages.json")
    with open(combined_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Save combined plain text (all natural_text joined with double newlines)
    combined_text_path = os.path.join(output_dir, f"{pdf_name}_linearized.txt")
    with open(combined_text_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_texts))

    print("-" * 60)
    print(f"Done! {num_pages} pages processed.")
    print(f"  Per-page JSON:   {output_dir}/{pdf_name}_page_*.json")
    print(f"  Combined JSON:   {combined_json_path}")
    print(f"  Linearized text: {combined_text_path}")

    return all_results
