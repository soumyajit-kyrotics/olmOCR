"""
run_pipeline.py — End-to-end synthetic data pipeline orchestrator.

Runs all 3 steps of the OLMoCR v2 synthetic data pipeline:
  1. generate_html — PDF pages → faithful HTML via VLM
  2. render_html  — HTML → PNG/PDF screenshots via Playwright
  3. generate_tests — HTML → unit tests JSONL

Usage:
  # Via OpenRouter:
  python -m synth.run_pipeline --pdf-dir data/ --output-dir synth_output/

  # Via local vLLM:
  python -m synth.run_pipeline --pdf-dir data/ \\
    --api-url http://localhost:8000/v1/chat/completions \\
    --model Qwen/Qwen2.5-VL-32B-Instruct-AWQ
"""

import argparse
import asyncio
import glob
import os
import sys
import time

from dotenv import load_dotenv

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(
        description="OLMoCR v2 Synthetic Data Pipeline (end-to-end)"
    )
    parser.add_argument("--pdf-dir", required=True, help="Directory containing PDF files")
    parser.add_argument("--output-dir", default="synth_output", help="Output directory")
    parser.add_argument("--max-pages", type=int, default=50, help="Max pages to process")
    parser.add_argument("--max-dim", type=int, default=2048, help="Max image dimension")
    parser.add_argument("--render-format", choices=["png", "pdf", "both"], default="png",
                        help="Rendering output format")
    parser.add_argument("--api-key", help="API key (or set OPENROUTER_API_KEY). Use 'none' for local vLLM.")
    parser.add_argument("--api-url", default="https://openrouter.ai/api/v1/chat/completions",
                        help="API endpoint URL (default: OpenRouter)")
    parser.add_argument("--model", default="qwen/qwen3-vl-235b-a22b-thinking",
                        help="Model name (default: qwen/qwen3-vl-235b-a22b-thinking)")
    parser.add_argument("--skip-render", action="store_true", help="Skip HTML rendering step")
    parser.add_argument("--skip-tests", action="store_true", help="Skip test extraction step")
    parser.add_argument("--workers", type=int, default=5,
                        help="Parallel workers per PDF (default: 5). Each worker makes one API call.")
    args = parser.parse_args()

    load_dotenv()
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY") or ""

    # For local vLLM, no API key is needed
    is_local = "localhost" in args.api_url or "127.0.0.1" in args.api_url
    if not api_key and not is_local:
        print("Error: No API key. Use --api-key or set OPENROUTER_API_KEY in .env")
        sys.exit(1)
    if is_local and not api_key:
        api_key = "none"

    os.makedirs(args.output_dir, exist_ok=True)
    html_dir = os.path.join(args.output_dir, "html")
    rendered_dir = os.path.join(args.output_dir, "rendered")
    tests_path = os.path.join(args.output_dir, "tests.jsonl")

    t0 = time.time()

    # ── Step 1: Generate HTML from PDFs ──────────────────────────────
    print("=" * 60)
    print("STEP 1: Generate HTML from PDF pages")
    print("=" * 60)

    from synth.generate_html import process_pdf

    # Recursively discover all PDFs under --pdf-dir (including subdirectories)
    pdf_files = sorted(glob.glob(
        os.path.join(args.pdf_dir, "**", "*.pdf"), recursive=True
    ))
    # Also pick up PDFs directly in the top-level dir (non-recursive glob misses none)
    pdf_files = sorted(set(pdf_files))  # deduplicate, keep sorted

    if not pdf_files:
        print(f"No PDFs found under {args.pdf_dir} (searched recursively)")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF(s) under {args.pdf_dir}")

    total_pages = 0
    for pdf_path in pdf_files:
        if total_pages >= args.max_pages:
            break
        print(f"\nProcessing: {pdf_path}")
        results = process_pdf(
            pdf_path, api_key, args.output_dir, args.max_dim,
            api_url=args.api_url, model=args.model,
            workers=args.workers,
        )
        total_pages += len(results)

    t1 = time.time()
    print(f"\n✓ Step 1 complete: {total_pages} pages → {html_dir}")
    print(f"  Time: {t1 - t0:.1f}s")

    # ── Step 2: Render HTML to PNG/PDF ───────────────────────────────
    if not args.skip_render:
        print("\n" + "=" * 60)
        print("STEP 2: Render HTML to images")
        print("=" * 60)

        from synth.render_html import render_directory

        if args.render_format in ("png", "both"):
            png_out = os.path.join(rendered_dir, "png") if args.render_format == "both" else rendered_dir
            n = asyncio.run(render_directory(html_dir, png_out, "png"))
            print(f"  Rendered {n} PNG files")

        if args.render_format in ("pdf", "both"):
            pdf_out = os.path.join(rendered_dir, "pdf") if args.render_format == "both" else rendered_dir
            n = asyncio.run(render_directory(html_dir, pdf_out, "pdf"))
            print(f"  Rendered {n} PDF files")

        t2 = time.time()
        print(f"\n✓ Step 2 complete → {rendered_dir}")
        print(f"  Time: {t2 - t1:.1f}s")
    else:
        t2 = t1
        print("\n⏭  Skipping Step 2 (--skip-render)")

    # ── Step 3: Extract unit tests from HTML ─────────────────────────
    if not args.skip_tests:
        print("\n" + "=" * 60)
        print("STEP 3: Extract unit tests from HTML")
        print("=" * 60)

        from synth.generate_tests import process_directory

        total_tests = process_directory(html_dir, tests_path)

        t3 = time.time()
        print(f"\n✓ Step 3 complete: {total_tests} tests → {tests_path}")
        print(f"  Time: {t3 - t2:.1f}s")
    else:
        t3 = t2
        print("\n⏭  Skipping Step 3 (--skip-tests)")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Total time: {t3 - t0:.1f}s")
    print(f"  Output directory: {args.output_dir}")
    print(f"  HTML files:  {html_dir}")
    if not args.skip_render:
        print(f"  Rendered:    {rendered_dir}")
    if not args.skip_tests:
        print(f"  Unit tests:  {tests_path}")


if __name__ == "__main__":
    main()
