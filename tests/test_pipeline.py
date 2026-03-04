
import glob
import os
import sys
from dotenv import load_dotenv
from synth.run_pipeline import main as run_pipeline_main
from unittest.mock import patch


def test_pipeline():
    load_dotenv()
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    if not openrouter_key:
        print("WARNING: OPENROUTER_API_KEY not found in .env file.")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir  = os.path.join(base_dir, "pdf_data")   # 10 test PDFs
    output_dir = os.path.join(base_dir, "synth_output_test")

    print(f"Data Dir:   {data_dir}")
    print(f"Output Dir: {output_dir}")

    if not os.path.exists(data_dir):
        print(f"FAILURE: {data_dir} does not exist.")
        return

    # Discover PDFs (recursive so structure doesn't matter)
    pdfs = glob.glob(os.path.join(data_dir, "**", "*.pdf"), recursive=True)
    if not pdfs:
        print(f"FAILURE: No PDFs found under {data_dir}")
        return
    print(f"Found {len(pdfs)} PDF(s)")

    # NOTE: We do NOT wipe output_dir between runs so resume/skip can be tested.
    # Delete output_dir manually if you want a clean run.
    os.makedirs(output_dir, exist_ok=True)

    test_args = [
        "run_pipeline.py",
        "--pdf-dir",     data_dir,
        "--output-dir",  output_dir,
        "--max-pages",   "9999",          # process all pages in the 10 test PDFs
        "--render-format", "png",
        "--api-url",     "https://openrouter.ai/api/v1/chat/completions",
        "--api-key",     openrouter_key,
        "--model",       "qwen/qwen3-vl-235b-a22b-thinking",
        "--workers",     "5",             # 5 pages per PDF in parallel
    ]

    print(f"\nRunning pipeline with args:\n  {' '.join(test_args)}\n")

    with patch.object(sys, "argv", test_args):
        try:
            run_pipeline_main()
        except SystemExit as e:
            if e.code != 0:
                print(f"Pipeline exited with code {e.code}")
                return
        except Exception as e:
            import traceback
            print(f"Pipeline raised an exception: {e}")
            traceback.print_exc()
            return

    # ── Verify outputs ────────────────────────────────────────────────
    html_dir     = os.path.join(output_dir, "html")
    rendered_dir = os.path.join(output_dir, "rendered")
    tests_path   = os.path.join(output_dir, "tests.jsonl")

    success = True

    html_files = glob.glob(os.path.join(html_dir, "*.html"))
    if not html_files:
        print("FAILURE: No HTML files generated.")
        success = False
    else:
        print(f"SUCCESS: {len(html_files)} HTML file(s) in {html_dir}")

    pngs = glob.glob(os.path.join(rendered_dir, "*.png"))
    if not pngs:
        print("FAILURE: No rendered PNG files found.")
        success = False
    else:
        print(f"SUCCESS: {len(pngs)} PNG file(s) in {rendered_dir}")

    if not os.path.exists(tests_path):
        print("FAILURE: tests.jsonl not created.")
        success = False
    else:
        with open(tests_path) as f:
            n_tests = sum(1 for _ in f)
        print(f"SUCCESS: tests.jsonl created ({n_tests} tests)")

    print("\nTest PASSED ✓" if success else "\nTest FAILED ✗")


if __name__ == "__main__":
    test_pipeline()
