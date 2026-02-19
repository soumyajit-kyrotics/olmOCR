
import os
import shutil
import sys
from synth.run_pipeline import main as run_pipeline_main
from unittest.mock import patch

def test_pipeline():
    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    output_dir = os.path.join(base_dir, "synth_output_test")

    print(f"Data Dir: {data_dir}")
    print(f"Output Dir: {output_dir}")

    # Clean previous output
    if os.path.exists(output_dir):
        print("Cleaning previous output...")
        shutil.rmtree(output_dir)
    
    # Ensure data dir exists and has pdf
    if not os.path.exists(data_dir):
        print(f"FAILURE: Data directory {data_dir} does not exist.")
        return

    pdfs = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
    if not pdfs:
        print("FAILURE: No PDFs found in data directory. Please add a PDF to test.")
        return
    else:
        print(f"Found PDFs: {pdfs}")

    # Mock sys.argv
    # We use a small max-pages to ensure it runs quickly if there are many pages
    test_args = [
        "run_pipeline.py",
        "--pdf-dir", data_dir,
        "--output-dir", output_dir,
        "--max-pages", "2", 
        "--render-format", "png",
        "--api-url", "http://localhost:11434/v1/chat/completions",
        "--api-key", "ollama",
        "--model", "qwen2.5vl:7b"
    ]
    
    print(f"Running pipeline with args: {test_args}")
    
    # We need to make sure we're in the right directory or that imports work
    # Since we imported run_pipeline_main, it should share the context
    
    with patch.object(sys, 'argv', test_args):
        try:
            run_pipeline_main()
        except SystemExit as e:
            if e.code != 0:
                print(f"Pipeline failed with exit code {e.code}")
                return
            else:
                print("Pipeline completed successfully (exit code 0).")
        except Exception as e:
            print(f"Pipeline failed with error: {e}")
            import traceback
            traceback.print_exc()
            return

    # Verify outputs
    html_dir = os.path.join(output_dir, "html")
    rendered_dir = os.path.join(output_dir, "rendered")
    tests_path = os.path.join(output_dir, "tests.jsonl")

    success = True

    if not os.path.exists(html_dir) or not os.listdir(html_dir):
        print("FAILURE: No HTML files generated.")
        success = False
    else:
        print(f"SUCCESS: HTML files generated in {html_dir}")
        print(f"Files: {os.listdir(html_dir)}")

    if not os.path.exists(rendered_dir):
        print("FAILURE: Rendered directory not created.")
        success = False
    else:
        # Check for pngs
        pngs = [f for f in os.listdir(rendered_dir) if f.endswith('.png')]
        if not pngs:
             print("FAILURE: No rendered files found.")
             success = False
        else:
             print(f"SUCCESS: Rendered files found in {rendered_dir}")
             print(f"Files: {pngs}")

    if not os.path.exists(tests_path):
        print("FAILURE: tests.jsonl not created.")
        success = False
    else:
        print(f"SUCCESS: tests.jsonl created at {tests_path}")

    if success:
        print("\nTest passed successfully!")
    else:
        print("\nTest failed.")

if __name__ == "__main__":
    test_pipeline()
