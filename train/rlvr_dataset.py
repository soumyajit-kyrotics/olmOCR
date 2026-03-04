"""
rlvr_dataset.py — Dataset class for RLVR (Reinforcement Learning with Verifiable Rewards).

Pairs rendered page images with their corresponding unit tests and prompts
for use in the GRPO training loop.

Input sources:
  - synth_output/rendered/*.png  — rendered page images
  - synth_output/tests.jsonl     — unit tests per page
  - data/*.pdf                   — original PDFs (for anchor text)

Output format per example:
  {
      "image": "/path/to/rendered/page.png",
      "prompt": "...olmOCR prompt with anchor text...",
      "tests": [list of test dicts for this page],
      "metadata": {"pdf": "...", "page": 1}
  }
"""

import glob
import json
import os
import re
import sys

from datasets import Dataset

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.anchor import get_anchor_text


# ---------------------------------------------------------------------------
# Prompt template (matches prepare_sft_data.py and src/prompt.py)
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = (
    "Below is the image of one page of a PDF document, as well as some raw textual content "
    "that was previously extracted for it that includes position information for each image "
    "and block of text (The origin [0x0] of the coordinates is in the lower left corner of the image). "
    "Just return the plain text representation of this document as if you were reading it naturally.\n"
    "Turn equations into a LaTeX representation, and tables into markdown format.\n"
    "IMPORTANT: Reflow the text into continuous paragraphs. Do NOT preserve the visual line breaks "
    "from the PDF layout. Join lines that belong to the same paragraph into a single flowing paragraph. "
    "If a word is split across lines with a hyphen (e.g., 'argu-\\nment'), rejoin it as one word ('argument').\n"
    "IMPORTANT: Extract ONLY the main body text of the page. You MUST exclude ALL of the following:\n"
    "  - Headers and footers (any repeating text at the top or bottom of the page)\n"
    "  - Page numbers (whether at the top, bottom, or margins of the page)\n"
    "  - Margin notes and side annotations\n"
    "  - Any text that appears outside the main text column/body area\n"
    "Keep references, footnotes, and citations that are part of the main body text.\n"
    "Read any natural handwriting.\n"
    "This is likely one page out of several in the document, so be sure to preserve "
    "any sentences that come from the previous page, or continue onto the next page, exactly as they are.\n"
    "If there is no text at all that you think you should read, output null for natural_text.\n"
    "Do not hallucinate.\n\n"
    "Respond in the following YAML format (do not wrap in code fences):\n"
    "primary_language: <two-letter language code, or null if no readable text>\n"
    "is_rotation_valid: <true if page is correctly oriented for reading, false otherwise>\n"
    "rotation_correction: <0, 90, 180, or 270 — degrees of clockwise rotation needed>\n"
    "is_table: <true if majority of page content is tabular>\n"
    "is_diagram: <true if majority of page content is a visual diagram>\n"
    "natural_text: |\n"
    "  <the extracted plain text, indented by 2 spaces>"
)


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def _load_tests_by_page(tests_path: str) -> dict:
    """
    Load tests.jsonl and group tests by (pdf_name, page) key.

    Returns:
        Dict mapping (pdf_name, page_num) -> list of test dicts
    """
    tests_by_page = {}
    with open(tests_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            test = json.loads(line)
            pdf_name = test.get("pdf", "").replace(".pdf", "")
            page = test.get("page", 0)
            key = (pdf_name, page)
            tests_by_page.setdefault(key, []).append(test)
    return tests_by_page


def build_rlvr_dataset(
    synth_dir: str,
    data_dir: str,
    min_tests: int = 2,
) -> Dataset:
    """
    Build the RLVR dataset from synth pipeline outputs.

    Pairs each rendered page image with its unit tests and prompt.
    Only includes pages that have at least `min_tests` tests.

    Args:
        synth_dir: synth pipeline output directory (contains rendered/, tests.jsonl)
        data_dir: directory containing original PDFs (for anchor text)
        min_tests: minimum number of tests required per page

    Returns:
        HuggingFace Dataset with columns: image, prompt, tests, metadata
    """
    rendered_dir = os.path.join(synth_dir, "rendered")
    tests_path = os.path.join(synth_dir, "tests.jsonl")

    if not os.path.exists(tests_path):
        raise FileNotFoundError(f"Tests file not found: {tests_path}")
    if not os.path.isdir(rendered_dir):
        raise FileNotFoundError(f"Rendered directory not found: {rendered_dir}")

    # Load and group tests
    tests_by_page = _load_tests_by_page(tests_path)
    print(f"Loaded tests for {len(tests_by_page)} pages")

    # Find rendered images
    png_files = sorted(glob.glob(os.path.join(rendered_dir, "*.png")))
    if not png_files:
        raise FileNotFoundError(f"No PNG files found in {rendered_dir}")

    examples = []
    skipped = 0

    for png_path in png_files:
        basename = os.path.splitext(os.path.basename(png_path))[0]

        # Parse filename: {pdf_name}_page_{N}
        match = re.match(r"(.+)_page_(\d+)$", basename)
        if not match:
            skipped += 1
            continue

        pdf_name = match.group(1)
        page_num = int(match.group(2))
        key = (pdf_name, page_num)

        # Get tests for this page
        page_tests = tests_by_page.get(key, [])
        if len(page_tests) < min_tests:
            skipped += 1
            continue

        # Build prompt with anchor text from source PDF
        pdf_path = os.path.join(data_dir, f"{pdf_name}.pdf")
        if os.path.exists(pdf_path):
            anchor_text = get_anchor_text(pdf_path, page_num)
            prompt = (
                PROMPT_TEMPLATE + "\n\n"
                f"<|anchor_start|>\n{anchor_text}\n<|anchor_end|>"
            )
        else:
            prompt = PROMPT_TEMPLATE

        examples.append({
            "image": os.path.abspath(png_path),
            "prompt": prompt,
            "tests": json.dumps(page_tests),  # Serialize as JSON string for Dataset
            "pdf_name": pdf_name,
            "page_num": page_num,
        })

    print(f"Built {len(examples)} RLVR examples (skipped {skipped})")

    return Dataset.from_list(examples)


# ---------------------------------------------------------------------------
# CLI — for inspecting the dataset
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build and inspect the RLVR dataset")
    parser.add_argument("--synth-dir", default="synth_output", help="Synth pipeline output dir")
    parser.add_argument("--data-dir", default="data", help="Original PDFs directory")
    parser.add_argument("--min-tests", type=int, default=2, help="Min tests per page")
    args = parser.parse_args()

    dataset = build_rlvr_dataset(args.synth_dir, args.data_dir, args.min_tests)

    print(f"\nDataset: {len(dataset)} examples")
    print(f"Columns: {dataset.column_names}")

    if len(dataset) > 0:
        ex = dataset[0]
        tests = json.loads(ex["tests"])
        print(f"\nExample 0:")
        print(f"  Image: {ex['image']}")
        print(f"  PDF: {ex['pdf_name']}, Page: {ex['page_num']}")
        print(f"  Prompt length: {len(ex['prompt'])} chars")
        print(f"  Tests: {len(tests)}")
        test_types = {}
        for t in tests:
            test_types[t["type"]] = test_types.get(t["type"], 0) + 1
        for ttype, count in sorted(test_types.items()):
            print(f"    {ttype}: {count}")


if __name__ == "__main__":
    main()
