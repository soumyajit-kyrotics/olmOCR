"""
prepare_sft_data.py — Convert pipeline outputs into SFT training data.

Generates (image + prompt → natural_text) triplets for supervised fine-tuning
of a Vision-Language Model to do OCR, following OLMoCR's training approach.

Page images are saved as PNG files to an images directory, and referenced
by file path in the training data (not embedded as base64).

Output format (JSONL):
  Each line is a JSON object with chat messages:
    {
      "messages": [
        {"role": "user", "content": [
          {"type": "text", "text": "<anchor prompt>"},
          {"type": "image", "image": "images/1950_1_25_29_page_1.png"}
        ]},
        {"role": "assistant", "content": "<natural_text>"}
      ],
      "metadata": {"source": "file.pdf", "page": 1, "language": "en", ...}
    }

Usage:
  # Process a single PDF (runs GPT-4o + saves training data)
  python prepare_sft_data.py data/1950_1_25_29.pdf --output train.jsonl

  # Process a folder of PDFs
  python prepare_sft_data.py data/ --output train.jsonl

  # Use existing pipeline outputs (skip re-running GPT-4o)
  python prepare_sft_data.py --from-outputs output/ --output train.jsonl
"""

import argparse
import base64
import glob
import json
import os
import sys

from src.anchor import get_anchor_text
from src.render import render_pdf_to_base64png
from src.prompt import build_prompt
from src.pipeline import process_page

from pypdf import PdfReader


def _save_page_image(pdf_path: str, page_num: int, images_dir: str, max_dim: int = 2048) -> str:
    """
    Render a PDF page to PNG and save it to the images directory.

    Returns the relative path to the saved image (e.g. 'images/doc_page_1.png').
    """
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    image_filename = f"{pdf_name}_page_{page_num}.png"
    image_path = os.path.join(images_dir, image_filename)

    # Render and decode from base64 to raw bytes, then write to disk
    image_b64 = render_pdf_to_base64png(pdf_path, page_num, max_dim)
    with open(image_path, "wb") as f:
        f.write(base64.b64decode(image_b64))

    return image_path


def build_sft_example(
    pdf_path: str,
    page_num: int,
    natural_text: str,
    metadata: dict,
    images_dir: str,
    max_dim: int = 2048,
) -> dict:
    """
    Build a single SFT training example.

    Saves the page image as a PNG file and returns a chat message pair:
      - User message: anchor prompt + image file path
      - Assistant message: the natural_text (ground truth)
    """
    # Generate anchor text and prompt
    anchor_text = get_anchor_text(pdf_path, page_num)
    prompt_text = build_prompt(anchor_text)

    # Save page image to disk and get its path
    image_path = _save_page_image(pdf_path, page_num, images_dir, max_dim)

    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image", "image": image_path},
                ],
            },
            {
                "role": "assistant",
                "content": natural_text,
            },
        ],
        "metadata": metadata,
    }


def from_existing_outputs(output_dir: str, data_dir: str, output_jsonl: str, images_dir: str, max_dim: int = 2048):
    """
    Convert existing pipeline output JSON files into SFT training data.
    Skips re-running GPT-4o — uses the natural_text already extracted.

    Args:
        output_dir: directory containing *_page_N.json files from the pipeline
        data_dir: directory containing the original PDF files
        output_jsonl: path to write the training JSONL file
    """
    page_files = sorted(glob.glob(os.path.join(output_dir, "*_page_*.json")))

    if not page_files:
        print(f"ERROR: No page JSON files found in {output_dir}")
        sys.exit(1)

    print(f"Found {len(page_files)} page outputs in {output_dir}")
    print(f"Saving page images to {images_dir}")
    os.makedirs(images_dir, exist_ok=True)

    count = 0
    skipped = 0
    with open(output_jsonl, "w", encoding="utf-8") as out:
        for page_file in page_files:
            with open(page_file) as f:
                page_data = json.load(f)

            natural_text = page_data.get("natural_text")
            if not natural_text:
                skipped += 1
                continue

            # Extract PDF name and page number from filename
            # Format: {pdf_name}_page_{N}.json
            basename = os.path.basename(page_file)
            parts = basename.rsplit("_page_", 1)
            pdf_name = parts[0]
            page_num = int(parts[1].replace(".json", ""))

            # Find the original PDF
            pdf_path = os.path.join(data_dir, f"{pdf_name}.pdf")
            if not os.path.exists(pdf_path):
                print(f"  WARNING: PDF not found for {basename}, skipping ({pdf_path})")
                skipped += 1
                continue

            print(f"  Processing {pdf_name} page {page_num}...")

            metadata = {
                "source": f"{pdf_name}.pdf",
                "page": page_num,
                "primary_language": page_data.get("primary_language"),
                "is_table": page_data.get("is_table", False),
                "is_diagram": page_data.get("is_diagram", False),
            }

            example = build_sft_example(pdf_path, page_num, natural_text, metadata, images_dir, max_dim)
            out.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1

    print(f"\nDone! Wrote {count} training examples to {output_jsonl}")
    if skipped:
        print(f"  Skipped {skipped} pages (no text or missing PDF)")


def from_pdfs(pdf_paths: list, api_key: str, output_jsonl: str, images_dir: str, max_dim: int = 2048):
    """
    Process PDFs end-to-end: run GPT-4o on each page and save as SFT training data.

    Args:
        pdf_paths: list of PDF file paths to process
        api_key: OpenRouter API key
        output_jsonl: path to write the training JSONL file
    """
    count = 0
    skipped = 0
    os.makedirs(images_dir, exist_ok=True)

    with open(output_jsonl, "w", encoding="utf-8") as out:
        for pdf_path in pdf_paths:
            reader = PdfReader(pdf_path)
            num_pages = len(reader.pages)
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

            print(f"\nProcessing: {pdf_path} ({num_pages} pages)")

            for page_num in range(1, num_pages + 1):
                try:
                    # Run the full pipeline for this page
                    result = process_page(pdf_path, page_num, api_key, max_dim)

                    natural_text = result.get("natural_text")
                    if not natural_text:
                        skipped += 1
                        continue

                    metadata = {
                        "source": f"{pdf_name}.pdf",
                        "page": page_num,
                        "primary_language": result.get("primary_language"),
                        "is_table": result.get("is_table", False),
                        "is_diagram": result.get("is_diagram", False),
                    }

                    example = build_sft_example(pdf_path, page_num, natural_text, metadata, images_dir, max_dim)
                    out.write(json.dumps(example, ensure_ascii=False) + "\n")
                    count += 1

                except Exception as e:
                    print(f"  [Page {page_num}] ERROR: {e}")
                    skipped += 1

    print(f"\nDone! Wrote {count} training examples to {output_jsonl}")
    if skipped:
        print(f"  Skipped {skipped} pages (errors or no text)")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare SFT training data from PDFs using OLMoCR-style extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # From existing pipeline outputs (no API calls needed):\n"
            "  python prepare_sft_data.py --from-outputs output/ --data-dir data/ -o train.jsonl\n\n"
            "  # Process new PDFs end-to-end:\n"
            "  python prepare_sft_data.py data/doc.pdf -o train.jsonl\n\n"
            "  # Process all PDFs in a folder:\n"
            "  python prepare_sft_data.py data/ -o train.jsonl\n"
        ),
    )

    # Input source (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "pdf_input",
        nargs="?",
        help="Path to a PDF file or directory of PDFs to process",
    )
    group.add_argument(
        "--from-outputs",
        help="Use existing pipeline output directory (skip GPT-4o calls)",
    )

    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing original PDFs (used with --from-outputs, default: data)",
    )
    parser.add_argument(
        "-o", "--output",
        default="train.jsonl",
        help="Output JSONL file path (default: train.jsonl)",
    )
    parser.add_argument(
        "--images-dir",
        default="images",
        help="Directory to save page images (default: images)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENROUTER_API_KEY"),
        help="OpenRouter API key (needed when processing new PDFs)",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=2048,
        help="Max pixel dimension for rendered pages (default: 2048)",
    )
    args = parser.parse_args()

    if args.from_outputs:
        # Mode 1: Convert existing pipeline outputs
        from_existing_outputs(args.from_outputs, args.data_dir, args.output, args.images_dir, args.max_dim)
    else:
        # Mode 2: Process PDFs end-to-end
        if not args.api_key:
            print("ERROR: API key required when processing new PDFs.")
            print("  Use --api-key or set OPENROUTER_API_KEY env var.")
            sys.exit(1)

        # Collect PDF paths
        if os.path.isdir(args.pdf_input):
            pdf_paths = sorted(glob.glob(os.path.join(args.pdf_input, "*.pdf")))
            if not pdf_paths:
                print(f"ERROR: No PDF files found in {args.pdf_input}")
                sys.exit(1)
        else:
            pdf_paths = [args.pdf_input]

        from_pdfs(pdf_paths, args.api_key, args.output, args.images_dir, args.max_dim)


if __name__ == "__main__":
    main()
