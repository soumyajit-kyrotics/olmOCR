"""
main.py — CLI entry point for the OLMoCR-style PDF linearization pipeline.

This file lives at the project root (OLM-OCR/) so file paths like
'data/1950_1_25_29.pdf' resolve correctly relative to the project.

Usage:
  python main.py <pdf_path> [--api-key KEY] [--output-dir DIR] [--max-dim 2048]

Examples:
  # Using environment variable for API key
  export OPENROUTER_API_KEY="your-key-here"
  python main.py data/1950_1_25_29.pdf

  # With explicit options
  python main.py document.pdf --api-key sk-... --output-dir results --max-dim 1024
"""

import argparse
import os
import sys

from dotenv import load_dotenv
load_dotenv()

from src.pipeline import process_pdf


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF to linearized plain text using GPT-4o (OLMoCR-style)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "This tool implements the 'Generating Linearized Plain Text' approach\n"
            "from the OLMoCR paper (https://arxiv.org/abs/2502.18443).\n\n"
            "It uses document anchoring (text + image positions from the PDF's\n"
            "digital layer) to guide GPT-4o for accurate text extraction."
        ),
    )
    parser.add_argument(
        "pdf_path",
        help="Path to the PDF file to process",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENROUTER_API_KEY"),
        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory to save results (default: output)",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=2048,
        help="Max pixel dimension for rendered pages (default: 2048)",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-4o",
        help="Model to use via OpenRouter (default: openai/gpt-4o)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent page processing threads (default: 4)",
    )
    args = parser.parse_args()

    # Validate inputs
    if not args.api_key:
        print("ERROR: No API key provided.")
        print("  Use --api-key YOUR_KEY or set OPENROUTER_API_KEY environment variable.")
        sys.exit(1)

    if not os.path.exists(args.pdf_path):
        print(f"ERROR: PDF not found: {args.pdf_path}")
        sys.exit(1)

    # Run the pipeline
    process_pdf(args.pdf_path, args.api_key, args.output_dir, args.max_dim, args.workers)


if __name__ == "__main__":
    main()
