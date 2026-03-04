"""
select_sc_sample.py — Randomly sample PDFs from data/sc/ up to a target page count.

Creates symlinks in data/sc_sample/ pointing to the selected PDFs.
Run generate_html.py on data/sc_sample/ to process only this subset.

Usage:
  python select_sc_sample.py                        # ~4500 pages, into data/sc_sample/
  python select_sc_sample.py --target-pages 6000    # larger sample
  python select_sc_sample.py --sc-dir data/sc/ --out-dir data/sc_sample/ --seed 99
"""

import argparse
import glob
import os
import random
import sys


def count_pages(pdf_path: str) -> int:
    """Return number of pages in a PDF, or 0 on error."""
    try:
        from pypdf import PdfReader
        return len(PdfReader(pdf_path).pages)
    except Exception:
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Randomly sample PDFs from sc/ up to a target page count",
    )
    parser.add_argument("--sc-dir",       default="data/sc",
                        help="Source directory of scanned PDFs (default: data/sc)")
    parser.add_argument("--out-dir",      default="data/sc_sample",
                        help="Output directory for symlinks (default: data/sc_sample)")
    parser.add_argument("--target-pages", type=int, default=1408,
                        help="Target total page count to sample (default: 4500)")
    parser.add_argument("--seed",         type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--dry-run",      action="store_true",
                        help="Print selected PDFs without creating symlinks")
    args = parser.parse_args()

    sc_dir  = os.path.abspath(args.sc_dir)
    out_dir = os.path.abspath(args.out_dir)

    if not os.path.isdir(sc_dir):
        print(f"ERROR: sc directory not found: {sc_dir}")
        sys.exit(1)

    pdf_files = sorted(glob.glob(os.path.join(sc_dir, "**", "*.pdf"), recursive=True))
    if not pdf_files:
        print(f"ERROR: No PDFs found in {sc_dir}")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDFs in {sc_dir}")
    print(f"Target: ~{args.target_pages} pages  (seed={args.seed})\n")

    # Shuffle randomly
    rng = random.Random(args.seed)
    rng.shuffle(pdf_files)

    # Accumulate until we reach the target
    selected = []
    total_pages = 0
    skipped = 0

    print("Counting pages...")
    for pdf_path in pdf_files:
        if total_pages >= args.target_pages:
            break
        n = count_pages(pdf_path)
        if n == 0:
            skipped += 1
            continue
        selected.append((pdf_path, n))
        total_pages += n
        if len(selected) % 100 == 0:
            print(f"  {len(selected)} PDFs selected, {total_pages} pages so far...")

    print(f"\nSelected {len(selected)} PDFs → {total_pages} total pages")
    if skipped:
        print(f"  ({skipped} PDFs skipped due to read errors)")

    if args.dry_run:
        print("\n--- DRY RUN: would create symlinks for ---")
        for pdf_path, n in selected:
            print(f"  {os.path.basename(pdf_path)}  ({n} pages)")
        return

    # Create output directory and symlinks
    os.makedirs(out_dir, exist_ok=True)

    created = 0
    already = 0
    for pdf_path, n in selected:
        link_name = os.path.join(out_dir, os.path.basename(pdf_path))
        if os.path.exists(link_name) or os.path.islink(link_name):
            already += 1
            continue
        os.symlink(pdf_path, link_name)
        created += 1

    print(f"\nDone!")
    print(f"  Created : {created} symlinks in {out_dir}")
    if already:
        print(f"  Skipped : {already} already existed")
    print(f"\nNow run:")
    print(f"  python -m synth.generate_html \\")
    print(f"    --pdf-dir {out_dir} \\")
    print(f"    --output-dir synth_output/ \\")
    print(f"    --workers 4")


if __name__ == "__main__":
    main()
