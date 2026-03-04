"""
create_balanced_dataset.py — Build a balanced HTML dataset by:
  - Randomly sampling 1407 pages from hc (4413 generated)
  - Taking ALL generated pages from hi (1407)
  - Taking ALL generated pages from sc (~764)

Output: synth_output/html_balanced/ with symlinks to original files.

Usage:
  python create_balanced_dataset.py
  python create_balanced_dataset.py --hc-sample 1407 --out-dir synth_output/html_balanced/
"""

import argparse
import glob
import os
import random
import re
import sys


def get_pdf_stems(pdf_dir: str) -> set[str]:
    """Return set of PDF basenames (without .pdf) from a folder."""
    pdfs = glob.glob(os.path.join(pdf_dir, "**", "*.pdf"), recursive=True)
    return {re.sub(r"\.pdf$", "", os.path.basename(p), flags=re.IGNORECASE) for p in pdfs}


def find_html_pairs(stem: str, html_dir: str) -> list[tuple[str, str]]:
    """Return list of (html_path, json_path) for all pages of a given PDF stem."""
    pairs = []
    html_files = glob.glob(os.path.join(html_dir, f"{glob.escape(stem)}_page_*.html"))
    for html_path in html_files:
        json_path = html_path.replace(".html", ".json")
        if os.path.exists(json_path):
            pairs.append((html_path, json_path))
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Create a balanced HTML dataset from hc/hi/sc output"
    )
    parser.add_argument("--html-dir",  default="synth_output/html",
                        help="Source HTML directory (default: synth_output/html)")
    parser.add_argument("--hc-dir",    default="data/hc",
                        help="hc PDF directory   (default: data/hc)")
    parser.add_argument("--hi-dir",    default="data/hi",
                        help="hi PDF directory   (default: data/hi)")
    parser.add_argument("--sc-dir",    default="data/sc",
                        help="sc PDF directory   (default: data/sc)")
    parser.add_argument("--hc-sample", type=int, default=1407,
                        help="How many hc pages to randomly sample (default: 1407)")
    parser.add_argument("--out-dir",   default="synth_output/html_balanced",
                        help="Output directory for symlinks (default: synth_output/html_balanced)")
    parser.add_argument("--seed",      type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Print counts without creating symlinks")
    args = parser.parse_args()

    html_dir = os.path.abspath(args.html_dir)
    out_dir  = os.path.abspath(args.out_dir)
    rng      = random.Random(args.seed)

    print("Scanning PDF stems...")
    hc_stems = get_pdf_stems(args.hc_dir)
    hi_stems = get_pdf_stems(args.hi_dir)
    sc_stems = get_pdf_stems(args.sc_dir)

    print(f"  hc PDFs: {len(hc_stems)}")
    print(f"  hi PDFs: {len(hi_stems)}")
    print(f"  sc PDFs: {len(sc_stems)}")

    # Collect all (html, json) pairs per folder
    print("\nFinding generated HTML files...")

    def collect_pairs(stems):
        all_pairs = []
        for stem in stems:
            all_pairs.extend(find_html_pairs(stem, html_dir))
        return all_pairs

    hc_pairs = collect_pairs(hc_stems)
    hi_pairs = collect_pairs(hi_stems)
    sc_pairs = collect_pairs(sc_stems)

    print(f"  hc pages generated: {len(hc_pairs)}")
    print(f"  hi pages generated: {len(hi_pairs)}")
    print(f"  sc pages generated: {len(sc_pairs)}")

    # Randomly sample hc
    if len(hc_pairs) > args.hc_sample:
        rng.shuffle(hc_pairs)
        hc_selected = hc_pairs[:args.hc_sample]
        print(f"\n  hc: randomly selected {len(hc_selected)} / {len(hc_pairs)} pages")
    else:
        hc_selected = hc_pairs
        print(f"\n  hc: using all {len(hc_selected)} pages (fewer than target {args.hc_sample})")

    hi_selected = hi_pairs
    sc_selected = sc_pairs

    total = len(hc_selected) + len(hi_selected) + len(sc_selected)
    print(f"  hi: using all {len(hi_selected)} pages")
    print(f"  sc: using all {len(sc_selected)} pages")
    print(f"\nTotal selected: {total} pages")
    print(f"  hc={len(hc_selected)}  hi={len(hi_selected)}  sc={len(sc_selected)}")

    if args.dry_run:
        print("\nDry run — no symlinks created.")
        return

    os.makedirs(out_dir, exist_ok=True)
    created = 0
    skipped = 0

    all_pairs = hc_selected + hi_selected + sc_selected
    for html_path, json_path in all_pairs:
        for src in (html_path, json_path):
            link = os.path.join(out_dir, os.path.basename(src))
            if os.path.exists(link) or os.path.islink(link):
                skipped += 1
                continue
            os.symlink(src, link)
            created += 1

    print(f"\nDone!")
    print(f"  Created : {created} symlinks in {out_dir}")
    if skipped:
        print(f"  Skipped : {skipped} already existed")


if __name__ == "__main__":
    main()
