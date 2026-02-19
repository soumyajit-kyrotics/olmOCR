"""
prepare_sft_data.py — Convert synth pipeline outputs into SFT training data.

Generates (image + prompt → YAML) training pairs for supervised fine-tuning
of a Vision-Language Model to do OCR, following OLMoCR's training approach.

The assistant response is formatted as YAML matching the OLMoCR output schema:
  primary_language: en
  is_rotation_valid: true
  rotation_correction: 0
  is_table: false
  is_diagram: false
  natural_text: |
    extracted body text here...

Page images are saved as PNG files to an images directory, and referenced
by file path in the training data (not embedded as base64).

Usage:
  python prepare_sft_data.py --from-html synth_output/ --data-dir data/ -o train.jsonl
"""

import argparse
import glob
import json
import os
import re
import sys
import textwrap

from bs4 import BeautifulSoup
from PIL import Image

from src.anchor import get_anchor_text


# ---------------------------------------------------------------------------
# Prompt template
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
# HTML parsing
# ---------------------------------------------------------------------------

def _extract_text_from_html(html_path: str) -> str:
    """
    Extract body text from a synth pipeline HTML file.

    Extracts content from .main-col only, excluding <header> and <aside>.
    <table> elements are preserved as raw HTML (with <tr>, <td> tags intact).
    All other elements are extracted as plain text via .get_text(strip=True).
    Blocks are joined with double newlines.
    """
    with open(html_path, encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    # Remove header and aside elements entirely
    for tag in soup.find_all(["header", "aside"]):
        tag.decompose()

    # Select the content root
    root = soup.find(class_="main-col") or soup.find("body")
    if not root:
        return ""

    # Walk direct children — preserve <table> as raw HTML, extract text otherwise
    blocks = []
    for child in root.children:
        # Skip bare whitespace / newline nodes
        if isinstance(child, str):
            stripped = child.strip()
            if stripped:
                blocks.append(stripped)
            continue

        # Preserve raw HTML for tables (keeps <table>, <tr>, <td>, etc.)
        if child.name == "table":
            blocks.append(str(child))
            continue

        # For everything else, extract plain text
        text = child.get_text(separator="\n", strip=True)
        if text:
            blocks.append(text)

    result = "\n\n".join(blocks)
    # Clean up excessive whitespace
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def _detect_html_metadata(html_path: str) -> dict:
    """
    Detect metadata fields from the HTML structure.

    Returns dict with primary_language, is_table, is_diagram.
    """
    with open(html_path, encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    # Detect language from <html lang="..."> attribute
    html_tag = soup.find("html")
    lang = None
    if html_tag and html_tag.get("lang"):
        lang = html_tag["lang"][:2].lower()  # e.g. "en-US" → "en"

    # Detect if majority of content is a table
    main_col = soup.find(class_="main-col") or soup.find("body")
    is_table = False
    is_diagram = False
    if main_col:
        tables = main_col.find_all("table")
        all_text = main_col.get_text(strip=True)
        table_text = "".join(t.get_text(strip=True) for t in tables)
        if all_text and len(table_text) > len(all_text) * 0.5:
            is_table = True

        # Detect diagram-heavy pages (many images, little text)
        images = main_col.find_all("img")
        if len(images) > 0 and len(all_text) < 100:
            is_diagram = True

    return {
        "primary_language": lang or "en",
        "is_table": is_table,
        "is_diagram": is_diagram,
    }


# ---------------------------------------------------------------------------
# YAML formatting
# ---------------------------------------------------------------------------

def _format_as_yaml(
    natural_text: str,
    primary_language: str = "en",
    is_rotation_valid: bool = True,
    rotation_correction: int = 0,
    is_table: bool = False,
    is_diagram: bool = False,
) -> str:
    """
    Format the extracted text as OLMoCR-style YAML output.

    This matches the YAML schema the model is prompted to produce:
      primary_language: en
      is_rotation_valid: true
      rotation_correction: 0
      is_table: false
      is_diagram: false
      natural_text: |
        the text indented by 2 spaces
    """
    # Indent each line of natural_text by 2 spaces for YAML block scalar
    indented = textwrap.indent(natural_text, "  ")

    yaml_str = (
        f"primary_language: {primary_language}\n"
        f"is_rotation_valid: {'true' if is_rotation_valid else 'false'}\n"
        f"rotation_correction: {rotation_correction}\n"
        f"is_table: {'true' if is_table else 'false'}\n"
        f"is_diagram: {'true' if is_diagram else 'false'}\n"
        f"natural_text: |\n"
        f"{indented}"
    )
    return yaml_str


# ---------------------------------------------------------------------------
# from_html mode (primary)
# ---------------------------------------------------------------------------

def from_html(synth_dir: str, data_dir: str, output_jsonl: str, images_dir: str):
    """
    Build SFT training data from synth pipeline HTML outputs.

    Reads HTML files from synth_dir/html/, extracts body text (excluding
    header/aside), wraps it in YAML format, pairs with rendered PNGs
    from synth_dir/rendered/, and writes training examples to output_jsonl.

    Args:
        synth_dir: synth pipeline output directory (contains html/ and rendered/)
        data_dir: directory containing original PDFs (for anchor text fallback)
        output_jsonl: path to write the training JSONL file
        images_dir: directory to copy/save page images
    """
    html_dir = os.path.join(synth_dir, "html")
    rendered_dir = os.path.join(synth_dir, "rendered")

    if not os.path.isdir(html_dir):
        print(f"ERROR: HTML directory not found: {html_dir}")
        sys.exit(1)

    html_files = sorted(glob.glob(os.path.join(html_dir, "*.html")))
    if not html_files:
        print(f"ERROR: No HTML files found in {html_dir}")
        sys.exit(1)

    print(f"Found {len(html_files)} HTML files in {html_dir}")
    os.makedirs(images_dir, exist_ok=True)

    count = 0
    skipped = 0
    with open(output_jsonl, "w", encoding="utf-8") as out:
        for html_path in html_files:
            basename = os.path.basename(html_path)  # e.g. 1950_1_25_29_page_1.html
            stem = os.path.splitext(basename)[0]     # e.g. 1950_1_25_29_page_1

            # Parse filename: {pdf_name}_page_{N}
            match = re.match(r"(.+)_page_(\d+)$", stem)
            if not match:
                print(f"  WARNING: Unexpected filename format: {basename}, skipping")
                skipped += 1
                continue

            pdf_name = match.group(1)
            page_num = int(match.group(2))

            # Extract text from HTML
            natural_text = _extract_text_from_html(html_path)
            if not natural_text:
                print(f"  WARNING: No text extracted from {basename}, skipping")
                skipped += 1
                continue

            # Detect metadata from HTML
            meta = _detect_html_metadata(html_path)

            # Format assistant response as YAML
            yaml_response = _format_as_yaml(
                natural_text=natural_text,
                primary_language=meta["primary_language"],
                is_table=meta["is_table"],
                is_diagram=meta["is_diagram"],
            )

            # Find rendered PNG, resize to 1288px max edge, and save
            png_name = f"{stem}.png"
            rendered_png = os.path.join(rendered_dir, png_name)
            dest_image = os.path.join(images_dir, png_name)

            if os.path.exists(rendered_png):
                img = Image.open(rendered_png).convert("RGB")
                w, h = img.size
                max_edge = 1288
                longest = max(w, h)
                if longest > max_edge:
                    scale = max_edge / longest
                    img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
                img.save(dest_image)
            else:
                print(f"  WARNING: No rendered PNG found for {basename}, skipping")
                skipped += 1
                continue

            # Build prompt with anchor text from source PDF
            pdf_path = os.path.join(data_dir, f"{pdf_name}.pdf")
            if os.path.exists(pdf_path):
                anchor_text = get_anchor_text(pdf_path, page_num)
                prompt_text = (
                    PROMPT_TEMPLATE + "\n\n"
                    f"<|anchor_start|>\n{anchor_text}\n<|anchor_end|>"
                )
            else:
                prompt_text = PROMPT_TEMPLATE

            example = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image", "image": dest_image},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": yaml_response,
                    },
                ],
                "metadata": {
                    "source": f"{pdf_name}.pdf",
                    "page": page_num,
                    "html_source": html_path,
                    "primary_language": meta["primary_language"],
                    "is_table": meta["is_table"],
                    "is_diagram": meta["is_diagram"],
                },
            }
            out.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1
            print(f"  [{stem}] ✓ {len(natural_text)} chars")

    print(f"\nDone! Wrote {count} training examples to {output_jsonl}")
    if skipped:
        print(f"  Skipped {skipped} pages (no text or missing files)")





# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare SFT training data from PDFs using OLMoCR-style extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python prepare_sft_data.py --from-html synth_output/ --data-dir data/ -o train.jsonl\n"
        ),
    )

    parser.add_argument(
        "--from-html",
        required=True,
        help="Synth pipeline output directory containing html/ and rendered/ subdirs",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing original PDFs (default: data)",
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
    args = parser.parse_args()

    from_html(args.from_html, args.data_dir, args.output, args.images_dir)


if __name__ == "__main__":
    main()
