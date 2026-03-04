"""
html_to_yaml.py — Convert HTML files to standalone YAML files using BeautifulSoup.

No API calls needed. Reads each HTML file, extracts body text + metadata,
and writes OLMoCR-schema YAML to synth_output/yaml/.

Skips pages that already have a .yaml file (from generate_yaml.py).
"""

import argparse
import glob
import os
import re
import textwrap

from bs4 import BeautifulSoup


def _extract_text_from_html(html_path: str) -> str:
    """Extract body text from HTML, excluding header/aside."""
    with open(html_path, encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    for tag in soup.find_all(["header", "aside"]):
        tag.decompose()

    root = soup.find(class_="main-col") or soup.find("body")
    if not root:
        return ""

    blocks = []
    for child in root.children:
        if isinstance(child, str):
            stripped = child.strip()
            if stripped:
                blocks.append(stripped)
            continue
        if child.name == "table":
            # Convert table to markdown
            blocks.append(_table_to_markdown(child))
            continue
        text = child.get_text(separator="\n", strip=True)
        if text:
            blocks.append(text)

    result = "\n\n".join(blocks)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def _table_to_markdown(table_tag) -> str:
    """Convert an HTML <table> to markdown table format."""
    rows = table_tag.find_all("tr")
    if not rows:
        return str(table_tag)

    md_rows = []
    for i, row in enumerate(rows):
        cells = row.find_all(["th", "td"])
        cell_texts = [c.get_text(strip=True).replace("|", "\\|") for c in cells]
        md_rows.append("| " + " | ".join(cell_texts) + " |")
        if i == 0:
            md_rows.append("| " + " | ".join(["---"] * len(cell_texts)) + " |")

    return "\n".join(md_rows)


def _detect_metadata(html_path: str) -> dict:
    """Detect language, is_table, is_diagram from HTML structure."""
    with open(html_path, encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    html_tag = soup.find("html")
    lang = None
    if html_tag and html_tag.get("lang"):
        lang = html_tag["lang"][:2].lower()

    main_col = soup.find(class_="main-col") or soup.find("body")
    is_table = False
    is_diagram = False
    if main_col:
        tables = main_col.find_all("table")
        all_text = main_col.get_text(strip=True)
        table_text = "".join(t.get_text(strip=True) for t in tables)
        if all_text and len(table_text) > len(all_text) * 0.5:
            is_table = True
        images = main_col.find_all("img")
        if len(images) > 0 and len(all_text) < 100:
            is_diagram = True

    return {
        "primary_language": lang or "en",
        "is_table": is_table,
        "is_diagram": is_diagram,
    }


def _format_as_yaml(natural_text, document_id="", primary_language="en",
                     is_table=False, is_diagram=False) -> str:
    """Format extracted text as OLMoCR YAML schema."""
    indented = textwrap.indent(natural_text, "  ")
    return (
        f"document_id: {document_id}\n"
        f"primary_language: {primary_language}\n"
        f"is_rotation_valid: true\n"
        f"rotation_correction: 0\n"
        f"is_table: {'true' if is_table else 'false'}\n"
        f"is_diagram: {'true' if is_diagram else 'false'}\n"
        f"natural_text: |\n"
        f"{indented}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert HTML files to YAML (no API calls)")
    parser.add_argument("--html-dir", default="synth_output/html",
                        help="Directory containing HTML files")
    parser.add_argument("--output-dir", default="synth_output/yaml",
                        help="Directory for YAML output")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing YAML files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    html_files = sorted(glob.glob(os.path.join(args.html_dir, "*.html")))
    print(f"Found {len(html_files)} HTML files in {args.html_dir}")

    done = 0
    skipped = 0
    failed = 0

    for html_path in html_files:
        stem = os.path.splitext(os.path.basename(html_path))[0]
        yaml_path = os.path.join(args.output_dir, f"{stem}.yaml")

        if os.path.exists(yaml_path) and not args.overwrite:
            skipped += 1
            continue

        try:
            text = _extract_text_from_html(html_path)
            if not text:
                print(f"  [SKIP] {stem} — no text extracted")
                skipped += 1
                continue

            meta = _detect_metadata(html_path)
            yaml_str = _format_as_yaml(
                text,
                document_id=stem,
                primary_language=meta["primary_language"],
                is_table=meta["is_table"],
                is_diagram=meta["is_diagram"],
            )

            with open(yaml_path, "w", encoding="utf-8") as f:
                f.write(yaml_str)
            done += 1

            if done % 200 == 0:
                print(f"  Progress: {done} converted, {skipped} skipped, {failed} failed")

        except Exception as e:
            print(f"  [FAIL] {stem}: {e}")
            failed += 1

    print(f"\nDone: {done} converted, {skipped} skipped, {failed} failed")
    print(f"YAML files in {args.output_dir}: {len(glob.glob(os.path.join(args.output_dir, '*.yaml')))}")


if __name__ == "__main__":
    main()
