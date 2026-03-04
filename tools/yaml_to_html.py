"""
yaml_to_html.py — Convert YAML OCR output back into styled HTML document pages.

Reads YAML files (with natural_text field) and generates clean HTML pages
that visually represent the document content. Works for both original
and translated YAML files.

Usage:
  # Single file
  python yaml_to_html.py --input file.yaml --output file.html

  # Batch: all YAML files in a directory
  python yaml_to_html.py --yaml-dir synth_output/yaml/ --output-dir synth_output/html_from_yaml/
"""

import argparse
import glob
import os
import re
import textwrap

import yaml


# A4-like page dimensions
PAGE_WIDTH = 595
PAGE_HEIGHT = 842

CSS_TEMPLATE = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Bengali:wght@400;700&family=Noto+Serif:ital,wght@0,400;0,700;1,400&display=swap');

  * { margin: 0; padding: 0; box-sizing: border-box; }

  html {
    width: {width}px;
    min-height: {height}px;
  }

  body {
    width: {width}px;
    min-height: {height}px;
    font-family: 'Noto Serif', 'Noto Sans Bengali', Georgia, serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #222;
    background: #fff;
    padding: 60px 50px;
  }

  .main-col {
    width: 100%;
  }

  p {
    margin-bottom: 0.8em;
    text-align: justify;
  }

  h1, h2, h3 {
    font-weight: 700;
    margin: 1em 0 0.4em 0;
  }

  h1 { font-size: 14pt; }
  h2 { font-size: 12pt; }
  h3 { font-size: 11pt; }

  .meta-header {
    font-weight: 700;
    font-size: 12pt;
    margin-bottom: 1em;
    text-align: center;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    margin: 1em 0;
    font-size: 10pt;
  }

  th, td {
    border: 1px solid #666;
    padding: 4px 8px;
    text-align: left;
  }

  th {
    background: #f0f0f0;
    font-weight: 700;
  }
</style>
"""


def _detect_language(text: str) -> str:
    """Detect if text contains Bengali/Devanagari script."""
    bengali_chars = len(re.findall(r'[\u0980-\u09FF]', text))
    devanagari_chars = len(re.findall(r'[\u0900-\u097F]', text))
    total = len(text)
    if total == 0:
        return "en"
    if bengali_chars / total > 0.1:
        return "bn"
    if devanagari_chars / total > 0.1:
        return "hi"
    return "en"


def _text_to_html_body(text: str) -> str:
    """Convert natural_text content to HTML paragraphs."""
    if not text or text.strip().lower() == "null":
        return "<p><em>(No text content)</em></p>"

    # Split into paragraphs on double newlines
    paragraphs = re.split(r'\n{2,}', text.strip())

    html_parts = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Detect markdown tables
        if re.match(r'\|.*\|', para):
            html_parts.append(_markdown_table_to_html(para))
            continue

        # Escape HTML
        para = para.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        # Convert single newlines to spaces (reflow)
        para = re.sub(r'\n', ' ', para)
        para = re.sub(r'  +', ' ', para)

        html_parts.append(f"<p>{para}</p>")

    return "\n".join(html_parts)


def _markdown_table_to_html(md_table: str) -> str:
    """Convert markdown table to HTML table."""
    lines = [l.strip() for l in md_table.strip().split('\n') if l.strip()]
    if len(lines) < 2:
        return f"<p>{md_table}</p>"

    html = ["<table>"]
    for i, line in enumerate(lines):
        # Skip separator line (|---|---|)
        if re.match(r'\|[\s\-:|]+\|', line):
            continue

        cells = [c.strip() for c in line.split('|')[1:-1]]
        tag = "th" if i == 0 else "td"
        row = "".join(f"<{tag}>{c}</{tag}>" for c in cells)
        html.append(f"  <tr>{row}</tr>")

    html.append("</table>")
    return "\n".join(html)


def yaml_to_html(yaml_path: str) -> str:
    """Convert a YAML file to an HTML document string."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f.read())

    if not content:
        return "<html><body><p>Empty YAML</p></body></html>"

    lang = content.get("primary_language", "en") or "en"
    doc_id = content.get("document_id", os.path.splitext(os.path.basename(yaml_path))[0])

    # Check which format: raw (natural_text) or postprocessed (sentences+tables)
    if "natural_text" in content and content["natural_text"]:
        natural_text = content["natural_text"]
        body_html = _text_to_html_body(natural_text)
    elif "sentences" in content:
        body_html = _sentences_to_html(content.get("sentences", []),
                                        content.get("tables", []))
    else:
        body_html = "<p><em>(No content)</em></p>"

    # Auto-detect script for font selection
    detected_lang = _detect_language(body_html)
    if detected_lang in ("bn", "hi"):
        lang = detected_lang

    css = CSS_TEMPLATE.replace("{width}", str(PAGE_WIDTH)).replace("{height}", str(PAGE_HEIGHT))

    html = f"""<!DOCTYPE html>
<html lang="{lang}">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width={PAGE_WIDTH}">
{css}
</head>
<body>
<div class="main-col">
{body_html}
</div>
</body>
</html>"""

    return html


def _sentences_to_html(sentences: list, tables: list) -> str:
    """Convert postprocessed sentences + tables into HTML body."""
    parts = []

    for s in sentences:
        text = s.get("text", "")
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        # Preserve double newlines as paragraph breaks
        paras = re.split(r'\n{2,}', text)
        for p in paras:
            p = p.strip()
            if p:
                p = re.sub(r'\n', ' ', p)
                parts.append(f"<p>{p}</p>")

    for t in tables:
        headers = t.get("headers", [])
        rows = t.get("rows", [])
        tbl = ["<table>"]
        if headers:
            tbl.append("  <tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>")
        for row in rows:
            tbl.append("  <tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>")
        tbl.append("</table>")
        parts.append("\n".join(tbl))

    return "\n".join(parts)



def main():
    parser = argparse.ArgumentParser(description="Convert YAML to HTML pages")
    parser.add_argument("--input", help="Single YAML file to convert")
    parser.add_argument("--output", help="Output HTML file path (for single file)")
    parser.add_argument("--yaml-dir", help="Directory of YAML files for batch conversion")
    parser.add_argument("--output-dir", help="Output directory for batch HTML files")
    args = parser.parse_args()

    if args.input:
        html = yaml_to_html(args.input)
        out_path = args.output or args.input.replace(".yaml", ".html")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Created: {out_path}")
        return

    if args.yaml_dir:
        out_dir = args.output_dir or os.path.join(args.yaml_dir, "html_from_yaml")
        os.makedirs(out_dir, exist_ok=True)
        yaml_files = sorted(glob.glob(os.path.join(args.yaml_dir, "*.yaml")))
        print(f"Found {len(yaml_files)} YAML files")

        done = 0
        for yf in yaml_files:
            stem = os.path.splitext(os.path.basename(yf))[0]
            out_path = os.path.join(out_dir, f"{stem}.html")
            try:
                html = yaml_to_html(yf)
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(html)
                done += 1
                if done % 500 == 0:
                    print(f"  Progress: {done}/{len(yaml_files)}")
            except Exception as e:
                print(f"  [FAIL] {stem}: {e}")

        print(f"\nDone: {done} HTML files in {out_dir}")
        return

    parser.error("Provide either --input or --yaml-dir")


if __name__ == "__main__":
    main()
