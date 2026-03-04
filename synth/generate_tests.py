"""
generate_tests.py — Extract programmatic unit tests from generated HTML.

Parses the HTML produced by generate_html.py and creates binary unit tests
that verify whether an OCR system correctly extracted the content.

Test types (matching OLMoCR v2's olmOCR-Bench):
  - text_presence: a sentence/phrase must exist in OCR output
  - text_absence:  header/footer text must NOT be in OCR output
  - table:         table cell relationships are preserved (up/down/left/right)
  - order:         reading order is correct (sentence A comes before B)
"""

import argparse
import glob
import json
import os
import random
import re
import uuid

from bs4 import BeautifulSoup, NavigableString


# ---------------------------------------------------------------------------
# Test Extraction Functions
# ---------------------------------------------------------------------------

def extract_text_presence_tests(soup: BeautifulSoup, pdf_name: str, page: int, rng: random.Random) -> list:
    """
    Extract text_presence tests by sampling sentences from body content.

    Targets: <p>, <li>, <h1>-<h6>, <td>, <th> tags (excluding header/footer).
    """
    tests = []

    # Get all body text elements, excluding header/footer/aside
    body_tags = []
    for tag in soup.find_all(["p", "li", "h1", "h2", "h3", "h4", "h5", "h6"]):
        # Skip if inside <header>, <footer>, or <aside> (margin notes)
        if tag.find_parent(["header", "footer", "aside"]):
            continue
        text = tag.get_text(strip=True)
        if text and len(text) >= 15:  # minimum meaningful length
            body_tags.append(text)

    # Sample up to 5 text presence tests
    sample_size = min(5, len(body_tags))
    if sample_size > 0:
        sampled = rng.sample(body_tags, sample_size)
        for text in sampled:
            # Truncate long texts to a sentence fragment
            if len(text) > 200:
                text = text[:200]

            tests.append({
                "id": f"{pdf_name}_p{page}_tp_{uuid.uuid4().hex[:8]}",
                "type": "text_presence",
                "pdf": f"{pdf_name}.pdf",
                "page": page,
                "text": text,
                "max_diffs": max(3, len(text) // 50),  # allow ~2% character diffs
            })

    return tests


def extract_text_absence_tests(soup: BeautifulSoup, pdf_name: str, page: int) -> list:
    """
    Extract text_absence tests from <header>, <footer>, and <aside> content.

    These tests verify that the OCR output does NOT include
    headers, footers, page numbers, and margin/side notes.
    """
    tests = []

    for tag_name in ["header", "footer", "aside"]:
        for tag in soup.find_all(tag_name):
            text = tag.get_text(strip=True)
            if text and len(text) >= 3:
                tests.append({
                    "id": f"{pdf_name}_p{page}_ta_{uuid.uuid4().hex[:8]}",
                    "type": "text_absence",
                    "pdf": f"{pdf_name}.pdf",
                    "page": page,
                    "text": text,
                    "max_diffs": 0,
                })

    return tests


def extract_table_tests(soup: BeautifulSoup, pdf_name: str, page: int, rng: random.Random) -> list:
    """
    Extract table tests by sampling cells and checking positional relationships.

    For each sampled cell, records its value and the values of adjacent cells
    (up, down, left, right) to verify table structure preservation.
    """
    tests = []
    tables = soup.find_all("table")

    for table_idx, table in enumerate(tables):
        # Parse the table into a 2D array
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue

        # Build a 2D array of cell texts
        grid = []
        for row in rows:
            cells = row.find_all(["td", "th"])
            grid.append([cell.get_text(strip=True) for cell in cells])

        if not grid or not grid[0]:
            continue

        num_rows = len(grid)
        num_cols = max(len(row) for row in grid)

        # Pad rows to uniform width
        for row in grid:
            while len(row) < num_cols:
                row.append("")

        # Determine header rows (first row if it contains <th>)
        header_rows = set()
        for i, row in enumerate(rows):
            if row.find("th"):
                header_rows.add(i)

        # Sample non-header cells
        non_header_rows = [i for i in range(num_rows) if i not in header_rows]
        if not non_header_rows:
            non_header_rows = list(range(num_rows))

        # Sample up to 6 cells
        cell_positions = []
        sample_rows = rng.sample(non_header_rows, min(3, len(non_header_rows)))
        sample_cols = rng.sample(range(num_cols), min(2, num_cols))
        for r in sample_rows:
            for c in sample_cols:
                cell_positions.append((r, c))

        rng.shuffle(cell_positions)

        for row_idx, col_idx in cell_positions:
            cell_text = grid[row_idx][col_idx]
            if not cell_text or len(cell_text) < 2:
                continue

            test_data = {
                "id": f"{pdf_name}_p{page}_tbl{table_idx}_{uuid.uuid4().hex[:8]}",
                "type": "table",
                "pdf": f"{pdf_name}.pdf",
                "page": page,
                "cell": cell_text,
                "max_diffs": 0,
            }

            # Adjacent cells
            if row_idx > 0 and grid[row_idx - 1][col_idx]:
                test_data["up"] = grid[row_idx - 1][col_idx]
            if row_idx < num_rows - 1 and grid[row_idx + 1][col_idx]:
                test_data["down"] = grid[row_idx + 1][col_idx]
            if col_idx > 0 and grid[row_idx][col_idx - 1]:
                test_data["left"] = grid[row_idx][col_idx - 1]
            if col_idx < num_cols - 1 and grid[row_idx][col_idx + 1]:
                test_data["right"] = grid[row_idx][col_idx + 1]

            # Column header
            if header_rows and col_idx < len(grid[min(header_rows)]):
                header_text = grid[min(header_rows)][col_idx]
                if header_text and row_idx not in header_rows:
                    test_data["top_heading"] = header_text

            # Only add if we have at least one relationship
            if any(k in test_data for k in ["up", "down", "left", "right", "top_heading"]):
                tests.append(test_data)

    return tests


def extract_order_tests(soup: BeautifulSoup, pdf_name: str, page: int, rng: random.Random) -> list:
    """
    Extract reading order tests by sampling pairs of sentences
    and verifying they appear in the correct sequence.
    """
    tests = []

    # Get all body text blocks in document order
    texts = []
    for tag in soup.find_all(["p", "li", "h1", "h2", "h3", "h4", "h5", "h6"]):
        if tag.find_parent(["header", "footer", "aside"]):
            continue
        text = tag.get_text(strip=True)
        if text and len(text) >= 20:
            texts.append(text)

    # Need at least 2 sentences for order tests
    if len(texts) < 2:
        return tests

    # Sample up to 3 pairs of sentences
    num_pairs = min(3, len(texts) - 1)
    for _ in range(num_pairs):
        i = rng.randint(0, len(texts) - 2)
        j = rng.randint(i + 1, len(texts) - 1)

        before_text = texts[i][:150] if len(texts[i]) > 150 else texts[i]
        after_text = texts[j][:150] if len(texts[j]) > 150 else texts[j]

        tests.append({
            "id": f"{pdf_name}_p{page}_ord_{uuid.uuid4().hex[:8]}",
            "type": "order",
            "pdf": f"{pdf_name}.pdf",
            "page": page,
            "before": before_text,
            "after": after_text,
        })

    return tests


def extract_math_tests(soup: BeautifulSoup, pdf_name: str, page: int) -> list:
    """
    Extract math tests from LaTeX/MathJax/KaTeX content in the HTML.

    Looks for:
      - <math> elements (MathML)
      - Elements with class 'MathJax' or 'katex'
      - Inline LaTeX delimiters: \\( ... \\) or $ ... $
      - Display LaTeX delimiters: \\[ ... \\] or $$ ... $$
    """
    tests = []

    # Strategy 1: Look for <math> or MathJax/KaTeX elements
    math_elements = soup.find_all(["math"])
    math_elements += soup.find_all(class_=re.compile(r"MathJax|katex", re.IGNORECASE))

    for i, elem in enumerate(math_elements):
        latex_text = elem.get_text(strip=True)
        if latex_text and len(latex_text) >= 3:
            tests.append({
                "id": f"{pdf_name}_p{page}_math_{uuid.uuid4().hex[:8]}",
                "type": "math",
                "pdf": f"{pdf_name}.pdf",
                "page": page,
                "latex": latex_text,
            })

    # Strategy 2: Look for inline LaTeX in text nodes
    body = soup.find("body")
    if body:
        full_text = body.get_text()
        # Match \( ... \) and \[ ... \]
        latex_patterns = [
            re.compile(r'\\\((.+?)\\\)', re.DOTALL),
            re.compile(r'\\\[(.+?)\\\]', re.DOTALL),
            re.compile(r'\$\$(.+?)\$\$', re.DOTALL),
        ]
        for pattern in latex_patterns:
            for match in pattern.finditer(full_text):
                latex = match.group(1).strip()
                if latex and len(latex) >= 3:
                    tests.append({
                        "id": f"{pdf_name}_p{page}_math_{uuid.uuid4().hex[:8]}",
                        "type": "math",
                        "pdf": f"{pdf_name}.pdf",
                        "page": page,
                        "latex": latex,
                    })

    return tests


def extract_formatting_tests(soup: BeautifulSoup, pdf_name: str, page: int, rng: random.Random) -> list:
    """
    Extract formatting tests that verify structural markers are preserved.

    Tests that bold text, italic text, and heading structure survive OCR.
    """
    tests = []

    # Bold text (<strong>, <b>)
    bold_elements = soup.find_all(["strong", "b"])
    for elem in bold_elements:
        if elem.find_parent(["header", "footer", "aside"]):
            continue
        text = elem.get_text(strip=True)
        if text and len(text) >= 3:
            tests.append({
                "id": f"{pdf_name}_p{page}_fmt_{uuid.uuid4().hex[:8]}",
                "type": "formatting",
                "pdf": f"{pdf_name}.pdf",
                "page": page,
                "text": text,
                "format_type": "bold",
            })

    # Italic text (<em>, <i>) — sample up to 3
    italic_elements = soup.find_all(["em", "i"])
    italic_texts = []
    for elem in italic_elements:
        if elem.find_parent(["header", "footer", "aside"]):
            continue
        text = elem.get_text(strip=True)
        if text and len(text) >= 5:
            italic_texts.append(text)

    for text in rng.sample(italic_texts, min(3, len(italic_texts))):
        tests.append({
            "id": f"{pdf_name}_p{page}_fmt_{uuid.uuid4().hex[:8]}",
            "type": "formatting",
            "pdf": f"{pdf_name}.pdf",
            "page": page,
            "text": text,
            "format_type": "italic",
        })

    # Headings (h1-h6) — verify heading text is present
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        if tag.find_parent(["header", "footer", "aside"]):
            continue
        text = tag.get_text(strip=True)
        if text and len(text) >= 3:
            tests.append({
                "id": f"{pdf_name}_p{page}_fmt_{uuid.uuid4().hex[:8]}",
                "type": "formatting",
                "pdf": f"{pdf_name}.pdf",
                "page": page,
                "text": text,
                "format_type": f"heading_{tag.name}",
            })

    return tests


# ---------------------------------------------------------------------------
# Process HTML Files
# ---------------------------------------------------------------------------

def process_html_file(html_path: str, seed: int = 42) -> list:
    """
    Extract all unit tests from a single HTML file.

    Returns a list of test dicts.
    """
    rng = random.Random(seed)

    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, "html.parser")

    basename = os.path.splitext(os.path.basename(html_path))[0]
    # Extract pdf_name and page from filename like "1950_1_25_29_page_1"
    match = re.match(r"(.+)_page_(\d+)$", basename)
    if match:
        pdf_name = match.group(1)
        page = int(match.group(2))
    else:
        pdf_name = basename
        page = 1

    tests = []
    tests.extend(extract_text_presence_tests(soup, pdf_name, page, rng))
    tests.extend(extract_text_absence_tests(soup, pdf_name, page))
    tests.extend(extract_table_tests(soup, pdf_name, page, rng))
    tests.extend(extract_order_tests(soup, pdf_name, page, rng))
    tests.extend(extract_math_tests(soup, pdf_name, page))
    tests.extend(extract_formatting_tests(soup, pdf_name, page, rng))

    return tests


def process_directory(html_dir: str, output_path: str) -> int:
    """
    Process all HTML files in a directory and write tests to a JSONL file.

    Returns the total number of tests generated.
    """
    html_files = sorted(glob.glob(os.path.join(html_dir, "*.html")))
    if not html_files:
        print(f"No HTML files found in {html_dir}")
        return 0

    all_tests = []
    test_type_counts = {}

    for html_path in html_files:
        tests = process_html_file(html_path)
        all_tests.extend(tests)

        for t in tests:
            ttype = t["type"]
            test_type_counts[ttype] = test_type_counts.get(ttype, 0) + 1

        print(f"  {os.path.basename(html_path)}: {len(tests)} tests")

    # Write JSONL
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for test in all_tests:
            f.write(json.dumps(test, ensure_ascii=False) + "\n")

    print(f"\nTest type breakdown:")
    for ttype, count in sorted(test_type_counts.items()):
        print(f"  {ttype}: {count}")

    return len(all_tests)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract unit tests from generated HTML files"
    )
    parser.add_argument("--html-dir", required=True, help="Directory containing HTML files")
    parser.add_argument("--output", default="synth_output/tests.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    total = process_directory(args.html_dir, args.output)
    print(f"\nDone! {total} tests → {args.output}")


if __name__ == "__main__":
    main()
