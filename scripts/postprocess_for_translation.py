"""
postprocess_for_translation.py — Transform OCR YAML output into a structured format
ready for sentence-level translation.

Reads YAML files from synth_output/yaml/ (and optionally HTML-sourced YAML from
prepare_sft_data.py) and produces a per-page JSON with:

  document_id:   unique stable ID (e.g. "1959~scr_1959_1_861_867_e_page_2")
  primary_language: "en"
  is_table:      true/false
  sentences:     [{id: 1, text: "..."}, ...]   — from natural_text
  tables:        [{table_id: 1, headers: [...], rows: [[...], ...]}, ...]

Usage:
  python postprocess_for_translation.py \\
      --yaml-dir synth_output/yaml/ \\
      --html-yaml-dir synth_output/html/ \\   # optional: YAML embedded in HTML pipeline
      --out-dir translation_input/

  # Then give translation_input/ to your translation teammate.
"""

import argparse
import glob
import json
import os
import re
import sys

import yaml  # pyyaml


# ---------------------------------------------------------------------------
# Sentence splitter (rule-based, handles legal text)
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> list[dict]:
    """
    Split text into sentences and return [{id: N, text: "..."}].

    Uses a regex-based splitter that handles:
    - Common legal abbreviations (No., Ltd., Hon., vs., etc.)
    - Single-letter initials (P. G. Gokhale, J., O. XLI)
    - Quotation marks and numbered lists
    """
    if not text or text.strip().lower() == "null":
        return []

    # Protect common abbreviations from being split on
    ABBREVS = [
        r"Mr\.", r"Mrs\.", r"Ms\.", r"Dr\.", r"Hon\.", r"No\.", r"vs\.",
        r"Ltd\.", r"Inc\.", r"Jr\.", r"Sr\.", r"Vol\.", r"Art\.", r"Sec\.",
        r"Para\.", r"v\.", r"pp\.", r"Fig\.", r"approx\.", r"est\.", r"etc\.",
        r"Ors\.", r"Anr\.", r"Govt\.", r"Pvt\.", r"Assn\.", r"Commr\.",
        r"Admn\.", r"Crl\.", r"Cr\.", r"Dist\.", r"Ref\.",
        r"e\.g\.", r"i\.e\.",
    ]
    placeholder_map = {}
    protected = text
    idx = 0

    for abbrev in ABBREVS:
        placeholder = f"__ABBREV{idx}__"
        protected, count = re.subn(abbrev, placeholder, protected)
        if count:
            original = re.sub(r"\\", "", abbrev)
            placeholder_map[placeholder] = original
            idx += 1

    # Protect single-letter initials: "P." "G." "J." "A." "O." etc.
    # A period after a single uppercase letter is almost never a sentence end.
    # Matches: "P.", " J.", "(A." but NOT "He." or "The."
    def _protect_initial(m):
        nonlocal idx
        placeholder = f"__ABBREV{idx}__"
        placeholder_map[placeholder] = m.group(0)
        idx += 1
        return m.group(1) + placeholder
    # Match a single uppercase letter followed by period, where preceded by
    # start-of-string, space, quote, or parenthesis
    protected = re.sub(r"((?:^|[\s\"'(,]))([A-Z]\.)", _protect_initial, protected)

    # Also protect "p." (lowercase, meaning "page") when followed by a digit
    def _protect_p_page(m):
        nonlocal idx
        placeholder = f"__ABBREV{idx}__"
        placeholder_map[placeholder] = m.group(0)
        idx += 1
        return placeholder
    protected = re.sub(r"\bp\.\s*(?=\d)", _protect_p_page, protected)

    # Split on sentence-ending punctuation followed by whitespace + uppercase
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"\(])', protected)

    # Restore abbreviations
    sentences = []
    for part in parts:
        restored = part
        for placeholder, original in placeholder_map.items():
            restored = restored.replace(placeholder, original)
        restored = restored.strip()
        if restored:
            sentences.append(restored)

    return [{"id": i + 1, "text": s} for i, s in enumerate(sentences)]



# ---------------------------------------------------------------------------
# Markdown table parser
# ---------------------------------------------------------------------------

def parse_markdown_tables(text: str) -> tuple[str, list[dict]]:
    """
    Extract markdown tables from text, return (text_without_tables, [table_dicts]).

    Each table dict:
      {table_id: N, headers: [...], rows: [[...]]}
    """
    if not text:
        return text, []

    tables = []
    table_id = 0

    def _parse_table_block(match):
        nonlocal table_id
        raw = match.group(0)
        lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]

        rows_raw = []
        for line in lines:
            if re.match(r"^\|[-| :]+\|$", line):
                continue  # separator line
            cells = [c.strip() for c in line.strip("|").split("|")]
            rows_raw.append(cells)

        if len(rows_raw) < 2:
            return raw  # not a real table

        headers = rows_raw[0]
        rows    = rows_raw[1:]

        table_id += 1
        tables.append({
            "table_id": table_id,
            "headers":  headers,
            "rows":     rows,
        })
        return f"__TABLE_{table_id}__"



    # Match markdown table blocks (header row | separator | data rows)
    table_pattern = re.compile(
        r"(\|.+\|\n)(\|[-| :]+\|\n)(\|.+\|\n?)+",
        re.MULTILINE,
    )
    cleaned = table_pattern.sub(_parse_table_block, text)

    # Remove table placeholders from body text
    cleaned = re.sub(r"__TABLE_\d+__\n?", "", cleaned).strip()

    return cleaned, tables


# ---------------------------------------------------------------------------
# YAML parser
# ---------------------------------------------------------------------------

def _parse_yaml_content(raw_yaml: str) -> dict:
    """Parse the raw YAML string from the VLM output. Tolerates minor formatting issues."""
    try:
        return yaml.safe_load(raw_yaml) or {}
    except yaml.YAMLError:
        # Fallback: extract fields with regex
        result = {}
        for field in ("primary_language", "is_rotation_valid", "rotation_correction",
                       "is_table", "is_diagram"):
            m = re.search(rf"^{field}:\s*(.+)$", raw_yaml, re.MULTILINE)
            if m:
                result[field] = m.group(1).strip()
        # Extract natural_text block
        m = re.search(r"natural_text:\s*\|\n((?:  .+\n?)*)", raw_yaml)
        if m:
            result["natural_text"] = textwrap.dedent(m.group(1)).strip()
        return result


# ---------------------------------------------------------------------------
# Per-file transformation
# ---------------------------------------------------------------------------

def transform_yaml_file(yaml_path: str, doc_id: str) -> dict | None:
    """Read one YAML file and return the translation-ready dict."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = f.read()

    parsed = _parse_yaml_content(raw)
    if not parsed:
        return None

    natural_text = parsed.get("natural_text") or ""
    if isinstance(natural_text, str):
        natural_text = natural_text.strip()

    # Extract and structure tables
    text_no_tables, tables = parse_markdown_tables(natural_text)

    # Split remaining body text into sentences
    sentences = split_sentences(text_no_tables)

    return {
        "document_id":       doc_id,
        "primary_language":  parsed.get("primary_language", "en"),
        "is_rotation_valid": parsed.get("is_rotation_valid", True),
        "rotation_correction": int(parsed.get("rotation_correction", 0) or 0),
        "is_table":          bool(parsed.get("is_table", False)),
        "is_diagram":        bool(parsed.get("is_diagram", False)),
        "sentences":         sentences,
        "tables":            tables,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Post-process OCR YAML output into translation-ready structured YAML",
    )
    parser.add_argument("--yaml-dir",  default="synth_output/yaml",
                        help="Directory of .yaml files from generate_yaml.py")
    parser.add_argument("--out-dir",   default="translation_input",
                        help="Output directory for per-page YAML files")
    parser.add_argument("--out-yaml",  default=None,
                        help="Optional: write all pages to a single multi-document YAML file")
    args = parser.parse_args()

    yaml_dir = args.yaml_dir
    out_dir  = args.out_dir

    yaml_files = sorted(glob.glob(os.path.join(yaml_dir, "*.yaml")))
    if not yaml_files:
        print(f"No .yaml files found in {yaml_dir}")
        sys.exit(1)

    print(f"Found {len(yaml_files)} YAML files in {yaml_dir}")
    os.makedirs(out_dir, exist_ok=True)

    count = 0
    skipped = 0

    out_yaml_handle = None
    if args.out_yaml:
        out_yaml_handle = open(args.out_yaml, "w", encoding="utf-8")

    for yaml_path in yaml_files:
        stem   = os.path.splitext(os.path.basename(yaml_path))[0]
        doc_id = stem

        result = transform_yaml_file(yaml_path, doc_id)
        if result is None:
            print(f"  WARNING: Could not parse {stem}, skipping")
            skipped += 1
            continue

        if out_yaml_handle:
            out_yaml_handle.write("---\n")
            out_yaml_handle.write(yaml.dump(result, allow_unicode=True, default_flow_style=False))
        else:
            out_path = os.path.join(out_dir, f"{stem}.yaml")
            with open(out_path, "w", encoding="utf-8") as f:
                yaml.dump(result, f, allow_unicode=True, default_flow_style=False)

        count += 1
        if count % 500 == 0:
            print(f"  Processed {count} files...")

    if out_yaml_handle:
        out_yaml_handle.close()

    print(f"\nDone!")
    print(f"  Processed : {count} pages")
    if skipped:
        print(f"  Skipped   : {skipped} pages (parse errors)")
    if args.out_yaml:
        print(f"  Output    : {args.out_yaml}")
    else:
        print(f"  Output    : {out_dir}/ ({count} YAML files)")


if __name__ == "__main__":
    import textwrap
    main()
