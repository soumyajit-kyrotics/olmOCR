"""
prompt.py — OLMoCR prompt construction with YAML response format.

Contains:
  - build_prompt(): the prompt adapted from OLMoCR's approach,
    with YAML response format instructions instead of JSON schema.

    The chain-of-thought fields (language, rotation, table, diagram) are
    ordered BEFORE natural_text in the YAML template — this ordering
    ensures the model reasons about the page before producing text.

    Anchor text is wrapped in <|anchor_start|> / <|anchor_end|> tags
    that are registered as special tokens in the tokenizer.

Reference: https://github.com/allenai/olmocr/blob/main/olmocr/prompts/prompts.py
"""


def build_prompt(anchor_text: str) -> str:
    """
    Build the OLMoCR-style prompt with YAML response format.

    This prompt instructs the model to:
      - Read the page naturally, preserving reading order
      - Convert equations to LaTeX, tables to markdown
      - Remove headers/footers but keep references and footnotes
      - Read handwriting
      - Preserve partial sentences spanning page boundaries
      - Respond in YAML format for token efficiency
      - Not hallucinate

    The anchor text (extracted by anchor.py) provides spatial layout context
    wrapped in <|anchor_start|> / <|anchor_end|> markers.

    Args:
        anchor_text: formatted anchor text from get_anchor_text()

    Returns:
        Complete prompt string to send alongside the page image
    """
    return (
        "Below is the image of one page of a PDF document, as well as some raw textual content "
        "that was previously extracted for it that includes position information for each image "
        "and block of text (The origin [0x0] of the coordinates is in the lower left corner of the image). "
        "Just return the plain text representation of this document as if you were reading it naturally.\n"
        "Turn equations into a LaTeX representation, and tables into markdown format.\n"
        "IMPORTANT: Reflow the text into continuous paragraphs. Do NOT preserve the visual line breaks "
        "from the PDF layout. Join lines that belong to the same paragraph into a single flowing paragraph. "
        "If a word is split across lines with a hyphen (e.g., 'argu-\\nment'), rejoin it as one word ('argument').\n"
        "IMPORTANT: Extract ONLY the main body text of the page. You MUST exclude ALL of the following:\n"
        "  - Headers and footers (any repeating text at the top or bottom of the page, such as running titles, document titles, or section names)\n"
        "  - Page numbers (whether at the top, bottom, or margins of the page)\n"
        "  - Margin notes and side annotations (any text printed in the left or right margins, such as names, dates, labels, or short annotations)\n"
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
        "  <the extracted plain text, indented by 2 spaces>\n\n"
        f"<|anchor_start|>\n{anchor_text}\n<|anchor_end|>"
    )


