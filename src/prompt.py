"""
prompt.py — OLMoCR prompt construction and structured output schema.

Contains:
  - build_prompt(): the exact GPT-4o prompt from OLMoCR's build_openai_silver_data_prompt()
  - get_response_format_schema(): the structured JSON schema that forces GPT-4o to
    answer chain-of-thought fields (language, rotation, table, diagram) BEFORE
    producing the natural_text — this ordering trick improves output quality.

Reference: https://github.com/allenai/olmocr/blob/main/olmocr/prompts/prompts.py
"""


def build_prompt(anchor_text: str) -> str:
    """
    Build the exact prompt from OLMoCR's build_openai_silver_data_prompt().
    
    This prompt instructs GPT-4o to:
      - Read the page naturally, preserving reading order
      - Convert equations to LaTeX, tables to markdown
      - Remove headers/footers but keep references and footnotes
      - Read handwriting
      - Preserve partial sentences spanning page boundaries
      - Not hallucinate
    
    The anchor text (extracted by anchor.py) provides spatial layout context
    wrapped in RAW_TEXT_START / RAW_TEXT_END markers.
    
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
        "IMPORTANT: Extract ONLY the main body text of the page. You MUST exclude ALL of the following:\n"
        "  - Headers and footers (any repeating text at the top or bottom of the page, such as running titles, document titles, or section names)\n"
        "  - Page numbers (whether at the top, bottom, or margins of the page)\n"
        "  - Margin notes and side annotations (any text printed in the left or right margins, such as names, dates, labels, or short annotations)\n"
        "  - Any text that appears outside the main text column/body area\n"
        "Keep references, footnotes, and citations that are part of the main body text.\n"
        "Read any natural handwriting.\n"
        "This is likely one page out of several in the document, so be sure to preserve "
        "any sentences that come from the previous page, or continue onto the next page, exactly as they are.\n"
        "If there is no text at all that you think you should read, you can output null.\n"
        "Do not hallucinate.\n"
        f"RAW_TEXT_START\n{anchor_text}\nRAW_TEXT_END"
    )


def get_response_format_schema() -> dict:
    """
    OLMoCR's structured JSON schema for GPT-4o response.
    
    The field ordering is intentional:
      1. primary_language  → forces model to identify the language first
      2. is_rotation_valid → forces model to assess page orientation
      3. rotation_correction → if rotated, how much to correct
      4. is_table          → forces model to notice tabular content
      5. is_diagram        → forces model to notice visual diagrams
      6. natural_text      → LAST: the actual linearized text output
    
    This "chain-of-thought via schema ordering" trick ensures the model
    reasons about the page before producing the final text.
    
    Returns:
        OpenAI-compatible response_format dict with JSON schema
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "page_response",
            "schema": {
                "type": "object",
                "properties": {
                    "primary_language": {
                        "type": ["string", "null"],
                        "description": (
                            "The primary language of the text using two-letter codes "
                            "or null if there is no text at all that you think you should read."
                        ),
                    },
                    "is_rotation_valid": {
                        "type": "boolean",
                        "description": (
                            "Is this page oriented correctly for reading? Answer only considering "
                            "the textual content, do not factor in the rotation of any charts, "
                            "tables, drawings, or figures."
                        ),
                    },
                    "rotation_correction": {
                        "type": "integer",
                        "description": "Indicates the degree of clockwise rotation needed if the page is not oriented correctly.",
                        "enum": [0, 90, 180, 270],
                        "default": 0,
                    },
                    "is_table": {
                        "type": "boolean",
                        "description": "Indicates if the majority of the page content is in tabular format.",
                    },
                    "is_diagram": {
                        "type": "boolean",
                        "description": "Indicates if the majority of the page content is a visual diagram.",
                    },
                    "natural_text": {
                        "type": ["string", "null"],
                        "description": "The natural text content extracted from the page.",
                    },
                },
                "additionalProperties": False,
                "required": [
                    "primary_language",
                    "is_rotation_valid",
                    "rotation_correction",
                    "is_table",
                    "is_diagram",
                    "natural_text",
                ],
            },
            "strict": True,
        },
    }
