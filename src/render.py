"""
render.py — PDF page rendering to base64-encoded PNG images.

Converts individual PDF pages to high-resolution PNG images (up to 2048px)
and encodes them as base64 strings for inclusion in GPT-4o API requests.

Uses pdf2image (poppler backend) for rendering.
"""

import base64
from io import BytesIO

from pdf2image import convert_from_path


def render_pdf_to_base64png(pdf_path: str, page_num: int, max_dim: int = 2048) -> str:
    """
    Render a single PDF page to a base64-encoded PNG string.

    Args:
        pdf_path: path to the PDF file
        page_num: 1-indexed page number
        max_dim: maximum pixel dimension for the longest edge (default: 2048,
                 matching OLMoCR's TARGET_IMAGE_DIM)

    Returns:
        Base64-encoded PNG string ready for embedding in API requests
    """
    # pdf2image's convert_from_path uses 1-based page numbering
    pages = convert_from_path(
        pdf_path,
        first_page=page_num,
        last_page=page_num,
        fmt="png",
        size=(max_dim, None),  # Cap width; pdf2image maintains aspect ratio
    )
    if not pages:
        raise ValueError(f"Failed to render page {page_num} from {pdf_path}")

    img = pages[0]

    # Ensure longest edge does not exceed max_dim
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)))

    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
