"""
anchor.py — Document Anchoring (the core OLMoCR innovation).

Extracts text blocks with (x, y) positions and image bounding boxes from
a PDF page's digital layer using pypdf. This metadata is formatted into
"anchor text" that is included in the GPT-4o prompt alongside the page image.

Key insight from the paper: providing spatial layout information to the VLM
dramatically reduces hallucination and improves reading order accuracy.

Reference: https://github.com/allenai/olmocr/blob/main/olmocr/prompts/anchor.py
"""

import random
import re
from typing import List

from pypdf import PdfReader

from .models import BoundingBox, TextElement, ImageElement, PageReport


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers — affine transformation math
# ─────────────────────────────────────────────────────────────────────────────

def _transform_point(x: float, y: float, m: List[float]) -> tuple:
    """Apply a 2D affine transformation matrix [a,b,c,d,e,f] to point (x, y)."""
    x_new = m[0] * x + m[2] * y + m[4]
    y_new = m[1] * x + m[3] * y + m[5]
    return x_new, y_new


def _mult(m: List[float], n: List[float]) -> List[float]:
    """Multiply two 2D affine transformation matrices (6-element arrays)."""
    return [
        m[0] * n[0] + m[1] * n[2],
        m[0] * n[1] + m[1] * n[3],
        m[2] * n[0] + m[3] * n[2],
        m[2] * n[1] + m[3] * n[3],
        m[4] * n[0] + m[5] * n[2] + n[4],
        m[4] * n[1] + m[5] * n[3] + n[5],
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Page report extraction — reads text + image positions from PDF
# ─────────────────────────────────────────────────────────────────────────────

def _extract_page_report(pdf_path: str, page_num: int) -> PageReport:
    """
    Extract text elements with positions and image bounding boxes from a PDF page.
    This is the 'document anchoring' technique from the OLMoCR paper.
    
    Args:
        pdf_path: path to the PDF file
        page_num: 1-indexed page number
    
    Returns:
        PageReport with mediabox, text_elements, and image_elements
    """
    reader = PdfReader(pdf_path)
    page = reader.pages[page_num - 1]
    resources = page.get("/Resources", {})
    xobjects = resources.get("/XObject", {})
    text_elements, image_elements = [], []

    def visitor_body(text, cm, tm, font_dict, font_size):
        """Callback for each text block — records text and its transformed position."""
        txt2user = _mult(tm, cm)
        text_elements.append(TextElement(text, txt2user[4], txt2user[5]))

    def visitor_op(op, args, cm, tm):
        """Callback for each PDF operator — captures image placements."""
        if op == b"Do":
            xobject_name = args[0]
            xobject = xobjects.get(xobject_name)
            if xobject and xobject.get("/Subtype") == "/Image":
                x0, y0 = _transform_point(0, 0, cm)
                x1, y1 = _transform_point(1, 1, cm)
                image_elements.append(
                    ImageElement(
                        xobject_name,
                        BoundingBox(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)),
                    )
                )

    page.extract_text(visitor_text=visitor_body, visitor_operand_before=visitor_op)

    return PageReport(
        mediabox=BoundingBox.from_rectangle(page.mediabox),
        text_elements=text_elements,
        image_elements=image_elements,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Text cleanup + image merging
# ─────────────────────────────────────────────────────────────────────────────

def _cleanup_text(text: str, max_length: int = 250) -> str:
    """
    Clean up an extracted text element:
      - Strip whitespace
      - Escape brackets and control characters
      - Truncate long strings (keep head + tail with '...' in between)
    """
    text = text.strip()
    if not text:
        return ""

    # Escape special characters that would confuse the model
    replacements = {"[": "\\[", "]": "\\]", "\n": "\\n", "\r": "\\r", "\t": "\\t"}
    pattern = re.compile("|".join(re.escape(k) for k in replacements))
    text = pattern.sub(lambda m: replacements[m.group(0)], text)

    # Truncate long text elements (keep head + tail)
    if len(text) > max_length:
        half = max_length // 2 - 3
        head = text[:half].rsplit(" ", 1)[0] or text[:half]
        tail = text[-half:].split(" ", 1)[-1] or text[-half:]
        text = f"{head} ... {tail}"

    return text


def _merge_image_elements(images: List[ImageElement], tolerance: float = 0.5) -> List[ImageElement]:
    """
    Merge overlapping or adjacent image bounding boxes using union-find.
    Many PDFs split a single visual image into multiple XObject tiles;
    this reconstruction step merges them back into logical images.
    """
    if not images:
        return []

    n = len(images)
    parent = list(range(n))

    def find(i):
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != i:
            parent[i], i = root, parent[i]
        return root

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    # Union boxes that overlap or are within tolerance distance
    for i in range(n):
        for j in range(i + 1, n):
            b1, b2 = images[i].bbox, images[j].bbox
            h_dist = max(0, max(b1.x0, b2.x0) - min(b1.x1, b2.x1))
            v_dist = max(0, max(b1.y0, b2.y0) - min(b1.y1, b2.y1))
            if h_dist <= tolerance and v_dist <= tolerance:
                union(i, j)

    # Group by connected component and merge bounding boxes
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged = []
    for indices in groups.values():
        bbox = images[indices[0]].bbox
        name = images[indices[0]].name
        for idx in indices[1:]:
            b = images[idx].bbox
            bbox = BoundingBox(min(bbox.x0, b.x0), min(bbox.y0, b.y0), max(bbox.x1, b.x1), max(bbox.y1, b.y1))
            name += f"+{images[idx].name}"
        merged.append(ImageElement(name=name, bbox=bbox))
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Public API — get_anchor_text()
# ─────────────────────────────────────────────────────────────────────────────

def get_anchor_text(pdf_path: str, page_num: int, max_length: int = 4000) -> str:
    """
    Build the anchor text for a PDF page — the OLMoCR 'document anchoring' technique.

    Extracts text blocks (with coordinates) and image bounding boxes from the PDF's
    digital layer, then formats them as a structured string for the model prompt.

    Strategy for fitting within max_length:
      1. Edge elements (min/max x, y coordinates) are always included
      2. Remaining elements are randomly sampled to fill available space
      3. All selected elements are sorted by position for logical reading order

    Args:
        pdf_path: path to the PDF file
        page_num: 1-indexed page number
        max_length: maximum character length for the anchor text (default: 4000)

    Returns:
        Formatted string like:
            Page dimensions: 612.0x792.0
            [72x700]Introduction
            [Image 100x200 to 400x500]
    """
    report = _extract_page_report(pdf_path, page_num)
    result = f"Page dimensions: {report.mediabox.x1:.1f}x{report.mediabox.y1:.1f}\n"

    if max_length < 20:
        return result

    images = _merge_image_elements(report.image_elements)

    # Build image strings
    image_strings = []
    for elem in images:
        s = f"[Image {elem.bbox.x0:.0f}x{elem.bbox.y0:.0f} to {elem.bbox.x1:.0f}x{elem.bbox.y1:.0f}]\n"
        image_strings.append((elem, s))

    # Build text strings
    text_strings = []
    for elem in report.text_elements:
        cleaned = _cleanup_text(elem.text)
        if not cleaned:
            continue
        s = f"[{elem.x:.0f}x{elem.y:.0f}]{cleaned}\n"
        text_strings.append((elem, s))

    # Combine all elements with position info for sorting
    all_elements = []
    for elem, s in image_strings:
        all_elements.append(("image", elem, s, (elem.bbox.x0, elem.bbox.y0)))
    for elem, s in text_strings:
        all_elements.append(("text", elem, s, (elem.x, elem.y)))

    total_length = len(result) + sum(len(s) for _, _, s, _ in all_elements)

    # If everything fits, include all elements
    if total_length <= max_length:
        for _, _, s, _ in all_elements:
            result += s
        return result

    # Otherwise, prioritize edge elements (min/max x,y) then randomly sample the rest
    edge_elements = set()
    if images:
        edge_elements.update([
            min(images, key=lambda e: e.bbox.x0),
            max(images, key=lambda e: e.bbox.x1),
            min(images, key=lambda e: e.bbox.y0),
            max(images, key=lambda e: e.bbox.y1),
        ])
    text_elems = [e for e in report.text_elements if len(e.text.strip()) > 0]
    if text_elems:
        edge_elements.update([
            min(text_elems, key=lambda e: e.x),
            max(text_elems, key=lambda e: e.x),
            min(text_elems, key=lambda e: e.y),
            max(text_elems, key=lambda e: e.y),
        ])

    selected_ids = set()
    selected = []

    # Include edge elements first (they provide spatial context)
    for t, elem, s, pos in all_elements:
        if elem in edge_elements and id(elem) not in selected_ids:
            selected.append((t, elem, s, pos))
            selected_ids.add(id(elem))

    current_length = len(result) + sum(len(s) for _, _, s, _ in selected)
    remaining = [(t, e, s, p) for t, e, s, p in all_elements if id(e) not in selected_ids]
    random.shuffle(remaining)

    # Fill remaining space with randomly sampled elements
    for t, e, s, p in remaining:
        if current_length + len(s) > max_length:
            break
        selected.append((t, e, s, p))
        selected_ids.add(id(e))
        current_length += len(s)

    # Sort by position for logical reading order
    selected.sort(key=lambda x: (x[3][0], x[3][1]))
    for _, _, s, _ in selected:
        result += s

    return result
