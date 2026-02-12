"""
models.py — Shared data classes for the OLMoCR-style pipeline.

Contains dataclass definitions used across all modules:
  - BoundingBox: rectangular region on a PDF page
  - TextElement: a text block with its (x, y) position
  - ImageElement: an embedded image with its bounding box
  - PageReport: full extraction report for a single PDF page
  - PageResponse: structured response returned by GPT-4o
"""

from dataclasses import dataclass
from typing import List, Optional

from pypdf.generic import RectangleObject


@dataclass(frozen=True)
class BoundingBox:
    """A rectangular region defined by two corner points (x0, y0) and (x1, y1)."""
    x0: float
    y0: float
    x1: float
    y1: float

    @staticmethod
    def from_rectangle(rect: RectangleObject) -> "BoundingBox":
        """Create a BoundingBox from a pypdf RectangleObject."""
        return BoundingBox(float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3]))


@dataclass(frozen=True)
class TextElement:
    """A single text block extracted from a PDF page, with its position."""
    text: str
    x: float  # horizontal position (origin at lower-left)
    y: float  # vertical position (origin at lower-left)


@dataclass(frozen=True)
class ImageElement:
    """An embedded image in a PDF page, identified by name and bounding box."""
    name: str
    bbox: BoundingBox


@dataclass(frozen=True)
class PageReport:
    """
    Full extraction report for a single PDF page.
    Contains the page dimensions (mediabox) and all extracted text/image elements.
    """
    mediabox: BoundingBox
    text_elements: List[TextElement]
    image_elements: List[ImageElement]


@dataclass(frozen=True)
class PageResponse:
    """
    Structured response from GPT-4o following OLMoCR's schema.
    
    The chain-of-thought fields (language, rotation, table, diagram) are
    intentionally ordered BEFORE natural_text — GPT-4o answers fields in
    schema order, so reasoning about the page first improves text quality.
    """
    primary_language: Optional[str]
    is_rotation_valid: bool
    rotation_correction: int
    is_table: bool
    is_diagram: bool
    natural_text: Optional[str]
