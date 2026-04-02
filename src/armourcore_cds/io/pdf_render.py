from __future__ import annotations

from pathlib import Path

import fitz  # pymupdf
import numpy as np


def render_pdf_first_page(path: Path, dpi: int = 400) -> np.ndarray:
    """Render the first page of a single-page PDF to a BGR uint8 image."""
    doc = fitz.open(path)
    if doc.page_count < 1:
        raise ValueError(f"PDF has no pages: {path}")

    page = doc.load_page(0)
    scale = dpi / 72.0
    matrix = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    rgb = data[:, :, :3]
    bgr = rgb[:, :, ::-1].copy()
    return bgr
