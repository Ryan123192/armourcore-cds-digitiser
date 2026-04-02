from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from armourcore_cds.io.pdf_render import render_pdf_first_page

SUPPORTED_IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
SUPPORTED_PDF_SUFFIXES = {'.pdf'}


def load_input(path: Path, pdf_dpi: int = 400) -> tuple[np.ndarray, dict]:
    """Load one image or single-page PDF as a BGR uint8 raster."""
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix in SUPPORTED_PDF_SUFFIXES:
        image = render_pdf_first_page(path, dpi=pdf_dpi)
        meta = {
            'input_kind': 'pdf',
            'source_suffix': suffix,
            'pdf_render_dpi': pdf_dpi,
        }
        return image, meta

    if suffix in SUPPORTED_IMAGE_SUFFIXES:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'OpenCV could not load image: {path}')
        meta = {
            'input_kind': 'image',
            'source_suffix': suffix,
            'pdf_render_dpi': None,
        }
        return image, meta

    raise ValueError(f'Unsupported input type: {suffix}')
