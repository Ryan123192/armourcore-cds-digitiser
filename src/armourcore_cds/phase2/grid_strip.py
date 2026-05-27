"""Phase 2 sibling: subtract template major-grid lines from cleaned image.

Symptom this fixes
==================
After Phase 2 v14 adaptive, both img1 and img2 still have faint major
grid line residue.  Major grid lines are PRINTED at known geometric
positions (every 50mm).  Where two major lines cross a tool outline,
they leave small light-grey segments that:

  * cv2.findContours sees as small closed RECTANGLES (the four corners
    of a grid cell)
  * vectorise reports as extra paths
  * the user sees as "weird squares" intruding into shape vectors

Strategy
========
The image is rectified to PAPER_W_MM x PAPER_H_MM exactly.  We know
every major grid line is at multiples of 50mm.  So just MASK those
bands directly - lift those pixels to paper-white, but only where the
existing pixel is BRIGHTER than a tool-ink threshold (so we don't
erase ink that happens to cross a grid line).

This is a Phase 2 POST-PROCESS that runs AFTER the v14 adaptive
cleaner.  v14 stays untouched.
"""
from __future__ import annotations

import cv2
import numpy as np

from armourcore_cds.phase1.marker_rectify_fast_v4 import (
    PAPER_W_MM, PAPER_H_MM,
)


def strip_major_grid_lines(
    cleaned_bgr: np.ndarray,
    major_spacing_mm: float = 50.0,
    band_half_px: int = 5,
    ink_protect_max_gray: int = 90,
) -> np.ndarray:
    """Erase template major-grid-line bands in *cleaned_bgr*.

    Pixels within ``band_half_px`` of any major-line position are set to
    white, UNLESS they're dark enough to be tool ink (gray
    <= ``ink_protect_max_gray``) - those stay.  This preserves any tool
    stroke that happens to lie under or cross a grid line.

    Returns a new BGR image; input is not mutated.
    """
    H, W = cleaned_bgr.shape[:2]
    px_per_mm_x = W / PAPER_W_MM
    px_per_mm_y = H / PAPER_H_MM

    # Build a mask of grid-line band pixels
    band_mask = np.zeros((H, W), dtype=np.uint8)
    n_v = int(PAPER_W_MM / major_spacing_mm) + 1
    n_h = int(PAPER_H_MM / major_spacing_mm) + 1
    t = 2 * band_half_px + 1
    for i in range(n_v + 1):
        x = int(round(i * major_spacing_mm * px_per_mm_x))
        if 0 <= x < W:
            cv2.line(band_mask, (x, 0), (x, H - 1), 255, t)
    for j in range(n_h + 1):
        y = int(round(j * major_spacing_mm * px_per_mm_y))
        if 0 <= y < H:
            cv2.line(band_mask, (0, y), (W - 1, y), 255, t)

    # Protect tool ink (dark pixels)
    gray = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
    ink = gray <= ink_protect_max_gray
    kill = (band_mask > 0) & (~ink)

    out = cleaned_bgr.copy()
    out[kill] = (255, 255, 255)
    return out


def strip_minor_grid_residue(
    cleaned_bgr: np.ndarray,
    minor_spacing_mm: float = 10.0,
    band_half_px: int = 2,
    ink_protect_max_gray: int = 110,
) -> np.ndarray:
    """Same idea but for minor grid lines.  Thinner bands, slightly
    higher ink-protect threshold so very-thin pen strokes survive."""
    H, W = cleaned_bgr.shape[:2]
    px_per_mm_x = W / PAPER_W_MM
    px_per_mm_y = H / PAPER_H_MM

    band_mask = np.zeros((H, W), dtype=np.uint8)
    n_v = int(PAPER_W_MM / minor_spacing_mm) + 1
    n_h = int(PAPER_H_MM / minor_spacing_mm) + 1
    t = 2 * band_half_px + 1
    for i in range(n_v + 1):
        x = int(round(i * minor_spacing_mm * px_per_mm_x))
        if 0 <= x < W:
            cv2.line(band_mask, (x, 0), (x, H - 1), 255, t)
    for j in range(n_h + 1):
        y = int(round(j * minor_spacing_mm * px_per_mm_y))
        if 0 <= y < H:
            cv2.line(band_mask, (0, y), (W - 1, y), 255, t)

    gray = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
    ink = gray <= ink_protect_max_gray
    kill = (band_mask > 0) & (~ink)

    out = cleaned_bgr.copy()
    out[kill] = (255, 255, 255)
    return out


def strip_all_grid_residue(cleaned_bgr: np.ndarray) -> np.ndarray:
    """Sequential: strip major then minor.  Returns cleaned BGR."""
    out = strip_major_grid_lines(cleaned_bgr)
    out = strip_minor_grid_residue(out)
    return out
