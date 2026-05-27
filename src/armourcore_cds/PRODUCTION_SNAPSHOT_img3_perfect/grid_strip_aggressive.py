"""Phase 2 sibling: aggressive major-grid removal for dark-border images.

Why a separate module
=====================
The standard ``grid_strip.py`` protects very-dark pixels (gray <= 90)
to avoid erasing tool ink that happens to cross a grid line.  But on
img1, the dark camera border pulls EVERY pixel near the left edge
below gray 90 - so the grid-line bands inside the border are protected
and survive grid_strip, then connect everything together for findContours.

This sibling strips major grid lines MORE aggressively:
  * Wider bands (8px half-width instead of 5px)
  * Lower ink-protect threshold (gray <= 60 instead of 90)
  * Optional: kill grid lines completely (no ink protect) within a
    user-specified "border zone" along image edges

Tool ink that genuinely crosses a major-grid line gets a tiny break
(1-2mm) but phase 3's per-shape rescue + RETR_CCOMP hole-preference
will bridge those without distortion.
"""
from __future__ import annotations

import cv2
import numpy as np

from armourcore_cds.phase1.marker_rectify_fast_v4 import (
    PAPER_W_MM, PAPER_H_MM,
)


def strip_major_grid_aggressive(
    cleaned_bgr: np.ndarray,
    major_spacing_mm: float = 50.0,
    band_half_px: int = 8,
    ink_protect_max_gray: int = 60,
    border_zone_px: int = 250,
    border_ink_protect: int = 0,
) -> np.ndarray:
    """Strip major grid lines with wider bands + lower ink protect.

    In the border zone (within ``border_zone_px`` of image edge), ink
    protect is overridden to ``border_ink_protect`` (0 = kill everything
    along the band, even dark pixels) since the dark-border pixels are
    NOT tool ink, just camera artefact.
    """
    H, W = cleaned_bgr.shape[:2]
    px_per_mm_x = W / PAPER_W_MM
    px_per_mm_y = H / PAPER_H_MM

    band_mask = np.zeros((H, W), dtype=np.uint8)
    t = 2 * band_half_px + 1
    n_v = int(PAPER_W_MM / major_spacing_mm) + 1
    n_h = int(PAPER_H_MM / major_spacing_mm) + 1
    for i in range(n_v + 1):
        x = int(round(i * major_spacing_mm * px_per_mm_x))
        if 0 <= x < W:
            cv2.line(band_mask, (x, 0), (x, H - 1), 255, t)
    for j in range(n_h + 1):
        y = int(round(j * major_spacing_mm * px_per_mm_y))
        if 0 <= y < H:
            cv2.line(band_mask, (0, y), (W - 1, y), 255, t)

    gray = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)

    # Border zone has lower / zero ink-protect (border pixels aren't ink)
    border_zone = np.zeros((H, W), dtype=bool)
    if border_zone_px > 0:
        border_zone[:border_zone_px, :] = True
        border_zone[-border_zone_px:, :] = True
        border_zone[:, :border_zone_px] = True
        border_zone[:, -border_zone_px:] = True

    # Decide which band pixels to erase
    in_band = band_mask > 0
    in_border = in_band & border_zone
    in_interior = in_band & ~border_zone

    # Interior bands: protect ink darker than ink_protect_max_gray
    kill_interior = in_interior & (gray > ink_protect_max_gray)
    # Border bands: protect ONLY very dark ink (or nothing if border_ink_protect=0)
    if border_ink_protect <= 0:
        kill_border = in_border   # erase regardless of darkness
    else:
        kill_border = in_border & (gray > border_ink_protect)

    kill = kill_interior | kill_border

    out = cleaned_bgr.copy()
    out[kill] = (255, 255, 255)
    return out


def strip_dark_border(cleaned_bgr: np.ndarray,
                     border_zone_px: int = 200,
                     ink_threshold_gray: int = 100) -> np.ndarray:
    """Whiten anything in the border zone that's an even-darker-than-paper
    blob (camera vignetting / framing offset).  Tool ink in the border
    zone (darker still) survives.

    Image-edge dark border becomes paper-white, allowing per-shape close
    operations to work without merging the giant border component.
    """
    H, W = cleaned_bgr.shape[:2]
    gray = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
    border_zone = np.zeros((H, W), dtype=bool)
    border_zone[:border_zone_px, :] = True
    border_zone[-border_zone_px:, :] = True
    border_zone[:, :border_zone_px] = True
    border_zone[:, -border_zone_px:] = True

    # In border zone: anything not very dark (i.e. not tool ink) -> paper
    # Tool ink in pen drawings is typically gray < 100.
    not_ink = gray > ink_threshold_gray
    kill = border_zone & not_ink
    out = cleaned_bgr.copy()
    out[kill] = (255, 255, 255)
    return out
