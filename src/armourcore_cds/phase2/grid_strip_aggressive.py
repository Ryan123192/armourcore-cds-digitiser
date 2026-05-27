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


def strip_major_grid_unconditional(
    cleaned_bgr: np.ndarray,
    major_spacing_mm: float = 50.0,
    band_half_px: int = 4,
) -> np.ndarray:
    """UNCONDITIONAL major grid kill - no ink protection.

    Strips every pixel within ``band_half_px`` of a major grid line.
    Tool ink that genuinely crosses a major-grid line gets a tiny
    (6-9 px ~= 0.6-0.9 mm) break.  Phase 3's per-shape rescue + RETR_CCOMP
    hole-preference bridge those breaks.

    Use this when EVEN DARK grid lines survive (e.g. img1 where the
    template print came out as dark as pen ink).
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

    out = cleaned_bgr.copy()
    out[band_mask > 0] = (255, 255, 255)
    return out


def strip_grid_by_component(
    cleaned_bgr: np.ndarray,
    major_spacing_mm: float = 50.0,
    minor_spacing_mm: float = 10.0,
    major_band_half_px: int = 4,
    minor_band_half_px: int = 2,
    in_band_kill_frac: float = 0.55,
    min_component_area: int = 80,
) -> np.ndarray:
    """Component-based grid kill.

    For each connected component of ink:
      * Count how many pixels lie inside a major OR minor grid band.
      * If `in_band_fraction > in_band_kill_frac` -> entire component is
        grid residue; ERASE it.
      * Otherwise it's a tool outline that crosses the band; KEEP it
        completely intact.

    A pure grid-line residue has ~100% in-band pixels.  A tool outline
    even when crossing 3 grid lines only has ~5-10% in-band pixels.
    The 0.55 threshold safely splits the two.
    """
    H, W = cleaned_bgr.shape[:2]
    px_per_mm_x = W / PAPER_W_MM
    px_per_mm_y = H / PAPER_H_MM
    gray = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
    ink = (gray < 170).astype(np.uint8) * 255

    band_mask = np.zeros((H, W), dtype=np.uint8)
    # Major bands
    n_v = int(PAPER_W_MM / major_spacing_mm) + 1
    n_h = int(PAPER_H_MM / major_spacing_mm) + 1
    t = 2 * major_band_half_px + 1
    for i in range(n_v + 1):
        x = int(round(i * major_spacing_mm * px_per_mm_x))
        if 0 <= x < W:
            cv2.line(band_mask, (x, 0), (x, H - 1), 255, t)
    for j in range(n_h + 1):
        y = int(round(j * major_spacing_mm * px_per_mm_y))
        if 0 <= y < H:
            cv2.line(band_mask, (0, y), (W - 1, y), 255, t)
    # Minor bands
    n_v_min = int(PAPER_W_MM / minor_spacing_mm) + 1
    n_h_min = int(PAPER_H_MM / minor_spacing_mm) + 1
    tm = 2 * minor_band_half_px + 1
    for i in range(n_v_min + 1):
        x = int(round(i * minor_spacing_mm * px_per_mm_x))
        if 0 <= x < W:
            cv2.line(band_mask, (x, 0), (x, H - 1), 255, tm)
    for j in range(n_h_min + 1):
        y = int(round(j * minor_spacing_mm * px_per_mm_y))
        if 0 <= y < H:
            cv2.line(band_mask, (0, y), (W - 1, y), 255, tm)

    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(
        ink, connectivity=8)
    kill_mask = np.zeros_like(ink)
    for cid in range(1, n_lbl):
        area = stats[cid, cv2.CC_STAT_AREA]
        if area < min_component_area:
            continue
        comp = (lbl == cid)
        in_band = comp & (band_mask > 0)
        frac = in_band.sum() / max(area, 1)
        if frac >= in_band_kill_frac:
            kill_mask[comp] = 255

    out = cleaned_bgr.copy()
    out[kill_mask > 0] = (255, 255, 255)
    return out


def strip_grid_intersection_safe(
    cleaned_bgr: np.ndarray,
    major_spacing_mm: float = 50.0,
    band_half_px: int = 3,
    curve_protect_dilation: int = 9,
) -> np.ndarray:
    """Smart major-grid kill that preserves tool outlines.

    For each pixel inside a major-grid band:
      * If it's part of a CURVED (non-axis-aligned) structure, KEEP it
        (it's part of a tool outline crossing the grid line).
      * Otherwise REMOVE it (it's grid-line residue).

    "Curved structure" = the pixel sits within ``curve_protect_dilation``
    of any ink pixel that is NOT itself in a grid band.  Tool outlines
    have most of their pixels OFF the grid band, so band pixels near
    them get protected.  Pure grid-line residue has all its mass ON the
    band so no off-band ink protects it.
    """
    H, W = cleaned_bgr.shape[:2]
    px_per_mm_x = W / PAPER_W_MM
    px_per_mm_y = H / PAPER_H_MM
    gray = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
    ink = (gray < 170).astype(np.uint8) * 255

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

    # Ink that lies OUTSIDE any band - this is the curved tool outline.
    off_band_ink = (ink > 0) & (band_mask == 0)
    off_band_ink_u8 = off_band_ink.astype(np.uint8) * 255
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (curve_protect_dilation * 2 + 1, curve_protect_dilation * 2 + 1))
    near_curve = cv2.dilate(off_band_ink_u8, k)

    # Kill = in-band ink that is NOT near a curve
    in_band_ink = (ink > 0) & (band_mask > 0)
    kill = in_band_ink & (near_curve == 0)

    out = cleaned_bgr.copy()
    out[kill] = (255, 255, 255)
    return out


def strip_minor_grid_unconditional(
    cleaned_bgr: np.ndarray,
    minor_spacing_mm: float = 10.0,
    band_half_px: int = 2,
) -> np.ndarray:
    """Unconditional minor grid kill - thinner bands.  Tool ink across
    a minor grid line gets a 2-4 px (~0.3 mm) break which is sub-stroke
    width and gets bridged by Phase 3 trivially.
    """
    H, W = cleaned_bgr.shape[:2]
    px_per_mm_x = W / PAPER_W_MM
    px_per_mm_y = H / PAPER_H_MM

    band_mask = np.zeros((H, W), dtype=np.uint8)
    t = 2 * band_half_px + 1
    n_v = int(PAPER_W_MM / minor_spacing_mm) + 1
    n_h = int(PAPER_H_MM / minor_spacing_mm) + 1
    for i in range(n_v + 1):
        x = int(round(i * minor_spacing_mm * px_per_mm_x))
        if 0 <= x < W:
            cv2.line(band_mask, (x, 0), (x, H - 1), 255, t)
    for j in range(n_h + 1):
        y = int(round(j * minor_spacing_mm * px_per_mm_y))
        if 0 <= y < H:
            cv2.line(band_mask, (0, y), (W - 1, y), 255, t)

    out = cleaned_bgr.copy()
    out[band_mask > 0] = (255, 255, 255)
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
