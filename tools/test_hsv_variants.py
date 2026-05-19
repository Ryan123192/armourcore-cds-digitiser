"""Batch test of HSV-only grid-removal variants on V2BlueColourTest01.

The user observed:
* The current production (LAB-primary) leaves minor grid lines and one
  major grid line completely intact on V2 prints
* Curved-band geometry detection is too destructive on pen / tool ink
* Wants to disable bands for now and dial in HSV-only detection

Approach: take the rectified image, apply ONLY an HSV mask (no LAB, no
bands), fill with paper_blended_fill, normalise to white.  Try several
HSV threshold combinations to find the sweet spot for V2 inputs.

    python tools/test_hsv_variants.py
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np

from armourcore_cds.phase2.trace_isolation import (
    normalise_lighting,
    normalise_paper_to_white,
    boost_chroma,
    paper_blended_fill,
    _remove_small_components,
    build_lab_orange_mask,
)

REPO = Path(__file__).parent.parent


# =====================================================================
# Helpers
# =====================================================================

def _hsv_mask(
    image_bgr: np.ndarray,
    hue_low: int = 0,
    hue_high: int = 40,
    sat_min: int = 25,
    val_min: int = 10,
    chroma_boost: float = 1.0,
    close_px: int = 2,
    min_component_px: int = 15,
) -> np.ndarray:
    if chroma_boost and chroma_boost > 1.0:
        det = boost_chroma(image_bgr, gain=chroma_boost)
    else:
        det = image_bgr
    hsv = cv2.cvtColor(det, cv2.COLOR_BGR2HSV)
    lower = np.array([hue_low, sat_min, val_min], dtype=np.uint8)
    upper = np.array([hue_high, 255, 255], dtype=np.uint8)
    m = cv2.inRange(hsv, lower, upper)
    if close_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_px * 2 + 1, close_px * 2 + 1))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    if min_component_px > 0:
        m = _remove_small_components(m, min_component_px)
    return m


def _apply_removal(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Standard paper-blend + normalise-to-white cleaning."""
    cleaned = paper_blended_fill(image_bgr, mask)
    cleaned = normalise_paper_to_white(cleaned)
    return cleaned


# =====================================================================
# Variants
# =====================================================================

def H0_lab_only_baseline(img):
    """Current production: LAB a* primary, no HSV."""
    return build_lab_orange_mask(img, a_threshold=2)


def H1_hsv_no_boost_sat25(img):
    """No chroma boost — raw HSV at sat>=25 (V2 grid has raw S=75, tool ink raw S=15)."""
    return _hsv_mask(img, sat_min=25, chroma_boost=1.0)


def H2_hsv_no_boost_sat15(img):
    """No boost, lower threshold (catches more, riskier on tool ink)."""
    return _hsv_mask(img, sat_min=15, chroma_boost=1.0)


def H3_hsv_no_boost_sat35(img):
    """No boost, higher threshold (safer, may miss faint grid)."""
    return _hsv_mask(img, sat_min=35, chroma_boost=1.0)


def H4_hsv_no_boost_sat45(img):
    """No boost, very strict threshold."""
    return _hsv_mask(img, sat_min=45, chroma_boost=1.0)


def H5_hsv_boost2_sat35(img):
    """Mild chroma boost 2.0x, sat>=35 (tool ink S=15 -> 30, below 35)."""
    return _hsv_mask(img, sat_min=35, chroma_boost=2.0)


def H6_hsv_boost1_5_sat25(img):
    """Mild boost 1.5x, default threshold (tool S=15 -> 22.5, just below 25)."""
    return _hsv_mask(img, sat_min=25, chroma_boost=1.5)


def H7_hsv_AND_lab(img):
    """Intersection: HSV chromatic AND LAB warm — strictest, cleanest."""
    hsv = _hsv_mask(img, sat_min=15, chroma_boost=1.0)
    lab = build_lab_orange_mask(img, a_threshold=2)
    return cv2.bitwise_and(hsv, lab)


def H8_hsv_OR_lab(img):
    """Union: HSV chromatic OR LAB warm — broadest, may grab tool ink."""
    hsv = _hsv_mask(img, sat_min=25, chroma_boost=1.0)
    lab = build_lab_orange_mask(img, a_threshold=2)
    return cv2.bitwise_or(hsv, lab)


def H9_hsv_no_boost_sat20_widehue(img):
    """No boost, sat>=20, wider hue range (5-50)."""
    return _hsv_mask(img, hue_low=0, hue_high=50, sat_min=20, chroma_boost=1.0)


METHODS: list[tuple[str, Callable]] = [
    ("H0_lab_only_baseline",       H0_lab_only_baseline),
    ("H1_hsv_no_boost_sat25",      H1_hsv_no_boost_sat25),
    ("H2_hsv_no_boost_sat15",      H2_hsv_no_boost_sat15),
    ("H3_hsv_no_boost_sat35",      H3_hsv_no_boost_sat35),
    ("H4_hsv_no_boost_sat45",      H4_hsv_no_boost_sat45),
    ("H5_hsv_boost2_sat35",        H5_hsv_boost2_sat35),
    ("H6_hsv_boost1_5_sat25",      H6_hsv_boost1_5_sat25),
    ("H7_hsv_AND_lab",             H7_hsv_AND_lab),
    ("H8_hsv_OR_lab",              H8_hsv_OR_lab),
    ("H9_hsv_no_boost_sat20_wide", H9_hsv_no_boost_sat20_widehue),
]


# =====================================================================
# Runner
# =====================================================================

def _overlay(base, mask, colour=(0, 0, 255), alpha=0.55):
    out = base.copy()
    if not np.any(mask):
        return out
    paint = np.zeros_like(base)
    paint[:] = colour
    blend = cv2.addWeighted(out, 1.0 - alpha, paint, alpha, 0)
    return np.where(mask[..., None] > 0, blend, out)


def _label(img, txt, scale=0.6):
    out = img.copy()
    H, W = out.shape[:2]
    band = int(34 * scale)
    cv2.rectangle(out, (0, 0), (W, band), (0, 0, 0), -1)
    cv2.putText(out, txt, (8, band - 8), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main():
    case = "V2BlueColourTest01"
    p1_run = sorted((REPO / "outputs" / "runs").glob(f"*{case}*"))[-1]
    rect = cv2.imread(str(p1_run / "scaled_design_area.png"))
    rect_norm = normalise_lighting(rect)

    out_root = (REPO / "data" / "outputs" / "hsv_variants" / case
                / time.strftime("%Y%m%d-%H%M%S"))
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"output: {out_root.relative_to(REPO)}\n")

    cleaned_crops = []
    overlay_crops = []
    H, W = rect_norm.shape[:2]
    # Crop of interest: full top-left (where text + minor grid live)
    cy0, cy1 = 0, int(H * 0.55)
    cx0, cx1 = 0, int(W * 0.55)

    for name, fn in METHODS:
        print(f"[{name}]")
        t0 = time.time()
        mask = fn(rect_norm)
        cleaned = _apply_removal(rect_norm, mask)
        elapsed = time.time() - t0

        sub = out_root / name
        sub.mkdir(exist_ok=True)
        cv2.imwrite(str(sub / "cleaned.png"), cleaned)
        overlay = _overlay(rect_norm, mask)
        cv2.imwrite(str(sub / "overlay.png"), overlay)
        print(f"  mask {int((mask > 0).sum()):>9,} px | {elapsed:.1f}s")

        cleaned_crops.append(_label(cleaned[cy0:cy1, cx0:cx1], name))
        overlay_crops.append(_label(overlay[cy0:cy1, cx0:cx1], name))

    # 5x2 grid for 10 methods
    h_c, w_c = cleaned_crops[0].shape[:2]
    cols, rows = 5, 2
    grid = np.full((rows * h_c, cols * w_c, 3), 255, dtype=np.uint8)
    for i, c in enumerate(cleaned_crops):
        r_, c_ = divmod(i, cols)
        grid[r_ * h_c:(r_ + 1) * h_c, c_ * w_c:(c_ + 1) * w_c] = c
    cv2.imwrite(str(out_root / "summary_cleaned_grid.png"), grid)

    grid_o = np.full((rows * h_c, cols * w_c, 3), 255, dtype=np.uint8)
    for i, c in enumerate(overlay_crops):
        r_, c_ = divmod(i, cols)
        grid_o[r_ * h_c:(r_ + 1) * h_c, c_ * w_c:(c_ + 1) * w_c] = c
    cv2.imwrite(str(out_root / "summary_overlay_grid.png"), grid_o)

    print(f"\nopen:  {out_root.relative_to(REPO)}/summary_cleaned_grid.png")
    print(f"open:  {out_root.relative_to(REPO)}/summary_overlay_grid.png")


if __name__ == "__main__":
    main()
