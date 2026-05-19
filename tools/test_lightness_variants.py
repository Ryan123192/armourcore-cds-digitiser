"""L-series variants for the MINOR grid that HSV misses.

H1 (sat>=25 no boost) cleanly catches major + text but misses minor
grid (FFEBCC is too pale).  Idea: drawn tracings are MUCH darker than
the minor grid, so a brightness threshold cleanly separates them.

Approaches tested here combine H1's major-line removal with various
brightness-based methods to additionally catch minor grid:

    L0  H1 baseline (major only — minor grid stays)
    L1  H1 + gray>=170 light-pixel paper-conversion
    L2  H1 + gray>=185 light-pixel paper-conversion
    L3  H1 + gray>=200 light-pixel paper-conversion (strictest, safest)
    L4  Light warm pixel: V>=170 + warm-hue + any chroma
    L5  Light orange: V>=180 + warm-hue + S>=10
    L6  Pure brightness: any pixel gray>=160 -> paper (binarize)
    L7  H1 + L4
    L8  H1 + L5

The "light warm" variants only catch pixels that are BOTH light AND
chromatic — pure white paper stays as-is.  The "H1 + grayscale"
variants catch literally anything bright, including pure paper, but
that's harmless (paper -> paper).

    python tools/test_lightness_variants.py
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
    normalise_lighting, normalise_paper_to_white,
    paper_blended_fill, _remove_small_components,
)

REPO = Path(__file__).parent.parent


# =====================================================================
# Helpers
# =====================================================================

def _h1_mask(image_bgr: np.ndarray) -> np.ndarray:
    """H1 from the previous batch: sat>=25 no boost."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array([0, 25, 10], dtype=np.uint8),
                          np.array([40, 255, 255], dtype=np.uint8))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return _remove_small_components(m, 15)


def _light_paper_mask(image_bgr: np.ndarray, gray_min: int) -> np.ndarray:
    """Every pixel brighter than gray_min — will be force-paper'd."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return (gray >= gray_min).astype(np.uint8) * 255


def _light_warm_mask(
    image_bgr: np.ndarray, v_min: int, sat_min: int = 5,
    hue_low: int = 0, hue_high: int = 50,
) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(
        hsv,
        np.array([hue_low,  sat_min, v_min], dtype=np.uint8),
        np.array([hue_high, 255, 255],       dtype=np.uint8),
    )


# =====================================================================
# Variants
# =====================================================================

def L0_baseline_h1(img):
    return _h1_mask(img)


def L1_h1_plus_gray170(img):
    return cv2.bitwise_or(_h1_mask(img), _light_paper_mask(img, 170))


def L2_h1_plus_gray185(img):
    return cv2.bitwise_or(_h1_mask(img), _light_paper_mask(img, 185))


def L3_h1_plus_gray200(img):
    return cv2.bitwise_or(_h1_mask(img), _light_paper_mask(img, 200))


def L4_light_warm_v170(img):
    return _light_warm_mask(img, v_min=170, sat_min=5)


def L5_light_warm_v180_s10(img):
    return _light_warm_mask(img, v_min=180, sat_min=10)


def L6_pure_gray160(img):
    """Any pixel brighter than 160 -> paper (binarize approach)."""
    return _light_paper_mask(img, 160)


def L7_h1_plus_lightwarm(img):
    return cv2.bitwise_or(_h1_mask(img), _light_warm_mask(img, v_min=170, sat_min=5))


def L8_h1_plus_lightorange(img):
    return cv2.bitwise_or(_h1_mask(img),
                          _light_warm_mask(img, v_min=180, sat_min=10))


def L9_h1_plus_lightwarm_strict(img):
    """L7 but tighter on the warm side (V>=170, S>=8)."""
    return cv2.bitwise_or(_h1_mask(img),
                          _light_warm_mask(img, v_min=170, sat_min=8))


METHODS: list[tuple[str, Callable]] = [
    ("L0_baseline_h1",         L0_baseline_h1),
    ("L1_h1_plus_gray170",     L1_h1_plus_gray170),
    ("L2_h1_plus_gray185",     L2_h1_plus_gray185),
    ("L3_h1_plus_gray200",     L3_h1_plus_gray200),
    ("L4_light_warm_v170",     L4_light_warm_v170),
    ("L5_light_warm_v180_s10", L5_light_warm_v180_s10),
    ("L6_pure_gray160",        L6_pure_gray160),
    ("L7_h1_plus_lightwarm",   L7_h1_plus_lightwarm),
    ("L8_h1_plus_lightorange", L8_h1_plus_lightorange),
    ("L9_h1_plus_lightwarm_strict", L9_h1_plus_lightwarm_strict),
]


# =====================================================================
# Runner
# =====================================================================

def _apply(img, mask):
    cleaned = paper_blended_fill(img, mask)
    cleaned = normalise_paper_to_white(cleaned)
    return cleaned


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
    img = normalise_lighting(rect)

    out_root = (REPO / "data" / "outputs" / "hsv_variants" / case
                / time.strftime("%Y%m%d-%H%M%S_lightness"))
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"output: {out_root.relative_to(REPO)}\n")

    cleaned_crops, overlay_crops = [], []
    H, W = img.shape[:2]
    cy0, cy1 = 0, int(H * 0.55)
    cx0, cx1 = 0, int(W * 0.55)

    for name, fn in METHODS:
        print(f"[{name}]")
        t0 = time.time()
        mask = fn(img)
        cleaned = _apply(img, mask)
        elapsed = time.time() - t0
        sub = out_root / name
        sub.mkdir(exist_ok=True)
        cv2.imwrite(str(sub / "cleaned.png"), cleaned)
        ov = _overlay(img, mask)
        cv2.imwrite(str(sub / "overlay.png"), ov)
        print(f"  mask {int((mask > 0).sum()):>9,} px | {elapsed:.1f}s")
        cleaned_crops.append(_label(cleaned[cy0:cy1, cx0:cx1], name))
        overlay_crops.append(_label(ov[cy0:cy1, cx0:cx1], name))

    h_c, w_c = cleaned_crops[0].shape[:2]
    cols, rows = 5, 2
    grid = np.full((rows * h_c, cols * w_c, 3), 255, dtype=np.uint8)
    for i, c in enumerate(cleaned_crops):
        r_, c_ = divmod(i, cols)
        grid[r_ * h_c:(r_ + 1) * h_c, c_ * w_c:(c_ + 1) * w_c] = c
    cv2.imwrite(str(out_root / "summary_cleaned.png"), grid)

    grid_o = np.full((rows * h_c, cols * w_c, 3), 255, dtype=np.uint8)
    for i, c in enumerate(overlay_crops):
        r_, c_ = divmod(i, cols)
        grid_o[r_ * h_c:(r_ + 1) * h_c, c_ * w_c:(c_ + 1) * w_c] = c
    cv2.imwrite(str(out_root / "summary_overlay.png"), grid_o)

    print(f"\nopen: {out_root.relative_to(REPO)}/summary_cleaned.png")
    print(f"open: {out_root.relative_to(REPO)}/summary_overlay.png")


if __name__ == "__main__":
    main()
