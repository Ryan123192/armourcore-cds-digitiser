"""Lightness/threshold variants v2: hard-threshold-to-white approach.

Insight: when the removal mask is HUGE (most of the image), the Gaussian
paper_blended_fill fails because there's not enough "kept" paper to blend
from.  Instead, just FORCE light pixels to pure paper-white (255,255,255).

Also tries a "soft" gradient — pixels are nudged toward white in
proportion to how light they already are.

    python tools/test_lightness_v2.py
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


def _h1_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array([0, 25, 10], dtype=np.uint8),
                          np.array([40, 255, 255], dtype=np.uint8))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return _remove_small_components(m, 15)


def _apply_h1_then_force_white(img: np.ndarray, gray_threshold: int) -> np.ndarray:
    """Apply H1 removal first, then hard-threshold anything gray>=T to white."""
    # Step 1 — H1 paper-blend on the major orange
    h1 = _h1_mask(img)
    cleaned = paper_blended_fill(img, h1)
    cleaned = normalise_paper_to_white(cleaned)
    # Step 2 — hard-clamp any light pixel to pure white
    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    cleaned[gray >= gray_threshold] = (255, 255, 255)
    return cleaned


def _apply_h1_then_soft_lift(img: np.ndarray, lift_start: int = 140,
                              lift_end: int = 200) -> np.ndarray:
    """Apply H1, then smoothly lift pixels toward white.

    Pixels darker than lift_start stay as-is.  Pixels between lift_start and
    lift_end get partial brightening proportional to position.  Pixels at or
    above lift_end go to pure white.
    """
    h1 = _h1_mask(img)
    cleaned = paper_blended_fill(img, h1)
    cleaned = normalise_paper_to_white(cleaned)
    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lift_factor = np.clip(
        (gray - lift_start) / max(1, (lift_end - lift_start)), 0.0, 1.0
    )
    out = cleaned.astype(np.float32)
    target = np.array([255.0, 255.0, 255.0])
    for c in range(3):
        out[..., c] = out[..., c] + lift_factor * (target[c] - out[..., c])
    return out.astype(np.uint8)


# =====================================================================
# Variants
# =====================================================================

def L0_baseline_h1_only(img):
    h1 = _h1_mask(img)
    cleaned = paper_blended_fill(img, h1)
    return normalise_paper_to_white(cleaned)


def L10_h1_then_hard_white_140(img):
    return _apply_h1_then_force_white(img, 140)


def L11_h1_then_hard_white_160(img):
    return _apply_h1_then_force_white(img, 160)


def L12_h1_then_hard_white_180(img):
    return _apply_h1_then_force_white(img, 180)


def L13_h1_then_hard_white_200(img):
    return _apply_h1_then_force_white(img, 200)


def L14_h1_then_soft_140_200(img):
    return _apply_h1_then_soft_lift(img, 140, 200)


def L15_h1_then_soft_160_220(img):
    return _apply_h1_then_soft_lift(img, 160, 220)


def L16_h1_then_soft_180_240(img):
    return _apply_h1_then_soft_lift(img, 180, 240)


def L17_h3_then_hard_white_180(img):
    """H3 (sat>=35) is even stricter than H1 — try with white-clamp 180."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h3 = cv2.inRange(hsv, np.array([0, 35, 10], dtype=np.uint8),
                          np.array([40, 255, 255], dtype=np.uint8))
    h3 = _remove_small_components(h3, 15)
    cleaned = paper_blended_fill(img, h3)
    cleaned = normalise_paper_to_white(cleaned)
    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    cleaned[gray >= 180] = (255, 255, 255)
    return cleaned


METHODS: list[tuple[str, Callable]] = [
    ("L0_baseline_h1_only",     L0_baseline_h1_only),
    ("L10_h1_hardwhite_140",    L10_h1_then_hard_white_140),
    ("L11_h1_hardwhite_160",    L11_h1_then_hard_white_160),
    ("L12_h1_hardwhite_180",    L12_h1_then_hard_white_180),
    ("L13_h1_hardwhite_200",    L13_h1_then_hard_white_200),
    ("L14_h1_soft_140_200",     L14_h1_then_soft_140_200),
    ("L15_h1_soft_160_220",     L15_h1_then_soft_160_220),
    ("L16_h1_soft_180_240",     L16_h1_then_soft_180_240),
    ("L17_h3_hardwhite_180",    L17_h3_then_hard_white_180),
]


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
                / time.strftime("%Y%m%d-%H%M%S_lightness_v2"))
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"output: {out_root.relative_to(REPO)}\n")

    crops = []
    H, W = img.shape[:2]
    cy0, cy1 = 0, int(H * 0.55); cx0, cx1 = 0, int(W * 0.55)

    for name, fn in METHODS:
        print(f"[{name}]", end=" ")
        t0 = time.time()
        cleaned = fn(img)
        print(f"{time.time() - t0:.1f}s")
        sub = out_root / name
        sub.mkdir(exist_ok=True)
        cv2.imwrite(str(sub / "cleaned.png"), cleaned)
        crops.append(_label(cleaned[cy0:cy1, cx0:cx1], name))

    h_c, w_c = crops[0].shape[:2]
    cols, rows = 3, 3
    grid = np.full((rows * h_c, cols * w_c, 3), 255, dtype=np.uint8)
    for i, c in enumerate(crops):
        r_, c_ = divmod(i, cols)
        grid[r_ * h_c:(r_ + 1) * h_c, c_ * w_c:(c_ + 1) * w_c] = c
    cv2.imwrite(str(out_root / "summary_cleaned.png"), grid)
    print(f"\nopen: {out_root.relative_to(REPO)}/summary_cleaned.png")


if __name__ == "__main__":
    main()
