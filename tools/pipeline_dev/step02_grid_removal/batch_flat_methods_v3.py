"""Step 02 — IMPROVED batch test on the 4 flat cases.

After the first 8-method round we learned:
  - M2 CMYK-K with fixed threshold = 40 over-detects (catches grid + paper)
  - M3 blackhat = too noisy
  - M5 bg_subtract = grid sneaks through
  - M7 frangi = too sparse, loses pencil

This v3 replaces those with smarter alternatives, pivoting to a key
insight: **identify ink directly by its DUAL signature** (dark AND
chromatically neutral) instead of "remove grid, hope what's left is ink."

Methods
=======

    M0_L11_baseline
        Production L11.  Reference.

    M1_lab_strict_clamp
        Lab Δa* (with brightness floor L>=120) + L11 hard clamp.
        Current best from previous compare.

    M2_cmyk_k_otsu
        Same CMYK-K extraction but Otsu auto-thresholds the K channel
        instead of using a fixed cutoff.  Adapts to per-image
        K distribution.

    M3_lab_ink_signature  *** NEW PRIMARY CANDIDATE ***
        Identifies INK directly by its dual color signature:
          1. Significantly darker than paper L baseline (>= 30 grey)
          2. Chromatically neutral (|a-ap| < 8 AND |b-bp| < 8)
        Pixels matching BOTH are ink; everything else becomes paper white.
        This directly attacks the "neutral dark = ink, warm = grid"
        separation the user described.

    M4_rgb_neutral_dark
        Pure-RGB version of M3:
            ink = max(R,G,B) < dark_threshold  AND  (max - min) < chromaticity_threshold
        No Lab transform, faster, similar semantics.

    M5_lab_signature_plus_bg_subtract
        M3 + background-subtraction sanity check.  Pixel is ink iff M3
        flags it AND its local Gaussian-blur surroundings are brighter
        by at least 15 grey (excludes broad uniform dark regions).
        Designed to reject any false-positive ink in broad shadowed
        areas (relevant later for creases).

    M6_lab_decorrelation
        Best non-trivial method from previous round.  Project (Δa, Δb)
        onto paper-to-orange axis; suppress chromatic pixels by
        setting them to grey; then threshold L.

    M7_sauvola
        Industry-standard adaptive doc binarisation.  Reference for
        comparison.

Outputs: same shape as batch_flat_methods.py.

    python tools/pipeline_dev/step02_grid_removal/batch_flat_methods_v3.py
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

REPO = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

import cv2
import numpy as np
from skimage.filters import threshold_sauvola

from armourcore_cds.phase1.marker_rectify_fast_v4 import (
    rectify_with_markers_fast_v4,
    PAPER_W_MM, PAPER_H_MM, DEFAULT_PX_PER_MM,
)
from armourcore_cds.phase2.trace_isolation import (
    isolate_trace_candidates, normalise_lighting,
    paper_blended_fill, normalise_paper_to_white,
)
from armourcore_cds.phase2.trace_isolation_v2 import (
    estimate_paper_lab,
)
from armourcore_cds.templates.registry import load_template_config
from tools.pipeline_dev.corpus import (
    RAW_IMAGES_DIR, make_run_dir, label_image, grid_montage,
)


TEMPLATE_ID = "cds_colour_test_260x350"
FLAT_CASES = [
    "BLUE_PEN_FLAT_01",
    "BLUE_PEN_FLAT_02",
    "BLUE_PEN_FLAT_03",
    "BLUE_PENCIL_FLAT_01",
]


def _ink_visual(binary_mask: np.ndarray) -> np.ndarray:
    out = np.full((binary_mask.shape[0], binary_mask.shape[1], 3),
                  255, dtype=np.uint8)
    out[binary_mask > 0] = (40, 40, 40)
    return out


def _drop_small(mask: np.ndarray, min_px: int) -> np.ndarray:
    """Drop connected components smaller than ``min_px``."""
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.where(stats[:, cv2.CC_STAT_AREA] >= min_px,
                    np.uint8(255), np.uint8(0))
    keep[0] = 0
    return keep[lbl].astype(np.uint8)


# ===========================================================================
# M0 — L11 baseline
# ===========================================================================

def method_M0_L11(rect_bgr, template):
    return isolate_trace_candidates(rect_bgr, template).cleaned_bgr


# ===========================================================================
# M1 — Lab Δa* strict + L11 hard clamp (the previous-round winner)
# ===========================================================================

def method_M1_lab_strict(rect_bgr, template):
    norm = normalise_lighting(rect_bgr)
    paper_lab = estimate_paper_lab(cv2.cvtColor(norm, cv2.COLOR_BGR2LAB))
    lab = cv2.cvtColor(norm, cv2.COLOR_BGR2LAB).astype(np.int16)
    L = lab[:, :, 0]; a = lab[:, :, 1]; b = lab[:, :, 2]
    _, ap, bp = paper_lab
    mask = (
        ((a - ap) >= 6)
        & (((b - bp) - (a - ap)) <= 8)
        & (L >= 120)
    ).astype(np.uint8) * 255
    filled = paper_blended_fill(norm, mask)
    lifted = normalise_paper_to_white(filled)
    gray = cv2.cvtColor(lifted, cv2.COLOR_BGR2GRAY)
    out = lifted.copy()
    out[gray >= 160] = (255, 255, 255)
    return out


# ===========================================================================
# M2 — CMYK K extraction with Otsu adaptive thresholding
# ===========================================================================

def method_M2_cmyk_k_otsu(rect_bgr, template):
    """K = 1 - max(R,G,B).  Auto-threshold via Otsu so we don't have a
    fixed cutoff that fails when the K-histogram is skewed by warm
    paper or strong grid contrast.
    """
    norm = normalise_lighting(rect_bgr)
    bgr_f = norm.astype(np.float32) / 255.0
    K = 1.0 - bgr_f.max(axis=2)
    K_8bit = (K * 255).clip(0, 255).astype(np.uint8)
    _, ink_mask = cv2.threshold(
        K_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    ink_mask = _drop_small(ink_mask, 5)
    return _ink_visual(ink_mask)


# ===========================================================================
# M3 — Lab ink-signature (PRIMARY NEW CANDIDATE)
# ===========================================================================

def method_M3_lab_ink_signature(rect_bgr, template):
    """Identify ink DIRECTLY by its dual signature:
       (1) L significantly below paper baseline
       (2) a, b channels close to paper baseline (chromatically neutral)
    """
    norm = normalise_lighting(rect_bgr)
    lab = cv2.cvtColor(norm, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    paper_lab = estimate_paper_lab(lab.astype(np.uint8))
    Lp, ap, bp = paper_lab

    darker_than_paper = (Lp - L) > 30
    neutral_a = np.abs(a - ap) < 8
    neutral_b = np.abs(b - bp) < 8
    ink_mask = (
        darker_than_paper & neutral_a & neutral_b
    ).astype(np.uint8) * 255
    ink_mask = _drop_small(ink_mask, 5)
    return _ink_visual(ink_mask)


# ===========================================================================
# M4 — pure-RGB neutral-dark check
# ===========================================================================

def method_M4_rgb_neutral_dark(rect_bgr, template):
    """No Lab transform.  Pixel is ink iff:
       max(R,G,B) < dark_threshold  AND  (max - min) < chromaticity_threshold
    """
    norm = normalise_lighting(rect_bgr)
    bgr_i = norm.astype(np.int16)
    R, G, B = bgr_i[:, :, 2], bgr_i[:, :, 1], bgr_i[:, :, 0]
    max_c = np.maximum(np.maximum(R, G), B)
    min_c = np.minimum(np.minimum(R, G), B)
    sat = max_c - min_c

    # Tier 1: definite ink — very dark (pen)
    strong_ink = max_c < 110
    # Tier 2: neutral dark-ish — pencil
    weak_neutral = (max_c < 200) & (sat < 20)

    ink_mask = (
        (strong_ink | weak_neutral)
    ).astype(np.uint8) * 255
    ink_mask = _drop_small(ink_mask, 5)
    return _ink_visual(ink_mask)


# ===========================================================================
# M5 — M3 + bg-subtract sanity check
# ===========================================================================

def method_M5_lab_signature_plus_bg(rect_bgr, template):
    """M3 + require the pixel to be locally darker than its surroundings.

    Excludes false-positive ink in broad uniformly-dark regions.
    Insurance against creases / shadows (not relevant for these flat
    cases but here for completeness).
    """
    norm = normalise_lighting(rect_bgr)
    lab = cv2.cvtColor(norm, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    paper_lab = estimate_paper_lab(lab.astype(np.uint8))
    Lp, ap, bp = paper_lab

    # M3 signature
    sig = ((Lp - L) > 30) & (np.abs(a - ap) < 8) & (np.abs(b - bp) < 8)

    # Local-darkness sanity check
    gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY).astype(np.float32)
    local_bg = cv2.GaussianBlur(gray, (0, 0), 30.0)
    locally_dark = (local_bg - gray) > 12

    ink_mask = (sig & locally_dark).astype(np.uint8) * 255
    ink_mask = _drop_small(ink_mask, 5)
    return _ink_visual(ink_mask)


# ===========================================================================
# M6 — Lab decorrelation along paper-to-orange axis
# ===========================================================================

def method_M6_lab_decorrelation(rect_bgr, template):
    norm = normalise_lighting(rect_bgr)
    lab = cv2.cvtColor(norm, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    paper_lab = estimate_paper_lab(lab)
    _, ap, bp = paper_lab
    da = a.astype(np.float32) - ap
    db = b.astype(np.float32) - bp
    proj = (da + db) / np.sqrt(2.0)
    orange_pixels = proj > 6
    a[orange_pixels] = 128
    b[orange_pixels] = 128
    desat = cv2.merge([L, a, b])
    out_bgr = cv2.cvtColor(desat, cv2.COLOR_LAB2BGR)
    out_bgr = normalise_paper_to_white(out_bgr)
    gray = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2GRAY)
    out_bgr[gray >= 160] = (255, 255, 255)
    return out_bgr


# ===========================================================================
# M7 — Sauvola adaptive binarisation (reference)
# ===========================================================================

def method_M7_sauvola(rect_bgr, template):
    norm = normalise_lighting(rect_bgr)
    gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)
    thresh = threshold_sauvola(gray, window_size=51, k=0.2)
    binary = (gray < thresh).astype(np.uint8) * 255
    binary = _drop_small(binary, 5)
    return _ink_visual(binary)


METHODS: list[tuple[str, Callable]] = [
    ("M0_L11_baseline",                 method_M0_L11),
    ("M1_lab_strict_clamp",             method_M1_lab_strict),
    ("M2_cmyk_k_otsu",                  method_M2_cmyk_k_otsu),
    ("M3_lab_ink_signature",            method_M3_lab_ink_signature),
    ("M4_rgb_neutral_dark",             method_M4_rgb_neutral_dark),
    ("M5_lab_signature_plus_bg",        method_M5_lab_signature_plus_bg),
    ("M6_lab_decorrelation",            method_M6_lab_decorrelation),
    ("M7_sauvola",                      method_M7_sauvola),
]


# ===========================================================================
# Runner
# ===========================================================================

def _per_case_compare(rectified, per_method_cleaned, case_stem):
    tiles = [label_image(rectified, "rectified", height_px=48, scale=0.85)]
    for name, cleaned in per_method_cleaned.items():
        tiles.append(label_image(cleaned, name, height_px=48, scale=0.85))
    half = (len(tiles) + 1) // 2
    row_a, row_b = tiles[:half], tiles[half:]
    h_max = max(t.shape[0] for t in tiles)

    def _pad_h(img, h):
        if img.shape[0] == h:
            return img
        pad = np.full((h - img.shape[0], img.shape[1], 3),
                      240, dtype=np.uint8)
        return np.vstack([img, pad])

    row_a_img = np.hstack([_pad_h(t, h_max) for t in row_a])
    row_b_img = np.hstack([_pad_h(t, h_max) for t in row_b])
    w_max = max(row_a_img.shape[1], row_b_img.shape[1])

    def _pad_w(img, w):
        if img.shape[1] == w:
            return img
        pad = np.full((img.shape[0], w - img.shape[1], 3),
                      240, dtype=np.uint8)
        return np.hstack([img, pad])

    grid = np.vstack([_pad_w(row_a_img, w_max), _pad_w(row_b_img, w_max)])
    header = np.full((52, grid.shape[1], 3), 25, dtype=np.uint8)
    cv2.putText(header, case_stem, (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([header, grid])


def main():
    run_dir = make_run_dir("step02_grid_removal", "flat_methods_v3")
    print(f"Run dir: {run_dir.relative_to(REPO)}\n")
    template = load_template_config(TEMPLATE_ID)
    per_case_dir = run_dir / "per_case_compare"
    per_case_dir.mkdir(exist_ok=True)
    per_method_summary_tiles: dict[str, list[np.ndarray]] = {
        name: [] for name, _ in METHODS
    }

    print(f"{'case':<24s}  " + "  ".join(
        f"{name[:18]:>18s}" for name, _ in METHODS))
    print("-" * (24 + 4 + len(METHODS) * 20))

    for stem in FLAT_CASES:
        case_path = next(RAW_IMAGES_DIR.glob(f"{stem}.*"), None)
        if case_path is None:
            print(f"{stem}: NOT FOUND")
            continue
        img = cv2.imread(str(case_path))
        try:
            rect_result = rectify_with_markers_fast_v4(
                img, paper_w_mm=PAPER_W_MM, paper_h_mm=PAPER_H_MM,
                px_per_mm=DEFAULT_PX_PER_MM,
            )
            rectified = rect_result.warped
        except Exception as exc:
            print(f"{stem}: Phase 1 failed: {exc}")
            continue

        per_method_cleaned: dict[str, np.ndarray] = {}
        timings: dict[str, float] = {}
        for name, fn in METHODS:
            method_dir = run_dir / name / stem
            method_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            try:
                cleaned = fn(rectified, template)
                timings[name] = time.time() - t0
                per_method_cleaned[name] = cleaned
                cv2.imwrite(str(method_dir / "cleaned.png"), cleaned)
                per_method_summary_tiles[name].append(
                    label_image(cleaned, stem, height_px=48, scale=0.85)
                )
            except Exception as exc:
                timings[name] = float("nan")
                per_method_cleaned[name] = np.full(
                    rectified.shape, 200, dtype=np.uint8,
                )
                (method_dir / "info.json").write_text(json.dumps({
                    "method": name, "stem": stem,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }, indent=2), encoding="utf-8")

        tile = _per_case_compare(rectified, per_method_cleaned, stem)
        cv2.imwrite(str(per_case_dir / f"{stem}.png"), tile)

        line = f"{stem:<24s}  "
        line += "  ".join(
            f"{timings[name]:>16.2f}s" if not np.isnan(timings[name])
            else f"{'ERR':>18s}"
            for name, _ in METHODS
        )
        print(line)

    for name, tiles in per_method_summary_tiles.items():
        if not tiles:
            continue
        summary = grid_montage(tiles, cols=2, tile_max_dim=900)
        cv2.imwrite(str(run_dir / f"summary_{name}.png"), summary)

    print(f"\nFolder: {run_dir}")


if __name__ == "__main__":
    main()
