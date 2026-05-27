"""Step 02 — v4 batch: fix the M3 paper-baseline bias.

The v3 M3 Lab-ink-signature was clean in concept but the paper Lab
baseline got biased toward orange whenever the central 10% patch had
heavy grid coverage.  Once the baseline drifts toward orange, real
orange grid pixels look "neutral" relative to that biased baseline
and slip through the ink criteria.

v4 makes three fixes
====================

1) BETTER PAPER BASELINE.  Sample paper from the WHOLE image, not just
   the central patch.  Take the top 5% brightest pixels (paper is the
   brightest large region regardless of where it sits) and use the
   median Lab of those.  Robust to grid-heavy central crops.

2) TIGHTER THRESHOLDS.
       darker_than_paper:  30 -> 45 grey
       neutral_a, neutral_b:  8 -> 5 grey

3) NOISE CLEANUP.
       3x3 morphological OPEN (removes single-pixel JPEG specks)
       drop components < 30 px

Plus one new alternative for comparison:

    K-means 3-class clustering on Lab.  Cluster the image into
    paper / grid / ink.  Keep only the ink cluster.

Methods
=======

    M0_L11_baseline             — production reference
    M1_lab_strict_clamp         — previous round's winner (Lab + L11 clamp)
    M3v2_ink_signature_tight    — the fixed M3 ink-signature
    M3v3_ink_sig_dilated        — M3v2 + slight dilation to thicken thin ink
    M_kmeans_3class             — K-means 3-cluster on Lab

    python tools/pipeline_dev/step02_grid_removal/batch_flat_methods_v4.py
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

from armourcore_cds.phase1.marker_rectify_fast_v4 import (
    rectify_with_markers_fast_v4,
    PAPER_W_MM, PAPER_H_MM, DEFAULT_PX_PER_MM,
)
from armourcore_cds.phase2.trace_isolation import (
    isolate_trace_candidates, normalise_lighting,
    paper_blended_fill, normalise_paper_to_white,
)
from armourcore_cds.phase2.trace_isolation_v2 import (
    estimate_paper_lab as _legacy_paper_lab,  # the central-patch one
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


# ===========================================================================
# Fix 1 — robust paper Lab baseline
# ===========================================================================

def estimate_paper_lab_robust(lab_uint8: np.ndarray) -> tuple:
    """Median Lab of the TOP 5% brightest pixels in the WHOLE image.

    Paper is always the brightest large region.  By sampling globally
    and taking only the brightest 5%, we avoid the central-patch bias
    that drifted the baseline toward orange when grid lines sat in the
    centre.
    """
    L = lab_uint8[:, :, 0]
    bright_thresh = np.percentile(L, 95)         # top 5% brightest
    mask = L >= bright_thresh
    chosen = lab_uint8[mask]
    return tuple(float(v) for v in np.median(chosen, axis=0))


def _ink_visual(binary_mask: np.ndarray) -> np.ndarray:
    out = np.full((binary_mask.shape[0], binary_mask.shape[1], 3),
                  255, dtype=np.uint8)
    out[binary_mask > 0] = (40, 40, 40)
    return out


def _drop_small(mask: np.ndarray, min_px: int) -> np.ndarray:
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.where(stats[:, cv2.CC_STAT_AREA] >= min_px,
                    np.uint8(255), np.uint8(0))
    keep[0] = 0
    return keep[lbl].astype(np.uint8)


# ===========================================================================
# M0 / M1 reference methods (unchanged)
# ===========================================================================

def method_M0_L11(rect_bgr, template):
    return isolate_trace_candidates(rect_bgr, template).cleaned_bgr


def method_M1_lab_strict(rect_bgr, template):
    norm = normalise_lighting(rect_bgr)
    paper_lab = _legacy_paper_lab(cv2.cvtColor(norm, cv2.COLOR_BGR2LAB))
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
# M3v2 — Lab ink-signature with all three v4 fixes
# ===========================================================================

def method_M3v2_ink_signature_tight(rect_bgr, template):
    """Lab ink-signature with robust baseline + tight thresholds + noise clean."""
    norm = normalise_lighting(rect_bgr)
    lab = cv2.cvtColor(norm, cv2.COLOR_BGR2LAB)
    lab_f = lab.astype(np.float32)
    L = lab_f[:, :, 0]
    a = lab_f[:, :, 1]
    b = lab_f[:, :, 2]

    # ---- FIX 1: robust paper baseline (whole image, top 5%) ----
    paper_lab = estimate_paper_lab_robust(lab)
    Lp, ap, bp = paper_lab

    # ---- FIX 2: tighter thresholds ----
    darker = (Lp - L) > 45                 # at least 45 grey darker
    neutral_a = np.abs(a - ap) < 5
    neutral_b = np.abs(b - bp) < 5
    ink_mask = (darker & neutral_a & neutral_b).astype(np.uint8) * 255

    # ---- FIX 3: noise cleanup ----
    ink_mask = cv2.morphologyEx(
        ink_mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    ink_mask = _drop_small(ink_mask, 30)
    return _ink_visual(ink_mask)


# ===========================================================================
# M3v3 — same as M3v2 but slightly dilated to thicken thin pencil
# ===========================================================================

def method_M3v3_ink_sig_dilated(rect_bgr, template):
    norm = normalise_lighting(rect_bgr)
    lab = cv2.cvtColor(norm, cv2.COLOR_BGR2LAB)
    lab_f = lab.astype(np.float32)
    L = lab_f[:, :, 0]
    a = lab_f[:, :, 1]
    b = lab_f[:, :, 2]

    paper_lab = estimate_paper_lab_robust(lab)
    Lp, ap, bp = paper_lab
    darker = (Lp - L) > 45
    neutral_a = np.abs(a - ap) < 5
    neutral_b = np.abs(b - bp) < 5
    ink_mask = (darker & neutral_a & neutral_b).astype(np.uint8) * 255
    ink_mask = cv2.morphologyEx(
        ink_mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    ink_mask = _drop_small(ink_mask, 30)
    # Slight dilate to recover thin pencil pixels lost in the open
    ink_mask = cv2.dilate(
        ink_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    return _ink_visual(ink_mask)


# ===========================================================================
# M_kmeans — 3-cluster K-means on Lab
# ===========================================================================

def method_M_kmeans_3class(rect_bgr, template):
    """Cluster image pixels in Lab space into 3 groups: paper, grid, ink.
    Keep only the cluster with the lowest L (ink).
    """
    norm = normalise_lighting(rect_bgr)
    # Downsample for kmeans speed
    h, w = norm.shape[:2]
    scale = 600 / max(h, w)
    small = cv2.resize(norm, (int(w * scale), int(h * scale)),
                       interpolation=cv2.INTER_AREA)
    lab_small = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    pixels = lab_small.reshape(-1, 3)
    # Initialise k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 8, 1.0)
    _, _, centers = cv2.kmeans(
        pixels, 3, None, criteria, 3, cv2.KMEANS_PP_CENTERS,
    )
    # The cluster with the lowest L is ink
    ink_centre = centers[np.argmin(centers[:, 0])]

    # Build mask on the FULL-resolution image by nearest-centroid
    lab_full = cv2.cvtColor(norm, cv2.COLOR_BGR2LAB).astype(np.float32)
    flat = lab_full.reshape(-1, 3)
    # Distance to each centre
    dists = np.linalg.norm(
        flat[:, None, :] - centers[None, :, :],
        axis=2,
    )
    labels = np.argmin(dists, axis=1)
    ink_mask_flat = (labels == np.argmin(centers[:, 0])).astype(np.uint8) * 255
    ink_mask = ink_mask_flat.reshape(lab_full.shape[:2])
    ink_mask = cv2.morphologyEx(
        ink_mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    ink_mask = _drop_small(ink_mask, 30)
    return _ink_visual(ink_mask)


METHODS: list[tuple[str, Callable]] = [
    ("M0_L11_baseline",                 method_M0_L11),
    ("M1_lab_strict_clamp",             method_M1_lab_strict),
    ("M3v2_ink_signature_tight",        method_M3v2_ink_signature_tight),
    ("M3v3_ink_sig_dilated",            method_M3v3_ink_sig_dilated),
    ("M_kmeans_3class",                 method_M_kmeans_3class),
]


# ===========================================================================
# Runner
# ===========================================================================

def _per_case_compare(rectified, per_method_cleaned, case_stem):
    tiles = [label_image(rectified, "rectified", height_px=48, scale=0.85)]
    for name, cleaned in per_method_cleaned.items():
        tiles.append(label_image(cleaned, name, height_px=48, scale=0.85))
    h_max = max(t.shape[0] for t in tiles)

    def _pad(img):
        if img.shape[0] == h_max:
            return img
        pad = np.full((h_max - img.shape[0], img.shape[1], 3),
                      240, dtype=np.uint8)
        return np.vstack([img, pad])

    row = np.hstack([_pad(t) for t in tiles])
    header = np.full((52, row.shape[1], 3), 25, dtype=np.uint8)
    cv2.putText(header, case_stem, (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([header, row])


def main():
    run_dir = make_run_dir("step02_grid_removal", "flat_methods_v4")
    print(f"Run dir: {run_dir.relative_to(REPO)}\n")
    template = load_template_config(TEMPLATE_ID)
    per_case_dir = run_dir / "per_case_compare"
    per_case_dir.mkdir(exist_ok=True)
    per_method_summary_tiles: dict[str, list[np.ndarray]] = {
        name: [] for name, _ in METHODS
    }

    print(f"{'case':<24s}  " + "  ".join(
        f"{name[:24]:>24s}" for name, _ in METHODS))
    print("-" * (24 + 4 + len(METHODS) * 26))

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
            f"{timings[name]:>22.2f}s" if not np.isnan(timings[name])
            else f"{'ERR':>24s}"
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
