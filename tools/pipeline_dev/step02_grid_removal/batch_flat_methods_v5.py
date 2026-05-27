"""Step 02 — v5 batch: ABSOLUTE Lab neutrality (no CLAHE, no relative baseline).

Diagnosis from the v4 run + debug_m3v2.py:

    1) CLAHE shifts the paper baseline so dramatically that 19% of all
       pixels pass the "darker than paper" check.  Skip CLAHE -- work
       on the rectified image directly.

    2) Paper baseline a/b channels drift under colored lighting
       (blue tint -> b=100, warm sun -> b=140).  Using a paper-relative
       neutral check then mis-classifies grid as neutral whenever the
       grid happens to share the paper's drifted chromaticity.
       Solution: use ABSOLUTE Lab neutrality (|a-128|<5 AND |b-128|<5)
       because pure black/grey ink absorbs light uniformly and is
       captured at neutral chromaticity regardless of lighting cast.

This is the cleanest formulation we've tried.  Tested below.

Methods
=======

    M0_L11_baseline
        Production reference.

    M3v4_abs_neutral
        ** new primary candidate **
        - No CLAHE
        - Lp from top 5% L of full image (paper baseline)
        - INK = (Lp - L) > 45  AND  |a-128| < 5  AND  |b-128| < 5
        - 3x3 morph open + drop sub-30 px components

    M3v5_abs_neutral_open5
        M3v4 with a slightly bigger 5x5 morph open to kill more JPEG
        noise.  Loses some thin-pencil detail but cleaner output.

    M3v6_paperblend_then_M3v4
        Apply M3v4 to detect ink, then paper-blend-fill EVERYTHING not
        detected as ink (i.e., wipe non-ink to white).  Visual output:
        clean black ink on pure white.

    M3v7_lab_AND_K
        Belt-and-braces: pixel is ink iff M3v4's criteria pass AND its
        CMYK-K value is high enough.  Should be even cleaner.

    python tools/pipeline_dev/step02_grid_removal/batch_flat_methods_v5.py
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
from armourcore_cds.phase2.trace_isolation import isolate_trace_candidates
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


def estimate_paper_L_only(lab_uint8: np.ndarray) -> float:
    """Return the L channel median of the top 5% brightest pixels.

    We DON'T need a/b baseline anymore -- absolute neutrality replaces
    the relative-baseline approach.
    """
    L = lab_uint8[:, :, 0]
    bright_thresh = np.percentile(L, 95)
    chosen = L[L >= bright_thresh]
    return float(np.median(chosen))


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
# M0 — production baseline
# ===========================================================================

def method_M0_L11(rect_bgr, template):
    return isolate_trace_candidates(rect_bgr, template).cleaned_bgr


# ===========================================================================
# M3v4 — no-CLAHE, absolute Lab neutrality (PRIMARY)
# ===========================================================================

def method_M3v4_abs_neutral(rect_bgr, template):
    """No CLAHE.  Ink iff dark AND absolutely Lab-neutral."""
    lab = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    Lp = estimate_paper_L_only(lab.astype(np.uint8))

    darker = (Lp - L) > 45
    neutral = (np.abs(a - 128) < 5) & (np.abs(b - 128) < 5)
    ink_mask = (darker & neutral).astype(np.uint8) * 255

    ink_mask = cv2.morphologyEx(
        ink_mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    ink_mask = _drop_small(ink_mask, 30)
    return _ink_visual(ink_mask)


# ===========================================================================
# M3v5 — same but with bigger morph open
# ===========================================================================

def method_M3v5_abs_neutral_open5(rect_bgr, template):
    lab = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]; a = lab[:, :, 1]; b = lab[:, :, 2]
    Lp = estimate_paper_L_only(lab.astype(np.uint8))
    darker = (Lp - L) > 45
    neutral = (np.abs(a - 128) < 5) & (np.abs(b - 128) < 5)
    ink_mask = (darker & neutral).astype(np.uint8) * 255
    ink_mask = cv2.morphologyEx(
        ink_mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    )
    ink_mask = _drop_small(ink_mask, 50)
    return _ink_visual(ink_mask)


# ===========================================================================
# M3v6 — wider neutral threshold (catches pencil with slight chromatic noise)
# ===========================================================================

def method_M3v6_abs_neutral_wider(rect_bgr, template):
    """Same as M3v4 but neutral threshold is < 8 instead of < 5.

    Pencil graphite can have very mild warm tint from JPEG, so widening
    might recover lost pencil pixels without re-admitting grid.
    """
    lab = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]; a = lab[:, :, 1]; b = lab[:, :, 2]
    Lp = estimate_paper_L_only(lab.astype(np.uint8))
    darker = (Lp - L) > 45
    neutral = (np.abs(a - 128) < 8) & (np.abs(b - 128) < 8)
    ink_mask = (darker & neutral).astype(np.uint8) * 255
    ink_mask = cv2.morphologyEx(
        ink_mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    ink_mask = _drop_small(ink_mask, 30)
    return _ink_visual(ink_mask)


# ===========================================================================
# M3v7 — M3v4 + CMYK-K confirmation
# ===========================================================================

def method_M3v7_lab_AND_K(rect_bgr, template):
    """Pixel is ink iff M3v4 passes AND CMYK K > threshold."""
    bgr_f = rect_bgr.astype(np.float32) / 255.0
    K = 1.0 - bgr_f.max(axis=2)
    K_pass = K > 0.30                              # K >= ~76/255

    lab = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]; a = lab[:, :, 1]; b = lab[:, :, 2]
    Lp = estimate_paper_L_only(lab.astype(np.uint8))
    darker = (Lp - L) > 45
    neutral = (np.abs(a - 128) < 5) & (np.abs(b - 128) < 5)

    ink_mask = (darker & neutral & K_pass).astype(np.uint8) * 255
    ink_mask = cv2.morphologyEx(
        ink_mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    ink_mask = _drop_small(ink_mask, 30)
    return _ink_visual(ink_mask)


METHODS: list[tuple[str, Callable]] = [
    ("M0_L11_baseline",                method_M0_L11),
    ("M3v4_abs_neutral",               method_M3v4_abs_neutral),
    ("M3v5_abs_neutral_open5",         method_M3v5_abs_neutral_open5),
    ("M3v6_abs_neutral_wider",         method_M3v6_abs_neutral_wider),
    ("M3v7_lab_AND_K",                 method_M3v7_lab_AND_K),
]


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
                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([header, row])


def main():
    run_dir = make_run_dir("step02_grid_removal", "flat_methods_v5")
    print(f"Run dir: {run_dir.relative_to(REPO)}\n")
    template = load_template_config(TEMPLATE_ID)
    per_case_dir = run_dir / "per_case_compare"
    per_case_dir.mkdir(exist_ok=True)
    per_method_summary_tiles: dict[str, list[np.ndarray]] = {
        name: [] for name, _ in METHODS
    }

    print(f"{'case':<24s}  " + "  ".join(
        f"{name[:22]:>22s}" for name, _ in METHODS))
    print("-" * (24 + 4 + len(METHODS) * 24))

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
            f"{timings[name]:>20.2f}s" if not np.isnan(timings[name])
            else f"{'ERR':>22s}"
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
