"""Step 02 - v7 batch: wider b-channel tolerance + score-based methods.

v6 finding
==========
Even with CLAHE removed and paper-relative neutrality, the strict |b-bp|<5
clamp rejected pen ink under blue-tinted lighting because the camera's AWB
shifts ink and paper b channels by DIFFERENT amounts.  Paper bp ~= 97 but
pen-ink b ~= 115, so |b-bp| ~= 18 - well outside the <5 window.

v7 strategy
===========
Open up the b tolerance (or drop b entirely) while keeping a tight to reject
the orange grid (the grid's signature is large +a shift from paper).  Also
introduce a continuous "ink-likeness score" that lets darkness compensate
for moderate chromatic drift.

Methods
-------
    M0_L11_baseline             production reference
    M_widerb20                  |a-ap|<5 AND |b-bp|<20
    M_only_a_strict             |a-ap|<5  (no b check at all)
    M_chroma_dist20             sqrt(da^2+db^2) < 20
    M_chroma_dist25             sqrt(da^2+db^2) < 25
    M_score_dark_minus_chroma   (Lp-L) - 2*chroma_dist > 30
    M_score_v2                  (Lp-L) - 1.5*chroma_dist > 35
    M_grid_reject_a             darker AND (a-ap) < 8     (only reject strong +a, keep neutral and cool tints)

Run:
    python tools/pipeline_dev/step02_grid_removal/batch_flat_methods_v7.py
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


def estimate_paper_lab_robust(lab_uint8: np.ndarray) -> tuple:
    L = lab_uint8[:, :, 0]
    bright_thresh = np.percentile(L, 95)
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


def _finalise(mask: np.ndarray) -> np.ndarray:
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    mask = _drop_small(mask, 30)
    return _ink_visual(mask)


def _lab_no_clahe(rect_bgr):
    lab = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2LAB)
    lab_f = lab.astype(np.float32)
    Lp, ap, bp = estimate_paper_lab_robust(lab)
    return lab_f[:, :, 0], lab_f[:, :, 1], lab_f[:, :, 2], Lp, ap, bp


# ===========================================================================
# Methods
# ===========================================================================

def method_M0_L11(rect_bgr, template):
    return isolate_trace_candidates(rect_bgr, template).cleaned_bgr


def method_M_widerb20(rect_bgr, template):
    L, a, b, Lp, ap, bp = _lab_no_clahe(rect_bgr)
    darker = (Lp - L) > 45
    neutral_a = np.abs(a - ap) < 5
    neutral_b = np.abs(b - bp) < 20
    mask = (darker & neutral_a & neutral_b).astype(np.uint8) * 255
    return _finalise(mask)


def method_M_only_a_strict(rect_bgr, template):
    L, a, b, Lp, ap, bp = _lab_no_clahe(rect_bgr)
    darker = (Lp - L) > 45
    neutral_a = np.abs(a - ap) < 5
    mask = (darker & neutral_a).astype(np.uint8) * 255
    return _finalise(mask)


def method_M_chroma_dist20(rect_bgr, template):
    L, a, b, Lp, ap, bp = _lab_no_clahe(rect_bgr)
    darker = (Lp - L) > 45
    chroma = np.sqrt((a - ap) ** 2 + (b - bp) ** 2)
    mask = (darker & (chroma < 20)).astype(np.uint8) * 255
    return _finalise(mask)


def method_M_chroma_dist25(rect_bgr, template):
    L, a, b, Lp, ap, bp = _lab_no_clahe(rect_bgr)
    darker = (Lp - L) > 45
    chroma = np.sqrt((a - ap) ** 2 + (b - bp) ** 2)
    mask = (darker & (chroma < 25)).astype(np.uint8) * 255
    return _finalise(mask)


def method_M_score_dark_minus_chroma(rect_bgr, template):
    L, a, b, Lp, ap, bp = _lab_no_clahe(rect_bgr)
    chroma = np.sqrt((a - ap) ** 2 + (b - bp) ** 2)
    score = (Lp - L) - 2.0 * chroma
    mask = (score > 30).astype(np.uint8) * 255
    return _finalise(mask)


def method_M_score_v2(rect_bgr, template):
    L, a, b, Lp, ap, bp = _lab_no_clahe(rect_bgr)
    chroma = np.sqrt((a - ap) ** 2 + (b - bp) ** 2)
    score = (Lp - L) - 1.5 * chroma
    mask = (score > 35).astype(np.uint8) * 255
    return _finalise(mask)


def method_M_grid_reject_a(rect_bgr, template):
    """Only reject strong +a (orange grid).  Keep everything else dark."""
    L, a, b, Lp, ap, bp = _lab_no_clahe(rect_bgr)
    darker = (Lp - L) > 45
    not_orange = (a - ap) < 8
    mask = (darker & not_orange).astype(np.uint8) * 255
    return _finalise(mask)


METHODS: list[tuple[str, Callable]] = [
    ("M0_L11_baseline",          method_M0_L11),
    ("M_widerb20",               method_M_widerb20),
    ("M_only_a_strict",          method_M_only_a_strict),
    ("M_chroma_dist20",          method_M_chroma_dist20),
    ("M_chroma_dist25",          method_M_chroma_dist25),
    ("M_score_dark_minus_chroma",method_M_score_dark_minus_chroma),
    ("M_score_v2",               method_M_score_v2),
    ("M_grid_reject_a",          method_M_grid_reject_a),
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
    run_dir = make_run_dir("step02_grid_removal", "flat_methods_v7")
    print(f"Run dir: {run_dir.relative_to(REPO)}\n")
    template = load_template_config(TEMPLATE_ID)
    per_case_dir = run_dir / "per_case_compare"
    per_case_dir.mkdir(exist_ok=True)
    per_method_summary_tiles: dict[str, list[np.ndarray]] = {
        name: [] for name, _ in METHODS
    }

    print(f"{'case':<24s}  " + "  ".join(
        f"{name[:20]:>20s}" for name, _ in METHODS))
    print("-" * (24 + 4 + len(METHODS) * 22))

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
            f"{timings[name]:>18.2f}s" if not np.isnan(timings[name])
            else f"{'ERR':>20s}"
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
