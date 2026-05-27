"""Step 02 — v6 batch: NO CLAHE + paper-relative Lab neutrality.

Critical discovery: CLAHE was the root cause of every previous failure.
It boosted the L channel's local contrast so dramatically that the
"darker than paper" check matched ~19% of all pixels (vs ~0.6% in the
raw image).  Stripping CLAHE collapses the ink mask from 1.18M pixels
to 52k pixels — a 24x reduction.

Also confirmed: pen ink IS shifted by the camera AWB under coloured
lighting, so absolute Lab neutrality fails.  We need paper-RELATIVE
neutrality (|a - ap| < eps, |b - bp| < eps) where ap, bp are
sampled from actual paper pixels.

Methods
=======

    M0_L11_baseline
        Production reference.

    M_final_no_clahe
        ** main candidate **
        - NO CLAHE preprocessing
        - Paper Lab baseline from top-5% brightest pixels of whole image
        - Ink iff (Lp - L > 45) AND (|a - ap| < 5) AND (|b - bp| < 5)
        - 3x3 morph open + drop sub-30 px components

    M_final_neutral_a_strict
        Same as M_final but only requires |a - ap| < 5 (no b check).
        Pencil graphite has slight b drift under some lighting; this
        might rescue pencil cases.

    M_final_widerb
        Same as M_final but |b - bp| < 8 (wider).

    M_final_chroma_dist
        Same darkness check, but neutrality uses Euclidean chroma:
        sqrt((a-ap)^2 + (b-bp)^2) < 7

    python tools/pipeline_dev/step02_grid_removal/batch_flat_methods_v6.py
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
    """Median Lab of the top 5% brightest pixels in the whole image."""
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


# ===========================================================================
# M0 — production
# ===========================================================================

def method_M0_L11(rect_bgr, template):
    return isolate_trace_candidates(rect_bgr, template).cleaned_bgr


# ===========================================================================
# M_final family — NO CLAHE, paper-relative neutrality
# ===========================================================================

def method_M_final_no_clahe(rect_bgr, template):
    """The main candidate."""
    lab = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2LAB)
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
    return _ink_visual(ink_mask)


def method_M_final_neutral_a_only(rect_bgr, template):
    """Drop the b-channel check; pencil sometimes has b drift."""
    lab = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2LAB)
    lab_f = lab.astype(np.float32)
    L = lab_f[:, :, 0]; a = lab_f[:, :, 1]
    paper_lab = estimate_paper_lab_robust(lab)
    Lp, ap, _ = paper_lab
    darker = (Lp - L) > 45
    neutral_a = np.abs(a - ap) < 5
    ink_mask = (darker & neutral_a).astype(np.uint8) * 255
    ink_mask = cv2.morphologyEx(
        ink_mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    ink_mask = _drop_small(ink_mask, 30)
    return _ink_visual(ink_mask)


def method_M_final_widerb(rect_bgr, template):
    """Wider b-channel tolerance (8 instead of 5)."""
    lab = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2LAB)
    lab_f = lab.astype(np.float32)
    L = lab_f[:, :, 0]; a = lab_f[:, :, 1]; b = lab_f[:, :, 2]
    paper_lab = estimate_paper_lab_robust(lab)
    Lp, ap, bp = paper_lab
    darker = (Lp - L) > 45
    neutral_a = np.abs(a - ap) < 5
    neutral_b = np.abs(b - bp) < 8
    ink_mask = (darker & neutral_a & neutral_b).astype(np.uint8) * 255
    ink_mask = cv2.morphologyEx(
        ink_mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    ink_mask = _drop_small(ink_mask, 30)
    return _ink_visual(ink_mask)


def method_M_final_chroma_dist(rect_bgr, template):
    """Euclidean chromaticity distance from paper baseline."""
    lab = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2LAB)
    lab_f = lab.astype(np.float32)
    L = lab_f[:, :, 0]; a = lab_f[:, :, 1]; b = lab_f[:, :, 2]
    paper_lab = estimate_paper_lab_robust(lab)
    Lp, ap, bp = paper_lab
    darker = (Lp - L) > 45
    chroma_dist = np.sqrt((a - ap) ** 2 + (b - bp) ** 2)
    neutral = chroma_dist < 7
    ink_mask = (darker & neutral).astype(np.uint8) * 255
    ink_mask = cv2.morphologyEx(
        ink_mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    ink_mask = _drop_small(ink_mask, 30)
    return _ink_visual(ink_mask)


METHODS: list[tuple[str, Callable]] = [
    ("M0_L11_baseline",           method_M0_L11),
    ("M_final_no_clahe",          method_M_final_no_clahe),
    ("M_final_neutral_a_only",    method_M_final_neutral_a_only),
    ("M_final_widerb",            method_M_final_widerb),
    ("M_final_chroma_dist",       method_M_final_chroma_dist),
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
    run_dir = make_run_dir("step02_grid_removal", "flat_methods_v6")
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
