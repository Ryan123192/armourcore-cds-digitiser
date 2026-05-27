"""Step 02 - v9 batch: brightness-gated chromatic + shadow-aware geometry.

What v8 taught us
=================
* `M_pwb_lab_a` on img 1: GRID gone, but pen ink ALSO eaten.  Under strong
  blue AWB cast, pen ink's a* sits 1-2 above 128 (chromatic).  A pure a*
  threshold can't separate it from light orange grid.
* `M_pwb_lab_a` on img 2: top half PERFECT (grid gone, traces clean).
  Bottom-right shadow zone keeps the orange grid because in shadow the
  chromaticity collapses and a* doesn't cross the threshold.

Two new mechanisms in v9
========================
1. **Brightness-gated chromaticity** -- kill pixels iff
       (a* - 128 > delta_a)  AND  (gray > brightness_floor)
   Orange grid (and minor grid) is BRIGHT and chromatic - both true.
   Pen ink is DARK so the gate protects it even if a* is slightly above 128.
   Paper is bright but not chromatic - first clause is false.

2. **Shadow-aware fill via template grid projection** -- after the colour
   removal, find regions where colour signal is weak (low local contrast
   AND no orange-mask coverage) and project the regular grid spacing
   inferred from the rest of the image into those zones.  Removes the
   grid in shadow without needing per-pixel orange evidence.

Methods
=======
    M0_L11_baseline             reference

    M_v8_pwb_lab_a              v8's main method, kept for comparison.

    M_bright_gated              *** primary candidate ***
        Paper-WB -> Lab a* > 130 AND gray > 170 (brightness gate)
        -> paper-blend fill -> global lift -> simple trace.
        Should protect dark pen/pencil ink even if chromatic.

    M_bright_gated_geom         brightness-gated + geometry assist
        Same as bright_gated, then union the projected grid bands
        (chromatic-OR-bright inside bands, brightness-gated outside).
        Should clean the shadow zone in img 2.

    M_bright_gated_strong       wider brightness gate
        Paper-WB -> a* > 128.5 AND gray > 150 -> very permissive
        chromatic removal but still protects darker ink.

    M_bright_gated_widegate     brightness gate floored at 200
        Only kill pixels brighter than 200 (minor grid territory).
        Tool ink is much darker so 100% safe; only very bright orange
        gets removed.  Safest, weakest grid removal.
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
    isolate_trace_candidates,
    _find_local_peaks, _infer_grid_positions, _remove_small_components,
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
# Building blocks (shared with v8)
# ===========================================================================

def paper_wb(image_bgr: np.ndarray, percentile: float = 95.0) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    thr = np.percentile(gray, percentile)
    mask = gray >= thr
    paper = image_bgr[mask].astype(np.float32)
    means = paper.mean(axis=0)
    target = float(means.mean())
    scale = target / np.maximum(means, 1.0)
    out = image_bgr.astype(np.float32) * scale[None, None, :]
    return np.clip(out, 0, 255).astype(np.uint8)


def global_paper_lift(image_bgr: np.ndarray, target: int = 245) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    paper_val = float(np.percentile(gray, 95))
    if paper_val < 1.0:
        return image_bgr
    scale = min(float(target) / paper_val, 1.6)
    out = image_bgr.astype(np.float32) * scale
    return np.clip(out, 0, 255).astype(np.uint8)


def paper_blended_fill(image_bgr: np.ndarray, removal_mask: np.ndarray,
                      bg_sigma: float = 40.0) -> np.ndarray:
    keep = (removal_mask == 0).astype(np.float32)
    masked = image_bgr.astype(np.float32) * keep[..., None]
    img_blur = cv2.GaussianBlur(masked, (0, 0), bg_sigma)
    w_blur   = cv2.GaussianBlur(keep,   (0, 0), bg_sigma)
    bg = img_blur / np.maximum(w_blur[..., None], 1e-3)
    bg = np.clip(bg, 0, 255).astype(np.uint8)
    out = image_bgr.copy()
    out[removal_mask > 0] = bg[removal_mask > 0]
    return out


def bright_gated_orange_mask(image_bgr: np.ndarray,
                             a_threshold: float = 2.0,
                             brightness_floor: int = 170,
                             close_px: int = 2,
                             min_component: int = 12) -> np.ndarray:
    """Orange iff (a*-128 > delta) AND (gray > brightness_floor).

    The brightness gate is the key: it protects dark pen / pencil ink
    even when the AWB has nudged its a* slightly above 128.  Grid
    (major and minor) is always bright enough to clear the gate.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    a = lab[..., 1].astype(np.float32)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    chromatic = (a - 128.0) > a_threshold
    bright = gray > brightness_floor
    mask = (chromatic & bright).astype(np.uint8) * 255
    if close_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_px * 2 + 1, close_px * 2 + 1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    if min_component > 0:
        mask = _remove_small_components(mask, min_area_px=min_component)
    return mask


def project_grid_from_peaks(orange_mask: np.ndarray,
                            band_half_px: int = 3) -> np.ndarray:
    H, W = orange_mask.shape
    if not np.any(orange_mask):
        return np.zeros_like(orange_mask)
    strong = orange_mask > 0
    row_d = strong.sum(axis=1) / max(1, W)
    col_d = strong.sum(axis=0) / max(1, H)
    h_peaks = _find_local_peaks(row_d, 14, 0.10)
    v_peaks = _find_local_peaks(col_d, 14, 0.10)
    h_all = _infer_grid_positions(h_peaks, H)
    v_all = _infer_grid_positions(v_peaks, W)
    bands = np.zeros((H, W), dtype=np.uint8)
    t = 2 * band_half_px + 1
    for y in h_all:
        cv2.line(bands, (0, int(y)), (W - 1, int(y)), 255, t)
    for x in v_all:
        cv2.line(bands, (int(x), 0), (int(x), H - 1), 255, t)
    return bands


def build_trace_mask_simple(cleaned_bgr: np.ndarray,
                           dark_threshold: int = 170,
                           min_component_px: int = 12) -> np.ndarray:
    gray = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray < dark_threshold).astype(np.uint8) * 255
    if min_component_px > 0:
        mask = _remove_small_components(mask, min_area_px=min_component_px)
    return mask


def trace_to_visual(trace_mask: np.ndarray) -> np.ndarray:
    out = np.full((trace_mask.shape[0], trace_mask.shape[1], 3),
                  255, dtype=np.uint8)
    out[trace_mask > 0] = (40, 40, 40)
    return out


# v8 reference
def lab_a_mask_v8(image_bgr: np.ndarray, a_threshold: float = 2.0,
                 close_px: int = 2, min_component: int = 12) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    a = lab[..., 1].astype(np.float32)
    mask = ((a - 128.0) > a_threshold).astype(np.uint8) * 255
    if close_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_px * 2 + 1, close_px * 2 + 1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    if min_component > 0:
        mask = _remove_small_components(mask, min_area_px=min_component)
    return mask


# ===========================================================================
# Methods
# ===========================================================================

def method_M0_L11(rect_bgr, template):
    return isolate_trace_candidates(rect_bgr, template).cleaned_bgr


def method_M_v8_pwb_lab_a(rect_bgr, template):
    wb = paper_wb(rect_bgr)
    orange = lab_a_mask_v8(wb, a_threshold=2.0)
    cleaned = paper_blended_fill(wb, orange)
    cleaned = global_paper_lift(cleaned)
    trace = build_trace_mask_simple(cleaned)
    return trace_to_visual(trace)


def method_M_bright_gated(rect_bgr, template):
    """Primary candidate: brightness-gated chromatic removal."""
    wb = paper_wb(rect_bgr)
    orange = bright_gated_orange_mask(
        wb, a_threshold=2.0, brightness_floor=170,
    )
    cleaned = paper_blended_fill(wb, orange)
    cleaned = global_paper_lift(cleaned)
    trace = build_trace_mask_simple(cleaned)
    return trace_to_visual(trace)


def method_M_bright_gated_geom(rect_bgr, template):
    """Brightness-gated + projected grid bands (shadow zone recovery)."""
    wb = paper_wb(rect_bgr)
    orange = bright_gated_orange_mask(
        wb, a_threshold=2.0, brightness_floor=170,
    )
    bands = project_grid_from_peaks(orange, band_half_px=3)

    # Inside bands: remove anything BRIGHT (kills minor grid in shadow zone
    # where chromaticity collapsed) but NOT dark (protects ink crossings).
    gray = cv2.cvtColor(wb, cv2.COLOR_BGR2GRAY)
    band_kill = (bands > 0) & (gray > 170)
    full_mask = ((orange > 0) | band_kill).astype(np.uint8) * 255

    cleaned = paper_blended_fill(wb, full_mask)
    cleaned = global_paper_lift(cleaned)
    trace = build_trace_mask_simple(cleaned)
    return trace_to_visual(trace)


def method_M_bright_gated_strong(rect_bgr, template):
    """Weaker chromatic threshold + lower brightness floor.
    More aggressive grid removal at the cost of slightly higher risk
    of nibbling at light pen edges."""
    wb = paper_wb(rect_bgr)
    orange = bright_gated_orange_mask(
        wb, a_threshold=0.5, brightness_floor=150,
    )
    cleaned = paper_blended_fill(wb, orange)
    cleaned = global_paper_lift(cleaned)
    trace = build_trace_mask_simple(cleaned)
    return trace_to_visual(trace)


def method_M_bright_gated_widegate(rect_bgr, template):
    """Brightness floor very high (200) - only kills very bright pixels.
    100% safe for ink, but only catches the brightest grid sections."""
    wb = paper_wb(rect_bgr)
    orange = bright_gated_orange_mask(
        wb, a_threshold=1.0, brightness_floor=200,
    )
    cleaned = paper_blended_fill(wb, orange)
    cleaned = global_paper_lift(cleaned)
    trace = build_trace_mask_simple(cleaned)
    return trace_to_visual(trace)


def method_M_bright_gated_strong_geom(rect_bgr, template):
    """Strong (aggressive) + geometry assist - best chance of cleaning
    the shadow zone in img 2 while still protecting ink."""
    wb = paper_wb(rect_bgr)
    orange = bright_gated_orange_mask(
        wb, a_threshold=0.5, brightness_floor=150,
    )
    bands = project_grid_from_peaks(orange, band_half_px=4)
    gray = cv2.cvtColor(wb, cv2.COLOR_BGR2GRAY)
    band_kill = (bands > 0) & (gray > 150)
    full_mask = ((orange > 0) | band_kill).astype(np.uint8) * 255
    cleaned = paper_blended_fill(wb, full_mask)
    cleaned = global_paper_lift(cleaned)
    trace = build_trace_mask_simple(cleaned)
    return trace_to_visual(trace)


METHODS: list[tuple[str, Callable]] = [
    ("M0_L11_baseline",             method_M0_L11),
    ("M_v8_pwb_lab_a",              method_M_v8_pwb_lab_a),
    ("M_bright_gated",              method_M_bright_gated),
    ("M_bright_gated_geom",         method_M_bright_gated_geom),
    ("M_bright_gated_strong",       method_M_bright_gated_strong),
    ("M_bright_gated_strong_geom",  method_M_bright_gated_strong_geom),
    ("M_bright_gated_widegate",     method_M_bright_gated_widegate),
]


# ===========================================================================
# Batch driver
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
                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([header, row])


def main():
    run_dir = make_run_dir("step02_grid_removal", "flat_methods_v9")
    print(f"Run dir: {run_dir.relative_to(REPO)}\n")
    template = load_template_config(TEMPLATE_ID)
    per_case_dir = run_dir / "per_case_compare"
    per_case_dir.mkdir(exist_ok=True)
    per_method_summary_tiles: dict[str, list[np.ndarray]] = {
        n: [] for n, _ in METHODS
    }

    print(f"{'case':<24s}  " + "  ".join(
        f"{n[:24]:>24s}" for n, _ in METHODS))
    print("-" * (24 + 4 + len(METHODS) * 26))

    for stem in FLAT_CASES:
        case_path = next(RAW_IMAGES_DIR.glob(f"{stem}.*"), None)
        if case_path is None:
            print(f"{stem}: NOT FOUND")
            continue
        img = cv2.imread(str(case_path))
        try:
            rect = rectify_with_markers_fast_v4(
                img, paper_w_mm=PAPER_W_MM, paper_h_mm=PAPER_H_MM,
                px_per_mm=DEFAULT_PX_PER_MM,
            ).warped
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
                cleaned = fn(rect, template)
                timings[name] = time.time() - t0
                per_method_cleaned[name] = cleaned
                cv2.imwrite(str(method_dir / "cleaned.png"), cleaned)
                per_method_summary_tiles[name].append(
                    label_image(cleaned, stem, height_px=48, scale=0.85)
                )
            except Exception as exc:
                timings[name] = float("nan")
                per_method_cleaned[name] = np.full(
                    rect.shape, 200, dtype=np.uint8)
                (method_dir / "info.json").write_text(json.dumps({
                    "method": name, "stem": stem,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }, indent=2), encoding="utf-8")

        tile = _per_case_compare(rect, per_method_cleaned, stem)
        cv2.imwrite(str(per_case_dir / f"{stem}.png"), tile)

        line = f"{stem:<24s}  "
        line += "  ".join(
            f"{timings[n]:>22.2f}s" if not np.isnan(timings[n])
            else f"{'ERR':>24s}"
            for n, _ in METHODS
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
