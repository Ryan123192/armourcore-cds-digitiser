"""Step 02 - v10 batch: v9 brightness-gated method + pencil-aware trace mask.

v9 won on 3/4 images.  Only failure: PENCIL_FLAT_01 loses faint pencil
because build_trace_mask_simple's gray<170 absolute threshold misses
light graphite.  Fix: bring back the relative-darkness branch from the
baseline trace_mask, but require larger min_component_px so the
"dotty noise" failure from earlier v8 work doesn't return.

Methods
=======
    M0_L11_baseline                  reference
    M9_bright_gated_geom             v9 winner (kept for direct compare)
    M_pencil_aware                   v9 orange logic + abs + rel trace mask
    M_pencil_aware_stricter_noise    same but min_component_px=30
    M_pencil_aware_relaxed_dark      abs threshold lifted to 200
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
# Shared blocks (carried over from v9)
# ===========================================================================

def paper_wb(image_bgr, percentile=95.0):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    thr = np.percentile(gray, percentile)
    mask = gray >= thr
    paper = image_bgr[mask].astype(np.float32)
    means = paper.mean(axis=0)
    target = float(means.mean())
    scale = target / np.maximum(means, 1.0)
    return np.clip(image_bgr.astype(np.float32) * scale[None, None, :],
                  0, 255).astype(np.uint8)


def global_paper_lift(image_bgr, target=245):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    paper_val = float(np.percentile(gray, 95))
    if paper_val < 1.0:
        return image_bgr
    scale = min(float(target) / paper_val, 1.6)
    return np.clip(image_bgr.astype(np.float32) * scale, 0, 255).astype(np.uint8)


def paper_blended_fill(image_bgr, removal_mask, bg_sigma=40.0):
    keep = (removal_mask == 0).astype(np.float32)
    masked = image_bgr.astype(np.float32) * keep[..., None]
    img_blur = cv2.GaussianBlur(masked, (0, 0), bg_sigma)
    w_blur = cv2.GaussianBlur(keep, (0, 0), bg_sigma)
    bg = img_blur / np.maximum(w_blur[..., None], 1e-3)
    bg = np.clip(bg, 0, 255).astype(np.uint8)
    out = image_bgr.copy()
    out[removal_mask > 0] = bg[removal_mask > 0]
    return out


def bright_gated_orange_mask(image_bgr, a_threshold=2.0,
                            brightness_floor=170, close_px=2,
                            min_component=12):
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


def project_grid_from_peaks(orange_mask, band_half_px=3):
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


def trace_mask_pencil_aware(cleaned_bgr, dark_threshold=180,
                           rel_dark_sigma=25.0, rel_dark_min=15,
                           min_component_px=20):
    """Absolute dark OR locally-much-darker-than-bg.

    rel_dark_min raised from baseline 10 -> 15 to suppress paper-texture
    noise.  min_component_px raised from 8 -> 20 to drop dotty specks.
    """
    gray = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
    abs_dark = (gray < dark_threshold).astype(np.uint8) * 255
    local_bg = cv2.GaussianBlur(gray, (0, 0),
                               sigmaX=rel_dark_sigma, sigmaY=rel_dark_sigma)
    rel_dark = cv2.subtract(local_bg, gray)
    rel_mask = (rel_dark >= rel_dark_min).astype(np.uint8) * 255
    combined = cv2.bitwise_or(abs_dark, rel_mask)
    if min_component_px > 0:
        combined = _remove_small_components(combined, min_area_px=min_component_px)
    return combined


def trace_to_visual(trace_mask):
    out = np.full((trace_mask.shape[0], trace_mask.shape[1], 3),
                  255, dtype=np.uint8)
    out[trace_mask > 0] = (40, 40, 40)
    return out


# ===========================================================================
# Methods
# ===========================================================================

def method_M0_L11(rect_bgr, template):
    return isolate_trace_candidates(rect_bgr, template).cleaned_bgr


def method_M9_bright_gated_geom(rect_bgr, template):
    wb = paper_wb(rect_bgr)
    orange = bright_gated_orange_mask(wb, 2.0, 170)
    bands = project_grid_from_peaks(orange, 3)
    gray = cv2.cvtColor(wb, cv2.COLOR_BGR2GRAY)
    band_kill = (bands > 0) & (gray > 170)
    full_mask = ((orange > 0) | band_kill).astype(np.uint8) * 255
    cleaned = paper_blended_fill(wb, full_mask)
    cleaned = global_paper_lift(cleaned)
    gray2 = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    trace = (gray2 < 170).astype(np.uint8) * 255
    trace = _remove_small_components(trace, 12)
    return trace_to_visual(trace)


def _bright_gated_pipeline(rect_bgr, trace_fn):
    wb = paper_wb(rect_bgr)
    orange = bright_gated_orange_mask(wb, 2.0, 170)
    bands = project_grid_from_peaks(orange, 3)
    gray = cv2.cvtColor(wb, cv2.COLOR_BGR2GRAY)
    band_kill = (bands > 0) & (gray > 170)
    full_mask = ((orange > 0) | band_kill).astype(np.uint8) * 255
    cleaned = paper_blended_fill(wb, full_mask)
    cleaned = global_paper_lift(cleaned)
    return trace_to_visual(trace_fn(cleaned))


def method_M_pencil_aware(rect_bgr, template):
    """Baseline-style abs + rel trace mask, with stricter noise."""
    return _bright_gated_pipeline(
        rect_bgr,
        lambda c: trace_mask_pencil_aware(
            c, dark_threshold=180, rel_dark_min=15, min_component_px=20),
    )


def method_M_pencil_aware_stricter_noise(rect_bgr, template):
    return _bright_gated_pipeline(
        rect_bgr,
        lambda c: trace_mask_pencil_aware(
            c, dark_threshold=180, rel_dark_min=18, min_component_px=30),
    )


def method_M_pencil_aware_relaxed_dark(rect_bgr, template):
    """Lift abs threshold to 200 so very faint pencil registers."""
    return _bright_gated_pipeline(
        rect_bgr,
        lambda c: trace_mask_pencil_aware(
            c, dark_threshold=200, rel_dark_min=15, min_component_px=25),
    )


def method_M_pencil_aware_widebg(rect_bgr, template):
    """Wider Gaussian for local-bg estimate — catches broader pencil strokes."""
    return _bright_gated_pipeline(
        rect_bgr,
        lambda c: trace_mask_pencil_aware(
            c, dark_threshold=185, rel_dark_sigma=40.0,
            rel_dark_min=12, min_component_px=25),
    )


METHODS: list[tuple[str, Callable]] = [
    ("M0_L11_baseline",                method_M0_L11),
    ("M9_bright_gated_geom",           method_M9_bright_gated_geom),
    ("M_pencil_aware",                 method_M_pencil_aware),
    ("M_pencil_aware_stricter_noise",  method_M_pencil_aware_stricter_noise),
    ("M_pencil_aware_relaxed_dark",    method_M_pencil_aware_relaxed_dark),
    ("M_pencil_aware_widebg",          method_M_pencil_aware_widebg),
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
    run_dir = make_run_dir("step02_grid_removal", "flat_methods_v10")
    print(f"Run dir: {run_dir.relative_to(REPO)}\n")
    template = load_template_config(TEMPLATE_ID)
    per_case_dir = run_dir / "per_case_compare"
    per_case_dir.mkdir(exist_ok=True)
    per_method_summary_tiles: dict[str, list[np.ndarray]] = {
        n: [] for n, _ in METHODS
    }

    print(f"{'case':<24s}  " + "  ".join(
        f"{n[:26]:>26s}" for n, _ in METHODS))
    print("-" * (24 + 4 + len(METHODS) * 28))

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
            f"{timings[n]:>24.2f}s" if not np.isnan(timings[n])
            else f"{'ERR':>26s}"
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
