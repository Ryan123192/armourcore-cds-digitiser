"""Step 02 - v8 batch: paper-WB + Lab a* + geometry assist.

Findings from v6/v7 + L11 diagnostic
====================================
1. L11 baseline FAILS on image 1 (BLUE_PEN_FLAT_01) because the blue
   camera cast pushes ALL hues to ~109; orange-coverage probe = 0.00%,
   so auto-detect falls to the BLACK path and nothing gets removed.
2. A paper-WB step (estimate paper from top-5% brightest, scale each
   channel so paper is neutral gray) recovers the orange:
       img1 probe: 0.00% -> 3.07%   (now ORANGE path)
       img2/3/4: barely changed - safe to apply universally.
3. After paper-WB, paper Lab is near (255, 128, 128). Orange grid
   (major AND minor) shifts a* above 130; tool ink stays a* ~ 128.
   So a single a* threshold on the paper-WB'd image should kill BOTH
   major and minor grid - simpler than HSV + brightness clamp.
4. `normalise_paper_to_white` (local-background divide) overshoots in
   shadow zones - eats tool ink on img 2.  Use a GLOBAL paper-target
   lift instead.
5. The relative-darkness branch of `build_trace_mask` produces dotty
   noise on paper-WB output.  Drop it: absolute threshold is enough
   when paper has been WB'd to near-white.

Methods
=======
    M0_L11_baseline             production reference

    M_pwb_lab_a                 *** main candidate ***
        paper-WB -> Lab a* > 130 mask -> paper-blend fill -> global lift
        -> trace mask (absolute only).  Single-stage chromatic removal,
        no HSV, no hard clamp, no local lift.

    M_pwb_lab_a_geom            geometry-assisted variant
        same as M_pwb_lab_a BUT after a*>130 mask, also detect grid line
        positions (peaks in the chromatic mask), project full grid, and
        union into the removal mask.  Catches minor-grid sections too
        faint for the chroma threshold.

    M_pwb_strong_clamp          brightness-only fallback
        Strong paper-WB pulling paper to (255,255,255), then hard clamp
        gray >= 200.  Should kill minor grid (FFEBCC) by brightness.

    M_pwb_lab_a_wide            wider a* tolerance for shadow-orange
        same as M_pwb_lab_a but a* > 128 + 1.5 (catches very faint orange)
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
# Building blocks
# ===========================================================================

def paper_wb(image_bgr: np.ndarray, percentile: float = 95.0) -> np.ndarray:
    """Estimate paper colour from top-(100-percentile)% brightest pixels and
    scale each channel so paper becomes neutral gray."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    thr = np.percentile(gray, percentile)
    mask = gray >= thr
    paper = image_bgr[mask].astype(np.float32)
    means = paper.mean(axis=0)  # B, G, R
    target = float(means.mean())
    scale = target / np.maximum(means, 1.0)
    out = image_bgr.astype(np.float32) * scale[None, None, :]
    return np.clip(out, 0, 255).astype(np.uint8)


def paper_wb_to_white(image_bgr: np.ndarray, percentile: float = 95.0,
                     target_value: float = 250.0) -> np.ndarray:
    """Stronger version: pull paper toward (target, target, target)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    thr = np.percentile(gray, percentile)
    mask = gray >= thr
    paper = image_bgr[mask].astype(np.float32)
    means = paper.mean(axis=0)
    scale = target_value / np.maximum(means, 1.0)
    scale = np.clip(scale, 0.5, 3.0)
    out = image_bgr.astype(np.float32) * scale[None, None, :]
    return np.clip(out, 0, 255).astype(np.uint8)


def global_paper_lift(image_bgr: np.ndarray, target: int = 245) -> np.ndarray:
    """Multiply image by single GLOBAL scalar so the top-5% paper sits at
    `target`.  Unlike normalise_paper_to_white this does NOT use a local
    background divide, so it never blows up shadow regions."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    paper_val = float(np.percentile(gray, 95))
    if paper_val < 1.0:
        return image_bgr
    scale = float(target) / paper_val
    scale = min(scale, 1.6)  # cap so we don't crush ink contrast
    out = image_bgr.astype(np.float32) * scale
    return np.clip(out, 0, 255).astype(np.uint8)


def paper_blended_fill(image_bgr: np.ndarray, removal_mask: np.ndarray,
                      bg_sigma: float = 40.0) -> np.ndarray:
    """Fill masked pixels with smoothed local-paper estimate."""
    keep = (removal_mask == 0).astype(np.float32)
    masked = image_bgr.astype(np.float32) * keep[..., None]
    img_blur = cv2.GaussianBlur(masked, (0, 0), bg_sigma)
    w_blur   = cv2.GaussianBlur(keep,   (0, 0), bg_sigma)
    bg = img_blur / np.maximum(w_blur[..., None], 1e-3)
    bg = np.clip(bg, 0, 255).astype(np.uint8)
    out = image_bgr.copy()
    out[removal_mask > 0] = bg[removal_mask > 0]
    return out


def lab_a_mask(image_bgr: np.ndarray, a_threshold: float = 2.0,
              close_px: int = 2, min_component: int = 12) -> np.ndarray:
    """Pixels with a* sufficiently above the neutral 128 - i.e. orange/red."""
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


def project_grid_from_peaks(orange_mask: np.ndarray,
                            band_half_px: int = 3) -> np.ndarray:
    """Infer regular grid positions from peaks in orange-mask projection
    and rasterise as thin bands.  Catches minor-grid lines that are too
    faint for the chroma threshold but lie at predictable spacings."""
    H, W = orange_mask.shape
    if not np.any(orange_mask):
        return np.zeros_like(orange_mask)
    strong = orange_mask > 0
    row_d = strong.sum(axis=1) / max(1, W)
    col_d = strong.sum(axis=0) / max(1, H)
    h_peaks = _find_local_peaks(row_d, 14, 0.15)
    v_peaks = _find_local_peaks(col_d, 14, 0.15)
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
    """Absolute-only trace detector.  No local-darkness branch, so paper
    texture / JPEG noise on paper-WB output doesn't produce dotty noise."""
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


# ===========================================================================
# Methods
# ===========================================================================

def method_M0_L11(rect_bgr, template):
    return isolate_trace_candidates(rect_bgr, template).cleaned_bgr


def method_M_pwb_lab_a(rect_bgr, template):
    """Paper-WB + a*>130 mask + paper-blend fill + global lift + simple trace."""
    wb = paper_wb(rect_bgr)
    orange = lab_a_mask(wb, a_threshold=2.0)
    cleaned = paper_blended_fill(wb, orange)
    cleaned = global_paper_lift(cleaned)
    trace = build_trace_mask_simple(cleaned)
    return trace_to_visual(trace)


def method_M_pwb_lab_a_geom(rect_bgr, template):
    """Paper-WB + a* mask + geometry assist (union of projected grid bands).
    Chromatic-only inside bands: a band pixel is removed only if it's actually
    chromatic OR bright (light minor grid), but tool-ink-dark stays."""
    wb = paper_wb(rect_bgr)
    orange = lab_a_mask(wb, a_threshold=2.0)
    bands = project_grid_from_peaks(orange, band_half_px=3)

    # In-band removal: only chromatic OR bright (kills minor grid, keeps ink).
    lab = cv2.cvtColor(wb, cv2.COLOR_BGR2LAB)
    a = lab[..., 1].astype(np.float32)
    gray = cv2.cvtColor(wb, cv2.COLOR_BGR2GRAY)
    band_kill = (bands > 0) & (((a - 128) > 1.0) | (gray > 200))
    full_mask = (orange > 0) | band_kill
    full_mask = full_mask.astype(np.uint8) * 255

    cleaned = paper_blended_fill(wb, full_mask)
    cleaned = global_paper_lift(cleaned)
    trace = build_trace_mask_simple(cleaned)
    return trace_to_visual(trace)


def method_M_pwb_strong_clamp(rect_bgr, template):
    """Strong paper-WB to push paper close to (255,255,255), then brightness
    clamp gray>=200 to kill the light minor grid (FFEBCC)."""
    wb = paper_wb_to_white(rect_bgr, target_value=252.0)
    orange = lab_a_mask(wb, a_threshold=2.0)
    cleaned = paper_blended_fill(wb, orange)
    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    cleaned[gray >= 200] = (255, 255, 255)
    trace = build_trace_mask_simple(cleaned)
    return trace_to_visual(trace)


def method_M_pwb_lab_a_wide(rect_bgr, template):
    """Wider a* tolerance for shadow-orange."""
    wb = paper_wb(rect_bgr)
    orange = lab_a_mask(wb, a_threshold=1.0)  # lower threshold = more aggressive
    cleaned = paper_blended_fill(wb, orange)
    cleaned = global_paper_lift(cleaned)
    trace = build_trace_mask_simple(cleaned)
    return trace_to_visual(trace)


def method_M_pwb_lab_a_geom_wide(rect_bgr, template):
    """Wider a* + geometry assist combined."""
    wb = paper_wb(rect_bgr)
    orange = lab_a_mask(wb, a_threshold=1.0)
    bands = project_grid_from_peaks(orange, band_half_px=4)
    lab = cv2.cvtColor(wb, cv2.COLOR_BGR2LAB)
    a = lab[..., 1].astype(np.float32)
    gray = cv2.cvtColor(wb, cv2.COLOR_BGR2GRAY)
    band_kill = (bands > 0) & (((a - 128) > 0.5) | (gray > 195))
    full_mask = ((orange > 0) | band_kill).astype(np.uint8) * 255
    cleaned = paper_blended_fill(wb, full_mask)
    cleaned = global_paper_lift(cleaned)
    trace = build_trace_mask_simple(cleaned)
    return trace_to_visual(trace)


METHODS: list[tuple[str, Callable]] = [
    ("M0_L11_baseline",        method_M0_L11),
    ("M_pwb_lab_a",            method_M_pwb_lab_a),
    ("M_pwb_lab_a_geom",       method_M_pwb_lab_a_geom),
    ("M_pwb_lab_a_geom_wide",  method_M_pwb_lab_a_geom_wide),
    ("M_pwb_lab_a_wide",       method_M_pwb_lab_a_wide),
    ("M_pwb_strong_clamp",     method_M_pwb_strong_clamp),
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
    run_dir = make_run_dir("step02_grid_removal", "flat_methods_v8")
    print(f"Run dir: {run_dir.relative_to(REPO)}\n")
    template = load_template_config(TEMPLATE_ID)
    per_case_dir = run_dir / "per_case_compare"
    per_case_dir.mkdir(exist_ok=True)
    per_method_summary_tiles: dict[str, list[np.ndarray]] = {
        n: [] for n, _ in METHODS
    }

    print(f"{'case':<24s}  " + "  ".join(
        f"{n[:22]:>22s}" for n, _ in METHODS))
    print("-" * (24 + 4 + len(METHODS) * 24))

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
            f"{timings[n]:>20.2f}s" if not np.isnan(timings[n])
            else f"{'ERR':>22s}"
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
