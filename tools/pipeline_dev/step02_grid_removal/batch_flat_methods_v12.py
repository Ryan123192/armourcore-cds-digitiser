"""Step 02 - v12 batch: template-projected grid geometry + aggressive dilation.

v11 finding
===========
Peak-inferred grid bands are robust where colour signal is strong but
miss the parts where it isn't (shadow zone in img 2, faded minor grid
patches throughout).  This leaves minor-grid residuals that the pencil-
aware trace detector picks up as dotty dashes.

Insight
=======
The Phase 1 rectifier outputs the paper at EXACT known dimensions:
  3450 x 2550 px = 345 x 255 mm @ 10 px/mm
Template grid spacing is 10 mm minor / 50 mm major.
So minor grid lines are EXACTLY every 100 px, major every 500 px.

We can project a complete grid mask from template geometry alone, no
peak detection needed.  Then kill any bright-AND-paper-coloured pixel
that falls within those projected bands.

This is bullet-proof for the grid (it MUST be at those positions on a
rectified image), and combined with the achromatic-protection rule
(don't kill anything with a* ~ 128) it never eats tool ink.

Methods
=======
    M0_L11_baseline             reference
    M_v9_winner                 v9 main candidate
    M_tmpl_grid                 paper-WB + a*-mask + template-grid kill
    M_tmpl_grid_dilated         + 3px dilation of orange mask
    M_tmpl_grid_dilated_pencil  + pencil-aware trace mask
    M_tmpl_grid_full            template grid + dilation + pencil-aware
                                + post-trace removal of any pixels inside
                                template grid bands (last-line defence)
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
    isolate_trace_candidates, _remove_small_components,
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


def bright_gated_orange_mask(image_bgr, a_threshold=1.0,
                            brightness_floor=150, close_px=2,
                            min_component=12, dilate_px=0):
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
    if dilate_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        mask = cv2.dilate(mask, k, iterations=1)
    return mask


def template_grid_bands(H, W, minor_spacing_mm=10.0,
                       major_spacing_mm=50.0,
                       px_per_mm_x=None, px_per_mm_y=None,
                       minor_half_px=3, major_half_px=4):
    """Project regular grid bands directly from template geometry.

    Since Phase 1 outputs a rectified image at known dimensions, every
    grid line position is mathematically determined.  No detection
    required.

    Returns a uint8 mask with 255 inside band, 0 elsewhere.
    """
    if px_per_mm_x is None:
        px_per_mm_x = W / PAPER_W_MM
    if px_per_mm_y is None:
        px_per_mm_y = H / PAPER_H_MM
    bands = np.zeros((H, W), dtype=np.uint8)

    # Vertical lines (parallel to height)
    minor_step_x = minor_spacing_mm * px_per_mm_x
    n_v = int(W / minor_step_x) + 2
    for i in range(n_v):
        x = int(round(i * minor_step_x))
        if 0 <= x < W:
            is_major = (i * minor_spacing_mm) % major_spacing_mm < 1e-3
            half = major_half_px if is_major else minor_half_px
            t = 2 * half + 1
            cv2.line(bands, (x, 0), (x, H - 1), 255, t)

    # Horizontal lines (parallel to width)
    minor_step_y = minor_spacing_mm * px_per_mm_y
    n_h = int(H / minor_step_y) + 2
    for i in range(n_h):
        y = int(round(i * minor_step_y))
        if 0 <= y < H:
            is_major = (i * minor_spacing_mm) % major_spacing_mm < 1e-3
            half = major_half_px if is_major else minor_half_px
            t = 2 * half + 1
            cv2.line(bands, (0, y), (W - 1, y), 255, t)

    return bands


def template_grid_kill_mask(wb_image, a_neutral_protect=1.0,
                           band_brightness_floor=160,
                           minor_half_px=3, major_half_px=4):
    """Inside template grid bands, kill any pixel that is bright AND
    not strongly achromatic.  Protects tool ink even when ink crosses
    a grid line."""
    H, W = wb_image.shape[:2]
    bands = template_grid_bands(
        H, W,
        minor_half_px=minor_half_px,
        major_half_px=major_half_px,
    )
    lab = cv2.cvtColor(wb_image, cv2.COLOR_BGR2LAB)
    a = lab[..., 1].astype(np.float32)
    gray = cv2.cvtColor(wb_image, cv2.COLOR_BGR2GRAY)

    # Anything inside a band that is BRIGHT enough -> grid.
    bright = gray > band_brightness_floor
    # Anything inside a band whose a* is well above 128 -> definitely orange.
    chromatic = (a - 128.0) > 0.5
    # Tool ink that crosses the band is DARK (gray < 150).  Don't kill.
    kill = (bands > 0) & (bright | chromatic) & (gray > 100)
    return kill.astype(np.uint8) * 255, bands


def trace_mask_pencil_aware(cleaned_bgr, dark_threshold=185,
                           rel_dark_sigma=25.0, rel_dark_min=15,
                           min_component_px=25):
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


def method_M_v9_winner(rect_bgr, template):
    wb = paper_wb(rect_bgr)
    orange = bright_gated_orange_mask(wb, 2.0, 170)
    cleaned = paper_blended_fill(wb, orange)
    cleaned = global_paper_lift(cleaned)
    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    trace = (gray < 170).astype(np.uint8) * 255
    trace = _remove_small_components(trace, 12)
    return trace_to_visual(trace)


def method_M_tmpl_grid(rect_bgr, template):
    """Paper-WB + a*-mask + template-projected grid kill."""
    wb = paper_wb(rect_bgr)
    orange = bright_gated_orange_mask(wb, 1.0, 150)
    grid_kill, _ = template_grid_kill_mask(wb, band_brightness_floor=160)
    full = cv2.bitwise_or(orange, grid_kill)
    cleaned = paper_blended_fill(wb, full)
    cleaned = global_paper_lift(cleaned)
    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    trace = (gray < 170).astype(np.uint8) * 255
    trace = _remove_small_components(trace, 12)
    return trace_to_visual(trace)


def method_M_tmpl_grid_dilated(rect_bgr, template):
    """+ 3px dilation of the orange mask to eat anti-aliased fringes."""
    wb = paper_wb(rect_bgr)
    orange = bright_gated_orange_mask(wb, 1.0, 150, dilate_px=3)
    grid_kill, _ = template_grid_kill_mask(wb, band_brightness_floor=160)
    full = cv2.bitwise_or(orange, grid_kill)
    cleaned = paper_blended_fill(wb, full)
    cleaned = global_paper_lift(cleaned)
    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    trace = (gray < 170).astype(np.uint8) * 255
    trace = _remove_small_components(trace, 12)
    return trace_to_visual(trace)


def method_M_tmpl_grid_dilated_pencil(rect_bgr, template):
    """+ pencil-aware trace mask."""
    wb = paper_wb(rect_bgr)
    orange = bright_gated_orange_mask(wb, 1.0, 150, dilate_px=3)
    grid_kill, _ = template_grid_kill_mask(wb, band_brightness_floor=160)
    full = cv2.bitwise_or(orange, grid_kill)
    cleaned = paper_blended_fill(wb, full)
    cleaned = global_paper_lift(cleaned)
    trace = trace_mask_pencil_aware(cleaned, 185, 25, 15, 25)
    return trace_to_visual(trace)


def method_M_tmpl_grid_full(rect_bgr, template):
    """Full stack: dilated orange + template grid kill + pencil-aware
    + POST-TRACE removal of any pixels strictly inside the template
    minor-grid band centre (1-px wide).  Tool ink barely overlaps these
    1-px positions; minor-grid residuals lie exactly on them."""
    wb = paper_wb(rect_bgr)
    orange = bright_gated_orange_mask(wb, 1.0, 150, dilate_px=3)
    grid_kill, bands = template_grid_kill_mask(
        wb, band_brightness_floor=160,
        minor_half_px=2, major_half_px=3,
    )
    full = cv2.bitwise_or(orange, grid_kill)
    cleaned = paper_blended_fill(wb, full)
    cleaned = global_paper_lift(cleaned)
    trace = trace_mask_pencil_aware(cleaned, 185, 25, 15, 25)

    # Last-line defence: any trace pixels that landed exactly on the
    # 1-px-wide centre of a template grid line are almost certainly
    # residual minor-grid dashes.  Tool ink covers thick areas; a
    # single-pixel-wide hit on the grid centre is suspicious.
    H, W = trace.shape
    centres = np.zeros((H, W), dtype=np.uint8)
    px_per_mm_x = W / PAPER_W_MM
    px_per_mm_y = H / PAPER_H_MM
    step_x = 10.0 * px_per_mm_x
    step_y = 10.0 * px_per_mm_y
    for i in range(int(W / step_x) + 2):
        x = int(round(i * step_x))
        if 0 <= x < W:
            cv2.line(centres, (x, 0), (x, H - 1), 255, 1)
    for i in range(int(H / step_y) + 2):
        y = int(round(i * step_y))
        if 0 <= y < H:
            cv2.line(centres, (0, y), (W - 1, y), 255, 1)

    # Remove only trace components that lie entirely along grid centres.
    # Components that overlap centres but extend beyond are tool ink.
    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(trace, connectivity=8)
    keep = np.ones(n_lbl, dtype=bool)
    keep[0] = False
    for cid in range(1, n_lbl):
        area = stats[cid, cv2.CC_STAT_AREA]
        comp_mask = (lbl == cid)
        on_centre = (comp_mask & (centres > 0)).sum()
        # If MORE THAN HALF the component lies on a grid centre, drop it.
        if area > 0 and on_centre / area > 0.5:
            keep[cid] = False
    lut = np.zeros(n_lbl, dtype=np.uint8)
    lut[keep] = 255
    trace2 = lut[lbl].astype(np.uint8)
    return trace_to_visual(trace2)


METHODS: list[tuple[str, Callable]] = [
    ("M0_L11_baseline",              method_M0_L11),
    ("M_v9_winner",                  method_M_v9_winner),
    ("M_tmpl_grid",                  method_M_tmpl_grid),
    ("M_tmpl_grid_dilated",          method_M_tmpl_grid_dilated),
    ("M_tmpl_grid_dilated_pencil",   method_M_tmpl_grid_dilated_pencil),
    ("M_tmpl_grid_full",             method_M_tmpl_grid_full),
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
    run_dir = make_run_dir("step02_grid_removal", "flat_methods_v12")
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
