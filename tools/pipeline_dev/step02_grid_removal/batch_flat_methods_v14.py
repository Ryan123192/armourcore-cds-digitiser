"""Step 02 - v14 batch: dark-border detector + L11 for pencil.

Adaptive routing
================
The conservative-vs-aggressive tradeoff is image-dependent:
* img 1 has a dark image border (camera framing offset).  Aggressive
  thresholds (a*>1.0, bright>150, dilate=3) interact badly with that
  border and produce false dark fills.  -> use conservative params.
* img 2 has a shadow zone in the bottom-right.  Conservative params
  can't penetrate the shadow.  -> use aggressive params.
* img 3 has neither.  Either works.
* img 4 is pencil.  Per user: baseline L11 is best.  -> route directly.

Detectors
=========
* `detect_pencil_image`     dark-pixel percentile heuristic (from v13)
* `detect_dark_border`      large contiguous gray<60 region within 50px
                            of image edge -> conservative needed

Methods
=======
    M0_L11_baseline               reference
    M_v13_pen_conservative        v13 pen pipeline (works for img1)
    M_v14_pen_aggressive          aggressive grid kill (works for img2)
    M_v14_adaptive                detect characteristics + route
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
from armourcore_cds.phase3.vectorise import (
    extract_vector_paths, write_svg, render_vector_overlay,
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
# Shared blocks
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


def template_grid_bands(H, W, minor_spacing_mm=10.0, major_spacing_mm=50.0,
                       minor_half_px=3, major_half_px=5):
    px_per_mm_x = W / PAPER_W_MM
    px_per_mm_y = H / PAPER_H_MM
    bands = np.zeros((H, W), dtype=np.uint8)
    sx = minor_spacing_mm * px_per_mm_x
    sy = minor_spacing_mm * px_per_mm_y
    for i in range(int(W / sx) + 2):
        x = int(round(i * sx))
        if 0 <= x < W:
            is_major = (i * minor_spacing_mm) % major_spacing_mm < 1e-3
            half = major_half_px if is_major else minor_half_px
            t = 2 * half + 1
            cv2.line(bands, (x, 0), (x, H - 1), 255, t)
    for i in range(int(H / sy) + 2):
        y = int(round(i * sy))
        if 0 <= y < H:
            is_major = (i * minor_spacing_mm) % major_spacing_mm < 1e-3
            half = major_half_px if is_major else minor_half_px
            t = 2 * half + 1
            cv2.line(bands, (0, y), (W - 1, y), 255, t)
    return bands


def template_grid_kill(wb_image, brightness_floor=160,
                      minor_half_px=3, major_half_px=5,
                      protect_dark_below=100,
                      minor_spacing_mm=10.0, major_spacing_mm=50.0):
    H, W = wb_image.shape[:2]
    bands = template_grid_bands(H, W,
                                minor_spacing_mm=minor_spacing_mm,
                                major_spacing_mm=major_spacing_mm,
                                minor_half_px=minor_half_px,
                                major_half_px=major_half_px)
    lab = cv2.cvtColor(wb_image, cv2.COLOR_BGR2LAB)
    a = lab[..., 1].astype(np.float32)
    gray = cv2.cvtColor(wb_image, cv2.COLOR_BGR2GRAY)
    bright = gray > brightness_floor
    chromatic = (a - 128.0) > 0.5
    kill = (bands > 0) & (bright | chromatic) & (gray > protect_dark_below)
    return kill.astype(np.uint8) * 255


# ===========================================================================
# Detectors
# ===========================================================================

def detect_pencil_image(rect_bgr) -> bool:
    """Pencil if even the darkest 0.5% of WB'd pixels are > 75 gray."""
    wb = paper_wb(rect_bgr)
    gray = cv2.cvtColor(wb, cv2.COLOR_BGR2GRAY)
    p05 = float(np.percentile(gray, 0.5))
    return p05 > 75.0


def detect_shadow_zone(rect_bgr, strip_px: int = 100,
                      gap_threshold: float = 25.0,
                      brightest_min: float = 195.0) -> bool:
    """Returns True if the image has a partial shadow zone (one side
    significantly darker than another, but at least one side bright).

    Distinguishes from globally-dim paper (img1) where all sides are
    dim.  Aggressive grid kill helps shadow zones but hurts globally-dim
    images.

    Measured across corpus (strip p50 values):
      img1: max=184, min=154, gap=30 -> brightest<195 -> not shadow (dim)
      img2: max=199, min=163, gap=36 -> brightest>=195 -> SHADOW
      img3: max=224, min=214, gap=10 -> gap<25         -> not shadow
      img4: max=219, min=199, gap=20 -> gap<25         -> not shadow
    """
    gray = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2GRAY)
    strips = [
        gray[:, :strip_px],
        gray[:, -strip_px:],
        gray[:strip_px, :],
        gray[-strip_px:, :],
    ]
    p50s = [float(np.percentile(s, 50)) for s in strips]
    gap = max(p50s) - min(p50s)
    return gap > gap_threshold and max(p50s) >= brightest_min


# ===========================================================================
# Pipelines
# ===========================================================================

def _trace_to_visual(trace_mask, shape):
    out = np.full(shape, 255, dtype=np.uint8)
    out[trace_mask > 0] = (40, 40, 40)
    return out


def pipeline_conservative(rect_bgr):
    """v13 pen pipeline: gentle params, major-only template grid kill.
    Best when dark border exists (img 1)."""
    wb = paper_wb(rect_bgr)
    orange = bright_gated_orange_mask(wb, 2.0, 170, dilate_px=0)
    # major-only template grid kill (use 50mm spacing for both minor+major
    # so only the major-line positions are drawn)
    major_kill = template_grid_kill(
        wb, brightness_floor=160,
        minor_spacing_mm=50.0, major_spacing_mm=50.0,
        minor_half_px=6, major_half_px=6,
        protect_dark_below=100,
    )
    full = cv2.bitwise_or(orange, major_kill)
    cleaned = paper_blended_fill(wb, full)
    cleaned = global_paper_lift(cleaned)
    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    trace = (gray < 170).astype(np.uint8) * 255
    trace = _remove_small_components(trace, 12)
    return _trace_to_visual(trace, cleaned.shape)


def pipeline_aggressive(rect_bgr):
    """v12-style aggressive: bright-gated with a*>1.0, bright>150,
    dilate=3, full minor+major template grid kill.  Penetrates shadow
    zones (img 2)."""
    wb = paper_wb(rect_bgr)
    orange = bright_gated_orange_mask(wb, 1.0, 150, dilate_px=3)
    grid_kill = template_grid_kill(wb, brightness_floor=160,
                                   minor_half_px=3, major_half_px=6)
    full = cv2.bitwise_or(orange, grid_kill)
    cleaned = paper_blended_fill(wb, full)
    cleaned = global_paper_lift(cleaned)
    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    trace = (gray < 170).astype(np.uint8) * 255
    trace = _remove_small_components(trace, 12)
    return _trace_to_visual(trace, cleaned.shape)


def pipeline_l11_baseline(rect_bgr, template):
    """Route to production L11 - per user, best for pencil."""
    return isolate_trace_candidates(rect_bgr, template).cleaned_bgr


# ===========================================================================
# Methods
# ===========================================================================

def method_M0_L11(rect_bgr, template):
    return pipeline_l11_baseline(rect_bgr, template)


def method_M_v13_pen_conservative(rect_bgr, template):
    return pipeline_conservative(rect_bgr)


def method_M_v14_pen_aggressive(rect_bgr, template):
    return pipeline_aggressive(rect_bgr)


def method_M_v14_adaptive(rect_bgr, template):
    """Detect image characteristics and pick the right pipeline.

    Decision tree (inverted from v14 first cut - conservative is the
    better default; aggressive only helps shadow-zone images):
      pencil          -> L11 baseline (per user)
      shadow zone     -> aggressive pen (img2)
      otherwise       -> conservative pen (img1, img3)
    """
    if detect_pencil_image(rect_bgr):
        return pipeline_l11_baseline(rect_bgr, template)
    if detect_shadow_zone(rect_bgr):
        return pipeline_aggressive(rect_bgr)
    return pipeline_conservative(rect_bgr)


METHODS: list[tuple[str, Callable]] = [
    ("M0_L11_baseline",              method_M0_L11),
    ("M_v13_pen_conservative",       method_M_v13_pen_conservative),
    ("M_v14_pen_aggressive",         method_M_v14_pen_aggressive),
    ("M_v14_adaptive",               method_M_v14_adaptive),
]


def _fit_tile(img, max_w, max_h):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    return cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                      interpolation=cv2.INTER_AREA)


def _panel(img, title, tile_w, tile_h):
    canvas = np.full((tile_h, tile_w, 3), 250, dtype=np.uint8)
    header_h = 42
    cv2.rectangle(canvas, (0, 0), (tile_w, header_h), (35, 35, 35), -1)
    cv2.putText(canvas, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    body_h = tile_h - header_h - 6
    fit = _fit_tile(img, tile_w - 6, body_h)
    fh, fw = fit.shape[:2]
    y0 = header_h + (body_h - fh) // 2
    x0 = (tile_w - fw) // 2
    canvas[y0:y0 + fh, x0:x0 + fw] = fit
    return canvas


def _per_case_compare(rectified, per_method_cleaned, vector_overlay,
                     case_stem, routing_info,
                     tile_w=950, tile_h=720):
    """Big side-by-side compare tile: rectified + each method's cleaned +
    final vector overlay.  Each panel is letterboxed to a uniform size so
    every output is the same scale + labelled clearly."""
    panels = [_panel(rectified, f"rectified  [route: {routing_info}]",
                    tile_w, tile_h)]
    for name, cleaned in per_method_cleaned.items():
        panels.append(_panel(cleaned, name, tile_w, tile_h))
    if vector_overlay is not None:
        panels.append(_panel(vector_overlay, "VECTORISED (adaptive output)",
                            tile_w, tile_h))

    cols = 3
    rows = (len(panels) + cols - 1) // cols
    row_imgs = []
    for r in range(rows):
        row_tiles = panels[r * cols:(r + 1) * cols]
        while len(row_tiles) < cols:
            row_tiles.append(np.full((tile_h, tile_w, 3), 250, dtype=np.uint8))
        row_imgs.append(np.hstack(row_tiles))
    body = np.vstack(row_imgs)

    # Big title header
    H = body.shape[1]
    header = np.full((64, H, 3), 18, dtype=np.uint8)
    cv2.putText(header, case_stem, (24, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                (255, 255, 255), 3, cv2.LINE_AA)
    return np.vstack([header, body])


def vectorise_cleaned(cleaned_bgr, out_dir, case_stem):
    """Run Phase 3 on the adaptive output.  Saves SVG + overlay PNG.
    Returns (n_paths, overlay_bgr_or_None)."""
    gray = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
    trace_mask = (gray < 170).astype(np.uint8) * 255
    trace_mask = _remove_small_components(trace_mask, min_area_px=80)
    try:
        paths = extract_vector_paths(
            trace_mask, min_area_px=300.0, rdp_epsilon=2.5,
            gap_close_px=11, max_gap_close_px=41, circularity_min=0.04,
        )
    except Exception as exc:
        print(f"  vectorise FAILED on {case_stem}: {exc}")
        return 0, None

    H, W = trace_mask.shape
    write_svg(paths, out_dir / f"{case_stem}.svg",
             mask_shape=(H, W),
             design_width_mm=PAPER_W_MM, design_height_mm=PAPER_H_MM)

    base = np.full((H, W, 3), 255, dtype=np.uint8)
    overlay = render_vector_overlay(
        base, paths, mask_shape=(H, W),
        colour_bgr=(40, 50, 220), thickness=3,
    )
    cv2.imwrite(str(out_dir / f"{case_stem}_overlay.png"), overlay)
    return len(paths), overlay


def main():
    run_dir = make_run_dir("step02_grid_removal", "flat_methods_v14")
    print(f"Run dir: {run_dir.relative_to(REPO)}\n")
    template = load_template_config(TEMPLATE_ID)
    per_case_dir = run_dir / "per_case_compare"
    per_case_dir.mkdir(exist_ok=True)
    vector_dir = run_dir / "vectors"
    vector_dir.mkdir(exist_ok=True)
    per_method_summary_tiles: dict[str, list[np.ndarray]] = {
        n: [] for n, _ in METHODS
    }

    print(f"{'case':<24s}  {'route':<16s}  " + "  ".join(
        f"{n[:22]:>22s}" for n, _ in METHODS) + f"  {'vectors':>8s}")
    print("-" * (24 + 4 + 16 + 4 + len(METHODS) * 24 + 10))

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

        is_pencil = detect_pencil_image(rect)
        has_shadow = detect_shadow_zone(rect)
        if is_pencil:
            route = "L11(pencil)"
        elif has_shadow:
            route = "aggressive"
        else:
            route = "conservative"

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

        # Vectorise the adaptive output
        adaptive_cleaned = per_method_cleaned.get("M_v14_adaptive")
        n_paths = 0
        overlay = None
        if adaptive_cleaned is not None:
            n_paths, overlay = vectorise_cleaned(
                adaptive_cleaned, vector_dir, stem)

        tile = _per_case_compare(rect, per_method_cleaned, overlay,
                                 stem, route)
        cv2.imwrite(str(per_case_dir / f"{stem}.png"), tile)

        line = f"{stem:<24s}  {route:<16s}  "
        line += "  ".join(
            f"{timings[n]:>20.2f}s" if not np.isnan(timings[n])
            else f"{'ERR':>22s}"
            for n, _ in METHODS
        )
        line += f"  {n_paths:>8d}"
        print(line)

    for name, tiles in per_method_summary_tiles.items():
        if not tiles:
            continue
        summary = grid_montage(tiles, cols=2, tile_max_dim=900)
        cv2.imwrite(str(run_dir / f"summary_{name}.png"), summary)

    print(f"\nFolder: {run_dir}")


if __name__ == "__main__":
    main()
