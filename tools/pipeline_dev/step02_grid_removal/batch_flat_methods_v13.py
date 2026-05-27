"""Step 02 - v13 batch: adaptive routing + major-grid-line pass + inpaint fill.

User-directed strategy (after v12 review)
=========================================
* Image 1 (PEN_FLAT_01, blue cast): v9_winner close to best - need grid lines gone
* Image 2 (PEN_FLAT_02, shadow):    tmpl_grid close - need major grid + shadow
* Image 3 (PEN_FLAT_03):            tmpl_grid PERFECT - keep as is
* Image 4 (PENCIL_FLAT_01):         baseline best - any v12 variant loses pencil

Three key additions
===================
1. Major-grid-line removal pass.  After chromatic removal, project the
   1mm-thick MAJOR grid lines (every 50mm) from template geometry and
   suppress any bright pixels along them.  Catches the "outer frame"
   that v9 leaves visible on img 1.

2. Shadow-zone recovery.  Detect low-brightness sub-regions (paper
   gray < 200 in a 200-px window) and apply MORE AGGRESSIVE grid kill
   there.  Fixes img 2's bottom-right corner.

3. cv2.inpaint instead of paper_blended_fill.  TELEA inpainting
   eliminates the grid-shaped intensity ripples that paper-blend leaves
   behind, so pencil-aware trace mask can be used safely (img 4 fix).

Methods
=======
    M0_L11_baseline             reference (best for pencil per user)
    M12_tmpl_grid_dilated       v12 winner (best for img2, img3 per user)
    M_v13_pen                   tmpl_grid_dilated + major-grid + shadow zone
    M_v13_pencil                bright-gated + inpaint fill + pencil-aware
                                trace.  Aims to beat baseline on img 4.
    M_v13_adaptive              detect pen vs pencil from image stats,
                                route to v13_pen or v13_pencil.
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
# Shared building blocks
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


def inpaint_fill(image_bgr, removal_mask, radius=5):
    """Use TELEA inpainting - smoother fill, no grid-shaped residuals."""
    if not np.any(removal_mask):
        return image_bgr.copy()
    return cv2.inpaint(image_bgr, removal_mask, radius, cv2.INPAINT_TELEA)


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
                      protect_dark_below=100):
    H, W = wb_image.shape[:2]
    bands = template_grid_bands(H, W,
                                minor_half_px=minor_half_px,
                                major_half_px=major_half_px)
    lab = cv2.cvtColor(wb_image, cv2.COLOR_BGR2LAB)
    a = lab[..., 1].astype(np.float32)
    gray = cv2.cvtColor(wb_image, cv2.COLOR_BGR2GRAY)
    bright = gray > brightness_floor
    chromatic = (a - 128.0) > 0.5
    kill = (bands > 0) & (bright | chromatic) & (gray > protect_dark_below)
    return kill.astype(np.uint8) * 255


def shadow_zone_mask(wb_image, dark_threshold=180, window_px=200,
                    bright_floor=90):
    """Detect regions where the paper itself is darker than usual (shadow).
    Returns binary mask: 255 in shadow, 0 in well-lit zone.

    Excludes solid-dark regions (gray < bright_floor) which are typically
    image borders or large ink blobs, not paper shadow.
    """
    gray = cv2.cvtColor(wb_image, cv2.COLOR_BGR2GRAY)
    local_max = cv2.dilate(
        gray,
        cv2.getStructuringElement(cv2.MORPH_RECT,
                                  (window_px, window_px)),
    )
    local_min = cv2.erode(
        gray,
        cv2.getStructuringElement(cv2.MORPH_RECT,
                                  (window_px, window_px)),
    )
    # Shadow zone: dim paper (local_max in [bright_floor, dark_threshold]).
    # If local_min is also very dark, it's a border or ink blob -> exclude.
    shadow = ((local_max < dark_threshold) &
              (local_max > bright_floor) &
              (local_min > 40)).astype(np.uint8) * 255
    return shadow


def detect_pencil_image(rect_bgr) -> bool:
    """Heuristic: does the image contain pencil traces (light gray) vs
    pen traces (dark/saturated)?

    Method: after paper-WB, look at the darkest 0.5% of pixels.  Pen
    traces include very dark (< 80) pixels.  Pencil traces top out around
    100-150 gray.
    """
    wb = paper_wb(rect_bgr)
    gray = cv2.cvtColor(wb, cv2.COLOR_BGR2GRAY)
    p05 = float(np.percentile(gray, 0.5))
    return p05 > 75.0   # pencil if even the darkest 0.5% are > 75 gray


# ===========================================================================
# Methods
# ===========================================================================

def method_M0_L11(rect_bgr, template):
    return isolate_trace_candidates(rect_bgr, template).cleaned_bgr


def method_M12_tmpl_grid_dilated(rect_bgr, template):
    wb = paper_wb(rect_bgr)
    orange = bright_gated_orange_mask(wb, 1.0, 150, dilate_px=3)
    grid_kill = template_grid_kill(wb, brightness_floor=160)
    full = cv2.bitwise_or(orange, grid_kill)
    cleaned = paper_blended_fill(wb, full)
    cleaned = global_paper_lift(cleaned)
    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    trace = (gray < 170).astype(np.uint8) * 255
    trace = _remove_small_components(trace, 12)
    out = np.full(cleaned.shape, 255, dtype=np.uint8)
    out[trace > 0] = (40, 40, 40)
    return out


def method_M_v13_pen(rect_bgr, template):
    """Pen pipeline: v9_winner's conservative params + template MAJOR-only
    grid kill.

    Why conservative: v12's aggressive thresholds (a*>1.0, bright>150,
    dilate 3px) interact badly with dark image-border regions on img 1,
    smearing dark pixels into bigger solid blobs.  v9_winner's
    (a*>2.0, bright>170, no dilate) leaves the border alone.

    To still remove the visible MAJOR grid lines on img 1, we add the
    template grid kill but ONLY for major lines (every 50mm), no minor.
    Major lines are wider and survive v9's chromatic threshold; minor
    lines do not.
    """
    wb = paper_wb(rect_bgr)
    # v9 conservative chromatic removal
    orange = bright_gated_orange_mask(wb, 2.0, 170, dilate_px=0)

    # MAJOR-only template grid kill (thick bands every 50mm, no minor)
    H, W = wb.shape[:2]
    bands = template_grid_bands(
        H, W,
        minor_spacing_mm=50.0,   # treat 50mm as "minor" so only major lines drawn
        major_spacing_mm=50.0,
        minor_half_px=6, major_half_px=6,
    )
    gray = cv2.cvtColor(wb, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(wb, cv2.COLOR_BGR2LAB)
    a = lab[..., 1].astype(np.float32)
    bright = gray > 160
    chromatic = (a - 128.0) > 0.5
    major_kill = ((bands > 0) & (bright | chromatic) &
                  (gray > 100)).astype(np.uint8) * 255

    full = cv2.bitwise_or(orange, major_kill)
    cleaned = paper_blended_fill(wb, full)
    cleaned = global_paper_lift(cleaned)

    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    trace = (gray < 170).astype(np.uint8) * 255
    trace = _remove_small_components(trace, 12)

    out = np.full(cleaned.shape, 255, dtype=np.uint8)
    out[trace > 0] = (40, 40, 40)
    return out


def method_M_v13_pencil(rect_bgr, template):
    """Pencil pipeline: same grid-removal as pen, but
    1) extra Gaussian smooth pass over killed regions to flatten ripples
    2) wider absolute trace threshold (catches light graphite)
    3) NO relative-darkness branch (avoids grid-shape false positives)
    """
    wb = paper_wb(rect_bgr)
    orange = bright_gated_orange_mask(wb, 1.0, 150, dilate_px=3)
    grid_kill = template_grid_kill(wb, brightness_floor=160,
                                   minor_half_px=3, major_half_px=5)
    full = cv2.bitwise_or(orange, grid_kill)

    cleaned = paper_blended_fill(wb, full, bg_sigma=60.0)  # wider blur
    cleaned = global_paper_lift(cleaned)

    # Post-fill smoothing INSIDE killed regions only - flattens ripples.
    blurred = cv2.GaussianBlur(cleaned, (0, 0), 5.0)
    cleaned[full > 0] = blurred[full > 0]

    # Wider abs threshold (catches light pencil ~ gray 150-190)
    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    trace = (gray < 195).astype(np.uint8) * 255
    trace = _remove_small_components(trace, 40)

    out = np.full(cleaned.shape, 255, dtype=np.uint8)
    out[trace > 0] = (40, 40, 40)
    return out


def method_M_v13_adaptive(rect_bgr, template):
    """Detect pen vs pencil, route accordingly."""
    if detect_pencil_image(rect_bgr):
        return method_M_v13_pencil(rect_bgr, template)
    return method_M_v13_pen(rect_bgr, template)


METHODS: list[tuple[str, Callable]] = [
    ("M0_L11_baseline",              method_M0_L11),
    ("M12_tmpl_grid_dilated",        method_M12_tmpl_grid_dilated),
    ("M_v13_pen",                    method_M_v13_pen),
    ("M_v13_pencil",                 method_M_v13_pencil),
    ("M_v13_adaptive",               method_M_v13_adaptive),
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
    run_dir = make_run_dir("step02_grid_removal", "flat_methods_v13")
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

        is_pencil = detect_pencil_image(rect)
        print(f"{stem}: detected as {'PENCIL' if is_pencil else 'PEN'}")

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
