"""Step 02 — Focused batch test on the 4 FLAT cases.

User direction: "in ALL examples the major grid lines and text still exist
so there HAS to be some kind of image filter or correction process to
remove the orange but keep the black/grey lines."

This batch tests **8 different methods** of isolating tracings from the
orange-grid background — from boring (L11 baseline) to creative
(black-hat morphology, Sauvola, color decorrelation, Frangi ridges).

Tested on the four FLAT cases only (no creases / fold shadows yet):
    1.  BLUE_PEN_FLAT_01
    2.  BLUE_PEN_FLAT_02
    3.  BLUE_PEN_FLAT_03
    7.  BLUE_PENCIL_FLAT_01

Methods
=======

    M0_L11_baseline
        Production L11 (HSV chroma + paper-blend + hard clamp @ 160)

    M1_lab_strict_L11clamp
        Lab Δa* mask + brightness floor (L>=120) + L11 hard clamp.
        The current best from the previous compare round (M2).

    M2_cmyk_k
        Extract the K channel via K = 1 - max(R,G,B).  Black ink has
        high K, orange grid is C+M+Y with very low K.  Threshold K
        for ink-only output.

    M3_blackhat
        Morphological black-hat with a 35-px ellipse highlights small
        dark features regardless of background brightness.  Threshold
        the response.

    M4_sauvola
        Sauvola adaptive document binarization (industry standard for
        OCR/text extraction).  Window 51, k=0.2.

    M5_bg_subtract
        Subtract a heavy Gaussian-blurred background.  The residual is
        all the small dark features.  Threshold.

    M6_lab_decorrelation
        In Lab a/b space, the paper-to-orange direction is roughly
        (+a, +b).  For every pixel, project (Δa, Δb) onto that
        direction.  Strongly positive projection -> orange grid ->
        set a,b = 128 (desaturated grey).  Then threshold L.

    M7_frangi_ridges
        Frangi vesselness filter (multi-scale Hessian ridge detector).
        Designed for thin curvilinear features.  Threshold the
        response.

Outputs
=======

    data/outputs/pipeline_dev/step02_grid_removal/<stamp>_flat_methods/
      M0_L11_baseline/
        <stem>/cleaned.png
        summary.png        4 cleaned images for this method
      M1_..._lab/...
      ...
      per_case_compare/
        <stem>.png         rectified + all 8 methods side-by-side

    python tools/pipeline_dev/step02_grid_removal/batch_flat_methods.py
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
from skimage.filters import threshold_sauvola, frangi
from skimage import img_as_ubyte

from armourcore_cds.phase1.marker_rectify_fast_v4 import (
    rectify_with_markers_fast_v4,
    PAPER_W_MM, PAPER_H_MM, DEFAULT_PX_PER_MM,
)
from armourcore_cds.phase2.trace_isolation import (
    isolate_trace_candidates, normalise_lighting,
    paper_blended_fill, normalise_paper_to_white,
)
from armourcore_cds.phase2.trace_isolation_v2 import (
    estimate_paper_lab, lab_delta_a_grid_mask,
)
from armourcore_cds.templates.registry import load_template_config
from tools.pipeline_dev.corpus import (
    RAW_IMAGES_DIR, make_run_dir, label_image, grid_montage,
)


TEMPLATE_ID = "cds_colour_test_260x350"

# Focus corpus: the four flat cases
FLAT_CASES = [
    "BLUE_PEN_FLAT_01",
    "BLUE_PEN_FLAT_02",
    "BLUE_PEN_FLAT_03",
    "BLUE_PENCIL_FLAT_01",
]


def _ink_visual(binary_mask: np.ndarray) -> np.ndarray:
    """Render a binary ink mask as black-on-white BGR."""
    h, w = binary_mask.shape[:2]
    out = np.full((h, w, 3), 255, dtype=np.uint8)
    out[binary_mask > 0] = (40, 40, 40)
    return out


# ===========================================================================
# Method 0 — L11 production baseline
# ===========================================================================

def method_M0_L11(rect_bgr, template):
    result = isolate_trace_candidates(rect_bgr, template)
    return result.cleaned_bgr


# ===========================================================================
# Method 1 — Lab Δa* (strict, L>=120) + L11's hard clamp
# ===========================================================================

def method_M1_lab_strict(rect_bgr, template):
    norm = normalise_lighting(rect_bgr)
    paper_lab = estimate_paper_lab(cv2.cvtColor(norm, cv2.COLOR_BGR2LAB))

    # Build Lab mask with brightness floor inline
    lab = cv2.cvtColor(norm, cv2.COLOR_BGR2LAB).astype(np.int16)
    L = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    _, ap, bp = paper_lab
    delta_a = a - ap
    delta_b = b - bp
    mask = (
        (delta_a >= 6)
        & ((delta_b - delta_a) <= 8)
        & (L >= 120)                              # brightness floor
    ).astype(np.uint8) * 255

    filled = paper_blended_fill(norm, mask)
    lifted = normalise_paper_to_white(filled)
    gray = cv2.cvtColor(lifted, cv2.COLOR_BGR2GRAY)
    out = lifted.copy()
    out[gray >= 160] = (255, 255, 255)
    return out


# ===========================================================================
# Method 2 — CMYK K-channel extraction
# ===========================================================================

def method_M2_cmyk_k(rect_bgr, template):
    """Black ink is the K channel; orange grid is C+M+Y with low K.

    K = 1 - max(R, G, B)
    Threshold K to find ink pixels only.
    """
    # Optional CLAHE first to flatten lighting
    norm = normalise_lighting(rect_bgr)
    bgr_f = norm.astype(np.float32) / 255.0
    # OpenCV is BGR; max across channels
    rgb_max = bgr_f.max(axis=2)
    K = 1.0 - rgb_max
    K_8bit = (K * 255).clip(0, 255).astype(np.uint8)
    # Threshold: ink has K > ~50/255 (0.20).  Tune.
    _, ink_mask = cv2.threshold(K_8bit, 40, 255, cv2.THRESH_BINARY)
    # Small noise removal
    ink_mask = cv2.morphologyEx(
        ink_mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
    )
    return _ink_visual(ink_mask)


# ===========================================================================
# Method 3 — Morphological black-hat
# ===========================================================================

def method_M3_blackhat(rect_bgr, template):
    """Black-hat = closing(img) - img.  Highlights dark features narrower
    than the structuring element regardless of background level.
    """
    norm = normalise_lighting(rect_bgr)
    gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, ink_mask = cv2.threshold(blackhat, 25, 255, cv2.THRESH_BINARY)
    # Remove single-pixel noise
    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(
        ink_mask, connectivity=8,
    )
    keep = np.where(stats[:, cv2.CC_STAT_AREA] >= 5, np.uint8(255), np.uint8(0))
    keep[0] = 0
    ink_mask = keep[lbl].astype(np.uint8)
    return _ink_visual(ink_mask)


# ===========================================================================
# Method 4 — Sauvola adaptive binarization
# ===========================================================================

def method_M4_sauvola(rect_bgr, template):
    """Industry-standard adaptive thresholding for document binarization.

    For every pixel:  threshold = mean_local * (1 + k * (std_local/R - 1))
    where R is the dynamic range of std (usually 128).
    """
    norm = normalise_lighting(rect_bgr)
    gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)
    thresh = threshold_sauvola(gray, window_size=51, k=0.2)
    binary = (gray < thresh).astype(np.uint8) * 255
    # Remove sub-5-px specks
    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8,
    )
    keep = np.where(stats[:, cv2.CC_STAT_AREA] >= 5, np.uint8(255), np.uint8(0))
    keep[0] = 0
    binary = keep[lbl].astype(np.uint8)
    return _ink_visual(binary)


# ===========================================================================
# Method 5 — Background subtraction
# ===========================================================================

def method_M5_bg_subtract(rect_bgr, template):
    """Heavy Gaussian = background.  Residual = small dark features.

    A pixel is ink when (background - pixel) exceeds a threshold.
    """
    norm = normalise_lighting(rect_bgr)
    gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = cv2.GaussianBlur(gray, (0, 0), 40.0)
    diff = bg - gray
    diff = np.clip(diff, 0, 255).astype(np.uint8)
    _, ink_mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
    return _ink_visual(ink_mask)


# ===========================================================================
# Method 6 — Lab color decorrelation along paper-to-orange axis
# ===========================================================================

def method_M6_lab_decorrelation(rect_bgr, template):
    """Project every pixel's (Δa, Δb) onto the paper-to-orange direction.

    Orange ink is +a AND +b shifted from paper, so the direction is
    roughly (+1, +1).  Pixels with strong positive projection are
    desaturated to neutral grey (a=128, b=128).  Then the L channel
    contains only the brightness signal: paper (high L) vs ink (low L).
    """
    norm = normalise_lighting(rect_bgr)
    lab = cv2.cvtColor(norm, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    paper_lab = estimate_paper_lab(lab)
    _, ap, bp = paper_lab
    da = a.astype(np.float32) - ap
    db = b.astype(np.float32) - bp
    # Project onto (1, 1)/sqrt(2)
    proj = (da + db) / np.sqrt(2.0)
    # Threshold: anything significantly along paper->orange axis becomes grey
    orange_pixels = proj > 6
    a[orange_pixels] = 128
    b[orange_pixels] = 128
    desat = cv2.merge([L, a, b])
    out_bgr = cv2.cvtColor(desat, cv2.COLOR_LAB2BGR)
    # Now lift paper to white
    out_bgr = normalise_paper_to_white(out_bgr)
    gray = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2GRAY)
    out_bgr[gray >= 160] = (255, 255, 255)
    return out_bgr


# ===========================================================================
# Method 7 — Frangi vesselness filter (multi-scale ridge detection)
# ===========================================================================

def method_M7_frangi(rect_bgr, template):
    """Frangi filter is a Hessian-based ridge detector originally for
    blood vessels.  It responds strongly to thin curvilinear structures
    and weakly to broad regions or noise.  Promising for thin pen lines.
    """
    norm = normalise_lighting(rect_bgr)
    gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)
    # Invert so dark = bright (Frangi expects bright ridges)
    inv = 255 - gray
    # Frangi at multiple stroke widths (sigmas in pixels)
    response = frangi(
        inv.astype(np.float32) / 255.0,
        sigmas=(1, 2, 3),
        black_ridges=False,
    )
    # Normalise + threshold
    response = (response * 255 / max(response.max(), 1e-6)).astype(np.uint8)
    _, ink_mask = cv2.threshold(response, 30, 255, cv2.THRESH_BINARY)
    return _ink_visual(ink_mask)


METHODS: list[tuple[str, Callable]] = [
    ("M0_L11_baseline",        method_M0_L11),
    ("M1_lab_strict",          method_M1_lab_strict),
    ("M2_cmyk_k",              method_M2_cmyk_k),
    ("M3_blackhat",            method_M3_blackhat),
    ("M4_sauvola",             method_M4_sauvola),
    ("M5_bg_subtract",         method_M5_bg_subtract),
    ("M6_lab_decorrelation",   method_M6_lab_decorrelation),
    ("M7_frangi",              method_M7_frangi),
]


# ===========================================================================
# Runner
# ===========================================================================

def _per_case_compare(rectified: np.ndarray,
                      per_method_cleaned: dict[str, np.ndarray],
                      case_stem: str) -> np.ndarray:
    """Tile rectified + each method's cleaned output for a single case."""
    tiles = [label_image(rectified, "rectified", height_px=48, scale=0.85)]
    for name, cleaned in per_method_cleaned.items():
        tiles.append(
            label_image(cleaned, name, height_px=48, scale=0.85)
        )
    # Two rows (4+5) for readability
    half = (len(tiles) + 1) // 2
    row_a = tiles[:half]
    row_b = tiles[half:]
    h = max(t.shape[0] for t in tiles)

    def _pad(img):
        if img.shape[0] == h:
            return img
        pad = np.full((h - img.shape[0], img.shape[1], 3), 240, dtype=np.uint8)
        return np.vstack([img, pad])

    row_a_img = np.hstack([_pad(t) for t in row_a])
    row_b_img = np.hstack([_pad(t) for t in row_b])
    w_max = max(row_a_img.shape[1], row_b_img.shape[1])

    def _pad_w(img, w):
        if img.shape[1] == w:
            return img
        pad = np.full((img.shape[0], w - img.shape[1], 3), 240, dtype=np.uint8)
        return np.hstack([img, pad])

    grid = np.vstack([_pad_w(row_a_img, w_max), _pad_w(row_b_img, w_max)])
    header = np.full((52, grid.shape[1], 3), 25, dtype=np.uint8)
    cv2.putText(
        header, case_stem, (16, 36),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return np.vstack([header, grid])


def main() -> None:
    run_dir = make_run_dir("step02_grid_removal", "flat_methods")
    print(f"Run dir: {run_dir.relative_to(REPO)}\n")

    template = load_template_config(TEMPLATE_ID)
    per_case_dir = run_dir / "per_case_compare"
    per_case_dir.mkdir(exist_ok=True)
    per_method_summary_tiles: dict[str, list[np.ndarray]] = {
        name: [] for name, _ in METHODS
    }

    header_methods = "  ".join(f"{name[:13]:>13s}" for name, _ in METHODS)
    print(f"{'case':<24s}  {header_methods}")
    print("-" * (24 + 4 + len(METHODS) * 15))

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
                (method_dir / "info.json").write_text(json.dumps({
                    "method": name, "stem": stem,
                    "elapsed_s": round(timings[name], 3),
                }, indent=2), encoding="utf-8")
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

        # Per-case all-method compare
        tile = _per_case_compare(rectified, per_method_cleaned, stem)
        cv2.imwrite(str(per_case_dir / f"{stem}.png"), tile)

        line = f"{stem:<24s}  "
        line += "  ".join(
            f"{timings[name]:>11.2f}s" if not np.isnan(timings[name])
            else f"{'ERR':>13s}"
            for name, _ in METHODS
        )
        print(line)

    # Per-method summaries (4 cases each)
    for name, tiles in per_method_summary_tiles.items():
        if not tiles:
            continue
        summary = grid_montage(tiles, cols=2, tile_max_dim=900)
        cv2.imwrite(str(run_dir / f"summary_{name}.png"), summary)

    (run_dir / "summary.json").write_text(json.dumps({
        "step": "step02_grid_removal",
        "label": "flat_methods",
        "template_id": TEMPLATE_ID,
        "methods": [n for n, _ in METHODS],
        "cases": FLAT_CASES,
    }, indent=2), encoding="utf-8")

    print(f"\nFolder: {run_dir}")
    print("Look at:")
    for name, _ in METHODS:
        print(f"  summary_{name}.png")
    print(f"  per_case_compare/<stem>.png")


if __name__ == "__main__":
    main()
