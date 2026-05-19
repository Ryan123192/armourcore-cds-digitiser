"""Batch comparison of cross-reference repair methods.

Problem
-------
After Phase-2 grid removal (HSV ∪ curved-band-envelope), every grid
crossing has cut through tool ink, leaving tool outlines visibly
dashed.  Our existing minor-gap-fill struggles on the photo cases.

Goal
----
Produce a CLEAN binary trace mask where tool outlines are continuous,
ready to be vectorised by the existing pipeline.  The customer ink is
DARKER than the orange grid lines in the original, so we can use the
ORIGINAL rectified image as cross-reference evidence to recover the
tool ink at exactly the band-crossing positions.

This script tries 8 different repair strategies and saves an overlay
PNG and a binary trace-mask PNG for each so you can compare quality
visually:

    R0_no_repair               baseline: cleaned image's trace mask
    R1_orig_dark_restore       restore any pixel inside band envelope
                                that was dark in the ORIGINAL image
    R2_chromaticity_restore    restore dark AND achromatic in original
                                (so dark ORANGE grid is not restored)
    R3_morph_close             small morphological close on trace mask
    R4_inpaint_telea           cv2.inpaint Telea on band envelope
    R5_inpaint_ns              cv2.inpaint Navier-Stokes on band envelope
    R6_skel_endpoint_bridge    skeletonize trace, bridge close endpoints
    R7_orig_dark_plus_close    R1 + R3 (cross-ref restore then small close)
    R8_orig_dark_plus_bridge   R1 + R6

    python tools/test_cross_ref_repair.py
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np

from armourcore_cds.phase2.trace_isolation import (
    normalise_lighting,
    build_combined_orange_mask,
    detect_per_line_grid_bands,
)
from armourcore_cds.templates.registry import load_template_config

REPO = Path(__file__).parent.parent


# =====================================================================
# Inputs
# =====================================================================

def _load_inputs(case: str = "BlueColourTest01"):
    """Load original rectified + Phase-2 cleaned + recompute band envelope."""
    p1_run = sorted((REPO / "outputs" / "runs").glob(f"*{case}*"))[-1]
    rect_path = p1_run / "scaled_design_area.png"
    if not rect_path.exists():
        sys.exit(f"missing {rect_path}")
    cleaned_path = p1_run / "phase2" / "phase2_cleaned_raster.png"
    if not cleaned_path.exists():
        sys.exit(f"missing {cleaned_path} — run end_to_end first")

    rect = cv2.imread(str(rect_path))
    cleaned = cv2.imread(str(cleaned_path))
    if cleaned.shape != rect.shape:
        cleaned = cv2.resize(cleaned, (rect.shape[1], rect.shape[0]),
                             interpolation=cv2.INTER_AREA)

    # Re-derive the band envelope from the rectified image — matches what
    # Phase 2 used.  This is the region where bands cut through tool ink.
    rect_norm = normalise_lighting(rect)
    orange = build_combined_orange_mask(rect_norm)
    bands = detect_per_line_grid_bands(orange)
    envelope = cv2.dilate(
        bands,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    return rect, rect_norm, cleaned, envelope


def _trace_mask_from(cleaned_bgr: np.ndarray, thresh: int = 170) -> np.ndarray:
    """Phase-3's stock trace-mask detector."""
    gray = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
    return ((gray < thresh).astype(np.uint8)) * 255


# =====================================================================
# Repair methods
# =====================================================================

def R0_no_repair(rect, rect_norm, cleaned, envelope):
    """Baseline: just the stock trace mask from the cleaned image."""
    return _trace_mask_from(cleaned)


def R1_orig_dark_restore(rect, rect_norm, cleaned, envelope,
                          dark_thresh: int = 110):
    """Cross-reference repair using ORIGINAL darkness only.

    For every pixel inside the band envelope: if the ORIGINAL rectified
    image was darker than ``dark_thresh``, that pixel WAS customer ink
    (or grid line, but grid was orange and we're inside a band so it's
    going to be removed anyway).  Restore those pixels into the trace
    mask.
    """
    base = _trace_mask_from(cleaned)
    gray_orig = cv2.cvtColor(rect_norm, cv2.COLOR_BGR2GRAY)
    restore = (envelope > 0) & (gray_orig < dark_thresh)
    out = base.copy()
    out[restore] = 255
    return out


def R2_chromaticity_restore(rect, rect_norm, cleaned, envelope,
                             dark_thresh: int = 110,
                             sat_max: int = 30):
    """Cross-reference repair with chromaticity check.

    Same as R1 but also requires the ORIGINAL pixel to be ACHROMATIC
    (saturation < ``sat_max``).  This skips dark *orange* pixels
    (grid major-line shadow) and only restores dark grey / black
    pixels (real customer ink — pencil, pen, graphite).
    """
    base = _trace_mask_from(cleaned)
    gray_orig = cv2.cvtColor(rect_norm, cv2.COLOR_BGR2GRAY)
    hsv_orig  = cv2.cvtColor(rect_norm, cv2.COLOR_BGR2HSV)
    sat_orig  = hsv_orig[..., 1]
    restore = ((envelope > 0)
               & (gray_orig < dark_thresh)
               & (sat_orig < sat_max))
    out = base.copy()
    out[restore] = 255
    return out


def R3_morph_close(rect, rect_norm, cleaned, envelope,
                    close_radius: int = 5):
    """Just a small morphological close on the trace mask."""
    base = _trace_mask_from(cleaned)
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (close_radius * 2 + 1, close_radius * 2 + 1),
    )
    return cv2.morphologyEx(base, cv2.MORPH_CLOSE, k)


def R4_inpaint_telea(rect, rect_norm, cleaned, envelope):
    """OpenCV inpainting (Telea) on the band envelope region.

    Treats the band-envelope as a damaged region and reconstructs it
    from surrounding pixels.  Inpaints in BGR space, then re-derives
    the trace mask.
    """
    inpainted = cv2.inpaint(cleaned, envelope, 5, cv2.INPAINT_TELEA)
    return _trace_mask_from(inpainted)


def R5_inpaint_ns(rect, rect_norm, cleaned, envelope):
    """OpenCV inpainting (Navier-Stokes) on the band envelope region."""
    inpainted = cv2.inpaint(cleaned, envelope, 5, cv2.INPAINT_NS)
    return _trace_mask_from(inpainted)


def R6_skel_endpoint_bridge(rect, rect_norm, cleaned, envelope,
                             max_bridge_px: int = 20):
    """Skeletonise + endpoint detection + short-distance bridging.

    1. Skeletonise the trace mask to 1-px-thick centerlines.
    2. Find endpoint pixels (pixels with exactly 1 neighbour).
    3. For each pair of endpoints within ``max_bridge_px``, draw a
       straight line connecting them on the trace mask.
    """
    from skimage.morphology import skeletonize

    base = _trace_mask_from(cleaned)
    skel = skeletonize(base > 0).astype(np.uint8)

    # Find endpoints — pixels with exactly 1 lit 8-neighbour
    k8 = np.ones((3, 3), dtype=np.uint8)
    neigh_count = cv2.filter2D(skel, ddepth=cv2.CV_8U, kernel=k8) - skel
    endpoints_yx = np.column_stack(np.where((skel == 1) & (neigh_count == 1)))
    print(f"    R6: {len(endpoints_yx)} endpoints, bridging within "
          f"{max_bridge_px}px")

    out = base.copy()
    if len(endpoints_yx) >= 2:
        # Build a simple KD-tree of endpoints for efficient neighbour query
        from scipy.spatial import cKDTree
        tree = cKDTree(endpoints_yx)
        pairs = tree.query_pairs(r=max_bridge_px)
        n_bridged = 0
        for i, j in pairs:
            p0 = tuple(int(v) for v in endpoints_yx[i][::-1])  # x, y
            p1 = tuple(int(v) for v in endpoints_yx[j][::-1])
            cv2.line(out, p0, p1, 255, 2)
            n_bridged += 1
        print(f"    R6: bridged {n_bridged} endpoint pairs")
    return out


def R7_orig_dark_plus_close(rect, rect_norm, cleaned, envelope):
    """Combine R1 restoration + small morph close."""
    out = R1_orig_dark_restore(rect, rect_norm, cleaned, envelope)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)


def R8_orig_dark_plus_bridge(rect, rect_norm, cleaned, envelope):
    """Combine R1 restoration + skeleton-endpoint bridging."""
    repaired_cleaned = cleaned.copy()
    # Apply R1 restoration first by darkening cleaned where restoration says so
    gray_orig = cv2.cvtColor(rect_norm, cv2.COLOR_BGR2GRAY)
    restore = (envelope > 0) & (gray_orig < 110)
    repaired_cleaned[restore] = (0, 0, 0)
    return R6_skel_endpoint_bridge(rect, rect_norm, repaired_cleaned, envelope)


METHODS: list[tuple[str, Callable]] = [
    ("R0_no_repair",             R0_no_repair),
    ("R1_orig_dark_restore",     R1_orig_dark_restore),
    ("R2_chromaticity_restore",  R2_chromaticity_restore),
    ("R3_morph_close",           R3_morph_close),
    ("R4_inpaint_telea",         R4_inpaint_telea),
    ("R5_inpaint_ns",            R5_inpaint_ns),
    ("R6_skel_endpoint_bridge",  R6_skel_endpoint_bridge),
    ("R7_orig_dark_plus_close",  R7_orig_dark_plus_close),
    ("R8_orig_dark_plus_bridge", R8_orig_dark_plus_bridge),
]


# =====================================================================
# Visualisation
# =====================================================================

def _trace_to_visual(trace_mask: np.ndarray) -> np.ndarray:
    """Black ink on white paper visualisation of a binary trace mask."""
    img = np.full((*trace_mask.shape, 3), 255, dtype=np.uint8)
    img[trace_mask > 0] = (40, 40, 40)
    return img


def _label(img, txt, scale=0.7):
    out = img.copy()
    band = int(38 * scale)
    cv2.rectangle(out, (0, 0), (out.shape[1], band), (0, 0, 0), -1)
    cv2.putText(out, txt, (10, band - 10), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main():
    case = "BlueColourTest01"
    print("loading inputs...")
    rect, rect_norm, cleaned, envelope = _load_inputs(case)
    print(f"  rect {rect.shape}, cleaned {cleaned.shape}, "
          f"envelope coverage {envelope.mean() / 2.55:.2f}%")

    out_root = (REPO / "data" / "outputs" / "cross_ref_repair" / case
                / time.strftime("%Y%m%d-%H%M%S"))
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"output: {out_root.relative_to(REPO)}\n")

    crops: list[np.ndarray] = []
    H, W = cleaned.shape[:2]
    # Detail crop on a central area with several tool outlines
    cy0, cy1 = int(H * 0.10), int(H * 0.60)
    cx0, cx1 = int(W * 0.30), int(W * 0.70)

    for name, fn in METHODS:
        print(f"[{name}]")
        t0 = time.time()
        try:
            trace = fn(rect, rect_norm, cleaned, envelope)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue
        elapsed = time.time() - t0

        sub = out_root / name
        sub.mkdir(exist_ok=True)
        cv2.imwrite(str(sub / "trace_mask.png"), trace)
        visual = _trace_to_visual(trace)
        cv2.imwrite(str(sub / "trace_visual.png"), visual)

        # Detail crop
        crop = visual[cy0:cy1, cx0:cx1]
        crops.append(_label(crop, f"{name}  ink={int((trace > 0).sum()):,}"))
        print(f"  ink px {int((trace > 0).sum()):,} | {elapsed:.2f}s")

    # 3x3 grid of crops for at-a-glance comparison
    if crops:
        h_c, w_c = crops[0].shape[:2]
        cols = 3
        rows = (len(crops) + cols - 1) // cols
        grid = np.full((rows * h_c, cols * w_c, 3), 255, dtype=np.uint8)
        for i, c in enumerate(crops):
            r_, c_ = divmod(i, cols)
            grid[r_ * h_c:(r_ + 1) * h_c, c_ * w_c:(c_ + 1) * w_c] = c
        cv2.imwrite(str(out_root / "summary_crops_3x3.png"), grid)

    print(f"\nopen:  {out_root.relative_to(REPO)}/summary_crops_3x3.png")


if __name__ == "__main__":
    main()
