"""Marker-based rectification — lighting-invariant (Lab) variant.

A drop-in replacement for the ``marker_rectify`` module that swaps the
HSV "non-paper-hue" coarse pass for a Lab-colour-space delta approach.

Why
---
The original HSV mask only worked when paper was blue-shadowed.  When
the paper went warm (sunlight), the *paper itself* satisfied the
"not-blue" hue check and the discriminator collapsed.  Adding a
saturation floor helped some cases but still wasn't robust.

This module instead samples the paper's Lab colour from a central
patch, then accepts pixels with:

    delta_a >= 12          (markers are red-shifted vs. paper)
    delta_b - delta_a <= 6 (NOT orange/yellow shifts -- excludes wood,
                            orange grid lines)

Because both the paper and the marker get the same AWB transform from
the camera, the **delta** is invariant under lighting.  Cool blue
paper, warm yellow paper, mixed shade -- the markers always show a
~12-30 unit red shift relative to the paper baseline.

Corner-aware template matching
------------------------------
The refine pass now uses **corner-specific synthetic templates**.
The printed markers have ruler ticks on their two INWARD-facing edges:

    TL marker  ->  ticks on RIGHT and BOTTOM
    TR marker  ->  ticks on LEFT  and BOTTOM
    BR marker  ->  ticks on LEFT  and TOP
    BL marker  ->  ticks on RIGHT and TOP

Generating a unique template per corner gives a much stronger match
than the previous single generic template.

The original ``marker_rectify.py`` is left untouched -- this is a new
sibling module that imports the proven refine methods (canny, hough,
harris, dark_bbox) from it.

Adapted from
``Working Modules/working python files/detect_markers_robust.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# Re-use the proven refine methods + dataclasses from the original module
from armourcore_cds.phase1 import marker_rectify as mr
from armourcore_cds.phase1.marker_rectify import (
    MARKER_MM, PAPER_W_MM, PAPER_H_MM, DEFAULT_PX_PER_MM,
    MarkerDetection, RectifyResult,
    _save, _candidate_from_quad, _score_candidate, _inner_corner,
)


# ---------------------------------------------------------------------------
# Tuning constants (lifted verbatim from working module)
# ---------------------------------------------------------------------------

# Lab delta-a thresholds
DELTA_A_MIN      = 12        # min red-shift to consider a pixel "marker ink"
ORANGE_TOLERANCE = 6         # max (delta_b - delta_a) to still be "red"

# Paper-sampling
PAPER_PATCH_FRAC = 0.10      # sample paper from the central 10% of the image
PAPER_BRIGHT_PCT = 75        # use the top-25% brightest of those (avoid creases)

# Coarse-search constants
ROI_HALF             = 110
CORNER_PAD_FRAC      = 0.25
COARSE_CLOSE_KSIZE   = 18
MAX_ASPECT           = 1.8
MIN_AREA             = 2000


# ===========================================================================
# A. Paper-colour estimation (median Lab of bright central pixels)
# ===========================================================================

def estimate_paper_lab(lab: np.ndarray) -> Tuple[float, float, float]:
    """Median Lab of the brightest pixels in a central patch.

    Paper is the brightest object in the central image region (away
    from corners), so the brightest sub-set inside that patch
    overwhelmingly belongs to the paper itself rather than the printed
    shapes.  Median over those pixels is robust to a few stray dark
    pixels from text or grid that creep in.
    """
    h, w = lab.shape[:2]
    fx, fy = int(w * PAPER_PATCH_FRAC), int(h * PAPER_PATCH_FRAC)
    cx, cy = w // 2, h // 2
    central = lab[cy - fy: cy + fy, cx - fx: cx + fx].reshape(-1, 3)
    L = central[:, 0]
    bright_thresh = np.percentile(L, PAPER_BRIGHT_PCT)
    chosen = central[L >= bright_thresh]
    return tuple(float(v) for v in np.median(chosen, axis=0))


# ===========================================================================
# B. Lighting-invariant red-ink mask via Lab delta
# ===========================================================================

def red_ink_mask(
    bgr: np.ndarray,
    paper_lab: Optional[Tuple[float, float, float]] = None,
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Return ``(uint8_mask, paper_lab)``.

    ``mask`` is 255 where the pixel is a likely marker-ink pixel
    (red shifted from the paper baseline), 0 elsewhere.  ``paper_lab``
    is the sampled baseline so callers can log / debug it.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
    if paper_lab is None:
        paper_lab = estimate_paper_lab(lab.astype(np.uint8))
    _Lp, ap, bp = paper_lab

    a = lab[:, :, 1]
    b = lab[:, :, 2]
    delta_a = a - ap
    delta_b = b - bp

    mask = (
        (delta_a >= DELTA_A_MIN)
        & ((delta_b - delta_a) <= ORANGE_TOLERANCE)
    ).astype(np.uint8) * 255
    return mask, paper_lab


# ===========================================================================
# C. Coarse marker ROI search using the Lab mask
# ===========================================================================

def _coarse_marker_rois_lab(
    bgr: np.ndarray,
    debug_dir: Path | None = None,
) -> dict[str, tuple[int, int]]:
    """Find a rough centroid for each corner marker via the Lab mask.

    Returns ``{"TL": (cx, cy), ...}``.  Missing labels are simply absent.
    """
    mask, _paper_lab = red_ink_mask(bgr)

    k_close = cv2.getStructuringElement(
        cv2.MORPH_RECT, (COARSE_CLOSE_KSIZE, COARSE_CLOSE_KSIZE)
    )
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)
    _save(debug_dir, "01a_red_ink_mask.jpg", mask)
    _save(debug_dir, "01b_red_ink_closed.jpg", closed)

    # Kill thin orange-grid stubs that stretch INWARD from the marker
    # into the quadrant: an open with a moderately big kernel.
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    core = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k_open, iterations=1)
    _save(debug_dir, "01c_marker_core.jpg", core)

    h, w = bgr.shape[:2]
    pad = CORNER_PAD_FRAC
    quadrants = {
        "TL": (0,                  0,                  int(w * pad),       int(h * pad)),
        "TR": (int(w * (1 - pad)), 0,                  w,                  int(h * pad)),
        "BR": (int(w * (1 - pad)), int(h * (1 - pad)), w,                  h),
        "BL": (0,                  int(h * (1 - pad)), int(w * pad),       h),
    }

    rois: dict[str, tuple[int, int]] = {}
    for label, (x1, y1, x2, y2) in quadrants.items():
        roi = core[y1:y2, x1:x2]
        cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        best, best_area = None, 0
        for c in cnts:
            a = cv2.contourArea(c)
            if a < 300:
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            asp = max(bw, bh) / max(min(bw, bh), 1)
            if asp > MAX_ASPECT:
                continue
            if a > best_area:
                best_area, best = a, c
        if best is None:
            continue
        # Image-moments centroid is robust to ruler-tick asymmetry: the
        # marker CORE itself is symmetric (the close blob isn't because
        # it includes ticks pointing one direction).
        M = cv2.moments(best)
        if M["m00"] > 0:
            cx = int(round(M["m10"] / M["m00"])) + x1
            cy = int(round(M["m01"] / M["m00"])) + y1
        else:
            bx, by, bw, bh = cv2.boundingRect(best)
            cx, cy = bx + bw // 2 + x1, by + bh // 2 + y1
        rois[label] = (cx, cy)
    return rois


# ===========================================================================
# D. Corner-aware synthetic templates
# ===========================================================================

# corner -> (has_top, has_right, has_bottom, has_left)  -- inward-facing
_CORNER_TICKS = {
    "TL": (False, True,  True,  False),
    "TR": (False, False, True,  True),
    "BR": (True,  False, False, True),
    "BL": (True,  True,  False, False),
}


def make_marker_template(corner: str, side: int) -> np.ndarray:
    """Synthesise a calibration-marker template for the given corner.

    The marker has:
      1. A square outline
      2. An inscribed circle
      3. A central crosshair
      4. Ruler ticks on the two INWARD-facing edges (corner-specific)

    Returns a uint8 image (black ink on white background) sized to fit
    ``side`` pixels for the marker plus a small margin.
    """
    pad = max(side // 12, 6)
    img_size = side + 2 * pad
    t = np.full((img_size, img_size), 255, np.uint8)
    s, e = pad, pad + side
    mid = (s + e) // 2

    # Outline
    cv2.rectangle(t, (s, s), (e, e), 0, 3)
    # Inscribed circle
    r = (e - s) // 2 - max(side // 40, 3)
    cv2.circle(t, (mid, mid), r, 0, 2)
    # Crosshair
    ch = (e - s) // 4
    cv2.line(t, (mid - ch, mid), (mid + ch, mid), 0, 2)
    cv2.line(t, (mid, mid - ch), (mid, mid + ch), 0, 2)

    # Ruler ticks on inward-facing edges
    n_ticks = 14
    tick_len = max(side // 18, 5)
    long_tick_len = tick_len + 4
    has_top, has_right, has_bottom, has_left = _CORNER_TICKS[corner]

    def _ticks_along(p0, p1, normal_dx, normal_dy):
        for i in range(n_ticks + 1):
            t_frac = i / n_ticks
            x = int(round(p0[0] + (p1[0] - p0[0]) * t_frac))
            y = int(round(p0[1] + (p1[1] - p0[1]) * t_frac))
            this_len = long_tick_len if i == n_ticks // 2 else tick_len
            cv2.line(t, (x, y),
                     (x + int(this_len * normal_dx),
                      y + int(this_len * normal_dy)),
                     0, 1)

    if has_top:    _ticks_along((s, s), (e, s),  0,  +1)
    if has_bottom: _ticks_along((s, e), (e, e),  0,  -1)
    if has_left:   _ticks_along((s, s), (s, e), +1,   0)
    if has_right:  _ticks_along((e, s), (e, e), -1,   0)
    return t


def _method_template_match_oriented(
    roi_gray: np.ndarray,
    corner: str,
    expected_side: int,
):
    """Corner-aware multi-scale template match.

    Tries a small ladder of scales (0.85 - 1.25) around the expected
    marker side and keeps the highest-correlation match.
    """
    blurred = cv2.GaussianBlur(roi_gray, (3, 3), 0)
    best_val = -1.0
    best_match = None
    best_template = None
    for s in (0.85, 0.92, 1.0, 1.08, 1.15, 1.25):
        side = max(int(round(expected_side * s)), 50)
        tmpl = make_marker_template(corner, side)
        if (tmpl.shape[0] > roi_gray.shape[0]
                or tmpl.shape[1] > roi_gray.shape[1]):
            continue
        res = cv2.matchTemplate(blurred, tmpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, mloc = cv2.minMaxLoc(res)
        if mx > best_val:
            best_val = mx
            best_match = (mloc, tmpl.shape, side)
            best_template = tmpl
    if best_match is None or best_val < 0.15:
        return None
    mloc, (th, tw), side = best_match
    cx = mloc[0] + tw / 2
    cy = mloc[1] + th / 2
    return (float(cx), float(cy), float(side), float(side), 0.0), best_template


# ===========================================================================
# E. Lab-based ink_bbox refinement (replaces HSV ink_bbox)
# ===========================================================================

def _method_ink_bbox_lab(roi_bgr: np.ndarray):
    """Direct min-area rect of all Lab-red-shift pixels in the ROI.

    Drop-in replacement for the HSV ``ink_bbox`` method.  Works for
    cool / warm / mixed lighting because the Lab delta is invariant.
    """
    mask, _ = red_ink_mask(roi_bgr)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12)),
    )
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    pts = np.vstack([c.reshape(-1, 2) for c in cnts]).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    (cx, cy), (rw, rh), ang = rect
    if min(rw, rh) < 30:
        return None
    if rw < rh:
        rw, rh, ang = rh, rw, ang + 90
    return (float(cx), float(cy), float(rw), float(rh), float(ang)), mask


# ===========================================================================
# F. Public API
# ===========================================================================

def detect_markers_robust(
    image_bgr: np.ndarray,
    *,
    paper_w_mm: float = PAPER_W_MM,
    debug_dir: Path | None = None,
) -> dict[str, MarkerDetection]:
    """Lighting-invariant marker detector.

    Same return shape as ``marker_rectify.detect_markers``.  Uses the
    Lab-delta coarse pass, the corner-aware template match, and the
    Lab-based ink_bbox refinement on top of the original module's
    proven canny / hough / harris / dark_bbox methods.
    """
    h, w = image_bgr.shape[:2]
    _save(debug_dir, "00_original.jpg", image_bgr)

    # Expected marker side: assume paper spans ~92% of image width
    px_per_mm = (w * 0.92) / paper_w_mm
    expected_side = MARKER_MM * px_per_mm

    # A: Lab-delta coarse pass
    rois = _coarse_marker_rois_lab(image_bgr, debug_dir=debug_dir)

    # Save the synthetic templates once for visual inspection
    if debug_dir is not None:
        for corner in ("TL", "TR", "BR", "BL"):
            _save(
                debug_dir,
                f"04_template_{corner}.png",
                make_marker_template(corner, 120),
                max_w=300,
            )

    # B: Refine each ROI with the multi-method ensemble.
    # The corner-aware template match is the primary new addition; the
    # other methods are kept as fallbacks so a tough ROI still has many
    # bites at the apple.
    methods = [
        ("ink_lab",   "bgr",  _method_ink_bbox_lab),
        ("tmpl_cnr",  "gray", _method_template_match_oriented),     # NEW
        ("dark_bbox", "gray", mr._method_dark_bbox_gray),
        ("canny",     "gray", mr._method_canny_minrect),
        ("hough",     "gray", mr._method_hough_square),
        ("harris",    "gray", mr._method_harris_corners),
    ]

    final: dict[str, MarkerDetection] = {}
    for label, (cx0, cy0) in rois.items():
        x1, y1 = max(cx0 - ROI_HALF, 0), max(cy0 - ROI_HALF, 0)
        x2, y2 = min(cx0 + ROI_HALF, w), min(cy0 + ROI_HALF, h)
        roi = image_bgr[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _save(debug_dir, f"02_roi_{label}.jpg", roi)

        candidates: list[tuple[float, str, tuple]] = []
        for mname, src, mfn in methods:
            try:
                if mname == "tmpl_cnr":
                    out = mfn(gray, label, int(expected_side))
                elif src == "bgr":
                    out = mfn(roi)
                else:
                    out = mfn(gray)
            except Exception:
                continue
            if out is None:
                continue
            cand, dbg = out
            s = _score_candidate(cand, expected_side, roi.shape[0])
            candidates.append((s, mname, cand))
            if dbg is not None and debug_dir is not None:
                dbg_u8 = (
                    dbg if dbg.dtype == np.uint8
                    else cv2.normalize(dbg, None, 0, 255,
                                        cv2.NORM_MINMAX).astype(np.uint8)
                )
                _save(debug_dir, f"03_{label}_{mname}.jpg", dbg_u8)

        if not candidates:
            continue
        candidates.sort(key=lambda t: t[0])
        best_score, best_name, best_cand = candidates[0]
        cx, cy, rw, rh, ang = best_cand
        full_cx = cx + x1
        full_cy = cy + y1
        box = cv2.boxPoints(((full_cx, full_cy), (rw, rh), ang))
        final[label] = MarkerDetection(
            label=label,
            centre_xy=(full_cx, full_cy),
            size_wh=(rw, rh),
            angle_deg=ang,
            method=best_name,
            score=best_score,
            box=box,
        )
    return final


def rectify_with_markers_robust(
    image_bgr: np.ndarray,
    *,
    paper_w_mm: float = PAPER_W_MM,
    paper_h_mm: float = PAPER_H_MM,
    px_per_mm: float = DEFAULT_PX_PER_MM,
    debug_dir: Path | None = None,
) -> RectifyResult:
    """End-to-end Step-1: detect markers (robust), build inner-corner quad, warp.

    Same return shape as ``marker_rectify.rectify_with_markers`` so callers
    can swap modules without other code changes.
    """
    out_w = int(round(paper_w_mm * px_per_mm))
    out_h = int(round(paper_h_mm * px_per_mm))

    markers = detect_markers_robust(
        image_bgr, paper_w_mm=paper_w_mm, debug_dir=debug_dir,
    )
    missing = {"TL", "TR", "BR", "BL"} - markers.keys()
    if missing:
        raise RuntimeError(
            f"Required markers missing: {sorted(missing)}.  Detected: "
            f"{sorted(markers.keys())}"
        )

    paper_centre = np.mean(
        [m.centre_xy for m in markers.values()], axis=0,
    )
    inner_corners = {
        label: _inner_corner(det.box, paper_centre)
        for label, det in markers.items()
    }
    src_quad = np.array(
        [inner_corners[k] for k in ("TL", "TR", "BR", "BL")],
        dtype=np.float32,
    )
    dst_quad = np.array(
        [[0, 0],
         [out_w - 1, 0],
         [out_w - 1, out_h - 1],
         [0, out_h - 1]],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(src_quad, dst_quad)
    warped = cv2.warpPerspective(
        image_bgr, H, (out_w, out_h), flags=cv2.INTER_LANCZOS4,
    )

    if debug_dir is not None:
        vis = image_bgr.copy()
        for label, det in markers.items():
            cv2.polylines(vis, [det.box.astype(np.int32)],
                          True, (0, 255, 0), 4)
            cx, cy = det.centre_xy
            cv2.circle(vis, (int(cx), int(cy)), 8, (0, 0, 255), -1)
            cv2.putText(vis, label, (int(cx + 20), int(cy - 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 4)
        cv2.polylines(vis, [src_quad.astype(np.int32)],
                      True, (255, 200, 0), 5)
        for lbl, pt in zip(("TL", "TR", "BR", "BL"), src_quad):
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 20,
                       (0, 255, 255), -1)
            cv2.putText(vis, f"{lbl}-inner",
                        (int(pt[0]) + 25, int(pt[1]) + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (0, 255, 255), 3)
        _save(debug_dir, "10_inner_corners_and_quad.jpg", vis)
        _save(debug_dir, "20_rectified.jpg", warped)

    return RectifyResult(
        warped=warped,
        markers=markers,
        inner_corners_xy=inner_corners,
        src_quad=src_quad,
        homography=H,
        output_size_px=(out_w, out_h),
        paper_size_mm=(paper_w_mm, paper_h_mm),
        px_per_mm=px_per_mm,
    )
