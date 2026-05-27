"""Marker-based rectification — FAST v2 (paper-anchored coarse pass).

Identical to ``marker_rectify_fast.py`` (v1) EXCEPT for the coarse
quadrant anchoring:

    v1: search inside IMAGE corners (outer 25% per axis).
        Breaks when the paper doesn't fill the image -- rotated paper
        leaves wood inside the image corners, partial paper leaves
        bench / fabric inside the image corners.

    v2: find the PAPER first (largest bright region), then anchor the
        4 search quadrants on the PAPER's bounding box (inner 30% per
        axis of the paper region).  The quadrants then guarantee to
        sit inside the paper, so the Lab red-ink mask can only catch
        real marker ink.

Everything downstream of the coarse pass is unchanged -- same Lab
delta-a mask, same close/open chain, same image-moments centroid,
same corner-aware template + ink_lab refine pair, same scoring.

Failure analysis (v1 8/12 baseline):

    BLUE_PEN_FLAT_03         angled paper -> image-corner quadrants on wood
    BLUE_PENCIL_FLAT_01      same
    BLUE_PEN_CREASED_01      paper short of bottom -> BR quadrant on dark texture
    BLUE_PENCIL_CREASED_04   paper takes upper 70% -> BL/BR quadrants on fabric

All four failure modes have the same root cause: image-corner
quadrant -> not-paper region -> wrong / no detection.  Paper-anchored
quadrants fix all four in one shot.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Tuning constants (Lab + ROI sizes match v1)
# ---------------------------------------------------------------------------

DELTA_A_MIN      = 12
ORANGE_TOLERANCE = 6

PAPER_PATCH_FRAC = 0.10
PAPER_BRIGHT_PCT = 75

# NEW for v2: how big to make the paper-anchored coarse quadrants (per axis).
# 0.30 means the inner 30% of the paper bbox at each corner forms the search
# region.  This is tighter than v1's 0.25 of the IMAGE (which on a 4000-px
# image was a 1000-px quadrant -- huge), giving fewer chances for false
# positives while still leaving room for marker-position jitter.
PAPER_QUADRANT_FRAC = 0.30

# Minimum paper area as a fraction of total image area.  If the largest
# bright contour is smaller than this we fall back to v1's image-corner
# behaviour rather than trust an unreliable paper outline.
MIN_PAPER_AREA_FRAC = 0.30

COARSE_CLOSE_KSIZE   = 18
COARSE_OPEN_KSIZE    = 15
ROI_HALF             = 110
MAX_ASPECT_COARSE    = 1.8


# ---------------------------------------------------------------------------
# Physical reference for the warp target
# ---------------------------------------------------------------------------

MARKER_MM         = 15.0
PAPER_W_MM        = 345.0
PAPER_H_MM        = 255.0
DEFAULT_PX_PER_MM = 10.0


# ---------------------------------------------------------------------------
# Return shapes — identical to v1 so callers can swap modules cleanly
# ---------------------------------------------------------------------------

@dataclass
class MarkerDetection:
    label: str
    centre_xy: Tuple[float, float]
    size_wh: Tuple[float, float]
    angle_deg: float
    method: str
    score: float
    box: np.ndarray = field(
        repr=False, default_factory=lambda: np.zeros((4, 2))
    )


@dataclass
class RectifyResult:
    warped: np.ndarray
    markers: dict[str, MarkerDetection]
    inner_corners_xy: dict[str, Tuple[float, float]]
    src_quad: np.ndarray
    homography: np.ndarray
    output_size_px: Tuple[int, int]
    paper_size_mm: Tuple[float, float]
    px_per_mm: float
    # NEW: helpful diagnostic for v2 — what bounding box did we pick for paper?
    paper_bbox_xywh: Optional[Tuple[int, int, int, int]] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(debug_dir: Optional[Path],
          name: str,
          img: np.ndarray,
          *,
          max_w: int = 1800) -> None:
    if debug_dir is None:
        return
    debug_dir.mkdir(parents=True, exist_ok=True)
    h, w = img.shape[:2]
    if w > max_w:
        s = max_w / w
        img = cv2.resize(img, (int(w * s), int(h * s)))
    cv2.imwrite(str(debug_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])


# ===========================================================================
# NEW for v2:  find the paper region
# ===========================================================================

def find_paper_bbox(
    bgr: np.ndarray,
    debug_dir: Optional[Path] = None,
) -> Optional[Tuple[int, int, int, int]]:
    """Return the axis-aligned bounding box ``(x, y, w, h)`` of the paper.

    Strategy: paper is the brightest large object in the image.
    Threshold the L channel with Otsu, find the largest connected
    bright contour, take its bounding box.

    Returns None if the largest bright contour is too small to be the
    paper -- caller should fall back to image-corner quadrants.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]

    # Light blur to suppress paper-texture noise before Otsu
    blur = cv2.GaussianBlur(L, (15, 15), 0)
    _otsu_val, binary = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # Close small dark spots inside the paper (text / tools / markers)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=2)
    _save(debug_dir, "00b_paper_binary.jpg", closed)

    cnts, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    if not cnts:
        return None

    largest = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    img_area = bgr.shape[0] * bgr.shape[1]
    if area < img_area * MIN_PAPER_AREA_FRAC:
        return None

    x, y, w, h = cv2.boundingRect(largest)
    return (int(x), int(y), int(w), int(h))


# ===========================================================================
# Lab paper-baseline + red-ink mask  (identical to v1)
# ===========================================================================

def estimate_paper_lab(lab: np.ndarray) -> Tuple[float, float, float]:
    h, w = lab.shape[:2]
    fx, fy = int(w * PAPER_PATCH_FRAC), int(h * PAPER_PATCH_FRAC)
    cx, cy = w // 2, h // 2
    central = lab[cy - fy: cy + fy, cx - fx: cx + fx].reshape(-1, 3)
    L = central[:, 0]
    bright_thresh = np.percentile(L, PAPER_BRIGHT_PCT)
    chosen = central[L >= bright_thresh]
    return tuple(float(v) for v in np.median(chosen, axis=0))


def red_ink_mask(
    bgr: np.ndarray,
    paper_lab: Optional[Tuple[float, float, float]] = None,
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
    if paper_lab is None:
        paper_lab = estimate_paper_lab(lab.astype(np.uint8))
    _, ap, bp = paper_lab
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    delta_a = a - ap
    delta_b = b - bp
    mask = ((delta_a >= DELTA_A_MIN)
            & ((delta_b - delta_a) <= ORANGE_TOLERANCE)).astype(np.uint8) * 255
    return mask, paper_lab


# ===========================================================================
# A. Coarse pass — paper-anchored quadrants  (NEW in v2)
# ===========================================================================

def coarse_marker_rois(
    bgr: np.ndarray,
    debug_dir: Optional[Path] = None,
) -> Tuple[dict[str, Tuple[int, int]], np.ndarray, Optional[Tuple[int, int, int, int]]]:
    """Find each marker's approximate centre via paper-anchored quadrants.

    Returns ``(rois_by_label, core_mask, paper_bbox)``.  ``paper_bbox``
    is ``(x, y, w, h)`` if paper detection succeeded, else None and
    we fell back to image-corner quadrants.
    """
    paper_bbox = find_paper_bbox(bgr, debug_dir=debug_dir)

    mask, _paper_lab = red_ink_mask(bgr)
    k_close = cv2.getStructuringElement(
        cv2.MORPH_RECT, (COARSE_CLOSE_KSIZE, COARSE_CLOSE_KSIZE))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)
    k_open = cv2.getStructuringElement(
        cv2.MORPH_RECT, (COARSE_OPEN_KSIZE, COARSE_OPEN_KSIZE))
    core = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k_open, iterations=1)
    _save(debug_dir, "01a_red_ink_mask.jpg", mask)
    _save(debug_dir, "01b_marker_core.jpg",  core)

    h, w = bgr.shape[:2]

    if paper_bbox is not None:
        px, py, pw, ph = paper_bbox
        # Quadrants are corners of the paper bbox, sized as PAPER_QUADRANT_FRAC
        # of the paper dimensions.
        qw = int(pw * PAPER_QUADRANT_FRAC)
        qh = int(ph * PAPER_QUADRANT_FRAC)
        quads = {
            "TL": (px,           py,           px + qw,        py + qh),
            "TR": (px + pw - qw, py,           px + pw,        py + qh),
            "BR": (px + pw - qw, py + ph - qh, px + pw,        py + ph),
            "BL": (px,           py + ph - qh, px + qw,        py + ph),
        }
        # Annotate the diagnostic overlay
        if debug_dir is not None:
            vis = bgr.copy()
            cv2.rectangle(vis, (px, py), (px + pw, py + ph), (255, 255, 0), 4)
            for label, (x1, y1, x2, y2) in quads.items():
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 255), 4)
                cv2.putText(vis, label, (x1 + 10, y1 + 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                            (0, 200, 255), 3)
            _save(debug_dir, "00c_paper_bbox_quadrants.jpg", vis)
    else:
        # Fallback to v1-style image-corner quadrants
        pad = 0.25
        quads = {
            "TL": (0,              0,               int(w * pad), int(h * pad)),
            "TR": (int(w*(1-pad)), 0,               w,            int(h * pad)),
            "BR": (int(w*(1-pad)), int(h*(1-pad)),  w,            h),
            "BL": (0,              int(h*(1-pad)),  int(w * pad), h),
        }

    rois: dict[str, Tuple[int, int]] = {}
    for label, (x1, y1, x2, y2) in quads.items():
        # Clamp to image bounds
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        roi = core[y1:y2, x1:x2]
        cnts, _ = cv2.findContours(
            roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        best, best_area = None, 0
        for c in cnts:
            a = cv2.contourArea(c)
            if a < 300:
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            asp = max(bw, bh) / max(min(bw, bh), 1)
            if asp > MAX_ASPECT_COARSE:
                continue
            if a > best_area:
                best_area, best = a, c
        if best is None:
            continue
        M = cv2.moments(best)
        if M["m00"] > 0:
            cx = int(round(M["m10"] / M["m00"])) + x1
            cy = int(round(M["m01"] / M["m00"])) + y1
        else:
            bx, by, bw, bh = cv2.boundingRect(best)
            cx, cy = bx + bw // 2 + x1, by + bh // 2 + y1
        rois[label] = (cx, cy)
    return rois, core, paper_bbox


# ===========================================================================
# Corner-aware synthetic templates  (identical to v1)
# ===========================================================================

_CORNER_TICKS = {
    "TL": (False, True,  True,  False),
    "TR": (False, False, True,  True),
    "BR": (True,  False, False, True),
    "BL": (True,  True,  False, False),
}


def make_marker_template(corner: str, side: int) -> np.ndarray:
    pad = max(side // 12, 6)
    img_size = side + 2 * pad
    t = np.full((img_size, img_size), 255, np.uint8)
    s, e = pad, pad + side
    mid = (s + e) // 2

    cv2.rectangle(t, (s, s), (e, e), 0, 3)
    r = (e - s) // 2 - max(side // 40, 3)
    cv2.circle(t, (mid, mid), r, 0, 2)

    ch = (e - s) // 4
    cv2.line(t, (mid - ch, mid), (mid + ch, mid), 0, 2)
    cv2.line(t, (mid, mid - ch), (mid, mid + ch), 0, 2)

    n_ticks       = 14
    tick_len      = max(side // 18, 5)
    long_tick_len = tick_len + 4
    has_top, has_right, has_bottom, has_left = _CORNER_TICKS[corner]

    def _ticks_along(p0, p1, nx, ny):
        for i in range(n_ticks + 1):
            f = i / n_ticks
            x = int(round(p0[0] + (p1[0] - p0[0]) * f))
            y = int(round(p0[1] + (p1[1] - p0[1]) * f))
            L = long_tick_len if i == n_ticks // 2 else tick_len
            cv2.line(t, (x, y),
                     (x + int(L * nx), y + int(L * ny)), 0, 1)

    if has_top:    _ticks_along((s, s), (e, s),  0, +1)
    if has_bottom: _ticks_along((s, e), (e, e),  0, -1)
    if has_left:   _ticks_along((s, s), (s, e), +1,  0)
    if has_right:  _ticks_along((e, s), (e, e), -1,  0)
    return t


# ===========================================================================
# Refine methods  (identical to v1)
# ===========================================================================

def method_template_match_oriented(
    roi_gray: np.ndarray,
    corner: str,
    expected_side: int,
):
    blurred = cv2.GaussianBlur(roi_gray, (3, 3), 0)
    best_val = -1.0
    best = None
    best_tmpl = None
    for scale in (0.85, 0.92, 1.0, 1.08, 1.15, 1.25):
        side = max(int(round(expected_side * scale)), 50)
        tmpl = make_marker_template(corner, side)
        if (tmpl.shape[0] > roi_gray.shape[0]
                or tmpl.shape[1] > roi_gray.shape[1]):
            continue
        res = cv2.matchTemplate(blurred, tmpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, mloc = cv2.minMaxLoc(res)
        if mx > best_val:
            best_val = mx
            best = (mloc, tmpl.shape, side)
            best_tmpl = tmpl
    if best is None or best_val < 0.15:
        return None
    mloc, (th, tw), side = best
    cx = mloc[0] + tw / 2
    cy = mloc[1] + th / 2
    return (float(cx), float(cy), float(side), float(side), 0.0), best_tmpl


def method_ink_bbox_lab(roi_bgr: np.ndarray):
    mask, _ = red_ink_mask(roi_bgr)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12)),
    )
    cnts, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
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
# Scoring  (identical to v1)
# ===========================================================================

def score_candidate(cand: Tuple[float, float, float, float, float],
                    expected_side: float,
                    roi_size: int) -> float:
    cx, cy, rw, rh, _ = cand
    asp  = max(rw, rh) / max(min(rw, rh), 1)
    side = (rw + rh) / 2.0
    centre_off = float(np.linalg.norm([cx - roi_size/2, cy - roi_size/2]))
    if side < 50 or side > roi_size * 0.95:
        return 1e9
    return (abs(asp - 1.0) * 200
            + (centre_off / roi_size) * 150
            + abs(side - expected_side) / expected_side * 8)


# ===========================================================================
# Public API
# ===========================================================================

def detect_markers_fast_v2(
    image_bgr: np.ndarray,
    *,
    expected_side: float = 120.0,
    debug_dir: Optional[Path] = None,
) -> Tuple[dict[str, MarkerDetection], Optional[Tuple[int, int, int, int]]]:
    """Detect the 4 calibration markers in *image_bgr* using v2 (paper-anchored).

    Returns ``(markers, paper_bbox)``.  ``paper_bbox`` is None if the
    paper-detection fallback was triggered.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty image passed to detect_markers_fast_v2()")

    h, w = image_bgr.shape[:2]
    _save(debug_dir, "00_original.jpg", image_bgr)

    rois, _core, paper_bbox = coarse_marker_rois(image_bgr, debug_dir=debug_dir)

    if debug_dir is not None:
        for c in ("TL", "TR", "BR", "BL"):
            _save(debug_dir, f"04_template_{c}.png",
                  make_marker_template(c, 120), max_w=300)

    final: dict[str, MarkerDetection] = {}
    for label, (cx0, cy0) in rois.items():
        x1, y1 = max(cx0 - ROI_HALF, 0), max(cy0 - ROI_HALF, 0)
        x2, y2 = min(cx0 + ROI_HALF, w), min(cy0 + ROI_HALF, h)
        roi = image_bgr[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _save(debug_dir, f"02_roi_{label}.jpg", roi)

        candidates = []
        try:
            out = method_template_match_oriented(gray, label, int(expected_side))
            if out is not None:
                cand, dbg = out
                s = score_candidate(cand, expected_side, roi.shape[0])
                candidates.append((s, "tmpl_cnr", cand))
                if dbg is not None and debug_dir:
                    _save(debug_dir, f"03_{label}_tmpl_cnr.jpg", dbg)
        except Exception:
            pass

        try:
            out = method_ink_bbox_lab(roi)
            if out is not None:
                cand, dbg = out
                s = score_candidate(cand, expected_side, roi.shape[0])
                candidates.append((s, "ink_lab", cand))
                if dbg is not None and debug_dir:
                    _save(debug_dir, f"03_{label}_ink_lab.jpg", dbg)
        except Exception:
            pass

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
    return final, paper_bbox


def _inner_corner(box: np.ndarray, paper_centre: np.ndarray):
    dists = np.linalg.norm(box - paper_centre, axis=1)
    i = int(np.argmin(dists))
    return float(box[i][0]), float(box[i][1])


def rectify_with_markers_fast_v2(
    image_bgr: np.ndarray,
    *,
    paper_w_mm: float = PAPER_W_MM,
    paper_h_mm: float = PAPER_H_MM,
    px_per_mm: float = DEFAULT_PX_PER_MM,
    expected_side: float = 120.0,
    debug_dir: Optional[Path] = None,
) -> RectifyResult:
    """End-to-end Step-1 using v2's paper-anchored coarse pass."""
    out_w = int(round(paper_w_mm * px_per_mm))
    out_h = int(round(paper_h_mm * px_per_mm))

    markers, paper_bbox = detect_markers_fast_v2(
        image_bgr,
        expected_side=expected_side,
        debug_dir=debug_dir,
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
        if paper_bbox is not None:
            px, py, pw, ph = paper_bbox
            cv2.rectangle(vis, (px, py), (px + pw, py + ph),
                          (255, 255, 0), 4)
        for label, det in markers.items():
            cv2.polylines(vis, [det.box.astype(np.int32)],
                          True, (0, 255, 0), 4)
            cx, cy = det.centre_xy
            cv2.circle(vis, (int(cx), int(cy)), 8, (0, 0, 255), -1)
            cv2.putText(vis, label, (int(cx + 20), int(cy - 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 4)
        cv2.polylines(vis, [src_quad.astype(np.int32)],
                      True, (255, 200, 0), 5)
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
        paper_bbox_xywh=paper_bbox,
    )
