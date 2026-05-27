"""Marker-based rectification — FAST v3 (global candidate detection).

Designed to be **background-independent**.  v2 anchored quadrants on
the paper bounding box, which only works when paper is the most
prominent bright region.  In production we may see:

    * paper on a white tabletop                  -> paper not bright vs bg
    * paper on top of another paper              -> two bright blobs
    * mixed lighting / shadows                   -> Otsu mis-segments
    * angled paper extending outside the frame   -> bbox truncated

In every case though, the 4 red calibration markers are reliably
present at the corners of the printed border.  v3 uses ONLY those
markers as the anchor:

    A. Lab delta-a mask of the FULL image (lighting invariant).
    B. Close + open to leave clean marker cores while erasing grid lines.
    C. Find ALL connected components and filter to marker-shaped blobs:
         * area in [3000, 60000] px (covers ~13 mm to ~30 mm at 3–5 K wide photos)
         * minAreaRect aspect ratio <= 1.5  (allows mild perspective)
         * solidity >= 0.70                  (rejects L-shapes from grid joining)
         * bbox at least N px away from image edge
    D. Of the surviving candidates, pick the 4 that best form a quadrilateral:
         * compute centroid of all candidates
         * partition by position (TL/TR/BR/BL quadrant relative to centroid)
         * within each quadrant, prefer the candidate FURTHEST from centroid
           (corners, not interior elements)
    E. Refine each with the unchanged template + ink_lab pair from v1/v2.

This module reuses the synthetic-template + ink_lab refine code by
defining them locally (same algorithm as v1) so there's no cross-file
coupling that could be broken by future tweaks elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Lab / morph constants — unchanged from v1
# ---------------------------------------------------------------------------

DELTA_A_MIN      = 12
ORANGE_TOLERANCE = 6
PAPER_PATCH_FRAC = 0.10
PAPER_BRIGHT_PCT = 75

COARSE_CLOSE_KSIZE = 18
COARSE_OPEN_KSIZE  = 15
ROI_HALF           = 110

# ---------------------------------------------------------------------------
# Global-candidate filter parameters
# ---------------------------------------------------------------------------

# Marker physical size = 15 mm.  In 4–5 K wide photos that's typically
# 100–160 px across, so area ≈ 10 000 – 25 000 px.  Allow generous slack
# for closer / further shots.
CAND_MIN_AREA_PX     = 3000      # filters out specks
CAND_MAX_AREA_PX     = 60000     # filters out big regions (text, marks)

# minAreaRect aspect ratio limit (perspective allowance)
CAND_MAX_ASPECT      = 1.6

# Solidity (contour area / hull area) — rejects L-shapes from grid joining
CAND_MIN_SOLIDITY    = 0.65

# Distance from image edge — markers can be near the edge but not AT it.
# Allow them within EDGE_TOLERANCE_PX of the image bounds.
EDGE_TOLERANCE_PX    = 4

# When too many candidates survive the filters, keep the N largest by
# area for downstream quadrant assignment.
MAX_CANDIDATES_KEPT  = 30

# ---------------------------------------------------------------------------
# Physical reference for the warp target
# ---------------------------------------------------------------------------

MARKER_MM         = 15.0
PAPER_W_MM        = 345.0
PAPER_H_MM        = 255.0
DEFAULT_PX_PER_MM = 10.0


# ---------------------------------------------------------------------------
# Return shapes — same as v1/v2 so the dev runners can share code
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
    # Diagnostic: how many marker candidates survived global filtering
    candidate_count: int = 0


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
# Lab paper-baseline + red-ink mask (identical to v1)
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
# A. Global candidate detection — find ALL marker-shaped red blobs
# ===========================================================================

@dataclass
class _Candidate:
    contour: np.ndarray
    centre_xy: Tuple[float, float]
    minrect_size: Tuple[float, float]
    minrect_angle: float
    area: float
    solidity: float


def find_marker_candidates(
    bgr: np.ndarray,
    debug_dir: Optional[Path] = None,
) -> list[_Candidate]:
    """Return all blobs in the full image that look like marker squares.

    Uses a TIERED area threshold: tries the strictest first, relaxes
    until at least 4 candidates survive (or we run out of options).
    A faded / partially-detected marker can shrink to 200-500 px after
    the morph chain, but every other filter (aspect, solidity, edge
    distance) keeps random small blobs out.
    """
    mask, _paper_lab = red_ink_mask(bgr)
    k_close = cv2.getStructuringElement(
        cv2.MORPH_RECT, (COARSE_CLOSE_KSIZE, COARSE_CLOSE_KSIZE))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)
    k_open = cv2.getStructuringElement(
        cv2.MORPH_RECT, (COARSE_OPEN_KSIZE, COARSE_OPEN_KSIZE))
    core = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k_open, iterations=1)
    _save(debug_dir, "01a_red_ink_mask.jpg", mask)
    _save(debug_dir, "01b_marker_core.jpg", core)

    h, w = bgr.shape[:2]
    cnts, _ = cv2.findContours(core, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

    # Pre-compute each contour's properties once
    rows = []
    for c in cnts:
        area = float(cv2.contourArea(c))
        if area > CAND_MAX_AREA_PX:
            continue
        rect = cv2.minAreaRect(c)
        (cx, cy), (rw, rh), ang = rect
        if rw == 0 or rh == 0:
            continue
        aspect = max(rw, rh) / min(rw, rh)
        if aspect > CAND_MAX_ASPECT:
            continue
        hull = cv2.convexHull(c)
        hull_area = float(cv2.contourArea(hull)) or 1.0
        solidity = area / hull_area
        if solidity < CAND_MIN_SOLIDITY:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if (x < EDGE_TOLERANCE_PX or y < EDGE_TOLERANCE_PX
                or x + bw > w - EDGE_TOLERANCE_PX
                or y + bh > h - EDGE_TOLERANCE_PX):
            continue
        rows.append((area, c, cx, cy, rw, rh, ang, solidity))

    # Try strict-first, then relax the area floor until we have at least 4.
    # If we get >= 4 at the strict threshold we use them; we only loosen if
    # the strict pass under-detects.
    AREA_THRESHOLDS = [CAND_MIN_AREA_PX, 1500, 800, 400, 200]
    chosen_rows = []
    for threshold in AREA_THRESHOLDS:
        chosen_rows = [r for r in rows if r[0] >= threshold]
        if len(chosen_rows) >= 4:
            break

    survivors: list[_Candidate] = []
    for area, c, cx, cy, rw, rh, ang, solidity in chosen_rows:
        survivors.append(_Candidate(
            contour=c,
            centre_xy=(float(cx), float(cy)),
            minrect_size=(float(rw), float(rh)),
            minrect_angle=float(ang),
            area=area,
            solidity=solidity,
        ))
    survivors.sort(key=lambda s: s.area, reverse=True)
    survivors = survivors[:MAX_CANDIDATES_KEPT]

    if debug_dir is not None:
        vis = bgr.copy()
        for s in survivors:
            box = cv2.boxPoints(((s.centre_xy[0], s.centre_xy[1]),
                                  s.minrect_size, s.minrect_angle))
            cv2.polylines(vis, [box.astype(np.int32)],
                          True, (0, 255, 0), 3)
            cv2.circle(vis, (int(s.centre_xy[0]), int(s.centre_xy[1])),
                       8, (0, 255, 255), -1)
            cv2.putText(vis,
                        f"a={s.area:.0f} s={s.solidity:.2f}",
                        (int(s.centre_xy[0]) - 40, int(s.centre_xy[1]) - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        _save(debug_dir, "01c_candidates.jpg", vis)

    return survivors


def assign_corner_labels(
    candidates: list[_Candidate],
) -> dict[str, _Candidate]:
    """Assign TL/TR/BR/BL labels to the 4 best candidates.

    Strategy:
        1. Compute centroid of all candidate centres.
        2. Partition candidates into 4 quadrants by signs of (x - cx, y - cy).
        3. In each quadrant, pick the candidate FURTHEST from the centroid
           (corners should be furthest out; interior blobs sit near centre).
    """
    if len(candidates) < 4:
        return {}
    centres = np.array([c.centre_xy for c in candidates], dtype=np.float32)
    cx_all = float(centres[:, 0].mean())
    cy_all = float(centres[:, 1].mean())

    buckets: dict[str, list[_Candidate]] = {
        "TL": [], "TR": [], "BR": [], "BL": [],
    }
    for cand in candidates:
        x, y = cand.centre_xy
        if x <= cx_all and y <= cy_all:
            buckets["TL"].append(cand)
        elif x > cx_all and y <= cy_all:
            buckets["TR"].append(cand)
        elif x > cx_all and y > cy_all:
            buckets["BR"].append(cand)
        else:
            buckets["BL"].append(cand)

    chosen: dict[str, _Candidate] = {}
    for label, items in buckets.items():
        if not items:
            continue
        # furthest from the candidate centroid
        items.sort(
            key=lambda c: (c.centre_xy[0] - cx_all) ** 2
                          + (c.centre_xy[1] - cy_all) ** 2,
            reverse=True,
        )
        chosen[label] = items[0]
    return chosen


# ===========================================================================
# B. Synthetic templates + refine methods (identical to v1)
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


def score_candidate(cand, expected_side, roi_size):
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

def detect_markers_fast_v3(
    image_bgr: np.ndarray,
    *,
    expected_side: float = 120.0,
    debug_dir: Optional[Path] = None,
) -> Tuple[dict[str, MarkerDetection], int]:
    """Detect the 4 calibration markers via the global-candidate strategy.

    Returns ``(markers, n_candidates_found)``.  ``n_candidates_found``
    is the number of marker-shaped blobs that survived global filtering
    -- if ``< 4`` the algorithm couldn't produce a complete result.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty image passed to detect_markers_fast_v3()")

    h, w = image_bgr.shape[:2]
    _save(debug_dir, "00_original.jpg", image_bgr)

    candidates = find_marker_candidates(image_bgr, debug_dir=debug_dir)
    if len(candidates) < 4:
        return {}, len(candidates)

    chosen = assign_corner_labels(candidates)

    # Estimate expected_side from the chosen blobs themselves: average
    # of their minAreaRect sizes is a great metric-aware seed for the
    # multi-scale template ladder.
    side_estimates = [
        (s.minrect_size[0] + s.minrect_size[1]) / 2.0
        for s in chosen.values()
    ]
    if side_estimates:
        expected_side = float(np.median(side_estimates))

    if debug_dir is not None:
        vis = image_bgr.copy()
        for label, c in chosen.items():
            cx, cy = c.centre_xy
            cv2.circle(vis, (int(cx), int(cy)), 14, (0, 0, 255), -1)
            cv2.putText(vis, label, (int(cx) + 18, int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 4)
        _save(debug_dir, "01d_chosen_4.jpg", vis)
        for c in ("TL", "TR", "BR", "BL"):
            _save(debug_dir, f"04_template_{c}.png",
                  make_marker_template(c, int(expected_side)), max_w=300)

    final: dict[str, MarkerDetection] = {}
    for label, cand in chosen.items():
        cx0, cy0 = cand.centre_xy
        x1, y1 = max(int(cx0) - ROI_HALF, 0), max(int(cy0) - ROI_HALF, 0)
        x2, y2 = min(int(cx0) + ROI_HALF, w), min(int(cy0) + ROI_HALF, h)
        roi = image_bgr[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _save(debug_dir, f"02_roi_{label}.jpg", roi)

        refined: list[Tuple[float, str, Tuple[float, float, float, float, float]]] = []

        try:
            out = method_template_match_oriented(gray, label, int(expected_side))
            if out is not None:
                cand_t, dbg = out
                s = score_candidate(cand_t, expected_side, roi.shape[0])
                refined.append((s, "tmpl_cnr", cand_t))
                if dbg is not None and debug_dir:
                    _save(debug_dir, f"03_{label}_tmpl_cnr.jpg", dbg)
        except Exception:
            pass

        try:
            out = method_ink_bbox_lab(roi)
            if out is not None:
                cand_i, dbg = out
                s = score_candidate(cand_i, expected_side, roi.shape[0])
                refined.append((s, "ink_lab", cand_i))
                if dbg is not None and debug_dir:
                    _save(debug_dir, f"03_{label}_ink_lab.jpg", dbg)
        except Exception:
            pass

        if not refined:
            # Fallback: use the original candidate's minAreaRect directly.
            (rw, rh) = cand.minrect_size
            if rw < rh:
                rw, rh = rh, rw
                ang = cand.minrect_angle + 90
            else:
                ang = cand.minrect_angle
            full_cx, full_cy = cand.centre_xy
            box = cv2.boxPoints(((full_cx, full_cy), (rw, rh), ang))
            final[label] = MarkerDetection(
                label=label,
                centre_xy=(full_cx, full_cy),
                size_wh=(rw, rh),
                angle_deg=ang,
                method="raw_candidate",
                score=99.0,
                box=box,
            )
            continue

        refined.sort(key=lambda t: t[0])
        best_score, best_name, best_cand = refined[0]
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

    return final, len(candidates)


# ===========================================================================
# Inner-corner extraction + perspective warp (identical to v1)
# ===========================================================================

def _inner_corner(box: np.ndarray, paper_centre: np.ndarray):
    dists = np.linalg.norm(box - paper_centre, axis=1)
    i = int(np.argmin(dists))
    return float(box[i][0]), float(box[i][1])


def rectify_with_markers_fast_v3(
    image_bgr: np.ndarray,
    *,
    paper_w_mm: float = PAPER_W_MM,
    paper_h_mm: float = PAPER_H_MM,
    px_per_mm: float = DEFAULT_PX_PER_MM,
    expected_side: float = 120.0,
    debug_dir: Optional[Path] = None,
) -> RectifyResult:
    out_w = int(round(paper_w_mm * px_per_mm))
    out_h = int(round(paper_h_mm * px_per_mm))

    markers, n_candidates = detect_markers_fast_v3(
        image_bgr, expected_side=expected_side, debug_dir=debug_dir,
    )
    missing = {"TL", "TR", "BR", "BL"} - markers.keys()
    if missing:
        raise RuntimeError(
            f"Required markers missing: {sorted(missing)}.  Detected: "
            f"{sorted(markers.keys())}  (candidates found: {n_candidates})"
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
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
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
        candidate_count=n_candidates,
    )
