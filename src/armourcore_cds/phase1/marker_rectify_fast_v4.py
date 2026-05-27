"""Marker-based rectification — FAST v4 (geometric rescue for weak markers).

v3 reliably finds 4 candidates on every corpus image, but on
BLUE_PENCIL_CREASED_01 the TL marker is partly in shadow and the
morph chain reduces it to a small (~400 px) fragment.  The fragment's
centroid sits on the bright remnant, NOT on the marker's true centre,
which throws the rectification quad off.

v4 only changes the path AFTER global candidate detection has chosen
4 markers.  For each chosen marker it asks:

    "Is this candidate's area suspiciously small compared with the
     other three?  If yes, treat its centroid as unreliable and:
         (1) predict where it SHOULD be using parallelogram closure
             from the other 3 strong markers
         (2) re-search a generous ROI anchored on the predicted spot
             with the corner-aware template match and the ink_lab
             fallback (same methods as v3, just bigger neighbourhood)."

Every case that v3 already handled flows through unchanged because
their candidate areas are similar to each other (no suspect flag).

Returns the same shapes as v1/v2/v3 so the dev runner can call it
identically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Tuning constants (Lab + morph + global filter identical to v3)
# ---------------------------------------------------------------------------

DELTA_A_MIN      = 12
ORANGE_TOLERANCE = 6
PAPER_PATCH_FRAC = 0.10
PAPER_BRIGHT_PCT = 75

COARSE_CLOSE_KSIZE = 18
COARSE_OPEN_KSIZE  = 15

CAND_MIN_AREA_PX     = 3000
CAND_MAX_AREA_PX     = 60000
CAND_MAX_ASPECT      = 1.6
CAND_MIN_SOLIDITY    = 0.65
EDGE_TOLERANCE_PX    = 4
MAX_CANDIDATES_KEPT  = 30

# Standard refine ROI (px on either side of the seed centroid)
ROI_HALF             = 110

# --- NEW for v4 ---
# Area-ratio at which a chosen marker is considered "suspect" relative to
# the other three.  A ratio of 0.40 means the smallest must be at least
# 40% of the median to be trusted; less than that triggers the
# geometric-rescue path.
SUSPECT_AREA_RATIO          = 0.40

# When a marker is suspect, we re-search around its geometrically
# predicted position with this larger ROI half-size.  220 px gives a
# 440x440 search window — enough to absorb a ~100 px prediction error
# and still keep the actual ~120 px marker inside.
SUSPECT_ROI_HALF            = 220

# When predicting the position from 3 strong corners, the predicted
# position must be inside the image with this much margin.  If not,
# the prediction is rejected (presumably caused by mis-labelling).
PREDICTION_INSET_PX         = 60


MARKER_MM         = 15.0
PAPER_W_MM        = 345.0
PAPER_H_MM        = 255.0
DEFAULT_PX_PER_MM = 10.0


# ---------------------------------------------------------------------------
# Return shapes — same as v1/v2/v3
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
    candidate_count: int = 0
    # Diagnostic: which labels (if any) were geometrically rescued
    rescued_labels: list[str] = field(default_factory=list)


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
# Lab paper-baseline + red-ink mask (identical to v3)
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
# Global candidate detection (identical to v3, with progressive relaxation)
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
    cnts, _ = cv2.findContours(
        core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )

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
                        f"a={s.area:.0f}",
                        (int(s.centre_xy[0]) - 40, int(s.centre_xy[1]) - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        _save(debug_dir, "01c_candidates.jpg", vis)
    return survivors


def assign_corner_labels(
    candidates: list[_Candidate],
) -> dict[str, _Candidate]:
    """Bucket candidates into TL / TR / BR / BL quadrants by position
    relative to the candidate centroid, then pick the LARGEST in each
    quadrant.

    Why "largest" and not "furthest from centroid" (v3 logic)?
    The progressive area-relaxation in ``find_marker_candidates`` lets
    in small fragments (~300-700 px) that come from the morph open
    chipping ruler-tick lines off the marker body.  Those fragments
    sit slightly OUTSIDE the real marker centres, so they're
    further-from-centroid than the actual ~10 000-px marker bodies.
    Furthest-from-centroid would mis-select them as the marker.

    Largest-in-quadrant correctly grabs the big marker body whenever
    it survives the morph chain.  In the edge case where the real
    marker is reduced to a small fragment (PCC01 TL with deep
    shadow) the only candidate in that quadrant IS the small one,
    so it's still chosen -- and the suspect-area check downstream
    will flag it for geometric rescue.
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
        # Largest by area wins; tie-break by furthest-from-centroid
        items.sort(
            key=lambda c: (
                c.area,
                (c.centre_xy[0] - cx_all) ** 2 + (c.centre_xy[1] - cy_all) ** 2,
            ),
            reverse=True,
        )
        chosen[label] = items[0]
    return chosen


# ===========================================================================
# Templates + refine methods (identical to v3)
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
# NEW for v4: geometric prediction + per-corner refine helper
# ===========================================================================

def predict_corner_position(
    known: dict[str, Tuple[float, float]],
    target_label: str,
) -> Optional[Tuple[float, float]]:
    """Predict where ``target_label`` should sit, given the other 3 corners.

    Uses parallelogram closure -- exact for a parallelogram, close enough
    for a mildly perspective-warped rectangle.

        target = adjacent_A + adjacent_B  -  opposite

    Returns None if any of the 3 needed corners is missing from *known*.
    """
    # For each target, identify the OPPOSITE corner (across the diagonal)
    # and the two ADJACENT corners.  In the rectangle TL-TR-BR-BL clockwise:
    #   * TL opposite BR, adjacent TR + BL
    #   * TR opposite BL, adjacent TL + BR
    #   * BR opposite TL, adjacent TR + BL  (note: BL and TR are diagonals
    #                                         of the OTHER diagonal -- BR
    #                                         is the "free" corner of the
    #                                         TL-TR-BR or TL-BL-BR triple)
    # Concretely the closure formula is target = A + B - opposite where
    # A, B are the two non-opposite corners.
    opposites = {"TL": "BR", "TR": "BL", "BR": "TL", "BL": "TR"}
    op = opposites[target_label]
    others = [k for k in ("TL", "TR", "BR", "BL")
              if k not in (target_label, op)]
    if op not in known or any(k not in known for k in others):
        return None
    a = np.asarray(known[others[0]], dtype=np.float64)
    b = np.asarray(known[others[1]], dtype=np.float64)
    o = np.asarray(known[op], dtype=np.float64)
    pred = a + b - o
    return (float(pred[0]), float(pred[1]))


def _refine_marker_in_roi(
    image_bgr: np.ndarray,
    label: str,
    seed_xy: Tuple[float, float],
    roi_half: int,
    expected_side: float,
    debug_dir: Optional[Path],
    debug_tag: str,
) -> Optional[Tuple[Tuple[float, float, float, float, float], str, float]]:
    """Run the standard template + ink_lab refine at *seed_xy* with the
    given ROI half-size.

    Returns ``((cx, cy, rw, rh, ang), method_name, score)`` in full-image
    coords, or None if both methods fail.
    """
    h, w = image_bgr.shape[:2]
    cx0, cy0 = seed_xy
    x1, y1 = max(int(cx0) - roi_half, 0), max(int(cy0) - roi_half, 0)
    x2, y2 = min(int(cx0) + roi_half, w), min(int(cy0) + roi_half, h)
    if x2 <= x1 or y2 <= y1:
        return None
    roi = image_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    if debug_dir is not None:
        _save(debug_dir, f"02_roi_{label}{debug_tag}.jpg", roi)

    refined: list[Tuple[float, str, Tuple[float, float, float, float, float]]] = []
    try:
        out = method_template_match_oriented(gray, label, int(expected_side))
        if out is not None:
            cand_t, dbg = out
            s = score_candidate(cand_t, expected_side, roi.shape[0])
            refined.append((s, "tmpl_cnr", cand_t))
            if dbg is not None and debug_dir:
                _save(debug_dir,
                      f"03_{label}{debug_tag}_tmpl_cnr.jpg", dbg)
    except Exception:
        pass
    try:
        out = method_ink_bbox_lab(roi)
        if out is not None:
            cand_i, dbg = out
            s = score_candidate(cand_i, expected_side, roi.shape[0])
            refined.append((s, "ink_lab", cand_i))
            if dbg is not None and debug_dir:
                _save(debug_dir,
                      f"03_{label}{debug_tag}_ink_lab.jpg", dbg)
    except Exception:
        pass
    if not refined:
        return None
    refined.sort(key=lambda t: t[0])
    best_score, best_name, best_cand = refined[0]
    cx_local, cy_local, rw, rh, ang = best_cand
    return ((cx_local + x1, cy_local + y1, rw, rh, ang),
            best_name, best_score)


# ===========================================================================
# Public API
# ===========================================================================

def detect_markers_fast_v4(
    image_bgr: np.ndarray,
    *,
    expected_side: float = 120.0,
    debug_dir: Optional[Path] = None,
) -> Tuple[dict[str, MarkerDetection], int, list[str]]:
    """Detect markers via global candidates + geometric rescue.

    Returns ``(markers, n_candidates, rescued_labels)``.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty image passed to detect_markers_fast_v4()")

    h, w = image_bgr.shape[:2]
    _save(debug_dir, "00_original.jpg", image_bgr)

    candidates = find_marker_candidates(image_bgr, debug_dir=debug_dir)
    if len(candidates) < 4:
        return {}, len(candidates), []

    chosen = assign_corner_labels(candidates)
    if len(chosen) < 4:
        return {}, len(candidates), []

    # ---- 1) Identify suspect candidates by area ratio ------------------
    areas = [c.area for c in chosen.values()]
    median_area = float(np.median(areas))
    suspect: dict[str, bool] = {
        label: (c.area / median_area) < SUSPECT_AREA_RATIO
        for label, c in chosen.items()
    }

    # Estimate expected_side from the NON-suspect candidates only --
    # using a partial detection's size would underestimate.
    side_estimates_strong = [
        (c.minrect_size[0] + c.minrect_size[1]) / 2.0
        for label, c in chosen.items()
        if not suspect[label]
    ]
    if side_estimates_strong:
        expected_side = float(np.median(side_estimates_strong))

    rescued_labels: list[str] = []
    final: dict[str, MarkerDetection] = {}

    # ---- 2) Refine the strong candidates first ------------------------
    # Their results feed the geometric prediction for the suspect ones.
    strong_centres: dict[str, Tuple[float, float]] = {}
    pending_suspect: list[str] = []
    for label, cand in chosen.items():
        if suspect[label]:
            pending_suspect.append(label)
            continue
        refined = _refine_marker_in_roi(
            image_bgr, label, cand.centre_xy, ROI_HALF,
            expected_side, debug_dir, debug_tag="",
        )
        if refined is None:
            # Fallback: keep the raw candidate
            (rw, rh) = cand.minrect_size
            ang = cand.minrect_angle
            if rw < rh:
                rw, rh = rh, rw
                ang += 90
            cx, cy = cand.centre_xy
            box = cv2.boxPoints(((cx, cy), (rw, rh), ang))
            final[label] = MarkerDetection(
                label=label, centre_xy=(cx, cy),
                size_wh=(rw, rh), angle_deg=ang,
                method="raw_candidate", score=99.0, box=box,
            )
            strong_centres[label] = (cx, cy)
            continue
        (cx, cy, rw, rh, ang), method, score = refined
        box = cv2.boxPoints(((cx, cy), (rw, rh), ang))
        final[label] = MarkerDetection(
            label=label, centre_xy=(cx, cy),
            size_wh=(rw, rh), angle_deg=ang,
            method=method, score=score, box=box,
        )
        strong_centres[label] = (cx, cy)

    # ---- 3) Rescue suspect ones via parallelogram closure -------------
    for label in pending_suspect:
        cand = chosen[label]
        predicted = predict_corner_position(strong_centres, label)
        seed: Tuple[float, float]
        if (predicted is not None
                and PREDICTION_INSET_PX <= predicted[0] <= w - PREDICTION_INSET_PX
                and PREDICTION_INSET_PX <= predicted[1] <= h - PREDICTION_INSET_PX):
            seed = predicted
            rescued_labels.append(label)
        else:
            # No reliable prediction -> fall back to the raw candidate centroid
            seed = cand.centre_xy

        refined = _refine_marker_in_roi(
            image_bgr, label, seed, SUSPECT_ROI_HALF,
            expected_side, debug_dir, debug_tag="_rescue",
        )

        if refined is None:
            # Last resort: synthesise a marker box using the predicted
            # position and the strong-marker median size.
            cx, cy = seed
            side = expected_side
            ang = 0.0
            box = cv2.boxPoints(((cx, cy), (side, side), ang))
            final[label] = MarkerDetection(
                label=label, centre_xy=(cx, cy),
                size_wh=(side, side), angle_deg=ang,
                method="predicted_only", score=99.0, box=box,
            )
            continue

        (cx, cy, rw, rh, ang), method, score = refined
        box = cv2.boxPoints(((cx, cy), (rw, rh), ang))
        final[label] = MarkerDetection(
            label=label, centre_xy=(cx, cy),
            size_wh=(rw, rh), angle_deg=ang,
            method=method + "_rescue", score=score, box=box,
        )

    if debug_dir is not None:
        # Final composite overlay (after rescue if any)
        vis = image_bgr.copy()
        for label, det in final.items():
            cv2.polylines(vis, [det.box.astype(np.int32)],
                          True, (0, 255, 0), 4)
            cx, cy = det.centre_xy
            cv2.circle(vis, (int(cx), int(cy)), 12, (0, 0, 255), -1)
            tag = f"{label} [{det.method}]"
            if label in rescued_labels:
                tag += " *RESCUED*"
            cv2.putText(vis, tag, (int(cx) + 18, int(cy) - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                        (0, 255, 0), 4)
        _save(debug_dir, "01e_final_after_rescue.jpg", vis)

    return final, len(candidates), rescued_labels


# ===========================================================================
# Inner-corner extraction + perspective warp (identical to v3)
# ===========================================================================

def _inner_corner(box: np.ndarray, paper_centre: np.ndarray):
    dists = np.linalg.norm(box - paper_centre, axis=1)
    i = int(np.argmin(dists))
    return float(box[i][0]), float(box[i][1])


def rectify_with_markers_fast_v4(
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

    markers, n_candidates, rescued = detect_markers_fast_v4(
        image_bgr, expected_side=expected_side, debug_dir=debug_dir,
    )
    missing = {"TL", "TR", "BR", "BL"} - markers.keys()
    if missing:
        raise RuntimeError(
            f"Required markers missing: {sorted(missing)}.  Detected: "
            f"{sorted(markers.keys())}  (candidates: {n_candidates})"
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
        rescued_labels=rescued,
    )
