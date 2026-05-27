"""Marker-based rectification (new Step-1 method).

A REWRITE of Step 1 (CDS border detection + perspective rectification)
that does NOT try to find the printed border as a single quadrilateral.
Instead it locates the four small printed CALIBRATION MARKERS (15x15 mm
fiducials at each corner of the CDS sheet) independently, then uses the
INNER corner of each marker (the one facing the page centre) as the
four corners of the rectification quad.

Why this is better than ``boundary_detection.py``
-------------------------------------------------
* Markers are small, high-contrast, in known quadrants -> easy to find
  even when creases / lighting break the full-border colour mask.
* Each corner is detected INDEPENDENTLY via 7 different methods scored
  on aspect-ratio + ROI-centre alignment + expected size, so a single
  bad marker doesn't sink the whole detection.
* No reliance on the four sides of the printed rectangle being
  continuous or evenly lit.

Approach (per marker)
---------------------
1. **Coarse pass** — HSV non-paper-hue mask (or adaptive threshold in
   grayscale-ish images) in each of the four image-corner quadrants.
   Pick the most square-ish blob as the rough ROI centre.
2. **Refine pass** — within a tight ROI around the rough centre, try
   seven shape-based methods and score each candidate:
     * ``ink_bbox``     min-area rect of all non-paper pixels (colour)
     * ``dark_bbox``    min-area rect of dark pixels (gray)
     * ``adaptive``     adaptive threshold + 4-point contour
     * ``canny``        Canny edges + min-area rect
     * ``template``     synthetic crosshair-square template match
     * ``hough``        4 perpendicular Hough lines -> intersections
     * ``harris``       Harris corner clustering -> min-area rect
3. **Score** each candidate (lower=better):
     ``200*|aspect-1| + 150*centre_offset/roi_size + 8*size_deviation``
4. **Take best per corner**, return centre + size + angle + method tag.

Then ``rectify_with_markers`` builds the inner-corner quad and warps
to an exact ``paper_w_mm x paper_h_mm`` rectangle at the requested
``px_per_mm``.

Adapted verbatim from the working module at
``Working Modules/WORKING CODE DONT TOUCH/detect_markers.py`` and
``rectify_corner_border.py``.  Algorithm is unchanged; this version
just removes module-level globals and the IMAGE_PATH constant so the
function can be called as a library.

The original ``boundary_detection.py`` remains in place and unused
pending the end-of-project cleanup pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Physical reference constants (matched to the current blue-orange CDS print)
# ---------------------------------------------------------------------------

MARKER_MM      = 15.0          # marker side length, mm
PAPER_W_MM     = 345.0         # printable region between inner marker corners (mm)
PAPER_H_MM     = 255.0
DEFAULT_PX_PER_MM = 10.0       # output resolution -> 3450 x 2550 px


# ROI half-size around the coarse detection point (px).  Markers are
# ~130 px so 110 is tight but leaves room for marker position error.
ROI_HALF = 110


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MarkerDetection:
    """One marker's final detection."""
    label: str                         # "TL" / "TR" / "BR" / "BL"
    centre_xy: tuple[float, float]
    size_wh: tuple[float, float]
    angle_deg: float
    method: str
    score: float
    box: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((4, 2)))

    @property
    def inner_corner_xy(self) -> tuple[float, float]:
        """Convenience: returns this marker's inner corner once
        ``compute_inner_corner`` has been called for a page-centre."""
        return self.centre_xy   # populated post-detection; replaced below


@dataclass
class RectifyResult:
    """Result of marker detection + perspective warp."""
    warped: np.ndarray
    markers: dict[str, MarkerDetection]
    inner_corners_xy: dict[str, tuple[float, float]]   # original-image coords
    src_quad: np.ndarray                                # (4, 2) TL TR BR BL
    homography: np.ndarray                              # 3x3
    output_size_px: tuple[int, int]                     # (w, h)
    paper_size_mm: tuple[float, float]                  # (w_mm, h_mm)
    px_per_mm: float


# ---------------------------------------------------------------------------
# Internal: save helper used only when a debug_dir is provided
# ---------------------------------------------------------------------------

def _save(debug_dir: Path | None,
          name: str,
          img: np.ndarray,
          *,
          max_w: int = 2000) -> None:
    """If ``debug_dir`` is provided, resize-and-save into it.  Otherwise no-op."""
    if debug_dir is None:
        return
    debug_dir.mkdir(parents=True, exist_ok=True)
    h, w = img.shape[:2]
    if w > max_w:
        s = max_w / w
        img = cv2.resize(img, (int(w * s), int(h * s)))
    cv2.imwrite(str(debug_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])


# ===========================================================================
# A. COARSE PASS — rough quadrant search for each marker centre
# ===========================================================================

def _coarse_marker_rois(bgr: np.ndarray,
                        debug_dir: Path | None = None) -> dict[str, tuple[int, int]]:
    """Return ``{"TL": (cx, cy), "TR": ..., "BR": ..., "BL": ...}``.

    Strategy: non-paper-hue HSV mask (or adaptive threshold for grayscale
    images), then within each corner quadrant pick the largest blob with
    aspect ratio under 2.5 (rejects edge / paper-corner artefacts).
    """
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mean_sat = float(hsv[:, :, 1].mean())
    use_colour = mean_sat > 25

    if use_colour:
        hue, sat, brt = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        # Marker mask = "chromatic ink that isn't blue".
        #
        #   * Original mask had no saturation floor, so when the page
        #     itself was warm (sun-lit, hue near yellow/red) it ALSO
        #     passed the "not-blue" hue test and the discriminator
        #     collapsed.  Adding sat > 60 cleanly separates the
        #     printed-ink markers (saturated) from warm-tinted paper
        #     (sat < ~40 even in strong sun).
        #   * Brightness floor widened from 130 -> 60 so dark-red /
        #     deep-shadow markers still pass.  Top capped from 240 ->
        #     250 to keep bright paper out without trimming the
        #     brightest marker pixels.
        not_blue        = (hue < 95) | (hue > 125)
        high_saturation = sat > 60
        brightness_ok   = (brt > 60) & (brt < 250)
        mask = (not_blue & high_saturation
                & brightness_ok).astype(np.uint8) * 255
    else:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 51, 8,
        )

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    _save(debug_dir, "01_coarse_mask.jpg", mask)

    pad = 0.25
    quadrants = {
        "TL": (0,              0,               int(w * pad), int(h * pad)),
        "TR": (int(w * (1 - pad)), 0,               w,            int(h * pad)),
        "BR": (int(w * (1 - pad)), int(h * (1 - pad)), w,         h),
        "BL": (0,              int(h * (1 - pad)),  int(w * pad), h),
    }

    rois: dict[str, tuple[int, int]] = {}
    for label, (x1, y1, x2, y2) in quadrants.items():
        roi = mask[y1:y2, x1:x2]
        cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best, best_area = None, 0
        for c in cnts:
            a = cv2.contourArea(c)
            if a < 300:
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            asp = max(bw, bh) / max(min(bw, bh), 1)
            if asp > 2.5:
                continue
            if a > best_area:
                best_area, best = a, c
        if best is None:
            continue
        bx, by, bw, bh = cv2.boundingRect(best)
        cx, cy = bx + bw // 2 + x1, by + bh // 2 + y1
        rois[label] = (cx, cy)
    return rois


# ===========================================================================
# B. REFINEMENT METHODS — return ((cx, cy, w, h, angle), debug_image) or None
# ===========================================================================

def _candidate_from_quad(quad: np.ndarray) -> tuple[float, float, float, float, float]:
    """Return (cx, cy, w, h, angle) from a 4-point quad via minAreaRect."""
    rect = cv2.minAreaRect(quad.astype(np.float32))
    (cx, cy), (rw, rh), ang = rect
    if rw < rh:
        rw, rh, ang = rh, rw, ang + 90
    return float(cx), float(cy), float(rw), float(rh), float(ang)


def _method_adaptive_threshold(roi_gray: np.ndarray):
    th = cv2.adaptiveThreshold(
        roi_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 35, 5,
    )
    th = cv2.morphologyEx(
        th, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    )
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        if peri < 200:
            continue
        for eps in (0.02, 0.04, 0.06):
            ap = cv2.approxPolyDP(c, eps * peri, True)
            if len(ap) == 4 and cv2.isContourConvex(ap):
                best = ap.reshape(4, 2)
                break
        if best is not None:
            break
    if best is None:
        return None
    return _candidate_from_quad(best), th


def _method_canny_minrect(roi_gray: np.ndarray):
    edges = cv2.Canny(cv2.GaussianBlur(roi_gray, (5, 5), 0), 30, 90)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = roi_gray.shape
    centre = np.array([w / 2, h / 2])
    best, best_score = None, 1e18
    for c in cnts:
        if cv2.contourArea(c) < 200:
            continue
        rect = cv2.minAreaRect(c)
        (cx, cy), (rw, rh), _ = rect
        if min(rw, rh) < 40 or max(rw, rh) > min(h, w) * 0.95:
            continue
        asp = max(rw, rh) / max(min(rw, rh), 1)
        if asp > 1.6:
            continue
        d = np.linalg.norm(np.array([cx, cy]) - centre)
        score = d + abs(asp - 1.0) * 30
        if score < best_score:
            best_score, best = score, rect
    if best is None:
        return None
    (cx, cy), (rw, rh), ang = best
    if rw < rh:
        rw, rh, ang = rh, rw, ang + 90
    return (float(cx), float(cy), float(rw), float(rh), float(ang)), edges


def _method_template_match(roi_gray: np.ndarray, expected_side: int):
    side = max(int(expected_side), 40)
    t = np.full((side + 12, side + 12), 255, np.uint8)
    s, e = 6, 6 + side
    cv2.rectangle(t, (s, s), (e, e), 0, 3)
    cv2.line(t, (s, (s + e) // 2), (e, (s + e) // 2), 0, 2)
    cv2.line(t, ((s + e) // 2, s), ((s + e) // 2, e), 0, 2)
    for ty in range(s + 6, e, 8):
        cv2.line(t, (s, ty), (s + 6, ty), 0, 1)
    blurred = cv2.GaussianBlur(roi_gray, (3, 3), 0)
    th_h, tw_w = t.shape
    res = cv2.matchTemplate(blurred, t, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val < 0.15:
        return None
    cx = max_loc[0] + tw_w / 2
    cy = max_loc[1] + th_h / 2
    return (float(cx), float(cy), float(side), float(side), 0.0), t


def _method_hough_square(roi_gray: np.ndarray):
    edges = cv2.Canny(cv2.GaussianBlur(roi_gray, (5, 5), 0), 30, 90)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                            minLineLength=40, maxLineGap=10)
    if lines is None or len(lines) < 4:
        return None
    lines = lines.reshape(-1, 4)
    horiz, vert = [], []
    for x1, y1, x2, y2 in lines:
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        if ang < 20 or ang > 160:
            horiz.append((x1, y1, x2, y2))
        elif 70 < ang < 110:
            vert.append((x1, y1, x2, y2))
    if len(horiz) < 2 or len(vert) < 2:
        return None
    horiz.sort(key=lambda l: (l[1] + l[3]) / 2)
    vert.sort(key=lambda l: (l[0] + l[2]) / 2)
    top, bot = horiz[0], horiz[-1]
    lft, rgt = vert[0], vert[-1]

    def _intersect(la, lb):
        x1, y1, x2, y2 = la; x3, y3, x4, y4 = lb
        d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(d) < 1e-6:
            return None
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / d
        return np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)])
    pts = [_intersect(top, lft), _intersect(top, rgt),
           _intersect(bot, rgt), _intersect(bot, lft)]
    if any(p is None for p in pts):
        return None
    pts = np.array(pts, dtype=np.float32)
    return _candidate_from_quad(pts), edges


def _method_ink_bbox(roi_bgr: np.ndarray):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    hue, brt = hsv[:, :, 0], hsv[:, :, 2]
    ink = (((hue < 95) | (hue > 125))
           & (brt > 130) & (brt < 240)).astype(np.uint8) * 255
    ink = cv2.morphologyEx(
        ink, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12)),
    )
    cnts, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    all_pts = np.vstack([c.reshape(-1, 2) for c in cnts])
    rect = cv2.minAreaRect(all_pts.astype(np.float32))
    (cx, cy), (rw, rh), ang = rect
    if min(rw, rh) < 30:
        return None
    if rw < rh:
        rw, rh, ang = rh, rw, ang + 90
    return (float(cx), float(cy), float(rw), float(rh), float(ang)), ink


def _method_dark_bbox_gray(roi_gray: np.ndarray):
    th = cv2.adaptiveThreshold(
        roi_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 25, 7,
    )
    th = cv2.morphologyEx(
        th, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    th = cv2.morphologyEx(
        th, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)),
    )
    ys, xs = np.where(th > 0)
    if len(xs) < 80:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    (cx, cy), (rw, rh), ang = rect
    if min(rw, rh) < 30:
        return None
    if rw < rh:
        rw, rh, ang = rh, rw, ang + 90
    return (float(cx), float(cy), float(rw), float(rh), float(ang)), th


def _method_harris_corners(roi_gray: np.ndarray):
    gray = roi_gray.astype(np.float32)
    harris = cv2.cornerHarris(gray, blockSize=4, ksize=3, k=0.04)
    harris = cv2.dilate(harris, None)
    h, w = roi_gray.shape
    thresh = harris.max() * 0.05
    ys, xs = np.where(harris > thresh)
    if len(xs) < 20:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    (cx, cy), (rw, rh), ang = rect
    if min(rw, rh) < 40 or max(rw, rh) > min(h, w) * 0.95:
        return None
    if rw < rh:
        rw, rh, ang = rh, rw, ang + 90
    asp = rw / max(rh, 1)
    if asp > 1.7:
        return None
    return (float(cx), float(cy), float(rw), float(rh), float(ang)), harris


def _score_candidate(cand: tuple[float, float, float, float, float],
                     expected_side: float,
                     roi_size: int) -> float:
    """Lower-is-better composite score."""
    cx, cy, rw, rh, _ = cand
    asp = max(rw, rh) / max(min(rw, rh), 1)
    side = (rw + rh) / 2.0
    centre_offset = float(np.linalg.norm([cx - roi_size / 2, cy - roi_size / 2]))
    aspect_pen = abs(asp - 1.0) * 200
    centre_pen = (centre_offset / roi_size) * 150
    size_pen   = abs(side - expected_side) / expected_side * 8
    if side < 50 or side > roi_size * 0.95:
        return 1e9
    return aspect_pen + centre_pen + size_pen


# ===========================================================================
# Public API
# ===========================================================================

def detect_markers(
    image_bgr: np.ndarray,
    *,
    paper_w_mm: float = PAPER_W_MM,
    debug_dir: Path | None = None,
) -> dict[str, MarkerDetection]:
    """Locate the 4 calibration markers in *image_bgr*.

    Returns ``{"TL": MarkerDetection, ...}``.  Missing markers are
    simply absent from the returned dict.  Caller is responsible for
    checking ``len(result) == 4`` before relying on it.

    If ``debug_dir`` is provided, intermediate masks / per-method
    debug images are saved into it.
    """
    h, w = image_bgr.shape[:2]
    _save(debug_dir, "00_original.jpg", image_bgr)

    # Expected marker side in pixels, assuming paper spans ~92% of image width
    px_per_mm = (w * 0.92) / paper_w_mm
    expected_side = MARKER_MM * px_per_mm

    # --- A. Coarse ROIs -----------------------------------------------
    rois = _coarse_marker_rois(image_bgr, debug_dir=debug_dir)

    # --- B. Refine within each ROI ------------------------------------
    methods: list[tuple[str, str, Any]] = [
        ("ink_bbox",  "bgr",  _method_ink_bbox),
        ("dark_bbox", "gray", _method_dark_bbox_gray),
        ("adaptive",  "gray", _method_adaptive_threshold),
        ("canny",     "gray", _method_canny_minrect),
        ("template",  "gray", _method_template_match),
        ("hough",     "gray", _method_hough_square),
        ("harris",    "gray", _method_harris_corners),
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
                if mname == "template":
                    out = mfn(gray, int(expected_side))
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


def _inner_corner(box: np.ndarray, paper_centre: np.ndarray) -> tuple[float, float]:
    """Pick the corner of *box* closest to *paper_centre*."""
    dists = np.linalg.norm(box - paper_centre, axis=1)
    idx = int(np.argmin(dists))
    return float(box[idx][0]), float(box[idx][1])


def rectify_with_markers(
    image_bgr: np.ndarray,
    *,
    paper_w_mm: float = PAPER_W_MM,
    paper_h_mm: float = PAPER_H_MM,
    px_per_mm: float = DEFAULT_PX_PER_MM,
    debug_dir: Path | None = None,
) -> RectifyResult:
    """End-to-end Step-1: detect markers, build inner-corner quad, warp.

    Raises ``RuntimeError`` if any of the 4 markers cannot be detected.

    Output dimensions are ``round(paper_w_mm * px_per_mm)`` x
    ``round(paper_h_mm * px_per_mm)`` so the resulting raster has a
    known, exact px/mm scale.
    """
    out_w = int(round(paper_w_mm * px_per_mm))
    out_h = int(round(paper_h_mm * px_per_mm))

    markers = detect_markers(
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

    inner_corners: dict[str, tuple[float, float]] = {}
    for label, det in markers.items():
        inner_corners[label] = _inner_corner(det.box, paper_centre)

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

    # Debug overlays
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
        wb = warped.copy()
        cv2.rectangle(wb, (0, 0), (out_w - 1, out_h - 1),
                      (0, 255, 0), 8)
        _save(debug_dir, "21_rectified_with_border.jpg", wb)

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
