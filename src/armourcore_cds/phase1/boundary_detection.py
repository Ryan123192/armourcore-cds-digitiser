from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class BorderDetectionResult:
    ordered_corners_xy: list[list[float]]
    contour_area_px: float
    score: float
    candidate_count: int
    confidence: str = "high"
    diagnostics: dict[str, Any] | None = None
    debug_images: dict[str, np.ndarray] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _hex_to_bgr(hex_colour: str) -> tuple[int, int, int]:
    s = hex_colour.strip().lstrip("#")
    if len(s) != 6:
        raise ValueError(f"Expected 6-digit hex colour, got: {hex_colour!r}")
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return (b, g, r)


def _hex_to_hsv(hex_colour: str) -> tuple[int, int, int]:
    bgr = np.uint8([[list(_hex_to_bgr(hex_colour))]])
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0]
    return int(hsv[0]), int(hsv[1]), int(hsv[2])


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]   # top-left
    ordered[1] = pts[np.argmin(d)]   # top-right
    ordered[2] = pts[np.argmax(s)]   # bottom-right
    ordered[3] = pts[np.argmax(d)]   # bottom-left
    return ordered


def _quad_side_lengths(quad: np.ndarray) -> tuple[float, float, float, float]:
    tl, tr, br, bl = quad
    top    = float(np.linalg.norm(tr - tl))
    right  = float(np.linalg.norm(br - tr))
    bottom = float(np.linalg.norm(br - bl))
    left   = float(np.linalg.norm(bl - tl))
    return top, right, bottom, left


def _quad_area(quad: np.ndarray) -> float:
    return float(abs(cv2.contourArea(np.asarray(quad, dtype=np.float32))))


def _line_intersect(pa: np.ndarray, pb: np.ndarray,
                    pc: np.ndarray, pd: np.ndarray) -> np.ndarray | None:
    """Intersection of lines pa→pb and pc→pd. Returns None if parallel."""
    d1 = pb - pa
    d2 = pd - pc
    cross = float(d1[0] * d2[1] - d1[1] * d2[0])
    if abs(cross) < 1e-6:
        return None
    t = ((pc[0] - pa[0]) * d2[1] - (pc[1] - pa[1]) * d2[0]) / cross
    return pa + t * d1


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _sample_line_single(
    channel_2d: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    half_width: int = 3,
    samples: int = 256,
) -> np.ndarray:
    h, w = channel_2d.shape[:2]
    xs = np.linspace(float(p0[0]), float(p1[0]), samples)
    ys = np.linspace(float(p0[1]), float(p1[1]), samples)
    dx = float(p1[0] - p0[0])
    dy = float(p1[1] - p0[1])
    length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    nx = -dy / length
    ny =  dx / length
    strips = []
    for off in range(-half_width, half_width + 1):
        xso = np.clip(np.round(xs + nx * off).astype(int), 0, w - 1)
        yso = np.clip(np.round(ys + ny * off).astype(int), 0, h - 1)
        strips.append(channel_2d[yso, xso].astype(np.float32))
    return np.stack(strips, axis=0)


def _sample_line_mask(
    mask: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    half_width: int = 2,
    samples: int = 256,
) -> np.ndarray:
    return _sample_line_single(mask, p0, p1, half_width=half_width, samples=samples)


# ---------------------------------------------------------------------------
# Scoring primitives
# ---------------------------------------------------------------------------

def _side_evidence(gray: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> tuple[float, float]:
    strip = _sample_line_single(gray, p0, p1, half_width=4, samples=256)
    centre = strip[strip.shape[0] // 2]
    darkness    = 1.0 - float(np.mean(centre) / 255.0)
    support     = float(np.mean(centre < 190))
    band_support = float(np.mean(np.mean(strip < 200, axis=0) > 0.25))
    good = (np.mean(strip < 200, axis=0) > 0.25).astype(np.uint8)
    longest = current = 0
    for v in good:
        if v:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    longest_run = float(longest / max(len(good), 1))
    score = 0.32 * darkness + 0.26 * support + 0.26 * band_support + 0.16 * longest_run
    return score, band_support


def _colour_side_evidence(
    colour_mask: np.ndarray, p0: np.ndarray, p1: np.ndarray
) -> float:
    """Coverage of colour pixels along a side — complement to grayscale side_evidence."""
    strip = _sample_line_mask(colour_mask, p0, p1, half_width=4, samples=256)
    centre = strip[strip.shape[0] // 2]
    coverage = float(np.mean(centre > 0))
    band_support = float(np.mean(np.mean(strip > 0, axis=0) > 0.25))
    good = (np.mean(strip > 0, axis=0) > 0.25).astype(np.uint8)
    longest = current = 0
    for v in good:
        if v:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    longest_run = float(longest / max(len(good), 1))
    return 0.40 * coverage + 0.35 * band_support + 0.25 * longest_run


def _corner_support(gray: np.ndarray, corner: np.ndarray, radius: int = 36) -> float:
    h, w = gray.shape[:2]
    x = int(round(float(corner[0])))
    y = int(round(float(corner[1])))
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    dark = float(np.mean(patch < 195))
    edge = cv2.Canny(patch, 60, 160)
    return 0.60 * dark + 0.40 * float(np.mean(edge > 0))


def _occupancy_band_score(frac: float, lo: float, peak_lo: float, peak_hi: float, hi: float) -> float:
    if frac <= lo or frac >= hi:
        return 0.0
    if peak_lo <= frac <= peak_hi:
        return 1.0
    if frac < peak_lo:
        return float((frac - lo) / max(peak_lo - lo, 1e-6))
    return float((hi - frac) / max(hi - peak_hi, 1e-6))


def _outside_envelope_penalty(gray: np.ndarray, quad: np.ndarray) -> float:
    penalties = []
    pts = list(quad)
    for a, b in zip(pts, pts[1:] + pts[:1]):
        inner = _sample_line_single(gray, a, b, half_width=2, samples=192)
        outer = _sample_line_single(gray, a, b, half_width=10, samples=192)
        inner_dark = 1.0 - float(np.mean(inner[inner.shape[0] // 2]) / 255.0)
        outer_dark = 1.0 - float(np.mean(outer[0]) / 255.0)
        penalties.append(max(0.0, outer_dark - inner_dark))
    return float(np.mean(penalties))


def _colour_band_width_score(colour_border_mask: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> float:
    strip = _sample_line_mask(colour_border_mask, p0, p1, half_width=8, samples=192)
    row_occupancy = np.mean(strip > 0, axis=1)
    width_frac = float(np.mean(row_occupancy > 0.20))
    return _occupancy_band_score(width_frac, lo=0.08, peak_lo=0.16, peak_hi=0.55, hi=0.90)


def _colour_parallel_penalty(colour_mask: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> float:
    strip = _sample_line_mask(colour_mask, p0, p1, half_width=18, samples=160)
    row_occupancy = np.mean(strip > 0, axis=1)
    centre = row_occupancy.shape[0] // 2
    guard = 4
    far_rows = np.concatenate([
        row_occupancy[: max(0, centre - guard)],
        row_occupancy[min(row_occupancy.shape[0], centre + guard + 1):],
    ])
    return float(np.max(far_rows)) if far_rows.size > 0 else 0.0


def _marker_support_score(
    image_bgr: np.ndarray, quad: np.ndarray, fiducial_hex_candidates: list[str] | None
) -> float:
    if not fiducial_hex_candidates:
        return 0.0
    h, w = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    masks: list[np.ndarray] = []
    for hex_colour in fiducial_hex_candidates:
        try:
            hh, ss, vv = _hex_to_hsv(hex_colour)
        except Exception:
            continue
        s_lo = max(20, ss - 130)
        v_lo = max(40, vv - 150)
        # Primary hue window
        lower = np.array([max(0, hh - 12), s_lo, v_lo], dtype=np.uint8)
        upper = np.array([min(179, hh + 12), 255, 255], dtype=np.uint8)
        masks.append(cv2.inRange(hsv, lower, upper))
        # Red wraps in OpenCV HSV (H≈0 and H≈179 are both red).
        # Always add the complementary wrap-around window when near red.
        if hh <= 15:
            masks.append(cv2.inRange(
                hsv,
                np.array([max(0, 168), s_lo, v_lo], dtype=np.uint8),
                np.array([179, 255, 255], dtype=np.uint8),
            ))
        elif hh >= 164:
            masks.append(cv2.inRange(
                hsv,
                np.array([0, s_lo, v_lo], dtype=np.uint8),
                np.array([min(15, 179 - hh + 12), 255, 255], dtype=np.uint8),
            ))
    if not masks:
        return 0.0
    mask = _merge_masks(*masks)
    radius = max(16, int(round(0.025 * min(h, w))))
    scores = []
    for corner in quad:
        x = int(round(float(corner[0])))
        y = int(round(float(corner[1])))
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        patch = mask[y0:y1, x0:x1]
        scores.append(float(np.mean(patch > 0)) if patch.size > 0 else 0.0)
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def _clear_image_border(mask: np.ndarray, border_px: int) -> np.ndarray:
    out = mask.copy()
    b = max(1, int(border_px))
    out[:b, :] = 0
    out[-b:, :] = 0
    out[:, :b] = 0
    out[:, -b:] = 0
    return out


def _merge_masks(*masks: np.ndarray) -> np.ndarray:
    out = np.zeros_like(masks[0])
    for m in masks:
        out = cv2.bitwise_or(out, m)
    return out


def _build_colour_mask_from_hex_candidates(image_bgr: np.ndarray, hex_candidates: list[str]) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    masks: list[np.ndarray] = []
    for hex_colour in hex_candidates:
        try:
            hh, ss, vv = _hex_to_hsv(hex_colour)
        except Exception:
            continue
        lower = np.array([max(0, hh - 18), max(50, ss - 130), max(60, vv - 120)], dtype=np.uint8)
        upper = np.array([min(179, hh + 18), 255, min(255, vv + 120)], dtype=np.uint8)
        masks.append(cv2.inRange(hsv, lower, upper))
    if not masks:
        return np.zeros(hsv.shape[:2], dtype=np.uint8)
    return _merge_masks(*masks)


def _make_dark_mask(gray: np.ndarray, border_px: int) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(blur)
    dark_mask = cv2.inRange(clahe, 0, 170)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8), iterations=1)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    return _clear_image_border(dark_mask, border_px)


def _make_edge_mask(gray: np.ndarray, border_px: int) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(blur)
    edges = cv2.Canny(clahe, 60, 180)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    return _clear_image_border(edges, border_px)


def _make_orange_mask(
    image_bgr: np.ndarray,
    outer_border_hex_candidates: list[str] | None,
    orange_border_hex: str,
) -> np.ndarray:
    """Orange/terracotta mask — requires meaningful saturation to avoid warm-grey false positives."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    masks: list[np.ndarray] = []

    if outer_border_hex_candidates:
        masks.append(_build_colour_mask_from_hex_candidates(image_bgr, list(outer_border_hex_candidates)))
    else:
        masks.append(_build_colour_mask_from_hex_candidates(image_bgr, [orange_border_hex]))

    h0, s0, v0 = _hex_to_hsv(orange_border_hex)

    # Nominal orange window
    lower1 = np.array([max(0, h0 - 15), max(60, s0 - 120), max(60, v0 - 120)], dtype=np.uint8)
    upper1 = np.array([min(179, h0 + 15), 255, min(255, v0 + 100)], dtype=np.uint8)
    masks.append(cv2.inRange(hsv, lower1, upper1))

    # Slightly shifted / desaturated (printed terracotta)
    lower2 = np.array([max(0, h0 - 20), 50, 55], dtype=np.uint8)
    upper2 = np.array([min(24, h0 + 8), 230, 235], dtype=np.uint8)
    masks.append(cv2.inRange(hsv, lower2, upper2))

    # Static orange/red-orange guard: S_min raised to 55 to avoid warm-grey pickup
    lower3 = np.array([0, 55, 80], dtype=np.uint8)
    upper3 = np.array([20, 255, 235], dtype=np.uint8)
    masks.append(cv2.inRange(hsv, lower3, upper3))

    mask = _merge_masks(*masks)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return mask


def _make_orange_border_mask(
    image_bgr: np.ndarray,
    outer_border_hex_candidates: list[str] | None,
    orange_border_hex: str,
) -> np.ndarray:
    orange_mask = _make_orange_mask(image_bgr, outer_border_hex_candidates, orange_border_hex)
    border_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN,  np.ones((5, 5), np.uint8), iterations=1)
    border_mask = cv2.morphologyEx(border_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    border_mask = cv2.dilate(border_mask, np.ones((3, 3), np.uint8), iterations=1)
    return border_mask


def _make_blue_mask(
    image_bgr: np.ndarray,
    outer_border_hex_candidates: list[str] | None,
    blue_border_hex: str,
) -> np.ndarray:
    """Cyan/blue mask with tightened saturation floor to avoid grey-background false positives."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    masks: list[np.ndarray] = []

    if outer_border_hex_candidates:
        masks.append(_build_colour_mask_from_hex_candidates(image_bgr, list(outer_border_hex_candidates)))
    else:
        masks.append(_build_colour_mask_from_hex_candidates(image_bgr, [blue_border_hex]))

    h0, s0, v0 = _hex_to_hsv(blue_border_hex)

    # Tight nominal window (well-lit blue border)
    lower1 = np.array([max(0, h0 - 12), max(60, s0 - 145), max(70, v0 - 150)], dtype=np.uint8)
    upper1 = np.array([min(179, h0 + 12), 255, 255], dtype=np.uint8)
    masks.append(cv2.inRange(hsv, lower1, upper1))

    # Moderate window for phone white-balance shift / slight exposure change
    lower2 = np.array([max(0, h0 - 20), 55, 60], dtype=np.uint8)
    upper2 = np.array([min(179, h0 + 10), 255, 255], dtype=np.uint8)
    masks.append(cv2.inRange(hsv, lower2, upper2))

    # Broader fallback for very desaturated or overexposed print — S_min raised to 50
    # (old value was 18 which caused grey-background false positives)
    lower3 = np.array([max(0, h0 - 14), 50, 90], dtype=np.uint8)
    upper3 = np.array([min(179, h0 + 16), 200, 255], dtype=np.uint8)
    masks.append(cv2.inRange(hsv, lower3, upper3))

    # Static cyan-blue guard around #00CAEC — S_min raised to 50
    lower4 = np.array([78, 50, 70], dtype=np.uint8)
    upper4 = np.array([110, 255, 255], dtype=np.uint8)
    masks.append(cv2.inRange(hsv, lower4, upper4))

    mask = _merge_masks(*masks)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return mask


def _make_blue_border_mask(
    image_bgr: np.ndarray,
    outer_border_hex_candidates: list[str] | None,
    blue_border_hex: str,
) -> np.ndarray:
    blue_mask = _make_blue_mask(image_bgr, outer_border_hex_candidates, blue_border_hex)
    border_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN,  np.ones((5, 5), np.uint8), iterations=1)
    border_mask = cv2.morphologyEx(border_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    border_mask = cv2.dilate(border_mask, np.ones((3, 3), np.uint8), iterations=1)
    return border_mask


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def _is_frame_like_quad(quad: np.ndarray, w: int, h: int) -> bool:
    """Reject quads that are essentially the image frame itself.

    A legitimate sheet border typically occupies 0.65–0.90 of the image area.
    Candidates above 0.94 are almost always the literal image/paper frame and
    should be dropped before scoring.  The touches-all-4-edges check catches
    cases where the quad clips the image boundary even with a slightly smaller
    area fraction (e.g. a landscape sheet at 92 % area that grazes all corners).
    """
    q = np.asarray(quad, dtype=np.float32)
    tol = max(8.0, 0.01 * min(w, h))
    minx, maxx = float(np.min(q[:, 0])), float(np.max(q[:, 0]))
    miny, maxy = float(np.min(q[:, 1])), float(np.max(q[:, 1]))
    touches = minx <= tol and miny <= tol and maxx >= (w - 1 - tol) and maxy >= (h - 1 - tol)
    area_frac = _quad_area(q) / max(float(w * h), 1.0)
    return bool((touches and area_frac >= 0.90) or area_frac >= 0.94)


def _collect_candidate_quads(
    mask: np.ndarray,
    min_area_px: float,
    source: str,
    *,
    retrieval_mode: int = cv2.RETR_LIST,
    epsilons: tuple[float, ...] = (0.02, 0.01),
) -> list[dict[str, Any]]:
    """Extract quadrilateral candidates from a binary mask.

    Each contour is approximated at multiple epsilon values so that both
    coarse shapes (0.02 × perimeter) and tighter shapes (0.01 × perimeter)
    are represented.  The minAreaRect bounding box is always added as a
    reliable fallback.  Duplicates within this function are not removed here
    (deduplication happens in _dedupe_candidates).
    """
    contours, _ = cv2.findContours(mask, retrieval_mode, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[dict[str, Any]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area_px:
            continue
        peri = cv2.arcLength(contour, True)
        quads: list[np.ndarray] = []
        # Try each approximation tolerance — tighter values preserve the outer
        # border shape better when the contour has slight curvature or gaps.
        for eps in epsilons:
            approx = cv2.approxPolyDP(contour, eps * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                quads.append(approx.reshape(4, 2).astype(np.float32))
        # minAreaRect is always included — it fits the best enclosing rectangle
        # even for broken or partial contours.
        rect = cv2.minAreaRect(contour)
        quads.append(cv2.boxPoints(rect).astype(np.float32))
        for quad in quads:
            ordered = _order_quad_points(quad)
            if _quad_area(ordered) >= min_area_px:
                candidates.append({"quad": ordered, "contour_area_px": float(area), "source": source})
    return candidates


def _dark_band_thickness_frac(
    gray: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    dark_threshold: int = 140,
    num_sample_pts: int = 9,
) -> float:
    """Measure dark-band width along a side, normalised to ``min(h, w)``.

    Returns a value in [0, 1] that is independent of image resolution and
    scale.  Typical values:

    * Outer thick border  ≈ 0.006–0.025  (0.6 %–2.5 % of short dimension)
    * Interior grid line  ≈ 0.001–0.005  (0.1 %–0.5 % of short dimension)

    Because the ratio is relative to the image, the same threshold works for
    a 3 MP phone crop and a 70 MP flatbed scan.
    """
    h, w = gray.shape[:2]
    scale = min(h, w)
    max_scan = max(4, int(round(scale * 0.05)))  # scan up to 5 % of short side

    side_vec = p1 - p0
    side_len = float(np.linalg.norm(side_vec))
    if side_len < 1e-6:
        return 0.0
    side_unit = side_vec / side_len
    normal = np.array([-side_unit[1], side_unit[0]], dtype=np.float64)

    widths: list[float] = []
    for alpha in np.linspace(0.1, 0.9, num_sample_pts):
        p = p0 + alpha * side_vec
        total_dark = 0
        for sign in (1.0, -1.0):  # both sides of the line
            for d in range(1, max_scan + 1):
                px = int(round(float(p[0]) + sign * normal[0] * d))
                py = int(round(float(p[1]) + sign * normal[1] * d))
                if 0 <= px < w and 0 <= py < h and gray[py, px] < dark_threshold:
                    total_dark += 1
                else:
                    break
        widths.append(float(total_dark))

    if not widths:
        return 0.0
    return float(np.median(widths)) / float(max_scan)


def _dedupe_candidates(raw_candidates: list[dict[str, Any]], w: int, h: int) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for cand in raw_candidates:
        quad = cand["quad"]
        if _is_frame_like_quad(quad, w, h):
            continue
        top, right, bottom, left = _quad_side_lengths(quad)
        cx = int(round(float(np.mean(quad[:, 0])) / 24.0))
        cy = int(round(float(np.mean(quad[:, 1])) / 24.0))
        ww = int(round(((top + bottom) / 2.0) / 24.0))
        hh = int(round(((left + right) / 2.0) / 24.0))
        key = (cx, cy, ww, hh)
        if key not in seen:
            seen.add(key)
            deduped.append(cand)
    return deduped


def _build_black_candidates(
    gray: np.ndarray, border_px: int, min_area_px: float
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    edges     = _make_edge_mask(gray, border_px)
    dark_mask = _make_dark_mask(gray, border_px)
    combined  = _merge_masks(edges, dark_mask)
    combined  = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    combined  = _clear_image_border(combined, border_px)

    raw: list[dict[str, Any]] = []
    raw.extend(_collect_candidate_quads(edges,     min_area_px=min_area_px, source="edges"))
    raw.extend(_collect_candidate_quads(dark_mask, min_area_px=min_area_px, source="dark"))
    raw.extend(_collect_candidate_quads(combined,  min_area_px=min_area_px, source="combined"))

    # ------------------------------------------------------------------ #
    # Padded-image pass: when the sheet fills the frame its outer border  #
    # may lie within border_px of the image edge and get stripped by      #
    # _clear_image_border.  Padding with neutral grey lets those edge     #
    # contours close properly so minAreaRect yields a clean quad.         #
    # ------------------------------------------------------------------ #
    PAD = max(60, border_px * 3)
    padded_gray = cv2.copyMakeBorder(
        gray, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=200
    )
    pad_edges = _make_edge_mask(padded_gray, border_px)
    pad_dark  = _make_dark_mask(padded_gray, border_px)
    for cand in _collect_candidate_quads(pad_edges, min_area_px=min_area_px, source="padded_edges"):
        cand["quad"] = cand["quad"] - np.float32([PAD, PAD])
        raw.append(cand)
    for cand in _collect_candidate_quads(pad_dark, min_area_px=min_area_px, source="padded_dark"):
        cand["quad"] = cand["quad"] - np.float32([PAD, PAD])
        raw.append(cand)

    # ------------------------------------------------------------------ #
    # Envelope-hull candidate: build a convex hull from the LARGEST      #
    # dark contours only (≥ 1 % of image area each).  This lets the     #
    # hull trace the true outer extent of the sheet without being        #
    # contaminated by small corner artefacts at the image boundary.      #
    # The result is a quad that often captures the outer border even     #
    # when that border is too thin/patchy to form its own closed         #
    # 4-point contour.                                                   #
    # Two versions are generated: one from the edge mask and one from    #
    # the dark mask, giving scoring a chance to pick the better one.     #
    # ------------------------------------------------------------------ #
    hull_large_min = max(min_area_px * 0.01, min_area_px * 0.5)
    for hull_mask, hull_src in ((edges, "hull_edges"), (dark_mask, "hull_dark")):
        hull_pts: list[np.ndarray] = []
        h_contours, _ = cv2.findContours(hull_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in sorted(h_contours, key=cv2.contourArea, reverse=True)[:12]:
            if cv2.contourArea(cnt) >= hull_large_min:
                hull_pts.append(cnt.reshape(-1, 2))
        if not hull_pts:
            continue
        stacked = np.vstack(hull_pts)
        hull = cv2.convexHull(stacked)
        if hull is None or len(hull) < 4:
            continue
        hull_area = cv2.contourArea(hull)
        if hull_area < min_area_px:
            continue
        rect = cv2.minAreaRect(hull)
        box  = cv2.boxPoints(rect).astype(np.float32)
        ordered = _order_quad_points(box)
        raw.append({"quad": ordered, "contour_area_px": float(hull_area),
                    "source": hull_src})

    blank = np.zeros_like(edges)
    return raw, {"colour_mask": blank, "colour_border_mask": blank, "edge_mask": edges}


def _build_orange_candidates(
    gray: np.ndarray,
    image_bgr: np.ndarray,
    border_px: int,
    min_area_px: float,
    outer_border_hex_candidates: list[str] | None,
    orange_border_hex: str,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    edges              = _make_edge_mask(gray, border_px)
    orange_mask        = _make_orange_mask(image_bgr, outer_border_hex_candidates, orange_border_hex)
    orange_mask        = _clear_image_border(orange_mask, border_px)
    orange_border_mask = _make_orange_border_mask(image_bgr, outer_border_hex_candidates, orange_border_hex)
    orange_border_mask = _clear_image_border(orange_border_mask, border_px)

    edge_support = cv2.dilate(orange_border_mask, np.ones((9, 9), np.uint8), iterations=1)
    supported    = _merge_masks(orange_border_mask, cv2.bitwise_and(edges, edge_support))
    supported    = cv2.morphologyEx(supported, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    supported    = _clear_image_border(supported, border_px)

    raw: list[dict[str, Any]] = []
    raw.extend(_collect_candidate_quads(orange_border_mask, min_area_px=min_area_px * 0.45,
                                        source="orange_border", retrieval_mode=cv2.RETR_EXTERNAL))
    raw.extend(_collect_candidate_quads(supported,          min_area_px=min_area_px * 0.55,
                                        source="orange_supported", retrieval_mode=cv2.RETR_EXTERNAL))
    raw.extend(_collect_candidate_quads(orange_mask,        min_area_px=min_area_px * 0.75,
                                        source="orange_broad", retrieval_mode=cv2.RETR_EXTERNAL))
    # Geometry-first candidates as fallback
    raw.extend(_collect_candidate_quads(edges, min_area_px=min_area_px, source="edges"))
    return raw, {"colour_mask": orange_mask, "colour_border_mask": orange_border_mask, "edge_mask": edges}


def _build_blue_candidates(
    gray: np.ndarray,
    image_bgr: np.ndarray,
    border_px: int,
    min_area_px: float,
    outer_border_hex_candidates: list[str] | None,
    blue_border_hex: str,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    edges            = _make_edge_mask(gray, border_px)
    blue_mask        = _make_blue_mask(image_bgr, outer_border_hex_candidates, blue_border_hex)
    blue_mask        = _clear_image_border(blue_mask, border_px)
    blue_border_mask = _make_blue_border_mask(image_bgr, outer_border_hex_candidates, blue_border_hex)
    blue_border_mask = _clear_image_border(blue_border_mask, border_px)

    edge_support = cv2.dilate(blue_border_mask, np.ones((9, 9), np.uint8), iterations=1)
    supported    = _merge_masks(blue_border_mask, cv2.bitwise_and(edges, edge_support))
    supported    = cv2.morphologyEx(supported, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    supported    = _clear_image_border(supported, border_px)

    raw: list[dict[str, Any]] = []
    raw.extend(_collect_candidate_quads(blue_border_mask, min_area_px=min_area_px * 0.40,
                                        source="blue_border", retrieval_mode=cv2.RETR_EXTERNAL))
    raw.extend(_collect_candidate_quads(supported,        min_area_px=min_area_px * 0.50,
                                        source="blue_supported", retrieval_mode=cv2.RETR_EXTERNAL))
    raw.extend(_collect_candidate_quads(blue_mask,        min_area_px=min_area_px * 0.70,
                                        source="blue_broad", retrieval_mode=cv2.RETR_EXTERNAL))
    # Geometry-first candidates always included as reliable fallback
    raw.extend(_collect_candidate_quads(edges, min_area_px=min_area_px, source="edges"))
    return raw, {"colour_mask": blue_mask, "colour_border_mask": blue_border_mask, "edge_mask": edges}


# ---------------------------------------------------------------------------
# Inside-edge refinement
# ---------------------------------------------------------------------------

def _refine_to_inside_edge(
    quad: np.ndarray,
    gray: np.ndarray,
    colour_border_mask: np.ndarray | None,
    border_colour_mode: str,
    *,
    max_scan_px: int | None = None,
    num_sample_pts: int = 13,
) -> np.ndarray:
    """Move each side of the detected quad inward to the inside edge of the thick border.

    For black mode: scans for the dark-to-light transition along the inward normal.
    For colour modes: scans for where the colour mask ends going inward.
    """
    h, w = gray.shape[:2]
    if max_scan_px is None:
        max_scan_px = max(20, int(round(0.025 * min(h, w))))

    cx = float(np.mean(quad[:, 0]))
    cy = float(np.mean(quad[:, 1]))

    use_colour = (
        border_colour_mode in {"orange", "blue"}
        and colour_border_mask is not None
        and colour_border_mask.any()
    )

    side_offsets: list[float] = []

    for i in range(4):
        pa = quad[i].copy()
        pb = quad[(i + 1) % 4].copy()
        side_vec = pb - pa
        side_len = float(np.linalg.norm(side_vec))
        if side_len < 1e-6:
            side_offsets.append(0.0)
            continue

        side_unit = side_vec / side_len
        # Inward normal: perpendicular pointing toward centroid
        normal = np.array([-side_unit[1], side_unit[0]], dtype=np.float32)
        mid = (pa + pb) / 2.0
        if float(np.dot(normal, np.array([cx - mid[0], cy - mid[1]]))) < 0:
            normal = -normal

        # Scan range: look back half max_scan (outward) and forward max_scan (inward)
        scan_back   = max_scan_px // 2
        scan_range  = np.arange(-scan_back, max_scan_px + 1, 1)

        found_offsets: list[float] = []

        for alpha in np.linspace(0.1, 0.9, num_sample_pts):
            p = pa + alpha * side_vec
            pxs = np.clip(np.round(p[0] + normal[0] * scan_range).astype(int), 0, w - 1)
            pys = np.clip(np.round(p[1] + normal[1] * scan_range).astype(int), 0, h - 1)

            if use_colour:
                is_border = colour_border_mask[pys, pxs] > 0
            else:
                is_border = gray[pys, pxs] < 150   # dark pixel = black border

            # Find the farthest inward border pixel (= inside edge of thick line)
            inward_border = is_border & (scan_range > 0)
            if np.any(inward_border):
                last_idx = int(np.max(np.where(inward_border)[0]))
                found_offsets.append(float(scan_range[last_idx]))
            elif np.any(is_border & (scan_range >= 0)):
                found_offsets.append(0.0)
            # else: no border found inward → don't contribute an offset

        if found_offsets:
            median_off = float(np.median(found_offsets))
            # Sanity: ignore if offset is suspiciously large (> 80% of scan)
            if abs(median_off) <= max_scan_px * 0.85:
                side_offsets.append(median_off)
            else:
                side_offsets.append(0.0)
        else:
            side_offsets.append(0.0)

    # Rebuild quad by moving each side and intersecting adjacent moved sides
    # Represent each moved side as two offset endpoints
    moved_sides: list[tuple[np.ndarray, np.ndarray]] = []
    normals: list[np.ndarray] = []

    for i in range(4):
        pa = quad[i].copy()
        pb = quad[(i + 1) % 4].copy()
        side_vec = pb - pa
        side_len = float(np.linalg.norm(side_vec))
        side_unit = side_vec / max(side_len, 1e-6)
        normal = np.array([-side_unit[1], side_unit[0]], dtype=np.float32)
        mid = (pa + pb) / 2.0
        if float(np.dot(normal, np.array([cx - mid[0], cy - mid[1]]))) < 0:
            normal = -normal
        normals.append(normal)
        off = side_offsets[i]
        moved_sides.append((pa + normal * off, pb + normal * off))

    # Corner i = intersection of moved side (i-1) and moved side (i)
    refined_corners: list[np.ndarray] = []
    for i in range(4):
        prev_a, prev_b = moved_sides[(i - 1) % 4]
        curr_a, curr_b = moved_sides[i]
        pt = _line_intersect(prev_a, prev_b, curr_a, curr_b)
        if pt is None:
            # Fallback: midpoint of the two original endpoints
            pt = (prev_b + curr_a) / 2.0
        refined_corners.append(pt)

    refined = np.array(refined_corners, dtype=np.float32)

    # Validate: don't apply refinement if it moved corners absurdly far
    orig_area   = _quad_area(quad)
    refined_area = _quad_area(refined)
    if orig_area > 0 and not (0.50 <= refined_area / orig_area <= 1.20):
        return quad  # reject refinement — something went wrong

    refined[:, 0] = np.clip(refined[:, 0], 0, w - 1)
    refined[:, 1] = np.clip(refined[:, 1], 0, h - 1)
    return _order_quad_points(refined)


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

def _score_candidate(
    cand: dict[str, Any],
    gray: np.ndarray,
    image_bgr: np.ndarray,
    image_area: float,
    expected_aspect_ratio: float,
    colour_mask: np.ndarray,
    colour_border_mask: np.ndarray,
    border_colour_mode: str,
    fallback_to_fiducials: bool,
    fiducial_hex_candidates: list[str] | None,
) -> dict[str, Any]:
    quad = cand["quad"]

    # Aspect ratio
    top, right, bottom, left = _quad_side_lengths(quad)
    width  = max((top + bottom) / 2.0, 1e-6)
    height = max((left + right) / 2.0, 1e-6)
    ratio  = width / height
    ratio_err   = abs(np.log(max(ratio, 1e-6) / max(expected_aspect_ratio, 1e-6)))
    ratio_score = max(0.0, 1.0 - ratio_err / 0.40)

    # Area plausibility
    area      = _quad_area(quad)
    area_frac = area / image_area
    bbox_frac = (float(np.ptp(quad[:, 0])) * float(np.ptp(quad[:, 1]))) / image_area
    area_score = _occupancy_band_score(area_frac, lo=0.05, peak_lo=0.20, peak_hi=0.80, hi=0.96)

    # Side evidence (grayscale)
    pts = list(quad)
    side_scores_raw = [_side_evidence(gray, p0, p1)[0]
                       for p0, p1 in zip(pts, pts[1:] + pts[:1])]
    avg_side_score   = float(np.mean(side_scores_raw))
    worst_side_score = float(min(side_scores_raw))

    # Colour side evidence (if colour mode)
    colour_side_scores_raw: list[float] = []
    if border_colour_mode in {"orange", "blue"} and colour_mask.any():
        colour_side_scores_raw = [
            _colour_side_evidence(colour_mask, p0, p1)
            for p0, p1 in zip(pts, pts[1:] + pts[:1])
        ]
    avg_colour_side   = float(np.mean(colour_side_scores_raw)) if colour_side_scores_raw else 0.0
    worst_colour_side = float(min(colour_side_scores_raw)) if colour_side_scores_raw else 0.0

    # Colour band width and parallel penalty
    colour_band_width_score  = 0.0
    colour_parallel_penalty  = 0.0
    if border_colour_mode in {"orange", "blue"} and colour_border_mask.any():
        colour_band_width_score = float(np.mean([
            _colour_band_width_score(colour_border_mask, p0, p1)
            for p0, p1 in zip(pts, pts[1:] + pts[:1])
        ]))
        colour_parallel_penalty = float(np.mean([
            _colour_parallel_penalty(colour_mask, p0, p1)
            for p0, p1 in zip(pts, pts[1:] + pts[:1])
        ]))

    # Combined colour score
    colour_score = float(np.clip(
        0.55 * avg_colour_side + 0.55 * colour_band_width_score - 0.35 * colour_parallel_penalty,
        0.0, 1.0,
    )) if border_colour_mode in {"orange", "blue"} else 0.0

    # Scale-invariant dark-band thickness score (black mode only).
    # Measures the width of the dark band along each side, normalised by
    # min(h, w).  Outer borders (~1–2 % of short dimension) score much higher
    # than interior grid lines (~0.1–0.5 %).  The normalisation makes this
    # identical in meaning across phone photos, scans, and PDFs.
    thickness_score = 0.0
    if border_colour_mode == "black":
        thickness_fracs = [
            _dark_band_thickness_frac(gray, p0, p1)
            for p0, p1 in zip(pts, pts[1:] + pts[:1])
        ]
        # Reward minimum (worst side) so a single thin-grid-line side can't hide.
        thickness_score = float(min(thickness_fracs))

    # Corner / fiducial scores
    corner_scores = [_corner_support(gray, p) for p in quad]
    corner_score  = float(np.mean(corner_scores))
    worst_corner  = float(min(corner_scores))

    marker_support_score = (
        _marker_support_score(image_bgr, quad, fiducial_hex_candidates)
        if fallback_to_fiducials and fiducial_hex_candidates
        else 0.0
    )

    outside_penalty = _outside_envelope_penalty(gray, quad)

    # Page-like penalty — increases steeply for candidates that fill the frame.
    # A real CDS border occupies ~0.65–0.90 of the image; anything above that
    # is penalised progressively to avoid selecting the paper/image edge.
    page_like_penalty = 0.0
    if area_frac > 0.82:
        page_like_penalty += (area_frac - 0.82) * 3.0
    if area_frac > 0.90:                          # extra steep cliff above 90 %
        page_like_penalty += (area_frac - 0.90) * 12.0
    if bbox_frac > 0.88:
        page_like_penalty += (bbox_frac - 0.88) * 3.0
    if bbox_frac > 0.92:
        page_like_penalty += (bbox_frac - 0.92) * 10.0

    # ---- Scoring formula ----
    if border_colour_mode in {"orange", "blue"}:
        total = (
            2.00 * ratio_score
            + 0.70 * area_score
            + 1.40 * avg_side_score
            + 2.00 * worst_side_score       # penalise one bad side heavily
            + 1.20 * colour_score
            + 0.80 * avg_colour_side
            + 3.00 * worst_colour_side      # penalise one bad colour side heavily
            + 0.50 * corner_score
            + 2.00 * marker_support_score
            - 0.80 * outside_penalty
            - 0.60 * colour_parallel_penalty
            - page_like_penalty
        )
    else:  # black mode
        total = (
            2.20 * ratio_score
            + 0.80 * area_score
            + 1.80 * avg_side_score
            + 2.20 * worst_side_score       # heavily penalise one bad side
            + 2.50 * thickness_score        # outer border >> grid line, scale-invariant
            + 0.60 * corner_score
            + 0.50 * marker_support_score
            - 1.00 * outside_penalty
            - page_like_penalty
        )

    # Geometric plausibility gate (colour NOT required — colour contributes to score only)
    plausible = bool(
        0.06 <= area_frac <= 0.96
        and 0.08 <= bbox_frac <= 0.97
        and ratio_score >= 0.10
        and (avg_side_score >= 0.10 or worst_side_score >= 0.06)
    )

    return {
        "quad":                   quad,
        "contour_area_px":        cand["contour_area_px"],
        "source":                 cand.get("source", "unknown"),
        "score":                  float(total),
        "ratio_score":            float(ratio_score),
        "area_score":             float(area_score),
        "area_frac":              float(area_frac),
        "bbox_frac":              float(bbox_frac),
        "avg_side_score":         float(avg_side_score),
        "worst_side_score":       float(worst_side_score),
        "colour_score":           float(colour_score),
        "avg_colour_side":        float(avg_colour_side),
        "worst_colour_side":      float(worst_colour_side),
        "colour_band_width_score": float(colour_band_width_score),
        "colour_parallel_penalty": float(colour_parallel_penalty),
        "corner_score":           float(corner_score),
        "worst_corner":           float(worst_corner),
        "marker_support_score":   float(marker_support_score),
        "thickness_score":        float(thickness_score),
        "outside_penalty":        float(outside_penalty),
        "page_like_penalty":      float(page_like_penalty),
        "plausible":              bool(plausible),
    }


# ---------------------------------------------------------------------------
# Mode dispatcher
# ---------------------------------------------------------------------------

def _detect_outer_border_in_mode(
    image_bgr: np.ndarray,
    expected_aspect_ratio: float,
    *,
    use_colour_hint: bool,
    use_shape_constraints: bool,
    fallback_to_fiducials: bool,
    outer_border_hex_candidates: list[str] | None,
    fiducial_hex_candidates: list[str] | None,
    border_colour_mode: str,
    orange_border_hex: str,
    blue_border_hex: str,
) -> BorderDetectionResult:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    image_area = float(h * w)
    min_area_px = 0.02 * image_area
    border_px   = max(4, int(round(min(h, w) * 0.008)))

    # --- Candidate generation ---
    if border_colour_mode == "orange":
        raw_candidates, masks = _build_orange_candidates(
            gray, image_bgr, border_px, min_area_px,
            outer_border_hex_candidates if use_colour_hint else None,
            orange_border_hex,
        )
    elif border_colour_mode == "blue":
        raw_candidates, masks = _build_blue_candidates(
            gray, image_bgr, border_px, min_area_px,
            outer_border_hex_candidates if use_colour_hint else None,
            blue_border_hex,
        )
    else:
        raw_candidates, masks = _build_black_candidates(gray, border_px, min_area_px)

    candidates = _dedupe_candidates(raw_candidates, w, h)
    if not candidates:
        raise RuntimeError(
            f"No geometric candidates generated in {border_colour_mode!r} mode "
            f"(raw={len(raw_candidates)})."
        )

    colour_mask        = masks["colour_mask"]
    colour_border_mask = masks["colour_border_mask"]
    edge_mask          = masks["edge_mask"]

    # --- Score every candidate ---
    best: dict[str, Any] | None     = None   # best plausible (or unrestricted) candidate
    best_score                       = -1e9
    best_fallback: dict[str, Any] | None = None  # best regardless of plausibility
    best_fallback_score              = -1e9
    scored_candidates: list[dict[str, Any]] = []

    for cand in candidates:
        scored = _score_candidate(
            cand, gray, image_bgr, image_area, expected_aspect_ratio,
            colour_mask, colour_border_mask, border_colour_mode,
            fallback_to_fiducials, fiducial_hex_candidates,
        )
        scored_candidates.append({k: v for k, v in scored.items() if k != "quad"})

        total     = scored["score"]
        plausible = scored["plausible"]

        # Always track best overall (fallback)
        if total > best_fallback_score:
            best_fallback_score = total
            best_fallback = scored

        # Track best plausible candidate
        if plausible or not use_shape_constraints:
            if total > best_score:
                best_score = total
                best = scored

    # --- Fallback: if no plausible candidate found, use best available ---
    used_fallback = False
    if best is None:
        if best_fallback is not None:
            best = best_fallback
            used_fallback = True
        else:
            raise RuntimeError(
                f"Could not score any candidates in {border_colour_mode!r} mode."
            )

    # --- Inside-edge refinement ---
    refined_quad = _refine_to_inside_edge(
        best["quad"], gray,
        colour_border_mask if border_colour_mode in {"orange", "blue"} else None,
        border_colour_mode,
    )

    # --- Confidence ---
    if used_fallback:
        confidence = "low"
    elif border_colour_mode in {"orange", "blue"}:
        confidence = "high" if (
            best["score"] >= 4.5
            and best["ratio_score"] >= 0.25
            and best["worst_side_score"] >= 0.12
        ) else "low"
    else:
        confidence = "high" if (
            best["score"] >= 5.0
            and best["ratio_score"] >= 0.25
            and best["worst_side_score"] >= 0.15
        ) else "low"

    # --- Diagnostics ---
    source_counts: dict[str, int] = {}
    for c in candidates:
        s = str(c.get("source", "unknown"))
        source_counts[s] = source_counts.get(s, 0) + 1

    top_candidates = sorted(scored_candidates, key=lambda x: x["score"], reverse=True)[:5]

    diagnostics: dict[str, Any] = {
        "ratio_score":             float(best["ratio_score"]),
        "area_score":              float(best["area_score"]),
        "area_frac":               float(best["area_frac"]),
        "avg_side_score":          float(best["avg_side_score"]),
        "worst_side_score":        float(best["worst_side_score"]),
        "colour_score":            float(best["colour_score"]),
        "avg_colour_side":         float(best["avg_colour_side"]),
        "worst_colour_side":       float(best["worst_colour_side"]),
        "colour_band_width_score": float(best["colour_band_width_score"]),
        "colour_parallel_penalty": float(best["colour_parallel_penalty"]),
        "corner_score":            float(best["corner_score"]),
        "worst_corner":            float(best["worst_corner"]),
        "marker_support_score":    float(best["marker_support_score"]),
        "outside_penalty":         float(best["outside_penalty"]),
        "page_like_penalty":       float(best["page_like_penalty"]),
        "plausible":               bool(best["plausible"]),
        "used_fallback":           bool(used_fallback),
        "requested_colour_mode":   border_colour_mode,
        "detected_border_mode":    border_colour_mode,
        "candidate_source":        best["source"],
        "candidate_sources":       source_counts,
        "top_candidates":          top_candidates,
        "orange_border_hex":       orange_border_hex,
        "blue_border_hex":         blue_border_hex,
        "outer_border_hex_candidates": list(outer_border_hex_candidates or []),
        "fiducial_hex_candidates": fiducial_hex_candidates or [],
    }

    return BorderDetectionResult(
        ordered_corners_xy=[[float(x), float(y)] for x, y in refined_quad.tolist()],
        contour_area_px=float(best["contour_area_px"]),
        score=float(best["score"]),
        candidate_count=len(candidates),
        confidence=confidence,
        diagnostics=diagnostics,
        debug_images={
            "colour_mask":        colour_mask,
            "colour_border_mask": colour_border_mask,
            "edge_mask":          edge_mask,
        },
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def detect_outer_border(
    image_bgr: np.ndarray,
    expected_aspect_ratio: float,
    *,
    use_colour_hint: bool = True,
    use_shape_constraints: bool = True,
    fallback_to_fiducials: bool = True,
    outer_border_hex_candidates: list[str] | None = None,
    fiducial_hex_candidates: list[str] | None = None,
    border_colour_mode: str = "auto",
    orange_border_hex: str = "#D35400",
    blue_border_hex: str = "#00CAEC",
) -> BorderDetectionResult:
    if image_bgr is None or image_bgr.size == 0:
        raise RuntimeError("Empty image passed to detect_outer_border().")

    requested = (border_colour_mode or "auto").strip().lower()
    if requested not in {"auto", "black", "orange", "blue"}:
        requested = "auto"

    common_kwargs = dict(
        use_colour_hint=use_colour_hint,
        use_shape_constraints=use_shape_constraints,
        fallback_to_fiducials=fallback_to_fiducials,
        outer_border_hex_candidates=outer_border_hex_candidates,
        fiducial_hex_candidates=fiducial_hex_candidates,
        orange_border_hex=orange_border_hex,
        blue_border_hex=blue_border_hex,
    )

    if requested in {"black", "orange", "blue"}:
        return _detect_outer_border_in_mode(
            image_bgr, expected_aspect_ratio,
            border_colour_mode=requested,
            **common_kwargs,
        )

    # Auto mode: try all three, pick highest score with high-confidence bonus
    attempts: list[tuple[str, BorderDetectionResult]] = []
    errors: dict[str, str] = {}
    for mode in ("black", "orange", "blue"):
        try:
            attempts.append((
                mode,
                _detect_outer_border_in_mode(
                    image_bgr, expected_aspect_ratio,
                    border_colour_mode=mode,
                    **common_kwargs,
                ),
            ))
        except RuntimeError as exc:
            errors[mode] = str(exc)

    if not attempts:
        raise RuntimeError(f"Auto detection failed in all modes. Errors: {errors}")

    def _rank(item: tuple[str, BorderDetectionResult]) -> float:
        _, result = item
        diag = result.diagnostics or {}
        confidence_bonus = 0.35 if result.confidence == "high" else 0.0
        return float(result.score) + confidence_bonus

    selected_mode, selected = max(attempts, key=_rank)
    diag = dict(selected.diagnostics or {})
    diag["requested_colour_mode"]  = "auto"
    diag["detected_border_mode"]   = selected_mode
    diag["auto_candidates_tried"]  = [m for m, _ in attempts]
    diag["auto_attempt_errors"]    = errors
    selected.diagnostics = diag
    return selected
