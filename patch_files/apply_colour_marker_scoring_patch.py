from __future__ import annotations

import re
import sys
from pathlib import Path

BOUNDARY_DETECTION = '''from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class BorderDetectionResult:
    ordered_corners_xy: list[list[float]]
    contour_area_px: float
    score: float
    candidate_count: int
    confidence: str = "high"
    diagnostics: dict[str, Any] | None = None


def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[1] = pts[np.argmin(d)]
    ordered[2] = pts[np.argmax(s)]
    ordered[3] = pts[np.argmax(d)]
    return ordered


def _quad_side_lengths(quad: np.ndarray) -> tuple[float, float, float, float]:
    tl, tr, br, bl = quad
    top = float(np.linalg.norm(tr - tl))
    right = float(np.linalg.norm(br - tr))
    bottom = float(np.linalg.norm(br - bl))
    left = float(np.linalg.norm(bl - tl))
    return top, right, bottom, left


def _quad_area(quad: np.ndarray) -> float:
    return float(abs(cv2.contourArea(np.asarray(quad, dtype=np.float32))))


def _hex_to_hsv(hex_colour: str) -> np.ndarray:
    colour = hex_colour.strip().lstrip("#")
    if len(colour) != 6:
        raise ValueError(f"Expected 6-digit hex colour, got: {hex_colour!r}")
    rgb = np.array([[[int(colour[0:2], 16), int(colour[2:4], 16), int(colour[4:6], 16)]]], dtype=np.uint8)
    bgr = rgb[:, :, ::-1]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return hsv[0, 0].astype(np.int16)


def _hue_diff(h: np.ndarray, target_h: int) -> np.ndarray:
    diff = np.abs(h.astype(np.int16) - int(target_h))
    return np.minimum(diff, 180 - diff)


def _sample_strip(image: np.ndarray, p0: np.ndarray, p1: np.ndarray, half_width: int = 3, samples: int = 256) -> np.ndarray:
    h, w = image.shape[:2]
    xs = np.linspace(float(p0[0]), float(p1[0]), samples)
    ys = np.linspace(float(p0[1]), float(p1[1]), samples)

    dx = float(p1[0] - p0[0])
    dy = float(p1[1] - p0[1])
    length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    nx = -dy / length
    ny = dx / length

    strips = []
    for off in range(-half_width, half_width + 1):
        xso = np.clip(np.round(xs + nx * off).astype(int), 0, w - 1)
        yso = np.clip(np.round(ys + ny * off).astype(int), 0, h - 1)
        strips.append(image[yso, xso])
    return np.stack(strips, axis=0)


def _sample_line(gray: np.ndarray, p0: np.ndarray, p1: np.ndarray, half_width: int = 3, samples: int = 256) -> np.ndarray:
    return _sample_strip(gray, p0, p1, half_width=half_width, samples=samples).astype(np.float32)


def _side_evidence(gray: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> tuple[float, float]:
    strip = _sample_line(gray, p0, p1, half_width=5, samples=256)
    centre = strip[strip.shape[0] // 2]

    darkness = 1.0 - float(np.mean(centre) / 255.0)
    support = float(np.mean(centre < 185))
    band_support = float(np.mean(np.mean(strip < 195, axis=0) > 0.30))

    covered = (np.mean(strip < 195, axis=0) > 0.30).astype(np.uint8)
    if covered.size:
        padded = np.pad(covered, (1, 1), constant_values=0)
        transitions = np.diff(padded)
        starts = np.where(transitions == 1)[0]
        ends = np.where(transitions == -1)[0]
        longest_run = float(np.max(ends - starts) / covered.size) if starts.size and ends.size else 0.0
    else:
        longest_run = 0.0

    score = 0.34 * darkness + 0.28 * support + 0.28 * band_support + 0.10 * longest_run
    return score, band_support


def _colour_side_evidence(
    hsv: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    colour_mode: str,
    orange_hsv: np.ndarray,
) -> tuple[float, str]:
    strip = _sample_strip(hsv, p0, p1, half_width=6, samples=256).astype(np.int16)
    centre = strip[strip.shape[0] // 2]

    h = strip[:, :, 0]
    s = strip[:, :, 1]
    v = strip[:, :, 2]
    hc = centre[:, 0]
    sc = centre[:, 1]
    vc = centre[:, 2]

    black_centre = ((vc <= 105) & (sc <= 110)).astype(np.uint8)
    black_band = ((v <= 120) & (s <= 120)).astype(np.uint8)
    black_score = 0.60 * float(np.mean(black_centre)) + 0.40 * float(np.mean(np.mean(black_band, axis=0) > 0.35))

    orange_h = int(orange_hsv[0])
    orange_centre = ((_hue_diff(hc, orange_h) <= 14) & (sc >= 55) & (vc >= 45)).astype(np.uint8)
    orange_band = ((_hue_diff(h, orange_h) <= 16) & (s >= 45) & (v >= 35)).astype(np.uint8)
    orange_score = 0.55 * float(np.mean(orange_centre)) + 0.45 * float(np.mean(np.mean(orange_band, axis=0) > 0.30))

    mode = colour_mode.lower().strip()
    if mode == "black":
        return black_score, "black"
    if mode == "orange":
        return orange_score, "orange"
    if orange_score >= black_score:
        return orange_score, "orange"
    return black_score, "black"


def _marker_support(
    hsv: np.ndarray,
    corner: np.ndarray,
    fiducial_hsv_targets: list[np.ndarray],
    radius: int = 60,
) -> float:
    if not fiducial_hsv_targets:
        return 0.0

    h, w = hsv.shape[:2]
    x = int(round(float(corner[0])))
    y = int(round(float(corner[1])))
    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)
    patch = hsv[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0

    hue = patch[:, :, 0].astype(np.int16)
    sat = patch[:, :, 1].astype(np.int16)
    val = patch[:, :, 2].astype(np.int16)

    mask = np.zeros(patch.shape[:2], dtype=np.uint8)
    for target in fiducial_hsv_targets:
        target_h = int(target[0])
        candidate = ((_hue_diff(hue, target_h) <= 12) & (sat >= 65) & (val >= 45)).astype(np.uint8)
        mask = np.maximum(mask, candidate)

    density = float(np.mean(mask > 0))
    if density <= 0.0:
        return 0.0

    density_score = min(density / 0.05, 1.0)
    edges = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)
    edge_density = float(np.mean(edges > 0))
    edge_score = min(edge_density / 0.20, 1.0)
    return 0.60 * density_score + 0.40 * edge_score


def _corner_support(gray: np.ndarray, corner: np.ndarray, radius: int = 36) -> float:
    h, w = gray.shape[:2]
    x = int(round(float(corner[0])))
    y = int(round(float(corner[1])))
    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)
    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    dark = float(np.mean(patch < 195))
    edge = cv2.Canny(patch, 60, 160)
    edge_density = float(np.mean(edge > 0))
    return 0.60 * dark + 0.40 * edge_density


def _outside_envelope_penalty(gray: np.ndarray, quad: np.ndarray) -> float:
    penalties = []
    pts = list(quad)
    for a, b in zip(pts, pts[1:] + pts[:1]):
        inner = _sample_line(gray, a, b, half_width=2, samples=192)
        outer = _sample_line(gray, a, b, half_width=10, samples=192)
        inner_dark = 1.0 - float(np.mean(inner[inner.shape[0] // 2]) / 255.0)
        outer_dark = 1.0 - float(np.mean(outer[0]) / 255.0)
        penalties.append(max(0.0, outer_dark - inner_dark))
    return float(np.mean(penalties))


def _clear_image_border(mask: np.ndarray, border_px: int) -> np.ndarray:
    out = mask.copy()
    b = max(1, int(border_px))
    out[:b, :] = 0
    out[-b:, :] = 0
    out[:, :b] = 0
    out[:, -b:] = 0
    return out


def _occupancy_band_score(frac: float, lo: float, peak_lo: float, peak_hi: float, hi: float) -> float:
    if frac <= lo or frac >= hi:
        return 0.0
    if peak_lo <= frac <= peak_hi:
        return 1.0
    if frac < peak_lo:
        return float((frac - lo) / max(peak_lo - lo, 1e-6))
    return float((hi - frac) / max(hi - peak_hi, 1e-6))


def _is_frame_like_quad(quad: np.ndarray, w: int, h: int) -> bool:
    q = np.asarray(quad, dtype=np.float32)
    tol = max(8.0, 0.01 * min(w, h))
    minx = float(np.min(q[:, 0]))
    maxx = float(np.max(q[:, 0]))
    miny = float(np.min(q[:, 1]))
    maxy = float(np.max(q[:, 1]))
    touches_all_edges = minx <= tol and miny <= tol and maxx >= (w - 1 - tol) and maxy >= (h - 1 - tol)
    area_frac = _quad_area(q) / max(float(w * h), 1.0)
    return bool(touches_all_edges or area_frac >= 0.975)


def _collect_candidate_quads(mask: np.ndarray, min_area_px: float) -> list[dict[str, Any]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[dict[str, Any]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area_px:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        quads: list[np.ndarray] = []
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quads.append(approx.reshape(4, 2).astype(np.float32))
        rect = cv2.minAreaRect(c)
        quads.append(cv2.boxPoints(rect).astype(np.float32))

        for q in quads:
            ordered = _order_quad_points(q)
            qa = _quad_area(ordered)
            if qa < min_area_px:
                continue
            candidates.append({"quad": ordered, "contour_area_px": float(area)})
    return candidates


def _build_candidates_from_contours(gray: np.ndarray) -> list[dict[str, Any]]:
    h, w = gray.shape[:2]
    image_area = float(h * w)
    min_area_px = 0.02 * image_area
    border_px = max(4, int(round(min(h, w) * 0.008)))

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(blur)

    edges = cv2.Canny(clahe, 60, 180)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = _clear_image_border(edges, border_px)

    dark_mask = cv2.inRange(clahe, 0, 165)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    dark_mask = _clear_image_border(dark_mask, border_px)

    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    combined = cv2.bitwise_or(closed_edges, dark_mask)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    combined = _clear_image_border(combined, border_px)

    raw_candidates: list[dict[str, Any]] = []
    for mask in (edges, dark_mask, combined):
        raw_candidates.extend(_collect_candidate_quads(mask, min_area_px=min_area_px))

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for cand in raw_candidates:
        q = cand["quad"]
        if _is_frame_like_quad(q, w, h):
            continue
        top, right, bottom, left = _quad_side_lengths(q)
        cx = int(round(float(np.mean(q[:, 0])) / 24.0))
        cy = int(round(float(np.mean(q[:, 1])) / 24.0))
        ww = int(round(((top + bottom) / 2.0) / 24.0))
        hh = int(round(((left + right) / 2.0) / 24.0))
        key = (cx, cy, ww, hh)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cand)
    return deduped


def detect_outer_border(
    image_bgr: np.ndarray,
    expected_aspect_ratio: float,
    colour_mode: str = "auto",
    orange_border_hex: str = "#D35400",
    fiducial_hex_candidates: list[str] | None = None,
) -> BorderDetectionResult:
    if image_bgr is None or image_bgr.size == 0:
        raise RuntimeError("Empty image passed to detect_outer_border().")

    colour_mode = (colour_mode or "auto").strip().lower()
    if colour_mode not in {"auto", "black", "orange"}:
        raise ValueError(f"Unsupported colour_mode: {colour_mode!r}")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    candidates = _build_candidates_from_contours(gray)
    if not candidates:
        raise RuntimeError("Could not detect CDS outer border candidate.")

    h, w = gray.shape[:2]
    image_area = float(h * w)
    orange_hsv = _hex_to_hsv(orange_border_hex)
    fiducial_hsv_targets = [_hex_to_hsv(x) for x in (fiducial_hex_candidates or ["#FF0000"])]

    best: dict[str, Any] | None = None
    best_score = -1e9

    for cand in candidates:
        quad = cand["quad"]
        top, right, bottom, left = _quad_side_lengths(quad)
        width = max((top + bottom) / 2.0, 1e-6)
        height = max((left + right) / 2.0, 1e-6)

        ratio = width / height
        ratio_err = abs(np.log(max(ratio, 1e-6) / max(expected_aspect_ratio, 1e-6)))
        ratio_score = max(0.0, 1.0 - ratio_err / 0.40)

        area = _quad_area(quad)
        area_frac = area / image_area
        bbox_frac = (float(np.ptp(quad[:, 0])) * float(np.ptp(quad[:, 1]))) / image_area
        area_score = _occupancy_band_score(area_frac, lo=0.08, peak_lo=0.30, peak_hi=0.90, hi=0.965)
        extent_score = _occupancy_band_score(bbox_frac, lo=0.10, peak_lo=0.35, peak_hi=0.94, hi=0.985)

        side_scores = []
        band_scores = []
        colour_side_scores = []
        colour_side_modes = []
        pts = list(quad)
        for p0, p1 in zip(pts, pts[1:] + pts[:1]):
            s, b = _side_evidence(gray, p0, p1)
            cs, cm = _colour_side_evidence(hsv, p0, p1, colour_mode=colour_mode, orange_hsv=orange_hsv)
            side_scores.append(s)
            band_scores.append(b)
            colour_side_scores.append(cs)
            colour_side_modes.append(cm)
        side_score = float(np.mean(side_scores))
        band_score = float(np.mean(band_scores))
        colour_score = float(np.mean(colour_side_scores))

        orange_mean = float(np.mean([s for s, m in zip(colour_side_scores, colour_side_modes) if m == "orange"])) if "orange" in colour_side_modes else 0.0
        black_mean = float(np.mean([s for s, m in zip(colour_side_scores, colour_side_modes) if m == "black"])) if "black" in colour_side_modes else 0.0
        detected_border_mode = "orange" if orange_mean >= black_mean else "black"

        corner_scores = [_corner_support(gray, p) for p in quad]
        corner_score = float(np.mean(sorted(corner_scores, reverse=True)[:3]))
        marker_scores = [_marker_support(hsv, p, fiducial_hsv_targets) for p in quad]
        marker_support_score = float(np.mean(sorted(marker_scores, reverse=True)[:3]))
        outside_penalty = _outside_envelope_penalty(gray, quad)

        plausible = bool(
            area_frac >= 0.12
            and area_frac <= 0.965
            and bbox_frac >= 0.18
            and bbox_frac <= 0.985
            and ratio_score >= 0.18
            and (side_score >= 0.16 or band_score >= 0.18 or colour_score >= 0.18)
        )
        if not plausible:
            continue

        total = (
            2.30 * ratio_score
            + 1.10 * area_score
            + 0.95 * extent_score
            + 1.95 * side_score
            + 0.90 * band_score
            + 1.15 * colour_score
            + 0.80 * marker_support_score
            + 0.65 * corner_score
            - 1.30 * outside_penalty
        )
        if total > best_score:
            best_score = total
            best = {
                "quad": quad,
                "contour_area_px": cand["contour_area_px"],
                "score": float(total),
                "ratio_score": float(ratio_score),
                "area_score": float(area_score),
                "extent_score": float(extent_score),
                "side_score": float(side_score),
                "band_score": float(band_score),
                "colour_score": float(colour_score),
                "marker_support_score": float(marker_support_score),
                "corner_score": float(corner_score),
                "outside_penalty": float(outside_penalty),
                "plausible": plausible,
                "area_frac": float(area_frac),
                "bbox_frac": float(bbox_frac),
                "detected_border_mode": detected_border_mode,
                "requested_colour_mode": colour_mode,
                "orange_border_hex": orange_border_hex,
            }

    if best is None:
        raise RuntimeError("Could not detect CDS outer border candidate.")

    confidence = "high"
    if (
        best["score"] < 3.3
        or best["side_score"] < 0.20
        or best["ratio_score"] < 0.25
        or (best["requested_colour_mode"] == "orange" and best["colour_score"] < 0.20)
    ):
        confidence = "low"

    ordered = best["quad"]
    return BorderDetectionResult(
        ordered_corners_xy=[[float(x), float(y)] for x, y in ordered.tolist()],
        contour_area_px=float(best["contour_area_px"]),
        score=float(best["score"]),
        candidate_count=len(candidates),
        confidence=confidence,
        diagnostics={
            "ratio_score": float(best["ratio_score"]),
            "area_score": float(best["area_score"]),
            "extent_score": float(best["extent_score"]),
            "side_score": float(best["side_score"]),
            "band_score": float(best["band_score"]),
            "colour_score": float(best["colour_score"]),
            "marker_support_score": float(best["marker_support_score"]),
            "corner_score": float(best["corner_score"]),
            "outside_penalty": float(best["outside_penalty"]),
            "plausible": bool(best["plausible"]),
            "area_frac": float(best["area_frac"]),
            "bbox_frac": float(best["bbox_frac"]),
            "requested_colour_mode": best["requested_colour_mode"],
            "detected_border_mode": best["detected_border_mode"],
            "orange_border_hex": best["orange_border_hex"],
            "fiducial_hex_candidates": fiducial_hex_candidates or ["#FF0000"],
        },
    )
'''

HELPER = '''\n\ndef _corners_to_serialisable(corners: object) -> list[list[float]]:\n    arr = np.asarray(corners, dtype=np.float32).reshape(4, 2)\n    return [[round(float(x), 2), round(float(y), 2)] for x, y in arr.tolist()]\n'''


def patch_boundary_detection(repo: Path) -> None:
    path = repo / "src/armourcore_cds/phase1/boundary_detection.py"
    path.write_text(BOUNDARY_DETECTION, encoding="utf-8")


def patch_pipeline(repo: Path) -> None:
    path = repo / "src/armourcore_cds/phase1/pipeline.py"
    text = path.read_text(encoding="utf-8")

    if "import numpy as np" not in text:
        text = text.replace("import cv2\n", "import cv2\nimport numpy as np\n")

    import_anchor = "from armourcore_cds.utils.image_ops import draw_polygon, resize_long_edge, stack_debug_h, save_image\n"
    if "def _corners_to_serialisable" not in text:
        text = text.replace(import_anchor, import_anchor + HELPER + "\n")

    text, count = re.subn(
        r"expected_aspect_ratio = float\(template\.design_area_mm\.width\) / float\(template\.design_area_mm\.height\)\n\s*border = detect_outer_border\(input_image, expected_aspect_ratio=expected_aspect_ratio\)",
        """expected_aspect_ratio = float(template.design_area_mm.width) / float(template.design_area_mm.height)\n    template_border_cfg = template.border_detection or {}\n    template_colour_hints = template_border_cfg.get('colour_hints', {}) if isinstance(template_border_cfg, dict) else {}\n    border_colour_mode = str(app_config.get('processing', {}).get('border_colour_mode', 'auto')).strip().lower()\n    orange_border_hex = str(app_config.get('processing', {}).get('orange_border_hex', '#D35400')).strip()\n    fiducial_hex_candidates = template_colour_hints.get('fiducial_hex_candidates', ['#FF0000'])\n    border = detect_outer_border(\n        input_image,\n        expected_aspect_ratio=expected_aspect_ratio,\n        colour_mode=border_colour_mode,\n        orange_border_hex=orange_border_hex,\n        fiducial_hex_candidates=fiducial_hex_candidates,\n    )""",
        text,
        flags=re.MULTILINE,
    )
    if count == 0:
        raise RuntimeError("Could not patch detect_outer_border call in pipeline.py")

    if "border_confidence=" not in text:
        text = text.replace(
            "        'crop_mode=none_use_detected_border',\n",
            "        'crop_mode=none_use_detected_border',\n        f'border_confidence={border.confidence}',\n        f'border_score={border.score:.3f}',\n        f'border_mode_req={border.diagnostics.get(\"requested_colour_mode\", \"auto\") if border.diagnostics else \"auto\"}',\n        f'border_mode_hit={border.diagnostics.get(\"detected_border_mode\", \"unknown\") if border.diagnostics else \"unknown\"}',\n",
        )

    text, count = re.subn(
        r"'border_detection': \{.*?\n\s*\},",
        """'border_detection': {\n            'ordered_corners_xy': _corners_to_serialisable(border.ordered_corners_xy),\n            'contour_area_px': float(border.contour_area_px),\n            'score': float(border.score),\n            'candidate_count': int(border.candidate_count),\n            'confidence': border.confidence,\n            'requested_colour_mode': (border.diagnostics or {}).get('requested_colour_mode', border_colour_mode),\n            'detected_border_mode': (border.diagnostics or {}).get('detected_border_mode', 'unknown'),\n            'diagnostics': border.diagnostics or {},\n        },""",
        text,
        count=1,
        flags=re.MULTILINE | re.DOTALL,
    )
    if count == 0:
        raise RuntimeError("Could not patch report block in pipeline.py")

    path.write_text(text, encoding="utf-8")


def patch_template_models(repo: Path) -> None:
    path = repo / "src/armourcore_cds/templates/models.py"
    text = path.read_text(encoding="utf-8")
    if "from typing import Any" not in text:
        text = text.replace("from __future__ import annotations\n\n", "from __future__ import annotations\n\nfrom typing import Any\n\n")
    if "border_detection:" not in text:
        anchor = "    fiducials_enabled: bool\n"
        replacement = anchor + "    border_detection: dict[str, Any] | None = None\n    outputs: dict[str, Any] | None = None\n"
        text = text.replace(anchor, replacement)
    path.write_text(text, encoding="utf-8")


def patch_default_config(repo: Path) -> None:
    path = repo / "configs/app/default.yaml"
    text = path.read_text(encoding="utf-8")
    if "border_colour_mode:" not in text:
        anchor = "  preferred_output_dpi: 400\n"
        insertion = anchor + "  border_colour_mode: auto\n  orange_border_hex: \"#D35400\"\n"
        text = text.replace(anchor, insertion)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    repo = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd().resolve()
    if not (repo / "src/armourcore_cds").exists():
        raise SystemExit(f"Repo root not found: {repo}")

    patch_boundary_detection(repo)
    patch_pipeline(repo)
    patch_template_models(repo)
    patch_default_config(repo)

    print("Applied colour + calibration-marker scoring patch")
    print("Updated files:")
    print(" - src/armourcore_cds/phase1/boundary_detection.py")
    print(" - src/armourcore_cds/phase1/pipeline.py")
    print(" - src/armourcore_cds/templates/models.py")
    print(" - configs/app/default.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
