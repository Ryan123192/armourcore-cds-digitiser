from __future__ import annotations

import sys
from pathlib import Path

BOUNDARY_DETECTION_PY = '''from __future__ import annotations

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
    return float(abs(cv2.contourArea(quad.astype(np.float32))))


def _sample_line(gray: np.ndarray, p0: np.ndarray, p1: np.ndarray, half_width: int = 3, samples: int = 256) -> np.ndarray:
    h, w = gray.shape[:2]
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
        strips.append(gray[yso, xso].astype(np.float32))
    return np.stack(strips, axis=0)


def _side_evidence(gray: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> tuple[float, float, float]:
    strip = _sample_line(gray, p0, p1, half_width=6, samples=320)
    centre = strip[strip.shape[0] // 2]
    centre_band = strip[max(0, strip.shape[0] // 2 - 2): min(strip.shape[0], strip.shape[0] // 2 + 3)]

    darkness = 1.0 - float(np.mean(centre) / 255.0)
    support = float(np.mean(np.median(centre_band, axis=0) < 185))
    band_dark = float(np.mean(np.mean(strip < 195, axis=0) > 0.28))
    thick_dark = float(np.mean(np.mean(strip < 185, axis=0) > 0.45))
    score = 0.30 * darkness + 0.20 * support + 0.30 * band_dark + 0.20 * thick_dark
    return score, thick_dark, band_dark


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
    dark = float(np.mean(patch < 190))
    edge = cv2.Canny(patch, 60, 160)
    edge_density = float(np.mean(edge > 0))
    return 0.65 * dark + 0.35 * edge_density


def _outside_envelope_penalty(gray: np.ndarray, quad: np.ndarray) -> float:
    penalties = []
    tl, tr, br, bl = quad
    pts = [tl, tr, br, bl]
    for a, b in zip(pts, pts[1:] + pts[:1]):
        inner = _sample_line(gray, a, b, half_width=2, samples=192)
        outer = _sample_line(gray, a, b, half_width=8, samples=192)
        inner_dark = 1.0 - float(np.mean(inner[inner.shape[0] // 2]) / 255.0)
        outer_dark = 1.0 - float(np.mean(outer[0]) / 255.0)
        penalties.append(max(0.0, outer_dark - inner_dark))
    return float(np.mean(penalties))


def _image_extent_score(quad: np.ndarray, image_shape: tuple[int, int]) -> float:
    h, w = image_shape
    xs = quad[:, 0]
    ys = quad[:, 1]
    bbox_w = float(np.max(xs) - np.min(xs))
    bbox_h = float(np.max(ys) - np.min(ys))
    width_frac = bbox_w / max(float(w), 1.0)
    height_frac = bbox_h / max(float(h), 1.0)
    return float(
        0.5 * np.clip((width_frac - 0.45) / 0.40, 0.0, 1.0)
        + 0.5 * np.clip((height_frac - 0.45) / 0.40, 0.0, 1.0)
    )


def _build_candidates_from_contours(gray: np.ndarray) -> list[dict[str, Any]]:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(blur)

    edges = cv2.Canny(clahe, 55, 165)
    dark_mask = cv2.inRange(clahe, 0, 185)
    dark_mask = cv2.medianBlur(dark_mask, 5)

    kernel3 = np.ones((3, 3), np.uint8)
    dark_closed = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel3, iterations=2)
    dark_open = cv2.morphologyEx(dark_closed, cv2.MORPH_OPEN, kernel3, iterations=1)

    combined = cv2.bitwise_or(edges, dark_open)
    combined = cv2.dilate(combined, kernel3, iterations=1)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

    contour_sets = []
    contours_a, _ = cv2.findContours(combined, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_sets.extend(contours_a)
    contours_b, _ = cv2.findContours(dark_open, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_sets.extend(contours_b)

    candidates: list[dict[str, Any]] = []
    min_area = 0.015 * gray.shape[0] * gray.shape[1]
    for c in contour_sets:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        quads: list[np.ndarray] = []
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quads.append(approx.reshape(4, 2).astype(np.float32))
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect).astype(np.float32)
        quads.append(box)

        for q in quads:
            ordered = _order_quad_points(q)
            qa = _quad_area(ordered)
            if qa < min_area:
                continue
            candidates.append({"quad": ordered, "contour_area_px": float(area)})

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for cand in candidates:
        q = cand["quad"]
        top, right, bottom, left = _quad_side_lengths(q)
        cx = int(round(float(np.mean(q[:, 0])) / 20.0))
        cy = int(round(float(np.mean(q[:, 1])) / 20.0))
        ww = int(round(((top + bottom) / 2.0) / 20.0))
        hh = int(round(((left + right) / 2.0) / 20.0))
        key = (cx, cy, ww, hh)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cand)
    return deduped


def detect_outer_border(image_bgr: np.ndarray, expected_aspect_ratio: float) -> BorderDetectionResult:
    if image_bgr is None or image_bgr.size == 0:
        raise RuntimeError("Empty image passed to detect_outer_border().")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    candidates = _build_candidates_from_contours(gray)
    if not candidates:
        raise RuntimeError("Could not detect CDS outer border candidate.")

    h, w = gray.shape[:2]
    image_area = float(h * w)

    scored: list[dict[str, Any]] = []
    for cand in candidates:
        quad = cand["quad"]
        top, right, bottom, left = _quad_side_lengths(quad)
        width = max((top + bottom) / 2.0, 1e-6)
        height = max((left + right) / 2.0, 1e-6)
        ratio = width / height

        ratio_err = abs(np.log(max(ratio, 1e-6) / max(expected_aspect_ratio, 1e-6)))
        ratio_score = max(0.0, 1.0 - ratio_err / 0.35)

        area = _quad_area(quad)
        area_frac = area / image_area
        area_score = float(np.clip((area_frac - 0.10) / 0.50, 0.0, 1.0))
        extent_score = _image_extent_score(quad, (h, w))

        side_scores = []
        thickness_scores = []
        band_scores = []
        pts = list(quad)
        for p0, p1 in zip(pts, pts[1:] + pts[:1]):
            s, t, b = _side_evidence(gray, p0, p1)
            side_scores.append(s)
            thickness_scores.append(t)
            band_scores.append(b)
        side_score = float(np.mean(side_scores))
        thickness_score = float(np.mean(thickness_scores))
        band_score = float(np.mean(band_scores))

        corner_scores = [_corner_support(gray, p) for p in quad]
        corner_score = float(np.mean(sorted(corner_scores, reverse=True)[:3]))

        outside_penalty = _outside_envelope_penalty(gray, quad)

        plausible = (
            area_frac >= 0.08 and ratio_score >= 0.18 and extent_score >= 0.20
        ) or (
            area_frac >= 0.16 and extent_score >= 0.35
        )
        plausibility_penalty = 0.0 if plausible else 3.25

        total = (
            2.20 * ratio_score
            + 1.50 * area_score
            + 1.10 * extent_score
            + 2.20 * side_score
            + 1.00 * band_score
            + 0.90 * thickness_score
            + 0.70 * corner_score
            - 1.40 * outside_penalty
            - plausibility_penalty
        )

        scored.append(
            {
                "quad": quad,
                "contour_area_px": cand["contour_area_px"],
                "score": float(total),
                "ratio_score": float(ratio_score),
                "area_score": float(area_score),
                "extent_score": float(extent_score),
                "side_score": float(side_score),
                "band_score": float(band_score),
                "thickness_score": float(thickness_score),
                "corner_score": float(corner_score),
                "outside_penalty": float(outside_penalty),
                "plausible": bool(plausible),
                "area_frac": float(area_frac),
            }
        )

    plausible_scored = [s for s in scored if s["plausible"]]
    pool = plausible_scored if plausible_scored else scored
    best = max(pool, key=lambda s: s["score"], default=None)

    if best is None:
        raise RuntimeError("Could not detect CDS outer border candidate.")

    confidence = "high"
    if (
        best["score"] < 3.60
        or best["side_score"] < 0.30
        or best["band_score"] < 0.20
        or best["area_score"] < 0.10
        or best["extent_score"] < 0.18
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
            "thickness_score": float(best["thickness_score"]),
            "corner_score": float(best["corner_score"]),
            "outside_penalty": float(best["outside_penalty"]),
            "plausible": bool(best["plausible"]),
            "area_frac": float(best["area_frac"]),
        },
    )
'''


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python apply_phase1_border_band_detector_patch.py /path/to/armourcore-cds-digitiser")
        return 2

    repo_root = Path(sys.argv[1]).resolve()
    target = repo_root / "src" / "armourcore_cds" / "phase1" / "boundary_detection.py"
    if not target.exists():
        print(f"Target file not found: {target}")
        return 1

    target.write_text(BOUNDARY_DETECTION_PY, encoding="utf-8")
    print(f"Patched: {target}")
    print("Done. This patch assumes the earlier Phase 01 wiring repair has already been applied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
