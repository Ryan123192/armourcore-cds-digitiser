
from __future__ import annotations

from pathlib import Path
import textwrap

ROOT = Path.cwd()
TARGET = ROOT / "src" / "armourcore_cds" / "phase1" / "boundary_detection.py"

CONTENT = """from __future__ import annotations

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


def _side_evidence(gray: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> tuple[float, float]:
    strip = _sample_line(gray, p0, p1, half_width=4, samples=256)
    centre = strip[strip.shape[0] // 2]
    darkness = 1.0 - float(np.mean(centre) / 255.0)
    support = float(np.mean(centre < 180))
    thick_dark = float(np.mean(np.mean(strip < 190, axis=0) > 0.45))
    score = 0.45 * darkness + 0.35 * support + 0.20 * thick_dark
    return score, thick_dark


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


def _build_candidates_from_contours(gray: np.ndarray) -> list[dict[str, Any]]:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 180)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[dict[str, Any]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.02 * gray.shape[0] * gray.shape[1]:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        quads: list[np.ndarray] = []
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quads.append(approx.reshape(4, 2).astype(np.float32))
        else:
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect).astype(np.float32)
            quads.append(box)

        for q in quads:
            ordered = _order_quad_points(q)
            qa = _quad_area(ordered)
            if qa < 0.02 * gray.shape[0] * gray.shape[1]:
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

    best: dict[str, Any] | None = None
    best_score = -1e9

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
        area_score = np.clip((area_frac - 0.10) / 0.50, 0.0, 1.0)

        side_scores = []
        thickness_scores = []
        pts = list(quad)
        for p0, p1 in zip(pts, pts[1:] + pts[:1]):
            s, t = _side_evidence(gray, p0, p1)
            side_scores.append(s)
            thickness_scores.append(t)
        side_score = float(np.mean(side_scores))
        thickness_score = float(np.mean(thickness_scores))

        corner_scores = [_corner_support(gray, p) for p in quad]
        corner_score = float(np.mean(sorted(corner_scores, reverse=True)[:3]))

        outside_penalty = _outside_envelope_penalty(gray, quad)

        total = (
            2.20 * ratio_score
            + 1.30 * area_score
            + 2.40 * side_score
            + 1.10 * thickness_score
            + 0.75 * corner_score
            - 1.50 * outside_penalty
        )

        if total > best_score:
            best_score = total
            best = {
                "quad": quad,
                "contour_area_px": cand["contour_area_px"],
                "score": float(total),
                "ratio_score": float(ratio_score),
                "area_score": float(area_score),
                "side_score": float(side_score),
                "thickness_score": float(thickness_score),
                "corner_score": float(corner_score),
                "outside_penalty": float(outside_penalty),
            }

    if best is None:
        raise RuntimeError("Could not detect CDS outer border candidate.")

    confidence = "high"
    if best["score"] < 3.10 or best["side_score"] < 0.28:
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
            "side_score": float(best["side_score"]),
            "thickness_score": float(best["thickness_score"]),
            "corner_score": float(best["corner_score"]),
            "outside_penalty": float(best["outside_penalty"]),
        },
    )
"""

def main() -> None:
    if not TARGET.exists():
        raise SystemExit(f"Target file not found: {TARGET}")
    TARGET.write_text(textwrap.dedent(CONTENT), encoding="utf-8")
    print(f"Patched {TARGET}")

if __name__ == "__main__":
    main()
