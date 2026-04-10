from __future__ import annotations

import re
import sys
from pathlib import Path

NEW_FILE = '''from __future__ import annotations

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
    strip = _sample_line(gray, p0, p1, half_width=5, samples=256)
    centre = strip[strip.shape[0] // 2]
    darkness = 1.0 - float(np.mean(centre) / 255.0)
    support = float(np.mean(centre < 185))
    band_support = float(np.mean(np.mean(strip < 195, axis=0) > 0.30))
    score = 0.40 * darkness + 0.30 * support + 0.30 * band_support
    return score, band_support


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
    tl, tr, br, bl = quad
    pts = [tl, tr, br, bl]
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
        ratio_score = max(0.0, 1.0 - ratio_err / 0.40)

        area = _quad_area(quad)
        area_frac = area / image_area
        bbox_frac = (float(np.ptp(quad[:, 0])) * float(np.ptp(quad[:, 1]))) / image_area
        area_score = _occupancy_band_score(area_frac, lo=0.08, peak_lo=0.30, peak_hi=0.90, hi=0.965)
        extent_score = _occupancy_band_score(bbox_frac, lo=0.10, peak_lo=0.35, peak_hi=0.94, hi=0.985)

        side_scores = []
        band_scores = []
        pts = list(quad)
        for p0, p1 in zip(pts, pts[1:] + pts[:1]):
            s, b = _side_evidence(gray, p0, p1)
            side_scores.append(s)
            band_scores.append(b)
        side_score = float(np.mean(side_scores))
        band_score = float(np.mean(band_scores))

        corner_scores = [_corner_support(gray, p) for p in quad]
        corner_score = float(np.mean(sorted(corner_scores, reverse=True)[:3]))
        outside_penalty = _outside_envelope_penalty(gray, quad)

        plausible = bool(
            area_frac >= 0.12
            and area_frac <= 0.965
            and bbox_frac >= 0.18
            and bbox_frac <= 0.985
            and ratio_score >= 0.18
            and (side_score >= 0.16 or band_score >= 0.18)
        )
        if not plausible:
            continue

        total = (
            2.40 * ratio_score
            + 1.20 * area_score
            + 1.00 * extent_score
            + 2.10 * side_score
            + 1.00 * band_score
            + 0.75 * corner_score
            - 1.35 * outside_penalty
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
                "corner_score": float(corner_score),
                "outside_penalty": float(outside_penalty),
                "plausible": plausible,
                "area_frac": float(area_frac),
                "bbox_frac": float(bbox_frac),
            }

    if best is None:
        raise RuntimeError("Could not detect CDS outer border candidate.")

    confidence = "high"
    if best["score"] < 3.2 or best["side_score"] < 0.22 or best["ratio_score"] < 0.25:
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
            "corner_score": float(best["corner_score"]),
            "outside_penalty": float(best["outside_penalty"]),
            "plausible": bool(best["plausible"]),
            "area_frac": float(best["area_frac"]),
            "bbox_frac": float(best["bbox_frac"]),
        },
    )
'''


def maybe_patch_boundary_test(repo_root: Path) -> bool:
    test_path = repo_root / "tests" / "test_boundary_detection.py"
    if not test_path.exists():
        return False
    text = test_path.read_text(encoding="utf-8")
    updated = re.sub(r"expected_aspect_ratio\s*=\s*500\s*/\s*600", "expected_aspect_ratio=600 / 500", text)
    if updated != text:
        test_path.write_text(updated, encoding="utf-8")
        return True
    return False


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python apply_phase1_frame_reject_detector_patch.py <repo_root>")
        return 2

    repo_root = Path(sys.argv[1]).resolve()
    target = repo_root / "src" / "armourcore_cds" / "phase1" / "boundary_detection.py"
    if not target.exists():
        print(f"Could not find target file: {target}")
        return 1

    target.write_text(NEW_FILE, encoding="utf-8")
    test_patched = maybe_patch_boundary_test(repo_root)

    print(f"Patched: {target}")
    if test_patched:
        print("Patched stale synthetic boundary test aspect ratio to 600 / 500.")
    else:
        print("Boundary test did not need aspect-ratio patching.")
    print("Next: run pytest -q, then rerun Test02 and Test01.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
