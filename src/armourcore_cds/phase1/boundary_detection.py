from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from armourcore_cds.utils.image_ops import to_gray


@dataclass
class BorderDetectionResult:
    contour: np.ndarray
    ordered_corners: np.ndarray
    preview_edges: np.ndarray
    preview_mask: np.ndarray
    contour_area_px: float
    score: float
    candidate_count: int


def _order_points_clockwise(points: np.ndarray) -> np.ndarray:
    pts = points.astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]      # top-left
    ordered[2] = pts[np.argmax(s)]      # bottom-right
    ordered[1] = pts[np.argmin(diff)]   # top-right
    ordered[3] = pts[np.argmax(diff)]   # bottom-left
    return ordered


def _contour_score(contour: np.ndarray, image_shape: tuple[int, int], expected_aspect_ratio: float) -> tuple[float, np.ndarray] | None:
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return None

    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    if len(approx) != 4 or not cv2.isContourConvex(approx):
        return None

    area = cv2.contourArea(approx)
    if area <= 0:
        return None

    h, w = image_shape[:2]
    image_area = float(h * w)
    area_ratio = area / image_area
    if area_ratio < 0.12:
        return None

    rect = _order_points_clockwise(approx.reshape(4, 2))
    top_w = float(np.linalg.norm(rect[1] - rect[0]))
    bot_w = float(np.linalg.norm(rect[2] - rect[3]))
    left_h = float(np.linalg.norm(rect[3] - rect[0]))
    right_h = float(np.linalg.norm(rect[2] - rect[1]))
    mean_w = max((top_w + bot_w) * 0.5, 1.0)
    mean_h = max((left_h + right_h) * 0.5, 1.0)
    aspect_ratio = mean_w / mean_h
    aspect_error = abs(np.log(aspect_ratio / expected_aspect_ratio))

    contour_pts = approx.reshape(4, 2).astype(np.float32)
    contour_center = contour_pts.mean(axis=0)
    image_center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
    centre_distance = float(np.linalg.norm(contour_center - image_center))
    max_centre_distance = float(np.linalg.norm(image_center))
    centre_penalty = centre_distance / max(max_centre_distance, 1.0)

    extent = area / max(cv2.contourArea(cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)), 1.0)
    score = (area_ratio * 4.0) + (extent * 1.5) - (aspect_error * 2.5) - (centre_penalty * 0.5)
    return score, rect


def detect_outer_border(image: np.ndarray, expected_aspect_ratio: float) -> BorderDetectionResult:
    gray = to_gray(image)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilated = cv2.dilate(closed, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_score = None
    best_rect = None
    best_contour = None

    for contour in contours:
        scored = _contour_score(contour, image.shape, expected_aspect_ratio)
        if scored is None:
            continue
        score, rect = scored
        if best_score is None or score > best_score:
            best_score = score
            best_rect = rect
            best_contour = contour

    if best_rect is None or best_contour is None or best_score is None:
        raise RuntimeError('Could not detect CDS outer border candidate.')

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [best_contour], -1, 255, thickness=cv2.FILLED)

    return BorderDetectionResult(
        contour=best_contour,
        ordered_corners=best_rect,
        preview_edges=dilated,
        preview_mask=mask,
        contour_area_px=float(cv2.contourArea(best_contour)),
        score=float(best_score),
        candidate_count=len(contours),
    )
