"""Phase 2 module: contour_detection"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ContourCandidate:
    contour: np.ndarray
    area_px: float
    perimeter_px: float
    bbox_xywh: tuple[int, int, int, int]
    approx_vertices: int


def detect_trace_contours(
    trace_mask: np.ndarray,
    min_area_px: float = 12.0,
    min_perimeter_px: float = 20.0,
) -> list[ContourCandidate]:
    contours, _ = cv2.findContours(trace_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[ContourCandidate] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, closed=True))
        if area < min_area_px and perimeter < min_perimeter_px:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        approx = cv2.approxPolyDP(contour, 0.01 * max(perimeter, 1.0), closed=True)

        candidates.append(
            ContourCandidate(
                contour=contour,
                area_px=area,
                perimeter_px=perimeter,
                bbox_xywh=(int(x), int(y), int(w), int(h)),
                approx_vertices=int(len(approx)),
            )
        )

    candidates.sort(key=lambda item: item.perimeter_px, reverse=True)
    return candidates


def draw_contour_overlay(
    image_bgr: np.ndarray,
    candidates: list[ContourCandidate],
    max_count: int = 400,
) -> np.ndarray:
    overlay = image_bgr.copy()
    for candidate in candidates[:max_count]:
        cv2.drawContours(overlay, [candidate.contour], -1, (0, 255, 0), 1, lineType=cv2.LINE_AA)
    return overlay
