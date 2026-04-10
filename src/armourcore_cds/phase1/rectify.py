from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class RectifyResult:
    image: np.ndarray
    transform_matrix: np.ndarray
    rectified_size_px: tuple[int, int]


def _coerce_corners_xy(corners: object) -> np.ndarray:
    arr = np.asarray(corners, dtype=np.float32)
    if arr.shape != (4, 2):
        arr = arr.reshape(4, 2)
    return arr


def _target_size_from_corners(corners: object, expected_aspect_ratio: float) -> tuple[int, int]:
    corners_xy = _coerce_corners_xy(corners)
    tl, tr, br, bl = corners_xy
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)

    measured_w = max(width_a, width_b, 1.0)
    measured_h = max(height_a, height_b, 1.0)

    if measured_w / measured_h > expected_aspect_ratio:
        target_w = int(round(measured_w))
        target_h = int(round(target_w / expected_aspect_ratio))
    else:
        target_h = int(round(measured_h))
        target_w = int(round(target_h * expected_aspect_ratio))

    target_w = max(target_w, 64)
    target_h = max(target_h, 64)
    return target_w, target_h


def rectify_from_corners(image: np.ndarray, corners: object, expected_aspect_ratio: float) -> RectifyResult:
    corners_xy = _coerce_corners_xy(corners)
    target_w, target_h = _target_size_from_corners(corners_xy, expected_aspect_ratio)
    dst = np.array(
        [[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(corners_xy, dst)
    rectified = cv2.warpPerspective(image, matrix, (target_w, target_h), flags=cv2.INTER_CUBIC)
    return RectifyResult(image=rectified, transform_matrix=matrix, rectified_size_px=(target_w, target_h))
