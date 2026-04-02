from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def ensure_uint8_bgr(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_long_edge(image: np.ndarray, max_long_edge: int) -> np.ndarray:
    h, w = image.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_long_edge:
        return image.copy()
    scale = max_long_edge / float(long_edge)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = ensure_uint8_bgr(image)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise IOError(f'Failed to write image: {path}')


def draw_polygon(image: np.ndarray, points: np.ndarray, colour: tuple[int, int, int], thickness: int = 3) -> np.ndarray:
    canvas = ensure_uint8_bgr(image).copy()
    pts = points.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(canvas, [pts], isClosed=True, color=colour, thickness=thickness, lineType=cv2.LINE_AA)
    return canvas


def stack_debug_h(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = ensure_uint8_bgr(a)
    b = ensure_uint8_bgr(b)
    if a.shape[0] != b.shape[0]:
        target_h = min(a.shape[0], b.shape[0])
        a = cv2.resize(a, (int(round(a.shape[1] * target_h / a.shape[0])), target_h), interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, (int(round(b.shape[1] * target_h / b.shape[0])), target_h), interpolation=cv2.INTER_AREA)
    return np.hstack([a, b])
