"""Phase 1 inner-border crop recovery."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from armourcore_cds.utils.image_ops import to_gray


@dataclass
class InnerBorderCropResult:
    cropped_image: np.ndarray
    inner_rect_xyxy: tuple[int, int, int, int]
    border_samples_px: dict[str, int]
    preview_binary: np.ndarray


def _find_peak_run(profile: np.ndarray, prefer_start: bool) -> tuple[int, int]:
    if profile.size == 0:
        return 0, 0

    threshold = max(float(np.percentile(profile, 92)) * 0.55, 0.10)
    runs: list[tuple[int, int]] = []
    in_run = False
    start = 0

    for idx, value in enumerate(profile):
        if value >= threshold:
            if not in_run:
                start = idx
                in_run = True
        else:
            if in_run:
                runs.append((start, idx - 1))
                in_run = False

    if in_run:
        runs.append((start, len(profile) - 1))

    if not runs:
        peak = int(np.argmax(profile))
        return peak, peak

    best_run = runs[0]
    best_score = float('-inf')
    length = max(len(profile), 1)
    for run_start, run_end in runs:
        run_slice = profile[run_start:run_end + 1]
        run_strength = float(run_slice.mean())
        run_len = run_end - run_start + 1
        edge_distance = run_start if prefer_start else (length - 1 - run_end)
        score = (run_strength * run_len) - ((edge_distance / length) * 0.5)
        if score > best_score:
            best_score = score
            best_run = (run_start, run_end)
    return best_run


def find_inner_border_crop(rectified_image: np.ndarray) -> InnerBorderCropResult:
    gray = to_gray(rectified_image)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        9,
    )
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )

    h, w = rectified_image.shape[:2]
    search_y = max(20, int(round(h * 0.18)))
    search_x = max(20, int(round(w * 0.12)))

    centre_x0 = int(round(w * 0.12))
    centre_x1 = int(round(w * 0.88))
    centre_y0 = int(round(h * 0.12))
    centre_y1 = int(round(h * 0.88))

    top_strength = (binary[:search_y, centre_x0:centre_x1] > 0).mean(axis=1)
    bottom_strength = (binary[h - search_y:, centre_x0:centre_x1] > 0).mean(axis=1)
    left_strength = (binary[centre_y0:centre_y1, :search_x] > 0).mean(axis=0)
    right_strength = (binary[centre_y0:centre_y1, w - search_x:] > 0).mean(axis=0)

    _, top_end = _find_peak_run(top_strength, prefer_start=True)
    bottom_start, _ = _find_peak_run(bottom_strength, prefer_start=False)
    _, left_end = _find_peak_run(left_strength, prefer_start=True)
    right_start, _ = _find_peak_run(right_strength, prefer_start=False)

    y0 = int(np.clip(top_end + 1, 0, h - 2))
    y1 = int(np.clip((h - search_y) + bottom_start - 1, y0 + 1, h - 1))
    x0 = int(np.clip(left_end + 1, 0, w - 2))
    x1 = int(np.clip((w - search_x) + right_start - 1, x0 + 1, w - 1))

    if (x1 - x0) < (w * 0.50) or (y1 - y0) < (h * 0.50):
        fallback_pad_x = max(2, int(round(w * 0.01)))
        fallback_pad_y = max(2, int(round(h * 0.01)))
        x0 = fallback_pad_x
        x1 = w - fallback_pad_x
        y0 = fallback_pad_y
        y1 = h - fallback_pad_y

    cropped = rectified_image[y0:y1, x0:x1].copy()
    return InnerBorderCropResult(
        cropped_image=cropped,
        inner_rect_xyxy=(x0, y0, x1, y1),
        border_samples_px={
            'top': int(y0),
            'bottom': int(h - y1),
            'left': int(x0),
            'right': int(w - x1),
        },
        preview_binary=binary,
    )
