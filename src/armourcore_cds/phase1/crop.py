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


def _scan_dark_run(profile: np.ndarray, threshold: int, min_run: int) -> int:
    run = 0
    start = 0
    for idx, value in enumerate(profile):
        if value <= threshold:
            if run == 0:
                start = idx
            run += 1
            if run >= min_run:
                return start
        else:
            run = 0
    return 0


def _scan_dark_run_reverse(profile: np.ndarray, threshold: int, min_run: int) -> int:
    run = 0
    end = len(profile) - 1
    for rev_idx, value in enumerate(profile[::-1]):
        idx = len(profile) - 1 - rev_idx
        if value <= threshold:
            if run == 0:
                end = idx
            run += 1
            if run >= min_run:
                return end
        else:
            run = 0
    return len(profile) - 1


def find_inner_border_crop(rectified_image: np.ndarray) -> InnerBorderCropResult:
    gray = to_gray(rectified_image)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    margin_y = max(5, rectified_image.shape[0] // 20)
    margin_x = max(5, rectified_image.shape[1] // 20)

    top_band = blur[:margin_y, :]
    bottom_band = blur[-margin_y:, :]
    left_band = blur[:, :margin_x]
    right_band = blur[:, -margin_x:]

    top_profile = top_band.mean(axis=1)
    bottom_profile = bottom_band.mean(axis=1)
    left_profile = left_band.mean(axis=0)
    right_profile = right_band.mean(axis=0)

    threshold = int(np.clip(np.percentile(np.concatenate([top_profile, bottom_profile, left_profile, right_profile]), 35), 30, 220))
    min_run_y = max(3, margin_y // 6)
    min_run_x = max(3, margin_x // 6)

    top_outer = _scan_dark_run(top_profile, threshold, min_run_y)
    bottom_outer_local = _scan_dark_run_reverse(bottom_profile, threshold, min_run_y)
    left_outer = _scan_dark_run(left_profile, threshold, min_run_x)
    right_outer_local = _scan_dark_run_reverse(right_profile, threshold, min_run_x)

    top_band_dark = top_profile <= threshold
    left_band_dark = left_profile <= threshold
    bottom_band_dark = bottom_profile <= threshold
    right_band_dark = right_profile <= threshold

    top_inner = top_outer
    while top_inner < len(top_band_dark) and top_band_dark[top_inner]:
        top_inner += 1

    left_inner = left_outer
    while left_inner < len(left_band_dark) and left_band_dark[left_inner]:
        left_inner += 1

    bottom_inner_local = bottom_outer_local
    while bottom_inner_local >= 0 and bottom_band_dark[bottom_inner_local]:
        bottom_inner_local -= 1

    right_inner_local = right_outer_local
    while right_inner_local >= 0 and right_band_dark[right_inner_local]:
        right_inner_local -= 1

    x0 = int(np.clip(left_inner, 0, rectified_image.shape[1] - 2))
    y0 = int(np.clip(top_inner, 0, rectified_image.shape[0] - 2))
    x1 = int(np.clip(rectified_image.shape[1] - margin_x + right_inner_local, x0 + 1, rectified_image.shape[1] - 1))
    y1 = int(np.clip(rectified_image.shape[0] - margin_y + bottom_inner_local, y0 + 1, rectified_image.shape[0] - 1))

    cropped = rectified_image[y0:y1, x0:x1].copy()
    return InnerBorderCropResult(
        cropped_image=cropped,
        inner_rect_xyxy=(x0, y0, x1, y1),
        border_samples_px={
            'top': int(y0),
            'bottom': int(rectified_image.shape[0] - y1),
            'left': int(x0),
            'right': int(rectified_image.shape[1] - x1),
        },
        preview_binary=binary,
    )
