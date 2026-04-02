from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from armourcore_cds.templates.calibration import mm_to_pixels


@dataclass
class ScalingResult:
    image: np.ndarray
    output_size_px: tuple[int, int]
    px_per_mm_x: float
    px_per_mm_y: float
    effective_dpi_x: float
    effective_dpi_y: float


def scale_to_template_design_area(image: np.ndarray, width_mm: float, height_mm: float, output_dpi: int) -> ScalingResult:
    target_w = max(1, int(round(mm_to_pixels(width_mm, output_dpi))))
    target_h = max(1, int(round(mm_to_pixels(height_mm, output_dpi))))
    scaled = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    px_per_mm_x = target_w / float(width_mm)
    px_per_mm_y = target_h / float(height_mm)
    effective_dpi_x = px_per_mm_x * 25.4
    effective_dpi_y = px_per_mm_y * 25.4
    return ScalingResult(
        image=scaled,
        output_size_px=(target_w, target_h),
        px_per_mm_x=px_per_mm_x,
        px_per_mm_y=px_per_mm_y,
        effective_dpi_x=effective_dpi_x,
        effective_dpi_y=effective_dpi_y,
    )
