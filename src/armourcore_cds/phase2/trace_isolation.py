"""Phase 2 module: trace_isolation"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from armourcore_cds.templates.models import TemplateModel


ORANGE_GRID_HEXES: tuple[str, ...] = (
    "#D35400",
    "#F39C12",
    "#FAD7A0",
)


@dataclass
class TraceIsolationResult:
    cleaned_bgr: np.ndarray
    trace_candidate_mask: np.ndarray
    note_candidate_seed_mask: np.ndarray
    projected_grid_mask: np.ndarray
    orange_likelihood: np.ndarray
    orange_grid_mask: np.ndarray
    protect_mask: np.ndarray
    removal_mask: np.ndarray
    relative_darkness: np.ndarray


def _hex_to_bgr(hex_colour: str) -> tuple[int, int, int]:
    value = hex_colour.strip().lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected 6-digit hex colour, got: {hex_colour}")
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    return (b, g, r)


def _remove_small_components(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area >= min_area_px:
            cleaned[labels == idx] = 255
    return cleaned


def _palette_lab(hexes: tuple[str, ...]) -> np.ndarray:
    palette_bgr = np.array([_hex_to_bgr(x) for x in hexes], dtype=np.uint8)
    palette_lab = cv2.cvtColor(palette_bgr[None, :, :], cv2.COLOR_BGR2LAB)[0]
    return palette_lab.astype(np.float32)


def _min_lab_distance_map(image_bgr: np.ndarray, palette_lab: np.ndarray) -> np.ndarray:
    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    diffs = image_lab[:, :, None, :] - palette_lab[None, None, :, :]
    dists = np.linalg.norm(diffs, axis=3)
    return dists.min(axis=2)


def _build_orange_likelihood_map(
    image_bgr: np.ndarray,
    max_lab_distance: float = 55.0,
    sat_floor: float = 0.12,
) -> np.ndarray:
    palette_lab = _palette_lab(ORANGE_GRID_HEXES)
    min_dist = _min_lab_distance_map(image_bgr, palette_lab)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32) / 255.0

    colour_score = 1.0 - np.clip(min_dist / max_lab_distance, 0.0, 1.0)
    sat_score = np.clip((sat - sat_floor) / max(1e-6, (1.0 - sat_floor)), 0.0, 1.0)

    orange_likelihood = np.clip((0.7 * colour_score) + (0.3 * sat_score), 0.0, 1.0)
    return orange_likelihood.astype(np.float32)


def _line_positions_mm(total_mm: float, spacing_mm: float) -> list[float]:
    positions: list[float] = []
    value = 0.0
    while value <= (total_mm + 1e-6):
        positions.append(value)
        value += spacing_mm
    return positions


def _build_projected_grid_mask(
    image_bgr: np.ndarray,
    template: TemplateModel,
    minor_line_px: int = 1,
    major_line_px: int = 2,
    dilation_px: int = 1,
) -> np.ndarray:
    h, w = image_bgr.shape[:2]

    grid_minor_mm = float(template.grid.minor_spacing_mm if template.grid else 10.0)
    grid_major_mm = float(template.grid.major_spacing_mm if template.grid else 50.0)

    px_per_mm_x = w / float(template.design_area_mm.width)
    px_per_mm_y = h / float(template.design_area_mm.height)

    mask = np.zeros((h, w), dtype=np.uint8)

    for x_mm in _line_positions_mm(float(template.design_area_mm.width), grid_minor_mm):
        x = int(round(x_mm * px_per_mm_x))
        x = int(np.clip(x, 0, w - 1))
        is_major = abs((x_mm / grid_major_mm) - round(x_mm / grid_major_mm)) < 1e-6
        thickness = major_line_px if is_major else minor_line_px
        cv2.line(mask, (x, 0), (x, h - 1), 255, thickness=thickness, lineType=cv2.LINE_AA)

    for y_mm in _line_positions_mm(float(template.design_area_mm.height), grid_minor_mm):
        y = int(round(y_mm * px_per_mm_y))
        y = int(np.clip(y, 0, h - 1))
        is_major = abs((y_mm / grid_major_mm) - round(y_mm / grid_major_mm)) < 1e-6
        thickness = major_line_px if is_major else minor_line_px
        cv2.line(mask, (0, y), (w - 1, y), 255, thickness=thickness, lineType=cv2.LINE_AA)

    if dilation_px > 0:
        kernel = np.ones((2 * dilation_px + 1, 2 * dilation_px + 1), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def _build_dark_trace_protect_mask(
    image_bgr: np.ndarray,
    rel_dark_thresh: int = 10,
    abs_dark_thresh: int = 78,
    sat_max_for_graphite: int = 145,
) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]

    local_bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=7.0, sigmaY=7.0)
    relative_darkness = cv2.subtract(local_bg, gray)
    absolute_darkness = cv2.subtract(np.full_like(gray, 255), gray)

    edges = cv2.Canny(gray, threshold1=40, threshold2=120)

    protect = (
        ((relative_darkness >= rel_dark_thresh) & (sat <= sat_max_for_graphite))
        | (absolute_darkness >= abs_dark_thresh)
        | (edges > 0)
    ).astype(np.uint8) * 255

    protect = cv2.dilate(protect, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return protect, relative_darkness


def isolate_trace_candidates(
    image_bgr: np.ndarray,
    template: TemplateModel,
    orange_threshold: float = 0.42,
) -> TraceIsolationResult:
    projected_grid_mask = _build_projected_grid_mask(image_bgr, template)
    orange_likelihood = _build_orange_likelihood_map(image_bgr)
    protect_mask, relative_darkness = _build_dark_trace_protect_mask(image_bgr)

    orange_grid_mask = (
        ((orange_likelihood >= orange_threshold) & (projected_grid_mask > 0)).astype(np.uint8) * 255
    )
    orange_grid_mask = cv2.morphologyEx(
        orange_grid_mask,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )

    removal_mask = cv2.bitwise_and(orange_grid_mask, cv2.bitwise_not(protect_mask))
    removal_mask = _remove_small_components(removal_mask, min_area_px=8)

    background_bgr = cv2.GaussianBlur(image_bgr, (0, 0), sigmaX=5.0, sigmaY=5.0)
    cleaned_bgr = image_bgr.copy()
    cleaned_bgr[removal_mask > 0] = background_bgr[removal_mask > 0]

    gray_clean = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
    local_bg_clean = cv2.GaussianBlur(gray_clean, (0, 0), sigmaX=7.0, sigmaY=7.0)
    rel_dark_clean = cv2.subtract(local_bg_clean, gray_clean)
    abs_dark_clean = cv2.subtract(np.full_like(gray_clean, 255), gray_clean)

    trace_candidate_mask = (
        ((rel_dark_clean >= 8) | (abs_dark_clean >= 70) | (protect_mask > 0)).astype(np.uint8) * 255
    )
    trace_candidate_mask = cv2.bitwise_and(trace_candidate_mask, cv2.bitwise_not(removal_mask))
    trace_candidate_mask = _remove_small_components(trace_candidate_mask, min_area_px=6)

    note_candidate_seed_mask = trace_candidate_mask.copy()

    return TraceIsolationResult(
        cleaned_bgr=cleaned_bgr,
        trace_candidate_mask=trace_candidate_mask,
        note_candidate_seed_mask=note_candidate_seed_mask,
        projected_grid_mask=projected_grid_mask,
        orange_likelihood=orange_likelihood,
        orange_grid_mask=orange_grid_mask,
        protect_mask=protect_mask,
        removal_mask=removal_mask,
        relative_darkness=relative_darkness,
    )
