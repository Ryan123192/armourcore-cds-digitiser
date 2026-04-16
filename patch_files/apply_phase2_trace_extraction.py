from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil
import sys

FILES: dict[str, str] = {
    "src/armourcore_cds/templates/models.py": '''from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DesignArea(BaseModel):
    width: float
    height: float


class GridModel(BaseModel):
    minor_spacing_mm: float = 10.0
    major_spacing_mm: float = 50.0


class ColourHintsModel(BaseModel):
    outer_border_hex_candidates: list[str] = Field(default_factory=list)
    fiducial_hex_candidates: list[str] = Field(default_factory=list)


class BorderDetectionModel(BaseModel):
    use_colour_hint: bool | None = None
    use_shape_constraints: bool | None = None
    fallback_to_fiducials: bool | None = None
    colour_hints: ColourHintsModel | None = None


class TemplateOutputsModel(BaseModel):
    write_rectified: bool | None = None
    write_scaled: bool | None = None
    write_cropped: bool | None = None
    write_debug: bool | None = None


class TemplateModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    template_id: str
    display_name: str
    design_area_mm: DesignArea
    crop_rule: str
    preferred_output_dpi: int
    primary_geometry_truth: str
    fiducials_enabled: bool

    grid: GridModel | None = None
    border_detection: BorderDetectionModel | None = None
    outputs: TemplateOutputsModel | None = None
''',
    "src/armourcore_cds/phase2/trace_isolation.py": '''"""Phase 2 module: trace_isolation"""
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
''',
    "src/armourcore_cds/phase2/contour_detection.py": '''"""Phase 2 module: contour_detection"""
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
''',
    "src/armourcore_cds/phase2/note_candidates.py": '''"""Phase 2 module: note_candidates"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class NoteCandidate:
    bbox_xywh: tuple[int, int, int, int]
    area_px: int


def detect_note_candidate_mask(
    seed_mask: np.ndarray,
    max_width_fraction: float = 0.18,
    max_height_fraction: float = 0.12,
    min_area_px: int = 10,
) -> tuple[np.ndarray, list[NoteCandidate]]:
    h, w = seed_mask.shape[:2]
    max_w = int(round(w * max_width_fraction))
    max_h = int(round(h * max_height_fraction))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(seed_mask, connectivity=8)

    note_mask = np.zeros_like(seed_mask)
    candidates: list[NoteCandidate] = []

    for idx in range(1, num_labels):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        bw = int(stats[idx, cv2.CC_STAT_WIDTH])
        bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])

        if area < min_area_px:
            continue
        if bw > max_w or bh > max_h:
            continue

        note_mask[labels == idx] = 255
        candidates.append(NoteCandidate(bbox_xywh=(x, y, bw, bh), area_px=area))

    return note_mask, candidates
''',
    "src/armourcore_cds/phase2/pipeline.py": '''"""Phase 2 module: pipeline"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from armourcore_cds.phase2.contour_detection import (
    ContourCandidate,
    detect_trace_contours,
    draw_contour_overlay,
)
from armourcore_cds.phase2.note_candidates import NoteCandidate, detect_note_candidate_mask
from armourcore_cds.phase2.trace_isolation import TraceIsolationResult, isolate_trace_candidates
from armourcore_cds.templates.models import TemplateModel
from armourcore_cds.utils.debug import DebugWriter
from armourcore_cds.utils.image_ops import save_image


@dataclass
class Phase2PipelineResult:
    cleaned_raster_bgr: np.ndarray
    trace_candidate_mask: np.ndarray
    note_candidate_mask: np.ndarray
    contour_candidates: list[ContourCandidate]
    note_candidates: list[NoteCandidate]
    isolation: TraceIsolationResult


def _float_map_to_heatmap(values: np.ndarray) -> np.ndarray:
    scaled = np.clip(values * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)


def _mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 2:
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return mask


def run_phase2_pipeline(
    image_bgr: np.ndarray,
    template: TemplateModel,
    run_dir: Path | None = None,
    debug: DebugWriter | None = None,
) -> Phase2PipelineResult:
    isolation = isolate_trace_candidates(image_bgr=image_bgr, template=template)
    contour_candidates = detect_trace_contours(isolation.trace_candidate_mask)
    note_candidate_mask, note_candidates = detect_note_candidate_mask(isolation.note_candidate_seed_mask)
    contour_overlay = draw_contour_overlay(isolation.cleaned_bgr, contour_candidates)

    if debug is not None:
        debug.image("20_phase2_input", image_bgr)
        debug.image("21_grid_projection_mask", _mask_to_bgr(isolation.projected_grid_mask))
        debug.image("22_orange_likelihood", _float_map_to_heatmap(isolation.orange_likelihood))
        debug.image("23_orange_grid_mask", _mask_to_bgr(isolation.orange_grid_mask))
        debug.image("24_protect_mask", _mask_to_bgr(isolation.protect_mask))
        debug.image("25_removal_mask", _mask_to_bgr(isolation.removal_mask))
        debug.image("26_trace_cleaned_raster", isolation.cleaned_bgr)
        debug.image("27_trace_candidate_mask", _mask_to_bgr(isolation.trace_candidate_mask))
        debug.image("28_note_candidate_mask", _mask_to_bgr(note_candidate_mask))
        debug.image("29_contour_overlay", contour_overlay)

    if run_dir is not None:
        save_image(run_dir / "phase2_cleaned_raster.png", isolation.cleaned_bgr)
        save_image(run_dir / "phase2_trace_candidate_mask.png", isolation.trace_candidate_mask)
        save_image(run_dir / "phase2_note_candidate_mask.png", note_candidate_mask)
        save_image(run_dir / "phase2_contour_overlay.png", contour_overlay)

    return Phase2PipelineResult(
        cleaned_raster_bgr=isolation.cleaned_bgr,
        trace_candidate_mask=isolation.trace_candidate_mask,
        note_candidate_mask=note_candidate_mask,
        contour_candidates=contour_candidates,
        note_candidates=note_candidates,
        isolation=isolation,
    )
''',
}


def main() -> int:
    repo_root = Path.cwd()
    expected = repo_root / "src" / "armourcore_cds"
    if not expected.exists():
        print("Error: run this script from the repo root that contains src/armourcore_cds")
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = repo_root / ".backup_phase2_trace_extraction" / timestamp
    backup_root.mkdir(parents=True, exist_ok=True)

    print(f"Repo root: {repo_root}")
    print(f"Backup dir: {backup_root}")

    for rel_path, content in FILES.items():
        target = repo_root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)

        if target.exists():
            backup_target = backup_root / rel_path
            backup_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup_target)
            print(f"Backed up: {rel_path}")
        else:
            print(f"Creating new file: {rel_path}")

        target.write_text(content, encoding="utf-8", newline="\n")
        print(f"Wrote: {rel_path}")

    print("\nDone.")
    print("Next steps:")
    print("  git diff -- src/armourcore_cds/templates/models.py src/armourcore_cds/phase2/trace_isolation.py src/armourcore_cds/phase2/contour_detection.py src/armourcore_cds/phase2/note_candidates.py src/armourcore_cds/phase2/pipeline.py")
    print("  python -m pytest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
