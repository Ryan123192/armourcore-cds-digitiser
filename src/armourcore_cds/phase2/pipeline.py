"""Phase 2 module: pipeline"""
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
