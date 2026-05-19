"""Phase 2 pipeline — grid removal → trace isolation → contour extraction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from armourcore_cds.phase2.contour_detection import (
    ContourCandidate,
    detect_trace_contours,
    draw_contour_overlay,
)
from armourcore_cds.phase2.note_candidates import NoteCandidate, detect_note_candidate_mask
from armourcore_cds.phase2.trace_isolation import (
    GridColour,
    TraceIsolationResult,
    isolate_trace_candidates,
)
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
    grid_colour: GridColour


def _mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if mask.ndim == 2 else mask


def run_phase2_pipeline(
    image_bgr: np.ndarray,
    template: TemplateModel,
    run_dir: Path | None = None,
    debug: DebugWriter | None = None,
    grid_colour: GridColour = "auto",
    max_processing_dim: int | None = 1500,
) -> Phase2PipelineResult:
    """Run the full Phase 2 pipeline on a scaled design-area crop.

    Parameters
    ----------
    image_bgr:
        Output of Phase 1 ``scaled_design_area.png``.
    template:
        Matched template config.
    run_dir:
        Primary output images are saved here if provided.
    debug:
        Step-by-step debug images saved here if provided.
    grid_colour:
        ``"auto"`` (default) — detect orange vs black from the image.
        ``"orange"`` / ``"black"`` — force a specific mode.
    max_processing_dim:
        Downscale limit for the black-grid path.  ``None`` = full resolution.
    """
    isolation = isolate_trace_candidates(
        image_bgr=image_bgr,
        template=template,
        grid_colour=grid_colour,
        max_processing_dim=max_processing_dim,
    )
    contour_candidates = detect_trace_contours(isolation.trace_candidate_mask)
    note_candidate_mask, note_candidates = detect_note_candidate_mask(
        isolation.note_candidate_seed_mask
    )

    # Draw contour overlay at the same resolution as the trace mask (avoids
    # coordinate mismatch when trace mask was computed at reduced resolution).
    h_trace, w_trace = isolation.trace_candidate_mask.shape[:2]
    h_clean, w_clean = isolation.cleaned_bgr.shape[:2]
    if h_trace != h_clean or w_trace != w_clean:
        # Resize cleaned raster to trace-mask resolution for the overlay
        overlay_base = cv2.resize(
            isolation.cleaned_bgr, (w_trace, h_trace), interpolation=cv2.INTER_AREA
        )
    else:
        overlay_base = isolation.cleaned_bgr
    contour_overlay = draw_contour_overlay(overlay_base, contour_candidates)

    if debug is not None:
        debug.image("20_phase2_input", image_bgr)

        if isolation.grid_colour == "orange":
            debug.image("21_orange_mask", _mask_to_bgr(isolation.orange_mask))
        else:
            debug.image("21_black_grid_mask", _mask_to_bgr(isolation.black_grid_mask))
            debug.image("21b_black_removal", _mask_to_bgr(isolation.black_removal_mask))

        debug.image("22_removal_mask", _mask_to_bgr(isolation.removal_mask))
        debug.image("23_trace_cleaned_raster", isolation.cleaned_bgr)
        debug.image("24_trace_candidate_mask", _mask_to_bgr(isolation.trace_candidate_mask))
        debug.image("25_note_candidate_mask", _mask_to_bgr(note_candidate_mask))
        debug.image("26_contour_overlay", contour_overlay)

    if run_dir is not None:
        save_image(run_dir / "phase2_cleaned_raster.png", isolation.cleaned_bgr)
        save_image(run_dir / "phase2_orange_mask.png", isolation.orange_mask)
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
        grid_colour=isolation.grid_colour,
    )
