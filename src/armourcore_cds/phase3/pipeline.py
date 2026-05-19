"""Phase 3 pipeline — vectorise cleaned raster into smooth closed curves.

Consumes Phase 2 outputs (``phase2_cleaned_raster.png`` and
``phase2_trace_candidate_mask.png``) and produces:

* ``phase3_vectors.svg``   — scale-accurate SVG (coordinates in mm)
* ``phase3_overlay.png``   — Bézier curves drawn on the cleaned raster
* Debug images numbered 30–39
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from armourcore_cds.phase3.vectorise import (
    VectorPath,
    extract_vector_paths,
    render_vector_overlay,
    repair_orange_gaps,
    write_svg,
)
from armourcore_cds.templates.models import TemplateModel
from armourcore_cds.utils.debug import DebugWriter
from armourcore_cds.utils.image_ops import save_image


@dataclass
class Phase3PipelineResult:
    vector_paths: list[VectorPath]
    svg_path: Path
    overlay_path: Path
    mask_shape: tuple[int, int]   # (H, W) of the trace mask used
    n_paths: int


def run_phase3_pipeline(
    cleaned_bgr: np.ndarray,
    trace_mask: np.ndarray,
    template: TemplateModel,
    orange_mask: np.ndarray | None = None,
    run_dir: Path | None = None,
    debug: DebugWriter | None = None,
    min_area_px: float = 200.0,
    rdp_epsilon: float = 3.0,
    tension: float = 1.0,
    gap_close_px: int = 7,
    max_gap_close_px: int = 31,
    circularity_min: float = 0.05,
    orange_bridge_px: int = 20,
    orange_extra_passes: int = 4,
) -> Phase3PipelineResult:
    """Vectorise Phase 2 raster outputs into smooth cubic Bezier paths.

    Parameters
    ----------
    cleaned_bgr :
        ``phase2_cleaned_raster.png`` — used as the overlay base image.
    trace_mask :
        ``phase2_trace_candidate_mask.png`` — binary mask identifying trace
        pixels.  May be at reduced resolution relative to *cleaned_bgr*.
    template :
        Matched template config (supplies physical mm dimensions).
    run_dir :
        Primary output images saved here (SVG + overlay PNG).
    debug :
        Step-by-step debug images saved here if provided.
    orange_mask :
        Full-resolution orange removal mask (``phase2_orange_mask.png``).
        When provided, gaps left by grid removal are filled before contouring.
        Pass ``None`` to skip the repair (black-grid images).
    min_area_px :
        Minimum contour area in trace_mask pixels.
    rdp_epsilon :
        RDP simplification tolerance in trace_mask pixels.
    tension :
        Catmull-Rom tension (1.0 = standard smooth).
    gap_close_px :
        Residual MORPH_CLOSE kernel after the orange repair pass (mask pixels).
        Default 7 -- handles any tiny remaining breaks.
    max_gap_close_px :
        Max kernel for the per-ribbon local retry.  Default 31.
    circularity_min :
        Isoperimetric ratio threshold; contours below this are ribbons.
    orange_bridge_px :
        Directional bridge half-width (full-res pixels) for the orange repair.
        Default 20 (~1.3 mm at 5512px/350mm).  Raise for thicker grid lines.  Raise for thicker grid lines.
    """
    # --- orange-gap repair ---------------------------------------------------
    # Rebuild the trace mask from the full-res cleaned raster, filling gaps
    # where the orange grid was removed.  This is done before any morphological
    # close so the fix is geometrically confined to the grid footprint.
    if orange_mask is not None and np.any(orange_mask > 0):
        repaired_mask, _bridge_mask, _hermite_mask = repair_orange_gaps(
            cleaned_bgr=cleaned_bgr,
            orange_mask=orange_mask,
            bridge_px=orange_bridge_px,
            extra_passes=orange_extra_passes,
            max_processing_dim=3000,
        )
    else:
        repaired_mask = trace_mask

    mask_shape = repaired_mask.shape[:2]  # (H, W)

    # --- extract paths -------------------------------------------------------
    paths = extract_vector_paths(
        repaired_mask,
        min_area_px=min_area_px,
        rdp_epsilon=rdp_epsilon,
        tension=tension,
        gap_close_px=gap_close_px,
        max_gap_close_px=max_gap_close_px,
        circularity_min=circularity_min,
    )

    # --- physical dimensions from template -----------------------------------
    design_w_mm = float(template.design_area_mm.width)
    design_h_mm = float(template.design_area_mm.height)

    # --- SVG -----------------------------------------------------------------
    svg_path = (run_dir / "phase3_vectors.svg") if run_dir else Path("phase3_vectors.svg")
    write_svg(
        paths=paths,
        output_path=svg_path,
        mask_shape=mask_shape,
        design_width_mm=design_w_mm,
        design_height_mm=design_h_mm,
    )

    # --- overlay PNG ---------------------------------------------------------
    overlay = render_vector_overlay(
        base_bgr=cleaned_bgr,
        paths=paths,
        mask_shape=mask_shape,
        colour_bgr=(0, 48, 230),   # vivid red drawn in BGR
        thickness=3,
    )
    overlay_path = (run_dir / "phase3_overlay.png") if run_dir else Path("phase3_overlay.png")
    save_image(overlay_path, overlay)

    # --- debug ---------------------------------------------------------------
    if debug is not None:
        debug.image("30_phase3_repaired_mask", repaired_mask)
        debug.image("31_phase3_overlay", overlay)

    # --- console summary -----------------------------------------------------
    total_nodes = sum(len(p.points) for p in paths)
    total_segs = sum(len(p.bezier_segments) for p in paths)
    print(
        f"  Phase 3: {len(paths)} paths  |  {total_nodes} nodes  "
        f"|  {total_segs} Bezier segments  |  -> {svg_path.name}"
    )

    return Phase3PipelineResult(
        vector_paths=paths,
        svg_path=svg_path,
        overlay_path=overlay_path,
        mask_shape=mask_shape,
        n_paths=len(paths),
    )
