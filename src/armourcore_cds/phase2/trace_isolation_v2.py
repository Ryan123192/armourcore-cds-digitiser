"""Phase 2 v2 — Lab-baseline + adaptive-paper-clamp trace isolation.

Sibling of the production ``trace_isolation.py`` (the "L11" two-stage
HSV + hard-white-clamp approach).  Designed to attack the failure
modes documented in
``docs/phase_02_failure_analysis_2026-05-22.md``:

  * blue/warm white-balance shifts that move the grid out of HSV range
  * shadows that look "orange" under warm light (over-detection)
  * dark shadows that survive the fixed gray ≥ 160 clamp
  * the lighter V2 minor grid sitting right at the global clamp threshold

Algorithm
=========

A.  ``normalise_lighting``  --  unchanged: CLAHE on the LAB-L channel
                                flattens the worst of the cross-frame
                                brightness gradient.

B.  Sample the paper Lab baseline AGAIN inside this module (fresh
    sample, not passed in from Phase 1).  Same maths as Phase 1:
    central-patch + brightest-25% + median.  Lighting-invariant.

    NOTE FOR FUTURE EFFICIENCY: Phase 1 already computes this baseline
    once for marker detection.  Plumbing it through and reusing would
    save ~5 ms per image.  Negligible right now but worth revisiting
    if/when Phase 2 wall time becomes a concern.

C.  STAGE 1  --  **Lab Δa* mask** (Path A): mark every pixel whose
    ``a* − paper_a*`` exceeds a small threshold AND whose colour is
    NOT a yellow-orange shift.  This catches the printed grid in
    every lighting condition we've seen because the *relative*
    chromaticity shift is the same under blue tint OR warm sun --
    only the absolute hue moves.  Tool ink (neutral chromaticity)
    and shadows (low L but neutral a*) are NOT caught.

D.  ``paper_blended_fill`` + ``normalise_paper_to_white``  --
    unchanged: replace masked grid pixels with the local paper
    colour, then lift paper to a consistent ~245 globally.

E.  STAGE 2  --  **Adaptive paper-relative clamp** (Path C): compute
    a local paper baseline via Gaussian blur (excluding the dark
    marks).  A pixel becomes pure white when it sits within
    ``ink_relative_gap`` grey levels of the LOCAL baseline -- not
    when it crosses an arbitrary global value of 160.  Dark crease
    shadows that don't actually contain ink get wiped because the
    local baseline reflects the shadow paper; tool ink in a shadow
    survives because it's still meaningfully darker than the
    locally-darker paper around it.

F.  ``build_trace_mask``  --  unchanged: dual abs + relative dark
    detection on the cleaned image produces the binary trace mask
    that Phase 3 consumes.

The module returns the same ``TraceIsolationResult`` dataclass as the
production module so any downstream code can swap modules without
changes.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from armourcore_cds.templates.models import TemplateModel
# Re-use existing helpers from the production module so we don't fork
# every utility -- only the algorithm itself changes here.
from armourcore_cds.phase2.trace_isolation import (
    GridColour,
    TraceIsolationResult,
    normalise_lighting,
    normalise_paper_to_white,
    paper_blended_fill,
    build_trace_mask,
    _orange_coverage,
    isolate_trace_candidates as _isolate_legacy,
)


# ---------------------------------------------------------------------------
# Tuning constants — start conservative; tweak after seeing the per-case dumps
# ---------------------------------------------------------------------------

# Lab Δa* thresholds.  Phase 1's MARKER detector uses 12 because markers are
# vivid red ink; the V2 grid colours (FFEBCC / FFC466) are much paler so we
# start lower.  Empirically: a* shift of ~6-8 is usually enough to catch
# the lightest grid even under blue tint; anything below that risks
# false-positives on slightly-warm paper.
LAB_DELTA_A_MIN          = 6        # min a* shift to count a pixel as grid
LAB_ORANGE_TOLERANCE     = 8        # max (Δb - Δa) -- rejects yellow / wood

# Paper baseline sampling
PAPER_PATCH_FRAC         = 0.10
PAPER_BRIGHT_PCT         = 75

# Stage-1 morph close (anti-aliasing of grid line edges only).  Small so
# we never bridge across customer ink.
STAGE1_CLOSE_KSIZE       = 3

# Stage-1 minimum component size (drops single-pixel chromatic noise)
STAGE1_MIN_COMPONENT_PX  = 4

# Stage-2 adaptive clamp
ADAPTIVE_PAPER_SIGMA     = 80.0
ADAPTIVE_DARK_THRESHOLD  = 90       # pixels below this are excluded from
                                    # the local-paper-baseline estimate
ADAPTIVE_INK_GAP         = 50       # a pixel within this many grey levels
                                    # of the local paper baseline gets
                                    # clamped to pure white; further
                                    # below is considered ink and kept


# ===========================================================================
# Lab paper baseline + Δa* mask
# ===========================================================================

def estimate_paper_lab(lab_uint8: np.ndarray) -> Tuple[float, float, float]:
    """Median Lab of the brightest pixels in a central image patch.

    Identical maths to Phase 1's helper but kept local so v2 has no
    cross-module dependency on phase1.
    """
    h, w = lab_uint8.shape[:2]
    fx, fy = int(w * PAPER_PATCH_FRAC), int(h * PAPER_PATCH_FRAC)
    cx, cy = w // 2, h // 2
    central = lab_uint8[cy - fy: cy + fy, cx - fx: cx + fx].reshape(-1, 3)
    L = central[:, 0]
    bright_thresh = np.percentile(L, PAPER_BRIGHT_PCT)
    chosen = central[L >= bright_thresh]
    return tuple(float(v) for v in np.median(chosen, axis=0))


def lab_delta_a_grid_mask(
    image_bgr: np.ndarray,
    paper_lab: Tuple[float, float, float],
    *,
    delta_a_min: int = LAB_DELTA_A_MIN,
    orange_tolerance: int = LAB_ORANGE_TOLERANCE,
    close_px: int = STAGE1_CLOSE_KSIZE,
    min_component_px: int = STAGE1_MIN_COMPONENT_PX,
) -> np.ndarray:
    """Build a binary mask of pixels chromatically shifted toward orange.

    For each pixel:
        delta_a = a - paper_a*       (positive == redder than paper)
        delta_b = b - paper_b*       (positive == yellower than paper)
    A pixel is grid-ink-like iff
        delta_a >= delta_a_min  AND  (delta_b - delta_a) <= orange_tolerance

    The first half catches every red-warm-shifted pixel.  The second
    rejects yellow / brown shifts -- wood backgrounds, fingers, warm
    paper edges -- which have higher delta_b than delta_a.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
    _, ap, bp = paper_lab
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    delta_a = a - ap
    delta_b = b - bp
    mask = ((delta_a >= delta_a_min)
            & ((delta_b - delta_a) <= orange_tolerance)).astype(np.uint8) * 255

    if close_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_px * 2 + 1, close_px * 2 + 1)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    if min_component_px > 0:
        # Drop sub-min specks -- LUT-based to avoid Python loops
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8,
        )
        areas = stats[:, cv2.CC_STAT_AREA]
        keep = np.where(areas >= min_component_px, np.uint8(255), np.uint8(0))
        keep[0] = 0   # always drop the background label
        mask = keep[labels].astype(np.uint8)
    return mask


# ===========================================================================
# Stage 2 — adaptive paper-relative clamp
# ===========================================================================

def adaptive_paper_clamp(
    image_bgr: np.ndarray,
    *,
    paper_sigma: float = ADAPTIVE_PAPER_SIGMA,
    dark_mark_threshold: int = ADAPTIVE_DARK_THRESHOLD,
    ink_relative_gap: int = ADAPTIVE_INK_GAP,
) -> Tuple[np.ndarray, np.ndarray]:
    """Clamp pixels CLOSE to their local paper baseline to pure white.

    Replaces the global ``gray >= 160`` clamp used in the L11 module.
    Per pixel:
        diff = local_paper - gray
        diff <  ink_relative_gap  ->  paper-like  -> set to white
        diff >= ink_relative_gap  ->  ink-like    -> leave unchanged

    ``local_paper`` is a per-pixel Gaussian-blurred estimate of paper
    brightness that EXCLUDES the dark-mark pixels from the estimate.
    So in a shadow region surrounded by bright paper, the baseline
    near the centre of the shadow reflects the shadow's own paper
    value -- which means a true ink line inside the shadow still has
    a large positive diff and survives.

    Returns ``(clamped_bgr, clamp_mask)`` -- the boolean mask is
    saved alongside for diagnostics.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    paper_only = gray.copy()
    weight = np.ones_like(gray, dtype=np.float32)
    dark = gray < float(dark_mark_threshold)
    paper_only[dark] = 0.0
    weight[dark] = 0.0

    bg_num = cv2.GaussianBlur(paper_only, (0, 0), paper_sigma)
    bg_den = cv2.GaussianBlur(weight,     (0, 0), paper_sigma)
    local_paper = bg_num / np.maximum(bg_den, 1e-3)
    local_paper = np.maximum(local_paper, 1.0)  # numerical safety

    diff = local_paper - gray
    is_paperish = diff < float(ink_relative_gap)

    out = image_bgr.copy()
    out[is_paperish] = (255, 255, 255)
    return out, is_paperish.astype(np.uint8) * 255


# ===========================================================================
# Public entry point
# ===========================================================================

def isolate_trace_candidates_v2(
    image_bgr: np.ndarray,
    template: TemplateModel,
    grid_colour: GridColour = "auto",
    orange_threshold: float = 0.42,
    max_processing_dim: int = 1500,
    *,
    delta_a_min: int = LAB_DELTA_A_MIN,
    orange_tolerance: int = LAB_ORANGE_TOLERANCE,
    ink_relative_gap: int = ADAPTIVE_INK_GAP,
    paper_sigma: float = ADAPTIVE_PAPER_SIGMA,
) -> TraceIsolationResult:
    """Phase 2 v2 -- Lab-baseline + adaptive-paper-clamp cleanup.

    Same return shape as ``isolate_trace_candidates``.  Falls back to
    the legacy production routine when the auto-detect probe decides
    the grid is BLACK -- v2 only changes the ORANGE path.

    Parameters of interest (all keyword-only) for experimentation:
        delta_a_min       -- Stage-1 Lab Δa* threshold (default 6)
        orange_tolerance  -- Stage-1 Δb − Δa rejection (default 8)
        ink_relative_gap  -- Stage-2 paper-relative clamp gap (default 50)
        paper_sigma       -- Gaussian sigma for the local baseline (default 80)
    """
    h_orig, w_orig = image_bgr.shape[:2]

    # ---- A. Lighting normalisation (unchanged) ----
    image_bgr = normalise_lighting(image_bgr)

    # ---- Auto-detect grid colour (unchanged) ----
    if grid_colour == "auto":
        coverage = _orange_coverage(image_bgr)
        grid_colour = "orange" if coverage >= 0.005 else "black"

    # Black-grid path is untouched -- defer to the production routine.
    if grid_colour != "orange":
        return _isolate_legacy(
            image_bgr,
            template,
            grid_colour=grid_colour,
            orange_threshold=orange_threshold,
            max_processing_dim=max_processing_dim,
        )

    # ---- B. Sample paper Lab baseline ----
    lab_uint8 = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    paper_lab = estimate_paper_lab(lab_uint8)

    # ---- C. Stage 1: Lab Δa* mask ----
    orange_mask = lab_delta_a_grid_mask(
        image_bgr,
        paper_lab=paper_lab,
        delta_a_min=delta_a_min,
        orange_tolerance=orange_tolerance,
    )

    # ---- D. Paper-blended fill + normalise paper to white ----
    cleaned_bgr = paper_blended_fill(image_bgr, orange_mask)
    cleaned_bgr = normalise_paper_to_white(cleaned_bgr)

    # ---- E. Stage 2: adaptive paper-relative clamp ----
    cleaned_bgr, clamp_mask = adaptive_paper_clamp(
        cleaned_bgr,
        paper_sigma=paper_sigma,
        ink_relative_gap=ink_relative_gap,
    )

    # ---- F. Trace mask (unchanged) ----
    if max_processing_dim and max(h_orig, w_orig) > max_processing_dim:
        scale = max_processing_dim / max(h_orig, w_orig)
        cleaned_proc = cv2.resize(
            cleaned_bgr,
            (
                max(1, int(round(w_orig * scale))),
                max(1, int(round(h_orig * scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )
    else:
        cleaned_proc = cleaned_bgr
    trace_mask = build_trace_mask(cleaned_proc)

    return TraceIsolationResult(
        cleaned_bgr=cleaned_bgr,
        orange_mask=orange_mask,
        trace_candidate_mask=trace_mask,
        note_candidate_seed_mask=trace_mask.copy(),
        removal_mask=orange_mask,
        grid_colour="orange",
        orange_grid_mask=orange_mask,
    )
