"""Phase 3 - vectorise_v2 (sibling, non-invasive).

Wraps the existing ``vectorise.extract_vector_paths`` and adds:

1. ``filter_paths`` - subtractive post-pass that drops:
     * paths smaller than a minimum AREA (mm^2)              -> kills text + dust
     * paths smaller than a minimum bbox-diagonal (mm)        -> kills tiny letters
     * paths whose bbox aspect ratio is too extreme           -> kills slithers
     * paths whose area is tiny relative to bbox area         -> kills hollow stubs

2. ``extract_vector_paths_tuned`` - convenience wrapper that picks
   gap-close kernel sizes based on a route hint ("pen"/"pencil"), then
   filters.  Pencil traces have wider natural gaps than pen, so pencil
   needs a bigger close.

The original ``extract_vector_paths`` is untouched - this module is a
strict superset.

Coordinate notes
================
* Paths come back in MASK-PIXEL space (origin top-left of trace mask).
* ``filter_paths`` needs the mask shape + the physical design dimensions
  so it can convert size thresholds in mm to pixel equivalents.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from armourcore_cds.phase3.vectorise import (
    VectorPath, extract_vector_paths,
)


Route = Literal["pen", "pencil", "auto"]


# ---------------------------------------------------------------------------
# Per-route Phase 3 parameter presets
# ---------------------------------------------------------------------------

ROUTE_PRESETS: dict[str, dict] = {
    "pen": {
        # Global close kept small (15) to avoid merging adjacent shapes
        # on tight-layout images (img1 merged into 1 blob at gap=19).
        # max_gap_close raised to 91 - this is the per-ribbon local retry
        # which only operates within a ribbon's bbox so it CAN'T merge
        # separate shapes, only bridge wider gaps inside a single ribbon.
        "gap_close_px":      15,
        "max_gap_close_px":  91,
        "min_area_px":       1000,
        "rdp_epsilon":       3.0,
        "circularity_min":   0.04,
    },
    "pencil": {
        # Pencil traces have wider natural gaps (graphite breaks) plus
        # paper texture interruption AND the L11 trace mask is sparse,
        # so we need a much bigger close to bridge stroke skips.
        "gap_close_px":      41,
        "max_gap_close_px":  151,
        "min_area_px":       400,
        "rdp_epsilon":       3.0,
        "circularity_min":   0.02,
    },
}


# Per-route filter presets - pencil is more lenient because outlines
# are thinner and easier to fragment.
ROUTE_FILTER_PRESETS: dict[str, dict] = {
    "pen": {
        "min_area_mm2":      100.0,  # 10 x 10 mm
        "min_bbox_diag_mm":  8.0,
        "max_aspect_ratio":  20.0,
        "min_area_to_bbox":  0.01,
    },
    "pencil": {
        # Drop only obvious noise; keep fragmented outlines so user
        # can see what's recoverable.
        "min_area_mm2":      30.0,   # ~5 x 6 mm
        "min_bbox_diag_mm":  6.0,
        "max_aspect_ratio":  30.0,
        "min_area_to_bbox":  0.005,
    },
}


# ---------------------------------------------------------------------------
# Subtractive post-filter
# ---------------------------------------------------------------------------

@dataclass
class FilterReport:
    """What got dropped, why - for diagnostics."""
    kept: int = 0
    dropped_small_area: int = 0
    dropped_short_diag: int = 0
    dropped_slither: int = 0
    dropped_hollow: int = 0

    def total_dropped(self) -> int:
        return (self.dropped_small_area + self.dropped_short_diag
                + self.dropped_slither + self.dropped_hollow)

    def __str__(self) -> str:
        return (f"kept={self.kept}, dropped: "
                f"small={self.dropped_small_area} "
                f"short={self.dropped_short_diag} "
                f"slither={self.dropped_slither} "
                f"hollow={self.dropped_hollow}")


def filter_paths(
    paths: list[VectorPath],
    mask_shape: tuple[int, int],
    design_width_mm: float,
    design_height_mm: float,
    min_area_mm2: float = 100.0,
    min_bbox_diag_mm: float = 8.0,
    max_aspect_ratio: float = 20.0,
    min_area_to_bbox: float = 0.01,
) -> tuple[list[VectorPath], FilterReport]:
    """Drop paths that look like text, slithers, or tiny noise.

    Parameters
    ----------
    paths : output of ``extract_vector_paths`` (mask-pixel coords)
    mask_shape : (H, W) used for mm conversion
    design_width_mm / design_height_mm : physical extents of the mask
    min_area_mm2 :
        Minimum path area in mm^2.  Default 100 = 10x10 mm.  Tool
        tracings are >= 10x10 mm by spec, so anything smaller is
        either text, a grid dash, or noise.
    min_bbox_diag_mm :
        Minimum bounding-box diagonal in mm.  Drops text-character-sized
        outlines that pass the area test by being squarish but tiny.
    max_aspect_ratio :
        Max of (long_dim / short_dim).  Paths shaped like 20:1 slivers
        (boundary-line text, residual grid dashes) are dropped.
    min_area_to_bbox :
        Minimum ratio of contour-area to bbox-area.  Extremely hollow
        outlines (often a few stray pixels forming a long loop) get
        dropped.

    Returns
    -------
    (filtered_paths, report).  Original ``paths`` list is not mutated.
    """
    mask_h, mask_w = mask_shape
    mm_per_px_x = design_width_mm / float(mask_w)
    mm_per_px_y = design_height_mm / float(mask_h)
    # Use the smaller scale so we don't accidentally pass a thin-tall path.
    mm_per_px = min(mm_per_px_x, mm_per_px_y)

    min_area_px = min_area_mm2 / (mm_per_px_x * mm_per_px_y)
    min_diag_px = min_bbox_diag_mm / mm_per_px

    kept: list[VectorPath] = []
    report = FilterReport()

    for p in paths:
        if p.area_px < min_area_px:
            report.dropped_small_area += 1
            continue

        _, _, bw, bh = p.bbox_xywh
        diag_px = float(np.hypot(bw, bh))
        if diag_px < min_diag_px:
            report.dropped_short_diag += 1
            continue

        short_dim = float(min(bw, bh))
        long_dim = float(max(bw, bh))
        aspect = long_dim / max(short_dim, 1.0)
        if aspect > max_aspect_ratio:
            report.dropped_slither += 1
            continue

        bbox_area = float(bw * bh)
        if bbox_area > 0 and (p.area_px / bbox_area) < min_area_to_bbox:
            report.dropped_hollow += 1
            continue

        kept.append(p)

    report.kept = len(kept)
    return kept, report


# ---------------------------------------------------------------------------
# Tuned convenience wrapper
# ---------------------------------------------------------------------------

def extract_vector_paths_tuned(
    trace_mask: np.ndarray,
    design_width_mm: float,
    design_height_mm: float,
    route: Route = "pen",
    filter_overrides: dict | None = None,
    **gap_overrides,
) -> tuple[list[VectorPath], FilterReport, dict]:
    """Run extract_vector_paths with route-specific defaults, then filter.

    Route picks both the extraction presets AND the filter presets.
    Pass ``filter_overrides`` to tweak filter thresholds for a single call;
    pass keyword args to override extraction params (``gap_close_px`` etc).

    Returns
    -------
    (filtered_paths, filter_report, params_used).  ``params_used`` is the
    final extraction-param dict so callers can report what ran.
    """
    preset = ROUTE_PRESETS.get(route, ROUTE_PRESETS["pen"]).copy()
    preset.update(gap_overrides)

    filt = ROUTE_FILTER_PRESETS.get(route, ROUTE_FILTER_PRESETS["pen"]).copy()
    if filter_overrides:
        filt.update(filter_overrides)

    paths = extract_vector_paths(trace_mask, **preset)
    H, W = trace_mask.shape[:2]
    filtered, report = filter_paths(
        paths, (H, W), design_width_mm, design_height_mm, **filt
    )
    return filtered, report, preset
