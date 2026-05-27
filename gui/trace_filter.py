"""Pre-vectorisation trace filter — drop tiny noise / text components.

After Phase 2 cleans the orange grid, the resulting binary trace
mask still contains every dark blob on the paper.  In real customer
photos this includes:

  * the actual tool outlines we want                          (BIG components)
  * orange grid remnants that survived Phase 2                (small/medium)
  * handwritten notes / printed text fragments                (small)
  * scan noise, dust, paper imperfections                     (very small)

The Phase 3 tool detector and per-tool vectoriser end up processing
all of these, which is slow and produces hundreds of junk vectors.
Because Phase 1 rectifies to a known mm/pixel scale, we can apply a
**physical-size threshold** in real-world millimetres — a single
robust filter that works across every template, camera, and
resolution.

Default: drop any connected component below 5 mm² (roughly a 2.5x2
mm character).  Real tool outlines are 100+ mm of perimeter at
~0.5 mm stroke, well over 50 mm² of pixels.  The default leaves a
huge margin around real geometry while wiping the obvious noise.

Returns:
    (filtered_mask, stats_dict)
    where stats_dict contains diagnostic counts useful for the
    run_report.json (components seen, kept, dropped, drop reasons).
"""
from __future__ import annotations

import cv2
import numpy as np


def filter_text_and_noise(
    trace_mask: np.ndarray,
    px_per_mm: float,
    *,
    min_component_area_mm2: float = 5.0,
    max_text_aspect_ratio: float = 8.0,
    max_text_bbox_short_mm: float = 3.0,
) -> tuple[np.ndarray, dict]:
    """Filter a binary trace mask to remove tiny noise / text components.

    Parameters
    ----------
    trace_mask:
        ``uint8`` binary mask (0 / 255).  Foreground is "ink".
    px_per_mm:
        Pixels per millimetre, derived from the rectified image and
        template design-area dimensions.
    min_component_area_mm2:
        Drop any connected component whose AREA is below this
        threshold.  Default 5 mm² (about a small printed character).
    max_text_aspect_ratio:
        Components with bounding-box aspect ratio above this are
        considered text-like.  Combined with the next threshold.
    max_text_bbox_short_mm:
        If a component's bbox SHORT-edge is below this AND the
        aspect ratio is above the text threshold, drop it.  Tools
        almost never produce such a thin elongated shape; text and
        scan streaks do.

    Returns
    -------
    (filtered_mask, stats)
        filtered_mask : same shape / dtype as input, with rejected
                        components set to 0.
        stats         : diagnostic dict with component counts and
                        rejection reasons.
    """
    if px_per_mm <= 0:
        # Can't apply mm-scale logic; pass through unchanged.
        return trace_mask, {
            "px_per_mm": 0.0,
            "skipped": True,
            "reason": "px_per_mm not available",
        }

    # Convert mm thresholds to pixel equivalents
    min_area_px = max(1, int(round(min_component_area_mm2 * (px_per_mm ** 2))))
    text_short_px = max(1, int(round(max_text_bbox_short_mm * px_per_mm)))

    binary = (trace_mask > 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    # Always-rejected label 0 = background; iterate 1..n_labels-1
    keep = np.zeros(n_labels, dtype=bool)
    n_total = n_labels - 1
    n_kept = 0
    n_dropped_size = 0
    n_dropped_textlike = 0
    smallest_kept_area = 0
    largest_dropped_area = 0

    for i in range(1, n_labels):
        x, y, w, h, area = stats[i]
        if area < min_area_px:
            n_dropped_size += 1
            largest_dropped_area = max(largest_dropped_area, int(area))
            continue
        long_side = max(w, h)
        short_side = max(1, min(w, h))
        aspect = long_side / short_side
        if (aspect > max_text_aspect_ratio
                and short_side < text_short_px):
            n_dropped_textlike += 1
            largest_dropped_area = max(largest_dropped_area, int(area))
            continue
        keep[i] = True
        n_kept += 1
        if smallest_kept_area == 0 or area < smallest_kept_area:
            smallest_kept_area = int(area)

    # LUT-based remap: keep[label] -> 255, else 0
    lut = np.where(keep, np.uint8(255), np.uint8(0))
    filtered = lut[labels]

    stats_out = {
        "px_per_mm": round(float(px_per_mm), 4),
        "min_component_area_mm2": float(min_component_area_mm2),
        "min_component_area_px": int(min_area_px),
        "components_total": int(n_total),
        "components_kept": int(n_kept),
        "components_dropped_size": int(n_dropped_size),
        "components_dropped_textlike": int(n_dropped_textlike),
        "smallest_kept_area_px": int(smallest_kept_area),
        "largest_dropped_area_px": int(largest_dropped_area),
        "ink_pixels_before": int(binary.sum()),
        "ink_pixels_after": int((filtered > 0).sum()),
    }
    return filtered, stats_out
