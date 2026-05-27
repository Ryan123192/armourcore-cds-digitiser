"""Phase 2 sibling: per-shape gap bridging.

Symptom this fixes
==================
img1 (BLUE_PEN_FLAT_01) has all 13 shapes visible in the v14-cleaned image
but the LEFT-COLUMN shapes (bottle, hexagon, fish, etc.) have wider stroke
breaks because the camera dark-border zone interacts badly with the
conservative cleaning.  Phase 3's per-shape rescue can't bridge gaps
> ~12mm, so those shapes are dropped.

Strategy
========
Find every dark-ink BLOB in the cleaned image.  For each blob whose bbox
looks like a tool tracing (area >= 100 mm^2, not the border), apply a
HEAVY morphological close inside its bbox+pad.  The close is GEOMETRICALLY
LIMITED to that shape's neighbourhood so:
  * Adjacent shapes don't merge (each gets its own close)
  * The dark image border doesn't get touched (small bbox, won't include border)
  * Neighbouring small ink fragments inside the bbox get pulled into the
    same blob (which IS what we want - we're rescuing one shape)

Returns a new cleaned BGR with the bridged ink stamped back as dark pixels.
"""
from __future__ import annotations

import cv2
import numpy as np

from armourcore_cds.phase1.marker_rectify_fast_v4 import (
    PAPER_W_MM, PAPER_H_MM,
)


def bridge_shape_gaps(
    cleaned_bgr: np.ndarray,
    dark_threshold: int = 170,
    min_blob_mm2: float = 80.0,
    close_kernel_px: int = 51,        # bumped: 31 -> 51 for wider gaps
    discovery_close_px: int = 41,     # bumped: 21 -> 41 so fragments group
    bbox_pad_px: int = 16,
    max_image_area_frac: float = 0.25,
) -> np.ndarray:
    """For each promising ink blob, locally close gaps inside its bbox.

    Returns a new BGR image where the closed regions have been re-stamped
    as dark ink (gray=40).  Original cleaned input is not mutated.
    """
    H, W = cleaned_bgr.shape[:2]
    px_per_mm_x = W / PAPER_W_MM
    px_per_mm_y = H / PAPER_H_MM
    px_per_mm2 = px_per_mm_x * px_per_mm_y
    min_blob_px = int(min_blob_mm2 * px_per_mm2)

    gray = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
    ink = (gray < dark_threshold).astype(np.uint8) * 255

    # Pre-pass: bigger close at discovery so fragmented shape outlines
    # (e.g. img1 left-column with wide breaks) merge into ONE blob.
    discovery = cv2.morphologyEx(
        ink, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (discovery_close_px, discovery_close_px)),
    )

    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(
        discovery, connectivity=8)

    out = cleaned_bgr.copy()
    image_area = H * W
    for cid in range(1, n_lbl):
        x = stats[cid, cv2.CC_STAT_LEFT]
        y = stats[cid, cv2.CC_STAT_TOP]
        bw = stats[cid, cv2.CC_STAT_WIDTH]
        bh = stats[cid, cv2.CC_STAT_HEIGHT]
        area = stats[cid, cv2.CC_STAT_AREA]
        # Too small to be a tool tracing
        if bw * bh < min_blob_px:
            continue
        # Too big - probably image border or merged frame
        if bw * bh > image_area * max_image_area_frac:
            continue

        x1 = max(0, x - bbox_pad_px)
        y1 = max(0, y - bbox_pad_px)
        x2 = min(W, x + bw + bbox_pad_px)
        y2 = min(H, y + bh + bbox_pad_px)

        # Aggressive close within this bbox only
        sub_ink = ink[y1:y2, x1:x2].copy()
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_kernel_px, close_kernel_px))
        sub_closed = cv2.morphologyEx(sub_ink, cv2.MORPH_CLOSE, k)

        # Stamp the bridged ink back into the output as dark pixels.
        # Only stamp where we added NEW ink (avoid clobbering paper).
        new_ink = (sub_closed > 0) & (sub_ink == 0)
        if new_ink.any():
            region = out[y1:y2, x1:x2]
            region[new_ink] = (40, 40, 40)
            out[y1:y2, x1:x2] = region

    return out
