"""Phase 1 sibling for SCANNED INPUTS.

Why a separate module?
======================
The production ``marker_rectify_fast_v4`` was designed for phone
photographs where the sheet sits on a desk with non-paper background
around it.  It does a lot of work to find the paper edge, handle
perspective distortion, deal with lighting, etc.

A scan is fundamentally different:
* Background is paper-white (the platen lid)
* No perspective distortion - it's a flatbed
* Red corner markers (#FF0033) are reliably ~near the 4 corners
* Small skew at most (paper sits a few degrees off-square on platen)

So the rectifier just needs to:
1. Find red pixels via colour threshold
2. Cluster them into 4 components - one near each image corner
3. Compute centroid of each marker cluster
4. Perspective-warp those 4 points to (PAPER_W_MM, PAPER_H_MM)

That's it.  No edge detection, no Hough, no Lab dance.

API mirrors v4 so the rest of the pipeline can swap in unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from armourcore_cds.phase1.marker_rectify_fast_v4 import (
    PAPER_W_MM, PAPER_H_MM, DEFAULT_PX_PER_MM,
)


@dataclass
class ScanRectifyResult:
    """Mirrors RectifyResult shape - has .warped so downstream code works."""
    warped: np.ndarray
    marker_pixels_bgr: tuple   # the 4 (x, y) centroids - useful for debug
    debug_overlay: np.ndarray  # original with markers + bounding box drawn


def _red_marker_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Detect red marker pixels.

    Loose thresholds because PDF-rasterised / scanned red can drift from
    pure #FF0033 to something more like (B=127, G=92, R=204) due to
    anti-aliasing and scanner colour profile.

    The marker design is a SQUARE WITH AN X INSIDE - so we also close
    holes within ~15px so each marker becomes one solid blob.
    """
    B, G, R = cv2.split(image_bgr)
    red_bgr = (R.astype(np.int16) - np.maximum(G, B).astype(np.int16) > 30) & (R > 100)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    red_hsv = ((H < 15) | (H > 165)) & (S > 60) & (V > 60)

    mask = (red_bgr & red_hsv).astype(np.uint8) * 255
    # Close holes in the X-inside-square marker design so each marker is
    # one connected blob.
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    # Speckle clean
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return mask


def _find_4_corner_markers(red_mask: np.ndarray, image_shape) -> list[tuple[float, float]]:
    """Return 4 marker INNER-CORNER points ordered TL, TR, BR, BL.

    For each marker (a square in a page corner), the INNER CORNER is the
    bbox corner facing the page centre.  So the design area sits exactly
    INSIDE the four markers, not overlapping them.

      TL marker -> its bbox BR corner (page-centre-facing)
      TR marker -> its bbox BL corner
      BR marker -> its bbox TL corner
      BL marker -> its bbox TR corner
    """
    H, W = image_shape[:2]
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(
        red_mask, connectivity=8)
    # Build candidate list with bbox info; bbox-centre is used only for
    # the page-corner assignment step, not for the final warp source.
    cands_all = []
    for cid in range(1, n):
        area = stats[cid, cv2.CC_STAT_AREA]
        if area < 1500:
            continue
        x = stats[cid, cv2.CC_STAT_LEFT]
        y = stats[cid, cv2.CC_STAT_TOP]
        bw = stats[cid, cv2.CC_STAT_WIDTH]
        bh = stats[cid, cv2.CC_STAT_HEIGHT]
        cx_bbox = x + bw / 2.0
        cy_bbox = y + bh / 2.0
        cands_all.append({
            "cx": float(cx_bbox), "cy": float(cy_bbox),
            "x": int(x), "y": int(y),
            "w": int(bw), "h": int(bh),
            "area": int(area),
        })
    cands_all.sort(key=lambda c: -c["area"])
    cands = cands_all[:8]
    if len(cands) < 4:
        raise RuntimeError(
            f"Scan rectify: found only {len(cands)} red marker candidates "
            f"(need 4).  Is the sheet scanned in colour mode?")

    # Assign each candidate to a page corner (TL, TR, BR, BL) by which
    # page corner its bbox-centre is closest to.
    corners_ref = [
        ("TL", 0, 0),
        ("TR", W - 1, 0),
        ("BR", W - 1, H - 1),
        ("BL", 0, H - 1),
    ]
    picked = []
    used = set()
    for tag, ref_x, ref_y in corners_ref:
        best_idx = None
        best_d = float("inf")
        for i, c in enumerate(cands):
            if i in used:
                continue
            d = (c["cx"] - ref_x) ** 2 + (c["cy"] - ref_y) ** 2
            if d < best_d:
                best_d = d
                best_idx = i
        if best_idx is None:
            raise RuntimeError("Scan rectify: marker assignment failed")
        c = cands[best_idx]
        # Pick the marker's INNER BBOX CORNER (toward page centre).
        if tag == "TL":   # use BR of bbox
            px, py = c["x"] + c["w"], c["y"] + c["h"]
        elif tag == "TR": # use BL of bbox
            px, py = c["x"],          c["y"] + c["h"]
        elif tag == "BR": # use TL of bbox
            px, py = c["x"],          c["y"]
        else:             # BL -> use TR of bbox
            px, py = c["x"] + c["w"], c["y"]
        picked.append((float(px), float(py)))
        used.add(best_idx)
    return picked


def rectify_scan(
    image_bgr: np.ndarray,
    paper_w_mm: float = PAPER_W_MM,
    paper_h_mm: float = PAPER_H_MM,
    px_per_mm: float = DEFAULT_PX_PER_MM,
) -> ScanRectifyResult:
    """Simple scan-based rectifier.

    Caller is responsible for any pre-rotation (use GUI controls or
    rotate before calling).  This function does NOT auto-rotate -
    auto-detection by image dimensions guessed the wrong direction
    in some cases.
    """
    H, W = image_bgr.shape[:2]
    red = _red_marker_mask(image_bgr)
    pts = _find_4_corner_markers(red, image_bgr.shape)

    # Build debug overlay: draw markers + connecting box
    debug = image_bgr.copy()
    for i, (x, y) in enumerate(pts):
        cv2.circle(debug, (int(x), int(y)), 30, (0, 255, 0), 5)
        cv2.putText(debug, ["TL", "TR", "BR", "BL"][i],
                   (int(x) + 35, int(y) + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3,
                   cv2.LINE_AA)
    quad = np.array([(int(x), int(y)) for (x, y) in pts], dtype=np.int32)
    cv2.polylines(debug, [quad.reshape(-1, 1, 2)], True, (0, 255, 0), 4)

    # Compute homography to a clean (paper_w_mm * px) x (paper_h_mm * px) canvas
    out_w = int(round(paper_w_mm * px_per_mm))
    out_h = int(round(paper_h_mm * px_per_mm))
    dst = np.array([
        (0,         0),
        (out_w - 1, 0),
        (out_w - 1, out_h - 1),
        (0,         out_h - 1),
    ], dtype=np.float32)
    src = np.array(pts, dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image_bgr, M, (out_w, out_h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255))

    return ScanRectifyResult(
        warped=warped,
        marker_pixels_bgr=tuple(pts),
        debug_overlay=debug,
    )
