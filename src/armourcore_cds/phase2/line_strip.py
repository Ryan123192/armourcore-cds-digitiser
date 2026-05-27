"""Phase 2 sibling: remove dashed boundary lines via Hough line detection.

Previous version used morphological line-opening which mistakenly
caught tool outline edges (any bottle/rectangle has a ~100px vertical
left edge that survived the opening).

This version uses cv2.HoughLinesP to detect only PERFECTLY STRAIGHT
long axis-aligned lines.  Tool outlines are CURVED so Hough only
catches the very straight short segments, which we reject by minimum
length.  Boundary lines are TEMPLATE-PRINTED PERFECTLY STRAIGHT (even
when dashed) so Hough catches them after a small close to bridge dash
gaps.

Algorithm
=========
1. From cleaned BGR, build ink mask.
2. Close with thin vertical kernel (1 x 21) to bridge dash gaps WITHIN
   a single dashed line.  This doesn't merge tool outline curves
   because they're not vertical-aligned across the gap.
3. Run HoughLinesP looking for very-vertical lines:
       angle within ``angle_tolerance_deg`` of 90
       length >= ``min_line_length_px``
4. Same for horizontal lines.
5. Each detected line: draw thin band on a mask, then erase those
   pixels from the cleaned image.

Conservativeness: ``min_line_length_px`` defaults to 500px (~50mm at
9.86 px/mm) so only LONG straight lines get caught.  Tool outlines
with straight sections of even 30-40mm don't qualify because their
sections are CURVED enough to break the Hough detection threshold.
"""
from __future__ import annotations

import cv2
import numpy as np


def detect_long_straight_lines(
    ink: np.ndarray,
    angle_tolerance_deg: float = 1.5,
    min_line_length_px: int = 500,
    max_line_gap_px: int = 25,
    hough_vote_factor: float = 0.6,
    pre_close_dash_gap_px: int = 21,
) -> np.ndarray:
    """Return a band mask covering long straight lines in ink.

    Only lines within angle_tolerance of vertical or horizontal are
    accepted, so curved tool outlines are NEVER caught.
    """
    H, W = ink.shape

    # Bridge dash gaps - vertical and horizontal closes separately.
    vk = cv2.getStructuringElement(cv2.MORPH_RECT,
                                   (1, pre_close_dash_gap_px))
    hk = cv2.getStructuringElement(cv2.MORPH_RECT,
                                   (pre_close_dash_gap_px, 1))
    closed = cv2.morphologyEx(
        cv2.morphologyEx(ink, cv2.MORPH_CLOSE, vk),
        cv2.MORPH_CLOSE, hk,
    )

    lines = cv2.HoughLinesP(
        closed, rho=1, theta=np.pi / 720.0,
        threshold=max(20, int(min_line_length_px * hough_vote_factor)),
        minLineLength=min_line_length_px,
        maxLineGap=max_line_gap_px,
    )
    band = np.zeros_like(ink)
    if lines is None:
        return band

    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        if dx == 0 and dy == 0:
            continue
        ang = np.degrees(np.arctan2(abs(dy), abs(dx)))
        is_vert = ang >= (90.0 - angle_tolerance_deg)
        is_horiz = ang <= angle_tolerance_deg
        if not (is_vert or is_horiz):
            continue
        if is_vert:
            # Extend full image height at x = median x of segment
            xc = int(round((x1 + x2) / 2))
            cv2.line(band, (xc, 0), (xc, H - 1), 255, 7)
        else:
            yc = int(round((y1 + y2) / 2))
            cv2.line(band, (0, yc), (W - 1, yc), 255, 7)
    return band


def strip_dashed_boundary_lines(
    cleaned_bgr: np.ndarray,
    dark_threshold: int = 170,
    angle_tolerance_deg: float = 1.5,
    min_line_length_px: int = 500,
    max_line_gap_px: int = 25,
    band_half_px: int = 6,
    ink_protect_max_gray: int = 130,
    text_kill_max_area_px: int = 400,
    text_kill_proximity_px: int = 60,
) -> np.ndarray:
    """Erase dashed template-printed boundary lines + adjacent small text.

    Only catches lines that are within ``angle_tolerance_deg`` of
    axis-aligned AND at least ``min_line_length_px`` long.  Curved tool
    outlines are NEVER caught regardless of their height.

    Tool ink darker than ``ink_protect_max_gray`` along the band is
    preserved so any tool stroke that genuinely runs along a boundary
    line position survives.
    """
    H, W = cleaned_bgr.shape[:2]
    gray = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
    ink = (gray < dark_threshold).astype(np.uint8) * 255

    band = detect_long_straight_lines(
        ink, angle_tolerance_deg=angle_tolerance_deg,
        min_line_length_px=min_line_length_px,
        max_line_gap_px=max_line_gap_px,
    )
    if band_half_px != 3:
        # Resize band thickness to match band_half_px (rasterise at 7px
        # was the default).  Erode/dilate to adjust.
        target_thickness = 2 * band_half_px + 1
        actual_thickness = 7
        if target_thickness > actual_thickness:
            kdil = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (1, target_thickness - actual_thickness + 1) if False
                else (3, 3))
            band = cv2.dilate(band, kdil,
                              iterations=max(0, band_half_px - 3))
        elif target_thickness < actual_thickness:
            kero = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            band = cv2.erode(band, kero,
                             iterations=max(0, 3 - band_half_px))

    # Protect tool ink darker than threshold
    ink_protect = gray <= ink_protect_max_gray
    kill = (band > 0) & ~ink_protect

    out = cleaned_bgr.copy()
    out[kill] = (255, 255, 255)

    # Adjacent small text suppression: if there are tight components
    # within proximity of a detected line, drop them.
    if text_kill_max_area_px > 0 and band.any():
        prox = cv2.dilate(
            band,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (text_kill_proximity_px * 2 + 1,
                 text_kill_proximity_px * 2 + 1)),
        )
        ink_after = (cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
                     < dark_threshold).astype(np.uint8) * 255
        n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(
            ink_after, connectivity=8)
        text_kill = np.zeros_like(ink_after)
        for cid in range(1, n_lbl):
            area = stats[cid, cv2.CC_STAT_AREA]
            if area > text_kill_max_area_px:
                continue
            comp = (lbl == cid)
            if (comp & (prox > 0)).any():
                text_kill[comp] = 255
        out[text_kill > 0] = (255, 255, 255)

    return out
