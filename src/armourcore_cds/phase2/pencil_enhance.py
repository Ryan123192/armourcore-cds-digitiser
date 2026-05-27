"""Phase 2 sibling: enhance faint pencil traces before binarisation.

Why
===
Pencil traces have low contrast against paper (gray 160-200 ink vs
gray 230-245 paper).  A fixed threshold (gray<170) misses much of the
stroke, leaving sparse dots that gap-fill can't connect.

Strategy
========
Three different boost methods, callers can mix-and-match or pick one:

1. ``clahe_boost``    -- local contrast enhancement on L channel.  Pulls
                          faint pencil strokes down toward gray ~100.
2. ``paper_divide``   -- divide grayscale by smoothed local background
                          and rescale.  Paper -> uniform near-white;
                          pencil -> low-gray proportional to original
                          darkness.  Like Photoshop "subtract background".
3. ``sauvola_binarise`` -- Sauvola adaptive threshold.  Document-binarisation
                          standard; threshold varies per pixel based on
                          local mean+std.  Handles uneven lighting that
                          breaks fixed thresholds.

All three return a BGR image (so they slot into the existing pipeline)
with pencil traces darkened.
"""
from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# CLAHE boost
# ---------------------------------------------------------------------------

def clahe_boost(image_bgr: np.ndarray,
               clip_limit: float = 3.5,
               tile_grid: int = 16) -> np.ndarray:
    """Local-adaptive contrast boost on the L channel."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                            tileGridSize=(tile_grid, tile_grid))
    L_b = clahe.apply(L)
    return cv2.cvtColor(cv2.merge([L_b, a, b]), cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Paper-divide normalisation (most effective for pencil)
# ---------------------------------------------------------------------------

def paper_divide(image_bgr: np.ndarray,
                bg_sigma: float = 35.0,
                target_paper: float = 240.0,
                ink_protect_gray: int = 70) -> np.ndarray:
    """Divide by smoothed local background then rescale.

    Pixels darker than ``ink_protect_gray`` are excluded from the bg
    estimate so they don't darken their own surroundings.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    weight = np.ones_like(gray, dtype=np.float32)

    dark = gray < ink_protect_gray
    masked = gray.copy()
    masked[dark] = 0
    weight[dark] = 0

    bg_num = cv2.GaussianBlur(masked, (0, 0), bg_sigma)
    bg_den = cv2.GaussianBlur(weight, (0, 0), bg_sigma)
    bg = bg_num / np.maximum(bg_den, 1e-3)
    bg = np.maximum(bg, 30.0)

    # ratio: each pixel as a fraction of its local background paper
    ratio = gray / bg
    # rescale so 1.0 -> target_paper, ink ratios produce darker values
    out_gray = np.clip(ratio * target_paper, 0, 255).astype(np.uint8)

    # Return as BGR
    return cv2.cvtColor(out_gray, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# Sauvola binarisation (returns BGR with binary mask stamped dark)
# ---------------------------------------------------------------------------

def sauvola_binarise(image_bgr: np.ndarray,
                    window_size: int = 25,
                    k: float = 0.2,
                    R: float = 128.0) -> np.ndarray:
    """Sauvola adaptive threshold.  Returns binary BGR (dark ink / white).

    Threshold: T(x,y) = mean(x,y) * (1 + k*(std(x,y)/R - 1))

    Lower k -> more aggressive (catches fainter ink).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Local mean + std via box filter
    ksize = (window_size, window_size)
    mean = cv2.boxFilter(gray, ddepth=cv2.CV_32F, ksize=ksize)
    sq_mean = cv2.boxFilter(gray * gray, ddepth=cv2.CV_32F, ksize=ksize)
    var = np.maximum(sq_mean - mean * mean, 0.0)
    std = np.sqrt(var)
    T = mean * (1 + k * (std / R - 1))
    binary = (gray < T).astype(np.uint8) * 255   # ink = 255
    # Output BGR: ink stays dark (40,40,40), paper stays white
    out = np.full(image_bgr.shape, 255, dtype=np.uint8)
    out[binary > 0] = (40, 40, 40)
    return out


# ---------------------------------------------------------------------------
# Composite: best-for-pencil enhance pipeline
# ---------------------------------------------------------------------------

def enhance_pencil(cleaned_bgr: np.ndarray) -> np.ndarray:
    """Apply CLAHE + paper_divide.  Returns BGR image that has pencil
    traces noticeably darker, ready for fixed-threshold binarisation
    (gray < 170) in downstream code.
    """
    boosted = clahe_boost(cleaned_bgr, clip_limit=3.0, tile_grid=16)
    normalised = paper_divide(boosted, bg_sigma=40.0,
                              target_paper=240.0, ink_protect_gray=70)
    return normalised


def enhance_pencil_strong(cleaned_bgr: np.ndarray) -> np.ndarray:
    """More aggressive: CLAHE + paper_divide + final Sauvola pass.
    Use when standard enhance still misses too much pencil."""
    boosted = clahe_boost(cleaned_bgr, clip_limit=4.0, tile_grid=12)
    normalised = paper_divide(boosted, bg_sigma=35.0,
                              target_paper=235.0, ink_protect_gray=60)
    sauv = sauvola_binarise(normalised, window_size=31, k=0.15)
    return sauv
