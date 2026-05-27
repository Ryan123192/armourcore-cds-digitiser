"""Simplified Phase 2 cleaner for scanned controlled-input sheets.

Why a new module?
=================
The production pipeline (v14 adaptive + grid_strip + line_strip +
dark_border + aggressive variants) was built to survive PHONE PHOTOS
with coloured lighting cast, dark camera vignettes, and lens
distortion.  None of that applies to a scanner.

For scan inputs we have:
* Pure-white paper background
* Known template colours (cyan border, peach/orange grid, red markers,
  optionally orange dots)
* Tool ink that is BLACK/DARK-GRAY and ACHROMATIC

So we can clean by colour alone in a single HSV pass:
  * cyan band -> erase to white
  * orange/peach band -> erase to white
  * red band -> erase to white
  * what remains: the achromatic tool ink + (sometimes) faint pencil

This module replaces ALL of v14_adaptive + grid_strip_all + line_strip
+ grid_strip_aggressive + strip_dark_border for the scan workflow.
No Hough, no CLAHE, no per-route logic.  ~0.3s instead of ~5s.

Two passes
==========
  ``clean_scan_pen`` - direct colour strip, leaves any dark pixel intact
  ``clean_scan_pencil`` - colour strip THEN Sauvola adaptive binarise
                          (the only enhancement needed for faint pencil)
"""
from __future__ import annotations

import cv2
import numpy as np

from armourcore_cds.phase2.pencil_enhance_v2 import sauvola_adaptive


def _template_colour_mask(image_bgr: np.ndarray,
                         strip_blue_notes: bool = False) -> np.ndarray:
    """Union mask of TEMPLATE-PRINTED colours only.

    Catches:
      * cyan border (~#00CAEC) - tight hue band, high sat required
      * peach grid (~#FFEBCC, low-sat warm)
      * orange grid (~#FFC466) / orange dots (~#FF9C00) - chromatic only
      * red markers (~#FF0033)

    Does NOT touch black/dark pen ink.  Saturation thresholds tuned so
    scanner anti-aliasing edges of black strokes (which carry slight
    hue) stay out of the mask.

    ``strip_blue_notes`` flag is kept for API back-compat but is
    OFF by default and the underlying mask call is gated separately
    in the CLI (pen_dense route only).
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # CYAN BORDER: very tight band + high saturation requirement so
    # scanner-edge-tinted black pen never lands in here.
    cyan = ((H >= 82) & (H <= 96) & (S >= 100) & (V >= 100))

    # WARM family (peach, orange, dots) - chromatic only, high value.
    warm = ((H >= 0) & (H <= 30) & (S >= 50) & (V >= 130))

    # RED markers - strong sat + high value
    red = (((H <= 10) | (H >= 170)) & (S >= 120) & (V >= 80))

    mask = cyan | warm | red

    if strip_blue_notes:
        mask = mask | _blue_notes_mask(hsv)

    return mask.astype(np.uint8) * 255


def _blue_notes_mask(hsv: np.ndarray) -> np.ndarray:
    """Detect blue-pen scribbles for removal.

    Band catches royal blue (~110), navy (~115), light blue (~100),
    purple-blue (~130).  Cyan border (~88) excluded by lower bound.

    ``S >= 60`` ensures the pixel is GENUINELY chromatic.  Black tool
    ink scanned with slight hue noise typically sits at S < 30, so
    this floor protects black ink from being misclassified as blue.
    """
    H, S, V = cv2.split(hsv)
    return ((H >= 99) & (H <= 145) & (S >= 60) & (V >= 40))


def strip_blue_notes(image_bgr: np.ndarray,
                    protect_dark_below: int = 20) -> np.ndarray:
    """Dedicated Phase-2 step: erase any blue-pen scribbles.

    Place AFTER the rectifier and BEFORE the rest of Phase 2 cleaning
    so subsequent grid/text steps don't have to deal with blue noise.

    ``protect_dark_below``: gray below this is too dark to be blue pen
    (probably very dark navy that's nearly black - or genuine black
    tool ink with a hue noise blip).  Keep it.  Default 20 = only
    truly-black pixels protected.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blue = _blue_notes_mask(hsv)
    kill = blue & (gray > protect_dark_below)
    out = image_bgr.copy()
    out[kill] = (255, 255, 255)
    return out


def clean_scan_pen(image_bgr: np.ndarray,
                  protect_dark_below: int = 140) -> np.ndarray:
    """Erase template colours, leaving pen ink intact.

    ``protect_dark_below``: gray below this is treated as TOOL INK
    regardless of any colour match.  Raised from 90 to 140 so scanner
    anti-aliasing edges of black pen strokes (gray 90-130, sometimes
    blue-tinted) stay protected and the strokes remain continuous.
    Pen ink under any scanner reads gray < ~80, well below this floor.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    template_mask = _template_colour_mask(image_bgr)
    protect = gray <= protect_dark_below
    kill = (template_mask > 0) & (~protect)

    out = image_bgr.copy()
    out[kill] = (255, 255, 255)
    return out


def clean_scan_pencil(image_bgr: np.ndarray,
                     sauvola_window: int = 25,
                     sauvola_k: float = 0.15,
                     post_dilate_px: int = 1,
                     contrast_boost: bool = False) -> np.ndarray:
    """Erase template colours then Sauvola-binarise to darken pencil.

    ``contrast_boost`` (default True): before Sauvola, apply a CLAHE
    luminance boost so faint pencil strokes become more separable
    from paper.  Helps V3-style scans where pencil came out very light.

    ``post_dilate_px`` (default 1): after Sauvola, dilate ink pixels by
    1px to bridge sub-stroke skip-marks so faint outlines become
    continuous closed loops for Phase 3.
    """
    # 1) erase template colours
    stripped = clean_scan_pen(image_bgr, protect_dark_below=110)

    # 2) contrast boost via CLAHE on the L channel - makes faint pencil
    # stand out more before binarisation
    if contrast_boost:
        lab = cv2.cvtColor(stripped, cv2.COLOR_BGR2LAB)
        L, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(12, 12))
        L = clahe.apply(L)
        stripped = cv2.cvtColor(cv2.merge([L, a, b]), cv2.COLOR_LAB2BGR)

    # 3) Sauvola adaptive binarise
    binary_bgr = sauvola_adaptive(stripped, window_size=sauvola_window,
                                 k=sauvola_k)

    # 4) Small dilation: bridge skip-marks between adjacent dark pixels
    if post_dilate_px > 0:
        gray = cv2.cvtColor(binary_bgr, cv2.COLOR_BGR2GRAY)
        ink = (gray < 170).astype(np.uint8) * 255
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (post_dilate_px * 2 + 1, post_dilate_px * 2 + 1))
        ink = cv2.dilate(ink, k)
        out = np.full(binary_bgr.shape, 255, dtype=np.uint8)
        out[ink > 0] = (40, 40, 40)
        return out
    return binary_bgr
