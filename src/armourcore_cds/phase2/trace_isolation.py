"""Phase 2 — colour-based grid isolation for orange-grid CDS sheets.

Design philosophy
-----------------
The CDS grid is orange.  Customer marks are graphite, pen, or other colours
that are NOT orange.  So the approach is simple:

  1. Detect all orange-ish pixels using HSV hue range.
     HSV hue is brightness-independent: dark orange, light orange, and every
     shade in between share the same hue angle.  This is why the LAB-distance
     approach missed major/darker grid lines — a hue range catches them all.

  2. Apply a morphological CLOSE to bridge small gaps where dark customer
     marks cross grid lines (those crossing pixels become dark, not orange).

  3. Replace orange pixels with white (paper background).

  4. Detect remaining dark marks → trace candidates for downstream use.

No projected grid geometry is needed.  The orange colour IS the discriminator.
This also means the template grid spacing is irrelevant at this stage.

Black-grid support is preserved for backward compat / experimentation, but
the orange path is the primary focus going forward.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from armourcore_cds.templates.models import TemplateModel


GridColour = Literal["orange", "black", "auto"]

# ---------------------------------------------------------------------------
# Orange colour constants (HSV, OpenCV scale: H ∈ [0, 180])
#
# #D35400  deep orange    → H ≈ 12  (24° standard)
# #F39C12  golden orange  → H ≈ 18  (37° standard)
# #FAD7A0  pale peach     → H ≈ 18  (37° standard)
#
# Range tuned for BOTH digital-PDF sources (clean strong orange,
# saturation 40-200) AND photo-scan sources (lighting / JPEG / lens
# wash-out reduces saturation to 11-15 — only ~5 units above the
# off-white paper background's saturation of 6-9).
#
# Hue widened slightly to cover white-balance drift in cameras.
# Saturation min dropped to 10 (catches washed-out orange in photos);
# off-white paper background tops out around 9 so the discriminator
# still works.  Tool outlines (graphite / pen) have grey hue so are
# not caught regardless of saturation.  Any dark customer mark
# accidentally caught is restored by the DARK_MARK_THRESHOLD repair
# pass downstream.
# ---------------------------------------------------------------------------
ORANGE_HUE_LOW: int = 0    # ~0°  — into red
ORANGE_HUE_HIGH: int = 40  # ~80° — wider yellow-orange envelope (was 32)
ORANGE_SAT_MIN: int = 25   # min saturation on the CHROMA-BOOSTED image
ORANGE_VAL_MIN: int = 10   # very low — major grid lines in shadow can be dark

# Chroma-boost gain applied before HSV thresholding.  Multiplies the
# saturation channel so washed-out orange (photo-scan, shadowed major grid
# lines, orange text) becomes detectably chromatic.  Achromatic pencil /
# pen / paper stay near zero saturation no matter what gain is applied,
# so the boost cleanly separates "coloured ink" from "achromatic ink".
ORANGE_CHROMA_BOOST: float = 3.0

# Dark-mark restoration: pixels darker than this threshold in the original
# image AND with saturation below ACHROMATIC_SAT_MAX are protected from
# orange removal — that's the chromaticity-aware test for "this dark
# pixel is customer pencil / pen ink, not dark orange grid / text".
# A pixel that is both dark AND chromatic is dark *orange* and should be
# removed; achromatic-dark is customer ink and must survive.
DARK_MARK_THRESHOLD: int = 90
ACHROMATIC_SAT_MAX: int = 35  # original-image saturation below this = achromatic


# ---------------------------------------------------------------------------
# Lighting normalisation
# ---------------------------------------------------------------------------

def normalise_lighting(
    image_bgr: np.ndarray,
    clip_limit: float = 2.5,
    tile_grid: int = 32,
) -> np.ndarray:
    """Flatten uneven lighting / shadow gradients across the image.

    Photo scans often have one half darker than the other (overhead-light
    fall-off, hand shadow, page curve).  In dark regions every pixel is
    below the "dark mark" threshold which makes the orange grid
    indistinguishable from customer ink.  CLAHE on the LAB-L channel
    brightens shadows and darkens highlights LOCALLY so the whole image
    has comparable luminance.

    Parameters
    ----------
    clip_limit :
        CLAHE contrast clip limit.  ~2.5 = noticeable but not aggressive
        normalisation.  Larger = more aggressive.
    tile_grid :
        Number of CLAHE tiles per side.  32 splits the image into 32x32
        regions, each separately normalised.

    Returns
    -------
    Lighting-flattened BGR image at the same shape / dtype.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    L_norm = clahe.apply(L)
    lab_norm = cv2.merge([L_norm, a, b])
    return cv2.cvtColor(lab_norm, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Paper-background-aware fill (replaces grid pixels with local paper colour
# instead of pure white — prevents creating a "white grid on off-white paper"
# artefact that downstream stages mis-interpret as another grid)
# ---------------------------------------------------------------------------

def normalise_paper_to_white(
    image_bgr: np.ndarray,
    target_paper_value: int = 245,
    bg_sigma: float = 80.0,
    dark_mark_threshold: int = 90,
) -> np.ndarray:
    """Lift the local paper background to a consistent near-white value.

    Photographed paper sits around grey value 170-200; digital PDFs sit
    at 255.  Phase 3's trace detector uses a fixed grey threshold (170)
    tuned for digital paper, so photographed paper triggers it as if it
    were ink.  This function divides each pixel by an estimate of the
    local paper background and multiplies by *target_paper_value*,
    bringing paper to a consistent ~245 regardless of input source.

    Tool outlines (which are darker than paper) stay proportionally
    darker after scaling, so the trace threshold still separates them.

    Dark customer marks (gray < dark_mark_threshold) are excluded from
    the background estimate — otherwise ink-heavy regions would pull
    the bg estimate down and cause the surrounding paper to be
    over-brightened.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    paper_only = gray.astype(np.float32)
    weight = np.ones_like(paper_only, dtype=np.float32)

    dark = gray < dark_mark_threshold
    paper_only[dark] = 0
    weight[dark] = 0

    bg_num = cv2.GaussianBlur(paper_only, (0, 0), bg_sigma)
    bg_den = cv2.GaussianBlur(weight,     (0, 0), bg_sigma)
    bg = bg_num / np.maximum(bg_den, 1e-3)
    bg = np.maximum(bg, 1.0)

    scale = float(target_paper_value) / bg
    scale = np.clip(scale, 0.5, 3.0)  # cap extremes

    out = image_bgr.astype(np.float32)
    out *= scale[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def boost_chroma(image_bgr: np.ndarray, gain: float = ORANGE_CHROMA_BOOST) -> np.ndarray:
    """Multiply HSV saturation by *gain* to amplify chromatic separation.

    Real-world photographed orange grid lines can have a saturation as low as
    6-15 — barely above the achromatic noise floor of the paper background.
    Multiplying S by 3x lifts those grid pixels to 18-45, well above the
    detection threshold, while truly achromatic ink (pencil / pen / paper)
    stays near zero saturation.

    Hue and Value are left untouched so the colour identity of each pixel is
    preserved — only the chromaticity magnitude is exaggerated.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * float(gain), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def paper_blended_fill(
    image_bgr: np.ndarray,
    removal_mask: np.ndarray,
    bg_sigma: float = 40.0,
) -> np.ndarray:
    """Replace masked pixels with a smoothed estimate of the local paper.

    Algorithm:
      1. Zero out non-paper pixels (orange + customer ink) and their
         weights.
      2. Heavily Gaussian-blur the paper-only image AND the weight mask
         separately, then normalise — gives a per-pixel estimate of paper
         colour even in regions where paper was masked out.
      3. Copy that estimate into the masked positions of the original
         image.

    Result: removed grid pixels take on the actual paper background
    colour (slightly off-white, with any lighting variation preserved
    smoothly), so the cleaned output looks like the paper alone — no
    pure-white "ghost grid" that downstream stages would mis-interpret.
    """
    keep_mask = (removal_mask == 0).astype(np.float32)  # 1 where paper-only

    masked_img = image_bgr.astype(np.float32)
    masked_img *= keep_mask[..., None]   # zero out removed pixels

    img_blur    = cv2.GaussianBlur(masked_img,           (0, 0), bg_sigma)
    weight_blur = cv2.GaussianBlur(keep_mask,            (0, 0), bg_sigma)

    paper_bg = img_blur / np.maximum(weight_blur[..., None], 1e-3)
    paper_bg = np.clip(paper_bg, 0, 255).astype(np.uint8)

    out = image_bgr.copy()
    out[removal_mask > 0] = paper_bg[removal_mask > 0]
    return out


@dataclass
class TraceIsolationResult:
    """Results of the Phase 2 grid-isolation step."""
    cleaned_bgr: np.ndarray            # image with orange grid erased (white background)
    orange_mask: np.ndarray            # binary: pixels detected as orange grid
    trace_candidate_mask: np.ndarray   # binary: remaining dark customer marks
    note_candidate_seed_mask: np.ndarray  # same as trace_candidate_mask (downstream compat)
    removal_mask: np.ndarray           # alias of orange_mask (what was removed)
    grid_colour: GridColour            # resolved colour mode used

    # Legacy / compat fields populated only for specific modes
    orange_likelihood: np.ndarray = field(
        default_factory=lambda: np.zeros((1, 1), dtype=np.float32)
    )
    orange_grid_mask: np.ndarray = field(
        default_factory=lambda: np.zeros((1, 1), dtype=np.uint8)
    )
    protect_mask: np.ndarray = field(
        default_factory=lambda: np.zeros((1, 1), dtype=np.uint8)
    )
    relative_darkness: np.ndarray = field(
        default_factory=lambda: np.zeros((1, 1), dtype=np.uint8)
    )
    projected_grid_mask: np.ndarray = field(
        default_factory=lambda: np.zeros((1, 1), dtype=np.uint8)
    )
    black_grid_mask: np.ndarray = field(
        default_factory=lambda: np.zeros((1, 1), dtype=np.uint8)
    )
    black_removal_mask: np.ndarray = field(
        default_factory=lambda: np.zeros((1, 1), dtype=np.uint8)
    )


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def _remove_small_components(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    """Remove connected components smaller than min_area_px.

    Uses a lookup-table approach: O(n_labels) build + one vectorised O(HW)
    index.  Avoids the O(n_labels × HW) Python loop that is catastrophically
    slow when tens-of-thousands of components exist on a multi-megapixel image.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats[:, cv2.CC_STAT_AREA]
    lut = np.where(areas >= min_area_px, np.uint8(255), np.uint8(0))
    lut[0] = 0   # background always removed
    return lut[labels].astype(np.uint8)


# ---------------------------------------------------------------------------
# Orange path
# ---------------------------------------------------------------------------

def build_orange_mask(
    image_bgr: np.ndarray,
    hue_low: int = ORANGE_HUE_LOW,
    hue_high: int = ORANGE_HUE_HIGH,
    sat_min: int = ORANGE_SAT_MIN,
    val_min: int = ORANGE_VAL_MIN,
    close_px: int = 2,
    min_component_px: int = 15,
    chroma_boost: float = ORANGE_CHROMA_BOOST,
) -> np.ndarray:
    """Detect orange grid pixels via HSV hue range on a chroma-boosted image.

    **Chroma boost (NEW)** — saturation is multiplied by ``chroma_boost``
    before HSV thresholding.  This separates faintly-coloured grid lines and
    orange text from achromatic customer ink: the grid jumps from S~10 up to
    S~30, while pencil/pen ink (already near zero saturation) stays in the
    noise floor.  Without the boost, the saturation threshold had to be
    suicidally low (10) to catch shadowed major grid lines, which risked
    classifying coloured pencil as grid.

    **No large morphological close** — a large close bridges over black customer
    marks at grid-line crossings, incorrectly removing them.  Instead a tiny
    ``close_px=2`` handles only sub-pixel anti-aliasing gaps at line edges.
    Dark customer marks that happen to sit on orange grid positions are restored
    separately in the repair pass (see ``isolate_trace_candidates``).

    Parameters
    ----------
    chroma_boost:
        Multiplier applied to the HSV saturation channel before thresholding.
        Set 1.0 to disable (matches original behaviour).
    close_px:
        Radius (px) for morphological CLOSE.  Keep ≤ 3 to avoid bridging
        across black customer marks.  Handles anti-aliasing fringe only.
    min_component_px:
        Remove isolated noise specks smaller than this many pixels.
    """
    if chroma_boost and chroma_boost > 1.0:
        detection_bgr = boost_chroma(image_bgr, gain=chroma_boost)
    else:
        detection_bgr = image_bgr
    hsv = cv2.cvtColor(detection_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([hue_low, sat_min, val_min], dtype=np.uint8)
    upper = np.array([hue_high, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # Tiny close: anti-aliasing fringe only (NOT bridging across marks)
    if close_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_px * 2 + 1, close_px * 2 + 1)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    # Remove noise specks
    if min_component_px > 0:
        mask = _remove_small_components(mask, min_area_px=min_component_px)

    return mask


def build_lab_orange_mask(
    image_bgr: np.ndarray,
    a_threshold: int = 2,
    min_component_px: int = 12,
    close_px: int = 2,
) -> np.ndarray:
    """Brightness-INDEPENDENT orange detection via LAB a* channel.

    HSV saturation collapses to zero when a coloured pixel is very dark.
    LAB's a* axis (red-green chromaticity) is *decoupled* from luminance,
    so dark orange in shadow still has a* well above 128 (the neutral
    point), while pencil / pen / paper all sit very close to 128.

    Typical values:
      • bright orange grid: a* ≈ 150–180
      • dark orange in shadow: a* ≈ 132–145
      • graphite / pen / paper: a* ≈ 126–130

    A simple ``a* - 128 > a_threshold`` catches all orange regardless of
    how shadowed it is, with virtually no risk of false-positive on
    achromatic ink.  This is precisely the corner-of-image failure mode
    the HSV detector exhibits.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    a = lab[..., 1].astype(np.int16)
    mask = ((a - 128) > a_threshold).astype(np.uint8) * 255
    if close_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_px * 2 + 1, close_px * 2 + 1)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    if min_component_px > 0:
        mask = _remove_small_components(mask, min_area_px=min_component_px)
    return mask


def build_combined_orange_mask(
    image_bgr: np.ndarray,
) -> np.ndarray:
    """Union of HSV chroma-boost AND LAB a* channel detection.

    HSV catches typical / well-lit orange.  LAB a* catches shadowed orange
    the HSV path misses (brightness-independent).  Union of the two gives
    comprehensive coverage across every lighting condition.
    """
    hsv_mask = build_orange_mask(image_bgr)
    lab_mask = build_lab_orange_mask(image_bgr)
    return cv2.bitwise_or(hsv_mask, lab_mask)


def _find_local_peaks(
    profile: np.ndarray,
    min_separation: int = 14,
    min_height_relative: float = 0.18,
) -> list[int]:
    """Find indices of local maxima in a 1-D density profile."""
    if len(profile) == 0:
        return []
    smoothed = cv2.GaussianBlur(
        profile.astype(np.float32).reshape(-1, 1), (1, 5), 0
    ).ravel()
    threshold = float(smoothed.max()) * min_height_relative
    cands: list[tuple[int, float]] = []
    for i in range(1, len(smoothed) - 1):
        if (
            smoothed[i] >= threshold
            and smoothed[i] >= smoothed[i - 1]
            and smoothed[i] >= smoothed[i + 1]
        ):
            cands.append((i, float(smoothed[i])))
    if not cands:
        return []
    cands.sort()
    kept: list[tuple[int, float]] = []
    for idx, h in cands:
        if not kept or idx - kept[-1][0] >= min_separation:
            kept.append((idx, h))
        elif h > kept[-1][1]:
            kept[-1] = (idx, h)
    return [k[0] for k in kept]


def _infer_grid_positions(peaks: list[int], extent: int) -> list[int]:
    """Infer all expected grid-line positions from detected peaks.

    Returns a sorted list of positions covering 0..extent.

    BUG FIX (half-grid-offset)
    --------------------------
    ``np.median(peaks)`` for an **even** number of peaks returns the AVERAGE
    of the two middle peaks — which for an evenly-spaced grid is exactly
    halfway between two actual grid lines.  All projected positions then
    end up offset by half a grid spacing, sitting in the centres of grid
    cells instead of on the lines themselves.

    Fix: pick the middle element of the sorted peak list as the anchor.
    This guarantees the anchor is an ACTUAL detected peak, so projections
    align to real grid lines.
    """
    if len(peaks) < 3:
        return list(peaks)
    diffs = np.diff(peaks)
    diffs_sorted = np.sort(diffs)
    n_keep = max(1, int(0.7 * len(diffs_sorted)))
    spacing = float(np.median(diffs_sorted[:n_keep]))
    if spacing < 5:
        return list(peaks)
    # Anchor on an ACTUAL detected peak — the middle element of sorted peaks.
    # For odd N this matches np.median; for even N it avoids the half-step
    # offset that np.median would produce.
    peaks_sorted = sorted(peaks)
    anchor = float(peaks_sorted[len(peaks_sorted) // 2])
    n_below = int(np.ceil(anchor / spacing)) + 2
    n_above = int(np.ceil((extent - anchor) / spacing)) + 2
    out = []
    for k in range(-n_below, n_above + 1):
        pos = anchor + k * spacing
        if 0 <= pos < extent:
            out.append(int(round(pos)))
    return sorted(set(out))


def detect_per_line_grid_bands(
    orange_mask: np.ndarray,
    n_strips: int = 14,
    search_radius_px: int = 30,
    band_half_thickness_px: int = 5,
    peak_min_separation_px: int = 14,
    peak_min_height_relative: float = 0.15,
    min_axis_coverage: float = 0.50,
) -> np.ndarray:
    """**Per-line geometry mask with text-aware continuity filter.**

    Draws bands ONLY where there is direct orange evidence of a grid
    line (no projection/inference of missing lines).  Each detected peak
    is verified by axis-coverage: a real grid line spans roughly the
    entire image axis (>= 95%); orange TEXT labels span only a small
    fraction (~30%).  A coverage threshold of 50% cleanly separates
    grid lines from text labels, eliminating the duplicate bands that
    text would otherwise generate in the projection-peak step.

    Algorithm
    ---------
    1. Project orange mask onto Y (row density) and X (column density)
       to find candidate grid-line positions.
    2. **Continuity filter** — for each candidate, check what fraction
       of the perpendicular axis has orange near the candidate
       position.  Reject peaks below ``min_axis_coverage``.
    3. For each surviving peak, fit a quadratic curve through the
       actual orange centroids in ``n_strips`` strips along the line.
    4. Rasterise the curve as a thick band.

    This was chosen over the spacing-inference approach because the
    user verified empirically that direct per-line detection picks up
    every real grid line accurately (no need to project missing ones),
    while spacing inference is fragile to detection noise.
    """
    H, W = orange_mask.shape
    bands = np.zeros((H, W), dtype=np.uint8)
    if not np.any(orange_mask):
        return bands

    strong_bool = orange_mask > 0
    row_density = strong_bool.sum(axis=1) / max(1, W)
    col_density = strong_bool.sum(axis=0) / max(1, H)
    h_peaks = _find_local_peaks(
        row_density, peak_min_separation_px, peak_min_height_relative
    )
    v_peaks = _find_local_peaks(
        col_density, peak_min_separation_px, peak_min_height_relative
    )

    # ------------------------------------------------------------------
    # Continuity filter (G2b approach) — reject peaks that don't span
    # most of the image axis.  This drops orange-text false peaks while
    # preserving genuine grid-line peaks.
    # ------------------------------------------------------------------
    def _horiz_coverage(y: int) -> float:
        y0 = max(0, y - search_radius_px)
        y1 = min(H, y + search_radius_px + 1)
        return float(strong_bool[y0:y1, :].any(axis=0).mean())

    def _vert_coverage(x: int) -> float:
        x0 = max(0, x - search_radius_px)
        x1 = min(W, x + search_radius_px + 1)
        return float(strong_bool[:, x0:x1].any(axis=1).mean())

    h_peaks = [y for y in h_peaks if _horiz_coverage(y) >= min_axis_coverage]
    v_peaks = [x for x in v_peaks if _vert_coverage(x) >= min_axis_coverage]

    thickness = 2 * band_half_thickness_px + 1
    strip_w = max(1, W // n_strips)
    strip_h = max(1, H // n_strips)

    # ---- Horizontal lines: fit y as a polynomial of x ----
    for y_approx in h_peaks:
        xs_a: list[float] = []
        ys_a: list[float] = []
        for s in range(n_strips):
            x0 = s * strip_w
            x1 = min((s + 1) * strip_w, W)
            xc = (x0 + x1) / 2.0
            y0 = max(0, y_approx - search_radius_px)
            y1 = min(H, y_approx + search_radius_px + 1)
            region = strong_bool[y0:y1, x0:x1]
            ys_local, _ = np.where(region)
            if len(ys_local) >= 10:
                xs_a.append(xc)
                ys_a.append(y0 + float(np.median(ys_local)))
        if len(xs_a) < 4:
            # Insufficient anchors -> straight fallback at peak position
            cv2.line(bands, (0, int(y_approx)), (W - 1, int(y_approx)),
                     255, thickness)
            continue
        deg = 2 if len(xs_a) >= 5 else 1
        poly = np.poly1d(np.polyfit(xs_a, ys_a, deg=deg))
        xs_dense = np.arange(W)
        ys_dense = np.clip(poly(xs_dense), 0, H - 1).astype(np.int32)
        pts = np.column_stack([xs_dense, ys_dense]).reshape(-1, 1, 2)
        cv2.polylines(bands, [pts], isClosed=False, color=255,
                      thickness=thickness)

    # ---- Vertical lines: fit x as a polynomial of y ----
    for x_approx in v_peaks:
        ys_a = []
        xs_a = []
        for s in range(n_strips):
            y0 = s * strip_h
            y1 = min((s + 1) * strip_h, H)
            yc = (y0 + y1) / 2.0
            x0 = max(0, x_approx - search_radius_px)
            x1 = min(W, x_approx + search_radius_px + 1)
            region = strong_bool[y0:y1, x0:x1]
            _, xs_local = np.where(region)
            if len(xs_local) >= 10:
                ys_a.append(yc)
                xs_a.append(x0 + float(np.median(xs_local)))
        if len(ys_a) < 4:
            cv2.line(bands, (int(x_approx), 0), (int(x_approx), H - 1),
                     255, thickness)
            continue
        deg = 2 if len(ys_a) >= 5 else 1
        poly = np.poly1d(np.polyfit(ys_a, xs_a, deg=deg))
        ys_dense = np.arange(H)
        xs_dense = np.clip(poly(ys_dense), 0, W - 1).astype(np.int32)
        pts = np.column_stack([xs_dense, ys_dense]).reshape(-1, 1, 2)
        cv2.polylines(bands, [pts], isClosed=False, color=255,
                      thickness=thickness)

    print(
        f"  per-line bands: {len(h_peaks)}h+{len(v_peaks)}v lines after "
        f"continuity filter (>= {min_axis_coverage:.0%}), thickness "
        f"{thickness}px, coverage {bands.mean() / 2.55:.2f}%"
    )
    return bands


def detect_hybrid_grid_bands(
    orange_mask: np.ndarray,
    n_strips: int = 14,
    search_radius_px: int = 30,
    band_half_thickness_px: int = 5,
    peak_min_separation_px: int = 14,
    peak_min_height_relative: float = 0.15,
) -> np.ndarray:
    """**Hybrid M6 + M7**: spacing-inference + polynomial curve fitting.

    Combines the strengths of M7 (full-grid projection from spacing
    inference) and M6 (polynomial curve fit per line):

    1. Detect peaks in row / column orange-density projections — gives
       reliable grid-line positions wherever colour signal is strong.
    2. **M7 step** — infer regular spacing from the strong peaks, project
       a complete grid (all expected positions, even those with weak or
       missing colour evidence — typically the shadowed top-right corner).
    3. **M6 step** — for each projected position (detected OR inferred),
       attempt a quadratic curve fit using local orange centroids within
       a thin window.  If <4 anchors are found at an inferred-only
       position (weak colour signal), fall back to a straight line at
       that position.

    The result is grid coverage at EVERY expected line, with each line
    tracking its true curve wherever orange evidence exists.
    """
    H, W = orange_mask.shape
    bands = np.zeros((H, W), dtype=np.uint8)
    if not np.any(orange_mask):
        return bands

    strong_bool = orange_mask > 0
    row_density = strong_bool.sum(axis=1) / max(1, W)
    col_density = strong_bool.sum(axis=0) / max(1, H)
    h_peaks = _find_local_peaks(
        row_density, peak_min_separation_px, peak_min_height_relative
    )
    v_peaks = _find_local_peaks(
        col_density, peak_min_separation_px, peak_min_height_relative
    )

    # M7: infer all positions
    h_all = _infer_grid_positions(h_peaks, H)
    v_all = _infer_grid_positions(v_peaks, W)

    thickness = 2 * band_half_thickness_px + 1
    strip_w = max(1, W // n_strips)
    strip_h = max(1, H // n_strips)

    def _draw_horiz(y_approx: int) -> None:
        xs_anchor: list[float] = []
        ys_anchor: list[float] = []
        for s in range(n_strips):
            x0 = s * strip_w
            x1 = min((s + 1) * strip_w, W)
            xc = (x0 + x1) / 2.0
            y0 = max(0, y_approx - search_radius_px)
            y1 = min(H, y_approx + search_radius_px + 1)
            region = strong_bool[y0:y1, x0:x1]
            ys_local, _ = np.where(region)
            if len(ys_local) >= 10:
                xs_anchor.append(xc)
                ys_anchor.append(y0 + float(np.median(ys_local)))
        if len(xs_anchor) < 4:
            cv2.line(bands, (0, int(y_approx)), (W - 1, int(y_approx)),
                     255, thickness)
            return
        deg = 2 if len(xs_anchor) >= 5 else 1
        coeffs = np.polyfit(xs_anchor, ys_anchor, deg=deg)
        poly = np.poly1d(coeffs)
        xs_dense = np.arange(W)
        ys_dense = np.clip(poly(xs_dense), 0, H - 1).astype(np.int32)
        pts = np.column_stack([xs_dense, ys_dense]).reshape(-1, 1, 2)
        cv2.polylines(bands, [pts], isClosed=False, color=255,
                      thickness=thickness)

    def _draw_vert(x_approx: int) -> None:
        ys_anchor: list[float] = []
        xs_anchor: list[float] = []
        for s in range(n_strips):
            y0 = s * strip_h
            y1 = min((s + 1) * strip_h, H)
            yc = (y0 + y1) / 2.0
            x0 = max(0, x_approx - search_radius_px)
            x1 = min(W, x_approx + search_radius_px + 1)
            region = strong_bool[y0:y1, x0:x1]
            _, xs_local = np.where(region)
            if len(xs_local) >= 10:
                ys_anchor.append(yc)
                xs_anchor.append(x0 + float(np.median(xs_local)))
        if len(ys_anchor) < 4:
            cv2.line(bands, (int(x_approx), 0), (int(x_approx), H - 1),
                     255, thickness)
            return
        deg = 2 if len(ys_anchor) >= 5 else 1
        coeffs = np.polyfit(ys_anchor, xs_anchor, deg=deg)
        poly = np.poly1d(coeffs)
        ys_dense = np.arange(H)
        xs_dense = np.clip(poly(ys_dense), 0, W - 1).astype(np.int32)
        pts = np.column_stack([xs_dense, ys_dense]).reshape(-1, 1, 2)
        cv2.polylines(bands, [pts], isClosed=False, color=255,
                      thickness=thickness)

    for y in h_all:
        _draw_horiz(int(y))
    for x in v_all:
        _draw_vert(int(x))

    print(
        f"  hybrid bands: detected {len(h_peaks)}h+{len(v_peaks)}v -> "
        f"projected {len(h_all)}h+{len(v_all)}v with curve fitting, "
        f"thickness {thickness}px, coverage {bands.mean() / 2.55:.2f}%"
    )
    return bands


def detect_curved_grid_bands(
    orange_mask: np.ndarray,
    n_strips: int = 14,
    search_radius_px: int = 25,
    band_half_thickness_px: int = 5,
    peak_min_separation_px: int = 14,
    peak_min_height_relative: float = 0.18,
) -> np.ndarray:
    """Polynomial-fit curved grid bands — handles non-linear rectification.

    The straight-Hough version (``detect_grid_line_bands``) misses grid lines
    in image corners when residual rectification distortion bends them
    away from a single straight slope.  This curved version:

    1. Projects the orange mask onto Y (row density) and X (column density)
       to locate approximate grid-line positions.
    2. For each candidate row/column, divides the image into *n_strips*
       segments perpendicular to the line, and finds the median orange-
       pixel coordinate inside a thin window around the expected position.
       This yields a set of anchor points along the line's true path.
    3. Fits a quadratic polynomial through the anchors.
    4. Rasterises the polynomial as a thick band across the full image.

    This catches grid lines that curve away from a single straight slope —
    the failure mode the straight version exhibits in the top/bottom-right
    corners of photo scans.
    """
    H, W = orange_mask.shape
    bands = np.zeros((H, W), dtype=np.uint8)
    if not np.any(orange_mask):
        return bands

    strong_bool = orange_mask > 0
    row_density = strong_bool.sum(axis=1) / max(1, W)
    col_density = strong_bool.sum(axis=0) / max(1, H)
    h_peaks = _find_local_peaks(
        row_density, peak_min_separation_px, peak_min_height_relative
    )
    v_peaks = _find_local_peaks(
        col_density, peak_min_separation_px, peak_min_height_relative
    )
    thickness = 2 * band_half_thickness_px + 1

    # ---- Horizontal lines: fit y as a function of x ----
    strip_w = max(1, W // n_strips)
    for y_approx in h_peaks:
        xs_anchor: list[float] = []
        ys_anchor: list[float] = []
        for s in range(n_strips):
            x0 = s * strip_w
            x1 = min((s + 1) * strip_w, W)
            xc = (x0 + x1) / 2.0
            y0 = max(0, y_approx - search_radius_px)
            y1 = min(H, y_approx + search_radius_px + 1)
            region = strong_bool[y0:y1, x0:x1]
            ys_local, _ = np.where(region)
            if len(ys_local) >= 10:
                xs_anchor.append(xc)
                ys_anchor.append(y0 + float(np.median(ys_local)))
        if len(xs_anchor) < 4:
            cv2.line(bands, (0, int(y_approx)), (W - 1, int(y_approx)),
                     255, thickness)
            continue
        deg = 2 if len(xs_anchor) >= 5 else 1
        coeffs = np.polyfit(xs_anchor, ys_anchor, deg=deg)
        poly = np.poly1d(coeffs)
        xs_dense = np.arange(W)
        ys_dense = np.clip(poly(xs_dense), 0, H - 1).astype(np.int32)
        pts = np.column_stack([xs_dense, ys_dense]).reshape(-1, 1, 2)
        cv2.polylines(bands, [pts], isClosed=False, color=255,
                      thickness=thickness)

    # ---- Vertical lines: fit x as a function of y ----
    strip_h = max(1, H // n_strips)
    for x_approx in v_peaks:
        ys_anchor = []
        xs_anchor = []
        for s in range(n_strips):
            y0 = s * strip_h
            y1 = min((s + 1) * strip_h, H)
            yc = (y0 + y1) / 2.0
            x0 = max(0, x_approx - search_radius_px)
            x1 = min(W, x_approx + search_radius_px + 1)
            region = strong_bool[y0:y1, x0:x1]
            _, xs_local = np.where(region)
            if len(xs_local) >= 10:
                ys_anchor.append(yc)
                xs_anchor.append(x0 + float(np.median(xs_local)))
        if len(ys_anchor) < 4:
            cv2.line(bands, (int(x_approx), 0), (int(x_approx), H - 1),
                     255, thickness)
            continue
        deg = 2 if len(ys_anchor) >= 5 else 1
        coeffs = np.polyfit(ys_anchor, xs_anchor, deg=deg)
        poly = np.poly1d(coeffs)
        ys_dense = np.arange(H)
        xs_dense = np.clip(poly(ys_dense), 0, W - 1).astype(np.int32)
        pts = np.column_stack([xs_dense, ys_dense]).reshape(-1, 1, 2)
        cv2.polylines(bands, [pts], isClosed=False, color=255,
                      thickness=thickness)

    print(
        f"  curved bands: {len(h_peaks)} horiz + {len(v_peaks)} vert lines, "
        f"thickness {thickness}px, band coverage "
        f"{bands.mean() / 2.55:.2f}%"
    )
    return bands


def detect_grid_line_bands(
    orange_mask: np.ndarray,
    min_line_length_px: int = 200,
    max_line_gap_px: int = 60,
    angle_tolerance_deg: float = 3.0,
    band_half_thickness_px: int = 4,
    hough_vote_factor: float = 0.4,
) -> np.ndarray:
    """Find long near-axis-aligned grid lines and rasterise them as a band mask.

    Rationale
    ---------
    After Phase 1 rectification every grid line on the page is near-horizontal
    or near-vertical and stretches the full design area.  Even where tools are
    drawn over them the lines remain continuous on either side and produce
    plenty of orange-detected pixels.  Customer ink, by contrast, almost
    never forms a near-perfect 200+ px axis-aligned straight segment.

    Strategy
    --------
    1. Bridge tiny gaps in *orange_mask* with a 3 px dilation so Hough sees
       a continuous line at each grid position.
    2. Run probabilistic Hough on the bridged mask.
    3. Keep only segments whose orientation is within
       *angle_tolerance_deg* of horizontal or vertical.
    4. For each kept segment, **extend it across the full image** along its
       own slope (preserves the small tilt from rectification).
    5. Rasterise each extended line as a stripe of total thickness
       ``2 * band_half_thickness_px + 1`` onto the returned mask.

    The returned mask becomes the search territory for relaxed-threshold
    orange detection (see ``build_relaxed_orange_in_bands``).
    """
    H, W = orange_mask.shape
    band_mask = np.zeros((H, W), dtype=np.uint8)
    if not np.any(orange_mask):
        return band_mask

    # Bridge sub-px breaks so a single line votes as one
    bridged = cv2.dilate(
        orange_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )

    lines = cv2.HoughLinesP(
        bridged,
        rho=1,
        theta=np.pi / 720.0,  # 0.25 degree resolution
        threshold=max(20, int(min_line_length_px * hough_vote_factor)),
        minLineLength=min_line_length_px,
        maxLineGap=max_line_gap_px,
    )
    if lines is None:
        return band_mask

    # ------------------------------------------------------------------
    # Hough returns many overlapping segments per real grid line — cluster
    # by perpendicular coordinate so each physical grid line collapses to
    # a single band.  Cluster tolerance = band thickness, so adjacent
    # grid lines stay distinct.
    # ------------------------------------------------------------------
    horiz: list[tuple[float, float]] = []  # (y_mid, slope_dy_per_dx)
    verti: list[tuple[float, float]] = []  # (x_mid, slope_dx_per_dy)
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        if dx == 0 and dy == 0:
            continue
        ang = np.degrees(np.arctan2(abs(dy), abs(dx)))
        if ang <= angle_tolerance_deg and dx != 0:
            # Project to image-centre column to normalise mid-y across slopes
            slope = dy / dx
            y_at_centre = y1 + slope * (W / 2.0 - x1)
            horiz.append((y_at_centre, slope))
        elif ang >= (90.0 - angle_tolerance_deg) and dy != 0:
            inv_slope = dx / dy
            x_at_centre = x1 + inv_slope * (H / 2.0 - y1)
            verti.append((x_at_centre, inv_slope))

    def _cluster(entries: list[tuple[float, float]], tol: float) -> list[tuple[float, float]]:
        """Cluster (position, slope) entries by position; return mean of each cluster."""
        if not entries:
            return []
        entries = sorted(entries, key=lambda e: e[0])
        clusters: list[list[tuple[float, float]]] = [[entries[0]]]
        for e in entries[1:]:
            if e[0] - clusters[-1][-1][0] <= tol:
                clusters[-1].append(e)
            else:
                clusters.append([e])
        out = []
        for c in clusters:
            pos = float(np.mean([e[0] for e in c]))
            sl  = float(np.mean([e[1] for e in c]))
            out.append((pos, sl))
        return out

    cluster_tol = float(2 * band_half_thickness_px + 1)
    horiz_clusters = _cluster(horiz, cluster_tol)
    verti_clusters = _cluster(verti, cluster_tol)

    thickness = 2 * band_half_thickness_px + 1
    for y_centre, slope in horiz_clusters:
        # Extend across full width along the average slope
        y_left  = y_centre - slope * (W / 2.0)
        y_right = y_centre + slope * (W / 2.0 - 1)
        cv2.line(
            band_mask,
            (0, int(round(y_left))),
            (W - 1, int(round(y_right))),
            255, thickness,
        )

    for x_centre, inv_slope in verti_clusters:
        x_top    = x_centre - inv_slope * (H / 2.0)
        x_bottom = x_centre + inv_slope * (H / 2.0 - 1)
        cv2.line(
            band_mask,
            (int(round(x_top)), 0),
            (int(round(x_bottom)), H - 1),
            255, thickness,
        )

    print(
        f"  grid-line extension: {len(lines)} Hough segments -> "
        f"{len(horiz_clusters)} horizontal + {len(verti_clusters)} vertical lines "
        f"-> band coverage {band_mask.mean() / 255.0 * 100:.2f}%"
    )
    return band_mask


def catch_orange_near_bands(
    image_bgr: np.ndarray,
    band_mask: np.ndarray,
    proximity_px: int = 25,
    hue_low: int = 0,
    hue_high: int = 45,
    sat_min: int = 8,
    val_min: int = 5,
    chroma_boost: float = 4.5,
    max_component_area_px: int = 15000,
) -> np.ndarray:
    """Catch orange components that sit alongside (but not strictly inside)
    a confirmed Hough grid band.

    Why
    ---
    Hough detects only long *continuous* axis-aligned lines.  The CDS template
    also prints **dashed boundary lines** and **text labels** ("Boundary line
    (Small insert)", small alphabetic markers) in the same orange ink.  These
    are too broken / too short for Hough to extend across the page, so the
    per-band relaxed detector misses them.  However, they are always physically
    located *immediately adjacent to* a real grid line — that's the geometric
    property we exploit here.

    Strategy
    --------
    1. Dilate the Hough band mask by *proximity_px* in every direction.
       Anything within that envelope is in the "definitely template ink"
       neighbourhood of a grid line.
    2. Apply relaxed colour detection on the chroma-boosted image (same
       thresholds as ``build_relaxed_orange_in_bands``) intersected with the
       proximity envelope.
    3. Filter to **small connected components only** (``area <
       max_component_area_px``).  This keeps text characters, tiny dashes,
       and stray flecks — and lets any large blob through to the customer-
       trace pipeline.  Tools can extend along grid lines but they don't
       sit *as small isolated chromatic blobs* in a grid-line neighbourhood.

    Returns
    -------
    Binary uint8 mask of pixels to add to the removal set.
    """
    if not np.any(band_mask):
        return np.zeros_like(band_mask)

    # Step 1 — proximity envelope around the strict bands
    if proximity_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (proximity_px * 2 + 1, proximity_px * 2 + 1)
        )
        proximity = cv2.dilate(band_mask, kernel, iterations=1)
    else:
        proximity = band_mask

    # Step 2 — relaxed orange (chroma-boosted)
    boosted = boost_chroma(image_bgr, gain=chroma_boost)
    hsv = cv2.cvtColor(boosted, cv2.COLOR_BGR2HSV)
    lower = np.array([hue_low, sat_min, val_min], dtype=np.uint8)
    upper = np.array([hue_high, 255, 255], dtype=np.uint8)
    relaxed = cv2.inRange(hsv, lower, upper)
    near_band = cv2.bitwise_and(relaxed, proximity)

    if not np.any(near_band):
        return near_band

    # Step 3 — only keep small components (text, dashes, flecks)
    n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(near_band, connectivity=8)
    if n_lbl <= 1:
        return near_band
    areas = stats[:, cv2.CC_STAT_AREA]
    lut = np.where(
        (areas > 0) & (areas <= max_component_area_px), np.uint8(255), np.uint8(0)
    )
    lut[0] = 0
    filtered = lut[labels].astype(np.uint8)

    n_kept = int(((filtered > 0).sum()) / max(1, filtered.size) * 100000) / 1000.0
    print(f"  near-band sweep: proximity {proximity_px}px, "
          f"caught {int((filtered > 0).sum()):,} extra px ({n_kept:.3f}%)")
    return filtered


def build_relaxed_orange_in_bands(
    image_bgr: np.ndarray,
    band_mask: np.ndarray,
    hue_low: int = 0,
    hue_high: int = 45,
    sat_min: int = 8,
    val_min: int = 5,
    chroma_boost: float = 4.5,
) -> np.ndarray:
    """Catch faint orange remnants *inside* confirmed grid-line bands.

    Inside a confirmed grid band — a row/column that we already proved
    carries a long continuous orange line — any chromatic pixel is grid,
    because customer ink does not follow rectified-axis geometry.  So we
    can crank the boost up and the saturation floor down without worrying
    about false-positives on customer marks (and the downstream
    chromaticity-aware dark-ink repair still recovers anything dark and
    truly achromatic that gets caught at a tool/grid crossing).

    Returns a binary mask whose non-zero pixels are exactly the relaxed-
    orange-AND-in-band pixels.
    """
    if not np.any(band_mask):
        return np.zeros_like(band_mask)
    boosted = boost_chroma(image_bgr, gain=chroma_boost)
    hsv = cv2.cvtColor(boosted, cv2.COLOR_BGR2HSV)
    lower = np.array([hue_low, sat_min, val_min], dtype=np.uint8)
    upper = np.array([hue_high, 255, 255], dtype=np.uint8)
    relaxed = cv2.inRange(hsv, lower, upper)
    return cv2.bitwise_and(relaxed, band_mask)


def build_trace_mask(
    cleaned_bgr: np.ndarray,
    dark_abs_threshold: int = 170,
    rel_dark_sigma: float = 25.0,
    rel_dark_min: int = 10,
    min_component_px: int = 8,
) -> np.ndarray:
    """Find customer marks remaining after orange grid removal.

    Combines two detectors:
    * **Absolute**: gray < dark_abs_threshold  (catches solid pen / dark pencil)
    * **Relative**: pixel is darker than its local background by ≥ rel_dark_min
      (catches light pencil on slightly off-white paper / photos)

    Parameters
    ----------
    dark_abs_threshold:
        Gray value below which a pixel is always classified as a mark.
        170 is generous: catches graphite and most pen colours without
        false-triggering on cream-coloured paper.
    rel_dark_sigma:
        Gaussian sigma for local background estimation (px).  Large value =
        broad background region.
    rel_dark_min:
        Minimum local relative darkness to classify as a mark.
    min_component_px:
        Discard isolated specks smaller than this (noise / JPEG artefacts).
    """
    gray = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)

    # Absolute dark pixels
    abs_dark = (gray < dark_abs_threshold).astype(np.uint8) * 255

    # Locally dark pixels (relative to surrounding background)
    local_bg = cv2.GaussianBlur(
        gray, (0, 0), sigmaX=rel_dark_sigma, sigmaY=rel_dark_sigma
    )
    rel_dark = cv2.subtract(local_bg, gray)
    rel_mask = (rel_dark >= rel_dark_min).astype(np.uint8) * 255

    combined = cv2.bitwise_or(abs_dark, rel_mask)
    if min_component_px > 0:
        combined = _remove_small_components(combined, min_area_px=min_component_px)
    return combined


def _orange_coverage(
    image_bgr: np.ndarray,
    hue_low: int = ORANGE_HUE_LOW,
    hue_high: int = ORANGE_HUE_HIGH,
    sat_min: int = ORANGE_SAT_MIN,
    val_min: int = ORANGE_VAL_MIN,
    probe_dim: int = 600,
) -> float:
    """Return fraction of image pixels in the orange hue range (fast probe)."""
    h, w = image_bgr.shape[:2]
    scale = min(1.0, probe_dim / max(h, w))
    probe = (
        cv2.resize(image_bgr, (int(w * scale), int(h * scale)),
                   interpolation=cv2.INTER_AREA)
        if scale < 1.0 else image_bgr
    )
    probe_boosted = boost_chroma(probe, gain=ORANGE_CHROMA_BOOST)
    hsv = cv2.cvtColor(probe_boosted, cv2.COLOR_BGR2HSV)
    lower = np.array([hue_low, sat_min, val_min], dtype=np.uint8)
    upper = np.array([hue_high, 255, 255], dtype=np.uint8)
    return float(cv2.inRange(hsv, lower, upper).mean()) / 255.0


# ---------------------------------------------------------------------------
# Black grid path (preserved for experimentation / backward compat)
# ---------------------------------------------------------------------------

def _line_positions_mm(total_mm: float, spacing_mm: float) -> list[float]:
    positions: list[float] = []
    value = 0.0
    while value <= (total_mm + 1e-6):
        positions.append(value)
        value += spacing_mm
    return positions


def _build_projected_vertical_grid_mask(
    image_bgr: np.ndarray, template: TemplateModel, dilation_px: int = 2
) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    minor_mm = float(template.grid.minor_spacing_mm if template.grid else 10.0)
    major_mm = float(template.grid.major_spacing_mm if template.grid else 50.0)
    px_per_mm_x = w / float(template.design_area_mm.width)
    mask = np.zeros((h, w), dtype=np.uint8)
    for x_mm in _line_positions_mm(float(template.design_area_mm.width), minor_mm):
        x = int(np.clip(round(x_mm * px_per_mm_x), 0, w - 1))
        is_major = abs((x_mm / major_mm) - round(x_mm / major_mm)) < 1e-6
        cv2.line(mask, (x, 0), (x, h - 1), 255, thickness=2 if is_major else 1)
    if dilation_px > 0:
        mask = cv2.dilate(mask, np.ones((2 * dilation_px + 1,) * 2, dtype=np.uint8))
    return mask


def _build_projected_horizontal_grid_mask(
    image_bgr: np.ndarray, template: TemplateModel, dilation_px: int = 2
) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    minor_mm = float(template.grid.minor_spacing_mm if template.grid else 10.0)
    major_mm = float(template.grid.major_spacing_mm if template.grid else 50.0)
    px_per_mm_y = h / float(template.design_area_mm.height)
    mask = np.zeros((h, w), dtype=np.uint8)
    for y_mm in _line_positions_mm(float(template.design_area_mm.height), minor_mm):
        y = int(np.clip(round(y_mm * px_per_mm_y), 0, h - 1))
        is_major = abs((y_mm / major_mm) - round(y_mm / major_mm)) < 1e-6
        cv2.line(mask, (0, y), (w - 1, y), 255, thickness=2 if is_major else 1)
    if dilation_px > 0:
        mask = cv2.dilate(mask, np.ones((2 * dilation_px + 1,) * 2, dtype=np.uint8))
    return mask


def _build_black_grid_removal_mask(
    image_bgr: np.ndarray,
    template: TemplateModel,
    dark_threshold: int = 130,
    grid_line_width_mm: float = 0.90,
) -> tuple[np.ndarray, np.ndarray]:
    """Directional-erosion approach for black-on-white grids.

    Thin vertical lines disappear under wide horizontal erosion; thick customer
    marks survive.  Symmetrically for horizontal lines.  Avoids the connected-
    component contamination problem.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    px_per_mm = min(
        image_bgr.shape[1] / float(template.design_area_mm.width),
        image_bgr.shape[0] / float(template.design_area_mm.height),
    )
    erode_w = max(5, int(round(grid_line_width_mm * px_per_mm * 1.5)))
    if erode_w % 2 == 0:
        erode_w += 1

    dark = (gray < dark_threshold).astype(np.uint8) * 255

    k_h = np.ones((1, erode_w), dtype=np.uint8)
    k_v = np.ones((erode_w, 1), dtype=np.uint8)
    expand = np.ones((erode_w, erode_w), dtype=np.uint8)

    thick_h = cv2.erode(dark, k_h)
    protect_v = cv2.dilate(thick_h, expand)

    thick_v = cv2.erode(dark, k_v)
    protect_h = cv2.dilate(thick_v, expand)

    vert_proj = _build_projected_vertical_grid_mask(image_bgr, template)
    horiz_proj = _build_projected_horizontal_grid_mask(image_bgr, template)

    vert_rem = cv2.bitwise_and(dark, vert_proj)
    vert_rem = cv2.bitwise_and(vert_rem, cv2.bitwise_not(protect_v))

    horiz_rem = cv2.bitwise_and(dark, horiz_proj)
    horiz_rem = cv2.bitwise_and(horiz_rem, cv2.bitwise_not(protect_h))

    raw = cv2.bitwise_or(vert_rem, horiz_rem)
    cleaned = _remove_small_components(raw, min_area_px=4)
    return raw, cleaned


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def isolate_trace_candidates(
    image_bgr: np.ndarray,
    template: TemplateModel,
    grid_colour: GridColour = "auto",
    orange_threshold: float = 0.42,   # kept for API compat, not used in HSV path
    max_processing_dim: int = 1500,
) -> TraceIsolationResult:
    """Remove the printed grid, leaving only customer-drawn traces.

    Parameters
    ----------
    image_bgr:
        Scaled design-area crop from Phase 1.
    template:
        Matched template (needed only for black grid mode).
    grid_colour:
        ``"auto"`` — probe image to detect orange vs black grid (default).
        ``"orange"`` — force HSV hue-range masking.
        ``"black"``  — force directional-erosion thinness approach.
    max_processing_dim:
        Maximum dimension for internal processing (default 1500 px).  Both
        orange and black paths apply this: heavy per-pixel operations run at
        reduced resolution, then the removal mask is upscaled to full resolution
        and applied to the original image.  Set 0 or None to disable.
    """
    h_orig, w_orig = image_bgr.shape[:2]

    # --- Lighting normalisation (handles uneven shadow / overhead-light
    # fall-off in photo scans).  PDF inputs are unaffected because their
    # lighting is already uniform.  CLAHE on the LAB-L channel: brightens
    # dark regions, darkens bright regions, locally, so the whole image
    # has comparable luminance for downstream orange / dark-mark
    # detection.
    image_bgr = normalise_lighting(image_bgr)

    # --- Resolve auto ---
    if grid_colour == "auto":
        coverage = _orange_coverage(image_bgr)
        grid_colour = "orange" if coverage >= 0.005 else "black"

    # =========================================================
    # ORANGE PATH: full-resolution HSV masking + dark-mark repair
    #
    # Why full resolution?  Orange grid lines are often only 2-5 px wide.
    # INTER_AREA downsampling mixes them with surrounding white, dropping
    # saturation below detection threshold.  Intersections (2-D blocks)
    # survive downscaling; thin 1-D lines do not — producing the
    # "intersections removed, lines intact" symptom.
    #
    # The morph-close is intentionally tiny (2 px) — just anti-aliasing.
    # A large close would bridge over black customer marks at crossings.
    # Instead: a repair pass restores any genuinely dark pixels that the
    # orange mask accidentally covered.
    # =========================================================
    if grid_colour == "orange":
        # ==============================================================
        # L11 PRODUCTION APPROACH — two-stage HSV + hard-white-clamp.
        # ==============================================================
        # Stage 1: HSV chroma detection at sat>=25 (no boost, hue 0-40)
        #          catches the saturated MAJOR grid lines and orange text.
        #          paper_blended_fill replaces them with local paper colour.
        # Stage 2: Hard-clamp anything with gray >= 160 to pure white.
        #          The MINOR grid line colour (FFEBCC, gray ~200) and any
        #          anti-aliased grid edges sit above 160, so they get
        #          uniformly cleared.  Tool ink (gray < 160) is preserved.
        #
        # Why this works:
        # - Customer tool ink is meaningfully darker than ANY grid line
        #   colour (printed orange or legacy black), so a brightness
        #   threshold gives a clean discriminator that's robust to
        #   lighting / camera / JPEG noise.
        # - The HSV pass handles strongly-saturated grid colours that
        #   share brightness with tool ink (mid-tone oranges), and the
        #   hard-white clamp wipes the rest.
        # - Curved-band geometric detection is DISABLED because it was
        #   over-eating tool ink without solving the residual-grid issue.
        #
        # This approach also works on the legacy CDS template where the
        # grid is black/dark grey: the HSV pass catches nothing, but the
        # brightness clamp still cleans pages where the grid is printed
        # in a lighter tone than the customer ink.
        # --------------------------------------------------------------
        WHITE_CLAMP_GRAY_THRESHOLD = 160

        # Stage 1: HSV chroma detection (no boost) and paper-blend fill
        strong_orange = build_orange_mask(
            image_bgr,
            sat_min=25,
            chroma_boost=1.0,
        )
        orange_mask = strong_orange  # exposed via TraceIsolationResult
        removal_mask = strong_orange.copy()
        cleaned_bgr = paper_blended_fill(image_bgr, removal_mask)
        cleaned_bgr = normalise_paper_to_white(cleaned_bgr)

        # Stage 2: hard-clamp any pixel brighter than the threshold to
        # pure paper-white.  This removes the minor grid (FFEBCC) and
        # all remaining faint orange fringes without touching tool ink.
        gray_clean = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
        cleaned_bgr[gray_clean >= WHITE_CLAMP_GRAY_THRESHOLD] = (255, 255, 255)

        # Geometric-band placeholders for the result dataclass — disabled.
        grid_bands = np.zeros_like(strong_orange)

        # Trace mask at reduced resolution for speed
        if max_processing_dim and max(h_orig, w_orig) > max_processing_dim:
            scale = max_processing_dim / max(h_orig, w_orig)
            cleaned_proc = cv2.resize(
                cleaned_bgr,
                (max(1, int(round(w_orig * scale))), max(1, int(round(h_orig * scale)))),
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
            removal_mask=removal_mask,
            grid_colour="orange",
            orange_grid_mask=orange_mask,
        )

    # --- Shared downscale logic (black path only) ---
    if max_processing_dim and max(h_orig, w_orig) > max_processing_dim:
        scale = max_processing_dim / max(h_orig, w_orig)
        proc_w = max(1, int(round(w_orig * scale)))
        proc_h = max(1, int(round(h_orig * scale)))
        image_proc = cv2.resize(image_bgr, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
    else:
        image_proc = image_bgr
        scale = 1.0

    def _upsample_mask(mask: np.ndarray) -> np.ndarray:
        if scale >= 1.0:
            return mask
        up = cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        up = cv2.dilate(up, np.ones((3, 3), dtype=np.uint8), iterations=1)
        return up

    # =========================================================
    # BLACK PATH: directional-erosion (backward compat)
    # =========================================================
    raw_black, black_removal = _build_black_grid_removal_mask(image_proc, template)
    removal_full = _upsample_mask(black_removal)

    bg = cv2.GaussianBlur(image_bgr, (0, 0), sigmaX=5.0)
    cleaned_bgr = image_bgr.copy()
    cleaned_bgr[removal_full > 0] = bg[removal_full > 0]

    if scale < 1.0:
        cleaned_proc = cv2.resize(
            cleaned_bgr, (image_proc.shape[1], image_proc.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    else:
        cleaned_proc = cleaned_bgr

    trace_mask = build_trace_mask(cleaned_proc)

    return TraceIsolationResult(
        cleaned_bgr=cleaned_bgr,
        orange_mask=np.zeros(image_proc.shape[:2], dtype=np.uint8),
        trace_candidate_mask=trace_mask,
        note_candidate_seed_mask=trace_mask.copy(),
        removal_mask=black_removal,
        grid_colour="black",
        black_grid_mask=raw_black,
        black_removal_mask=black_removal,
    )
