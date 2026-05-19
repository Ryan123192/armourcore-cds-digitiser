"""Batch comparison of grid-removal strategies on a single rectified image.

Why this exists
---------------
The current production pipeline (chroma-boosted HSV + straight Hough bands +
near-band sweep) works well in the body of the image but leaves significant
grid remnants in shadowed corners — especially BlueColourTest01's top-right.
The root cause is that real rectified grids retain slight non-linear
distortion, so the bands' "straight-line" assumption misses curved grid
segments at the image periphery.

This script runs 8 different grid-removal approaches on the SAME input image
and writes:

  data/outputs/grid_method_comparison/<case>/<timestamp>/
    M0_strong_hsv/                # current strong HSV alone
      mask_overlay.png            # input with removal mask in red
      cleaned.png                 # image with mask filled by paper blend
    M1_full_current/              # current production
    M2_lab_a_channel/             # brightness-independent LAB a* threshold
    M3_lab_full/                  # LAB + Hough + near-band (LAB-based M1)
    M4_bg_divide/                 # divide by Gaussian background, then HSV
    M5_local_threshold/           # adaptive thresholding on saturation
    M6_curved_bands/              # polynomial-fit (curved) Hough bands
    M7_inferred_grid/             # spacing-inference projection of full grid
    summary.png                   # all 8 cleaned outputs side by side
    scoreboard.txt                # persistent-orange counts, runtimes

Usage:
    python tools/test_grid_removal_methods.py BlueColourTest01
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np

from armourcore_cds.phase2.trace_isolation import (
    normalise_lighting,
    normalise_paper_to_white,
    boost_chroma,
    paper_blended_fill,
    build_orange_mask,
    detect_grid_line_bands,
    build_relaxed_orange_in_bands,
    catch_orange_near_bands,
    DARK_MARK_THRESHOLD,
    ACHROMATIC_SAT_MAX,
)

REPO = Path(__file__).parent.parent


# =====================================================================
# Common helpers
# =====================================================================

def apply_dark_mark_protection(
    orange_mask: np.ndarray,
    geom_validated: np.ndarray | None,
    image_bgr: np.ndarray,
) -> np.ndarray:
    """Restore dark+achromatic pixels caught only by colour detection.

    Mirrors the production rule: pixels with geometric validation skip the
    protection (the geometry already proves they're template).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    sat  = hsv[..., 1]
    is_dark = gray < DARK_MARK_THRESHOLD
    is_ach  = sat  < ACHROMATIC_SAT_MAX
    colour_only = orange_mask > 0
    if geom_validated is not None:
        colour_only = colour_only & ~(geom_validated > 0)
    dark_ink = colour_only & is_dark & is_ach
    out = orange_mask.copy()
    out[dark_ink] = 0
    return out


def find_local_peaks(
    profile: np.ndarray,
    min_separation: int = 12,
    min_height_relative: float = 0.20,
) -> list[int]:
    """Find local maxima in a 1-D density profile.

    Used by both M6 (curved bands) and M7 (inferred grid).
    Returns indices of peaks above ``min_height_relative * max(profile)``,
    with neighbouring peaks within ``min_separation`` of each other
    collapsed to a single maximum.
    """
    if len(profile) == 0:
        return []
    # Smooth the profile slightly
    smoothed = cv2.GaussianBlur(profile.astype(np.float32).reshape(-1, 1),
                                (1, 5), 0).ravel()
    threshold = float(smoothed.max()) * min_height_relative
    candidates = []
    for i in range(1, len(smoothed) - 1):
        if (
            smoothed[i] >= threshold
            and smoothed[i] >= smoothed[i - 1]
            and smoothed[i] >= smoothed[i + 1]
        ):
            candidates.append((i, float(smoothed[i])))
    if not candidates:
        return []
    # Suppress neighbouring peaks: walk left to right, keep highest in window
    candidates.sort()
    kept: list[tuple[int, float]] = []
    for idx, h in candidates:
        if not kept or idx - kept[-1][0] >= min_separation:
            kept.append((idx, h))
        elif h > kept[-1][1]:
            kept[-1] = (idx, h)
    return [k[0] for k in kept]


# =====================================================================
# Method implementations
# =====================================================================

def m0_strong_hsv(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Just the chroma-boosted HSV detection — no geometric extension."""
    strong = build_orange_mask(img)
    return strong, np.zeros_like(strong)


def m1_full_current(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Current production: strong + straight Hough bands + near-band sweep."""
    strong = build_orange_mask(img)
    H, W = strong.shape
    min_len = max(150, int(0.10 * min(H, W)))
    bands = detect_grid_line_bands(strong, min_line_length_px=min_len)
    relaxed = build_relaxed_orange_in_bands(img, bands)
    near = catch_orange_near_bands(img, bands)
    mask = strong | relaxed | near
    geom = relaxed | near
    return mask, geom


def m2_lab_a_channel(img: np.ndarray, a_threshold: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """LAB a*-channel threshold — brightness-INDEPENDENT orange detection.

    Why this might fix the top-right corner:
    HSV saturation depends on Value; very dark orange in shadow has both
    low saturation AND low value, making it hard to detect.  LAB's a* axis
    is decoupled from luminance — it just measures red-green chromaticity.
    Orange ink has a* ≈ 145-170 (positive offset from neutral 128);
    pencil / paper have a* ≈ 126-132.  A simple threshold on a* catches
    orange independent of how shadowed it is.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    a = lab[..., 1].astype(np.int16)
    mask = ((a - 128) > a_threshold).astype(np.uint8) * 255
    # Small close to bridge anti-aliasing
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    # Drop very small components
    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8)
    areas = stats[:, cv2.CC_STAT_AREA]
    lut = np.where(areas >= 15, np.uint8(255), np.uint8(0))
    lut[0] = 0
    return lut[lbl].astype(np.uint8), np.zeros_like(mask)


def m3_lab_full(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """LAB a* detection + Hough bands + near-band — combines M2's strength
    with the current geometric extension framework."""
    strong, _ = m2_lab_a_channel(img)
    H, W = strong.shape
    min_len = max(150, int(0.10 * min(H, W)))
    bands = detect_grid_line_bands(strong, min_line_length_px=min_len)
    relaxed = build_relaxed_orange_in_bands(img, bands)
    near = catch_orange_near_bands(img, bands)
    mask = strong | relaxed | near
    geom = relaxed | near
    return mask, geom


def m4_bg_divide(img: np.ndarray, sigma: float = 60.0,
                 a_threshold: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Divide by Gaussian-blurred background, then LAB a* detection.

    Heavy Gaussian blur estimates the local paper colour at each pixel.
    Dividing the original by this background normalises away ALL lighting
    variation (shadows, hand-shadow, page-curve fall-off).  After division,
    the orange grid stands out as a uniform deviation from the now-flat
    paper background everywhere on the page.

    This is more powerful than CLAHE because it operates pointwise rather
    than per-tile, eliminating tile-boundary artefacts.
    """
    img_f = img.astype(np.float32)
    bg = cv2.GaussianBlur(img_f, (0, 0), sigma)
    bg = np.maximum(bg, 1.0)
    # Pointwise divide-and-rescale to paper-white reference
    normalised = (img_f / bg) * 220.0
    normalised = np.clip(normalised, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(normalised, cv2.COLOR_BGR2LAB)
    a = lab[..., 1].astype(np.int16)
    mask = ((a - 128) > a_threshold).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask, np.zeros_like(mask)


def m5_local_threshold(img: np.ndarray, block: int = 41,
                       offset: int = -4) -> tuple[np.ndarray, np.ndarray]:
    """Adaptive thresholding on LAB a* + S magnitude.

    cv2.adaptiveThreshold compares each pixel to its LOCAL mean (block × block).
    Even where the absolute orange saturation is low (shadowed corner), the
    grid still stands out from its immediate paper neighbourhood, so the
    relative threshold catches it.

    Combines a* channel adaptive threshold (chromaticity) with a check that
    the pixel really is in the orange hue range (not just any dark spot).
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    a = lab[..., 1]
    # Adaptive threshold on a* — pixel must be locally more red/orange than its
    # surrounding paper.
    a_adapt = cv2.adaptiveThreshold(
        a, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block, offset,
    )
    # Sanity check: the a* must still be > 128 (warm hue), not < 128 (cool)
    hue_warm = (a >= 128).astype(np.uint8) * 255
    mask = cv2.bitwise_and(a_adapt, hue_warm)
    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8)
    areas = stats[:, cv2.CC_STAT_AREA]
    lut = np.where(areas >= 12, np.uint8(255), np.uint8(0))
    lut[0] = 0
    return lut[lbl].astype(np.uint8), np.zeros_like(mask)


def m6_curved_bands(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Polynomial-fit curved bands for distorted grid lines.

    Replaces M1's straight bands with quadratic polynomials fit to the
    actual orange-pixel centroids along each grid line.  This handles
    the residual lens / rectification curvature that escapes the
    straight-Hough approach, especially near image corners.

    Algorithm
    ---------
    1. Strong HSV detection → seed mask.
    2. Find approximate horizontal grid row positions by projecting onto Y.
    3. For each row position, divide x into 12 strips; in each strip,
       find the median y of orange pixels in a thin window around the
       expected y.  This gives 12 (x_centre, y_centroid) anchor points.
    4. Polynomial-fit a quadratic y(x) through the anchors.
    5. Rasterise the curve as a thick band across the full image width.
    6. Repeat for vertical lines (x(y)).
    7. Apply relaxed colour detection within these curved bands.
    """
    strong = build_orange_mask(img)
    H, W = strong.shape
    n_strips = 14
    band_thickness = 9
    search_radius = 25

    # Approximate row/column line positions
    row_density = (strong > 0).sum(axis=1) / max(1, W)
    col_density = (strong > 0).sum(axis=0) / max(1, H)
    h_peaks = find_local_peaks(row_density, min_separation=14,
                               min_height_relative=0.18)
    v_peaks = find_local_peaks(col_density, min_separation=14,
                               min_height_relative=0.18)

    bands = np.zeros_like(strong)
    strong_bool = strong > 0

    # ---- Horizontal lines: fit y as a quadratic of x ----
    strip_w = W // n_strips
    for y_approx in h_peaks:
        xs_anchor: list[float] = []
        ys_anchor: list[float] = []
        for s in range(n_strips):
            x0 = s * strip_w
            x1 = min((s + 1) * strip_w, W)
            xc = (x0 + x1) / 2.0
            y0 = max(0, y_approx - search_radius)
            y1 = min(H, y_approx + search_radius + 1)
            region = strong_bool[y0:y1, x0:x1]
            ys_local, _ = np.where(region)
            if len(ys_local) >= 10:
                xs_anchor.append(xc)
                ys_anchor.append(y0 + float(np.median(ys_local)))
        if len(xs_anchor) < 4:
            cv2.line(bands, (0, int(y_approx)), (W - 1, int(y_approx)),
                     255, band_thickness)
            continue
        deg = 2 if len(xs_anchor) >= 5 else 1
        coeffs = np.polyfit(xs_anchor, ys_anchor, deg=deg)
        poly = np.poly1d(coeffs)
        xs_dense = np.arange(W)
        ys_dense = np.clip(poly(xs_dense), 0, H - 1).astype(np.int32)
        pts = np.column_stack([xs_dense, ys_dense]).reshape(-1, 1, 2)
        cv2.polylines(bands, [pts], isClosed=False, color=255,
                      thickness=band_thickness)

    # ---- Vertical lines: fit x as a quadratic of y ----
    strip_h = H // n_strips
    for x_approx in v_peaks:
        ys_anchor = []
        xs_anchor = []
        for s in range(n_strips):
            y0 = s * strip_h
            y1 = min((s + 1) * strip_h, H)
            yc = (y0 + y1) / 2.0
            x0 = max(0, x_approx - search_radius)
            x1 = min(W, x_approx + search_radius + 1)
            region = strong_bool[y0:y1, x0:x1]
            _, xs_local = np.where(region)
            if len(xs_local) >= 10:
                ys_anchor.append(yc)
                xs_anchor.append(x0 + float(np.median(xs_local)))
        if len(ys_anchor) < 4:
            cv2.line(bands, (int(x_approx), 0), (int(x_approx), H - 1),
                     255, band_thickness)
            continue
        deg = 2 if len(ys_anchor) >= 5 else 1
        coeffs = np.polyfit(ys_anchor, xs_anchor, deg=deg)
        poly = np.poly1d(coeffs)
        ys_dense = np.arange(H)
        xs_dense = np.clip(poly(ys_dense), 0, W - 1).astype(np.int32)
        pts = np.column_stack([xs_dense, ys_dense]).reshape(-1, 1, 2)
        cv2.polylines(bands, [pts], isClosed=False, color=255,
                      thickness=band_thickness)

    print(f"  M6 curved: {len(h_peaks)} horiz + {len(v_peaks)} vert lines, "
          f"band coverage {bands.mean()/2.55:.2f}%")

    relaxed = build_relaxed_orange_in_bands(img, bands)
    near = catch_orange_near_bands(img, bands)
    mask = strong | relaxed | near
    geom = relaxed | near
    return mask, geom


def m7_inferred_grid(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project a complete inferred grid using detected spacing.

    For a rectified design area the grid spacing is REGULAR.  If we detect
    even half of the grid positions reliably, we can interpolate / extrapolate
    the rest.  Once we have a full grid model, we project a thin band along
    every expected line, including ones with zero pixel-level evidence.

    This is the most aggressive approach: it inserts bands where the colour
    detector saw nothing.  Customer ink is protected by the chromaticity-
    aware dark-mark rule applied on the final union.
    """
    strong = build_orange_mask(img)
    H, W = strong.shape
    row_density = (strong > 0).sum(axis=1) / max(1, W)
    col_density = (strong > 0).sum(axis=0) / max(1, H)
    h_peaks = find_local_peaks(row_density, min_separation=14,
                               min_height_relative=0.18)
    v_peaks = find_local_peaks(col_density, min_separation=14,
                               min_height_relative=0.18)

    bands = np.zeros_like(strong)
    band_thickness = 9

    def _project(peaks: list[int], extent: int, axis: str) -> list[int]:
        if len(peaks) < 3:
            return list(peaks)
        diffs = np.diff(peaks)
        # Use the median of the LOWER 70% of diffs (avoid being thrown off
        # by missing-line gaps that double the spacing).
        diffs_sorted = np.sort(diffs)
        n_keep = max(1, int(0.7 * len(diffs_sorted)))
        spacing = float(np.median(diffs_sorted[:n_keep]))
        if spacing < 5:
            return list(peaks)
        # Anchor at the median of the detected peaks (most reliable centre)
        anchor = float(np.median(peaks))
        # Generate all grid positions within [0, extent]
        n_below = int(np.ceil(anchor / spacing)) + 2
        n_above = int(np.ceil((extent - anchor) / spacing)) + 2
        projected = [int(round(anchor + k * spacing))
                     for k in range(-n_below, n_above + 1)
                     if 0 <= anchor + k * spacing < extent]
        return projected

    inferred_h = _project(h_peaks, H, axis="y")
    inferred_v = _project(v_peaks, W, axis="x")

    for y in inferred_h:
        cv2.line(bands, (0, y), (W - 1, y), 255, band_thickness)
    for x in inferred_v:
        cv2.line(bands, (x, 0), (x, H - 1), 255, band_thickness)

    print(f"  M7 inferred: detected {len(h_peaks)}h+{len(v_peaks)}v, "
          f"projected to {len(inferred_h)}h+{len(inferred_v)}v lines")

    relaxed = build_relaxed_orange_in_bands(img, bands)
    near = catch_orange_near_bands(img, bands)
    mask = strong | relaxed | near
    geom = bands | relaxed | near
    return mask, geom


# =====================================================================
# Runner
# =====================================================================

@dataclass
class MethodResult:
    name: str
    elapsed_s: float
    mask_px: int
    persistent_px: int
    cleaned_path: Path
    overlay_path: Path


METHODS: list[tuple[str, callable]] = [
    ("M0_strong_hsv",     m0_strong_hsv),
    ("M1_full_current",   m1_full_current),
    ("M2_lab_a_channel",  m2_lab_a_channel),
    ("M3_lab_full",       m3_lab_full),
    ("M4_bg_divide",      m4_bg_divide),
    ("M5_local_threshold", m5_local_threshold),
    ("M6_curved_bands",   m6_curved_bands),
    ("M7_inferred_grid",  m7_inferred_grid),
]


def _persistent_score(image_norm: np.ndarray, cleaned: np.ndarray) -> int:
    """Count pixels that were chromatic in input AND retained ≥50% of
    saturation in the cleaned output — proxy for "grid not removed"."""
    hsv_in  = cv2.cvtColor(image_norm, cv2.COLOR_BGR2HSV)
    hsv_out = cv2.cvtColor(cleaned, cv2.COLOR_BGR2HSV)
    in_orange = (
        ((hsv_in[..., 0] <= 40) | (hsv_in[..., 0] >= 175))
        & (hsv_in[..., 1] > 5)
        & (hsv_in[..., 2] > 30)
    )
    sat_in  = hsv_in[..., 1].astype(np.int32)
    sat_out = hsv_out[..., 1].astype(np.int32)
    retained = sat_out / np.maximum(sat_in, 1)
    persistent = in_orange & (sat_out > 15) & (retained > 0.5)
    return int(persistent.sum())


def _overlay(base: np.ndarray, mask: np.ndarray,
             colour=(0, 0, 255), alpha: float = 0.55) -> np.ndarray:
    """Translucent red overlay of mask on base."""
    out = base.copy()
    paint = np.zeros_like(base)
    paint[:] = colour
    blend = cv2.addWeighted(out, 1.0 - alpha, paint, alpha, 0)
    out = np.where(mask[..., None] > 0, blend, out)
    return out


def _label(img: np.ndarray, text: str) -> np.ndarray:
    """Draw a label band at the top of the image."""
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", default="BlueColourTest01", nargs="?")
    ap.add_argument("--max-dim", type=int, default=1500,
                    help="Downscale max dimension for fair / fast comparison.")
    args = ap.parse_args()

    # ----- locate latest Phase 1 rectified output -----
    candidates = sorted((REPO / "outputs" / "runs").glob(f"*{args.name}*"))
    if not candidates:
        sys.exit(f"No Phase 1 runs found for {args.name}")
    rect_path = candidates[-1] / "scaled_design_area.png"
    if not rect_path.exists():
        sys.exit(f"Missing {rect_path}")

    out_root = (REPO / "data" / "outputs" / "grid_method_comparison"
                / args.name / time.strftime("%Y%m%d-%H%M%S"))
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"output: {out_root.relative_to(REPO)}\n")

    img = cv2.imread(str(rect_path))
    # Downscale for fair comparison
    H0, W0 = img.shape[:2]
    if max(H0, W0) > args.max_dim:
        scale = args.max_dim / max(H0, W0)
        img = cv2.resize(img, (int(W0 * scale), int(H0 * scale)),
                         interpolation=cv2.INTER_AREA)

    # Pre-process exactly as production does (lighting normalisation only —
    # paper-to-white normalisation happens AFTER orange removal).
    img_norm = normalise_lighting(img)

    results: list[MethodResult] = []
    for name, fn in METHODS:
        print(f"[{name}]")
        t0 = time.time()
        try:
            mask, geom = fn(img_norm)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue
        elapsed = time.time() - t0

        # Apply production-style dark-mark protection
        removal_mask = apply_dark_mark_protection(mask, geom, img_norm)

        # Build cleaned image (paper-blended fill + paper-to-white)
        cleaned = paper_blended_fill(img_norm, removal_mask)
        cleaned = normalise_paper_to_white(cleaned)

        # Score & save
        persistent = _persistent_score(img_norm, cleaned)

        sub = out_root / name
        sub.mkdir(exist_ok=True)
        overlay_path = sub / "mask_overlay.png"
        cleaned_path = sub / "cleaned.png"
        cv2.imwrite(str(overlay_path), _overlay(img_norm, removal_mask))
        cv2.imwrite(str(cleaned_path), cleaned)

        n_mask = int((removal_mask > 0).sum())
        print(f"  removed {n_mask:,} px | persistent {persistent:,} | "
              f"{elapsed:.1f}s")
        results.append(MethodResult(name, elapsed, n_mask, persistent,
                                    cleaned_path, overlay_path))

    # ----- summary image (4 cols × 2 rows of cleaned outputs) -----
    if results:
        cleans = [cv2.imread(str(r.cleaned_path)) for r in results]
        labelled = [_label(c, f"{r.name}  pers={r.persistent_px}")
                    for c, r in zip(cleans, results)]
        h, w = labelled[0].shape[:2]
        n = len(labelled)
        cols = 4
        rows = (n + cols - 1) // cols
        canvas = np.full((rows * h, cols * w, 3), 255, dtype=np.uint8)
        for i, img_i in enumerate(labelled):
            r_, c_ = divmod(i, cols)
            canvas[r_ * h:(r_ + 1) * h, c_ * w:(c_ + 1) * w] = img_i
        cv2.imwrite(str(out_root / "summary_cleaned.png"), canvas)

        overlays = [cv2.imread(str(r.overlay_path)) for r in results]
        labelled_ovr = [_label(c, f"{r.name}  mask={r.mask_px:,}")
                        for c, r in zip(overlays, results)]
        canvas_o = np.full((rows * h, cols * w, 3), 255, dtype=np.uint8)
        for i, img_i in enumerate(labelled_ovr):
            r_, c_ = divmod(i, cols)
            canvas_o[r_ * h:(r_ + 1) * h, c_ * w:(c_ + 1) * w] = img_i
        cv2.imwrite(str(out_root / "summary_overlay.png"), canvas_o)

        # Scoreboard
        sb_lines = [
            f"{'method':<22} {'mask_px':>10} {'persistent':>12} {'time_s':>8}",
            "-" * 60,
        ]
        results_sorted = sorted(results, key=lambda r: r.persistent_px)
        for r in results_sorted:
            sb_lines.append(
                f"{r.name:<22} {r.mask_px:>10,} {r.persistent_px:>12,} "
                f"{r.elapsed_s:>8.1f}"
            )
        scoreboard = "\n".join(sb_lines)
        (out_root / "scoreboard.txt").write_text(scoreboard, encoding="utf-8")
        print("\n" + scoreboard)
        print(f"\nsummary_cleaned.png and summary_overlay.png in: "
              f"{out_root.relative_to(REPO)}")


if __name__ == "__main__":
    main()
