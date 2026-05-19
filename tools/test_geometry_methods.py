"""Batch comparison of grid-GEOMETRY-detection methods on case 01.

After the user spotted that the previous hybrid bands were OFFSET by
half a grid spacing (median-anchor bug for even N peaks), this script
tests several alternative ways to compute the geometry mask, including
the bug-fixed version, a skeleton-centerline approach, and a morphology
based approach.

Each method outputs an OVERLAY showing where its bands fall ON TOP OF
the HSV orange detection — so it's visually obvious whether the bands
land on real grid lines or in between them.

    python tools/test_geometry_methods.py

Outputs:
    data/outputs/geometry_methods/BlueColourTest01/<timestamp>/
        G0_buggy_baseline/       # original median-anchor (offset bug)
            overlay.png          # bands GREEN + HSV detection RED
            mask.png             # band mask only
        G1_fixed_anchor/         # bug-fixed: use middle PEAK as anchor
        G2_per_line_strip/       # no projection; only draw detected peaks
                                 # with per-line strip-refined polynomial
        G3_skeleton/             # morph-skeleton centerlines + projection
        G4_long_morph/           # long-thin morphology opening
        G5_interpolate_missing/  # fixed-anchor + interp curve coeffs from
                                 # neighbours for missing-anchor lines
        summary_topright.png     # all methods' top-right corners stacked
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np

from armourcore_cds.phase2.trace_isolation import (
    normalise_lighting,
    build_orange_mask,
    build_lab_orange_mask,
    build_combined_orange_mask,
    _find_local_peaks,
)

REPO = Path(__file__).parent.parent


# =====================================================================
# Shared helpers
# =====================================================================

def _polyfit_horiz_line(
    orange_bool: np.ndarray,
    y_approx: int,
    n_strips: int = 14,
    search_radius_px: int = 30,
) -> np.poly1d | None:
    """Fit y as a polynomial of x for the horizontal line near y_approx."""
    H, W = orange_bool.shape
    strip_w = max(1, W // n_strips)
    xs: list[float] = []
    ys: list[float] = []
    for s in range(n_strips):
        x0 = s * strip_w
        x1 = min((s + 1) * strip_w, W)
        xc = (x0 + x1) / 2.0
        y0 = max(0, y_approx - search_radius_px)
        y1 = min(H, y_approx + search_radius_px + 1)
        region = orange_bool[y0:y1, x0:x1]
        ys_local, _ = np.where(region)
        if len(ys_local) >= 10:
            xs.append(xc)
            ys.append(y0 + float(np.median(ys_local)))
    if len(xs) < 4:
        return None
    deg = 2 if len(xs) >= 5 else 1
    return np.poly1d(np.polyfit(xs, ys, deg=deg))


def _polyfit_vert_line(
    orange_bool: np.ndarray,
    x_approx: int,
    n_strips: int = 14,
    search_radius_px: int = 30,
) -> np.poly1d | None:
    H, W = orange_bool.shape
    strip_h = max(1, H // n_strips)
    ys: list[float] = []
    xs: list[float] = []
    for s in range(n_strips):
        y0 = s * strip_h
        y1 = min((s + 1) * strip_h, H)
        yc = (y0 + y1) / 2.0
        x0 = max(0, x_approx - search_radius_px)
        x1 = min(W, x_approx + search_radius_px + 1)
        region = orange_bool[y0:y1, x0:x1]
        _, xs_local = np.where(region)
        if len(xs_local) >= 10:
            ys.append(yc)
            xs.append(x0 + float(np.median(xs_local)))
    if len(ys) < 4:
        return None
    deg = 2 if len(ys) >= 5 else 1
    return np.poly1d(np.polyfit(ys, xs, deg=deg))


def _draw_horiz_curve(bands: np.ndarray, poly: np.poly1d, thickness: int) -> None:
    H, W = bands.shape
    xs = np.arange(W)
    ys = np.clip(poly(xs), 0, H - 1).astype(np.int32)
    pts = np.column_stack([xs, ys]).reshape(-1, 1, 2)
    cv2.polylines(bands, [pts], isClosed=False, color=255, thickness=thickness)


def _draw_vert_curve(bands: np.ndarray, poly: np.poly1d, thickness: int) -> None:
    H, W = bands.shape
    ys = np.arange(H)
    xs = np.clip(poly(ys), 0, W - 1).astype(np.int32)
    pts = np.column_stack([xs, ys]).reshape(-1, 1, 2)
    cv2.polylines(bands, [pts], isClosed=False, color=255, thickness=thickness)


def _project_positions_buggy(peaks: list[int], extent: int) -> list[int]:
    """OLD median-anchor logic — reproduces the half-grid-offset bug."""
    if len(peaks) < 3:
        return list(peaks)
    diffs = np.diff(peaks)
    spacing = float(np.median(np.sort(diffs)[: max(1, int(0.7 * len(diffs)))]))
    if spacing < 5:
        return list(peaks)
    anchor = float(np.median(peaks))      # <- BUG: even N -> halfway
    out: list[int] = []
    k = 0
    while True:
        pos = anchor + k * spacing
        if pos >= extent: break
        if pos >= 0: out.append(int(round(pos)))
        k += 1
    k = -1
    while True:
        pos = anchor + k * spacing
        if pos < 0: break
        out.append(int(round(pos)))
        k -= 1
    return sorted(set(out))


def _project_positions_fixed(peaks: list[int], extent: int) -> list[int]:
    """Bug-fixed: anchor on an actual peak."""
    if len(peaks) < 3:
        return list(peaks)
    diffs = np.diff(peaks)
    spacing = float(np.median(np.sort(diffs)[: max(1, int(0.7 * len(diffs)))]))
    if spacing < 5:
        return list(peaks)
    peaks_sorted = sorted(peaks)
    anchor = float(peaks_sorted[len(peaks_sorted) // 2])
    out: list[int] = []
    k = 0
    while True:
        pos = anchor + k * spacing
        if pos >= extent: break
        if pos >= 0: out.append(int(round(pos)))
        k += 1
    k = -1
    while True:
        pos = anchor + k * spacing
        if pos < 0: break
        out.append(int(round(pos)))
        k -= 1
    return sorted(set(out))


# =====================================================================
# Methods
# =====================================================================

def G0_buggy_baseline(orange_mask: np.ndarray) -> np.ndarray:
    """Original median-anchor logic — shows the bug visually."""
    H, W = orange_mask.shape
    bands = np.zeros((H, W), dtype=np.uint8)
    strong_bool = orange_mask > 0
    h_peaks = _find_local_peaks((strong_bool.sum(axis=1) / max(1, W)), 14, 0.15)
    v_peaks = _find_local_peaks((strong_bool.sum(axis=0) / max(1, H)), 14, 0.15)
    h_all = _project_positions_buggy(h_peaks, H)
    v_all = _project_positions_buggy(v_peaks, W)
    th = 11
    for y in h_all:
        poly = _polyfit_horiz_line(strong_bool, int(y))
        if poly is not None:
            _draw_horiz_curve(bands, poly, th)
        else:
            cv2.line(bands, (0, int(y)), (W - 1, int(y)), 255, th)
    for x in v_all:
        poly = _polyfit_vert_line(strong_bool, int(x))
        if poly is not None:
            _draw_vert_curve(bands, poly, th)
        else:
            cv2.line(bands, (int(x), 0), (int(x), H - 1), 255, th)
    print(f"  G0: {len(h_peaks)}h+{len(v_peaks)}v peaks -> "
          f"{len(h_all)}h+{len(v_all)}v projected (buggy anchor)")
    return bands


def G1_fixed_anchor(orange_mask: np.ndarray) -> np.ndarray:
    """Bug-fixed: anchor on actual peak."""
    H, W = orange_mask.shape
    bands = np.zeros((H, W), dtype=np.uint8)
    strong_bool = orange_mask > 0
    h_peaks = _find_local_peaks((strong_bool.sum(axis=1) / max(1, W)), 14, 0.15)
    v_peaks = _find_local_peaks((strong_bool.sum(axis=0) / max(1, H)), 14, 0.15)
    h_all = _project_positions_fixed(h_peaks, H)
    v_all = _project_positions_fixed(v_peaks, W)
    th = 11
    for y in h_all:
        poly = _polyfit_horiz_line(strong_bool, int(y))
        if poly is not None:
            _draw_horiz_curve(bands, poly, th)
        else:
            cv2.line(bands, (0, int(y)), (W - 1, int(y)), 255, th)
    for x in v_all:
        poly = _polyfit_vert_line(strong_bool, int(x))
        if poly is not None:
            _draw_vert_curve(bands, poly, th)
        else:
            cv2.line(bands, (int(x), 0), (int(x), H - 1), 255, th)
    print(f"  G1: {len(h_peaks)}h+{len(v_peaks)}v peaks -> "
          f"{len(h_all)}h+{len(v_all)}v projected (fixed anchor)")
    return bands


def G2_per_line_only(orange_mask: np.ndarray) -> np.ndarray:
    """Draw only the DETECTED peaks (no projection of missing lines).

    Each detected peak gets a polynomial curve fit through its actual
    orange centroids.  No inference / projection — what you see is
    purely evidence-based.
    """
    H, W = orange_mask.shape
    bands = np.zeros((H, W), dtype=np.uint8)
    strong_bool = orange_mask > 0
    h_peaks = _find_local_peaks((strong_bool.sum(axis=1) / max(1, W)), 14, 0.15)
    v_peaks = _find_local_peaks((strong_bool.sum(axis=0) / max(1, H)), 14, 0.15)
    th = 11
    for y in h_peaks:
        poly = _polyfit_horiz_line(strong_bool, int(y))
        if poly is not None:
            _draw_horiz_curve(bands, poly, th)
    for x in v_peaks:
        poly = _polyfit_vert_line(strong_bool, int(x))
        if poly is not None:
            _draw_vert_curve(bands, poly, th)
    print(f"  G2: {len(h_peaks)}h+{len(v_peaks)}v detected peaks (no inference)")
    return bands


def G3_skeleton(orange_mask: np.ndarray) -> np.ndarray:
    """Skeleton-centerline approach.

    1. Filter the orange mask to keep only long-thin axis-aligned components
       (drops orange text blobs).
    2. Skeletonise each.
    3. For each skeleton, compute curve fit through its pixel centroids.
    4. Augment with fixed-anchor spacing inference for missing lines.
    """
    from skimage.morphology import skeletonize

    H, W = orange_mask.shape
    bands = np.zeros((H, W), dtype=np.uint8)

    # Long-thin filter: extract long horizontal and long vertical lines via
    # morphological opening with thin elongated kernels.
    line_len = max(50, int(0.15 * min(H, W)))
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (line_len, 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_len))
    h_lines = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, hk)
    v_lines = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, vk)

    # Skeletonise (gives 1-px-thick centerlines)
    h_skel = skeletonize(h_lines > 0).astype(np.uint8) * 255
    v_skel = skeletonize(v_lines > 0).astype(np.uint8) * 255

    strong_bool = orange_mask > 0
    h_peaks = _find_local_peaks((h_skel > 0).sum(axis=1).astype(np.float32) / max(1, W),
                                14, 0.15)
    v_peaks = _find_local_peaks((v_skel > 0).sum(axis=0).astype(np.float32) / max(1, H),
                                14, 0.15)
    h_all = _project_positions_fixed(h_peaks, H)
    v_all = _project_positions_fixed(v_peaks, W)
    th = 11
    for y in h_all:
        poly = _polyfit_horiz_line(h_skel > 0, int(y))
        if poly is not None:
            _draw_horiz_curve(bands, poly, th)
        else:
            cv2.line(bands, (0, int(y)), (W - 1, int(y)), 255, th)
    for x in v_all:
        poly = _polyfit_vert_line(v_skel > 0, int(x))
        if poly is not None:
            _draw_vert_curve(bands, poly, th)
        else:
            cv2.line(bands, (int(x), 0), (int(x), H - 1), 255, th)
    print(f"  G3: skel {len(h_peaks)}h+{len(v_peaks)}v -> "
          f"{len(h_all)}h+{len(v_all)}v projected")
    return bands


def G4_long_morph(orange_mask: np.ndarray) -> np.ndarray:
    """Long thin-kernel morphology opening — DIRECTLY extracts grid lines.

    No peak detection / projection.  The opened mask IS the grid line
    coverage; we just dilate it slightly for the band envelope.
    """
    H, W = orange_mask.shape
    line_len = max(80, int(0.20 * min(H, W)))
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (line_len, 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_len))
    h_lines = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, hk)
    v_lines = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, vk)
    bands = cv2.bitwise_or(h_lines, v_lines)
    # Thicken to a real band envelope
    bands = cv2.dilate(bands,
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                       iterations=1)
    print(f"  G4: long-morph opened directly")
    return bands


def G5_interpolate_coeffs(orange_mask: np.ndarray) -> np.ndarray:
    """G1 + interpolated polynomial coefficients for projected-only lines.

    For each projected position WITHOUT direct anchors (corners with weak
    detection), copy the polynomial coefficients from the nearest line
    that DOES have anchors, and just shift the constant term to the
    projected position.  This gives missing-line projections the same
    curve shape as their neighbours — capturing the global distortion
    even where colour evidence is too weak for local curve fitting.
    """
    H, W = orange_mask.shape
    bands = np.zeros((H, W), dtype=np.uint8)
    strong_bool = orange_mask > 0
    h_peaks = _find_local_peaks((strong_bool.sum(axis=1) / max(1, W)), 14, 0.15)
    v_peaks = _find_local_peaks((strong_bool.sum(axis=0) / max(1, H)), 14, 0.15)
    h_all = _project_positions_fixed(h_peaks, H)
    v_all = _project_positions_fixed(v_peaks, W)
    th = 11

    # Compute polynomial fits where possible; store as dict {y_approx: poly}
    h_fits: dict[int, np.poly1d] = {}
    for y in h_all:
        poly = _polyfit_horiz_line(strong_bool, int(y))
        if poly is not None:
            h_fits[int(y)] = poly
    v_fits: dict[int, np.poly1d] = {}
    for x in v_all:
        poly = _polyfit_vert_line(strong_bool, int(x))
        if poly is not None:
            v_fits[int(x)] = poly

    def _nearest_fit(target: int, fits: dict[int, np.poly1d]) -> np.poly1d | None:
        if not fits:
            return None
        keys = sorted(fits.keys())
        idx = min(range(len(keys)), key=lambda i: abs(keys[i] - target))
        return fits[keys[idx]]

    for y in h_all:
        if int(y) in h_fits:
            _draw_horiz_curve(bands, h_fits[int(y)], th)
        else:
            ref = _nearest_fit(int(y), h_fits)
            if ref is None:
                cv2.line(bands, (0, int(y)), (W - 1, int(y)), 255, th)
                continue
            # Shift the polynomial vertically so its anchor matches y
            # Reference poly evaluated at image centre gives its y position
            ref_centre = float(ref(W / 2.0))
            shifted = np.poly1d(list(ref))
            shifted[0] += (y - ref_centre)
            _draw_horiz_curve(bands, shifted, th)

    for x in v_all:
        if int(x) in v_fits:
            _draw_vert_curve(bands, v_fits[int(x)], th)
        else:
            ref = _nearest_fit(int(x), v_fits)
            if ref is None:
                cv2.line(bands, (int(x), 0), (int(x), H - 1), 255, th)
                continue
            ref_centre = float(ref(H / 2.0))
            shifted = np.poly1d(list(ref))
            shifted[0] += (x - ref_centre)
            _draw_vert_curve(bands, shifted, th)

    print(f"  G5: {len(h_fits)} h-fits / {len(h_all)} positions, "
          f"{len(v_fits)} v-fits / {len(v_all)} positions")
    return bands


METHODS: list[tuple[str, Callable[[np.ndarray], np.ndarray]]] = [
    ("G0_buggy_baseline",     G0_buggy_baseline),
    ("G1_fixed_anchor",       G1_fixed_anchor),
    ("G2_per_line_only",      G2_per_line_only),
    ("G3_skeleton",           G3_skeleton),
    ("G4_long_morph",         G4_long_morph),
    ("G5_interpolate_coeffs", G5_interpolate_coeffs),
]


# =====================================================================
# Runner
# =====================================================================

def _overlay_bands_and_orange(
    base: np.ndarray, bands: np.ndarray, orange: np.ndarray,
) -> np.ndarray:
    """Show bands (green) and orange detection (red) over the input.

    Where both overlap -> yellow.  Lets us see at a glance whether bands
    are landing ON the actual orange grid.
    """
    out = base.copy().astype(np.float32)
    green = np.zeros_like(out); green[..., 1] = 255
    red   = np.zeros_like(out); red[..., 2]   = 255
    alpha = 0.55
    mask_bands  = bands  > 0
    mask_orange = orange > 0
    out[mask_orange] = (1 - alpha) * out[mask_orange] + alpha * red  [mask_orange]
    out[mask_bands]  = (1 - alpha) * out[mask_bands]  + alpha * green[mask_bands]
    return np.clip(out, 0, 255).astype(np.uint8)


def _label(img, txt, scale=0.6):
    out = img.copy()
    band = int(34 * scale)
    cv2.rectangle(out, (0, 0), (out.shape[1], band), (0, 0, 0), -1)
    cv2.putText(out, txt, (8, band - 8), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main() -> None:
    name = "BlueColourTest01"
    p1_run = sorted((REPO / "outputs" / "runs").glob(f"*{name}*"))[-1]
    rect_path = p1_run / "scaled_design_area.png"
    img = cv2.imread(str(rect_path))
    img = normalise_lighting(img)

    out_root = (REPO / "data" / "outputs" / "geometry_methods" / name
                / time.strftime("%Y%m%d-%H%M%S"))
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"output: {out_root.relative_to(REPO)}\n")

    # Reference orange mask (HSV+LAB combined) — same for every method
    orange = build_combined_orange_mask(img)
    cv2.imwrite(str(out_root / "_orange_detection.png"),
                _overlay_bands_and_orange(img, np.zeros_like(orange), orange))

    topright_crops: list[np.ndarray] = []
    H, W = img.shape[:2]
    y2 = int(H * 0.40); x1 = int(W * 0.58)

    for name_m, fn in METHODS:
        print(f"[{name_m}]")
        t0 = time.time()
        bands = fn(orange)
        elapsed = time.time() - t0

        sub = out_root / name_m
        sub.mkdir(exist_ok=True)
        cv2.imwrite(str(sub / "mask.png"), bands)
        overlay = _overlay_bands_and_orange(img, bands, orange)
        cv2.imwrite(str(sub / "overlay.png"), overlay)
        print(f"  {elapsed:.2f}s")

        topright_crops.append(
            _label(overlay[:y2, x1:], name_m)
        )

    # Side-by-side top-right strip
    strip = np.vstack(topright_crops)
    cv2.imwrite(str(out_root / "summary_topright_stacked.png"), strip)

    # 3x2 grid
    h_c, w_c = topright_crops[0].shape[:2]
    rows, cols = 2, 3
    grid = np.full((rows * h_c, cols * w_c, 3), 255, dtype=np.uint8)
    for i, c in enumerate(topright_crops):
        r_, c_ = divmod(i, cols)
        grid[r_ * h_c:(r_ + 1) * h_c, c_ * w_c:(c_ + 1) * w_c] = c
    cv2.imwrite(str(out_root / "summary_topright_grid.png"), grid)

    print(f"\nview: {out_root.relative_to(REPO)}/summary_topright_grid.png")


if __name__ == "__main__":
    main()
