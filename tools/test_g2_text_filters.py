"""Variants of G2_per_line_only that suppress the false peaks created
by the orange template TEXT ("Boundary Line (Small Insert)" etc.).

G2 wins on accuracy because it draws ONLY what's evidenced — but the
orange text label is also evidenced and creates several extra peaks
in the column-density projection, drawing duplicate/false bands in
that area.

This script tries four different filters, each addressing the text
problem from a different angle, and produces overlay PNGs so we can
compare them visually:

    G2_baseline               -- original G2 (shows the text artefact)
    G2a_long_morph_prefilter  -- remove blob-shaped components BEFORE peak
                                 detection via long-thin morphology open
    G2b_continuity_post       -- reject peaks whose orange coverage along
                                 the line is too low (text only covers a
                                 small fraction of the axis length)
    G2c_aspect_ratio_compfilt -- filter connected components by aspect
                                 ratio: keep only long-thin axis-aligned
    G2d_spacing_outlier       -- after detection, drop peaks that break
                                 the global spacing pattern (text peaks
                                 cluster too close to existing peaks)

    python tools/test_g2_text_filters.py
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
    build_combined_orange_mask,
    _find_local_peaks,
)

REPO = Path(__file__).parent.parent


# =====================================================================
# Shared curve-fitting helpers (same as test_geometry_methods.py)
# =====================================================================

def _polyfit_horiz_line(orange_bool, y_approx, n_strips=14, search_radius_px=30):
    H, W = orange_bool.shape
    strip_w = max(1, W // n_strips)
    xs, ys = [], []
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


def _polyfit_vert_line(orange_bool, x_approx, n_strips=14, search_radius_px=30):
    H, W = orange_bool.shape
    strip_h = max(1, H // n_strips)
    ys, xs = [], []
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


def _draw_horiz_curve(bands, poly, thickness):
    H, W = bands.shape
    xs = np.arange(W)
    ys = np.clip(poly(xs), 0, H - 1).astype(np.int32)
    pts = np.column_stack([xs, ys]).reshape(-1, 1, 2)
    cv2.polylines(bands, [pts], isClosed=False, color=255, thickness=thickness)


def _draw_vert_curve(bands, poly, thickness):
    H, W = bands.shape
    ys = np.arange(H)
    xs = np.clip(poly(ys), 0, W - 1).astype(np.int32)
    pts = np.column_stack([xs, ys]).reshape(-1, 1, 2)
    cv2.polylines(bands, [pts], isClosed=False, color=255, thickness=thickness)


def _draw_bands_from_peaks(
    orange_bool: np.ndarray,
    h_peaks: list[int],
    v_peaks: list[int],
    thickness: int = 11,
) -> np.ndarray:
    H, W = orange_bool.shape
    bands = np.zeros((H, W), dtype=np.uint8)
    for y in h_peaks:
        poly = _polyfit_horiz_line(orange_bool, int(y))
        if poly is not None:
            _draw_horiz_curve(bands, poly, thickness)
    for x in v_peaks:
        poly = _polyfit_vert_line(orange_bool, int(x))
        if poly is not None:
            _draw_vert_curve(bands, poly, thickness)
    return bands


def _project_peaks(orange_bool: np.ndarray) -> tuple[list[int], list[int]]:
    H, W = orange_bool.shape
    row_density = orange_bool.sum(axis=1) / max(1, W)
    col_density = orange_bool.sum(axis=0) / max(1, H)
    h_peaks = _find_local_peaks(row_density, 14, 0.15)
    v_peaks = _find_local_peaks(col_density, 14, 0.15)
    return h_peaks, v_peaks


# =====================================================================
# Methods
# =====================================================================

def G2_baseline(orange_mask: np.ndarray) -> np.ndarray:
    """Original G2 — shows the orange-text false peaks."""
    strong_bool = orange_mask > 0
    h_peaks, v_peaks = _project_peaks(strong_bool)
    print(f"  G2 baseline: {len(h_peaks)}h+{len(v_peaks)}v peaks")
    return _draw_bands_from_peaks(strong_bool, h_peaks, v_peaks)


def G2a_long_morph_prefilter(orange_mask: np.ndarray) -> np.ndarray:
    """Apply long-thin morphological opening to drop text/blob components.

    Each character of "Boundary Line (Small Insert)" is roughly 10x15 px
    — a chunky blob, not a long thin line.  An opening with a long thin
    kernel (one direction) erases everything that ISN'T already long-thin
    in that direction.  We do horizontal AND vertical openings and union
    them so axis-aligned grid lines (which are long-thin in one direction)
    survive while text characters are wiped out.
    """
    H, W = orange_mask.shape
    line_len = max(50, int(0.12 * min(H, W)))
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (line_len, 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_len))
    h_only = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, hk)
    v_only = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, vk)
    filtered = cv2.bitwise_or(h_only, v_only)
    strong_bool = filtered > 0

    h_peaks, v_peaks = _project_peaks(strong_bool)
    print(f"  G2a long-morph: kept {filtered.mean() / orange_mask.mean() * 100:.0f}% "
          f"of orange, {len(h_peaks)}h+{len(v_peaks)}v peaks")
    return _draw_bands_from_peaks(strong_bool, h_peaks, v_peaks)


def G2b_continuity_post(
    orange_mask: np.ndarray,
    min_axis_coverage: float = 0.50,
    search_radius_px: int = 30,
) -> np.ndarray:
    """Reject peaks whose orange coverage along the line is too low.

    A grid line spans ~95% of the image axis.  The "Boundary Line" text
    label is only ~30% of the image height.  So for each candidate peak,
    we look at how many of the image's column/row strips have orange
    near the peak position.  Peaks with low coverage are dropped.
    """
    strong_bool = orange_mask > 0
    H, W = strong_bool.shape
    h_peaks, v_peaks = _project_peaks(strong_bool)

    def _horiz_coverage(y: int) -> float:
        y0 = max(0, y - search_radius_px)
        y1 = min(H, y + search_radius_px + 1)
        band = strong_bool[y0:y1, :]
        col_has_orange = band.any(axis=0)
        return float(col_has_orange.mean())

    def _vert_coverage(x: int) -> float:
        x0 = max(0, x - search_radius_px)
        x1 = min(W, x + search_radius_px + 1)
        band = strong_bool[:, x0:x1]
        row_has_orange = band.any(axis=1)
        return float(row_has_orange.mean())

    h_keep = [y for y in h_peaks if _horiz_coverage(y) >= min_axis_coverage]
    v_keep = [x for x in v_peaks if _vert_coverage(x) >= min_axis_coverage]
    n_h_drop = len(h_peaks) - len(h_keep)
    n_v_drop = len(v_peaks) - len(v_keep)
    print(f"  G2b continuity (>={min_axis_coverage:.0%}): "
          f"kept {len(h_keep)}h (dropped {n_h_drop}), "
          f"{len(v_keep)}v (dropped {n_v_drop})")
    return _draw_bands_from_peaks(strong_bool, h_keep, v_keep)


def G2c_aspect_ratio_compfilt(orange_mask: np.ndarray,
                               min_axis_extent_frac: float = 0.30,
                               max_thickness_px: int = 20) -> np.ndarray:
    """Connected-component filter by aspect ratio + extent.

    Each connected component has a bounding box.  Real grid-line pieces
    have one bbox dimension much larger than the other (long-thin), and
    the long dimension covers a substantial fraction of the image axis.
    Text-character components are roughly square (~15x15) and short.
    """
    H, W = orange_mask.shape
    n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(
        orange_mask, connectivity=8
    )
    keep_mask = np.zeros_like(orange_mask)
    n_kept = 0
    for i in range(1, n_lbl):
        x, y, w, h, area = stats[i]
        long_axis = max(w, h)
        short_axis = min(w, h)
        if short_axis > max_thickness_px:
            continue
        if (long_axis / max(W, H)) < min_axis_extent_frac:
            continue
        keep_mask[labels == i] = 255
        n_kept += 1
    print(f"  G2c aspect-filter: kept {n_kept}/{n_lbl - 1} components, "
          f"{keep_mask.mean() / max(1, orange_mask.mean()) * 100:.0f}% of orange")

    strong_bool = keep_mask > 0
    h_peaks, v_peaks = _project_peaks(strong_bool)
    print(f"    -> {len(h_peaks)}h+{len(v_peaks)}v peaks")
    return _draw_bands_from_peaks(strong_bool, h_peaks, v_peaks)


def G2d_spacing_outlier(orange_mask: np.ndarray) -> np.ndarray:
    """Drop peaks that break the global spacing pattern.

    A grid line should sit roughly at multiples of the median spacing.
    Text-induced peaks tend to cluster too close to existing peaks
    (each character creates a peak only a few px from its neighbours).
    We compute the median spacing of the inter-peak diffs (lower 70%),
    then drop any peak whose nearest neighbour is closer than
    0.6 * median_spacing.  This removes clustered text peaks while
    preserving evenly-spaced grid peaks.
    """
    strong_bool = orange_mask > 0
    h_peaks, v_peaks = _project_peaks(strong_bool)

    def _filter(peaks: list[int]) -> list[int]:
        if len(peaks) < 3:
            return peaks
        diffs = np.diff(peaks)
        diffs_sorted = np.sort(diffs)
        n_keep = max(1, int(0.7 * len(diffs_sorted)))
        spacing = float(np.median(diffs_sorted[:n_keep]))
        min_gap = 0.6 * spacing
        kept: list[int] = []
        for p in peaks:
            if not kept or (p - kept[-1]) >= min_gap:
                kept.append(p)
        return kept

    h_keep = _filter(h_peaks)
    v_keep = _filter(v_peaks)
    print(f"  G2d spacing-outlier: kept {len(h_keep)}h (was {len(h_peaks)}), "
          f"{len(v_keep)}v (was {len(v_peaks)})")
    return _draw_bands_from_peaks(strong_bool, h_keep, v_keep)


METHODS: list[tuple[str, Callable[[np.ndarray], np.ndarray]]] = [
    ("G2_baseline",                 G2_baseline),
    ("G2a_long_morph_prefilter",    G2a_long_morph_prefilter),
    ("G2b_continuity_post",         G2b_continuity_post),
    ("G2c_aspect_ratio_compfilt",   G2c_aspect_ratio_compfilt),
    ("G2d_spacing_outlier",         G2d_spacing_outlier),
]


# =====================================================================
# Runner
# =====================================================================

def _overlay(base, bands, orange):
    out = base.copy().astype(np.float32)
    g = np.zeros_like(out); g[..., 1] = 255
    r = np.zeros_like(out); r[..., 2] = 255
    alpha = 0.55
    mo = orange > 0
    mb = bands  > 0
    out[mo] = (1 - alpha) * out[mo] + alpha * r[mo]
    out[mb] = (1 - alpha) * out[mb] + alpha * g[mb]
    return np.clip(out, 0, 255).astype(np.uint8)


def _label(img, txt, scale=0.7):
    out = img.copy()
    band = int(38 * scale)
    cv2.rectangle(out, (0, 0), (out.shape[1], band), (0, 0, 0), -1)
    cv2.putText(out, txt, (10, band - 10), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main():
    name = "BlueColourTest01"
    p1_run = sorted((REPO / "outputs" / "runs").glob(f"*{name}*"))[-1]
    img = cv2.imread(str(p1_run / "scaled_design_area.png"))
    img = normalise_lighting(img)

    out_root = (REPO / "data" / "outputs" / "g2_text_filters" / name
                / time.strftime("%Y%m%d-%H%M%S"))
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"output: {out_root.relative_to(REPO)}\n")

    orange = build_combined_orange_mask(img)

    # Focus crop on the "Boundary Line" text area (left side of the image)
    H, W = img.shape[:2]
    text_crop = (int(H * 0.0), int(H * 0.50), int(W * 0.0), int(W * 0.20))
    full_crop = (0, H, 0, W)

    crops_text: list[np.ndarray] = []
    crops_full: list[np.ndarray] = []

    for nm, fn in METHODS:
        print(f"[{nm}]")
        t0 = time.time()
        bands = fn(orange)
        print(f"  {time.time() - t0:.2f}s")

        sub = out_root / nm
        sub.mkdir(exist_ok=True)
        full_ovr = _overlay(img, bands, orange)
        cv2.imwrite(str(sub / "overlay_full.png"), full_ovr)

        # text-area crop
        y0, y1, x0, x1 = text_crop
        text_ovr = full_ovr[y0:y1, x0:x1]
        cv2.imwrite(str(sub / "overlay_text_area.png"), text_ovr)

        crops_text.append(_label(text_ovr, nm, scale=0.7))
        crops_full.append(_label(full_ovr, nm, scale=0.9))

    # Stacked text-area comparison
    max_w = max(c.shape[1] for c in crops_text)
    padded = []
    for c in crops_text:
        if c.shape[1] < max_w:
            pad = np.full((c.shape[0], max_w - c.shape[1], 3), 255, dtype=np.uint8)
            c = np.hstack([c, pad])
        padded.append(c)
    strip_text = np.vstack(padded)
    cv2.imwrite(str(out_root / "compare_text_area_stacked.png"), strip_text)

    # 3x2 grid of full overlays
    h_c, w_c = crops_full[0].shape[:2]
    cols, rows = 3, 2
    grid = np.full((rows * h_c, cols * w_c, 3), 255, dtype=np.uint8)
    for i, c in enumerate(crops_full):
        r_, c_ = divmod(i, cols)
        grid[r_ * h_c:(r_ + 1) * h_c, c_ * w_c:(c_ + 1) * w_c] = c
    cv2.imwrite(str(out_root / "compare_full_grid.png"), grid)

    print(f"\nopen:  {out_root.relative_to(REPO)}/compare_text_area_stacked.png")
    print(f"open:  {out_root.relative_to(REPO)}/compare_full_grid.png")


if __name__ == "__main__":
    main()
