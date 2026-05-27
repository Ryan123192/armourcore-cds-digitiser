"""Step 03 - v15 batch: Phase 2 (v14 adaptive) -> Phase 3 vectorise_v2.

For each flat case:
  1. rectify (Phase 1)
  2. run v14 adaptive Phase 2 -> cleaned_bgr
  3. binarise -> trace_mask
  4. run vectorise_v2.extract_vector_paths_tuned with the appropriate route
     (pen or pencil) - returns FILTERED paths + report
  5. compare tile shows: rectified | cleaned | vectorised(current) |
                        vectorised(v2 filtered) | filter report

Side-by-side lets you SEE what got dropped.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

import cv2
import numpy as np

from armourcore_cds.phase1.marker_rectify_fast_v4 import (
    rectify_with_markers_fast_v4,
    PAPER_W_MM, PAPER_H_MM, DEFAULT_PX_PER_MM,
)
from armourcore_cds.phase2.trace_isolation import _remove_small_components
from armourcore_cds.phase3.vectorise import (
    extract_vector_paths, write_svg, render_vector_overlay,
)
from armourcore_cds.phase3.vectorise_v2 import (
    extract_vector_paths_tuned, ROUTE_PRESETS,
)
from armourcore_cds.templates.registry import load_template_config
from tools.pipeline_dev.corpus import RAW_IMAGES_DIR, make_run_dir

# Reuse v14 phase2
sys.path.insert(0, str(REPO / "tools/pipeline_dev/step02_grid_removal"))
from batch_flat_methods_v14 import (   # noqa: E402
    method_M_v14_adaptive, detect_pencil_image,
)


TEMPLATE_ID = "cds_colour_test_260x350"
FLAT_CASES = [
    "BLUE_PEN_FLAT_01",
    "BLUE_PEN_FLAT_02",
    "BLUE_PEN_FLAT_03",
    "BLUE_PENCIL_FLAT_01",
]


# ---------------------------------------------------------------------------
# Compare tile helpers
# ---------------------------------------------------------------------------

def _fit(img, max_w, max_h):
    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h, 1.0)
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))),
                      interpolation=cv2.INTER_AREA)


def _panel(img, title, tw, th, subtitle: str = ""):
    canvas = np.full((th, tw, 3), 250, dtype=np.uint8)
    head = 56 if subtitle else 42
    cv2.rectangle(canvas, (0, 0), (tw, head), (35, 35, 35), -1)
    cv2.putText(canvas, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.78,
                (255, 255, 255), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(canvas, subtitle, (12, 49),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (180, 200, 240), 1, cv2.LINE_AA)
    body_h = th - head - 4
    fit = _fit(img, tw - 6, body_h)
    fh, fw = fit.shape[:2]
    y0 = head + (body_h - fh) // 2
    x0 = (tw - fw) // 2
    canvas[y0:y0 + fh, x0:x0 + fw] = fit
    return canvas


def make_tile(case_stem, rectified, cleaned, overlay_v1, overlay_v2,
              route, n_v1, n_v2, report,
              tile_w=950, tile_h=720):
    panels = [
        _panel(rectified, f"rectified  [route: {route}]", tile_w, tile_h),
        _panel(cleaned,   "phase2 cleaned (v14 adaptive)", tile_w, tile_h),
        _panel(overlay_v1, "phase3 v1  (raw)",
              tile_w, tile_h, subtitle=f"{n_v1} paths"),
        _panel(overlay_v2, "phase3 v2  (filtered)",
              tile_w, tile_h,
              subtitle=f"{n_v2} paths  |  {report}"),
    ]
    cols = 2
    row_imgs = []
    for r in range(0, len(panels), cols):
        row = panels[r:r + cols]
        while len(row) < cols:
            row.append(np.full((tile_h, tile_w, 3), 250, dtype=np.uint8))
        row_imgs.append(np.hstack(row))
    body = np.vstack(row_imgs)
    header = np.full((64, body.shape[1], 3), 18, dtype=np.uint8)
    cv2.putText(header, case_stem, (24, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                (255, 255, 255), 3, cv2.LINE_AA)
    return np.vstack([header, body])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def cleaned_to_trace_mask(cleaned_bgr: np.ndarray,
                         dark_thr: int = 170,
                         min_component: int = 60,
                         pre_dilate_px: int = 0) -> np.ndarray:
    """Binarise cleaned phase2 output.  Optional pre-dilation thickens
    thin strokes so neighbouring stroke fragments merge into a single
    closed outline before vectorisation - critical for pencil."""
    gray = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray < dark_thr).astype(np.uint8) * 255
    mask = _remove_small_components(mask, min_area_px=min_component)
    if pre_dilate_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (pre_dilate_px * 2 + 1, pre_dilate_px * 2 + 1))
        mask = cv2.dilate(mask, k, iterations=1)
    return mask


def main():
    run_dir = make_run_dir("step03_vectorise", "phase3_v15")
    print(f"Run dir: {run_dir.relative_to(REPO)}\n")

    template = load_template_config(TEMPLATE_ID)
    tile_dir = run_dir / "compare"
    tile_dir.mkdir(exist_ok=True)
    svg_dir = run_dir / "svg"
    svg_dir.mkdir(exist_ok=True)

    header = (f"{'case':<22s}  {'route':<8s}  "
              f"{'v1_paths':>9s}  {'v2_paths':>9s}  "
              f"{'dropped (small/short/slither/hollow)':<42s}  "
              f"{'gap_close':>9s}")
    print(header)
    print("-" * len(header))

    for stem in FLAT_CASES:
        case_path = next(RAW_IMAGES_DIR.glob(f"{stem}.*"), None)
        if case_path is None:
            print(f"{stem}: NOT FOUND"); continue
        img = cv2.imread(str(case_path))
        rect = rectify_with_markers_fast_v4(
            img, paper_w_mm=PAPER_W_MM, paper_h_mm=PAPER_H_MM,
            px_per_mm=DEFAULT_PX_PER_MM,
        ).warped

        # Phase 2 via v14 adaptive
        cleaned = method_M_v14_adaptive(rect, template)
        cv2.imwrite(str(run_dir / f"{stem}_cleaned.png"), cleaned)

        # Determine route for Phase 3 tuning
        route = "pencil" if detect_pencil_image(rect) else "pen"

        # Build trace mask from cleaned.  Pencil gets pre-dilated so
        # the thin sparse strokes merge into a thick outline.
        pre_dilate = 5 if route == "pencil" else 0
        trace_mask = cleaned_to_trace_mask(cleaned, pre_dilate_px=pre_dilate)

        # Phase 3 v1 (current, no filter)
        paths_v1 = extract_vector_paths(
            trace_mask, **ROUTE_PRESETS[route]
        )

        # Phase 3 v2 (filtered)
        paths_v2, report, params = extract_vector_paths_tuned(
            trace_mask,
            design_width_mm=PAPER_W_MM, design_height_mm=PAPER_H_MM,
            route=route,
        )

        # Render overlays (white background)
        H, W = trace_mask.shape
        base = np.full((H, W, 3), 255, dtype=np.uint8)
        overlay_v1 = render_vector_overlay(
            base, paths_v1, mask_shape=(H, W),
            colour_bgr=(40, 50, 220), thickness=3,
        )
        overlay_v2 = render_vector_overlay(
            base, paths_v2, mask_shape=(H, W),
            colour_bgr=(40, 160, 40), thickness=3,
        )
        cv2.imwrite(str(run_dir / f"{stem}_overlay_v1.png"), overlay_v1)
        cv2.imwrite(str(run_dir / f"{stem}_overlay_v2.png"), overlay_v2)

        # SVG (filtered)
        write_svg(paths_v2, svg_dir / f"{stem}.svg",
                 mask_shape=(H, W),
                 design_width_mm=PAPER_W_MM,
                 design_height_mm=PAPER_H_MM)

        tile = make_tile(stem, rect, cleaned, overlay_v1, overlay_v2,
                        route, len(paths_v1), len(paths_v2), str(report))
        cv2.imwrite(str(tile_dir / f"{stem}.png"), tile)

        drop_summary = (f"{report.dropped_small_area}/"
                       f"{report.dropped_short_diag}/"
                       f"{report.dropped_slither}/"
                       f"{report.dropped_hollow}")
        print(f"{stem:<22s}  {route:<8s}  "
              f"{len(paths_v1):>9d}  {len(paths_v2):>9d}  "
              f"{drop_summary:<42s}  "
              f"{params['gap_close_px']:>9d}")

    print(f"\nFolder: {run_dir}")
    print(f"Compare tiles: {tile_dir}")
    print(f"SVGs: {svg_dir}")


if __name__ == "__main__":
    main()
