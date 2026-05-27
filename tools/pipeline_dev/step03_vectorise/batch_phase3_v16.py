"""Step 03 - v16 batch: skeleton-based (v3) vs contour-based (v2) side-by-side.

5-panel compare tile per case:
    rectified | cleaned | v2 contour | v3 skeleton | v3 + filter
"""
from __future__ import annotations

import sys
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
from armourcore_cds.phase3.vectorise import write_svg, render_vector_overlay
from armourcore_cds.phase3.vectorise_v2 import (
    extract_vector_paths_tuned, filter_paths, ROUTE_FILTER_PRESETS,
)
from armourcore_cds.phase3.vectorise_v3 import (
    extract_vector_paths_skeleton,
)
from armourcore_cds.templates.registry import load_template_config
from tools.pipeline_dev.corpus import RAW_IMAGES_DIR, make_run_dir

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


def _fit(img, max_w, max_h):
    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h, 1.0)
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))),
                      interpolation=cv2.INTER_AREA)


def _panel(img, title, tw, th, subtitle=""):
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
    canvas[head + (body_h - fh) // 2:head + (body_h - fh) // 2 + fh,
           (tw - fw) // 2:(tw - fw) // 2 + fw] = fit
    return canvas


def make_tile(case_stem, route, panels_with_titles, tile_w=950, tile_h=720):
    panels = [_panel(img, title, tile_w, tile_h, subtitle=sub)
              for img, title, sub in panels_with_titles]
    cols = 3
    row_imgs = []
    for r in range(0, len(panels), cols):
        row = panels[r:r + cols]
        while len(row) < cols:
            row.append(np.full((tile_h, tile_w, 3), 250, dtype=np.uint8))
        row_imgs.append(np.hstack(row))
    body = np.vstack(row_imgs)
    header = np.full((64, body.shape[1], 3), 18, dtype=np.uint8)
    cv2.putText(header, f"{case_stem}   [route: {route}]",
                (24, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                (255, 255, 255), 3, cv2.LINE_AA)
    return np.vstack([header, body])


def cleaned_to_trace_mask(cleaned_bgr, dark_thr=170, min_component=60):
    gray = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray < dark_thr).astype(np.uint8) * 255
    return _remove_small_components(mask, min_area_px=min_component)


def total_nodes(paths):
    return sum(len(p.points) for p in paths)


def main():
    run_dir = make_run_dir("step03_vectorise", "phase3_v16_skeleton")
    print(f"Run dir: {run_dir.relative_to(REPO)}\n")
    template = load_template_config(TEMPLATE_ID)
    tile_dir = run_dir / "compare"
    tile_dir.mkdir(exist_ok=True)
    svg_dir = run_dir / "svg"
    svg_dir.mkdir(exist_ok=True)

    print(f"{'case':<22s}  {'route':<8s}  "
          f"{'v2_paths':>9s}  {'v2_nodes':>9s}  "
          f"{'v3_paths':>9s}  {'v3_nodes':>9s}  "
          f"{'v3f_paths':>10s}")
    print("-" * 90)

    for stem in FLAT_CASES:
        case_path = next(RAW_IMAGES_DIR.glob(f"{stem}.*"), None)
        if case_path is None:
            print(f"{stem}: NOT FOUND"); continue
        img = cv2.imread(str(case_path))
        rect = rectify_with_markers_fast_v4(
            img, paper_w_mm=PAPER_W_MM, paper_h_mm=PAPER_H_MM,
            px_per_mm=DEFAULT_PX_PER_MM,
        ).warped

        cleaned = method_M_v14_adaptive(rect, template)
        route = "pencil" if detect_pencil_image(rect) else "pen"

        # Trace mask
        trace_mask = cleaned_to_trace_mask(cleaned)

        # ---- v2 contour-based ----
        paths_v2, report_v2, _ = extract_vector_paths_tuned(
            trace_mask, design_width_mm=PAPER_W_MM,
            design_height_mm=PAPER_H_MM, route=route,
        )

        # ---- v3 skeleton-based (raw) ----
        paths_v3_raw, _ = extract_vector_paths_skeleton(trace_mask, route=route)
        # ---- v3 skeleton-based + filter ----
        H, W = trace_mask.shape
        filt = ROUTE_FILTER_PRESETS.get(route).copy()
        # Skeleton paths have nominal small areas for open arcs -> use bbox
        # primarily.  Loosen area floor.
        filt["min_area_mm2"] = filt["min_area_mm2"] * 0.3
        paths_v3, report_v3 = filter_paths(
            paths_v3_raw, (H, W), PAPER_W_MM, PAPER_H_MM, **filt)

        # Render overlays (white background)
        base = np.full((H, W, 3), 255, dtype=np.uint8)
        ov_v2 = render_vector_overlay(
            base, paths_v2, (H, W), colour_bgr=(40, 50, 220), thickness=3)
        ov_v3_raw = render_vector_overlay(
            base, paths_v3_raw, (H, W), colour_bgr=(40, 160, 40), thickness=3)
        ov_v3 = render_vector_overlay(
            base, paths_v3, (H, W), colour_bgr=(180, 60, 200), thickness=3)

        cv2.imwrite(str(run_dir / f"{stem}_v2.png"), ov_v2)
        cv2.imwrite(str(run_dir / f"{stem}_v3_raw.png"), ov_v3_raw)
        cv2.imwrite(str(run_dir / f"{stem}_v3.png"), ov_v3)
        write_svg(paths_v3, svg_dir / f"{stem}_v3.svg",
                 mask_shape=(H, W),
                 design_width_mm=PAPER_W_MM, design_height_mm=PAPER_H_MM)

        # 5-panel tile
        n_v2 = len(paths_v2)
        n_v3r = len(paths_v3_raw)
        n_v3 = len(paths_v3)
        panels = [
            (rect, "rectified", ""),
            (cleaned, "phase2 cleaned (v14)", ""),
            (ov_v2, "v2 contour (filtered)",
             f"{n_v2} paths, {total_nodes(paths_v2)} nodes"),
            (ov_v3_raw, "v3 skeleton (raw)",
             f"{n_v3r} paths, {total_nodes(paths_v3_raw)} nodes"),
            (ov_v3, "v3 skeleton (filtered)",
             f"{n_v3} paths, {total_nodes(paths_v3)} nodes  |  {report_v3}"),
        ]
        tile = make_tile(stem, route, panels)
        cv2.imwrite(str(tile_dir / f"{stem}.png"), tile)

        print(f"{stem:<22s}  {route:<8s}  "
              f"{n_v2:>9d}  {total_nodes(paths_v2):>9d}  "
              f"{n_v3r:>9d}  {total_nodes(paths_v3_raw):>9d}  "
              f"{n_v3:>10d}")

    print(f"\nFolder: {run_dir}")


if __name__ == "__main__":
    main()
