"""Step 03 - v17: vectorise_v4 (closed-loops only) + per-stage logging.

For each case, saves:
  - <stem>_stage1_input.png      (binarised cleaned mask)
  - <stem>_stage2_stray_removed.png
  - <stem>_stage3_closed.png
  - <stem>_stage4_skeleton.png
  - <stem>_stage5_loops.png      (raw extracted loops, one colour each)
  - <stem>_report.json           (per-path properties + drop reasons)
  - <stem>_final_overlay.png     (filtered vectors on white)
  - compare/<stem>.png           (rectified | cleaned | skel | loops | final)
  - svg/<stem>.svg
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
from armourcore_cds.phase3.vectorise_v4 import (
    extract_vector_paths_closed_only,
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

EXPECTED_PATHS = {
    "BLUE_PEN_FLAT_01":    11,   # user reference shows 11
    "BLUE_PEN_FLAT_02":    11,
    "BLUE_PEN_FLAT_03":    11,
    "BLUE_PENCIL_FLAT_01": 11,
}


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
        cv2.putText(canvas, subtitle, (12, 49), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (180, 200, 240), 1, cv2.LINE_AA)
    body_h = th - head - 4
    fit = _fit(img, tw - 6, body_h)
    fh, fw = fit.shape[:2]
    canvas[head + (body_h - fh) // 2:head + (body_h - fh) // 2 + fh,
           (tw - fw) // 2:(tw - fw) // 2 + fw] = fit
    return canvas


def make_tile(case_stem, route, score_line, panels_titles, tile_w=920, tile_h=700):
    panels = [_panel(img, t, tile_w, tile_h, subtitle=sub)
              for img, t, sub in panels_titles]
    cols = 3
    rows = (len(panels) + cols - 1) // cols
    row_imgs = []
    for r in range(rows):
        row = panels[r * cols:(r + 1) * cols]
        while len(row) < cols:
            row.append(np.full((tile_h, tile_w, 3), 250, dtype=np.uint8))
        row_imgs.append(np.hstack(row))
    body = np.vstack(row_imgs)

    header = np.full((84, body.shape[1], 3), 18, dtype=np.uint8)
    cv2.putText(header, f"{case_stem}   [route: {route}]",
                (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(header, score_line, (24, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (160, 220, 255), 1, cv2.LINE_AA)
    return np.vstack([header, body])


def cleaned_to_trace_mask(cleaned_bgr, dark_thr=170, min_component=60):
    gray = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray < dark_thr).astype(np.uint8) * 255
    return _remove_small_components(mask, min_area_px=min_component)


def render_loops(loops_or_paths, H, W, colour, thickness=3):
    base = np.full((H, W, 3), 255, dtype=np.uint8)
    return render_vector_overlay(base, loops_or_paths, mask_shape=(H, W),
                                colour_bgr=colour, thickness=thickness)


def main():
    run_dir = make_run_dir("step03_vectorise", "phase3_v17_closed_only")
    print(f"Run dir: {run_dir.relative_to(REPO)}\n")
    template = load_template_config(TEMPLATE_ID)

    stages_dir = run_dir / "stages"
    compare_dir = run_dir / "compare"; compare_dir.mkdir(exist_ok=True)
    svg_dir = run_dir / "svg"; svg_dir.mkdir(exist_ok=True)
    final_dir = run_dir / "final"; final_dir.mkdir(exist_ok=True)

    print(f"{'case':<22s}  {'route':<8s}  "
          f"{'comps':>6s}  {'nat':>4s}  {'br':>4s}  "
          f"{'drop':>5s}  {'kept':>5s}  {'expected':>8s}  {'delta':>6s}")
    print("-" * 80)

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
        trace_mask = cleaned_to_trace_mask(cleaned)

        paths, report, cfg = extract_vector_paths_closed_only(
            trace_mask,
            design_width_mm=PAPER_W_MM, design_height_mm=PAPER_H_MM,
            route=route,
            save_stages_dir=stages_dir,
            case_stem=stem,
        )

        H, W = trace_mask.shape
        final_overlay = render_loops(paths, H, W, colour=(40, 50, 220))
        cv2.imwrite(str(final_dir / f"{stem}.png"), final_overlay)
        write_svg(paths, svg_dir / f"{stem}.svg",
                 mask_shape=(H, W),
                 design_width_mm=PAPER_W_MM, design_height_mm=PAPER_H_MM)

        # Read intermediate stage images for compare
        skel_img = cv2.imread(str(stages_dir / f"{stem}_stage4_skeleton.png"))
        if skel_img is None:
            skel_img = np.full((H, W, 3), 255, dtype=np.uint8)
        loops_img = cv2.imread(str(stages_dir / f"{stem}_stage5_loops.png"))
        if loops_img is None:
            loops_img = np.full((H, W, 3), 255, dtype=np.uint8)

        expected = EXPECTED_PATHS.get(stem, 11)
        delta = len(paths) - expected
        score_line = (f"skel components: {report.stage_skel_components}  |  "
                     f"natural closed: {report.stage_natural_closed_loops}  "
                     f"bridged: {report.stage_arcs_bridged}  "
                     f"dropped: {report.stage_arcs_dropped}  |  "
                     f"FINAL: {len(paths)} (expected ~{expected}, delta {delta:+d})")

        panels = [
            (rect,          "rectified", ""),
            (cleaned,       "phase2 cleaned", ""),
            (skel_img,      "skeleton (after prune)",
             f"{report.stage_skel_pixels} px"),
            (loops_img,     "raw closed loops",
             f"{report.stage_natural_closed_loops} natural + "
             f"{report.stage_arcs_bridged} bridged"),
            (final_overlay, "FINAL (filtered)",
             f"{len(paths)} paths, expected {expected}"),
        ]
        tile = make_tile(stem, route, score_line, panels)
        cv2.imwrite(str(compare_dir / f"{stem}.png"), tile)

        print(f"{stem:<22s}  {route:<8s}  "
              f"{report.stage_skel_components:>6d}  "
              f"{report.stage_natural_closed_loops:>4d}  "
              f"{report.stage_arcs_bridged:>4d}  "
              f"{report.stage_arcs_dropped:>5d}  "
              f"{len(paths):>5d}  {expected:>8d}  {delta:>+6d}")

    print(f"\nFolder: {run_dir}")
    print(f"Stages: {stages_dir}")
    print(f"Compare: {compare_dir}")


if __name__ == "__main__":
    main()
