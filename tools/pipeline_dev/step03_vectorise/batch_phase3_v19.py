"""Step 03 - v19: full pipeline with all three fixes.

Pipeline:
  1. rectify           (phase 1)
  2. v14 adaptive      (phase 2)
  3. shape_bridge      (phase 2 post: bridges per-shape gaps for img1)
  4. grid_strip        (phase 2 post: removes major/minor grid bands)
  5. vectorise route:
       - pen    -> vectorise_v4 closed-loops
       - pencil -> vectorise_pencil dilate-first
  6. classify          (REAL/GRID/TEXT/SLIV/NOISE)

6-panel compare tile per case + summary JSON.
"""
from __future__ import annotations

import json
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
from armourcore_cds.phase2.grid_strip import strip_all_grid_residue
from armourcore_cds.phase2.shape_bridge import bridge_shape_gaps
from armourcore_cds.phase3.vectorise import write_svg, render_vector_overlay
from armourcore_cds.phase3.vectorise_v4 import (
    extract_vector_paths_closed_only,
)
from armourcore_cds.phase3.vectorise_pencil import extract_vector_paths_pencil
from armourcore_cds.phase3.path_classifier import (
    classify_paths, render_classified, category_counts,
    write_classification_report,
)
from armourcore_cds.templates.registry import load_template_config
from tools.pipeline_dev.corpus import RAW_IMAGES_DIR, make_run_dir

sys.path.insert(0, str(REPO / "tools/pipeline_dev/step02_grid_removal"))
from batch_flat_methods_v14 import (   # noqa: E402
    method_M_v14_adaptive, detect_pencil_image,
)


TEMPLATE_ID = "cds_colour_test_260x350"
FLAT_CASES = [
    "BLUE_PEN_FLAT_01", "BLUE_PEN_FLAT_02",
    "BLUE_PEN_FLAT_03", "BLUE_PENCIL_FLAT_01",
]
EXPECTED_REAL = 11


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


def cleaned_to_trace_mask(cleaned_bgr, dark_thr=170, min_component=60):
    gray = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray < dark_thr).astype(np.uint8) * 255
    return _remove_small_components(mask, min_area_px=min_component)


def main():
    run_dir = make_run_dir("step03_vectorise", "phase3_v19_full")
    print(f"Run dir: {run_dir.relative_to(REPO)}\n")
    template = load_template_config(TEMPLATE_ID)
    stages_dir = run_dir / "stages"; stages_dir.mkdir(exist_ok=True)
    compare_dir = run_dir / "compare"; compare_dir.mkdir(exist_ok=True)
    svg_dir = run_dir / "svg"; svg_dir.mkdir(exist_ok=True)
    reports_dir = run_dir / "reports"; reports_dir.mkdir(exist_ok=True)

    print(f"{'case':<22s}  {'route':<8s}  "
          f"{'paths':>5s}  | "
          f"{'REAL':>5s}  {'GRID':>5s}  {'TEXT':>5s}  "
          f"{'SLIV':>5s}  {'NOISE':>5s}  | "
          f"{'verdict':<10s}")
    print("-" * 90)

    summary = []
    for stem in FLAT_CASES:
        case_path = next(RAW_IMAGES_DIR.glob(f"{stem}.*"), None)
        if case_path is None:
            print(f"{stem}: NOT FOUND"); continue
        img = cv2.imread(str(case_path))
        rect = rectify_with_markers_fast_v4(
            img, paper_w_mm=PAPER_W_MM, paper_h_mm=PAPER_H_MM,
            px_per_mm=DEFAULT_PX_PER_MM,
        ).warped

        # Phase 2 chain
        c_v14 = method_M_v14_adaptive(rect, template)
        route = "pencil" if detect_pencil_image(rect) else "pen"

        # shape_bridge REMOVED - it was filling shape interiors with solid
        # black via MORPH_CLOSE >= 51px, distorting the outlines.  Just
        # grid_strip is enough to clean residual grid/text.
        c_stripped = strip_all_grid_residue(c_v14)
        cv2.imwrite(str(stages_dir / f"{stem}_p2a_stripped.png"), c_stripped)
        c_bridged = c_stripped   # alias for the panel name

        trace_mask = cleaned_to_trace_mask(c_stripped)

        # Phase 3 route
        if route == "pencil":
            paths, pstats = extract_vector_paths_pencil(
                trace_mask,
                design_width_mm=PAPER_W_MM,
                design_height_mm=PAPER_H_MM,
            )
            loops_img_path = None   # pencil doesn't save stage5
        else:
            paths, report, cfg = extract_vector_paths_closed_only(
                trace_mask,
                design_width_mm=PAPER_W_MM, design_height_mm=PAPER_H_MM,
                route="pen",
                save_stages_dir=stages_dir,
                case_stem=stem,
            )
            loops_img_path = stages_dir / f"{stem}_stage5_loops.png"

        H, W = trace_mask.shape

        classified = classify_paths(paths, (H, W), PAPER_W_MM, PAPER_H_MM)
        counts = category_counts(classified)
        write_classification_report(classified,
                                    reports_dir / f"{stem}_classified.json")
        annotated = render_classified(paths, classified, (H, W))
        cv2.imwrite(str(run_dir / f"{stem}_annotated.png"), annotated)

        base = np.full((H, W, 3), 255, dtype=np.uint8)
        real_paths = [p for p, c in zip(paths, classified) if c.category == "REAL"]
        real_overlay = render_vector_overlay(
            base, real_paths, mask_shape=(H, W),
            colour_bgr=(40, 180, 40), thickness=3)
        cv2.imwrite(str(run_dir / f"{stem}_real_only.png"), real_overlay)
        write_svg(real_paths, svg_dir / f"{stem}.svg",
                 mask_shape=(H, W),
                 design_width_mm=PAPER_W_MM,
                 design_height_mm=PAPER_H_MM)

        loops_img = cv2.imread(str(loops_img_path)) if loops_img_path else None
        if loops_img is None:
            loops_img = trace_mask
            if len(loops_img.shape) == 2:
                loops_img = cv2.cvtColor(loops_img, cv2.COLOR_GRAY2BGR)

        delta_real = counts["REAL"] - EXPECTED_REAL
        if abs(delta_real) <= 2 and counts["GRID"] == 0:
            verdict = "EXCELLENT"
        elif abs(delta_real) <= 3:
            verdict = "GOOD"
        elif counts["GRID"] > 0:
            verdict = "GRID_LEAK"
        else:
            verdict = "POOR"

        panels = [
            (rect,            "rectified", ""),
            (c_v14,           "phase2 v14 adaptive", ""),
            (c_stripped,      "phase2 + grid_strip", ""),
            (annotated,       "CLASSIFIED",
             f"REAL={counts['REAL']} GRID={counts['GRID']} "
             f"TEXT={counts['TEXT']} SLIV={counts['SLIVER']} "
             f"NOISE={counts['NOISE']}"),
            (real_overlay,    "REAL ONLY (final export)",
             f"{len(real_paths)} paths  |  verdict: {verdict}"),
            (np.full(rect.shape, 250, dtype=np.uint8), "", ""),
        ]
        tile_w, tile_h = 920, 700
        rendered = [_panel(im, t, tile_w, tile_h, subtitle=s)
                    for (im, t, s) in panels]
        cols = 3
        rows = []
        for r in range(0, len(rendered), cols):
            row = rendered[r:r + cols]
            while len(row) < cols:
                row.append(np.full((tile_h, tile_w, 3), 250, dtype=np.uint8))
            rows.append(np.hstack(row))
        body = np.vstack(rows)
        header = np.full((96, body.shape[1], 3), 18, dtype=np.uint8)
        cv2.putText(header, f"{stem}   [route: {route}]   VERDICT: {verdict}",
                   (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                   (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(header,
                   f"REAL: {counts['REAL']}/{EXPECTED_REAL} (d {delta_real:+d})    "
                   f"GRID: {counts['GRID']}    TEXT: {counts['TEXT']}    "
                   f"SLIV: {counts['SLIVER']}    NOISE: {counts['NOISE']}",
                   (24, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                   (180, 220, 255), 1, cv2.LINE_AA)
        compare = np.vstack([header, body])
        cv2.imwrite(str(compare_dir / f"{stem}.png"), compare)

        print(f"{stem:<22s}  {route:<8s}  "
              f"{len(paths):>5d}  | "
              f"{counts['REAL']:>5d}  {counts['GRID']:>5d}  "
              f"{counts['TEXT']:>5d}  {counts['SLIVER']:>5d}  "
              f"{counts['NOISE']:>5d}  | {verdict}")

        summary.append({
            "case": stem, "route": route, "verdict": verdict,
            "counts": counts, "total_paths": len(paths),
            "real_paths": len(real_paths),
            "delta_from_expected": delta_real,
        })

    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nFolder: {run_dir}")


if __name__ == "__main__":
    main()
