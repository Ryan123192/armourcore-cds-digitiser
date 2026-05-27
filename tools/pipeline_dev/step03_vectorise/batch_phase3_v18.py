"""Step 03 - v18: phase2 v14 + grid_strip + vectorise_v4 + path classifier.

Pipeline:
  1. rectify  ............................. (phase 1)
  2. v14 adaptive .......................... (phase 2)
  3. grid_strip (major + minor) ............ (phase 2 post)
  4. vectorise_v4 closed-loops ............. (phase 3)
  5. classify each path -> annotated PNG ... (self-diagnostic)

Compare tile per case (6 panels):
  rectified | phase2 v14 | phase2 + grid_strip
  raw closed loops | FINAL | CLASSIFIED with REAL/GRID/TEXT colours
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
from armourcore_cds.phase3.vectorise import write_svg, render_vector_overlay
from armourcore_cds.phase3.vectorise_v4 import (
    extract_vector_paths_closed_only,
)
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
EXPECTED_REAL = 11   # per user reference


# ---------------------------------------------------------------------------
# Compare-tile helpers
# ---------------------------------------------------------------------------

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
    run_dir = make_run_dir("step03_vectorise", "phase3_v18_grid_strip")
    print(f"Run dir: {run_dir.relative_to(REPO)}\n")
    template = load_template_config(TEMPLATE_ID)
    stages_dir = run_dir / "stages"
    compare_dir = run_dir / "compare"; compare_dir.mkdir(exist_ok=True)
    svg_dir = run_dir / "svg"; svg_dir.mkdir(exist_ok=True)
    reports_dir = run_dir / "reports"; reports_dir.mkdir(exist_ok=True)

    print(f"{'case':<22s}  {'route':<8s}  "
          f"{'nat':>4s}  {'br':>4s}  {'drp':>4s}  "
          f"{'TOTAL':>5s}  | "
          f"{'REAL':>5s}  {'GRID':>5s}  {'TEXT':>5s}  "
          f"{'SLIV':>5s}  {'NOISE':>5s}  | "
          f"{'verdict':<10s}")
    print("-" * 100)

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

        # Phase 2 v14
        cleaned_v14 = method_M_v14_adaptive(rect, template)
        # Phase 2 + grid strip
        cleaned_stripped = strip_all_grid_residue(cleaned_v14)
        route = "pencil" if detect_pencil_image(rect) else "pen"

        # Phase 3
        trace_mask = cleaned_to_trace_mask(cleaned_stripped)
        paths, report, cfg = extract_vector_paths_closed_only(
            trace_mask,
            design_width_mm=PAPER_W_MM, design_height_mm=PAPER_H_MM,
            route=route,
            save_stages_dir=stages_dir,
            case_stem=stem,
        )

        H, W = trace_mask.shape

        # Self-diagnostic: classify each path
        classified = classify_paths(paths, (H, W), PAPER_W_MM, PAPER_H_MM)
        counts = category_counts(classified)
        write_classification_report(classified,
                                    reports_dir / f"{stem}_classified.json")

        annotated = render_classified(paths, classified, (H, W))
        cv2.imwrite(str(run_dir / f"{stem}_annotated.png"), annotated)

        # Renders
        base = np.full((H, W, 3), 255, dtype=np.uint8)
        final = render_vector_overlay(base, paths, mask_shape=(H, W),
                                     colour_bgr=(40, 50, 220), thickness=3)
        cv2.imwrite(str(run_dir / f"{stem}_final.png"), final)
        # Filter to REAL-only for the SVG to give the user a clean export
        real_paths = [p for p, c in zip(paths, classified) if c.category == "REAL"]
        real_overlay = render_vector_overlay(
            base, real_paths, mask_shape=(H, W),
            colour_bgr=(40, 180, 40), thickness=3)
        cv2.imwrite(str(run_dir / f"{stem}_real_only.png"), real_overlay)
        write_svg(real_paths, svg_dir / f"{stem}_real_only.svg",
                 mask_shape=(H, W),
                 design_width_mm=PAPER_W_MM,
                 design_height_mm=PAPER_H_MM)
        write_svg(paths, svg_dir / f"{stem}_all.svg",
                 mask_shape=(H, W),
                 design_width_mm=PAPER_W_MM,
                 design_height_mm=PAPER_H_MM)

        loops_img = cv2.imread(str(stages_dir / f"{stem}_stage5_loops.png"))

        # Verdict
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
            (rect,             "rectified", ""),
            (cleaned_v14,      "phase2 v14 adaptive", ""),
            (cleaned_stripped, "phase2 + grid_strip", ""),
            (loops_img if loops_img is not None else base,
             "raw closed loops", ""),
            (annotated,        "CLASSIFIED",
             f"REAL={counts['REAL']} GRID={counts['GRID']} "
             f"TEXT={counts['TEXT']} SLIV={counts['SLIVER']} "
             f"NOISE={counts['NOISE']}"),
            (real_overlay,     "REAL ONLY (final export)",
             f"{len(real_paths)} paths  |  verdict: {verdict}"),
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
                   (255, 255, 255) if verdict in ("EXCELLENT", "GOOD")
                   else (200, 220, 255), 3, cv2.LINE_AA)
        cv2.putText(header, f"REAL paths: {counts['REAL']} (target {EXPECTED_REAL}, "
                   f"delta {delta_real:+d})    |    "
                   f"GRID artefacts: {counts['GRID']}    "
                   f"TEXT: {counts['TEXT']}    SLIV: {counts['SLIVER']}    "
                   f"NOISE: {counts['NOISE']}",
                   (24, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                   (180, 220, 255), 1, cv2.LINE_AA)
        compare = np.vstack([header, body])
        cv2.imwrite(str(compare_dir / f"{stem}.png"), compare)

        print(f"{stem:<22s}  {route:<8s}  "
              f"{report.stage_natural_closed_loops:>4d}  "
              f"{report.stage_arcs_bridged:>4d}  "
              f"{report.stage_arcs_dropped:>4d}  "
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
