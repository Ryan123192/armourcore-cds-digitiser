"""Step 03 - v20: per-case adaptive pipeline.

PROTECTED PATH (must stay perfect for img3)
==========================================
  rectify -> v14 adaptive -> grid_strip -> vectorise_v4 (RETR_CCOMP)
  No extra processing.  This is the production snapshot.

ADAPTIVE EXTENSIONS (only fire when characteristic detected)
============================================================
* If `detect_dark_border(rect)` -> route "pen_darkborder":
    rectify -> v14 -> strip_dark_border -> grid_strip_aggressive
            -> grid_strip -> vectorise_v4
    (img1 fix - kill the border + extra-aggressive major grid removal)

* If `detect_pencil(rect)` -> route "pencil":
    rectify -> v14 -> enhance_pencil (CLAHE + paper_divide)
            -> grid_strip -> vectorise_pencil
    (img4 fix - boost faint pencil before binarisation)

* Otherwise -> route "pen_standard":
    rectify -> v14 -> grid_strip -> vectorise_v4   (img2 + img3)

Self-diagnostic remains via path_classifier.
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
from armourcore_cds.phase2.grid_strip_aggressive import (
    strip_major_grid_aggressive, strip_dark_border,
    strip_major_grid_unconditional, strip_minor_grid_unconditional,
    strip_grid_intersection_safe, strip_grid_by_component,
)
from armourcore_cds.phase2.line_strip import strip_dashed_boundary_lines
from armourcore_cds.phase2.pencil_enhance import (
    enhance_pencil, enhance_pencil_strong,
)
from armourcore_cds.phase2.pencil_enhance_v2 import sauvola_adaptive
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
    method_M_v14_adaptive, detect_pencil_image, detect_shadow_zone,
)


TEMPLATE_ID = "cds_colour_test_260x350"
FLAT_CASES = [
    "BLUE_PEN_FLAT_01", "BLUE_PEN_FLAT_02",
    "BLUE_PEN_FLAT_03", "BLUE_PENCIL_FLAT_01",
]
EXPECTED_REAL = 11


def detect_dark_border(rect_bgr) -> bool:
    """True if rectified image has a dim edge zone (p10 of any 150px
    edge strip < 165 gray).  This catches img1 (camera framing offset
    -> dim border ~140) while ignoring img2/img3 (edges p10 > 165)."""
    gray = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2GRAY)
    strip = 150
    edges = [gray[:, :strip], gray[:, -strip:],
             gray[:strip, :], gray[-strip:, :]]
    for e in edges:
        if float(np.percentile(e, 10)) < 165.0:
            return True
    return False


def detect_giant_component_after_v14(cleaned_bgr,
                                    threshold_frac: float = 0.25) -> bool:
    """True if v14-cleaned image has any single connected component whose
    bbox covers > threshold_frac of the image.  This is the actual
    failure signature for img1 (border + grid-web blob)."""
    gray = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
    ink = (gray < 170).astype(np.uint8) * 255
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    H, W = ink.shape
    image_area = H * W
    for cid in range(1, n):
        bw = stats[cid, cv2.CC_STAT_WIDTH]
        bh = stats[cid, cv2.CC_STAT_HEIGHT]
        if bw * bh > image_area * threshold_frac:
            return True
    return False


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


# ---------------------------------------------------------------------------
# Per-route processing functions
# ---------------------------------------------------------------------------

def process_pen_standard(rect, template, stages_dir, stem):
    """The PROTECTED path that produces img3 = 11/11.  Do not modify."""
    cleaned_v14 = method_M_v14_adaptive(rect, template)
    cleaned_p2  = strip_all_grid_residue(cleaned_v14)
    cv2.imwrite(str(stages_dir / f"{stem}_p2_final.png"), cleaned_p2)
    trace = cleaned_to_trace_mask(cleaned_p2)
    paths, report, _ = extract_vector_paths_closed_only(
        trace, design_width_mm=PAPER_W_MM, design_height_mm=PAPER_H_MM,
        route="pen", save_stages_dir=stages_dir, case_stem=stem,
    )
    return cleaned_v14, cleaned_p2, paths


def process_pen_darkborder(rect, template, stages_dir, stem):
    """img1-style: dark camera border + residual major grid.

    1. v14 adaptive cleaning (same as standard)
    2. strip_dark_border       <- new: wipes the border-zone non-ink pixels
    3. grid_strip_aggressive   <- new: wider bands + lower ink-protect
    4. strip_all_grid_residue  <- same as standard final pass
    5. vectorise_v4            <- same RETR_CCOMP hole-preference
    """
    cleaned_v14 = method_M_v14_adaptive(rect, template)
    cv2.imwrite(str(stages_dir / f"{stem}_p2a_v14.png"), cleaned_v14)
    c_no_border = strip_dark_border(cleaned_v14, border_zone_px=250,
                                    ink_threshold_gray=100)
    cv2.imwrite(str(stages_dir / f"{stem}_p2b_no_border.png"), c_no_border)
    c_strip_agg = strip_major_grid_aggressive(c_no_border,
                                              band_half_px=8,
                                              ink_protect_max_gray=70,
                                              border_zone_px=250,
                                              border_ink_protect=0)
    cv2.imwrite(str(stages_dir / f"{stem}_p2c_strip_agg.png"), c_strip_agg)
    c_grid = strip_all_grid_residue(c_strip_agg)
    cv2.imwrite(str(stages_dir / f"{stem}_p2d_grid.png"), c_grid)
    c_final = strip_dashed_boundary_lines(c_grid)
    cv2.imwrite(str(stages_dir / f"{stem}_p2e_final.png"), c_final)
    trace = cleaned_to_trace_mask(c_final)
    paths, report, _ = extract_vector_paths_closed_only(
        trace, design_width_mm=PAPER_W_MM, design_height_mm=PAPER_H_MM,
        route="pen", save_stages_dir=stages_dir, case_stem=stem,
    )
    return cleaned_v14, c_final, paths


def process_pencil(rect, template, stages_dir, stem):
    """Pencil pipeline (pencil_lab winner):
       v14 -> Sauvola adaptive binarise -> grid_strip
            -> extract_vector_paths_pencil(dilate=17, close=11)
    Sauvola is dramatically better at darkening pencil vs paper-divide
    because it locally adapts to lighting variations.  d17/c11 + the
    min_component=40 from lab tuned via sweep to give 12/11 EXCELLENT.
    """
    cleaned_v14 = method_M_v14_adaptive(rect, template)
    cv2.imwrite(str(stages_dir / f"{stem}_p2a_v14.png"), cleaned_v14)
    enhanced = sauvola_adaptive(cleaned_v14, window_size=25, k=0.15)
    cv2.imwrite(str(stages_dir / f"{stem}_p2b_sauvola.png"), enhanced)
    stripped = strip_all_grid_residue(enhanced)
    cv2.imwrite(str(stages_dir / f"{stem}_p2c_stripped.png"), stripped)
    # IMPORTANT: min_component=40 matches the pencil_lab sweep that
    # produced 12/11 EXCELLENT.  Larger min_component gave 16 (drops
    # small connector strokes between sub-shape segments, causing
    # what should be one shape to be counted as multiple).
    trace = cleaned_to_trace_mask(stripped, min_component=40)
    paths, _ = extract_vector_paths_pencil(
        trace,
        design_width_mm=PAPER_W_MM, design_height_mm=PAPER_H_MM,
        dilate_px=17, close_after_dilate_px=11,
    )
    return cleaned_v14, stripped, paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    run_dir = make_run_dir("step03_vectorise", "phase3_v20_adaptive")
    print(f"Run dir: {run_dir.relative_to(REPO)}\n")
    template = load_template_config(TEMPLATE_ID)
    stages_dir = run_dir / "stages"; stages_dir.mkdir(exist_ok=True)
    compare_dir = run_dir / "compare"; compare_dir.mkdir(exist_ok=True)
    svg_dir = run_dir / "svg"; svg_dir.mkdir(exist_ok=True)
    reports_dir = run_dir / "reports"; reports_dir.mkdir(exist_ok=True)

    print(f"{'case':<22s}  {'route':<18s}  "
          f"{'paths':>5s}  | "
          f"{'REAL':>5s}  {'GRID':>5s}  {'TEXT':>5s}  "
          f"{'SLIV':>5s}  {'NOISE':>5s}  | "
          f"{'verdict':<10s}")
    print("-" * 105)

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

        # Route decision: PROTECT img3-style cases first; only escalate
        # to darkborder path if the actual failure signature is present
        # (giant component covering > 25% of image after v14 cleaning).
        is_pencil = detect_pencil_image(rect)
        if is_pencil:
            route = "pencil"
            cleaned_v14, cleaned_final, paths = process_pencil(
                rect, template, stages_dir, stem)
        else:
            # Run v14 first; check for the giant-component failure signature
            cleaned_probe = method_M_v14_adaptive(rect, template)
            if detect_giant_component_after_v14(cleaned_probe):
                route = "pen_darkborder"
                cleaned_v14, cleaned_final, paths = process_pen_darkborder(
                    rect, template, stages_dir, stem)
            else:
                route = "pen_standard"
                cleaned_v14, cleaned_final, paths = process_pen_standard(
                    rect, template, stages_dir, stem)

        H, W = cleaned_final.shape[:2]

        # Classify
        classified = classify_paths(paths, (H, W), PAPER_W_MM, PAPER_H_MM)
        counts = category_counts(classified)
        write_classification_report(
            classified, reports_dir / f"{stem}_classified.json")
        annotated = render_classified(paths, classified, (H, W))
        cv2.imwrite(str(run_dir / f"{stem}_annotated.png"), annotated)

        # Renders
        base = np.full((H, W, 3), 255, dtype=np.uint8)
        real_paths = [p for p, c in zip(paths, classified)
                     if c.category == "REAL"]
        real_overlay = render_vector_overlay(
            base, real_paths, mask_shape=(H, W),
            colour_bgr=(40, 180, 40), thickness=3)
        cv2.imwrite(str(run_dir / f"{stem}_real_only.png"), real_overlay)
        write_svg(real_paths, svg_dir / f"{stem}.svg",
                 mask_shape=(H, W),
                 design_width_mm=PAPER_W_MM,
                 design_height_mm=PAPER_H_MM)

        # Verdict
        delta_real = counts["REAL"] - EXPECTED_REAL
        if abs(delta_real) <= 2 and counts["GRID"] == 0:
            verdict = "EXCELLENT"
        elif abs(delta_real) <= 3 and counts["GRID"] <= 1:
            verdict = "GOOD"
        elif counts["GRID"] > 0:
            verdict = "GRID_LEAK"
        else:
            verdict = "POOR"

        panels = [
            (rect,           "rectified", ""),
            (cleaned_v14,    "phase2 v14 adaptive", ""),
            (cleaned_final,  f"phase2 FINAL ({route})", ""),
            (annotated,      "CLASSIFIED",
             f"REAL={counts['REAL']} GRID={counts['GRID']} "
             f"TEXT={counts['TEXT']} SLIV={counts['SLIVER']} "
             f"NOISE={counts['NOISE']}"),
            (real_overlay,   "REAL ONLY (final export)",
             f"{len(real_paths)} paths  |  verdict: {verdict}"),
            (np.full(cleaned_final.shape, 250, dtype=np.uint8), "", ""),
        ]
        tw, th = 920, 700
        rendered = [_panel(im, t, tw, th, subtitle=s)
                    for (im, t, s) in panels]
        cols = 3
        rows = []
        for r in range(0, len(rendered), cols):
            row = rendered[r:r + cols]
            while len(row) < cols:
                row.append(np.full((th, tw, 3), 250, dtype=np.uint8))
            rows.append(np.hstack(row))
        body = np.vstack(rows)
        header = np.full((96, body.shape[1], 3), 18, dtype=np.uint8)
        cv2.putText(header,
                   f"{stem}   [route: {route}]   VERDICT: {verdict}",
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

        print(f"{stem:<22s}  {route:<18s}  "
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
