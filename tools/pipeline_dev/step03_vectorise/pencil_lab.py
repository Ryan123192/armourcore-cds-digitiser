"""Pencil focus lab - sweep enhancement methods + gap-fill params
on BLUE_PENCIL_FLAT_01.

For each (enhance_method, dilate_px, close_px) combo:
  1. enhance the v14-cleaned image
  2. binarise (gray < 170)
  3. dilate + close to merge sparse pencil dots into shape blobs
  4. extract external contours
  5. filter by area + compactness
  6. classify

Save a giant grid: rows = enhance methods, cols = (dilate, close) tunings.
Each cell shows the final REAL-vector overlay + path count.
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
from armourcore_cds.phase2.grid_strip import strip_all_grid_residue
from armourcore_cds.phase2 import pencil_enhance, pencil_enhance_v2
from armourcore_cds.phase3.vectorise import render_vector_overlay
from armourcore_cds.phase3.vectorise_pencil import extract_vector_paths_pencil
from armourcore_cds.phase3.path_classifier import (
    classify_paths, category_counts,
)
from armourcore_cds.templates.registry import load_template_config
from tools.pipeline_dev.corpus import RAW_IMAGES_DIR, make_run_dir

sys.path.insert(0, str(REPO / "tools/pipeline_dev/step02_grid_removal"))
from batch_flat_methods_v14 import method_M_v14_adaptive   # noqa


STEM = "BLUE_PENCIL_FLAT_01"
EXPECTED = 11


# ---------------------------------------------------------------------------
# Methods sweep
# ---------------------------------------------------------------------------

ENHANCE_METHODS = [
    ("baseline_v1",         lambda x: pencil_enhance.enhance_pencil(x)),
    ("clahe_only",          lambda x: pencil_enhance.clahe_boost(x, 3.5, 16)),
    ("paper_divide",        lambda x: pencil_enhance.paper_divide(x)),
    ("sauvola",             lambda x: pencil_enhance_v2.sauvola_adaptive(x, 25, 0.15)),
    ("niblack",             lambda x: pencil_enhance_v2.niblack_adaptive(x, 25, -0.2)),
    ("black_hat",           lambda x: pencil_enhance_v2.black_hat_tophat(x, 25)),
    ("multi_scale_tophat",  lambda x: pencil_enhance_v2.multi_scale_tophat(x)),
    ("frangi",              lambda x: pencil_enhance_v2.frangi_ridges(x)),
    ("gamma_clahe",         lambda x: pencil_enhance_v2.gamma_then_clahe(x, 1.6)),
    ("frangi_sauvola",      lambda x: pencil_enhance_v2.combined_frangi_sauvola(x)),
]

GAP_FILL_PRESETS = [
    ("d09_c07", 9, 7),
    ("d13_c09", 13, 9),
    ("d17_c11", 17, 11),
    ("d21_c13", 21, 13),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cleaned_to_trace(cleaned_bgr, thresh=170, min_comp=40):
    gray = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
    m = (gray < thresh).astype(np.uint8) * 255
    return _remove_small_components(m, min_area_px=min_comp)


def _fit(img, max_w, max_h):
    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h, 1.0)
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))),
                      interpolation=cv2.INTER_AREA)


def _panel(img, title, tw, th, subtitle=""):
    canvas = np.full((th, tw, 3), 250, dtype=np.uint8)
    head = 56 if subtitle else 36
    cv2.rectangle(canvas, (0, 0), (tw, head), (35, 35, 35), -1)
    cv2.putText(canvas, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(canvas, subtitle, (8, 46), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (180, 210, 255), 1, cv2.LINE_AA)
    body_h = th - head - 4
    fit = _fit(img, tw - 6, body_h)
    fh, fw = fit.shape[:2]
    canvas[head + (body_h - fh) // 2:head + (body_h - fh) // 2 + fh,
           (tw - fw) // 2:(tw - fw) // 2 + fw] = fit
    return canvas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    run_dir = make_run_dir("step03_vectorise", "PENCIL_LAB")
    print(f"Run dir: {run_dir.relative_to(REPO)}")
    template = load_template_config("cds_colour_test_260x350")

    img = cv2.imread(str(next(RAW_IMAGES_DIR.glob(f"{STEM}.*"))))
    rect = rectify_with_markers_fast_v4(
        img, paper_w_mm=PAPER_W_MM, paper_h_mm=PAPER_H_MM,
        px_per_mm=DEFAULT_PX_PER_MM,
    ).warped
    cv2.imwrite(str(run_dir / "00_rectified.png"), rect)
    cleaned_v14 = method_M_v14_adaptive(rect, template)
    cv2.imwrite(str(run_dir / "01_v14_adaptive.png"), cleaned_v14)

    methods_dir = run_dir / "methods"
    methods_dir.mkdir(exist_ok=True)
    tile_w, tile_h = 700, 530

    print(f"{'enhance':<22s}  {'tuning':<10s}  "
          f"{'paths':>5s}  {'REAL':>4s}  {'SLIV':>4s}  {'verdict':<10s}")
    print("-" * 70)

    results = []   # for the grid

    for ename, efn in ENHANCE_METHODS:
        try:
            enhanced = efn(cleaned_v14)
        except Exception as exc:
            print(f"{ename:<22s}  enhance FAILED: {exc}")
            continue
        cv2.imwrite(str(methods_dir / f"{ename}_enhanced.png"), enhanced)
        stripped = strip_all_grid_residue(enhanced)
        cv2.imwrite(str(methods_dir / f"{ename}_stripped.png"), stripped)

        row_panels = [
            _panel(enhanced, f"enhance: {ename}", tile_w, tile_h),
        ]
        best_for_method = None
        for tag, dil, close in GAP_FILL_PRESETS:
            trace = cleaned_to_trace(stripped)
            try:
                paths, pstats = extract_vector_paths_pencil(
                    trace, design_width_mm=PAPER_W_MM,
                    design_height_mm=PAPER_H_MM,
                    dilate_px=dil, close_after_dilate_px=close,
                )
            except Exception as exc:
                print(f"{ename}/{tag}  extract FAILED: {exc}")
                continue
            H, W = trace.shape
            classified = classify_paths(paths, (H, W),
                                       PAPER_W_MM, PAPER_H_MM)
            counts = category_counts(classified)
            real_paths = [p for p, c in zip(paths, classified)
                         if c.category == "REAL"]
            base = np.full((H, W, 3), 255, dtype=np.uint8)
            overlay = render_vector_overlay(
                base, real_paths, mask_shape=(H, W),
                colour_bgr=(40, 180, 40), thickness=3)
            delta = counts["REAL"] - EXPECTED
            if abs(delta) <= 1:
                verdict = "EXCELLENT"
            elif abs(delta) <= 3:
                verdict = "GOOD"
            elif counts["REAL"] < 4:
                verdict = "POOR"
            else:
                verdict = "FAIR"
            print(f"{ename:<22s}  {tag:<10s}  "
                  f"{len(paths):>5d}  {counts['REAL']:>4d}  "
                  f"{counts['SLIVER']:>4d}  {verdict:<10s}")

            sub = (f"d={dil} c={close} | "
                  f"REAL={counts['REAL']} SLIV={counts['SLIVER']} "
                  f"verdict={verdict}")
            row_panels.append(_panel(overlay, tag, tile_w, tile_h, subtitle=sub))
            results.append({
                "enhance": ename, "tuning": tag, "dil": dil, "close": close,
                "total_paths": len(paths),
                "real": counts["REAL"], "sliver": counts["SLIVER"],
                "verdict": verdict, "delta_from_expected": delta,
            })
            if best_for_method is None or abs(delta) < abs(best_for_method["delta_from_expected"]):
                best_for_method = results[-1]

        row = np.hstack(row_panels)
        cv2.imwrite(str(run_dir / f"row_{ename}.png"), row)

    # Build a grid: rows = enhance, cols = tunings
    print("\nBuilding mega-grid")
    rows_imgs = []
    for ename, _ in ENHANCE_METHODS:
        rp = run_dir / f"row_{ename}.png"
        if rp.exists():
            rows_imgs.append(cv2.imread(str(rp)))
    if rows_imgs:
        max_w = max(r.shape[1] for r in rows_imgs)
        padded = []
        for r in rows_imgs:
            if r.shape[1] < max_w:
                pad = np.full((r.shape[0], max_w - r.shape[1], 3), 250,
                             dtype=np.uint8)
                r = np.hstack([r, pad])
            padded.append(r)
        grid = np.vstack(padded)
        cv2.imwrite(str(run_dir / "MEGA_GRID.png"), grid)

    # Pick winner
    winners = sorted(results, key=lambda r: (abs(r["delta_from_expected"]),
                                            -r["real"]))
    print("\n=== TOP 5 CONFIGS ===")
    for w in winners[:5]:
        print(f"  {w['enhance']:<22s} {w['tuning']:<10s} "
              f"REAL={w['real']} delta={w['delta_from_expected']:+d} "
              f"{w['verdict']}")

    import json
    (run_dir / "results.json").write_text(
        json.dumps({"results": results, "top5": winners[:5]}, indent=2),
        encoding="utf-8")
    print(f"\nFolder: {run_dir}")


if __name__ == "__main__":
    main()
