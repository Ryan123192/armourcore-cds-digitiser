"""One-shot CLI for the in-house tracing workflow (SCAN inputs).

Each run goes into a TIMESTAMPED folder under
``data/outputs/InhouseProduction/<YYYYMMDD-HHMMSS>_<stem>/`` so a
back-and-forth iteration produces an ordered history, not a soup.

Output per run:
    diagnostic.png      <-- single 5-panel page: corners | rectified
                            | grid removed | trace + gap-fill | vectors
    vectors.svg         <-- production SVG (REAL paths only)
    vectors_all.svg     <-- includes anything classifier flagged
    report.json         <-- machine-readable verdict + counts
    cleaned.png         <-- raw phase 2 final cleaned image
    overlay.png         <-- vectors-on-white
    rectified.png       <-- phase 1 output

Usage:
    python tools/vectorise_cli.py path/to/scan.png
    python tools/vectorise_cli.py path/to/scan.png --route pencil
    python tools/vectorise_cli.py path/to/scan.png --phone   # use old rectifier
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

import cv2
import numpy as np

from armourcore_cds.phase1.marker_rectify_fast_v4 import (
    rectify_with_markers_fast_v4,
    PAPER_W_MM, PAPER_H_MM, DEFAULT_PX_PER_MM,
)
from armourcore_cds.phase1.marker_rectify_scan import rectify_scan
from armourcore_cds.phase2.trace_isolation import _remove_small_components
from armourcore_cds.phase2.grid_strip import strip_all_grid_residue
from armourcore_cds.phase2.grid_strip_aggressive import (
    strip_major_grid_aggressive, strip_dark_border,
)
from armourcore_cds.phase2.line_strip import strip_dashed_boundary_lines
from armourcore_cds.phase2.pencil_enhance_v2 import sauvola_adaptive
from armourcore_cds.phase2.clean_scan import (
    clean_scan_pen, clean_scan_pencil, strip_blue_notes,
)
from armourcore_cds.phase3.vectorise import write_svg, render_vector_overlay
from armourcore_cds.phase3.vectorise_v4 import (
    extract_vector_paths_closed_only,
)
from armourcore_cds.phase3.vectorise_pencil import extract_vector_paths_pencil
from armourcore_cds.phase3.path_classifier import (
    classify_paths, render_classified, category_counts,
)
from armourcore_cds.templates.registry import load_template_config

sys.path.insert(0, str(REPO / "tools/pipeline_dev/step02_grid_removal"))
from batch_flat_methods_v14 import (   # noqa
    method_M_v14_adaptive, detect_pencil_image,
)


TEMPLATE_ID = "cds_colour_test_260x350"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cleaned_to_trace_mask(cleaned_bgr, dark_thr=170, min_component=15,
                         drop_giant_frac=0.5):
    g = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
    m = (g < dark_thr).astype(np.uint8) * 255
    m = _remove_small_components(m, min_area_px=min_component)
    if drop_giant_frac > 0:
        H, W = m.shape
        image_area = H * W
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        kill = np.zeros_like(m)
        for cid in range(1, n):
            bw = stats[cid, cv2.CC_STAT_WIDTH]
            bh = stats[cid, cv2.CC_STAT_HEIGHT]
            if bw * bh > image_area * drop_giant_frac:
                kill[lbl == cid] = 255
        m[kill > 0] = 0
    return m


def render_trace_mask_for_diag(trace_mask: np.ndarray) -> np.ndarray:
    """Trace mask visualised in a way that shows the gap-fill state."""
    H, W = trace_mask.shape
    out = np.full((H, W, 3), 255, dtype=np.uint8)
    out[trace_mask > 0] = (40, 40, 40)
    return out


def _fit(img, max_w, max_h):
    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h, 1.0)
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))),
                      interpolation=cv2.INTER_AREA)


def _stage_panel(img, title, tile_w, tile_h, subtitle=""):
    canvas = np.full((tile_h, tile_w, 3), 250, dtype=np.uint8)
    head = 64 if subtitle else 44
    cv2.rectangle(canvas, (0, 0), (tile_w, head), (35, 35, 35), -1)
    cv2.putText(canvas, title, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.78,
                (255, 255, 255), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(canvas, subtitle, (14, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (180, 210, 255), 1, cv2.LINE_AA)
    body_h = tile_h - head - 8
    fit = _fit(img, tile_w - 10, body_h)
    fh, fw = fit.shape[:2]
    canvas[head + (body_h - fh) // 2:head + (body_h - fh) // 2 + fh,
           (tile_w - fw) // 2:(tile_w - fw) // 2 + fw] = fit
    return canvas


def build_diagnostic_page(stages: list[tuple[np.ndarray, str, str]],
                         header_text: str,
                         tile_w: int = 950,
                         tile_h: int = 720) -> np.ndarray:
    """Build a multi-panel diagnostic image.

    `stages` is a list of (image, title, subtitle).  Arranged 3-per-row.
    Header text is the big banner across the top.
    """
    panels = [_stage_panel(im, t, tile_w, tile_h, subtitle=s)
              for (im, t, s) in stages]
    cols = 3
    rows = (len(panels) + cols - 1) // cols
    row_imgs = []
    for r in range(rows):
        row = panels[r * cols:(r + 1) * cols]
        while len(row) < cols:
            row.append(np.full((tile_h, tile_w, 3), 250, dtype=np.uint8))
        row_imgs.append(np.hstack(row))
    body = np.vstack(row_imgs)
    header = np.full((96, body.shape[1], 3), 18, dtype=np.uint8)
    cv2.putText(header, header_text, (24, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                (255, 255, 255), 3, cv2.LINE_AA)
    return np.vstack([header, body])


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(image_path: Path, out_root: Path,
                forced_route: str | None = None,
                rectifier: str = "scan",
                ts_prefix: str | None = None):
    timings = {}
    overall_t0 = time.time()

    # Timestamped output folder
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder_name = f"{ts}_{image_path.stem}"
    if ts_prefix:
        folder_name = f"{ts_prefix}_{folder_name}"
    out_dir = out_root / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read {image_path}")

    # ---- Phase 1: rectify ----
    t0 = time.time()
    if rectifier == "scan":
        rr = rectify_scan(img, paper_w_mm=PAPER_W_MM,
                         paper_h_mm=PAPER_H_MM, px_per_mm=DEFAULT_PX_PER_MM)
        rect = rr.warped
        marker_debug = rr.debug_overlay
    else:
        rr = rectify_with_markers_fast_v4(
            img, paper_w_mm=PAPER_W_MM, paper_h_mm=PAPER_H_MM,
            px_per_mm=DEFAULT_PX_PER_MM,
        )
        rect = rr.warped
        marker_debug = img.copy()   # no overlay on old rectifier
    timings["rectify"] = time.time() - t0
    cv2.imwrite(str(out_dir / "rectified.png"), rect)
    cv2.imwrite(str(out_dir / "_01_corners.png"), marker_debug)

    template = load_template_config(TEMPLATE_ID)

    # ---- Route detection ----
    if forced_route:
        route = forced_route
    elif rectifier == "scan":
        # Simple ink-darkness probe for scan mode
        route = "pencil" if detect_pencil_image(rect) else "pen"
    else:
        if detect_pencil_image(rect):
            route = "pencil"
        else:
            probe = method_M_v14_adaptive(rect, template)
            g = cv2.cvtColor(probe, cv2.COLOR_BGR2GRAY)
            ink = (g < 170).astype(np.uint8) * 255
            n, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
            H, W = ink.shape
            image_area = H * W
            giant = any(
                stats[cid, cv2.CC_STAT_WIDTH] *
                stats[cid, cv2.CC_STAT_HEIGHT] > image_area * 0.25
                for cid in range(1, n)
            )
            route = "pen_darkborder" if giant else "pen_standard"

    # ---- Phase 2: grid removal / cleaning ----
    t0 = time.time()
    if rectifier == "scan":
        # SCAN PATH: simple colour-based filter only.
        # BLUE FILTERING IS COMPLETELY OFF in the default path - it
        # was eroding scanner anti-aliased edges of black pen strokes.
        # Users who want blue-note stripping can run with the
        # --blue-strip CLI flag (not wired in by default).
        if route == "pencil":
            cleaned = clean_scan_pencil(rect)
        else:
            cleaned = clean_scan_pen(rect)
    else:
        # PHONE PATH: full production pipeline with all the photo-survival
        # tricks (v14 adaptive + grid_strip + line_strip + darkborder).
        cleaned_v14 = method_M_v14_adaptive(rect, template)
        if route == "pencil":
            cleaned = sauvola_adaptive(cleaned_v14, window_size=25, k=0.15)
            cleaned = strip_all_grid_residue(cleaned)
        elif route == "pen_darkborder":
            c = strip_dark_border(cleaned_v14, border_zone_px=250,
                                 ink_threshold_gray=100)
            c = strip_major_grid_aggressive(
                c, band_half_px=8, ink_protect_max_gray=70,
                border_zone_px=250, border_ink_protect=0)
            c = strip_all_grid_residue(c)
            cleaned = strip_dashed_boundary_lines(c)
        else:
            cleaned = strip_all_grid_residue(cleaned_v14)
    timings["phase2"] = time.time() - t0
    cv2.imwrite(str(out_dir / "cleaned.png"), cleaned)

    # ---- Phase 2.5: trace mask (= gap-fill input) ----
    t0 = time.time()
    trace = cleaned_to_trace_mask(cleaned)
    timings["trace_mask"] = time.time() - t0
    trace_vis = render_trace_mask_for_diag(trace)

    # ---- Phase 3: vectorise ----
    t0 = time.time()
    chosen_dilate = None
    if route in ("pencil", "pen_dense"):
        # Dilate-first sweep - good for sparse pencil OR pen-with-scribbles.
        best_paths = []
        best_dil = 5
        for dil, cls in [(5, 3), (9, 5), (13, 7), (17, 11)]:
            try:
                cand, _ = extract_vector_paths_pencil(
                    trace, design_width_mm=PAPER_W_MM,
                    design_height_mm=PAPER_H_MM,
                    dilate_px=dil, close_after_dilate_px=cls,
                )
            except Exception:
                cand = []
            if len(cand) > len(best_paths):
                best_paths = cand
                best_dil = dil
        paths = best_paths
        chosen_dilate = best_dil
    else:
        # Pen "clean tracings" - vectorise_v4 RETR_CCOMP hole-preference
        # ONLY.  No auto-fallback - deterministic high-quality tracing
        # for clean controlled scans.
        paths, _, _ = extract_vector_paths_closed_only(
            trace, design_width_mm=PAPER_W_MM, design_height_mm=PAPER_H_MM,
            route="pen",
        )
    timings["phase3"] = time.time() - t0

    H, W = trace.shape
    classified = classify_paths(paths, (H, W), PAPER_W_MM, PAPER_H_MM)
    counts = category_counts(classified)
    real_paths = [p for p, c in zip(paths, classified)
                 if c.category == "REAL"]

    base = np.full((H, W, 3), 255, dtype=np.uint8)
    overlay = render_vector_overlay(
        base, real_paths, mask_shape=(H, W),
        colour_bgr=(40, 180, 40), thickness=3)
    cv2.imwrite(str(out_dir / "overlay.png"), overlay)
    annotated = render_classified(paths, classified, (H, W))

    write_svg(real_paths, out_dir / "vectors.svg",
             mask_shape=(H, W), design_width_mm=PAPER_W_MM,
             design_height_mm=PAPER_H_MM)
    write_svg(paths, out_dir / "vectors_all.svg",
             mask_shape=(H, W), design_width_mm=PAPER_W_MM,
             design_height_mm=PAPER_H_MM)

    # ---- Verdict ----
    if counts["REAL"] == 0:
        verdict = "EMPTY"
    elif counts["GRID"] == 0 and counts["TEXT"] == 0 and counts["NOISE"] == 0:
        verdict = "EXCELLENT"
    elif counts["GRID"] + counts["TEXT"] + counts["NOISE"] <= 2:
        verdict = "GOOD"
    elif counts["GRID"] > 0:
        verdict = "GRID_LEAK"
    else:
        verdict = "POOR"

    total_time = time.time() - overall_t0
    timings["total"] = total_time

    # ---- Diagnostic page (5 stages) ----
    header = (f"{image_path.name}   |   route: {route}   |   "
              f"VERDICT: {verdict}   |   {total_time:.1f}s")
    stages = [
        (marker_debug,
         "1. corner recognition",
         f"rectify={timings['rectify']:.2f}s   rectifier={rectifier}"),
        (rect,
         "2. rectified", ""),
        (cleaned,
         "3. grid + artefact removal",
         f"phase2={timings['phase2']:.2f}s"),
        (trace_vis,
         "4. trace mask (gap-fill input)",
         f"trace={timings['trace_mask']:.2f}s   ink_components={int((trace>0).sum())}"),
        (annotated,
         "5. classified vectors",
         f"phase3={timings['phase3']:.2f}s   "
         f"REAL={counts['REAL']} GRID={counts['GRID']} "
         f"TEXT={counts['TEXT']} SLIV={counts['SLIVER']} NOISE={counts['NOISE']}"),
        (overlay,
         "6. FINAL EXPORT (REAL only)",
         f"{len(real_paths)} paths to SVG"),
    ]
    diag = build_diagnostic_page(stages, header)
    cv2.imwrite(str(out_dir / "diagnostic.png"), diag)

    # ---- JSON report ----
    report = {
        "input": str(image_path),
        "timestamp": ts,
        "route": route,
        "rectifier": rectifier,
        "verdict": verdict,
        "counts": counts,
        "total_paths": len(paths),
        "real_paths": len(real_paths),
        "pencil_chosen_dilate": chosen_dilate,
        "timings_sec": {k: round(v, 3) for k, v in timings.items()},
        "vector_svg": str(out_dir / "vectors.svg"),
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n=== {image_path.name} ===")
    print(f"Folder:          {out_dir}")
    print(f"Rectifier:       {rectifier}")
    print(f"Route:           {route}")
    print(f"Verdict:         {verdict}")
    print(f"REAL paths:      {counts['REAL']}")
    print(f"GRID artefacts:  {counts['GRID']}")
    print(f"SLIVER rejected: {counts['SLIVER']}")
    print(f"Time:            {total_time:.1f}s "
          f"(P1 {timings['rectify']:.1f}s + P2 {timings['phase2']:.1f}s + "
          f"P3 {timings['phase3']:.1f}s)")
    print(f"Diagnostic:      {out_dir / 'diagnostic.png'}")
    print(f"SVG:             {out_dir / 'vectors.svg'}")
    return out_dir, verdict


def main():
    ap = argparse.ArgumentParser(description="ArmourCore CDS vectoriser")
    ap.add_argument("image", type=Path, help="Path to scan / photo")
    ap.add_argument("--out", type=Path,
                   default=Path("data/outputs/InhouseProduction"),
                   help="Output root (default: data/outputs/InhouseProduction/)")
    ap.add_argument("--route", choices=["pen", "pen_dense",
                                       "pen_standard", "pen_darkborder",
                                       "pencil"], default=None,
                   help="Force route (default: auto-detect). "
                        "pen=clean tracings, pen_dense=with notes/scribbles")
    ap.add_argument("--phone", action="store_true",
                   help="Use phone-photo rectifier instead of scan rectifier")
    ap.add_argument("--tag", default=None,
                   help="Optional prefix for the output folder name")
    args = ap.parse_args()
    rectifier = "phone" if args.phone else "scan"
    run_pipeline(args.image, args.out,
                forced_route=args.route, rectifier=rectifier,
                ts_prefix=args.tag)


if __name__ == "__main__":
    main()
