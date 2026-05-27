"""Step 01 — Diagnostic deep-dive on the 4 problem cases.

Runs the FAST marker detector with ``debug_dir`` enabled on only the
four cases the user has flagged:

    * BLUE_PEN_FLAT_03         (fail — angled markers)
    * BLUE_PENCIL_FLAT_01      (fail — angled markers)
    * BLUE_PEN_CREASED_01      (pass count but misidentified ROI)
    * BLUE_PENCIL_CREASED_04   (pass count but misidentified ROI)

For each case the runner writes every intermediate the algorithm
produces:

    00_original.jpg         input
    01a_red_ink_mask.jpg    Lab delta-a mask before morphology
    01b_marker_core.jpg     after close+open (this drives coarse ROI search)
    02_roi_<corner>.jpg     the per-corner refine ROI
    03_<corner>_tmpl_cnr.jpg  template image that scored highest
    03_<corner>_ink_lab.jpg   Lab ink mask inside the ROI
    04_template_<corner>.png  reference template for that corner
    10_inner_corners_and_quad.jpg   final overlay
    20_rectified.jpg        the warped output (if all 4 markers found)

It also adds a custom diagnostic showing where the coarse pass placed
each ROI relative to the actual marker location, so you can SEE the
"attached to dark area" failure mode visually.

    python tools/pipeline_dev/step01_border_detection/debug_problem_cases.py
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

import cv2
import numpy as np

from armourcore_cds.phase1.marker_rectify_fast import (
    coarse_marker_rois, detect_markers_fast, rectify_with_markers_fast,
    PAPER_W_MM, PAPER_H_MM, DEFAULT_PX_PER_MM, ROI_HALF,
    CORNER_PAD_FRAC, MARKER_MM,
)
from tools.pipeline_dev.corpus import (
    RAW_IMAGES_DIR, make_run_dir, label_image, grid_montage,
)


PROBLEM_CASES = [
    "BLUE_PEN_FLAT_03",        # angled markers
    "BLUE_PENCIL_FLAT_01",     # angled markers
    "BLUE_PEN_CREASED_01",     # misidentified ROI
    "BLUE_PENCIL_CREASED_04",  # misidentified ROI
]


def _draw_quadrants(img: np.ndarray) -> np.ndarray:
    """Show the four quadrant search regions on the input (yellow boxes)."""
    out = img.copy()
    h, w = out.shape[:2]
    pad = CORNER_PAD_FRAC
    quads = {
        "TL": (0,              0,               int(w * pad), int(h * pad)),
        "TR": (int(w*(1-pad)), 0,               w,            int(h * pad)),
        "BR": (int(w*(1-pad)), int(h*(1-pad)),  w,            h),
        "BL": (0,              int(h*(1-pad)),  int(w * pad), h),
    }
    for label, (x1, y1, x2, y2) in quads.items():
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 255), 4)
        cv2.putText(out, f"{label}-quad", (x1 + 10, y1 + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 200, 255), 3)
    return out


def _draw_coarse_rois(img: np.ndarray,
                      rois: dict,
                      roi_half: int) -> np.ndarray:
    """Draw the coarse-centroid + refine ROI box on the input."""
    out = img.copy()
    for label, (cx, cy) in rois.items():
        # Coarse centroid
        cv2.circle(out, (int(cx), int(cy)), 18, (255, 255, 0), -1)
        cv2.putText(out, label, (int(cx) + 22, int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)
        # Refine ROI bbox
        cv2.rectangle(
            out,
            (int(cx - roi_half), int(cy - roi_half)),
            (int(cx + roi_half), int(cy + roi_half)),
            (255, 200, 0), 4,
        )
    return out


def _per_case_diagnostic(case_path: Path, out_dir: Path) -> dict:
    """Run detection with full debug + extra diagnostic overlays."""
    out_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(case_path))
    if img is None:
        return {"stem": case_path.stem, "status": "load_failed"}

    info = {
        "stem": case_path.stem,
        "image_dimensions_px": [int(img.shape[1]), int(img.shape[0])],
    }

    # 1) Show the four quadrants on the input — this is where coarse searches.
    cv2.imwrite(str(out_dir / "QQ_quadrants.png"), _draw_quadrants(img))

    # 2) Run the coarse pass independently so we can show the rough centres
    #    and ROI boxes BEFORE the refine stage runs.
    rois, _core = coarse_marker_rois(img, debug_dir=out_dir)
    cv2.imwrite(
        str(out_dir / "QQ_coarse_rois_on_input.png"),
        _draw_coarse_rois(img, rois, ROI_HALF),
    )
    info["coarse_rois"] = {k: list(v) for k, v in rois.items()}

    # 3) Now run the full pipeline with debug_dir so all per-method
    #    images get saved too.
    t0 = time.time()
    try:
        markers = detect_markers_fast(img, debug_dir=out_dir)
        info["markers"] = {
            label: {
                "centre_xy": list(d.centre_xy),
                "size_wh":   list(d.size_wh),
                "angle_deg": d.angle_deg,
                "method":    d.method,
                "score":     d.score,
            }
            for label, d in markers.items()
        }
        info["n_markers"] = len(markers)
        # Try the full rectify (will raise if any marker missing)
        try:
            result = rectify_with_markers_fast(
                img,
                paper_w_mm=PAPER_W_MM,
                paper_h_mm=PAPER_H_MM,
                px_per_mm=DEFAULT_PX_PER_MM,
                debug_dir=out_dir,
            )
            info["rectified"] = "ok"
        except RuntimeError as exc:
            info["rectified"] = f"failed: {exc}"
        # Final overlay drawing
        vis = img.copy()
        for label, d in markers.items():
            cx, cy = d.centre_xy
            rw, rh = d.size_wh
            ang = d.angle_deg
            box = cv2.boxPoints(((cx, cy), (rw, rh), ang)).astype(np.int32)
            cv2.polylines(vis, [box], True, (0, 255, 0), 6)
            cv2.circle(vis, (int(cx), int(cy)), 14, (0, 0, 255), -1)
            cv2.putText(vis, f"{label}[{d.method}] s={d.score:.1f}",
                        (int(cx - rw/2), int(cy - rh/2) - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.imwrite(str(out_dir / "99_final_with_scores.png"), vis)
        info["status"] = "ok"
    except Exception as exc:
        info["status"] = "failed"
        info["error"] = str(exc)
        info["traceback"] = traceback.format_exc()
    info["elapsed_s"] = round(time.time() - t0, 3)
    (out_dir / "info.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8",
    )
    return info


def main() -> None:
    run_dir = make_run_dir(
        "step01_border_detection", "DEBUG_problem_cases",
    )
    print(f"Debug run dir: {run_dir.relative_to(REPO)}\n")

    summaries = []
    for stem in PROBLEM_CASES:
        # try .JPG, .jpg
        candidates = list(RAW_IMAGES_DIR.glob(f"{stem}.*"))
        if not candidates:
            print(f"!! could not find input for {stem}")
            continue
        case_path = candidates[0]
        sub = run_dir / stem
        print(f"\n--- {stem} ---")
        info = _per_case_diagnostic(case_path, sub)
        summaries.append(info)
        markers = info.get("markers", {})
        for label in ("TL", "TR", "BR", "BL"):
            if label in markers:
                m = markers[label]
                cx, cy = m["centre_xy"]
                print(f"  {label}: ({cx:7.1f},{cy:7.1f}) "
                      f"size={m['size_wh'][0]:.0f}x{m['size_wh'][1]:.0f}  "
                      f"method={m['method']:8s}  score={m['score']:.2f}")
            else:
                print(f"  {label}: MISSING")

    (run_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8",
    )
    print(f"\nFolder: {run_dir}")


if __name__ == "__main__":
    main()
