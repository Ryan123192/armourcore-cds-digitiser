"""Step 01 — Border detection: baseline run on the BLUE_* corpus.

Runs the EXISTING ``detect_outer_border`` from
``src/armourcore_cds/phase1/boundary_detection.py`` on every test file
and saves:

  data/outputs/pipeline_dev/step01_border_detection/<stamp>_baseline/
    <stem>/
      input.jpg                  -- copy / link of the original input
      overlay.png                -- detected quad drawn on the input
      info.json                  -- corners, score, confidence, timings
    summary.png                  -- 4x3 grid of overlays for at-a-glance review
    summary.json                 -- per-file headline stats

The baseline run is the reference point against which any future
method must justify itself.  Never modifies the production code.

    python tools/pipeline_dev/step01_border_detection/run_baseline.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

import cv2
import numpy as np

from armourcore_cds.phase1.boundary_detection import (
    detect_outer_border, BorderDetectionResult,
)
from armourcore_cds.templates.registry import load_template_config
from tools.pipeline_dev.corpus import (
    discover_corpus, make_run_dir, label_image, grid_montage,
)


# Blue-border CDS sheets use this template (260 x 350 mm design area).
TEMPLATE_ID = "cds_colour_test_260x350"

# Overlay colours
COLOUR_BORDER_OK   = (50, 200, 255)   # cyan-ish — detected quad
COLOUR_BORDER_BAD  = (40, 40, 220)    # red — used if detection failed


def _draw_quad(
    img: np.ndarray,
    quad: np.ndarray | list,
    colour: tuple[int, int, int],
    thickness: int = 6,
) -> np.ndarray:
    """Return *img* with the quad drawn as a closed polygon."""
    out = img.copy()
    pts = np.asarray(quad, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [pts], isClosed=True, color=colour,
                  thickness=thickness, lineType=cv2.LINE_AA)
    # Corner dots for clarity
    for pt in pts.reshape(-1, 2):
        cv2.circle(out, tuple(pt.tolist()), thickness * 2, colour, -1,
                   lineType=cv2.LINE_AA)
    return out


def _run_one(
    case_path: Path,
    expected_aspect_ratio: float,
    out_subdir: Path,
) -> dict:
    """Run the existing border detector on one image; save overlay + info."""
    out_subdir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(case_path))
    if img is None:
        return {
            "stem": case_path.stem,
            "status": "load_failed",
            "elapsed_s": 0.0,
        }

    info: dict = {
        "stem": case_path.stem,
        "input_path": str(case_path),
        "image_dimensions_px": [int(img.shape[1]), int(img.shape[0])],
    }

    t0 = time.time()
    try:
        result: BorderDetectionResult = detect_outer_border(
            img,
            expected_aspect_ratio=expected_aspect_ratio,
            border_colour_mode="blue",          # explicit: corpus is all blue
            use_colour_hint=True,
            use_shape_constraints=True,
            fallback_to_fiducials=True,
        )
        info["status"] = "ok"
        info["confidence"] = result.confidence
        info["score"] = float(result.score)
        info["candidate_count"] = int(result.candidate_count)
        info["contour_area_px"] = float(result.contour_area_px)
        info["corners_xy"] = [
            [float(x), float(y)] for x, y in result.ordered_corners_xy
        ]
        diag = result.diagnostics or {}
        # Subset of diagnostics we care about
        info["diagnostics"] = {
            k: v for k, v in diag.items()
            if k in {
                "selected_mode", "selected_score", "ranks",
                "marker_support", "outside_envelope_penalty",
                "side_quality", "corner_support",
            }
        }

        overlay = _draw_quad(img, result.ordered_corners_xy,
                             COLOUR_BORDER_OK, thickness=6)
        h_img, w_img = img.shape[:2]
        banner = (
            f"{case_path.stem}  |  score={result.score:.3f}  "
            f"conf={result.confidence}  candidates={result.candidate_count}"
        )
        overlay = label_image(overlay, banner, height_px=48, scale=0.9)
    except Exception as exc:
        info["status"] = "failed"
        info["error"] = str(exc)
        overlay = label_image(
            img.copy(),
            f"{case_path.stem}  |  DETECTION FAILED: {exc}",
            height_px=48, scale=0.8,
            bg=(0, 0, 80),
        )
    info["elapsed_s"] = round(time.time() - t0, 3)

    overlay_path = out_subdir / "overlay.png"
    cv2.imwrite(str(overlay_path), overlay)
    info["overlay_path"] = str(overlay_path)

    # write per-case info.json
    (out_subdir / "info.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8"
    )
    return info


def main() -> None:
    corpus = discover_corpus()
    if not corpus:
        print("No corpus files found in data/inputs/raw_images/")
        print("Looking for BLUE_<PEN|PENCIL>_<FLAT|CREASED>_<NN>.JPG")
        sys.exit(1)

    template = load_template_config(TEMPLATE_ID)
    aspect = (
        float(template.design_area_mm.width)
        / float(template.design_area_mm.height)
    )
    print(f"Loaded template '{TEMPLATE_ID}' (aspect = {aspect:.4f})")

    run_dir = make_run_dir("step01_border_detection", "baseline")
    print(f"Run dir: {run_dir.relative_to(REPO)}\n")

    print(f"{'case':<30s} {'elapsed':>7s}  {'conf':>6s}  {'score':>6s}  {'cands':>5s}  status")
    print("-" * 80)

    summaries: list[dict] = []
    overlays_for_montage: list[np.ndarray] = []

    for case in corpus:
        out_subdir = run_dir / case.stem
        info = _run_one(case.path, aspect, out_subdir)
        summaries.append({
            **info,
            "case_label": case.label,
            "medium": case.medium,
            "paper": case.paper,
            "index": case.index,
        })
        status = info["status"]
        conf = info.get("confidence", "-")
        score = info.get("score", float("nan"))
        cands = info.get("candidate_count", -1)
        score_str = f"{score:.3f}" if isinstance(score, (int, float)) and score == score else "-"
        print(
            f"{case.stem:<30s} {info['elapsed_s']:>6.2f}s  {conf:>6s}  "
            f"{score_str:>6s}  {cands:>5d}  {status}"
        )

        # Load the saved overlay back for the montage
        overlay = cv2.imread(str(out_subdir / "overlay.png"))
        if overlay is not None:
            overlays_for_montage.append(overlay)

    # Build the 4 x N summary grid
    if overlays_for_montage:
        montage = grid_montage(overlays_for_montage, cols=4, tile_max_dim=600)
        cv2.imwrite(str(run_dir / "summary.png"), montage)
        print(f"\nSummary grid: {run_dir / 'summary.png'}")

    # Write the per-step summary JSON
    summary_payload = {
        "step": "step01_border_detection",
        "method": "baseline",
        "template_id": TEMPLATE_ID,
        "aspect_ratio": aspect,
        "n_cases": len(summaries),
        "n_ok": sum(1 for s in summaries if s.get("status") == "ok"),
        "n_failed": sum(1 for s in summaries if s.get("status") != "ok"),
        "cases": summaries,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2), encoding="utf-8"
    )
    print(f"Summary JSON: {run_dir / 'summary.json'}")
    print(f"\nFolder: {run_dir}")


if __name__ == "__main__":
    main()
