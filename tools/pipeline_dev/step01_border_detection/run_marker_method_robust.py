"""Step 01 — Border detection: robust marker method (Lab + corner templates).

Same shape as ``run_marker_method.py`` but calls
``rectify_with_markers_robust`` from the new sibling module
``src/armourcore_cds/phase1/marker_rectify_robust.py``.

Outputs:

    data/outputs/pipeline_dev/step01_border_detection/<stamp>_marker_robust/
        <stem>/
            overlay.png
            rectified.png
            info.json
        summary_overlays.png
        summary_rectified.png
        summary.json

The non-robust ``marker_rectify.py`` and its dev runner stay intact for
side-by-side comparison.

    python tools/pipeline_dev/step01_border_detection/run_marker_method_robust.py
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

from armourcore_cds.phase1.marker_rectify_robust import (
    rectify_with_markers_robust,
    PAPER_W_MM, PAPER_H_MM, DEFAULT_PX_PER_MM,
)
from tools.pipeline_dev.corpus import (
    discover_corpus, make_run_dir, label_image, grid_montage,
)


def _run_one(case_path: Path, out_subdir: Path) -> dict:
    out_subdir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(case_path))
    if img is None:
        return {"stem": case_path.stem, "status": "load_failed",
                "elapsed_s": 0.0}

    info: dict = {
        "stem": case_path.stem,
        "input_path": str(case_path),
        "image_dimensions_px": [int(img.shape[1]), int(img.shape[0])],
    }
    t0 = time.time()
    try:
        result = rectify_with_markers_robust(
            img,
            paper_w_mm=PAPER_W_MM,
            paper_h_mm=PAPER_H_MM,
            px_per_mm=DEFAULT_PX_PER_MM,
            debug_dir=None,
        )
        info["status"] = "ok"
        info["markers"] = {
            label: {
                "centre_xy":   list(det.centre_xy),
                "size_wh":     list(det.size_wh),
                "angle_deg":   det.angle_deg,
                "method":      det.method,
                "score":       det.score,
            }
            for label, det in result.markers.items()
        }
        info["inner_corners_xy"] = {
            k: list(v) for k, v in result.inner_corners_xy.items()
        }
        info["output_size_px"] = list(result.output_size_px)
        info["px_per_mm"] = result.px_per_mm

        # Overlay
        overlay = img.copy()
        for label, det in result.markers.items():
            cv2.polylines(overlay, [det.box.astype(np.int32)],
                          True, (0, 255, 0), 6)
            cx, cy = det.centre_xy
            cv2.circle(overlay, (int(cx), int(cy)), 12, (0, 0, 255), -1)
            cv2.putText(
                overlay,
                f"{label}[{det.method}]",
                (int(cx + 18), int(cy - 18)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 4,
            )
        cv2.polylines(
            overlay, [result.src_quad.astype(np.int32)], True,
            (255, 200, 0), 6,
        )
        for _lbl, pt in zip(("TL", "TR", "BR", "BL"), result.src_quad):
            cv2.circle(overlay, (int(pt[0]), int(pt[1])), 22,
                       (0, 255, 255), -1)
        overlay = label_image(
            overlay,
            f"{case_path.stem}  |  {sorted(result.markers.keys())}",
            height_px=52, scale=1.0,
        )
        cv2.imwrite(str(out_subdir / "overlay.png"), overlay)
        cv2.imwrite(str(out_subdir / "rectified.png"), result.warped)

    except Exception as exc:
        info["status"] = "failed"
        info["error"] = str(exc)
        info["traceback"] = traceback.format_exc()
        overlay = label_image(
            img.copy(),
            f"{case_path.stem}  |  FAILED: {exc}",
            height_px=52, scale=0.9, bg=(0, 0, 80),
        )
        cv2.imwrite(str(out_subdir / "overlay.png"), overlay)

    info["elapsed_s"] = round(time.time() - t0, 3)
    (out_subdir / "info.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8",
    )
    return info


def main() -> None:
    corpus = discover_corpus()
    if not corpus:
        print("No corpus files found.")
        sys.exit(1)

    run_dir = make_run_dir("step01_border_detection", "marker_robust")
    print(f"Run dir: {run_dir.relative_to(REPO)}\n")

    print(f"{'case':<28s} {'elapsed':>7s}  {'markers':>7s}  status")
    print("-" * 60)

    summaries: list[dict] = []
    overlay_tiles: list[np.ndarray] = []
    rectified_tiles: list[np.ndarray] = []

    for case in corpus:
        out_subdir = run_dir / case.stem
        info = _run_one(case.path, out_subdir)
        summaries.append({**info, "case_label": case.label,
                          "medium": case.medium, "paper": case.paper,
                          "index": case.index})
        n_markers = (
            len(info.get("markers", {}))
            if info.get("status") == "ok" else 0
        )
        print(f"{case.stem:<28s} {info['elapsed_s']:>6.2f}s  "
              f"{n_markers:>4d}/4   {info['status']}")

        ov = cv2.imread(str(out_subdir / "overlay.png"))
        if ov is not None:
            overlay_tiles.append(ov)
        rp = out_subdir / "rectified.png"
        if rp.exists():
            rect = cv2.imread(str(rp))
            if rect is not None:
                rectified_tiles.append(
                    label_image(rect, case.stem,
                                height_px=48, scale=0.9)
                )

    if overlay_tiles:
        cv2.imwrite(
            str(run_dir / "summary_overlays.png"),
            grid_montage(overlay_tiles, cols=4, tile_max_dim=600),
        )
    if rectified_tiles:
        cv2.imwrite(
            str(run_dir / "summary_rectified.png"),
            grid_montage(rectified_tiles, cols=4, tile_max_dim=600),
        )

    summary_payload = {
        "step": "step01_border_detection",
        "method": "marker_robust",
        "paper_w_mm": PAPER_W_MM,
        "paper_h_mm": PAPER_H_MM,
        "px_per_mm": DEFAULT_PX_PER_MM,
        "n_cases": len(summaries),
        "n_ok": sum(1 for s in summaries if s.get("status") == "ok"),
        "n_failed": sum(1 for s in summaries if s.get("status") != "ok"),
        "cases": summaries,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2), encoding="utf-8",
    )
    print(f"\nFolder: {run_dir}")


if __name__ == "__main__":
    main()
