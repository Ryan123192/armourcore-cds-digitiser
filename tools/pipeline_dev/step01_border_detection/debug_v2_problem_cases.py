"""V2 deep-dive on the 2 still-failing angled cases.

Same shape as debug_problem_cases.py but uses v2.
"""
from __future__ import annotations
import json, sys, time, traceback
from pathlib import Path

REPO = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

import cv2
from armourcore_cds.phase1.marker_rectify_fast_v2 import (
    coarse_marker_rois, detect_markers_fast_v2, rectify_with_markers_fast_v2,
    ROI_HALF,
)
from tools.pipeline_dev.corpus import RAW_IMAGES_DIR, make_run_dir

CASES = ["BLUE_PEN_FLAT_03", "BLUE_PENCIL_FLAT_01"]


def main():
    out_root = make_run_dir("step01_border_detection", "DEBUG_v2_angled")
    print(f"Out: {out_root.relative_to(REPO)}\n")

    for stem in CASES:
        cands = list(RAW_IMAGES_DIR.glob(f"{stem}.*"))
        if not cands:
            print(f"missing {stem}"); continue
        img = cv2.imread(str(cands[0]))
        case_dir = out_root / stem
        case_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(case_dir / "00_original.jpg"), img)

        print(f"\n--- {stem} ---")
        # 1) Show v2's coarse-pass quadrants (with paper bbox visible)
        rois, core, paper_bbox = coarse_marker_rois(img, debug_dir=case_dir)
        print(f"  paper_bbox: {paper_bbox}")
        print(f"  coarse_rois: {rois}")

        # Annotated overlay
        vis = img.copy()
        if paper_bbox:
            px, py, pw, ph = paper_bbox
            cv2.rectangle(vis, (px, py), (px + pw, py + ph),
                          (255, 255, 0), 5)
            cv2.putText(vis, "PAPER", (px + 10, py + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 4)
        for label, (cx, cy) in rois.items():
            cv2.circle(vis, (int(cx), int(cy)), 22, (0, 255, 255), -1)
            cv2.rectangle(vis,
                          (int(cx - ROI_HALF), int(cy - ROI_HALF)),
                          (int(cx + ROI_HALF), int(cy + ROI_HALF)),
                          (0, 200, 0), 4)
            cv2.putText(vis, label, (int(cx) + 28, int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        cv2.imwrite(str(case_dir / "QQ_v2_overlay.jpg"), vis)

        # 2) Full v2 detect (with debug)
        try:
            markers, paper_bbox_again = detect_markers_fast_v2(
                img, debug_dir=case_dir,
            )
            print(f"  detect_markers_fast_v2 found {len(markers)}/4:")
            for label in ("TL", "TR", "BR", "BL"):
                if label in markers:
                    d = markers[label]
                    print(f"    {label}: ({d.centre_xy[0]:.0f},{d.centre_xy[1]:.0f}) "
                          f"size={d.size_wh[0]:.0f}x{d.size_wh[1]:.0f} "
                          f"method={d.method} score={d.score:.2f}")
                else:
                    print(f"    {label}: MISSING")
        except Exception as exc:
            print(f"  detect failed: {exc}")


if __name__ == "__main__":
    main()
