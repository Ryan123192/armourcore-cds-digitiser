"""V3 deep-dive on PENCIL_CREASED_01 (the one remaining failure).

Saves intermediates so we can see which marker is being filtered
out by the global candidate filter, and why.
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

from armourcore_cds.phase1.marker_rectify_fast_v3 import (
    find_marker_candidates, red_ink_mask,
    CAND_MIN_AREA_PX, CAND_MAX_AREA_PX,
    CAND_MAX_ASPECT, CAND_MIN_SOLIDITY,
    EDGE_TOLERANCE_PX,
    COARSE_CLOSE_KSIZE, COARSE_OPEN_KSIZE,
)
from tools.pipeline_dev.corpus import RAW_IMAGES_DIR, make_run_dir


def main():
    case_path = next(RAW_IMAGES_DIR.glob("BLUE_PENCIL_CREASED_01.*"))
    img = cv2.imread(str(case_path))
    h, w = img.shape[:2]
    out = make_run_dir("step01_border_detection", "DEBUG_v3_pcc01")
    print(f"Out: {out.relative_to(REPO)}\n")

    # 1) Filtered candidates as the function returns
    candidates = find_marker_candidates(img, debug_dir=out)
    print(f"Filtered candidates ({len(candidates)}):")
    for i, c in enumerate(candidates):
        print(f"  {i}: centre=({c.centre_xy[0]:.0f},{c.centre_xy[1]:.0f}) "
              f"area={c.area:.0f} aspect="
              f"{max(c.minrect_size)/max(min(c.minrect_size),1):.2f} "
              f"solidity={c.solidity:.2f}")

    # 2) ALL contours pre-filter -- show each one and which test it failed
    mask, _ = red_ink_mask(img)
    k_close = cv2.getStructuringElement(
        cv2.MORPH_RECT, (COARSE_CLOSE_KSIZE, COARSE_CLOSE_KSIZE))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)
    k_open = cv2.getStructuringElement(
        cv2.MORPH_RECT, (COARSE_OPEN_KSIZE, COARSE_OPEN_KSIZE))
    core = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k_open, iterations=1)

    cnts, _ = cv2.findContours(core, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

    print(f"\nALL {len(cnts)} contours before filtering:")
    print(f"  thresholds: area=[{CAND_MIN_AREA_PX},{CAND_MAX_AREA_PX}]  "
          f"aspect<={CAND_MAX_ASPECT}  solidity>={CAND_MIN_SOLIDITY}  "
          f"edge_tol={EDGE_TOLERANCE_PX}")

    vis = img.copy()
    for i, c in enumerate(cnts):
        area = float(cv2.contourArea(c))
        if area < 100:                          # only show non-trivial blobs
            continue
        rect = cv2.minAreaRect(c)
        (cx, cy), (rw, rh), ang = rect
        if rw == 0 or rh == 0:
            reasons = ["zero_dim"]
        else:
            aspect = max(rw, rh) / min(rw, rh)
            hull = cv2.convexHull(c)
            hull_area = float(cv2.contourArea(hull)) or 1.0
            solidity = area / hull_area
            x, y, bw, bh = cv2.boundingRect(c)
            reasons = []
            if area < CAND_MIN_AREA_PX:        reasons.append(f"area<{CAND_MIN_AREA_PX}")
            if area > CAND_MAX_AREA_PX:        reasons.append(f"area>{CAND_MAX_AREA_PX}")
            if aspect > CAND_MAX_ASPECT:       reasons.append(f"aspect={aspect:.2f}>{CAND_MAX_ASPECT}")
            if solidity < CAND_MIN_SOLIDITY:   reasons.append(f"solid={solidity:.2f}<{CAND_MIN_SOLIDITY}")
            if (x < EDGE_TOLERANCE_PX or y < EDGE_TOLERANCE_PX
                or x + bw > w - EDGE_TOLERANCE_PX
                or y + bh > h - EDGE_TOLERANCE_PX):
                reasons.append("edge")
        color = (0, 255, 0) if not reasons else (0, 0, 255)
        cv2.drawContours(vis, [c], 0, color, 3)
        label = "OK" if not reasons else ",".join(reasons)
        print(f"  c={i}: centre=({cx:.0f},{cy:.0f}) area={area:.0f} "
              f"size={rw:.0f}x{rh:.0f}  -> {label}")
        cv2.putText(vis, label, (int(cx) - 60, int(cy) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imwrite(str(out / "QQ_all_blobs_classified.jpg"), vis)
    print(f"\nView: {out / 'QQ_all_blobs_classified.jpg'}")


if __name__ == "__main__":
    main()
