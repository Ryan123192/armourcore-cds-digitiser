"""Diagnose why find_paper_bbox returns None on the 2 angled cases."""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

import cv2
from armourcore_cds.phase1.marker_rectify_fast_v2 import find_paper_bbox
from tools.pipeline_dev.corpus import RAW_IMAGES_DIR, make_run_dir

CASES = ["BLUE_PEN_FLAT_03", "BLUE_PENCIL_FLAT_01"]


def main():
    out = make_run_dir("step01_border_detection", "DEBUG_v2_paper")
    print(f"Out: {out.relative_to(REPO)}\n")
    for stem in CASES:
        candidates = list(RAW_IMAGES_DIR.glob(f"{stem}.*"))
        if not candidates:
            print(f"!! missing input for {stem}")
            continue
        img = cv2.imread(str(candidates[0]))
        case_dir = out / stem
        case_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(case_dir / "00_original.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        bbox = find_paper_bbox(img, debug_dir=case_dir)
        print(f"{stem}:  bbox = {bbox}   (image {img.shape[1]}x{img.shape[0]})")
        # Also save grayscale + Otsu for inspection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(str(case_dir / "_gray.jpg"), gray)
        blur = cv2.GaussianBlur(gray, (15, 15), 0)
        otsu_val, binary = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        print(f"  Otsu threshold = {otsu_val}")
        cv2.imwrite(str(case_dir / "_otsu_raw.jpg"), binary)
    print("\nView the 00b_paper_binary.jpg in each case folder.")


if __name__ == "__main__":
    main()
