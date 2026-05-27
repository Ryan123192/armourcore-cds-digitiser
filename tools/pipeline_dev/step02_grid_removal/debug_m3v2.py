"""Diagnose what M3v2 actually classifies as ink.

Saves intermediate images so we can see exactly which pixels are
being included by each criterion.
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
from armourcore_cds.phase2.trace_isolation import normalise_lighting
from tools.pipeline_dev.corpus import RAW_IMAGES_DIR, make_run_dir


CASES = ["BLUE_PEN_FLAT_01", "BLUE_PEN_FLAT_03", "BLUE_PENCIL_FLAT_01"]


def estimate_paper_lab_robust(lab_uint8):
    L = lab_uint8[:, :, 0]
    bright_thresh = np.percentile(L, 95)
    mask = L >= bright_thresh
    chosen = lab_uint8[mask]
    return tuple(float(v) for v in np.median(chosen, axis=0))


def main():
    out_root = make_run_dir("step02_grid_removal", "DEBUG_m3v2")
    print(f"Output: {out_root.relative_to(REPO)}\n")
    for stem in CASES:
        case_path = next(RAW_IMAGES_DIR.glob(f"{stem}.*"))
        img = cv2.imread(str(case_path))
        rect = rectify_with_markers_fast_v4(
            img, paper_w_mm=PAPER_W_MM, paper_h_mm=PAPER_H_MM,
            px_per_mm=DEFAULT_PX_PER_MM,
        ).warped
        case_dir = out_root / stem
        case_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(case_dir / "00_rectified.png"), rect)

        # Method we're debugging: with CLAHE
        norm = normalise_lighting(rect)
        cv2.imwrite(str(case_dir / "01_after_CLAHE.png"), norm)
        lab = cv2.cvtColor(norm, cv2.COLOR_BGR2LAB)
        lab_f = lab.astype(np.float32)
        L = lab_f[:, :, 0]; a = lab_f[:, :, 1]; b = lab_f[:, :, 2]

        paper_lab = estimate_paper_lab_robust(lab)
        Lp, ap, bp = paper_lab
        print(f"\n=== {stem} ===")
        print(f"  paper Lab (with CLAHE):  L={Lp:.1f}  a={ap:.1f}  b={bp:.1f}")

        # Save L channel grayscale to see what 'darker than paper' looks like
        cv2.imwrite(str(case_dir / "02_L_channel.png"), L.astype(np.uint8))

        # Mask: darker_than_paper > 45
        darker = (Lp - L) > 45
        darker_vis = (darker.astype(np.uint8)) * 255
        cv2.imwrite(str(case_dir / "03_darker_than_paper.png"), darker_vis)
        print(f"  darker_than_paper>45  : {int(darker.sum()):>10,} px "
              f"({100.0 * darker.mean():.1f}%)")

        # Mask: neutral_a
        neutral_a = np.abs(a - ap) < 5
        cv2.imwrite(str(case_dir / "04_neutral_a.png"),
                    neutral_a.astype(np.uint8) * 255)
        print(f"  |a-ap|<5              : {int(neutral_a.sum()):>10,} px "
              f"({100.0 * neutral_a.mean():.1f}%)")

        # Mask: neutral_b
        neutral_b = np.abs(b - bp) < 5
        cv2.imwrite(str(case_dir / "05_neutral_b.png"),
                    neutral_b.astype(np.uint8) * 255)
        print(f"  |b-bp|<5              : {int(neutral_b.sum()):>10,} px "
              f"({100.0 * neutral_b.mean():.1f}%)")

        ink_mask = (darker & neutral_a & neutral_b).astype(np.uint8) * 255
        cv2.imwrite(str(case_dir / "06_ink_mask_combined.png"), ink_mask)
        print(f"  combined ink mask     : {int((ink_mask > 0).sum()):>10,} px")

        # Also try WITHOUT CLAHE for comparison
        lab_noclahe = cv2.cvtColor(rect, cv2.COLOR_BGR2LAB).astype(np.float32)
        Lo = lab_noclahe[:, :, 0]; ao = lab_noclahe[:, :, 1]; bo = lab_noclahe[:, :, 2]
        paper_lab_noclahe = estimate_paper_lab_robust(
            lab_noclahe.astype(np.uint8),
        )
        Lp2, ap2, bp2 = paper_lab_noclahe
        print(f"  paper Lab (no CLAHE):    L={Lp2:.1f}  a={ap2:.1f}  b={bp2:.1f}")

        darker_n = (Lp2 - Lo) > 45
        neutral_a_n = np.abs(ao - ap2) < 5
        neutral_b_n = np.abs(bo - bp2) < 5
        ink_n = (darker_n & neutral_a_n & neutral_b_n).astype(np.uint8) * 255
        cv2.imwrite(str(case_dir / "07_ink_mask_NO_CLAHE.png"), ink_n)
        print(f"  combined no-CLAHE     : {int((ink_n > 0).sum()):>10,} px")


if __name__ == "__main__":
    main()
