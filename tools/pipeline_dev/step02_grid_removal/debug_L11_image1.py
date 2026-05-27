"""Diagnose why L11 baseline passes BLUE_PEN_FLAT_01 through ~unchanged.

Symptom: cleaned output of M0_L11_baseline on image 1 looks ~blue, like
the rectified input.  Possible causes:
  1. _orange_coverage probe < 0.005, falls to BLACK path (does nothing
     useful for a blue-tinted orange grid).
  2. Orange path runs but HSV hue threshold misses the blue-shifted grid.
  3. The gray>=160 clamp doesn't fire because blue tint keeps gray < 160.
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
from armourcore_cds.phase2.trace_isolation import (
    normalise_lighting, _orange_coverage, build_orange_mask,
    paper_blended_fill, normalise_paper_to_white,
)
from tools.pipeline_dev.corpus import RAW_IMAGES_DIR, make_run_dir


CASES = ["BLUE_PEN_FLAT_01", "BLUE_PEN_FLAT_02",
         "BLUE_PEN_FLAT_03", "BLUE_PENCIL_FLAT_01"]


def main():
    out = make_run_dir("step02_grid_removal", "DEBUG_L11_failure")
    print(f"Output: {out.relative_to(REPO)}\n")
    for stem in CASES:
        case_path = next(RAW_IMAGES_DIR.glob(f"{stem}.*"))
        img = cv2.imread(str(case_path))
        rect = rectify_with_markers_fast_v4(
            img, paper_w_mm=PAPER_W_MM, paper_h_mm=PAPER_H_MM,
            px_per_mm=DEFAULT_PX_PER_MM,
        ).warped
        case_dir = out / stem
        case_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(case_dir / "00_rectified.png"), rect)

        # Step 1: lighting normalisation (CLAHE)
        norm = normalise_lighting(rect)
        cv2.imwrite(str(case_dir / "01_after_CLAHE.png"), norm)

        # Step 2: orange coverage probe
        coverage = _orange_coverage(norm)
        chose_orange = coverage >= 0.005
        print(f"\n=== {stem} ===")
        print(f"  orange coverage probe = {coverage*100:.3f}%  "
              f"-> path: {'ORANGE' if chose_orange else 'BLACK (!)'}")

        # Step 3: HSV mask (the L11 detector)
        hsv = cv2.cvtColor(norm, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        print(f"  hue median={np.median(H):.0f} "
              f"sat median={np.median(S):.0f} val median={np.median(V):.0f}")
        # Look at orange-presumed pixels (paper-ish brightness, color)
        bright_color = (V > 150) & (S > 25)
        if bright_color.sum() > 0:
            hues = H[bright_color]
            print(f"  hue of bright-coloured pixels: "
                  f"min={hues.min()} p25={int(np.percentile(hues,25))} "
                  f"median={int(np.median(hues))} p75={int(np.percentile(hues,75))} "
                  f"max={hues.max()}  (L11 keeps 0..40)")

        strong_orange = build_orange_mask(norm, sat_min=25, chroma_boost=1.0)
        pct = 100.0 * (strong_orange > 0).mean()
        cv2.imwrite(str(case_dir / "02_HSV_orange_mask.png"), strong_orange)
        print(f"  HSV orange mask coverage: {pct:.3f}%")

        # Step 4: paper-blended fill + normalise paper to white
        cleaned = paper_blended_fill(norm, strong_orange)
        cleaned = normalise_paper_to_white(cleaned)
        cv2.imwrite(str(case_dir / "03_after_paper_blend_and_white_norm.png"), cleaned)

        # Step 5: hard clamp at gray>=160
        gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
        n_clamped = int((gray >= 160).sum())
        print(f"  gray>=160 hits: {n_clamped:,} px "
              f"({100.0*n_clamped/gray.size:.1f}%)")
        # Histogram quartiles to see if paper is actually below 160
        print(f"  gray after paper-norm: p10={int(np.percentile(gray,10))} "
              f"p50={int(np.percentile(gray,50))} "
              f"p90={int(np.percentile(gray,90))} "
              f"max={int(gray.max())}")
        cleaned2 = cleaned.copy()
        cleaned2[gray >= 160] = (255, 255, 255)
        cv2.imwrite(str(case_dir / "04_after_white_clamp.png"), cleaned2)


if __name__ == "__main__":
    main()
