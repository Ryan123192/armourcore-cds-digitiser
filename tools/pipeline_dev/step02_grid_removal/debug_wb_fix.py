"""Test if gray-world white-balance recovers the orange grid on image 1.

L11 fails on BLUE_PEN_FLAT_01 because the blue camera cast shifts every
hue to ~109 (cyan-blue), so the orange detector finds 0% coverage and
the auto-detector falls to the wrong (BLACK) path.

A gray-world or simple per-channel mean-equalisation WB should pull the
orange back to its true hue range.
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
    normalise_lighting, build_orange_mask, _orange_coverage,
    isolate_trace_candidates,
)
from armourcore_cds.templates.registry import load_template_config
from tools.pipeline_dev.corpus import RAW_IMAGES_DIR, make_run_dir


CASES = ["BLUE_PEN_FLAT_01", "BLUE_PEN_FLAT_02",
         "BLUE_PEN_FLAT_03", "BLUE_PENCIL_FLAT_01"]


def gray_world_wb(image_bgr: np.ndarray) -> np.ndarray:
    """Classic gray-world: scale each channel so its mean equals the
    overall luminance mean."""
    img = image_bgr.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0)  # B, G, R
    target = float(means.mean())
    scale = target / np.maximum(means, 1.0)
    out = img * scale[None, None, :]
    return np.clip(out, 0, 255).astype(np.uint8)


def paper_wb(image_bgr: np.ndarray) -> np.ndarray:
    """Estimate paper colour from top-5% brightest pixels and scale each
    channel so paper becomes neutral gray."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    thr = np.percentile(gray, 95)
    mask = gray >= thr
    paper = image_bgr[mask].astype(np.float32)
    means = paper.mean(axis=0)  # B, G, R of paper
    target = float(means.mean())
    scale = target / np.maximum(means, 1.0)
    out = image_bgr.astype(np.float32) * scale[None, None, :]
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    out = make_run_dir("step02_grid_removal", "DEBUG_wb_fix")
    print(f"Output: {out.relative_to(REPO)}\n")
    template = load_template_config("cds_colour_test_260x350")
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

        # Three WB approaches before CLAHE
        gw   = gray_world_wb(rect)
        pwb  = paper_wb(rect)
        cv2.imwrite(str(case_dir / "01_gray_world.png"),  gw)
        cv2.imwrite(str(case_dir / "02_paper_wb.png"),   pwb)

        print(f"\n=== {stem} ===")
        for label, src in [("rectified", rect),
                           ("gray_world", gw),
                           ("paper_wb",   pwb)]:
            norm = normalise_lighting(src)
            cov  = _orange_coverage(norm)
            mask = build_orange_mask(norm, sat_min=25, chroma_boost=1.0)
            print(f"  {label:<12s}  probe={cov*100:6.2f}%  "
                  f"orange_mask={100*(mask>0).mean():6.3f}%")
            cv2.imwrite(str(case_dir / f"orange_mask_{label}.png"), mask)

        # Run the full L11 pipeline on paper-WB input
        pwb_norm = normalise_lighting(pwb)
        result = isolate_trace_candidates(pwb_norm, template)
        cv2.imwrite(str(case_dir / "L11_on_paper_wb_cleaned.png"),
                    result.cleaned_bgr)


if __name__ == "__main__":
    main()
