"""Run Phase 2 only on a case and save a clean before/after image.

No Phase 3 — focuses on verifying grid removal alone.

    python tools/test_grid_removal_only.py BlueColourTest01

Output:
    data/outputs/grid_only/<case>/<timestamp>_<tag>/
        before.png            # lighting-normalised rectified input
        after.png             # grid-removed
        before_after.png      # side-by-side
        before_after_topright.png   # zoomed top-right corner
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np

from armourcore_cds.phase2.trace_isolation import (
    isolate_trace_candidates, normalise_lighting,
    build_combined_orange_mask, build_orange_mask, build_lab_orange_mask,
    detect_per_line_grid_bands,
)
from armourcore_cds.templates.registry import load_template_config

REPO = Path(__file__).parent.parent


def _label(img: np.ndarray, text: str, scale: float = 0.8) -> np.ndarray:
    out = img.copy()
    H, W = out.shape[:2]
    band_h = int(48 * scale)
    cv2.rectangle(out, (0, 0), (W, band_h), (0, 0, 0), -1)
    cv2.putText(out, text, (12, band_h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _template_for(name: str) -> str:
    n = name.lower()
    if "bluecolour" in n or "colourtest" in n:
        return "cds_colour_test_260x350"
    if "xlarge" in n:
        return "cds_xlarge_500x900"
    return "cds_regular_500x600"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", default="BlueColourTest01", nargs="?")
    ap.add_argument("--tag", default="hybrid")
    args = ap.parse_args()

    # Find the latest Phase 1 run for this case
    candidates = sorted((REPO / "outputs" / "runs").glob(f"*{args.name}*"))
    if not candidates:
        sys.exit(f"No Phase 1 runs found for {args.name}")
    p1_run = candidates[-1]
    rect_path = p1_run / "scaled_design_area.png"
    if not rect_path.exists():
        sys.exit(f"Missing {rect_path}")

    out_root = (REPO / "data" / "outputs" / "grid_only" / args.name
                / f"{time.strftime('%Y%m%d-%H%M%S')}_{args.tag}")
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"output: {out_root.relative_to(REPO)}")

    img = cv2.imread(str(rect_path))
    template = load_template_config(_template_for(args.name))

    # "Before" = lighting-normalised but not grid-removed (matches what
    # isolate_trace_candidates sees internally as its input).
    before = normalise_lighting(img)

    # ---- Visualise the band mask itself ----------------------------------
    # Run the band-detection pipeline standalone (mirrors what
    # isolate_trace_candidates does internally) so we can see the curved
    # polynomial fit overlaid on the input.
    hsv_only_mask  = build_orange_mask(before)
    lab_only_mask  = build_lab_orange_mask(before)
    combined_mask  = build_combined_orange_mask(before)
    bands_mask     = detect_per_line_grid_bands(combined_mask)

    def _overlay(base, mask, colour, alpha=0.55):
        out = base.copy()
        if not np.any(mask):
            return out
        paint = np.zeros_like(base)
        paint[:] = colour
        blend = cv2.addWeighted(out, 1.0 - alpha, paint, alpha, 0)
        return np.where(mask[..., None] > 0, blend, out)

    # Visualisations: each shows the band/detection in coloured overlay on
    # the lighting-normalised input.
    cv2.imwrite(str(out_root / "viz_hsv_detection.png"),
                _overlay(before, hsv_only_mask, (0, 0, 255)))     # red
    cv2.imwrite(str(out_root / "viz_lab_detection.png"),
                _overlay(before, lab_only_mask, (255, 0, 0)))     # blue
    cv2.imwrite(str(out_root / "viz_combined_detection.png"),
                _overlay(before, combined_mask, (0, 255, 255)))   # yellow
    cv2.imwrite(str(out_root / "viz_curved_bands.png"),
                _overlay(before, bands_mask, (0, 200, 0)))        # green

    t0 = time.time()
    result = isolate_trace_candidates(img, template)
    elapsed = time.time() - t0
    after = result.cleaned_bgr

    # Save individuals
    cv2.imwrite(str(out_root / "before.png"), before)
    cv2.imwrite(str(out_root / "after.png"),  after)

    # Side-by-side
    if after.shape != before.shape:
        after_resized = cv2.resize(after, (before.shape[1], before.shape[0]))
    else:
        after_resized = after
    side_by_side = np.hstack([
        _label(before, "BEFORE  (lighting-normalised rectified)"),
        _label(after_resized, "AFTER  (grid removed)"),
    ])
    cv2.imwrite(str(out_root / "before_after.png"), side_by_side)

    # Top-right crop (problem area)
    H, W = before.shape[:2]
    y2 = int(H * 0.45)
    x1 = int(W * 0.55)
    crop_before = before[:y2, x1:]
    crop_after  = after_resized[:y2, x1:]
    tr = np.hstack([
        _label(crop_before, "BEFORE  top-right"),
        _label(crop_after,  "AFTER  top-right"),
    ])
    cv2.imwrite(str(out_root / "before_after_topright.png"), tr)

    print(f"  Phase 2: {elapsed:.1f}s")
    print(f"  wrote: before.png, after.png, before_after.png, "
          f"before_after_topright.png")


if __name__ == "__main__":
    main()
