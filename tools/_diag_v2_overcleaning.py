"""Diagnose why tool ink is being eaten on the new V2 photos.

Compare HSV / LAB stats of the rectified input on:
- Likely tool-ink pixels (darker pixels in the design area)
- Likely grid pixels (orange-tinted pixels)
- Likely paper pixels (lightest pixels)

If tool ink has high boosted-saturation, it's being caught by the HSV mask
and removed by Phase 2.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import cv2
import numpy as np

from armourcore_cds.phase2.trace_isolation import (
    normalise_lighting, build_orange_mask, build_combined_orange_mask,
    detect_per_line_grid_bands,
)

REPO = Path(__file__).parent.parent

def diag(name: str):
    p1_run = sorted((REPO / "outputs" / "runs").glob(f"*{name}*"))[-1]
    rect = cv2.imread(str(p1_run / "scaled_design_area.png"))
    norm = normalise_lighting(rect)
    gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(norm, cv2.COLOR_BGR2HSV)
    lab  = cv2.cvtColor(norm, cv2.COLOR_BGR2LAB)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    L, A, B = lab[..., 0], lab[..., 1].astype(np.int16), lab[..., 2]

    print(f"\n=== {name} ===")
    print(f"shape: {norm.shape}")
    # Tool ink = the darkest 5% of pixels
    dark_thresh = float(np.percentile(gray, 5))
    tool_px = gray <= dark_thresh
    # Grid orange = warm pixels (a* > 130)
    orange_px = A > 130
    # Paper = lightest 60% of pixels
    paper_thresh = float(np.percentile(gray, 60))
    paper_px = gray >= paper_thresh

    def stats(label, mask):
        if mask.sum() == 0:
            print(f"  {label:<14} (empty)")
            return
        print(f"  {label:<14} n={int(mask.sum()):>9,}  "
              f"gray med={np.median(gray[mask]):5.0f}  "
              f"H med={np.median(H[mask]):3.0f}  "
              f"S med={np.median(S[mask]):3.0f} p95={np.percentile(S[mask], 95):3.0f}  "
              f"a*-128 med={int(np.median(A[mask])) - 128:+3d}")

    stats("tool ink",  tool_px)
    stats("grid orange", orange_px)
    stats("paper",     paper_px)

    # What does build_orange_mask catch?
    hsv_mask = build_orange_mask(norm)
    combined = build_combined_orange_mask(norm)
    # How much of the tool-ink mask is caught by HSV?
    tool_caught = (tool_px & (hsv_mask > 0)).sum()
    tool_total = int(tool_px.sum())
    print(f"  HSV mask catches {tool_caught:,}/{tool_total:,} tool-ink pixels "
          f"({100.0 * tool_caught / max(1, tool_total):.1f}%)")
    grid_caught = (orange_px & (hsv_mask > 0)).sum()
    grid_total = int(orange_px.sum())
    print(f"  HSV mask catches {grid_caught:,}/{grid_total:,} orange pixels "
          f"({100.0 * grid_caught / max(1, grid_total):.1f}%)")

    # And the per-line bands
    bands = detect_per_line_grid_bands(combined)
    print(f"  bands cover {bands.mean() / 2.55:.1f}% of image")

for n in ("V2BlueColourTest01", "V2BlueColourTest02", "V2BlueColourTest03"):
    diag(n)
