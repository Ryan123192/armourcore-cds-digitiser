"""Visualise the Hough band mask, near-band extras, and the strong mask
for BlueColourTest01 to see exactly what is and isn't being caught.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np

from armourcore_cds.phase2.trace_isolation import (
    normalise_lighting, build_orange_mask, detect_grid_line_bands,
    build_relaxed_orange_in_bands, catch_orange_near_bands,
)

REPO = Path(__file__).parent.parent
CASE_ROOT = REPO / "data" / "outputs" / "end_to_end" / "BlueColourTest01"
sub = sorted(p for p in CASE_ROOT.iterdir() if p.is_dir())[-1]
print(f"writing to: {sub.name}")

rect = cv2.imread(str(sub / "01_phase1_rectified.png"))
img = normalise_lighting(rect)

strong = build_orange_mask(img)
H, W = strong.shape
bands = detect_grid_line_bands(strong, min_line_length_px=max(150, int(0.10 * min(H, W))))
relaxed_in = build_relaxed_orange_in_bands(img, bands)
near = catch_orange_near_bands(img, bands)

# Three-panel overlay
def overlay(base, mask, colour):
    out = base.copy()
    out[mask > 0] = colour
    return out

cv2.imwrite(str(sub / "_dbg_strong.png"),   overlay(img, strong, (0, 0, 255)))   # red
cv2.imwrite(str(sub / "_dbg_bands.png"),    overlay(img, bands,  (255, 0, 0)))   # blue
cv2.imwrite(str(sub / "_dbg_relaxed.png"),  overlay(img, relaxed_in, (0, 255, 0)))  # green
cv2.imwrite(str(sub / "_dbg_nearband.png"), overlay(img, near, (0, 255, 255)))   # yellow

# Combined removal mask
all_removed = strong | relaxed_in | near
cv2.imwrite(str(sub / "_dbg_all_removed.png"), overlay(img, all_removed, (0, 0, 255)))

print("strong px:        ", int((strong > 0).sum()))
print("bands px (full):  ", int((bands > 0).sum()))
print("relaxed_in_band:  ", int((relaxed_in > 0).sum()))
print("near_band_extras: ", int((near > 0).sum()))
print("combined total:   ", int((all_removed > 0).sum()))

# Crop to the boundary-line text area for closer inspection
# from the diagnostic image, it's roughly at x~100, y~700..1200 in the
# rectified image dimensions (depends on actual dims)
H, W = img.shape[:2]
# look for the text by searching the persistent area
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
in_orange = ((hsv[..., 0] <= 40) & (hsv[..., 1] > 5))
# columns with high orange density (= vertical text/lines)
cols = (in_orange.sum(axis=0))
top_cols = np.argsort(cols)[-30:]
print("top 30 columns by orange density:", sorted(top_cols.tolist()))
