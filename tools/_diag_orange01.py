"""Diagnose persistent-orange HSV values on BlueColourTest01.

Reads the Phase 1 rectified output (lighting-normalised input to Phase 2)
and the Phase 2 cleaned output.  Pixels that are still orange-ish in the
cleaned image = grid that escaped detection.  Prints HSV stats for those
pixels so we know exactly how to widen the thresholds.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np

from armourcore_cds.phase2.trace_isolation import normalise_lighting

REPO = Path(__file__).parent.parent
OUT  = REPO / "data" / "outputs" / "end_to_end" / "BlueColourTest01"

rect    = cv2.imread(str(OUT / "01_phase1_rectified.png"))
cleaned = cv2.imread(str(OUT / "02_phase2_cleaned.png"))
assert rect is not None and cleaned is not None

# Match dimensions so we can compare
if cleaned.shape[:2] != rect.shape[:2]:
    cleaned = cv2.resize(cleaned, (rect.shape[1], rect.shape[0]),
                         interpolation=cv2.INTER_AREA)

# Re-apply normalise_lighting to match what Phase 2 sees
norm = normalise_lighting(rect)

# Find pixels that are orange-ish in the input but NOT removed in cleaned
hsv_in    = cv2.cvtColor(norm,    cv2.COLOR_BGR2HSV)
gray_in   = cv2.cvtColor(norm,    cv2.COLOR_BGR2GRAY)

# "Still visible orange" = pixel where the cleaned version still
# differs from a uniform paper background in a coloured way.  Define as:
#  - in input the pixel was chromatic (sat > 5) AND in hue 0-40 range
#  - in cleaned the pixel is still chromatic (sat > 5)
hsv_out   = cv2.cvtColor(cleaned, cv2.COLOR_BGR2HSV)

in_orange  = (
    ((hsv_in[..., 0] <= 40) | (hsv_in[..., 0] >= 175))  # broad hue
    & (hsv_in[..., 1] > 5)
    & (hsv_in[..., 2] > 30)
)
out_still_coloured = hsv_out[..., 1] > 8

persistent = in_orange & out_still_coloured

print(f"image shape: {norm.shape}")
print(f"in_orange (broad)         pixels: {int(in_orange.sum()):>8,}")
print(f"out_still_coloured        pixels: {int(out_still_coloured.sum()):>8,}")
print(f"persistent (overlap)      pixels: {int(persistent.sum()):>8,}")
print()

if persistent.sum() == 0:
    print("Nothing persistent found by this heuristic.")
else:
    h = hsv_in[..., 0][persistent]
    s = hsv_in[..., 1][persistent]
    v = hsv_in[..., 2][persistent]
    g = gray_in[persistent]
    print("HSV stats of persistent-orange pixels (input AFTER normalise_lighting):")
    print(f"  H : min={h.min():3d}  p05={np.percentile(h,5):6.1f}  "
          f"med={np.median(h):6.1f}  p95={np.percentile(h,95):6.1f}  max={h.max():3d}")
    print(f"  S : min={s.min():3d}  p05={np.percentile(s,5):6.1f}  "
          f"med={np.median(s):6.1f}  p95={np.percentile(s,95):6.1f}  max={s.max():3d}")
    print(f"  V : min={v.min():3d}  p05={np.percentile(v,5):6.1f}  "
          f"med={np.median(v):6.1f}  p95={np.percentile(v,95):6.1f}  max={v.max():3d}")
    print(f"  Gray : min={g.min():3d}  p05={np.percentile(g,5):6.1f}  "
          f"med={np.median(g):6.1f}  p95={np.percentile(g,95):6.1f}  max={g.max():3d}")

# Also: simulate a 2.5x saturation boost and check
hsv_boost = hsv_in.astype(np.float32).copy()
hsv_boost[..., 1] = np.clip(hsv_boost[..., 1] * 2.5, 0, 255)
hsv_boost = hsv_boost.astype(np.uint8)

# How many of the persistent pixels now pass S>=10 with chroma boost?
if persistent.sum() > 0:
    s_boost = hsv_boost[..., 1][persistent]
    passes = (s_boost >= 10).sum()
    print(f"\nAfter 2.5x sat boost, persistent pixels with S>=10: "
          f"{int(passes):,} / {int(persistent.sum()):,} "
          f"({100.0*passes/persistent.sum():.1f}%)")
    passes20 = (s_boost >= 20).sum()
    print(f"After 2.5x sat boost, persistent pixels with S>=20: "
          f"{int(passes20):,} / {int(persistent.sum()):,} "
          f"({100.0*passes20/persistent.sum():.1f}%)")

# Sample what *pencil ink* looks like for comparison: pixels that are dark
# in the cleaned image (true customer marks left in place).
gray_out  = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
ink = gray_out < 120
if ink.sum() > 0:
    s_ink = hsv_in[..., 1][ink]
    print(f"\nPencil/ink (gray<120 in cleaned)  sat stats:")
    print(f"  S : p05={np.percentile(s_ink,5):6.1f}  med={np.median(s_ink):6.1f}  "
          f"p95={np.percentile(s_ink,95):6.1f}")
    s_ink_b = hsv_boost[..., 1][ink]
    print(f"  S boosted x2.5: p05={np.percentile(s_ink_b,5):6.1f}  "
          f"med={np.median(s_ink_b):6.1f}  p95={np.percentile(s_ink_b,95):6.1f}")
