"""Save a visualisation of persistent-orange pixels on case 01."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np
from armourcore_cds.phase2.trace_isolation import normalise_lighting

REPO = Path(__file__).parent.parent
CASE_ROOT = REPO / "data" / "outputs" / "end_to_end" / "BlueColourTest01"
# Newest run subfolder (sorted lexically: timestamps sort chronologically)
subfolders = sorted(p for p in CASE_ROOT.iterdir() if p.is_dir())
if not subfolders:
    raise SystemExit(f"No iteration subfolders in {CASE_ROOT}")
OUT = subfolders[-1]
print(f"[diag] inspecting: {OUT.relative_to(REPO)}")

rect    = cv2.imread(str(OUT / "01_phase1_rectified.png"))
cleaned = cv2.imread(str(OUT / "02_phase2_cleaned.png"))
if cleaned.shape[:2] != rect.shape[:2]:
    cleaned = cv2.resize(cleaned, (rect.shape[1], rect.shape[0]),
                         interpolation=cv2.INTER_AREA)

norm    = normalise_lighting(rect)
hsv_in  = cv2.cvtColor(norm, cv2.COLOR_BGR2HSV)
hsv_out = cv2.cvtColor(cleaned, cv2.COLOR_BGR2HSV)

in_orange = (
    ((hsv_in[..., 0] <= 40) | (hsv_in[..., 0] >= 175))
    & (hsv_in[..., 1] > 5)
    & (hsv_in[..., 2] > 30)
)
# A pixel only counts as "persistent" if the cleaned version retained
# MOST of its chromatic strength — i.e., orange was not really removed.
# A pixel that was filled with paper-blend tint (S drops from 80 -> 12)
# is technically still "chromatic" but is in fact removed.  Require the
# cleaned saturation to be >= 50% of the original AND still > 15.
sat_in  = hsv_in[..., 1].astype(np.int32)
sat_out = hsv_out[..., 1].astype(np.int32)
retained_fraction = sat_out / np.maximum(sat_in, 1)
persistent = in_orange & (sat_out > 15) & (retained_fraction > 0.5)

vis = norm.copy()
vis[persistent] = (0, 255, 255)   # bright yellow where orange persisted

cv2.imwrite(str(OUT / "_diag_persistent_orange.png"), vis)
print(f"saved: {OUT / '_diag_persistent_orange.png'}")
print(f"persistent pixel count: {int(persistent.sum()):,}")
