"""Side-by-side comparison of grid-removal methods focused on the
top-right corner where the production pipeline leaves remnants."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import cv2
import numpy as np

REPO = Path(__file__).parent.parent
ROOT = REPO / "data" / "outputs" / "grid_method_comparison" / "BlueColourTest01"
LATEST = sorted(p for p in ROOT.iterdir() if p.is_dir())[-1]
print(f"reading: {LATEST.name}")

METHODS = ["M0_strong_hsv", "M1_full_current", "M6_curved_bands", "M7_inferred_grid"]


def crop_top_right(img: np.ndarray, frac: float = 0.4) -> np.ndarray:
    H, W = img.shape[:2]
    y2 = int(H * frac)
    x1 = int(W * (1 - frac))
    return img[:y2, x1:]


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return out


cleaned_crops = []
overlay_crops = []
for m in METHODS:
    c = cv2.imread(str(LATEST / m / "cleaned.png"))
    o = cv2.imread(str(LATEST / m / "mask_overlay.png"))
    cleaned_crops.append(label(crop_top_right(c), m + " cleaned"))
    overlay_crops.append(label(crop_top_right(o), m + " mask"))

cleaned_strip = np.hstack(cleaned_crops)
overlay_strip = np.hstack(overlay_crops)
both = np.vstack([overlay_strip, cleaned_strip])

out = LATEST / "_topright_compare.png"
cv2.imwrite(str(out), both)
print(f"wrote: {out.relative_to(REPO)}")
