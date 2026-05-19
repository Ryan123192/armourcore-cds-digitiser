"""Check whether residual 'grid' in the cleaned image is actual orange
that wasn't removed, or just tool-ink dashes at band crossings (which
form a grid-shaped illusion).

Approach: pick an area of the image with NO tools (e.g., a small gap
between rows of tools), zoom way in, and side-by-side it with the same
crop from before+after.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import cv2
import numpy as np

REPO = Path(__file__).parent.parent
ROOT = REPO / "data" / "outputs" / "grid_only" / "BlueColourTest01"
LATEST = sorted(p for p in ROOT.iterdir() if p.is_dir())[-1]
print(f"reading: {LATEST.name}")

before = cv2.imread(str(LATEST / "before.png"))
after  = cv2.imread(str(LATEST / "after.png"))

if after.shape != before.shape:
    after = cv2.resize(after, (before.shape[1], before.shape[0]))

H, W = before.shape[:2]

# Three zoom regions: top-right corner, mid empty gap, bottom-left
regions = {
    "topright":   (int(H * 0.0),  int(H * 0.20),  int(W * 0.78), int(W * 1.00)),
    "midgap":     (int(H * 0.55), int(H * 0.72),  int(W * 0.20), int(W * 0.50)),
    "bottomright":(int(H * 0.80), int(H * 1.00),  int(W * 0.78), int(W * 1.00)),
}

def label(img, txt, scale=0.7):
    out = img.copy()
    H_, W_ = out.shape[:2]
    band = int(36 * scale)
    cv2.rectangle(out, (0, 0), (W_, band), (0, 0, 0), -1)
    cv2.putText(out, txt, (8, band - 10), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), 2, cv2.LINE_AA)
    return out

panels = []
for name, (y1, y2, x1, x2) in regions.items():
    b_crop = before[y1:y2, x1:x2]
    a_crop = after [y1:y2, x1:x2]
    # 2x upscale for clarity
    b_zoom = cv2.resize(b_crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    a_zoom = cv2.resize(a_crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    panel = np.hstack([
        label(b_zoom, f"BEFORE  {name}"),
        label(a_zoom, f"AFTER  {name}"),
    ])
    panels.append(panel)

# Stack panels vertically
max_w = max(p.shape[1] for p in panels)
panels_padded = []
for p in panels:
    if p.shape[1] < max_w:
        pad = np.full((p.shape[0], max_w - p.shape[1], 3), 255, dtype=np.uint8)
        p = np.hstack([p, pad])
    panels_padded.append(p)
out = np.vstack(panels_padded)

cv2.imwrite(str(LATEST / "_zoomed_residual_check.png"), out)
print(f"wrote: {LATEST / '_zoomed_residual_check.png'}")
print(f"shape: {out.shape}")

# Also save a chromaticity-only view of the cleaned image: any remaining
# orange a* should show up as bright pixels
lab = cv2.cvtColor(after, cv2.COLOR_BGR2LAB)
a = lab[..., 1].astype(np.int16)
chrom = np.clip((a - 128) * 8, 0, 255).astype(np.uint8)
heatmap = cv2.applyColorMap(chrom, cv2.COLORMAP_HOT)
overlay = cv2.addWeighted(after, 0.5, heatmap, 0.5, 0)
cv2.imwrite(str(LATEST / "_chromaticity_heatmap.png"), overlay)
print(f"wrote chromaticity heatmap")
