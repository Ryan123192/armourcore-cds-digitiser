"""Quick iso check for variants A and E."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np

from test_phase3_iso_vectorise import (
    _avg_trace_thickness, render_uniform_band, find_inside_edge_contours,
)
from test_phase3_batch_experiments import (
    adaptive_baseline, adaptive_A_bridge_count,
    adaptive_C_endpoint_quality, adaptive_E_same_side,
)

REPO = Path(__file__).parent.parent

VARIANTS = {
    "baseline": adaptive_baseline,
    "A": adaptive_A_bridge_count,
    "C": adaptive_C_endpoint_quality,
    "E": adaptive_E_same_side,
}

print(f"{'case':<22} {'baseline':<10} {'A':<10} {'C':<10} {'E':<10}")
print("-" * 65)

for case in ["IsoToolUnfixed01", "IsoToolUnfixed02",
             "IsoToolUnfixed03", "IsoToolUnfixed04"]:
    img = cv2.imread(str(REPO / "data" / "inputs" / "phase03_testing"
                         / f"{case}.png"))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    trace = ((gray < 170).astype(np.uint8)) * 255
    bt = max(2, int(round(_avg_trace_thickness(trace))))

    line = f"{case:<22}"
    for vname, vfn in VARIANTS.items():
        try:
            cl, n_br, kk, ep, _, bm = vfn(
                trace, bt, bridge_max_dist_px=250,
                min_loop_abs_px=500, min_loop_rel=0.05,
            )
            band = render_uniform_band(cl, bt)
            loops = find_inside_edge_contours(
                band, min_area_abs_px=500, min_area_relative=0.05,
                bridge_mask=bm,
            )
            n_loops = len(loops)
            line += f" {n_loops}/{n_br:>2}b/k{kk:<3}"
        except Exception as e:
            line += f" ERR        "
    print(line)
