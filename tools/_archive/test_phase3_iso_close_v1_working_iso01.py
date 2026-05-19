"""Phase 3 isolated-tool repair via CLOSE + medial_axis.

This implements a fundamentally different (much simpler) approach to gap
filling: morphological CLOSE the trace first to merge all stubs into a
single continuous band, then take the medial axis of that band.  The
medial axis IS the customer's original centerline — gaps are bridged
automatically by the geometry, no endpoint pairing required.

Why this works
--------------
A tool outline drawn with thickness W, with internal gaps of size G < 2·W,
becomes a single continuous closed band after MORPH_CLOSE with a kernel
of size > G.  The medial axis of any closed band is its centerline —
which is what we want.

Crucially this handles ANY topology:
  * Convex tools  (pliers / heads): single cycle skeleton
  * Non-convex tools (wishbones / Y-shapes): single cycle skeleton
  * Multi-loop tools (handle + stem): multiple cycles joined at
    intersections — preserved by the medial axis

Limitations:
  * Closing kernel must be > largest within-tool gap, but < smallest
    inter-tool spacing.  Tunable per scan.
  * Slight bulges where the closing kernel modifies the trace shape near
    junctions — usually negligible.

Outputs (to ``data/outputs/phase03_testing/<name>_close/``):
    iso_original.png       — input trace mask (binary)
    iso_closed_trace.png   — trace after MORPH_CLOSE (gaps bridged)
    iso_filled.png         — repaired mask (original ∪ completion)
    iso_highlight.png      — overlay (dark=original, blue=skeleton, green=fill)
    iso_skeleton.png       — pure medial axis (the recovered centerline)

Usage:
    python tools/test_phase3_iso_close.py IsoToolUnfixed01
    python tools/test_phase3_iso_close.py IsoToolUnfixed01 --close-kernel 35
"""
from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np
from skimage.morphology import medial_axis

from armourcore_cds.utils.image_ops import save_image


def _open_file(path: Path) -> None:
    try:
        if platform.system() == "Windows":
            subprocess.Popen(["start", "", str(path)], shell=True)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as exc:
        print(f"  [open] {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLOSE + medial-axis gap repair (simple approach)."
    )
    parser.add_argument("name", help="Image name without extension")
    parser.add_argument("--threshold", type=int, default=170,
                        help="Dark-pixel threshold (default: 170).")
    parser.add_argument("--close-kernel", type=int, default=25, dest="close_kernel",
                        help="MORPH_CLOSE ellipse kernel size in px (default: 25). "
                             "Must be > largest within-tool gap.")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    in_path  = repo_root / "data" / "inputs"  / "phase03_testing" / f"{args.name}.png"
    out_dir  = repo_root / "data" / "outputs" / "phase03_testing" / f"{args.name}_close"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"ERROR: {in_path} not found")
        sys.exit(1)

    print(f"\n=== {args.name} (CLOSE + medial_axis method) ===")

    cleaned_bgr = cv2.imread(str(in_path))
    if cleaned_bgr is None:
        print("ERROR: cv2 could not read image")
        sys.exit(1)

    H, W = cleaned_bgr.shape[:2]
    print(f"Size: {W}x{H}")

    # ------------------------------------------------------------------
    # 1. Threshold → trace binary
    # ------------------------------------------------------------------
    gray = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
    trace_binary = ((gray < args.threshold).astype(np.uint8)) * 255
    if not np.any(trace_binary > 0):
        print("ERROR: no trace pixels detected (try lowering --threshold)")
        sys.exit(1)
    print(f"Trace pixels: {int(np.count_nonzero(trace_binary)):,}")

    # ------------------------------------------------------------------
    # 2. MORPH_CLOSE — bridge all gaps into one continuous band
    # ------------------------------------------------------------------
    k = max(3, args.close_kernel | 1)   # ensure odd
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed_trace = cv2.morphologyEx(trace_binary, cv2.MORPH_CLOSE, kernel)
    n_added_close = int(
        np.count_nonzero(cv2.bitwise_and(closed_trace, cv2.bitwise_not(trace_binary)))
    )
    print(f"Closing kernel: {k}px ellipse")
    print(f"Pixels filled by CLOSE: {n_added_close:,}")

    # ------------------------------------------------------------------
    # 3. Medial axis of CLOSED band — this IS the centerline of the
    #    customer's original outline, with gaps bridged automatically.
    # ------------------------------------------------------------------
    skel_closed_bool, dist_closed = medial_axis(
        closed_trace > 0, return_distance=True
    )
    skel_closed = skel_closed_bool.astype(np.uint8) * 255

    # Medial axis of ORIGINAL trace, used for local thickness reference
    skel_orig_bool, dist_orig = medial_axis(trace_binary > 0, return_distance=True)

    # Local trace thickness — diameter at each medial-axis pixel = 2 × dist.
    # Use the median across the original skeleton as a robust estimate.
    thicknesses = dist_orig[skel_orig_bool] * 2.0
    thicknesses = thicknesses[thicknesses >= 2.0]
    if thicknesses.size > 0:
        avg_thickness = float(np.median(thicknesses))
    else:
        avg_thickness = 4.0
    avg_thickness = float(np.clip(avg_thickness, 2.0, 30.0))
    print(f"Avg trace thickness (median): {avg_thickness:.1f}px")

    # ------------------------------------------------------------------
    # 4. Render the skeleton segments that fall in GAP regions as a
    #    thickened line.  Pixels in the original trace are unchanged.
    # ------------------------------------------------------------------
    skel_in_gaps = cv2.bitwise_and(skel_closed, cv2.bitwise_not(trace_binary))

    completion = np.zeros_like(trace_binary)
    gap_ys, gap_xs = np.where(skel_in_gaps > 0)
    radius = max(1, int(round(avg_thickness / 2.0)))
    for x, y in zip(gap_xs.tolist(), gap_ys.tolist()):
        cv2.circle(completion, (int(x), int(y)), radius, 255, -1)

    # Make completion strictly additive
    completion = cv2.bitwise_and(completion, cv2.bitwise_not(trace_binary))

    repaired = cv2.bitwise_or(trace_binary, completion)

    n_completion = int(np.count_nonzero(completion))
    n_skel_gaps = int(np.count_nonzero(skel_in_gaps))
    print(f"Gap-skeleton pixels: {n_skel_gaps:,}")
    print(f"Completion pixels: {n_completion:,}")

    # ------------------------------------------------------------------
    # 5. Diagnostic overlay
    # ------------------------------------------------------------------
    highlight = cleaned_bgr.copy()

    # Original trace — darken to grey
    orig_px = trace_binary > 0
    highlight[orig_px] = (
        highlight[orig_px].astype(np.int32) * 0.55
    ).clip(0, 255).astype(np.uint8)

    # Centerline (skeleton of closed band) — faint blue
    skel_only = (skel_closed > 0) & (completion == 0)
    highlight[skel_only] = (230, 80, 0)

    # Completion fills — vivid green
    highlight[completion > 0] = (0, 220, 60)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    p_orig      = out_dir / "iso_original.png"
    p_closed    = out_dir / "iso_closed_trace.png"
    p_filled    = out_dir / "iso_filled.png"
    p_highlight = out_dir / "iso_highlight.png"
    p_skeleton  = out_dir / "iso_skeleton.png"

    save_image(p_orig,      trace_binary)
    save_image(p_closed,    closed_trace)
    save_image(p_filled,    repaired)
    save_image(p_highlight, highlight)
    save_image(p_skeleton,  skel_closed)

    print(f"\nOutputs saved to: {out_dir}")
    print(f"  highlight: {p_highlight}")

    if not args.no_open:
        _open_file(p_highlight)


if __name__ == "__main__":
    main()
