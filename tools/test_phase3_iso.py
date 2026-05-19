"""Phase 3 isolated-tool repair tester.

Runs repair_orange_gaps on a single isolated-tool PNG from
``data/inputs/phase03_testing/`` and writes diagnostic outputs.

This bypasses the full Phase 1 / Phase 2 pipeline and operates directly
on a thresholded trace image — useful for narrowing down algorithm
behaviour to one tool at a time.

Usage:
    python tools/test_phase3_iso.py IsoToolUnfixed01
    python tools/test_phase3_iso.py IsoToolUnfixed03 --no-open

Outputs (to ``data/outputs/phase03_testing/<name>/``):
    iso_original.png  — input trace mask (binarised)
    iso_filled.png    — repaired mask after centerline completion
    iso_highlight.png — colour overlay:
                          dark grey   = original trace
                          faint blue  = pruned medial-axis skeleton
                          vivid green = Bézier completion fills
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

from armourcore_cds.phase3.vectorise import repair_orange_gaps
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
        description="Run Phase 3 centerline repair on one isolated-tool test image."
    )
    parser.add_argument("name", help="Image name without extension, e.g. IsoToolUnfixed01")
    parser.add_argument("--threshold", type=int, default=170,
                        help="Dark-pixel threshold for trace detection (default: 170).")
    parser.add_argument("--no-open", action="store_true",
                        help="Do not auto-open the highlight image.")
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    in_path  = repo_root / "data" / "inputs"  / "phase03_testing" / f"{args.name}.png"
    out_dir  = repo_root / "data" / "outputs" / "phase03_testing" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"ERROR: {in_path} not found")
        sys.exit(1)

    print(f"\n=== {args.name} ===")
    print(f"Input: {in_path}")

    cleaned_bgr = cv2.imread(str(in_path))
    if cleaned_bgr is None:
        print("ERROR: cv2 could not read image")
        sys.exit(1)

    H, W = cleaned_bgr.shape[:2]
    print(f"Size: {W}x{H}")

    # ------------------------------------------------------------------
    # Run repair at full resolution (these test images are small)
    # ------------------------------------------------------------------
    repaired_mask, completion_mask, skeleton_mask = repair_orange_gaps(
        cleaned_bgr=cleaned_bgr,
        orange_mask=None,
        dark_threshold=args.threshold,
        max_processing_dim=None,
    )

    # ------------------------------------------------------------------
    # Highlight overlay
    # ------------------------------------------------------------------
    h_m, w_m = repaired_mask.shape[:2]
    base = (
        cleaned_bgr if (h_m, w_m) == (H, W)
        else cv2.resize(cleaned_bgr, (w_m, h_m), interpolation=cv2.INTER_AREA)
    )

    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    orig_trace = ((gray < args.threshold).astype(np.uint8)) * 255

    highlight = base.copy()

    # Original trace — darken to grey for contrast
    orig_px = orig_trace > 0
    highlight[orig_px] = (
        highlight[orig_px].astype(np.int32) * 0.55
    ).clip(0, 255).astype(np.uint8)

    # Skeleton (faint blue)
    skel_only = (skeleton_mask > 0) & (completion_mask == 0)
    highlight[skel_only] = (230, 80, 0)

    # Completion (vivid green)
    completion_px = completion_mask > 0
    highlight[completion_px] = (0, 220, 60)

    p_orig      = out_dir / "iso_original.png"
    p_filled    = out_dir / "iso_filled.png"
    p_highlight = out_dir / "iso_highlight.png"

    save_image(p_orig,      orig_trace)
    save_image(p_filled,    repaired_mask)
    save_image(p_highlight, highlight)

    n_completion = int(np.count_nonzero(completion_mask))
    n_skel       = int(np.count_nonzero(skeleton_mask))
    print(f"\nResult: completion={n_completion:,}px  skeleton={n_skel:,}px")
    print(f"  original:  {p_orig}")
    print(f"  filled:    {p_filled}")
    print(f"  highlight: {p_highlight}")
    print(f"    green=Bézier completions  blue=medial-axis skeleton  dark=original trace")

    if not args.no_open:
        _open_file(p_highlight)


if __name__ == "__main__":
    main()
