"""Phase 3 repair-pass tester.

Runs only the orange-gap repair step and exports diagnostic PNGs for visual
analysis — no vectorisation.  Use this to tune --orange-dilation before
committing to a full Phase 3 run.

Outputs (saved to <phase2_dir>/phase3/repair_test/):
  repair_original.png  — Phase 2 trace mask as-is
  repair_filled.png    — trace mask after centerline-graph repair
  repair_highlight.png — cleaned raster with diagnostic colour overlay:
                         original trace = dark grey
                         medial-axis    = faint blue  (BGR 230, 80,  0)
                         Bézier fills   = vivid green (BGR   0, 220, 60)

Usage:
    python tools/test_phase3_repair.py --filter BlueColourTest00
    python tools/test_phase3_repair.py --filter BlueColourTest00 --dilation 35
    python tools/test_phase3_repair.py --filter BlueColourTest00 --dilation 50
"""
from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np

from armourcore_cds.phase3.vectorise import repair_orange_gaps
from armourcore_cds.utils.image_ops import save_image


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _latest_phase1_run(runs_dir: Path, stem: str) -> Path | None:
    candidates = sorted(
        (d for d in runs_dir.iterdir() if d.is_dir() and d.name.endswith(stem)),
        reverse=True,
    )
    return candidates[0] if candidates else None


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


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test Phase 3 orange-gap repair — outputs diagnostic PNGs only."
    )
    parser.add_argument("--phase1-dir", default="outputs/runs", dest="phase1_dir")
    parser.add_argument("--filter", default="",
                        help="Only process stems containing this string.")
    parser.add_argument("--bridge", type=int, default=20, dest="bridge",
                        help="Bridge half-width per pass in full-res pixels (default: 20 ~ 1.3 mm).")
    parser.add_argument("--passes", type=int, default=4, dest="passes",
                        help="Extra iterative passes after the first (default: 4). "
                             "Total reach = (passes+1) x bridge_px.")
    parser.add_argument("--threshold", type=int, default=170, dest="threshold",
                        help="Dark pixel threshold for trace detection (default: 170).")
    parser.add_argument("--no-open", action="store_true", dest="no_open")
    args = parser.parse_args()

    runs_dir = Path(args.phase1_dir)
    if not runs_dir.exists():
        print(f"ERROR: {runs_dir} not found")
        sys.exit(1)

    all_stems: set[str] = set()
    for d in runs_dir.iterdir():
        if d.is_dir():
            parts = d.name.rsplit("_", 1)
            if len(parts) == 2:
                all_stems.add(parts[1])

    stems = sorted(
        s for s in all_stems
        if (args.filter.lower() in s.lower() if args.filter else True)
    )
    if not stems:
        print("No matching runs found.")
        sys.exit(0)

    print(f"\nPhase 3 repair test — {len(stems)} image(s)  bridge={args.bridge}px  passes=1+{args.passes}")
    print("=" * 70)

    for stem in stems:
        run_dir = _latest_phase1_run(runs_dir, stem)
        if run_dir is None:
            print(f"  {stem}: SKIP — no Phase 1 run")
            continue

        phase2_dir = run_dir / "phase2"
        cleaned_path  = phase2_dir / "phase2_cleaned_raster.png"
        orange_path   = phase2_dir / "phase2_orange_mask.png"
        orig_mask_path = phase2_dir / "phase2_trace_candidate_mask.png"

        missing = [p for p in [cleaned_path, orange_path, orig_mask_path] if not p.exists()]
        if missing:
            print(f"  {stem}: SKIP — missing {[p.name for p in missing]}")
            continue

        print(f"  {stem}: loading images...")
        cleaned_bgr  = cv2.imread(str(cleaned_path))
        orange_mask  = cv2.imread(str(orange_path), cv2.IMREAD_GRAYSCALE)
        orig_mask    = cv2.imread(str(orig_mask_path), cv2.IMREAD_GRAYSCALE)

        if cleaned_bgr is None or orange_mask is None or orig_mask is None:
            print(f"  {stem}: FAIL — cv2 read error")
            continue

        t0 = time.time()

        # --- run repair ------------------------------------------------------
        repaired_mask, completion_mask, skeleton_mask = repair_orange_gaps(
            cleaned_bgr=cleaned_bgr,
            orange_mask=orange_mask,
            dark_threshold=args.threshold,
            max_processing_dim=3000,
        )

        elapsed = time.time() - t0

        # --- build diagnostic images -----------------------------------------
        orig_binary       = (orig_mask        > 0).astype(np.uint8) * 255
        rep_binary        = (repaired_mask    > 0).astype(np.uint8) * 255
        completion_binary = (completion_mask  > 0).astype(np.uint8) * 255
        skeleton_binary   = (skeleton_mask    > 0).astype(np.uint8) * 255

        # Resize cleaned raster to match repaired mask resolution
        h_m, w_m = rep_binary.shape[:2]
        base = cv2.resize(cleaned_bgr, (w_m, h_m), interpolation=cv2.INTER_AREA)

        orig_scaled       = cv2.resize(orig_binary,       (w_m, h_m), interpolation=cv2.INTER_NEAREST)
        completion_scaled = cv2.resize(completion_binary, (w_m, h_m), interpolation=cv2.INTER_NEAREST)
        skeleton_scaled   = cv2.resize(skeleton_binary,   (w_m, h_m), interpolation=cv2.INTER_NEAREST)

        # Build highlight overlay
        highlight = base.copy()

        # 1. Original trace pixels — darken to grey for contrast
        orig_px = orig_scaled > 0
        highlight[orig_px] = (
            highlight[orig_px].astype(np.int32) * 0.55
        ).clip(0, 255).astype(np.uint8)

        # 2. Centerline (medial axis) — faint blue overlay (BGR 230, 80, 0)
        skel_only = (skeleton_scaled > 0) & (completion_scaled == 0)
        highlight[skel_only] = (230, 80, 0)

        # 3. Bézier completion fills — vivid green (BGR 0, 220, 60)
        completion_px = completion_scaled > 0
        highlight[completion_px] = (0, 220, 60)

        # --- save ------------------------------------------------------------
        out_dir = phase2_dir / "phase3" / "repair_test"
        out_dir.mkdir(parents=True, exist_ok=True)

        p_orig      = out_dir / "repair_original.png"
        p_filled    = out_dir / "repair_filled.png"
        p_highlight = out_dir / "repair_highlight.png"

        save_image(p_orig,      orig_binary)
        save_image(p_filled,    rep_binary)
        save_image(p_highlight, highlight)

        n_completion = int(np.count_nonzero(completion_scaled))
        n_skel       = int(np.count_nonzero(skeleton_scaled))
        print(f"  {stem}: completion={n_completion:,}px  skeleton={n_skel:,}px  t={elapsed:.1f}s")
        print(f"    original:  {p_orig}")
        print(f"    filled:    {p_filled}")
        print(f"    highlight: {p_highlight}")
        print(f"      green=Bezier completions  blue=medial-axis skeleton  dark=original trace")

        if not args.no_open:
            _open_file(p_highlight)   # most useful for analysis
            _open_file(p_filled)


if __name__ == "__main__":
    main()
