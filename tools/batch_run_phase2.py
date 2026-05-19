"""Phase 2 batch runner — processes Phase 1 outputs through the grid-removal pipeline.

Picks up the most recent Phase 1 run for each input image and runs Phase 2 on it.

Usage (from repo root):
    python tools/batch_run_phase2.py
    python tools/batch_run_phase2.py --filter BlueColour
    python tools/batch_run_phase2.py --phase1-dir outputs/runs --filter Test
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np

from armourcore_cds.phase2.pipeline import run_phase2_pipeline
from armourcore_cds.templates.registry import load_template_config
from armourcore_cds.utils.debug import DebugWriter

# Filename → template mapping (mirrors batch_run.py)
TEMPLATE_PATTERNS: list[tuple[str, str]] = [
    ("bluecolour", "cds_colour_test_260x350"),
    ("colourtest", "cds_colour_test_260x350"),
]
DEFAULT_TEMPLATE = "cds_regular_500x600"


def _template_for(image_stem: str) -> str:
    low = image_stem.lower()
    for pattern, template_id in TEMPLATE_PATTERNS:
        if pattern in low:
            return template_id
    return DEFAULT_TEMPLATE


def _latest_phase1_run(runs_dir: Path, image_stem: str) -> Path | None:
    """Return the most-recent Phase 1 run directory for the given image stem."""
    candidates = sorted(
        (d for d in runs_dir.iterdir()
         if d.is_dir() and d.name.endswith(image_stem)),
        reverse=True,
    )
    return candidates[0] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-run Phase 02 on Phase 01 outputs.")
    parser.add_argument("--phase1-dir", default="outputs/runs",
                        dest="phase1_dir",
                        help="Directory containing Phase 1 run folders.")
    parser.add_argument("--filter", default="",
                        help="Only process image names containing this string.")
    args = parser.parse_args()

    runs_dir = Path(args.phase1_dir)
    if not runs_dir.exists():
        print(f"ERROR: Phase 1 output dir not found: {runs_dir}")
        sys.exit(1)

    # Collect unique image stems from run directory names
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
        print("No matching Phase 1 runs found.")
        sys.exit(0)

    print(f"\nPhase 2 batch — {len(stems)} image(s)\n{'='*70}")
    results = []

    for stem in stems:
        run_dir = _latest_phase1_run(runs_dir, stem)
        if run_dir is None:
            results.append((stem, "SKIP", "-", "-", "no Phase 1 run found"))
            continue

        scaled = run_dir / "scaled_design_area.png"
        if not scaled.exists():
            results.append((stem, "SKIP", "-", "-", "no scaled_design_area.png"))
            continue

        template_id = _template_for(stem)
        try:
            template = load_template_config(template_id)
        except Exception as e:
            results.append((stem, "FAIL", template_id, "-", str(e)))
            continue

        img = cv2.imread(str(scaled))
        if img is None:
            results.append((stem, "FAIL", template_id, "-", "cv2 read failed"))
            continue

        h, w = img.shape[:2]
        phase2_out = run_dir / "phase2"
        phase2_out.mkdir(exist_ok=True)
        debug = DebugWriter(phase2_out / "debug", enabled=True)

        t0 = time.time()
        try:
            result = run_phase2_pipeline(img, template, run_dir=phase2_out, debug=debug)
            elapsed = time.time() - t0
            gcol = result.grid_colour
            if gcol == "orange":
                grid_pct = float(np.mean(result.isolation.orange_grid_mask > 0) * 100)
                grid_label = "orange"
            else:
                grid_pct = float(np.mean(result.isolation.black_grid_mask > 0) * 100)
                grid_label = "black"
            removal_pct = float(np.mean(result.isolation.removal_mask > 0) * 100)
            n_traces    = len(result.contour_candidates)
            detail = (f"grid={grid_label} {grid_pct:.2f}%  removed={removal_pct:.2f}%  "
                      f"traces={n_traces}  t={elapsed:.1f}s")
            results.append((stem, "OK", template_id, f"{w}x{h}", detail))
        except Exception as exc:
            results.append((stem, "FAIL", template_id, f"{w}x{h}", str(exc)[:80]))

    # Summary
    print(f"\n{'Image':<35} {'Status':<6} {'Template':<28} {'Px':<12} Details")
    print("-" * 120)
    for stem, status, tmpl, px, detail in results:
        flag = "  " if status == "OK" else "!!"
        print(f"{flag} {stem:<33} {status:<6} {tmpl:<28} {px:<12} {detail}")

    n_ok = sum(1 for r in results if r[1] == "OK")
    print(f"\n{n_ok}/{len(results)} succeeded.")
    print(f"\nOutputs in: {runs_dir}/<run>/phase2/")


if __name__ == "__main__":
    main()
