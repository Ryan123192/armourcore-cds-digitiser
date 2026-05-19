"""Phase 3 runner — vectorise Phase 2 cleaned raster outputs.

Finds the most-recent Phase 1 / Phase 2 run for each image, runs Phase 3
vectorisation, and opens the resulting SVG + overlay PNG for inspection.

Usage (from repo root):
    python tools/run_phase3.py
    python tools/run_phase3.py --filter BlueColourTest00
    python tools/run_phase3.py --filter BlueColourTest00 --rdp 4 --min-area 100
    python tools/run_phase3.py --no-open
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

from armourcore_cds.phase3.pipeline import run_phase3_pipeline
from armourcore_cds.templates.registry import load_template_config
from armourcore_cds.utils.debug import DebugWriter

# --------------------------------------------------------------------------
# Template routing (mirrors batch_run_phase2.py)
# --------------------------------------------------------------------------
TEMPLATE_PATTERNS: list[tuple[str, str]] = [
    ("bluecolour", "cds_colour_test_260x350"),
    ("colourtest", "cds_colour_test_260x350"),
]
DEFAULT_TEMPLATE = "cds_regular_500x600"


def _template_for(stem: str) -> str:
    low = stem.lower()
    for pattern, tid in TEMPLATE_PATTERNS:
        if pattern in low:
            return tid
    return DEFAULT_TEMPLATE


def _latest_phase1_run(runs_dir: Path, stem: str) -> Path | None:
    candidates = sorted(
        (d for d in runs_dir.iterdir() if d.is_dir() and d.name.endswith(stem)),
        reverse=True,
    )
    return candidates[0] if candidates else None


def _open_file(path: Path) -> None:
    """Open *path* with the OS default viewer."""
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
# Main
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Batch Phase 3 vectorisation.")
    parser.add_argument("--phase1-dir", default="outputs/runs", dest="phase1_dir",
                        help="Directory containing Phase 1 run folders.")
    parser.add_argument("--filter", default="",
                        help="Only process image names containing this string.")
    parser.add_argument("--rdp", type=float, default=3.0, dest="rdp",
                        help="RDP simplification epsilon in mask pixels (default: 3.0).")
    parser.add_argument("--min-area", type=float, default=200.0, dest="min_area",
                        help="Minimum contour area in mask pixels (default: 200).")
    parser.add_argument("--tension", type=float, default=1.0,
                        help="Catmull-Rom tension (default: 1.0).")
    parser.add_argument("--gap-close", type=int, default=7, dest="gap_close",
                        help="Residual closing kernel after orange repair (mask px, default: 7). "
                             "Set 0 to disable.")
    parser.add_argument("--max-gap-close", type=int, default=31, dest="max_gap_close",
                        help="Max kernel for per-ribbon local retry (default: 31).")
    parser.add_argument("--circularity", type=float, default=0.05, dest="circularity",
                        help="Isoperimetric ratio threshold; below this = ribbon (default: 0.05).")
    parser.add_argument("--orange-bridge", type=int, default=20, dest="orange_bridge",
                        help="Directional bridge half-width in full-res pixels for orange gap repair "
                             "(default: 20 ~ 1.3 mm at 5512px/350mm). Raise for wider grid lines.")
    parser.add_argument("--no-open", action="store_true", dest="no_open",
                        help="Do not open output files automatically.")
    args = parser.parse_args()

    runs_dir = Path(args.phase1_dir)
    if not runs_dir.exists():
        print(f"ERROR: Phase 1 output dir not found: {runs_dir}")
        sys.exit(1)

    # Collect unique image stems
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

    print(f"\nPhase 3 vectorisation — {len(stems)} image(s)")
    print(f"  rdp={args.rdp}  min_area={args.min_area}  tension={args.tension}  "
          f"gap_close={args.gap_close}px  max_gap={args.max_gap_close}px  circ>={args.circularity}")
    print("=" * 70)
    results = []
    opened: list[Path] = []

    for stem in stems:
        run_dir = _latest_phase1_run(runs_dir, stem)
        if run_dir is None:
            results.append((stem, "SKIP", "no Phase 1 run found"))
            continue

        phase2_dir = run_dir / "phase2"
        cleaned_path = phase2_dir / "phase2_cleaned_raster.png"
        mask_path = phase2_dir / "phase2_trace_candidate_mask.png"

        if not cleaned_path.exists():
            results.append((stem, "SKIP", "no phase2_cleaned_raster.png — run Phase 2 first"))
            continue
        if not mask_path.exists():
            results.append((stem, "SKIP", "no phase2_trace_candidate_mask.png"))
            continue

        template_id = _template_for(stem)
        try:
            template = load_template_config(template_id)
        except Exception as exc:
            results.append((stem, "FAIL", str(exc)))
            continue

        cleaned_bgr = cv2.imread(str(cleaned_path))
        trace_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Orange mask — optional; enables the grid-gap repair pass
        orange_path = phase2_dir / "phase2_orange_mask.png"
        orange_mask = (
            cv2.imread(str(orange_path), cv2.IMREAD_GRAYSCALE)
            if orange_path.exists() else None
        )

        if cleaned_bgr is None:
            results.append((stem, "FAIL", "cv2 could not read cleaned raster"))
            continue
        if trace_mask is None:
            results.append((stem, "FAIL", "cv2 could not read trace mask"))
            continue

        phase3_dir = phase2_dir / "phase3"
        phase3_dir.mkdir(exist_ok=True)
        debug = DebugWriter(phase3_dir / "debug", enabled=True)

        t0 = time.time()
        try:
            result = run_phase3_pipeline(
                cleaned_bgr=cleaned_bgr,
                trace_mask=trace_mask,
                template=template,
                orange_mask=orange_mask,
                run_dir=phase3_dir,
                debug=debug,
                min_area_px=args.min_area,
                rdp_epsilon=args.rdp,
                tension=args.tension,
                gap_close_px=args.gap_close,
                max_gap_close_px=args.max_gap_close,
                circularity_min=args.circularity,
                orange_bridge_px=args.orange_bridge,
            )
            elapsed = time.time() - t0
            detail = (
                f"{result.n_paths} paths  "
                f"nodes={sum(len(p.points) for p in result.vector_paths)}  "
                f"t={elapsed:.1f}s"
            )
            results.append((stem, "OK", detail))

            if not args.no_open:
                opened.append(result.svg_path)
                opened.append(result.overlay_path)

        except Exception as exc:
            import traceback
            traceback.print_exc()
            results.append((stem, "FAIL", str(exc)[:100]))

    # Summary
    print(f"\n{'Image':<35} {'Status':<6} Details")
    print("-" * 90)
    for stem, status, detail in results:
        flag = "  " if status == "OK" else "!!"
        print(f"{flag} {stem:<33} {status:<6} {detail}")

    n_ok = sum(1 for r in results if r[1] == "OK")
    print(f"\n{n_ok}/{len(results)} succeeded.")

    # Open outputs
    if opened:
        print(f"\nOpening {len(opened)} output file(s)…")
        for p in opened:
            print(f"  {p}")
            _open_file(p)


if __name__ == "__main__":
    main()
