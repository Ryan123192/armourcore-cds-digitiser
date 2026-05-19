"""Quick batch runner — processes all test images and prints a result summary.

Usage (from repo root):
    python tools/batch_run.py
    python tools/batch_run.py --input-dir data/inputs/raw_images --template cds_regular_500x600
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from armourcore_cds.phase1.pipeline import run_phase1_pipeline

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".pdf"}

# Filename-pattern → template mapping. First match wins.
TEMPLATE_PATTERNS: list[tuple[str, str]] = [
    ("bluecolour", "cds_colour_test_260x350"),
    ("colourtest", "cds_colour_test_260x350"),
]
DEFAULT_TEMPLATE = "cds_regular_500x600"


def _template_for(path: Path, override: str | None) -> str:
    if override:
        return override
    name_lower = path.name.lower()
    for pattern, template_id in TEMPLATE_PATTERNS:
        if pattern in name_lower:
            return template_id
    return DEFAULT_TEMPLATE


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-run Phase 01 on a folder of images.")
    parser.add_argument("--input-dir", default="data/inputs/raw_images", help="Folder containing test images.")
    parser.add_argument("--template", default="", help="Force a specific template for all files (default: auto-select by filename).")
    parser.add_argument("--config", default="configs/app/default.yaml", help="App config path.")
    parser.add_argument("--filter", default="", help="Only process files whose name contains this string.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"ERROR: input dir not found: {input_dir}")
        sys.exit(1)

    files = sorted(
        f for f in input_dir.iterdir()
        if f.suffix.lower() in IMAGE_SUFFIXES
        and (args.filter.lower() in f.name.lower() if args.filter else True)
    )

    if not files:
        print(f"No image files found in {input_dir}")
        sys.exit(0)

    template_override = args.template or None
    print(f"\nProcessing {len(files)} file(s)  (template: {'forced=' + template_override if template_override else 'auto by filename'})\n{'='*70}")

    results = []
    for path in files:
        template_id = _template_for(path, template_override)
        try:
            run_dir = run_phase1_pipeline(
                input_path=path,
                template_id=template_id,
                config_path=Path(args.config),
            )
            report_path = run_dir / "run_report.json"
            report = json.loads(report_path.read_text())
            bd = report.get("border_detection", {})
            sc = report.get("scaling", {})
            status = "OK"
            confidence = bd.get("confidence", "?")
            score = bd.get("score", 0)
            mode = bd.get("detected_border_mode", "?")
            size = sc.get("scaled_size_px", {})
            px = f"{size.get('width','?')}x{size.get('height','?')}"
            results.append((path.name, template_id, status, confidence, f"{score:.2f}", mode, px, str(run_dir)))
        except Exception as exc:
            results.append((path.name, template_id, "FAIL", "-", "-", "-", "-", str(exc)))

    # Print summary table
    print(f"\n{'File':<45} {'Template':<28} {'Conf':<5} {'Score':<7} {'Mode':<8} {'OutputPx'}")
    print("-" * 115)
    for name, tmpl, status, conf, score, mode, px, detail in results:
        flag = "  " if status == "OK" else "!!"
        print(f"{flag} {name:<43} {tmpl:<28} {conf:<5} {score:<7} {mode:<8} {px}")
        if status == "FAIL":
            print(f"     ERROR: {detail[:90]}")

    n_ok = sum(1 for r in results if r[2] == "OK")
    print(f"\n{n_ok}/{len(results)} succeeded.")


if __name__ == "__main__":
    main()
