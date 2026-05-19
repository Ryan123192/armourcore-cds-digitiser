"""Bulk Phase 3 iso-tool tester — runs the vectorise script on every
``IsoToolUnfixed*.png`` in ``data/inputs/phase03_testing/`` and produces
a summary table of metrics + a side-by-side preview grid.

This is the regression-watcher: change something in the vectoriser, run
this, eyeball the table to see which cases got better or worse.

Metrics reported per image:
  * chosen kernel (adaptive)
  * # bridges added in Method F stage 2
  * # remaining centerline endpoints (0 = fully closed)
  * # final loops detected (after junction-artefact filter)
  * # raw inner-contour points (largest loop)
  * # RDP nodes (sum across loops)
  * # cusp corners (sum across loops)

Usage:
    python tools/test_phase3_iso_bulk.py
    python tools/test_phase3_iso_bulk.py --pattern "IsoTool*"
"""
from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import re
from pathlib import Path

REPO = Path(__file__).parent.parent


def _open_dir(path: Path) -> None:
    try:
        if platform.system() == "Windows":
            subprocess.Popen(["explorer", str(path)])
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as exc:
        print(f"  [open] {exc}")


def parse_summary(stdout: str) -> dict:
    """Pull metrics out of the vectorise script's stdout."""
    out: dict = {}

    m = re.search(r"Method F \(adaptive: kernel=(\d+)\): (\d+) bridges, "
                  r"(?:fully closed|(\d+) endpoints remain) "
                  r"\((\d+) loops\)", stdout)
    if m:
        out["kernel"] = int(m.group(1))
        out["bridges"] = int(m.group(2))
        out["endpoints"] = int(m.group(3) or 0)
        out["loops_during_search"] = int(m.group(4))

    m = re.search(r"Inside-edge loops detected: (\d+)", stdout)
    if m:
        out["loops_final"] = int(m.group(1))

    nodes = sum(int(x) for x in re.findall(r"(\d+) RDP nodes", stdout))
    cusps = sum(int(x) for x in re.findall(r"(\d+) cusps", stdout))
    out["rdp_nodes"] = nodes
    out["cusps"] = cusps

    out["dropped_artefacts"] = len(re.findall(r"DROP loop area=", stdout))

    m = re.search(r"Total: (\d+) nodes across (\d+) loops, "
                  r"(\d+) cusp corners", stdout)
    if m:
        out["total_nodes"] = int(m.group(1))

    m = re.search(r"Avg trace thickness: ([\d.]+)px", stdout)
    if m:
        out["avg_thickness"] = float(m.group(1))

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="IsoToolUnfixed*",
                        help="Glob for input PNG names (without .png).")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    in_dir  = REPO / "data" / "inputs" / "phase03_testing"
    paths = sorted(in_dir.glob(f"{args.pattern}.png"))
    if not paths:
        print(f"No matches for {args.pattern} in {in_dir}")
        sys.exit(1)

    rows: list[tuple[str, dict]] = []

    for in_path in paths:
        name = in_path.stem
        print(f"\n>>> Running {name}...")
        proc = subprocess.run(
            [sys.executable,
             str(REPO / "tools" / "test_phase3_iso_vectorise.py"),
             name, "--no-open"],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print(f"  FAILED (exit {proc.returncode})")
            print(proc.stdout)
            print(proc.stderr)
            rows.append((name, {"error": "failed"}))
            continue
        metrics = parse_summary(proc.stdout)
        rows.append((name, metrics))
        for k, v in metrics.items():
            print(f"    {k:24s} {v}")

    # Pretty table
    print("\n" + "=" * 96)
    print(f"{'name':<24} {'k':>4} {'br':>4} {'ep':>4} {'loops':>6} "
          f"{'drop':>5} {'nodes':>6} {'cusps':>6} {'thick':>6}")
    print("-" * 96)
    for name, m in rows:
        if "error" in m:
            print(f"{name:<24} ERROR")
            continue
        print(
            f"{name:<24} "
            f"{m.get('kernel', '?'):>4} "
            f"{m.get('bridges', '?'):>4} "
            f"{m.get('endpoints', 0):>4} "
            f"{m.get('loops_final', '?'):>6} "
            f"{m.get('dropped_artefacts', 0):>5} "
            f"{m.get('total_nodes', m.get('rdp_nodes', '?')):>6} "
            f"{m.get('cusps', '?'):>6} "
            f"{m.get('avg_thickness', 0.0):>6.1f}"
        )
    print("=" * 96)

    out_dir = REPO / "data" / "outputs" / "phase03_testing"
    if not args.no_open:
        _open_dir(out_dir)


if __name__ == "__main__":
    main()
