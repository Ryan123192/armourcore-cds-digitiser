from __future__ import annotations

import argparse
from pathlib import Path

from armourcore_cds.phase1.pipeline import run_phase1_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="armourcore-cds",
        description="ArmourCore CDS Digitiser development CLI.",
    )
    parser.add_argument("input_path", nargs="?", help="Path to an input image or scanned PDF.")
    parser.add_argument("--template", default="cds_regular_500x600", help="Template id defined in configs/templates.")
    parser.add_argument("--config", default="configs/app/default.yaml", help="Path to application config YAML.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.input_path:
        parser.print_help()
        return 0

    result = run_phase1_pipeline(
        input_path=Path(args.input_path),
        template_id=args.template,
        config_path=Path(args.config),
    )
    print(f"Run complete: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
