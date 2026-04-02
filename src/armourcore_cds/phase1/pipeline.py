from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml

from armourcore_cds.io.run_report import write_run_report
from armourcore_cds.io.save import ensure_run_dir
from armourcore_cds.templates.registry import load_template_config


def run_phase1_pipeline(input_path: Path, template_id: str, config_path: Path) -> Path:
    app_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    template = load_template_config(template_id)

    run_name = f"{datetime.now():%Y-%m-%d_%H%M%S}_{input_path.stem}"
    base_dir = Path(app_config["output"]["base_dir"])
    run_dir = ensure_run_dir(base_dir, run_name)

    report = {
        "input_path": str(input_path),
        "template_id": template.template_id,
        "template_display_name": template.display_name,
        "design_area_mm": {
            "width": template.design_area_mm.width,
            "height": template.design_area_mm.height,
        },
        "phase": "phase1_rectify_clean",
        "status": "scaffold_only",
        "notes": [
            "Pipeline scaffold created.",
            "Phase 01 implementation pending."
        ],
    }
    write_run_report(run_dir / "run_report.json", report)
    return run_dir
