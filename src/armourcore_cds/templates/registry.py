from __future__ import annotations

from pathlib import Path
import yaml

from armourcore_cds.templates.models import TemplateModel


def load_template_config(template_id: str, config_dir: Path | None = None) -> TemplateModel:
    config_dir = config_dir or Path("configs/templates")
    config_path = config_dir / f"{template_id}.yaml"
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return TemplateModel.model_validate(data)
