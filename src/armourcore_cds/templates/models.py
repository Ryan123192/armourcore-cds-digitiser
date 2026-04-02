from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class DesignArea(BaseModel):
    width: float
    height: float


class TemplateModel(BaseModel):
    model_config = ConfigDict(extra='ignore')

    template_id: str
    display_name: str
    design_area_mm: DesignArea
    crop_rule: str
    preferred_output_dpi: int
    primary_geometry_truth: str
    fiducials_enabled: bool
