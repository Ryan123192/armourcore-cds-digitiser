from armourcore_cds.templates.registry import load_template_config


def test_regular_template_loads():
    template = load_template_config('cds_regular_500x600')
    assert template.design_area_mm.width == 500
    assert template.design_area_mm.height == 600


def test_xlarge_template_loads():
    template = load_template_config('cds_xlarge_500x900')
    assert template.design_area_mm.width == 500
    assert template.design_area_mm.height == 900
