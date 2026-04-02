from armourcore_cds.phase1.scaling import scale_to_template_design_area
from armourcore_cds.templates.calibration import mm_to_pixels
import numpy as np


def test_mm_to_pixels_positive():
    assert mm_to_pixels(25.4, 400) == 400


def test_scale_to_template_design_area_shape():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    result = scale_to_template_design_area(image, width_mm=50, height_mm=25, output_dpi=254)
    assert result.output_size_px == (500, 250)
    assert result.image.shape[:2] == (250, 500)
