from armourcore_cds.templates.calibration import mm_to_pixels


def test_mm_to_pixels_positive():
    assert mm_to_pixels(25.4, 400) == 400
