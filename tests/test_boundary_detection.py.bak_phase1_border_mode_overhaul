import cv2
import numpy as np

from armourcore_cds.phase1.boundary_detection import detect_outer_border


def test_detect_outer_border_on_synthetic_sheet():
    image = np.full((900, 1200, 3), 255, dtype=np.uint8)
    pts = np.array([[150, 120], [1050, 160], [980, 760], [180, 720]], dtype=np.int32)
    cv2.polylines(image, [pts.reshape(-1, 1, 2)], isClosed=True, color=(0, 0, 0), thickness=18)

    result = detect_outer_border(image, expected_aspect_ratio=600 / 500)

    assert len(result.ordered_corners_xy) == 4
    assert all(len(pt) == 2 for pt in result.ordered_corners_xy)
    assert result.candidate_count > 0
    assert result.score > 0
    assert result.confidence in {'high', 'low'}
