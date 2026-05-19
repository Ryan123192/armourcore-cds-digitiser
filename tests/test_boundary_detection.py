import cv2
import numpy as np

from armourcore_cds.phase1.boundary_detection import detect_outer_border


def test_detect_outer_border_on_synthetic_black_sheet():
    image = np.full((900, 1200, 3), 255, dtype=np.uint8)
    pts = np.array([[150, 120], [1050, 160], [980, 760], [180, 720]], dtype=np.int32)
    cv2.polylines(image, [pts.reshape(-1, 1, 2)], isClosed=True, color=(0, 0, 0), thickness=18)

    result = detect_outer_border(image, expected_aspect_ratio=600 / 500, border_colour_mode="black")

    assert len(result.ordered_corners_xy) == 4
    assert all(len(pt) == 2 for pt in result.ordered_corners_xy)
    assert result.candidate_count > 0
    assert result.score > 0
    assert result.confidence in {"high", "low"}
    assert result.diagnostics["detected_border_mode"] == "black"


def test_detect_outer_border_on_synthetic_orange_sheet():
    image = np.full((900, 1200, 3), 255, dtype=np.uint8)
    outer = np.array([[120, 100], [1080, 135], [1010, 790], [155, 760]], dtype=np.int32)
    inner = np.array([[220, 200], [980, 220], [930, 700], [250, 675]], dtype=np.int32)
    orange_bgr = (0, 84, 211)

    cv2.polylines(image, [outer.reshape(-1, 1, 2)], isClosed=True, color=orange_bgr, thickness=20)
    cv2.polylines(image, [inner.reshape(-1, 1, 2)], isClosed=True, color=orange_bgr, thickness=4)

    for x in range(260, 940, 120):
        cv2.line(image, (x, 210), (x - 20, 690), orange_bgr, 2)
    for y in range(260, 680, 90):
        cv2.line(image, (235, y), (955, y - 10), orange_bgr, 2)

    result = detect_outer_border(
        image,
        expected_aspect_ratio=350 / 260,
        border_colour_mode="orange",
        outer_border_hex_candidates=["#D35400"],
        orange_border_hex="#D35400",
    )

    assert len(result.ordered_corners_xy) == 4
    assert result.candidate_count > 0
    assert result.score > 0
    assert result.diagnostics["detected_border_mode"] == "orange"
    assert result.diagnostics["colour_score"] >= 0.12
    assert result.diagnostics["colour_band_width_score"] >= 0.08


def test_detect_outer_border_on_synthetic_blue_sheet():
    image = np.full((900, 1200, 3), 255, dtype=np.uint8)
    outer = np.array([[120, 110], [1080, 110], [1070, 790], [130, 790]], dtype=np.int32)
    inner = np.array([[240, 210], [970, 220], [950, 690], [250, 680]], dtype=np.int32)
    blue_bgr = (236, 202, 0)  # #00CAEC in BGR

    cv2.polylines(image, [outer.reshape(-1, 1, 2)], isClosed=True, color=blue_bgr, thickness=20)
    cv2.polylines(image, [inner.reshape(-1, 1, 2)], isClosed=True, color=blue_bgr, thickness=4)

    result = detect_outer_border(
        image,
        expected_aspect_ratio=350 / 260,
        border_colour_mode="blue",
        outer_border_hex_candidates=["#00CAEC"],
        blue_border_hex="#00CAEC",
    )

    assert len(result.ordered_corners_xy) == 4
    assert result.candidate_count > 0
    assert result.score > 0
    assert result.diagnostics["detected_border_mode"] == "blue"
    assert result.diagnostics["colour_score"] >= 0.12
    assert result.diagnostics["colour_band_width_score"] >= 0.08
