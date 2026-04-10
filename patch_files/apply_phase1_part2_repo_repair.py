from __future__ import annotations

from pathlib import Path
import sys

PIPELINE_CONTENT = '''"""Phase 1 pipeline implementation."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

from armourcore_cds.io.loaders import load_input
from armourcore_cds.io.run_report import write_run_report
from armourcore_cds.io.save import ensure_run_dir
from armourcore_cds.phase1.boundary_detection import detect_outer_border
from armourcore_cds.phase1.rectify import rectify_from_corners
from armourcore_cds.phase1.scaling import scale_to_template_design_area
from armourcore_cds.templates.registry import load_template_config
from armourcore_cds.utils.debug import DebugWriter
from armourcore_cds.utils.image_ops import draw_polygon, resize_long_edge, stack_debug_h, save_image


def _corners_to_serialisable(corners: object) -> list[list[float]]:
    arr = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    return [[round(float(x), 2), round(float(y), 2)] for x, y in arr.tolist()]


def run_phase1_pipeline(input_path: Path, template_id: str, config_path: Path) -> Path:
    app_config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    template = load_template_config(template_id)

    run_name = f"{datetime.now():%Y-%m-%d_%H%M%S}_{input_path.stem}"
    base_dir = Path(app_config['output']['base_dir'])
    run_dir = ensure_run_dir(base_dir, run_name)
    debug = DebugWriter(run_dir / 'debug', enabled=bool(app_config['output'].get('write_debug_images', True)))

    preferred_output_dpi = int(template.preferred_output_dpi or app_config['processing']['preferred_output_dpi'])
    input_image, input_meta = load_input(input_path, pdf_dpi=preferred_output_dpi)

    debug.image('01_input_original', input_image)

    expected_aspect_ratio = float(template.design_area_mm.width) / float(template.design_area_mm.height)
    border = detect_outer_border(input_image, expected_aspect_ratio=expected_aspect_ratio)

    contour_overlay = draw_polygon(input_image, border.ordered_corners_xy, (0, 255, 0), thickness=6)
    debug.image('04_border_detected', contour_overlay)

    rectified = rectify_from_corners(input_image, border.ordered_corners_xy, expected_aspect_ratio=expected_aspect_ratio)
    debug.image('05_rectified_outer', rectified.image)

    design_area_preview = rectified.image.copy()
    h, w = design_area_preview.shape[:2]
    cv2.rectangle(design_area_preview, (0, 0), (w - 1, h - 1), (0, 255, 0), 4)
    debug.image('06_design_area_preview', design_area_preview)
    debug.image('07_design_area_raw', rectified.image)

    scaled = scale_to_template_design_area(
        rectified.image,
        width_mm=template.design_area_mm.width,
        height_mm=template.design_area_mm.height,
        output_dpi=preferred_output_dpi,
    )
    debug.image('08_scaled_design_area', scaled.image)

    save_image(run_dir / 'rectified_outer.png', rectified.image)
    save_image(run_dir / 'cropped_design_area_raw.png', rectified.image)
    save_image(run_dir / 'scaled_design_area.png', scaled.image)

    summary_preview = stack_debug_h(
        resize_long_edge(contour_overlay, 1400),
        resize_long_edge(design_area_preview, 1400),
    )
    summary_lines = [
        f'template={template.template_id}',
        f'input={input_meta["input_kind"]}',
        'crop_mode=none_use_detected_border',
        f'border_confidence={border.confidence}',
        f'border_score={border.score:.3f}',
        f'output_px={scaled.output_size_px[0]}x{scaled.output_size_px[1]}',
    ]
    summary_preview = debug.text_overlay(summary_preview, summary_lines)
    debug.image('10_summary_preview', summary_preview)

    report = {
        'input_path': str(input_path),
        'template_id': template.template_id,
        'template_display_name': template.display_name,
        'phase': 'phase1_rectify_clean',
        'status': 'success',
        'input': {
            **input_meta,
            'width_px': int(input_image.shape[1]),
            'height_px': int(input_image.shape[0]),
        },
        'design_area_mm': {
            'width': template.design_area_mm.width,
            'height': template.design_area_mm.height,
        },
        'border_detection': {
            'ordered_corners_xy': _corners_to_serialisable(border.ordered_corners_xy),
            'contour_area_px': float(border.contour_area_px),
            'score': float(border.score),
            'candidate_count': int(border.candidate_count),
            'confidence': border.confidence,
            'diagnostics': border.diagnostics or {},
        },
        'rectification': {
            'rectified_outer_size_px': {
                'width': rectified.rectified_size_px[0],
                'height': rectified.rectified_size_px[1],
            },
            'crop_mode': 'none_use_detected_border',
            'design_area_xyxy_px': {
                'x0': 0,
                'y0': 0,
                'x1': rectified.rectified_size_px[0] - 1,
                'y1': rectified.rectified_size_px[1] - 1,
            },
            'design_area_margins_px': {
                'top': 0,
                'bottom': 0,
                'left': 0,
                'right': 0,
            },
        },
        'scaling': {
            'output_dpi': preferred_output_dpi,
            'scaled_size_px': {
                'width': scaled.output_size_px[0],
                'height': scaled.output_size_px[1],
            },
            'px_per_mm_x': scaled.px_per_mm_x,
            'px_per_mm_y': scaled.px_per_mm_y,
            'effective_dpi_x': scaled.effective_dpi_x,
            'effective_dpi_y': scaled.effective_dpi_y,
        },
        'outputs': {
            'rectified_outer': 'rectified_outer.png',
            'cropped_design_area_raw': 'cropped_design_area_raw.png',
            'scaled_design_area': 'scaled_design_area.png',
            'debug_dir': 'debug',
        },
        'assumptions': [
            'The detected thick-border quadrilateral is the geometry truth for rectification, crop and scale.',
            'Calibration squares are support features for border confidence only and do not define the output crop.',
            'Phase 01 intentionally skips any secondary crop to avoid locking onto customer-drawn geometry.',
        ],
    }
    write_run_report(run_dir / 'run_report.json', report)
    return run_dir
'''

RECTIFY_CONTENT = '''from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class RectifyResult:
    image: np.ndarray
    transform_matrix: np.ndarray
    rectified_size_px: tuple[int, int]


def _coerce_corners_xy(corners: object) -> np.ndarray:
    arr = np.asarray(corners, dtype=np.float32)
    if arr.shape != (4, 2):
        arr = arr.reshape(4, 2)
    return arr


def _target_size_from_corners(corners: object, expected_aspect_ratio: float) -> tuple[int, int]:
    corners_xy = _coerce_corners_xy(corners)
    tl, tr, br, bl = corners_xy
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)

    measured_w = max(width_a, width_b, 1.0)
    measured_h = max(height_a, height_b, 1.0)

    if measured_w / measured_h > expected_aspect_ratio:
        target_w = int(round(measured_w))
        target_h = int(round(target_w / expected_aspect_ratio))
    else:
        target_h = int(round(measured_h))
        target_w = int(round(target_h * expected_aspect_ratio))

    target_w = max(target_w, 64)
    target_h = max(target_h, 64)
    return target_w, target_h


def rectify_from_corners(image: np.ndarray, corners: object, expected_aspect_ratio: float) -> RectifyResult:
    corners_xy = _coerce_corners_xy(corners)
    target_w, target_h = _target_size_from_corners(corners_xy, expected_aspect_ratio)
    dst = np.array(
        [[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(corners_xy, dst)
    rectified = cv2.warpPerspective(image, matrix, (target_w, target_h), flags=cv2.INTER_CUBIC)
    return RectifyResult(image=rectified, transform_matrix=matrix, rectified_size_px=(target_w, target_h))
'''

BOUNDARY_TEST_CONTENT = '''import cv2
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
'''

CONFTEST_CONTENT = '''from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
'''


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8', newline='\n')


def main() -> int:
    repo_root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd().resolve()
    targets = {
        repo_root / 'src/armourcore_cds/phase1/pipeline.py': PIPELINE_CONTENT,
        repo_root / 'src/armourcore_cds/phase1/rectify.py': RECTIFY_CONTENT,
        repo_root / 'tests/test_boundary_detection.py': BOUNDARY_TEST_CONTENT,
        repo_root / 'tests/conftest.py': CONFTEST_CONTENT,
    }
    missing = [str(path) for path in targets if not path.parent.exists()]
    if missing:
        raise SystemExit('Repo root looks wrong. Missing parent directories for:\n- ' + '\n- '.join(missing))
    for path, content in targets.items():
        write_text(path, content)
        print(f'Updated {path.relative_to(repo_root)}')
    print('\nDone. Suggested verification:')
    print('  pytest -q')
    print('  PYTHONPATH=src python -m armourcore_cds.cli "data/inputs/raw_images/Test02.jpg" --template cds_regular_500x600 --config configs/app/default.yaml')
    print('  PYTHONPATH=src python -m armourcore_cds.cli "data/inputs/raw_images/Test01.jpeg" --template cds_xlarge_500x900 --config configs/app/default.yaml')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
