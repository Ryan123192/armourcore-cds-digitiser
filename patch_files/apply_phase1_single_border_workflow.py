from pathlib import Path

ROOT = Path.cwd()

TEMPLATE_UPDATES = {
    'configs/templates/cds_regular_500x600.yaml': (
        '  width: 500\n  height: 600',
        '  width: 600\n  height: 500',
    ),
    'configs/templates/cds_xlarge_500x900.yaml': (
        '  width: 500\n  height: 900',
        '  width: 900\n  height: 500',
    ),
    'configs/templates/cds_colour_test_260x350.yaml': (
        '  width: 260\n  height: 350',
        '  width: 350\n  height: 260',
    ),
}

for rel_path, (old, new) in TEMPLATE_UPDATES.items():
    path = ROOT / rel_path
    text = path.read_text(encoding='utf-8')
    if old in text:
        text = text.replace(old, new)
    elif new in text:
        pass
    else:
        raise RuntimeError(f'Expected dimensions block not found in {rel_path}')
    path.write_text(text, encoding='utf-8')

pipeline_code = '''"""Phase 1 pipeline implementation."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
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

    contour_overlay = draw_polygon(input_image, border.ordered_corners, (0, 255, 0), thickness=6)
    debug.image('02_border_edges', cv2.cvtColor(border.preview_edges, cv2.COLOR_GRAY2BGR))
    debug.image('03_border_mask', cv2.cvtColor(border.preview_mask, cv2.COLOR_GRAY2BGR))
    debug.image('04_border_detected', contour_overlay)

    rectified = rectify_from_corners(input_image, border.ordered_corners, expected_aspect_ratio=expected_aspect_ratio)
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
    summary_preview = debug.text_overlay(summary_preview, [
        f'template={template.template_id}',
        f'input={input_meta["input_kind"]}',
        'crop_mode=none_use_detected_border',
        f'output_px={scaled.output_size_px[0]}x{scaled.output_size_px[1]}',
    ])
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
            'ordered_corners_xy': border.ordered_corners.round(2).tolist(),
            'contour_area_px': border.contour_area_px,
            'score': border.score,
            'candidate_count': border.candidate_count,
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

pipeline_path = ROOT / 'src/armourcore_cds/phase1/pipeline.py'
pipeline_path.write_text(pipeline_code, encoding='utf-8')

print('Patched files:')
for rel_path in TEMPLATE_UPDATES:
    print(f' - {rel_path}')
print(' - src/armourcore_cds/phase1/pipeline.py')
print('\nNotes:')
print(' - Template dimensions were normalised to landscape orientation.')
print(' - Phase 01 now uses a single detected border workflow: detect -> rectify -> scale.')
print(' - No secondary crop is performed.')
