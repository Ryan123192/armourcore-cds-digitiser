from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import yaml

from armourcore_cds.io.loaders import load_input
from armourcore_cds.io.run_report import write_run_report
from armourcore_cds.io.save import ensure_run_dir
from armourcore_cds.phase1.boundary_detection import detect_outer_border
from armourcore_cds.phase1.crop import find_inner_border_crop
from armourcore_cds.phase1.rectify import rectify_from_corners
from armourcore_cds.phase1.scaling import scale_to_template_design_area
from armourcore_cds.templates.registry import load_template_config
from armourcore_cds.utils.debug import DebugWriter
from armourcore_cds.utils.image_ops import draw_polygon, resize_long_edge, stack_debug_h, to_gray, save_image


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

    crop_result = find_inner_border_crop(rectified.image)
    crop_preview = rectified.image.copy()
    x0, y0, x1, y1 = crop_result.inner_rect_xyxy
    cv2.rectangle(crop_preview, (x0, y0), (x1, y1), (0, 255, 0), 4)
    debug.image('06_inner_border_binary', cv2.cvtColor(crop_result.preview_binary, cv2.COLOR_GRAY2BGR))
    debug.image('07_inner_crop_preview', crop_preview)
    debug.image('08_cropped_design_area_raw', crop_result.cropped_image)

    scaled = scale_to_template_design_area(
        crop_result.cropped_image,
        width_mm=template.design_area_mm.width,
        height_mm=template.design_area_mm.height,
        output_dpi=preferred_output_dpi,
    )
    debug.image('09_scaled_design_area', scaled.image)

    save_image(run_dir / 'rectified_outer.png', rectified.image)
    save_image(run_dir / 'cropped_design_area_raw.png', crop_result.cropped_image)
    save_image(run_dir / 'scaled_design_area.png', scaled.image)

    summary_preview = stack_debug_h(
        resize_long_edge(contour_overlay, 1400),
        resize_long_edge(crop_preview, 1400),
    )
    summary_preview = debug.text_overlay(summary_preview, [
        f'template={template.template_id}',
        f'input={input_meta["input_kind"]}',
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
            'inner_crop_xyxy_px': {
                'x0': x0,
                'y0': y0,
                'x1': x1,
                'y1': y1,
            },
            'inner_border_margins_px': crop_result.border_samples_px,
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
            'The strongest external quadrilateral contour corresponds to the CDS outer border.',
            'The inside-most edge of the thick border can be recovered after rectification by scanning inward from each edge.',
            'This first pass prioritises deterministic geometry recovery, not artefact suppression or trace extraction.',
        ],
    }
    write_run_report(run_dir / 'run_report.json', report)
    return run_dir
