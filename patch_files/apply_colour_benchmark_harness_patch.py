from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


def main() -> int:
    if len(sys.argv) != 2:
        print('Usage: python apply_colour_benchmark_harness_patch.py <repo_root>')
        return 2

    repo_root = Path(sys.argv[1]).resolve()
    if not repo_root.exists():
        print(f'Repo root does not exist: {repo_root}')
        return 2

    benchmark_tool = dedent('''
    """Batch benchmark runner for the 4-up CDS colour-test sheet.

    Expected input filenames:
        <lighting>__<angle>__<sheet_state>__<capture_id>.jpg

    Example:
        sunlight__badangle__creased__01.jpg

    The benchmark assumes a single photo contains all four 260x350 test sets on one A1 print:
        - top-left:     set_magenta
        - top-right:    set_cyan
        - bottom-left:  set_orange
        - bottom-right: set_green
    """
    from __future__ import annotations

    import argparse
    import csv
    import json
    from dataclasses import asdict, dataclass
    from datetime import datetime
    from pathlib import Path
    from typing import Any

    import cv2
    import numpy as np
    import yaml

    from armourcore_cds.io.loaders import SUPPORTED_IMAGE_SUFFIXES, load_input
    from armourcore_cds.phase1.boundary_detection import detect_outer_border
    from armourcore_cds.phase1.rectify import rectify_from_corners
    from armourcore_cds.phase1.scaling import scale_to_template_design_area
    from armourcore_cds.templates.registry import load_template_config
    from armourcore_cds.utils.image_ops import draw_polygon, ensure_uint8_bgr, resize_long_edge, save_image


    FILENAME_PARTS = ("lighting", "angle", "sheet_state", "capture_id")
    QUADRANTS = (
        {
            "set_id": "set_magenta",
            "label": "Magenta",
            "row": 0,
            "col": 0,
            "x0_frac": 0.00,
            "x1_frac": 0.58,
            "y0_frac": 0.00,
            "y1_frac": 0.58,
            "colour_bgr": (180, 30, 180),
        },
        {
            "set_id": "set_cyan",
            "label": "Cyan",
            "row": 0,
            "col": 1,
            "x0_frac": 0.42,
            "x1_frac": 1.00,
            "y0_frac": 0.00,
            "y1_frac": 0.58,
            "colour_bgr": (200, 180, 20),
        },
        {
            "set_id": "set_orange",
            "label": "Orange",
            "row": 1,
            "col": 0,
            "x0_frac": 0.00,
            "x1_frac": 0.58,
            "y0_frac": 0.42,
            "y1_frac": 1.00,
            "colour_bgr": (0, 140, 255),
        },
        {
            "set_id": "set_green",
            "label": "Green",
            "row": 1,
            "col": 1,
            "x0_frac": 0.42,
            "x1_frac": 1.00,
            "y0_frac": 0.42,
            "y1_frac": 1.00,
            "colour_bgr": (60, 180, 60),
        },
    )


    @dataclass
    class QuadrantResult:
        image_name: str
        lighting: str
        angle: str
        sheet_state: str
        capture_id: str
        set_id: str
        status: str
        confidence: str | None
        score: float | None
        candidate_count: int | None
        contour_area_px: float | None
        area_frac: float | None
        bbox_frac: float | None
        ratio_score: float | None
        area_score: float | None
        extent_score: float | None
        side_score: float | None
        band_score: float | None
        thickness_score: float | None
        corner_score: float | None
        outside_penalty: float | None
        rectified_width_px: int | None
        rectified_height_px: int | None
        scaled_width_px: int | None
        scaled_height_px: int | None
        error: str | None


    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description='Run the 4-up CDS colour benchmark on a folder of JPGs.')
        parser.add_argument(
            '--input-dir',
            default='data/inputs/colour_benchmark',
            help='Folder containing raw benchmark images named <lighting>__<angle>__<sheet_state>__<capture_id>.jpg',
        )
        parser.add_argument(
            '--output-dir',
            default=None,
            help='Optional output directory. Defaults to outputs/colour_benchmark/<timestamp>.',
        )
        parser.add_argument('--template', default='cds_colour_test_260x350', help='Template id for each quadrant.')
        parser.add_argument('--config', default='configs/app/default.yaml', help='Application config YAML.')
        return parser.parse_args()


    def parse_filename(path: Path) -> dict[str, str]:
        stem = path.stem
        parts = stem.split('__')
        if len(parts) != 4:
            raise ValueError(
                f'Expected filename format <lighting>__<angle>__<sheet_state>__<capture_id>, got: {path.name}'
            )
        return dict(zip(FILENAME_PARTS, parts))


    def quadrant_roi(full_image: np.ndarray, spec: dict[str, Any]) -> tuple[np.ndarray, tuple[int, int]]:
        h, w = full_image.shape[:2]
        x0 = int(round(w * spec['x0_frac']))
        x1 = int(round(w * spec['x1_frac']))
        y0 = int(round(h * spec['y0_frac']))
        y1 = int(round(h * spec['y1_frac']))
        x0 = max(0, min(x0, w - 1))
        x1 = max(x0 + 1, min(x1, w))
        y0 = max(0, min(y0, h - 1))
        y1 = max(y0 + 1, min(y1, h))
        return full_image[y0:y1, x0:x1].copy(), (x0, y0)


    def corners_with_offset(local_corners: list[list[float]], offset: tuple[int, int]) -> list[list[float]]:
        ox, oy = offset
        return [[float(x) + ox, float(y) + oy] for x, y in local_corners]


    def annotate_text(image: np.ndarray, origin: tuple[int, int], lines: list[str], colour: tuple[int, int, int]) -> None:
        x, y = origin
        yy = y
        for line in lines:
            cv2.putText(image, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(image, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 1, cv2.LINE_AA)
            yy += 24


    def make_contact_sheet(images: list[np.ndarray], labels: list[str], tile_w: int = 1000, tile_h: int = 740) -> np.ndarray:
        tiles = []
        for image, label in zip(images, labels):
            tile = ensure_uint8_bgr(image)
            tile = cv2.resize(tile, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            cv2.putText(tile, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(tile, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
        top = np.hstack(tiles[:2])
        bottom = np.hstack(tiles[2:])
        return np.vstack([top, bottom])


    def summarise_results(rows: list[QuadrantResult]) -> dict[str, Any]:
        per_set: dict[str, dict[str, Any]] = {}
        for set_id in [spec['set_id'] for spec in QUADRANTS]:
            subset = [r for r in rows if r.set_id == set_id]
            successes = [r for r in subset if r.status == 'success']
            per_set[set_id] = {
                'images_seen': len(subset),
                'successes': len(successes),
                'failures': len(subset) - len(successes),
                'mean_score': round(float(np.mean([r.score for r in successes])) if successes else 0.0, 4),
                'mean_candidate_count': round(float(np.mean([r.candidate_count for r in successes])) if successes else 0.0, 2),
                'high_confidence': sum(1 for r in successes if r.confidence == 'high'),
                'low_confidence': sum(1 for r in successes if r.confidence == 'low'),
            }
        return per_set


    def write_csv(path: Path, rows: list[QuadrantResult]) -> None:
        fieldnames = list(asdict(rows[0]).keys()) if rows else list(QuadrantResult.__annotations__.keys())
        with path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))


    def main() -> int:
        args = parse_args()

        app_config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
        template = load_template_config(args.template)
        preferred_output_dpi = int(template.preferred_output_dpi or app_config['processing']['preferred_output_dpi'])
        expected_aspect_ratio = float(template.design_area_mm.width) / float(template.design_area_mm.height)

        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f'Input directory not found: {input_dir}')

        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = Path('outputs/colour_benchmark') / datetime.now().strftime('%Y-%m-%d_%H%M%S')
        output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(
            p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        )
        if not image_paths:
            raise RuntimeError(f'No supported image files found in: {input_dir}')

        rows: list[QuadrantResult] = []
        image_summaries: list[dict[str, Any]] = []

        for image_path in image_paths:
            meta = parse_filename(image_path)
            image, input_meta = load_input(image_path, pdf_dpi=preferred_output_dpi)
            overlay = image.copy()
            rectified_tiles: list[np.ndarray] = []
            tile_labels: list[str] = []
            per_image_rows: list[dict[str, Any]] = []

            image_out_dir = output_dir / image_path.stem
            image_out_dir.mkdir(parents=True, exist_ok=True)
            save_image(image_out_dir / '00_input.jpg', image)

            for spec in QUADRANTS:
                roi, offset = quadrant_roi(image, spec)
                quadrant_dir = image_out_dir / spec['set_id']
                quadrant_dir.mkdir(parents=True, exist_ok=True)
                save_image(quadrant_dir / '01_roi.jpg', roi)

                try:
                    border = detect_outer_border(roi, expected_aspect_ratio=expected_aspect_ratio)
                    local_corners = np.asarray(border.ordered_corners_xy, dtype=np.float32)
                    global_corners = corners_with_offset(border.ordered_corners_xy, offset)

                    rectified = rectify_from_corners(roi, local_corners, expected_aspect_ratio=expected_aspect_ratio)
                    scaled = scale_to_template_design_area(
                        rectified.image,
                        width_mm=template.design_area_mm.width,
                        height_mm=template.design_area_mm.height,
                        output_dpi=preferred_output_dpi,
                    )

                    save_image(quadrant_dir / '02_rectified.jpg', rectified.image)
                    save_image(quadrant_dir / '03_scaled.jpg', scaled.image)

                    rectified_tiles.append(rectified.image)
                    tile_labels.append(
                        f"{spec['set_id']} | {border.confidence} | score={border.score:.3f}"
                    )

                    overlay = draw_polygon(overlay, np.asarray(global_corners, dtype=np.float32), spec['colour_bgr'], thickness=6)
                    label_x = int(min(pt[0] for pt in global_corners)) + 12
                    label_y = int(min(pt[1] for pt in global_corners)) + 28
                    annotate_text(
                        overlay,
                        (label_x, label_y),
                        [
                            spec['set_id'],
                            f"conf={border.confidence}",
                            f"score={border.score:.3f}",
                        ],
                        spec['colour_bgr'],
                    )

                    diagnostics = border.diagnostics or {}
                    row = QuadrantResult(
                        image_name=image_path.name,
                        lighting=meta['lighting'],
                        angle=meta['angle'],
                        sheet_state=meta['sheet_state'],
                        capture_id=meta['capture_id'],
                        set_id=spec['set_id'],
                        status='success',
                        confidence=border.confidence,
                        score=float(border.score),
                        candidate_count=int(border.candidate_count),
                        contour_area_px=float(border.contour_area_px),
                        area_frac=float(diagnostics.get('area_frac')) if diagnostics.get('area_frac') is not None else None,
                        bbox_frac=float(diagnostics.get('bbox_frac')) if diagnostics.get('bbox_frac') is not None else None,
                        ratio_score=float(diagnostics.get('ratio_score')) if diagnostics.get('ratio_score') is not None else None,
                        area_score=float(diagnostics.get('area_score')) if diagnostics.get('area_score') is not None else None,
                        extent_score=float(diagnostics.get('extent_score')) if diagnostics.get('extent_score') is not None else None,
                        side_score=float(diagnostics.get('side_score')) if diagnostics.get('side_score') is not None else None,
                        band_score=float(diagnostics.get('band_score')) if diagnostics.get('band_score') is not None else None,
                        thickness_score=float(diagnostics.get('thickness_score')) if diagnostics.get('thickness_score') is not None else None,
                        corner_score=float(diagnostics.get('corner_score')) if diagnostics.get('corner_score') is not None else None,
                        outside_penalty=float(diagnostics.get('outside_penalty')) if diagnostics.get('outside_penalty') is not None else None,
                        rectified_width_px=int(rectified.rectified_size_px[0]),
                        rectified_height_px=int(rectified.rectified_size_px[1]),
                        scaled_width_px=int(scaled.output_size_px[0]),
                        scaled_height_px=int(scaled.output_size_px[1]),
                        error=None,
                    )
                except Exception as exc:
                    failure_preview = ensure_uint8_bgr(roi)
                    annotate_text(failure_preview, (20, 40), [spec['set_id'], 'status=fail', str(exc)], (0, 0, 255))
                    save_image(quadrant_dir / '02_failure.jpg', failure_preview)
                    rectified_tiles.append(failure_preview)
                    tile_labels.append(f"{spec['set_id']} | fail")
                    row = QuadrantResult(
                        image_name=image_path.name,
                        lighting=meta['lighting'],
                        angle=meta['angle'],
                        sheet_state=meta['sheet_state'],
                        capture_id=meta['capture_id'],
                        set_id=spec['set_id'],
                        status='fail',
                        confidence=None,
                        score=None,
                        candidate_count=None,
                        contour_area_px=None,
                        area_frac=None,
                        bbox_frac=None,
                        ratio_score=None,
                        area_score=None,
                        extent_score=None,
                        side_score=None,
                        band_score=None,
                        thickness_score=None,
                        corner_score=None,
                        outside_penalty=None,
                        rectified_width_px=None,
                        rectified_height_px=None,
                        scaled_width_px=None,
                        scaled_height_px=None,
                        error=str(exc),
                    )

                rows.append(row)
                per_image_rows.append(asdict(row))

            save_image(image_out_dir / '10_overlay.jpg', overlay)
            contact = make_contact_sheet(rectified_tiles, tile_labels)
            save_image(image_out_dir / '11_contact_sheet.jpg', contact)

            image_summaries.append(
                {
                    'image_name': image_path.name,
                    'input_meta': input_meta,
                    'parsed_filename': meta,
                    'results': per_image_rows,
                    'files': {
                        'input': '00_input.jpg',
                        'overlay': '10_overlay.jpg',
                        'contact_sheet': '11_contact_sheet.jpg',
                    },
                }
            )

        write_csv(output_dir / 'benchmark_results.csv', rows)
        (output_dir / 'benchmark_results.json').write_text(
            json.dumps([asdict(r) for r in rows], indent=2), encoding='utf-8'
        )
        (output_dir / 'benchmark_summary.json').write_text(
            json.dumps(
                {
                    'input_dir': str(input_dir),
                    'template': template.template_id,
                    'quadrant_assignment': {
                        'top_left': 'set_magenta',
                        'top_right': 'set_cyan',
                        'bottom_left': 'set_orange',
                        'bottom_right': 'set_green',
                    },
                    'per_set': summarise_results(rows),
                    'image_summaries': image_summaries,
                },
                indent=2,
            ),
            encoding='utf-8',
        )

        print(f'Colour benchmark complete: {output_dir}')
        return 0


    if __name__ == '__main__':
        raise SystemExit(main())
    ''').strip() + "\n"

    readme = dedent('''
    Colour benchmark harness notes
    ==============================

    Input filenames must follow:
        <lighting>__<angle>__<sheet_state>__<capture_id>.jpg

    Example:
        lowlight__goodangle__flat__01.jpg

    This harness assumes a single A1 photo contains all four 260x350 colour-test sets:
        - top-left: set_magenta
        - top-right: set_cyan
        - bottom-left: set_orange
        - bottom-right: set_green

    It does not try to detect the outer A1 page first. Instead it searches four overlapping
    quadrant ROIs directly, which is simpler and more robust for this benchmark stage.
    ''').strip() + "\n"

    write_text(repo_root / 'tools' / 'benchmark_phase1.py', benchmark_tool)
    write_text(repo_root / 'docs' / 'COLOUR_BENCHMARK_NOTES.md', readme)

    print('Patched: tools/benchmark_phase1.py')
    print('Patched: docs/COLOUR_BENCHMARK_NOTES.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
