from __future__ import annotations

from pathlib import Path
import shutil
import sys
import textwrap

ROOT_SENTINELS = ["pyproject.toml", "configs", "src", "tests"]
BACKUP_SUFFIX = ".bak_phase1_border_mode_overhaul"

FILES: dict[str, str] = {
    "src/armourcore_cds/templates/models.py": textwrap.dedent(
        '''\
        from __future__ import annotations

        from typing import Literal

        from pydantic import BaseModel, ConfigDict, Field


        BorderColourMode = Literal["auto", "black", "orange"]


        class DesignArea(BaseModel):
            width: float
            height: float


        class GridModel(BaseModel):
            minor_spacing_mm: float = 10.0
            major_spacing_mm: float = 50.0


        class ColourHintsModel(BaseModel):
            outer_border_hex_candidates: list[str] = Field(default_factory=list)
            fiducial_hex_candidates: list[str] = Field(default_factory=list)


        class BorderDetectionModel(BaseModel):
            border_colour_mode: BorderColourMode | None = None
            use_colour_hint: bool | None = None
            use_shape_constraints: bool | None = None
            fallback_to_fiducials: bool | None = None
            colour_hints: ColourHintsModel | None = None


        class TemplateOutputsModel(BaseModel):
            write_rectified: bool | None = None
            write_scaled: bool | None = None
            write_cropped: bool | None = None
            write_debug: bool | None = None


        class TemplateModel(BaseModel):
            model_config = ConfigDict(extra="ignore")

            template_id: str
            display_name: str
            design_area_mm: DesignArea
            crop_rule: str
            preferred_output_dpi: int
            primary_geometry_truth: str
            fiducials_enabled: bool
            grid: GridModel | None = None
            border_detection: BorderDetectionModel | None = None
            outputs: TemplateOutputsModel | None = None
        '''
    ),
    "configs/app/default.yaml": textwrap.dedent(
        '''\
        app_name: armourcore-cds-digitiser
        mode: development
        log_level: INFO

        output:
          base_dir: outputs/runs
          timestamp_runs: true
          write_debug_images: true
          write_summary_text: true
          write_run_report: true

        processing:
          default_template: cds_regular_500x600
          preferred_output_dpi: 400
          border_colour_mode_override: null
          orange_border_hex: "#D35400"
          preserve_customer_dark_marks: true
          single_file_mode: true

        phase_toggles:
          phase1_rectify_clean: true
          phase2_trace_extract: false
          phase3_vectorise: false
          phase4_geometry_rules: false
        '''
    ),
    "configs/templates/cds_regular_500x600.yaml": textwrap.dedent(
        '''\
        template_id: cds_regular_500x600
        display_name: cds_regular_500x600

        design_area_mm:
          width: 600
          height: 500

        crop_rule: inside_most_edge_of_thick_outer_border
        preferred_output_dpi: 400
        primary_geometry_truth: outer_border
        fiducials_enabled: true

        grid:
          minor_spacing_mm: 10
          major_spacing_mm: 50

        border_detection:
          border_colour_mode: black
          use_colour_hint: false
          use_shape_constraints: true
          fallback_to_fiducials: true
          colour_hints:
            fiducial_hex_candidates: ["#FF0000"]

        outputs:
          write_rectified: true
          write_scaled: true
          write_cropped: true
          write_debug: true
        '''
    ),
    "configs/templates/cds_xlarge_500x900.yaml": textwrap.dedent(
        '''\
        template_id: cds_xlarge_500x900
        display_name: cds_xlarge_500x900

        design_area_mm:
          width: 900
          height: 500

        crop_rule: inside_most_edge_of_thick_outer_border
        preferred_output_dpi: 400
        primary_geometry_truth: outer_border
        fiducials_enabled: true

        grid:
          minor_spacing_mm: 10
          major_spacing_mm: 50

        border_detection:
          border_colour_mode: black
          use_colour_hint: false
          use_shape_constraints: true
          fallback_to_fiducials: true
          colour_hints:
            fiducial_hex_candidates: ["#FF0000"]

        outputs:
          write_rectified: true
          write_scaled: true
          write_cropped: true
          write_debug: true
        '''
    ),
    "configs/templates/cds_colour_test_260x350.yaml": textwrap.dedent(
        '''\
        template_id: cds_colour_test_260x350
        display_name: cds_colour_test_260x350

        design_area_mm:
          width: 350
          height: 260

        crop_rule: inside_most_edge_of_thick_outer_border
        preferred_output_dpi: 400
        primary_geometry_truth: outer_border
        fiducials_enabled: true

        grid:
          minor_spacing_mm: 10
          major_spacing_mm: 50

        border_detection:
          border_colour_mode: orange
          use_colour_hint: true
          use_shape_constraints: true
          fallback_to_fiducials: true
          colour_hints:
            outer_border_hex_candidates: ["#D35400", "#C2185B", "#007C91", "#2E7D32"]
            fiducial_hex_candidates: ["#FF0000"]

        outputs:
          write_rectified: true
          write_scaled: true
          write_cropped: true
          write_debug: true
        '''
    ),
    "src/armourcore_cds/phase1/pipeline.py": textwrap.dedent(
        '''\
        """Phase 1 pipeline implementation."""

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
        from armourcore_cds.utils.image_ops import draw_polygon, resize_long_edge, save_image, stack_debug_h


        def _corners_to_serialisable(corners: object) -> list[list[float]]:
            arr = np.asarray(corners, dtype=np.float32).reshape(4, 2)
            return [[round(float(x), 2), round(float(y), 2)] for x, y in arr.tolist()]


        def _normalise_mode(value: object) -> str | None:
            if value is None:
                return None
            mode = str(value).strip().lower()
            if mode in {"auto", "black", "orange"}:
                return mode
            return None


        def run_phase1_pipeline(input_path: Path, template_id: str, config_path: Path) -> Path:
            app_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            template = load_template_config(template_id)

            run_name = f"{datetime.now():%Y-%m-%d_%H%M%S}_{input_path.stem}"
            base_dir = Path(app_config["output"]["base_dir"])
            run_dir = ensure_run_dir(base_dir, run_name)
            debug = DebugWriter(run_dir / "debug", enabled=bool(app_config["output"].get("write_debug_images", True)))

            processing_cfg = app_config.get("processing", {})
            preferred_output_dpi = int(template.preferred_output_dpi or processing_cfg["preferred_output_dpi"])
            input_image, input_meta = load_input(input_path, pdf_dpi=preferred_output_dpi)
            debug.image("01_input_original", input_image)

            expected_aspect_ratio = float(template.design_area_mm.width) / float(template.design_area_mm.height)

            template_border_cfg = template.border_detection
            template_colour_hints = getattr(template_border_cfg, "colour_hints", None)

            template_mode = _normalise_mode(getattr(template_border_cfg, "border_colour_mode", None))
            override_mode = _normalise_mode(processing_cfg.get("border_colour_mode_override"))
            legacy_override_mode = _normalise_mode(processing_cfg.get("border_colour_mode"))
            resolved_border_mode = override_mode or legacy_override_mode or template_mode or "auto"

            use_colour_hint = True if getattr(template_border_cfg, "use_colour_hint", None) is None else bool(template_border_cfg.use_colour_hint)
            use_shape_constraints = True if getattr(template_border_cfg, "use_shape_constraints", None) is None else bool(template_border_cfg.use_shape_constraints)
            fallback_to_fiducials = True if getattr(template_border_cfg, "fallback_to_fiducials", None) is None else bool(template_border_cfg.fallback_to_fiducials)

            outer_border_hex_candidates = list(getattr(template_colour_hints, "outer_border_hex_candidates", []) or [])
            fiducial_hex_candidates = list(getattr(template_colour_hints, "fiducial_hex_candidates", []) or [])
            orange_border_hex = str(processing_cfg.get("orange_border_hex", "#D35400")).strip()

            border = detect_outer_border(
                input_image,
                expected_aspect_ratio=expected_aspect_ratio,
                use_colour_hint=use_colour_hint,
                use_shape_constraints=use_shape_constraints,
                fallback_to_fiducials=fallback_to_fiducials,
                outer_border_hex_candidates=outer_border_hex_candidates,
                fiducial_hex_candidates=fiducial_hex_candidates,
                border_colour_mode=resolved_border_mode,
                orange_border_hex=orange_border_hex,
            )

            contour_overlay = draw_polygon(input_image, border.ordered_corners_xy, (0, 255, 0), thickness=6)
            debug.image("04_border_detected", contour_overlay)

            rectified = rectify_from_corners(input_image, border.ordered_corners_xy, expected_aspect_ratio=expected_aspect_ratio)
            debug.image("05_rectified_outer", rectified.image)

            design_area_preview = rectified.image.copy()
            h, w = design_area_preview.shape[:2]
            cv2.rectangle(design_area_preview, (0, 0), (w - 1, h - 1), (0, 255, 0), 4)
            debug.image("06_design_area_preview", design_area_preview)
            debug.image("07_design_area_raw", rectified.image)

            scaled = scale_to_template_design_area(
                rectified.image,
                width_mm=template.design_area_mm.width,
                height_mm=template.design_area_mm.height,
                output_dpi=preferred_output_dpi,
            )
            debug.image("08_scaled_design_area", scaled.image)

            save_image(run_dir / "rectified_outer.png", rectified.image)
            save_image(run_dir / "cropped_design_area_raw.png", rectified.image)
            save_image(run_dir / "scaled_design_area.png", scaled.image)

            summary_preview = stack_debug_h(
                resize_long_edge(contour_overlay, 1400),
                resize_long_edge(design_area_preview, 1400),
            )
            summary_lines = [
                f"template={template.template_id}",
                f"input={input_meta['input_kind']}",
                f"border_mode={resolved_border_mode}",
                "crop_mode=none_use_detected_border",
                f"border_confidence={border.confidence}",
                f"border_score={border.score:.3f}",
                f"output_px={scaled.output_size_px[0]}x{scaled.output_size_px[1]}",
            ]
            summary_preview = debug.text_overlay(summary_preview, summary_lines)
            debug.image("10_summary_preview", summary_preview)

            report = {
                "input_path": str(input_path),
                "template_id": template.template_id,
                "template_display_name": template.display_name,
                "phase": "phase1_rectify_clean",
                "status": "success",
                "input": {
                    **input_meta,
                    "width_px": int(input_image.shape[1]),
                    "height_px": int(input_image.shape[0]),
                },
                "design_area_mm": {
                    "width": template.design_area_mm.width,
                    "height": template.design_area_mm.height,
                },
                "border_detection": {
                    "ordered_corners_xy": _corners_to_serialisable(border.ordered_corners_xy),
                    "contour_area_px": float(border.contour_area_px),
                    "score": float(border.score),
                    "candidate_count": int(border.candidate_count),
                    "confidence": border.confidence,
                    "template_border_mode": template_mode,
                    "resolved_border_mode": resolved_border_mode,
                    "requested_colour_mode": (border.diagnostics or {}).get("requested_colour_mode", resolved_border_mode),
                    "detected_border_mode": (border.diagnostics or {}).get("detected_border_mode", "unknown"),
                    "diagnostics": border.diagnostics or {},
                },
                "rectification": {
                    "rectified_outer_size_px": {
                        "width": rectified.rectified_size_px[0],
                        "height": rectified.rectified_size_px[1],
                    },
                    "crop_mode": "none_use_detected_border",
                    "design_area_xyxy_px": {
                        "x0": 0,
                        "y0": 0,
                        "x1": rectified.rectified_size_px[0] - 1,
                        "y1": rectified.rectified_size_px[1] - 1,
                    },
                    "design_area_margins_px": {
                        "top": 0,
                        "bottom": 0,
                        "left": 0,
                        "right": 0,
                    },
                },
                "scaling": {
                    "output_dpi": preferred_output_dpi,
                    "scaled_size_px": {
                        "width": scaled.output_size_px[0],
                        "height": scaled.output_size_px[1],
                    },
                    "px_per_mm_x": scaled.px_per_mm_x,
                    "px_per_mm_y": scaled.px_per_mm_y,
                    "effective_dpi_x": scaled.effective_dpi_x,
                    "effective_dpi_y": scaled.effective_dpi_y,
                },
                "outputs": {
                    "rectified_outer": "rectified_outer.png",
                    "cropped_design_area_raw": "cropped_design_area_raw.png",
                    "scaled_design_area": "scaled_design_area.png",
                    "debug_dir": "debug",
                },
                "assumptions": [
                    "The detected thick-border quadrilateral is the geometry truth for rectification, crop and scale.",
                    "Calibration squares are support features for border confidence only and do not define the output crop.",
                    "Phase 01 intentionally skips any secondary crop to avoid locking onto customer-drawn geometry.",
                ],
            }
            write_run_report(run_dir / "run_report.json", report)
            return run_dir
        '''
    ),
    "src/armourcore_cds/phase1/boundary_detection.py": textwrap.dedent(
        '''\
        from __future__ import annotations

        from dataclasses import dataclass
        from typing import Any

        import cv2
        import numpy as np


        @dataclass
        class BorderDetectionResult:
            ordered_corners_xy: list[list[float]]
            contour_area_px: float
            score: float
            candidate_count: int
            confidence: str = "high"
            diagnostics: dict[str, Any] | None = None


        def _hex_to_bgr(hex_colour: str) -> tuple[int, int, int]:
            s = hex_colour.strip().lstrip("#")
            if len(s) != 6:
                raise ValueError(f"Expected 6-digit hex colour, got: {hex_colour!r}")
            r = int(s[0:2], 16)
            g = int(s[2:4], 16)
            b = int(s[4:6], 16)
            return (b, g, r)


        def _hex_to_hsv(hex_colour: str) -> tuple[int, int, int]:
            bgr = np.uint8([[list(_hex_to_bgr(hex_colour))]])
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0]
            return int(hsv[0]), int(hsv[1]), int(hsv[2])


        def _order_quad_points(pts: np.ndarray) -> np.ndarray:
            pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
            s = pts.sum(axis=1)
            d = np.diff(pts, axis=1).reshape(-1)
            ordered = np.zeros((4, 2), dtype=np.float32)
            ordered[0] = pts[np.argmin(s)]  # top-left
            ordered[1] = pts[np.argmin(d)]  # top-right
            ordered[2] = pts[np.argmax(s)]  # bottom-right
            ordered[3] = pts[np.argmax(d)]  # bottom-left
            return ordered


        def _quad_side_lengths(quad: np.ndarray) -> tuple[float, float, float, float]:
            tl, tr, br, bl = quad
            top = float(np.linalg.norm(tr - tl))
            right = float(np.linalg.norm(br - tr))
            bottom = float(np.linalg.norm(br - bl))
            left = float(np.linalg.norm(bl - tl))
            return top, right, bottom, left


        def _quad_area(quad: np.ndarray) -> float:
            return float(abs(cv2.contourArea(np.asarray(quad, dtype=np.float32))))


        def _sample_line_single(
            channel_2d: np.ndarray,
            p0: np.ndarray,
            p1: np.ndarray,
            *,
            half_width: int = 3,
            samples: int = 256,
        ) -> np.ndarray:
            h, w = channel_2d.shape[:2]
            xs = np.linspace(float(p0[0]), float(p1[0]), samples)
            ys = np.linspace(float(p0[1]), float(p1[1]), samples)
            dx = float(p1[0] - p0[0])
            dy = float(p1[1] - p0[1])
            length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
            nx = -dy / length
            ny = dx / length
            strips = []
            for off in range(-half_width, half_width + 1):
                xso = np.clip(np.round(xs + nx * off).astype(int), 0, w - 1)
                yso = np.clip(np.round(ys + ny * off).astype(int), 0, h - 1)
                strips.append(channel_2d[yso, xso].astype(np.float32))
            return np.stack(strips, axis=0)


        def _sample_line_mask(
            mask: np.ndarray,
            p0: np.ndarray,
            p1: np.ndarray,
            *,
            half_width: int = 2,
            samples: int = 256,
        ) -> np.ndarray:
            return _sample_line_single(mask, p0, p1, half_width=half_width, samples=samples)


        def _side_evidence(gray: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> tuple[float, float]:
            strip = _sample_line_single(gray, p0, p1, half_width=4, samples=256)
            centre = strip[strip.shape[0] // 2]
            darkness = 1.0 - float(np.mean(centre) / 255.0)
            support = float(np.mean(centre < 190))
            band_support = float(np.mean(np.mean(strip < 200, axis=0) > 0.25))

            good = (np.mean(strip < 200, axis=0) > 0.25).astype(np.uint8)
            longest = 0
            current = 0
            for v in good:
                if v:
                    current += 1
                    longest = max(longest, current)
                else:
                    current = 0
            longest_run = float(longest / max(len(good), 1))
            score = 0.32 * darkness + 0.26 * support + 0.26 * band_support + 0.16 * longest_run
            return score, band_support


        def _corner_support(gray: np.ndarray, corner: np.ndarray, radius: int = 36) -> float:
            h, w = gray.shape[:2]
            x = int(round(float(corner[0])))
            y = int(round(float(corner[1])))
            x0 = max(0, x - radius)
            x1 = min(w, x + radius + 1)
            y0 = max(0, y - radius)
            y1 = min(h, y + radius + 1)
            patch = gray[y0:y1, x0:x1]
            if patch.size == 0:
                return 0.0
            dark = float(np.mean(patch < 195))
            edge = cv2.Canny(patch, 60, 160)
            edge_density = float(np.mean(edge > 0))
            return 0.60 * dark + 0.40 * edge_density


        def _clear_image_border(mask: np.ndarray, border_px: int) -> np.ndarray:
            out = mask.copy()
            b = max(1, int(border_px))
            out[:b, :] = 0
            out[-b:, :] = 0
            out[:, :b] = 0
            out[:, -b:] = 0
            return out


        def _occupancy_band_score(frac: float, lo: float, peak_lo: float, peak_hi: float, hi: float) -> float:
            if frac <= lo or frac >= hi:
                return 0.0
            if peak_lo <= frac <= peak_hi:
                return 1.0
            if frac < peak_lo:
                return float((frac - lo) / max(peak_lo - lo, 1e-6))
            return float((hi - frac) / max(hi - peak_hi, 1e-6))


        def _is_frame_like_quad(quad: np.ndarray, w: int, h: int) -> bool:
            q = np.asarray(quad, dtype=np.float32)
            tol = max(8.0, 0.01 * min(w, h))
            minx = float(np.min(q[:, 0]))
            maxx = float(np.max(q[:, 0]))
            miny = float(np.min(q[:, 1]))
            maxy = float(np.max(q[:, 1]))
            touches_all_edges = minx <= tol and miny <= tol and maxx >= (w - 1 - tol) and maxy >= (h - 1 - tol)
            area_frac = _quad_area(q) / max(float(w * h), 1.0)
            return bool(touches_all_edges or area_frac >= 0.975)


        def _collect_candidate_quads(
            mask: np.ndarray,
            min_area_px: float,
            source: str,
            *,
            retrieval_mode: int = cv2.RETR_LIST,
        ) -> list[dict[str, Any]]:
            contours, _ = cv2.findContours(mask, retrieval_mode, cv2.CHAIN_APPROX_SIMPLE)
            candidates: list[dict[str, Any]] = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area_px:
                    continue
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                quads: list[np.ndarray] = []
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    quads.append(approx.reshape(4, 2).astype(np.float32))
                rect = cv2.minAreaRect(contour)
                quads.append(cv2.boxPoints(rect).astype(np.float32))
                for quad in quads:
                    ordered = _order_quad_points(quad)
                    qa = _quad_area(ordered)
                    if qa < min_area_px:
                        continue
                    candidates.append(
                        {
                            "quad": ordered,
                            "contour_area_px": float(area),
                            "source": source,
                        }
                    )
            return candidates


        def _make_dark_mask(gray: np.ndarray, border_px: int) -> np.ndarray:
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(blur)
            dark_mask = cv2.inRange(clahe, 0, 170)
            dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
            dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
            return _clear_image_border(dark_mask, border_px)


        def _make_edge_mask(gray: np.ndarray, border_px: int) -> np.ndarray:
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(blur)
            edges = cv2.Canny(clahe, 60, 180)
            edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
            return _clear_image_border(edges, border_px)


        def _merge_masks(*masks: np.ndarray) -> np.ndarray:
            out = np.zeros_like(masks[0])
            for mask in masks:
                out = cv2.bitwise_or(out, mask)
            return out


        def _build_colour_mask_from_hex_candidates(image_bgr: np.ndarray, hex_candidates: list[str]) -> np.ndarray:
            hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
            masks: list[np.ndarray] = []
            for hex_colour in hex_candidates:
                try:
                    hh, ss, vv = _hex_to_hsv(hex_colour)
                except Exception:
                    continue
                lower = np.array([max(0, hh - 18), max(10, ss - 130), max(45, vv - 120)], dtype=np.uint8)
                upper = np.array([min(179, hh + 18), 255, min(255, vv + 120)], dtype=np.uint8)
                masks.append(cv2.inRange(hsv, lower, upper))
            if not masks:
                return np.zeros(hsv.shape[:2], dtype=np.uint8)
            return _merge_masks(*masks)


        def _make_orange_mask(
            image_bgr: np.ndarray,
            outer_border_hex_candidates: list[str] | None,
            orange_border_hex: str,
        ) -> np.ndarray:
            hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
            masks: list[np.ndarray] = []

            if outer_border_hex_candidates:
                masks.append(_build_colour_mask_from_hex_candidates(image_bgr, list(outer_border_hex_candidates)))
            else:
                masks.append(_build_colour_mask_from_hex_candidates(image_bgr, [orange_border_hex]))

            lower1 = np.array([4, 18, 70], dtype=np.uint8)
            upper1 = np.array([24, 200, 230], dtype=np.uint8)
            masks.append(cv2.inRange(hsv, lower1, upper1))

            h0, s0, v0 = _hex_to_hsv(orange_border_hex)
            lower2 = np.array([max(0, h0 - 18), max(10, s0 - 120), max(45, v0 - 120)], dtype=np.uint8)
            upper2 = np.array([min(179, h0 + 18), 255, min(255, v0 + 120)], dtype=np.uint8)
            masks.append(cv2.inRange(hsv, lower2, upper2))

            lower3 = np.array([0, 10, 80], dtype=np.uint8)
            upper3 = np.array([18, 170, 235], dtype=np.uint8)
            masks.append(cv2.inRange(hsv, lower3, upper3))

            mask = _merge_masks(*masks)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
            mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
            return mask


        def _make_orange_border_mask(
            image_bgr: np.ndarray,
            outer_border_hex_candidates: list[str] | None,
            orange_border_hex: str,
        ) -> np.ndarray:
            orange_mask = _make_orange_mask(image_bgr, outer_border_hex_candidates, orange_border_hex)
            border_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
            border_mask = cv2.morphologyEx(border_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
            border_mask = cv2.dilate(border_mask, np.ones((3, 3), np.uint8), iterations=1)
            return border_mask


        def _marker_support_score(image_bgr: np.ndarray, quad: np.ndarray, fiducial_hex_candidates: list[str] | None) -> float:
            if not fiducial_hex_candidates:
                return 0.0
            h, w = image_bgr.shape[:2]
            hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
            masks: list[np.ndarray] = []
            for hex_colour in fiducial_hex_candidates:
                try:
                    hh, ss, vv = _hex_to_hsv(hex_colour)
                except Exception:
                    continue
                lower = np.array([max(0, hh - 12), max(20, ss - 130), max(40, vv - 150)], dtype=np.uint8)
                upper = np.array([min(179, hh + 12), 255, 255], dtype=np.uint8)
                masks.append(cv2.inRange(hsv, lower, upper))
            if not masks:
                return 0.0
            mask = _merge_masks(*masks)
            scores = []
            radius = max(16, int(round(0.025 * min(h, w))))
            for corner in quad:
                x = int(round(float(corner[0])))
                y = int(round(float(corner[1])))
                x0 = max(0, x - radius)
                x1 = min(w, x + radius + 1)
                y0 = max(0, y - radius)
                y1 = min(h, y + radius + 1)
                patch = mask[y0:y1, x0:x1]
                if patch.size == 0:
                    scores.append(0.0)
                    continue
                scores.append(float(np.mean(patch > 0)))
            scores = sorted(scores, reverse=True)
            if not scores:
                return 0.0
            return float(np.mean(scores[:3]))


        def _outside_envelope_penalty(gray: np.ndarray, quad: np.ndarray) -> float:
            penalties = []
            pts = [quad[0], quad[1], quad[2], quad[3]]
            for a, b in zip(pts, pts[1:] + pts[:1]):
                inner = _sample_line_single(gray, a, b, half_width=2, samples=192)
                outer = _sample_line_single(gray, a, b, half_width=10, samples=192)
                inner_dark = 1.0 - float(np.mean(inner[inner.shape[0] // 2]) / 255.0)
                outer_dark = 1.0 - float(np.mean(outer[0]) / 255.0)
                penalties.append(max(0.0, outer_dark - inner_dark))
            return float(np.mean(penalties))


        def _orange_band_width_score(orange_border_mask: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> float:
            strip = _sample_line_mask(orange_border_mask, p0, p1, half_width=8, samples=192)
            row_occupancy = np.mean(strip > 0, axis=1)
            width_frac = float(np.mean(row_occupancy > 0.20))
            return _occupancy_band_score(width_frac, lo=0.08, peak_lo=0.16, peak_hi=0.55, hi=0.90)


        def _orange_parallel_penalty(orange_mask: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> float:
            strip = _sample_line_mask(orange_mask, p0, p1, half_width=18, samples=160)
            row_occupancy = np.mean(strip > 0, axis=1)
            centre = row_occupancy.shape[0] // 2
            guard = 4
            far_rows = np.concatenate([row_occupancy[: max(0, centre - guard)], row_occupancy[min(row_occupancy.shape[0], centre + guard + 1) :]])
            if far_rows.size == 0:
                return 0.0
            return float(np.max(far_rows))


        def _build_black_candidates(gray: np.ndarray, border_px: int, min_area_px: float) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
            edges = _make_edge_mask(gray, border_px)
            dark_mask = _make_dark_mask(gray, border_px)
            combined = _merge_masks(edges, dark_mask)
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
            combined = _clear_image_border(combined, border_px)

            raw_candidates: list[dict[str, Any]] = []
            raw_candidates.extend(_collect_candidate_quads(edges, min_area_px=min_area_px, source="edges"))
            raw_candidates.extend(_collect_candidate_quads(dark_mask, min_area_px=min_area_px, source="dark"))
            raw_candidates.extend(_collect_candidate_quads(combined, min_area_px=min_area_px, source="combined"))
            return raw_candidates, {"orange_mask": np.zeros_like(edges), "orange_border_mask": np.zeros_like(edges)}


        def _build_orange_candidates(
            gray: np.ndarray,
            image_bgr: np.ndarray,
            border_px: int,
            min_area_px: float,
            outer_border_hex_candidates: list[str] | None,
            orange_border_hex: str,
        ) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
            edges = _make_edge_mask(gray, border_px)
            orange_mask = _make_orange_mask(image_bgr, outer_border_hex_candidates, orange_border_hex)
            orange_mask = _clear_image_border(orange_mask, border_px)
            orange_border_mask = _make_orange_border_mask(image_bgr, outer_border_hex_candidates, orange_border_hex)
            orange_border_mask = _clear_image_border(orange_border_mask, border_px)

            edge_support_region = cv2.dilate(orange_border_mask, np.ones((9, 9), np.uint8), iterations=1)
            orange_supported_edges = cv2.bitwise_and(edges, edge_support_region)
            orange_supported = _merge_masks(orange_border_mask, orange_supported_edges)
            orange_supported = cv2.morphologyEx(orange_supported, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
            orange_supported = _clear_image_border(orange_supported, border_px)

            raw_candidates: list[dict[str, Any]] = []
            raw_candidates.extend(
                _collect_candidate_quads(
                    orange_border_mask,
                    min_area_px=min_area_px * 0.45,
                    source="orange_border",
                    retrieval_mode=cv2.RETR_EXTERNAL,
                )
            )
            raw_candidates.extend(
                _collect_candidate_quads(
                    orange_supported,
                    min_area_px=min_area_px * 0.55,
                    source="orange_supported",
                    retrieval_mode=cv2.RETR_EXTERNAL,
                )
            )
            raw_candidates.extend(
                _collect_candidate_quads(
                    orange_mask,
                    min_area_px=min_area_px * 0.75,
                    source="orange_broad",
                    retrieval_mode=cv2.RETR_EXTERNAL,
                )
            )
            return raw_candidates, {"orange_mask": orange_mask, "orange_border_mask": orange_border_mask}


        def _dedupe_candidates(raw_candidates: list[dict[str, Any]], w: int, h: int) -> list[dict[str, Any]]:
            deduped: list[dict[str, Any]] = []
            seen: set[tuple[int, int, int, int]] = set()
            for cand in raw_candidates:
                quad = cand["quad"]
                if _is_frame_like_quad(quad, w, h):
                    continue
                top, right, bottom, left = _quad_side_lengths(quad)
                cx = int(round(float(np.mean(quad[:, 0])) / 24.0))
                cy = int(round(float(np.mean(quad[:, 1])) / 24.0))
                ww = int(round(((top + bottom) / 2.0) / 24.0))
                hh = int(round(((left + right) / 2.0) / 24.0))
                key = (cx, cy, ww, hh)
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(cand)
            return deduped


        def _mode_specific_colour_score(
            gray: np.ndarray,
            orange_mask: np.ndarray,
            orange_border_mask: np.ndarray,
            quad: np.ndarray,
            requested_mode: str,
        ) -> tuple[str, float, dict[str, float]]:
            pts = [quad[0], quad[1], quad[2], quad[3]]
            black_side_scores = []
            orange_side_scores = []
            orange_band_width_scores = []
            orange_parallel_penalties = []

            for a, b in zip(pts, pts[1:] + pts[:1]):
                gray_strip = _sample_line_single(gray, a, b, half_width=2, samples=256)
                black_side_scores.append(float(np.mean(gray_strip[gray_strip.shape[0] // 2] < 175)))

                if orange_mask.size > 0:
                    orange_strip = _sample_line_mask(orange_mask, a, b, half_width=2, samples=256)
                    orange_side_scores.append(float(np.mean(orange_strip > 0)))
                    orange_band_width_scores.append(_orange_band_width_score(orange_border_mask, a, b))
                    orange_parallel_penalties.append(_orange_parallel_penalty(orange_mask, a, b))

            black_score = float(np.mean(black_side_scores)) if black_side_scores else 0.0
            orange_side_score = float(np.mean(orange_side_scores)) if orange_side_scores else 0.0
            orange_band_width_score = float(np.mean(orange_band_width_scores)) if orange_band_width_scores else 0.0
            orange_parallel_penalty = float(np.mean(orange_parallel_penalties)) if orange_parallel_penalties else 0.0
            orange_score = float(
                np.clip(
                    0.60 * orange_side_score + 0.55 * orange_band_width_score - 0.35 * orange_parallel_penalty,
                    0.0,
                    1.0,
                )
            )

            details = {
                "black_side_score": black_score,
                "orange_side_score": orange_side_score,
                "orange_band_width_score": orange_band_width_score,
                "orange_parallel_penalty": orange_parallel_penalty,
            }

            if requested_mode == "orange":
                return "orange", orange_score, details
            return "black", black_score, details


        def _detect_outer_border_in_mode(
            image_bgr: np.ndarray,
            expected_aspect_ratio: float,
            *,
            use_colour_hint: bool,
            use_shape_constraints: bool,
            fallback_to_fiducials: bool,
            outer_border_hex_candidates: list[str] | None,
            fiducial_hex_candidates: list[str] | None,
            border_colour_mode: str,
            orange_border_hex: str,
        ) -> BorderDetectionResult:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            image_area = float(h * w)
            min_area_px = 0.02 * image_area
            border_px = max(4, int(round(min(h, w) * 0.008)))

            if border_colour_mode == "orange":
                raw_candidates, masks = _build_orange_candidates(
                    gray,
                    image_bgr,
                    border_px,
                    min_area_px,
                    outer_border_hex_candidates if use_colour_hint else None,
                    orange_border_hex,
                )
            else:
                raw_candidates, masks = _build_black_candidates(gray, border_px, min_area_px)

            candidates = _dedupe_candidates(raw_candidates, w, h)
            if not candidates:
                raise RuntimeError(f"Could not detect CDS outer border candidate in {border_colour_mode!r} mode.")

            orange_mask = masks["orange_mask"]
            orange_border_mask = masks["orange_border_mask"]

            best: dict[str, Any] | None = None
            best_score = -1e9
            scored_candidates: list[dict[str, Any]] = []

            for cand in candidates:
                quad = cand["quad"]
                top, right, bottom, left = _quad_side_lengths(quad)
                width = max((top + bottom) / 2.0, 1e-6)
                height = max((left + right) / 2.0, 1e-6)
                ratio = width / height
                ratio_err = abs(np.log(max(ratio, 1e-6) / max(expected_aspect_ratio, 1e-6)))
                ratio_score = max(0.0, 1.0 - ratio_err / 0.40)

                area = _quad_area(quad)
                area_frac = area / image_area
                bbox_frac = (float(np.ptp(quad[:, 0])) * float(np.ptp(quad[:, 1]))) / image_area
                area_score = _occupancy_band_score(area_frac, lo=0.08, peak_lo=0.22, peak_hi=0.78, hi=0.93)
                extent_score = _occupancy_band_score(bbox_frac, lo=0.10, peak_lo=0.25, peak_hi=0.82, hi=0.95)

                side_scores = []
                band_scores = []
                pts = list(quad)
                for p0, p1 in zip(pts, pts[1:] + pts[:1]):
                    s, b = _side_evidence(gray, p0, p1)
                    side_scores.append(s)
                    band_scores.append(b)
                side_score = float(np.mean(side_scores))
                band_score = float(np.mean(band_scores))

                detected_border_mode, colour_score, colour_details = _mode_specific_colour_score(
                    gray=gray,
                    orange_mask=orange_mask,
                    orange_border_mask=orange_border_mask,
                    quad=quad,
                    requested_mode=border_colour_mode,
                )

                marker_support_score = (
                    _marker_support_score(image_bgr=image_bgr, quad=quad, fiducial_hex_candidates=fiducial_hex_candidates)
                    if fallback_to_fiducials
                    else 0.0
                )
                corner_scores = [_corner_support(gray, p) for p in quad]
                corner_score = float(np.mean(sorted(corner_scores, reverse=True)[:3]))
                outside_penalty = _outside_envelope_penalty(gray, quad)

                plausible = bool(
                    area_frac >= 0.10
                    and area_frac <= 0.94
                    and bbox_frac >= 0.16
                    and bbox_frac <= 0.95
                    and ratio_score >= 0.18
                    and (side_score >= 0.16 or band_score >= 0.18)
                )

                page_like_penalty = 0.0
                source_bonus = 0.0
                extra_colour_weight = 0.0
                extra_colour_penalty = 0.0
                size_bonus = 0.0

                if border_colour_mode == "orange":
                    orange_band_width_score = colour_details["orange_band_width_score"]
                    orange_parallel_penalty = colour_details["orange_parallel_penalty"]
                    plausible = bool(plausible and colour_score >= 0.12 and orange_band_width_score >= 0.08)
                    source_bonus = {
                        "orange_border": 0.45,
                        "orange_supported": 0.25,
                        "orange_broad": 0.10,
                    }.get(cand.get("source", ""), 0.0)
                    size_bonus = 0.35 * float(np.clip((area_frac - 0.18) / 0.35, 0.0, 1.0))
                    if area_frac > 0.78:
                        page_like_penalty += (area_frac - 0.78) * 3.5
                    if bbox_frac > 0.87:
                        page_like_penalty += (bbox_frac - 0.87) * 3.2
                    extra_colour_weight = 1.25 * orange_band_width_score
                    extra_colour_penalty = 1.15 * orange_parallel_penalty
                else:
                    source_bonus = 0.15 if cand.get("source") == detected_border_mode else 0.0

                if not plausible and use_shape_constraints:
                    continue

                total = (
                    2.35 * ratio_score
                    + 1.10 * area_score
                    + 0.95 * extent_score
                    + 2.00 * side_score
                    + 1.00 * band_score
                    + 1.70 * colour_score
                    + 0.90 * marker_support_score
                    + 0.70 * corner_score
                    + source_bonus
                    + size_bonus
                    + extra_colour_weight
                    - 1.20 * outside_penalty
                    - page_like_penalty
                    - extra_colour_penalty
                )

                summary = {
                    "score": float(total),
                    "source": cand.get("source", "unknown"),
                    "area_frac": float(area_frac),
                    "bbox_frac": float(bbox_frac),
                    "ratio_score": float(ratio_score),
                    "side_score": float(side_score),
                    "band_score": float(band_score),
                    "colour_score": float(colour_score),
                    "orange_band_width_score": float(colour_details["orange_band_width_score"]),
                    "orange_parallel_penalty": float(colour_details["orange_parallel_penalty"]),
                    "marker_support_score": float(marker_support_score),
                    "page_like_penalty": float(page_like_penalty),
                    "plausible": bool(plausible),
                }
                scored_candidates.append(summary)

                if total > best_score:
                    best_score = total
                    best = {
                        "quad": quad,
                        "contour_area_px": cand["contour_area_px"],
                        "score": float(total),
                        "ratio_score": float(ratio_score),
                        "area_score": float(area_score),
                        "extent_score": float(extent_score),
                        "side_score": float(side_score),
                        "band_score": float(band_score),
                        "colour_score": float(colour_score),
                        "marker_support_score": float(marker_support_score),
                        "corner_score": float(corner_score),
                        "outside_penalty": float(outside_penalty),
                        "page_like_penalty": float(page_like_penalty),
                        "plausible": bool(plausible),
                        "area_frac": float(area_frac),
                        "bbox_frac": float(bbox_frac),
                        "detected_border_mode": detected_border_mode,
                        "source": cand.get("source", "unknown"),
                        "orange_side_score": float(colour_details["orange_side_score"]),
                        "black_side_score": float(colour_details["black_side_score"]),
                        "orange_band_width_score": float(colour_details["orange_band_width_score"]),
                        "orange_parallel_penalty": float(colour_details["orange_parallel_penalty"]),
                    }

            if best is None:
                raise RuntimeError(f"Could not detect CDS outer border candidate in {border_colour_mode!r} mode.")

            confidence = "high"
            if border_colour_mode == "orange":
                if (
                    best["score"] < 3.3
                    or best["colour_score"] < 0.18
                    or best["orange_band_width_score"] < 0.12
                    or best["ratio_score"] < 0.25
                ):
                    confidence = "low"
            else:
                if best["score"] < 3.2 or best["side_score"] < 0.22 or best["ratio_score"] < 0.25:
                    confidence = "low"

            ordered = best["quad"]
            top_candidates = sorted(scored_candidates, key=lambda item: item["score"], reverse=True)[:5]
            source_counts: dict[str, int] = {}
            for cand in candidates:
                source = str(cand.get("source", "unknown"))
                source_counts[source] = source_counts.get(source, 0) + 1

            return BorderDetectionResult(
                ordered_corners_xy=[[float(x), float(y)] for x, y in ordered.tolist()],
                contour_area_px=float(best["contour_area_px"]),
                score=float(best["score"]),
                candidate_count=len(candidates),
                confidence=confidence,
                diagnostics={
                    "ratio_score": float(best["ratio_score"]),
                    "area_score": float(best["area_score"]),
                    "extent_score": float(best["extent_score"]),
                    "side_score": float(best["side_score"]),
                    "band_score": float(best["band_score"]),
                    "colour_score": float(best["colour_score"]),
                    "orange_side_score": float(best["orange_side_score"]),
                    "black_side_score": float(best["black_side_score"]),
                    "orange_band_width_score": float(best["orange_band_width_score"]),
                    "orange_parallel_penalty": float(best["orange_parallel_penalty"]),
                    "marker_support_score": float(best["marker_support_score"]),
                    "corner_score": float(best["corner_score"]),
                    "outside_penalty": float(best["outside_penalty"]),
                    "page_like_penalty": float(best["page_like_penalty"]),
                    "plausible": bool(best["plausible"]),
                    "area_frac": float(best["area_frac"]),
                    "bbox_frac": float(best["bbox_frac"]),
                    "requested_colour_mode": border_colour_mode,
                    "detected_border_mode": best["detected_border_mode"],
                    "candidate_source": best["source"],
                    "orange_border_hex": orange_border_hex,
                    "outer_border_hex_candidates": list(outer_border_hex_candidates or []),
                    "fiducial_hex_candidates": fiducial_hex_candidates or [],
                    "candidate_sources": source_counts,
                    "top_candidates": top_candidates,
                },
            )


        def detect_outer_border(
            image_bgr: np.ndarray,
            expected_aspect_ratio: float,
            *,
            use_colour_hint: bool = True,
            use_shape_constraints: bool = True,
            fallback_to_fiducials: bool = True,
            outer_border_hex_candidates: list[str] | None = None,
            fiducial_hex_candidates: list[str] | None = None,
            border_colour_mode: str = "auto",
            orange_border_hex: str = "#D35400",
        ) -> BorderDetectionResult:
            if image_bgr is None or image_bgr.size == 0:
                raise RuntimeError("Empty image passed to detect_outer_border().")

            requested_colour_mode = (border_colour_mode or "auto").strip().lower()
            if requested_colour_mode not in {"auto", "black", "orange"}:
                requested_colour_mode = "auto"

            if requested_colour_mode in {"black", "orange"}:
                return _detect_outer_border_in_mode(
                    image_bgr,
                    expected_aspect_ratio,
                    use_colour_hint=use_colour_hint,
                    use_shape_constraints=use_shape_constraints,
                    fallback_to_fiducials=fallback_to_fiducials,
                    outer_border_hex_candidates=outer_border_hex_candidates,
                    fiducial_hex_candidates=fiducial_hex_candidates,
                    border_colour_mode=requested_colour_mode,
                    orange_border_hex=orange_border_hex,
                )

            attempts: list[tuple[str, BorderDetectionResult]] = []
            errors: dict[str, str] = {}
            for mode in ("black", "orange"):
                try:
                    attempts.append(
                        (
                            mode,
                            _detect_outer_border_in_mode(
                                image_bgr,
                                expected_aspect_ratio,
                                use_colour_hint=use_colour_hint,
                                use_shape_constraints=use_shape_constraints,
                                fallback_to_fiducials=fallback_to_fiducials,
                                outer_border_hex_candidates=outer_border_hex_candidates,
                                fiducial_hex_candidates=fiducial_hex_candidates,
                                border_colour_mode=mode,
                                orange_border_hex=orange_border_hex,
                            ),
                        )
                    )
                except RuntimeError as exc:
                    errors[mode] = str(exc)

            if not attempts:
                raise RuntimeError(f"Could not detect CDS outer border candidate. Errors: {errors}")

            def _auto_rank(item: tuple[str, BorderDetectionResult]) -> tuple[float, float]:
                _, result = item
                diagnostics = result.diagnostics or {}
                quality_bonus = 0.35 if result.confidence == "high" else 0.0
                return (float(result.score) + quality_bonus, float(diagnostics.get("colour_score", 0.0)))

            selected_mode, selected = max(attempts, key=_auto_rank)
            selected.diagnostics = dict(selected.diagnostics or {})
            selected.diagnostics["requested_colour_mode"] = "auto"
            selected.diagnostics["detected_border_mode"] = selected_mode
            selected.diagnostics["auto_candidates_tried"] = [mode for mode, _ in attempts]
            selected.diagnostics["auto_attempt_errors"] = errors
            return selected
        '''
    ),
    "tests/test_boundary_detection.py": textwrap.dedent(
        '''\
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
            assert result.diagnostics["orange_band_width_score"] >= 0.08
        '''
    ),
    "tests/test_template_registry.py": textwrap.dedent(
        '''\
        from armourcore_cds.templates.registry import load_template_config


        def test_regular_template_loads():
            template = load_template_config("cds_regular_500x600")
            assert template.design_area_mm.width == 600
            assert template.design_area_mm.height == 500
            assert template.border_detection is not None
            assert template.border_detection.border_colour_mode == "black"


        def test_xlarge_template_loads():
            template = load_template_config("cds_xlarge_500x900")
            assert template.design_area_mm.width == 900
            assert template.design_area_mm.height == 500
            assert template.border_detection is not None
            assert template.border_detection.border_colour_mode == "black"


        def test_colour_template_loads_nested_colour_hints():
            template = load_template_config("cds_colour_test_260x350")
            assert template.border_detection is not None
            assert template.border_detection.border_colour_mode == "orange"
            assert template.border_detection.colour_hints is not None
            assert "#D35400" in template.border_detection.colour_hints.outer_border_hex_candidates
            assert "#FF0000" in template.border_detection.colour_hints.fiducial_hex_candidates
        '''
    ),
}


def _find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if all((candidate / sentinel).exists() for sentinel in ROOT_SENTINELS):
            return candidate
    raise RuntimeError(
        "Could not find repo root. Run this script from inside the armourcore-cds-digitiser repo."
    )



def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")



def main() -> int:
    repo_root = _find_repo_root(Path.cwd())
    print(f"Repo root: {repo_root}")

    written: list[Path] = []
    for relative_path, content in FILES.items():
        destination = repo_root / relative_path
        if destination.exists():
            backup_path = destination.with_suffix(destination.suffix + BACKUP_SUFFIX)
            shutil.copy2(destination, backup_path)
            print(f"Backup: {backup_path.relative_to(repo_root)}")
        _write_file(destination, content)
        written.append(destination)
        print(f"Wrote:  {relative_path}")

    print("\nPatch applied.")
    print("Next run:")
    print("  python -m pytest")
    print("  then re-run your Phase 01 CLI checks on one black sheet and one orange sheet.")
    return 0


        
if __name__ == "__main__":
    raise SystemExit(main())
