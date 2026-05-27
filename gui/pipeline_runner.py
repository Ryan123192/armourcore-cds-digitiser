"""Pipeline runner — orchestrates Phase 1, 2, 3 with progress callbacks.

Mirrors the logic of ``tools/test_end_to_end.py`` but exposes it as a
function suitable for invocation from a GUI thread.  Progress, log
lines, and the latest stage-preview image path are reported back via
callbacks so the GUI can update the UI in real time.

Output layout (matches test_end_to_end.py):

    data/outputs/end_to_end/<input_stem>/<YYYYMMDD-HHMMSS>/
        01_phase1_rectified.png
        02_phase2_cleaned.png
        03_phase3_tools_detected.png
        03_phase3_gap_filled.png
        03_phase3_vector_preview.png
        <stem>_with_image.svg
        summary.png
"""
from __future__ import annotations

import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

import cv2
import numpy as np

# Pipeline imports — these are the existing, untouched modules.
from armourcore_cds.phase1.pipeline import run_phase1_pipeline
from armourcore_cds.phase2.pipeline import run_phase2_pipeline
from armourcore_cds.templates.registry import load_template_config
from armourcore_cds.utils.debug import DebugWriter
from armourcore_cds.utils.image_ops import save_image

# Phase 3 helpers live in tools/ (legacy location)
from test_phase3_batch import detect_tools, vectorise_tool

# GUI-local helpers (input format handling + clean SVG output)
from gui.image_loader import ensure_loadable_image, ensure_landscape
from gui.svg_export import export_vector_svg
from gui.trace_filter import filter_text_and_noise
from gui.run_report import RunReport, ToolRecord, safe_format_exception


# ---------------------------------------------------------------------------
# Template registry — single source of truth for the GUI dropdown
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TemplateOption:
    """One entry in the GUI's template-type dropdown."""
    template_id: str
    display_name: str
    description: str


TEMPLATES: list[TemplateOption] = [
    TemplateOption(
        "cds_colour_test_260x350",
        "Colour Test CDS (260 x 350 mm)",
        "BlueColourTest / V2BlueColourTest sheet variants",
    ),
    TemplateOption(
        "cds_regular_500x600",
        "Regular CDS (500 x 600 mm)",
        "Standard ArmourCore template",
    ),
    TemplateOption(
        "cds_xlarge_500x900",
        "X-Large CDS (500 x 900 mm)",
        "Extended-height template",
    ),
]

DEFAULT_TEMPLATE_ID = "cds_regular_500x600"


def auto_detect_template(input_path: Path) -> str:
    """Best-effort guess at template type from filename.

    Mirrors the substring rules in ``tools/test_end_to_end.py``.  Returns
    the default if nothing matches — the GUI lets the user override.
    """
    low = input_path.stem.lower()
    if "bluecolour" in low or "colourtest" in low:
        return "cds_colour_test_260x350"
    if "xlarge" in low:
        return "cds_xlarge_500x900"
    return DEFAULT_TEMPLATE_ID


# ---------------------------------------------------------------------------
# Run-context dataclass
# ---------------------------------------------------------------------------

@dataclass
class RunCallbacks:
    """Bundle of optional callbacks the pipeline reports to during a run.

    All callbacks must be safe to call from a worker thread (the Qt
    worker uses queued signals to forward them to the main thread).
    """
    on_stage:   Callable[[str, int], None] | None = None    # (label, percent 0-100)
    on_log:     Callable[[str], None]      | None = None    # (single log line)
    on_preview: Callable[[Path], None]     | None = None    # (path to PNG)
    is_cancelled: Callable[[], bool]       | None = None    # check between stages
    on_rectified_saved: Callable[[Path], None] | None = None  # very-early progress hook


@dataclass
class RunResult:
    success: bool
    output_dir: Path | None = None
    summary_path: Path | None = None
    svg_path: Path | None = None
    n_tools: int = 0
    n_loops: int = 0
    n_beziers: int = 0
    elapsed_s: float = 0.0
    error: str | None = None
    output_paths: list[Path] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    input_path: Path,
    template_id: str,
    callbacks: RunCallbacks | None = None,
    output_root: Path | None = None,
    debug_mode: str = "off",
) -> RunResult:
    """Run Phase 1 → Phase 2 → Phase 3 on a single image file.

    Output layout (always)::

        <output_root>/<input_stem>/<YYYYMMDD-HHMMSS>/
            <stem>_rectified.png         (saved right after Phase 1)
            <stem>_vectors.svg           (saved at end if Phase 3 succeeds)
            <stem>_debug_summary.png     (saved at end if Phase 3 succeeds)
            run_report.json              (always — diagnostics)

    With ``debug_mode == "full"`` (D) the run additionally dumps::

            log.txt                       (every console line from this run)
            phase_1_debug/                (copies of Phase 1 intermediates)
            phase_2_debug/                (copies of Phase 2 intermediates)
            phase_3_debug/
                tool_001_crop.png
                tool_001_band.png
                ...

    With ``debug_mode == "off"`` (default), none of the extra dumps are
    produced — keeps the output folder lean for production runs.

    If ``output_root`` is omitted, defaults to
    ``data/outputs/end_to_end``.  The rectified PNG is written **as
    soon as Phase 1 finishes**, so a Phase 2/3 hang or failure does
    NOT cost the user the rectified output.
    """
    cb = callbacks or RunCallbacks()
    started = time.time()
    debug_mode = (debug_mode or "off").lower()
    if debug_mode not in {"off", "full"}:
        debug_mode = "off"

    # Capture every log line so we can write log.txt in Full mode.
    captured_log: list[str] = []

    def _log(line: str) -> None:
        captured_log.append(f"[{time.strftime('%H:%M:%S')}] {line}")
        if cb.on_log is not None:
            cb.on_log(line)

    def _stage(label: str, pct: int) -> None:
        _log(f"[{label}]")
        if cb.on_stage is not None:
            cb.on_stage(label, pct)

    def _cancel_check() -> bool:
        return bool(cb.is_cancelled and cb.is_cancelled())

    # ----- Start the structured run report ------------------------------
    report = RunReport()
    report.input_path = str(input_path)
    report.template_id = template_id

    try:
        input_path = Path(input_path)
        if not input_path.exists():
            report.final_status = "failed"
            report.error = f"Input file not found: {input_path}"
            return RunResult(False, error=report.error)

        # ---- Output folder ----------------------------------------------
        stem = input_path.stem
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_root = (
            Path(output_root) if output_root is not None
            else REPO / "data" / "outputs" / "end_to_end"
        )
        out_dir = out_root / stem / stamp
        out_dir.mkdir(parents=True, exist_ok=True)
        _log(f"Output folder: {out_dir}")

        # ---- Input metadata for the report ------------------------------
        try:
            report.input_format = input_path.suffix.lower().lstrip(".")
            report.input_size_mb = round(
                input_path.stat().st_size / 1_000_000.0, 2
            )
        except OSError:
            pass

        # =================================================================
        # Pre-processing — HEIC decode + auto-orient (ISSUE-001 / H)
        # =================================================================
        pre_stage = report.begin_stage("preprocess_input")
        if _cancel_check():
            report.end_stage(pre_stage, status="cancelled")
            report.final_status = "cancelled"
            report.write(out_dir / "run_report.json")
            return RunResult(False, error="Cancelled before pre-processing")

        # 1) HEIC -> PNG if needed
        load_path = ensure_loadable_image(input_path)
        report.heic_decoded = (load_path != input_path)
        if report.heic_decoded:
            _log(f"Converted HEIC -> PNG: {load_path.name}")

        # 2) Portrait -> 90deg-rotated landscape if needed.
        oriented_path, was_rotated = ensure_landscape(load_path)
        report.auto_oriented = was_rotated
        report.rotated_deg = 90 if was_rotated else 0
        if was_rotated:
            _log(f"Auto-rotated portrait -> landscape: {oriented_path.name}")

        # Capture input dimensions (post-orientation)
        try:
            _probe = cv2.imread(str(oriented_path))
            if _probe is not None:
                report.input_dimensions_px = (_probe.shape[1], _probe.shape[0])
        except Exception:
            pass

        report.end_stage(
            pre_stage,
            heic_decoded=report.heic_decoded,
            auto_oriented=report.auto_oriented,
            rotated_deg=report.rotated_deg,
            input_dimensions_px=report.input_dimensions_px,
        )

        # =================================================================
        # Phase 1 — rectify
        # =================================================================
        _stage("Loading image / Detecting CDS border / Rectifying", 5)
        if _cancel_check():
            report.final_status = "cancelled"
            report.write(out_dir / "run_report.json")
            return RunResult(False, error="Cancelled before Phase 1")

        p1_stage = report.begin_stage("phase_1_rectify")
        t0 = time.time()
        run_dir = run_phase1_pipeline(
            input_path=oriented_path,
            template_id=template_id,
            config_path=REPO / "configs" / "app" / "default.yaml",
        )
        _log(f"Phase 1 done in {time.time() - t0:.1f}s")

        scaled_path = run_dir / "scaled_design_area.png"
        rectified_path = run_dir / "rectified_outer.png"
        if not scaled_path.exists():
            report.end_stage(p1_stage, status="failed",
                             error=f"Phase 1 missing {scaled_path}")
            report.final_status = "failed"
            report.error = f"Phase 1 missing {scaled_path}"
            report.write(out_dir / "run_report.json")
            return RunResult(False, error=report.error)

        # ----- ISSUE-006 FIX -----------------------------------------------
        # Save the rectified PNG INTO THE FINAL OUTPUT FOLDER right now,
        # before Phase 2 even starts.  This way a Phase 2/3 hang, crash,
        # or user-cancel still leaves the user with a usable rectified
        # image they can hand off to manual workflow.
        p_rectified = out_dir / f"{stem}_rectified.png"
        rectified_img = (
            cv2.imread(str(rectified_path))
            if rectified_path.exists()
            else cv2.imread(str(scaled_path))
        )
        if rectified_img is not None:
            save_image(p_rectified, rectified_img)
            _log(f"Rectified image saved: {p_rectified.name}")
            report.outputs["rectified_png"] = str(p_rectified)
            if cb.on_rectified_saved is not None:
                cb.on_rectified_saved(p_rectified)
            report.rectified_dimensions_px = (
                int(rectified_img.shape[1]), int(rectified_img.shape[0])
            )
        # Show the Phase 1 preview immediately
        if cb.on_preview is not None:
            preview_img = rectified_path if rectified_path.exists() else scaled_path
            cb.on_preview(preview_img)

        report.end_stage(
            p1_stage,
            rectified_dimensions_px=report.rectified_dimensions_px,
        )

        # =================================================================
        # Phase 2 — clean grid
        # =================================================================
        _stage("Removing grid / Cleaning tracing", 35)
        if _cancel_check():
            report.final_status = "cancelled"
            report.write(out_dir / "run_report.json")
            return RunResult(False, error="Cancelled before Phase 2")

        p2_stage = report.begin_stage("phase_2_clean")
        phase2_dir = run_dir / "phase2"
        phase2_dir.mkdir(exist_ok=True)
        template = load_template_config(template_id)
        report.design_area_mm = (
            float(template.design_area_mm.width),
            float(template.design_area_mm.height),
        )
        img = cv2.imread(str(scaled_path))
        debug = DebugWriter(phase2_dir / "debug", enabled=True)
        t0 = time.time()
        run_phase2_pipeline(img, template, run_dir=phase2_dir, debug=debug)
        cleaned_path = phase2_dir / "phase2_cleaned_raster.png"
        if not cleaned_path.exists():
            report.end_stage(p2_stage, status="failed",
                             error=f"Phase 2 missing {cleaned_path}")
            report.final_status = "failed"
            report.error = f"Phase 2 missing {cleaned_path}"
            report.write(out_dir / "run_report.json")
            return RunResult(False, error=report.error)
        _log(f"Phase 2 done in {time.time() - t0:.1f}s")
        report.end_stage(p2_stage)

        if cb.on_preview is not None:
            cb.on_preview(cleaned_path)

        # =================================================================
        # Phase 3 — tool detect / gap fill / vectorise
        # =================================================================
        _stage("Detecting tools / Filling gaps / Vectorising", 55)
        if _cancel_check():
            report.final_status = "cancelled"
            report.write(out_dir / "run_report.json")
            return RunResult(False, error="Cancelled before Phase 3")

        cleaned_bgr = cv2.imread(str(cleaned_path))
        H, W = cleaned_bgr.shape[:2]
        gray = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
        trace = ((gray < 170).astype(np.uint8)) * 255

        # ----- ISSUE-018 FIX (F): pre-vectorisation noise filter --------
        # Drop tiny components (text / scan-noise / dust specks) so the
        # downstream tool-detection + per-tool vectoriser doesn't waste
        # time on hundreds of junk fragments.  Done in real-world mm
        # because we now know px/mm from the rectified scale.
        design_w_mm = float(template.design_area_mm.width)
        px_per_mm = W / design_w_mm if design_w_mm > 0 else 0.0
        report.px_per_mm = round(float(px_per_mm), 4)

        filt_stage = report.begin_stage("trace_filter_noise")
        trace_filtered, filter_stats = filter_text_and_noise(
            trace, px_per_mm=px_per_mm,
        )
        _log(
            f"Noise filter: kept {filter_stats['components_kept']}/"
            f"{filter_stats['components_total']} components "
            f"(dropped {filter_stats['components_dropped_size']} small, "
            f"{filter_stats['components_dropped_textlike']} text-like)"
        )
        report.end_stage(filt_stage, **filter_stats)
        trace = trace_filtered

        # ----- Tool detection -------------------------------------------
        td_stage = report.begin_stage("phase_3_tool_detect")
        t0 = time.time()
        tools = detect_tools(trace, group_dilate_radius=20,
                              min_tool_area_px=5000, verbose=False)
        _log(f"Detected {len(tools)} tool regions")
        report.end_stage(td_stage, n_tools=len(tools))

        # ----- Per-tool vectorisation -----------------------------------
        v_stage = report.begin_stage("phase_3_vectorise_all")
        all_segments_per_loop: list[list] = []
        full_band = np.zeros((H, W), dtype=np.uint8)

        # Full debug mode writes per-tool crop + band PNGs here.
        if debug_mode == "full":
            phase3_dbg_dir = out_dir / "phase_3_debug"
            phase3_dbg_dir.mkdir(exist_ok=True)
        else:
            phase3_dbg_dir = None

        for tool_i, tool in enumerate(tools):
            if _cancel_check():
                report.end_stage(v_stage, status="cancelled",
                                 tools_processed=tool_i)
                report.final_status = "cancelled"
                report.write(out_dir / "run_report.json")
                return RunResult(False, error="Cancelled during vectorisation")
            pct = 55 + int(35 * (tool_i + 1) / max(1, len(tools)))
            if cb.on_stage is not None:
                cb.on_stage(
                    f"Vectorising tool {tool_i + 1}/{len(tools)}", pct,
                )
            x, y, w, h = tool["bbox"]
            tool_rec = ToolRecord(
                index=tool_i + 1,
                bbox=(int(x), int(y), int(w), int(h)),
                area_px=int(w) * int(h),
            )
            tool_t0 = time.time()
            try:
                segments, info = vectorise_tool(
                    tool, padding_px=30,
                    rdp_epsilon=4.5, tension=0.5,
                    corner_angle_deg=60,
                    bridge_max_dist_px=250,
                    min_loop_abs_px=500, min_loop_rel=0.05,
                    verbose=False,
                )
                tool_rec.loops = len(segments)
                tool_rec.bezier_segments = sum(len(s) for s in segments)
            except Exception as exc:
                tool_rec.status = "failed"
                tool_rec.error = str(exc)
                segments, info = [], {}
                _log(f"  Tool {tool_i + 1} FAILED: {exc}")
            tool_rec.elapsed_s = round(time.time() - tool_t0, 3)
            report.add_tool(tool_rec)
            if tool_rec.slow:
                _log(
                    f"  Tool {tool_i + 1} slow ({tool_rec.elapsed_s:.1f}s) "
                    f"-- bbox=({x},{y},{w},{h}), loops={tool_rec.loops}"
                )
            all_segments_per_loop.extend(segments)
            bx, by = info.get("band_offset", (0, 0))
            band_crop = info.get("band_crop")
            if band_crop is not None:
                ch, cw = band_crop.shape[:2]
                full_band[by:by + ch, bx:bx + cw] = np.maximum(
                    full_band[by:by + ch, bx:bx + cw], band_crop,
                )

            # ----- Full-debug per-tool dumps ------------------------------
            if phase3_dbg_dir is not None:
                idx = f"{tool_i + 1:03d}"
                # Crop the trace mask down to the tool's bounding box.
                tx, ty, tw, th = int(x), int(y), int(w), int(h)
                ty2 = min(H, ty + th)
                tx2 = min(W, tx + tw)
                crop = trace[ty:ty2, tx:tx2]
                cv2.imwrite(str(phase3_dbg_dir / f"tool_{idx}_crop.png"), crop)
                if band_crop is not None:
                    cv2.imwrite(
                        str(phase3_dbg_dir / f"tool_{idx}_band.png"),
                        band_crop,
                    )
        n_beziers = sum(len(s) for s in all_segments_per_loop)
        report.end_stage(
            v_stage,
            tools_processed=len(tools),
            loops=len(all_segments_per_loop),
            bezier_segments=n_beziers,
            slow_tools=report.n_slow_tools,
        )
        _log(
            f"Phase 3 done in {time.time() - t0:.1f}s -- "
            f"{len(tools)} tools, {len(all_segments_per_loop)} loops, "
            f"{n_beziers} Bezier segments, "
            f"{report.n_slow_tools} slow tool(s)"
        )

        # =================================================================
        # Render outputs
        # =================================================================
        _stage("Saving output files", 92)

        trace_px = trace > 0

        # Tool-detection visualisation
        tools_vis = cleaned_bgr.copy()
        tools_vis[trace_px] = (
            tools_vis[trace_px].astype(np.int32) * 0.55
        ).clip(0, 255).astype(np.uint8)
        colours = [
            (255, 100, 100), (100, 255, 100), (100, 100, 255),
            (255, 255, 100), (255, 100, 255), (100, 255, 255),
            (200, 150, 50),  (50, 200, 150),  (150, 50, 200),
        ]
        for i, t in enumerate(tools):
            x, y, w, h = t["bbox"]
            colour = colours[i % len(colours)]
            cv2.rectangle(tools_vis, (x, y), (x + w, y + h), colour, 3)
            cv2.putText(tools_vis, f"#{i + 1}", (x + 8, y + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 3)

        # Vector preview
        vec_preview = cleaned_bgr.copy()
        vec_preview[trace_px] = (
            vec_preview[trace_px].astype(np.int32) * 0.55
        ).clip(0, 255).astype(np.uint8)
        for segments in all_segments_per_loop:
            parts: list[np.ndarray] = []
            for p1, cp1, cp2, p2 in segments:
                ts = np.linspace(0.0, 1.0, 24, endpoint=False)
                mt = 1.0 - ts
                seg = (
                    mt[:, None] ** 3 * p1
                    + 3 * mt[:, None] ** 2 * ts[:, None] * cp1
                    + 3 * mt[:, None] * ts[:, None] ** 2 * cp2
                    + ts[:, None] ** 3 * p2
                )
                parts.append(seg)
            if not parts:
                continue
            sampled = np.vstack(parts).round().astype(np.int32)
            cv2.polylines(vec_preview, [sampled], True,
                          (0, 0, 255), 2, cv2.LINE_AA)

        out_stage = report.begin_stage("write_outputs")

        # Gap-filled visual
        fixed_visual = np.full((H, W, 3), 255, dtype=np.uint8)
        fixed_visual[full_band > 0] = (0, 0, 0)
        ok, fixed_png_bytes = cv2.imencode(".png", fixed_visual)
        if not ok:
            report.end_stage(out_stage, status="failed",
                             error="Failed to encode gap-fixed PNG")
            report.final_status = "failed"
            report.error = "Failed to encode gap-fixed PNG"
            report.write(out_dir / "run_report.json")
            return RunResult(False, error=report.error)

        # ---- Production output set (only 3 files) -----------------------
        # 1) Rectified CDS PNG (already saved right after Phase 1 above)
        # 2) Vector SVG with outer-border for scaling
        # 3) Debug summary page (5-panel diagnostic)
        p_svg       = out_dir / f"{stem}_vectors.svg"
        p_summary   = out_dir / f"{stem}_debug_summary.png"
        # rectified_img reference is reused from the early-save block

        # Vector SVG with outer-border rectangle so "select all + scale
        # to design_area_mm" lands every contour at the correct
        # real-world size in one step inside Affinity Publisher.
        design_w_mm = float(template.design_area_mm.width)
        design_h_mm = float(template.design_area_mm.height)
        export_vector_svg(
            output_path=p_svg,
            rectified_png_path=p_rectified,
            segments_per_loop=all_segments_per_loop,
            image_size=(H, W),
            design_area_mm=(design_w_mm, design_h_mm),
            embed_image=True,
        )

        # Debug summary diagnostic (kept as the 3rd output for inspection)
        summary = _build_summary_page([
            ("Phase 1 - rectified", rectified_img),
            ("Phase 2 - cleaned",   cleaned_bgr),
            ("Phase 3 - tools",     tools_vis),
            ("Phase 3 - gap filled", fixed_visual),
            ("Phase 3 - vector",     vec_preview),
        ])
        save_image(p_summary, summary)

        if cb.on_preview is not None:
            cb.on_preview(p_summary)

        # Finalise outputs stage of the report
        report.outputs["svg"] = str(p_svg)
        report.outputs["debug_summary"] = str(p_summary)
        report.end_stage(out_stage)

        # ----- Full-debug bulk-copy of Phase 1 + Phase 2 intermediates --
        # Phase 1 leaves a folder of PNGs under outputs/runs/...; Phase 2
        # leaves debug PNGs under run_dir/phase2/debug.  In Full mode we
        # copy them into the user-visible output folder so the user has
        # ONE folder containing everything for the run.
        if debug_mode == "full":
            _dump_phase_debug_copies(
                run_dir=run_dir,
                phase2_dir=phase2_dir,
                out_dir=out_dir,
                logger=_log,
            )

        _stage("Done", 100)
        elapsed = time.time() - started

        # ----- Write the run report ------------------------------------
        report.n_tools = len(tools)
        report.n_loops = len(all_segments_per_loop)
        report.n_beziers = n_beziers
        report.total_elapsed_s = round(elapsed, 3)
        report.final_status = "success"
        report_path = out_dir / "run_report.json"
        report.write(report_path)
        _log(f"Run report: {report_path.name}")

        # ----- Full-debug log file (always written in full mode) -------
        if debug_mode == "full":
            try:
                log_path = out_dir / "log.txt"
                log_path.write_text("\n".join(captured_log), encoding="utf-8")
                _log(f"Full log written: {log_path.name}")
            except OSError:
                pass

        return RunResult(
            success=True,
            output_dir=out_dir,
            summary_path=p_summary,
            svg_path=p_svg,
            n_tools=len(tools),
            n_loops=len(all_segments_per_loop),
            n_beziers=n_beziers,
            elapsed_s=elapsed,
            output_paths=[p_rectified, p_svg, p_summary, report_path],
        )
    except Exception as e:
        _log(f"ERROR: {e}")
        _log(traceback.format_exc())
        # Write whatever we have so I can self-diagnose later.
        try:
            report.final_status = "failed"
            report.error = safe_format_exception()
            report.total_elapsed_s = round(time.time() - started, 3)
            # out_dir may not exist yet if we failed extremely early
            target_dir = (out_dir if "out_dir" in locals()
                          else REPO / "data" / "outputs" / "end_to_end")
            target_dir.mkdir(parents=True, exist_ok=True)
            report.write(target_dir / "run_report_FAILED.json")
            # In Full mode, persist the full log too so the failure is
            # easy to share / diagnose later.
            if debug_mode == "full":
                try:
                    (target_dir / "log.txt").write_text(
                        "\n".join(captured_log), encoding="utf-8"
                    )
                except OSError:
                    pass
        except Exception:
            pass
        return RunResult(False, error=str(e),
                         elapsed_s=time.time() - started)


# ---------------------------------------------------------------------------
# Full-debug helper — copy Phase 1/2 intermediates into the run folder
# ---------------------------------------------------------------------------

def _dump_phase_debug_copies(
    run_dir: Path,
    phase2_dir: Path,
    out_dir: Path,
    logger: Callable[[str], None] | None = None,
) -> None:
    """In Full debug mode, copy Phase 1 + Phase 2 intermediates into
    the user's output folder so everything lives in one place."""
    import shutil

    def _say(msg: str) -> None:
        if logger is not None:
            logger(msg)

    # Phase 1 — copy all of run_dir's top-level PNGs (rectified, scaled,
    # border-overlay, etc.) into phase_1_debug/.
    try:
        p1_target = out_dir / "phase_1_debug"
        p1_target.mkdir(exist_ok=True)
        copied = 0
        for src in run_dir.glob("*.png"):
            shutil.copy2(src, p1_target / src.name)
            copied += 1
        for src in run_dir.glob("*.json"):
            shutil.copy2(src, p1_target / src.name)
            copied += 1
        _say(f"Full-debug: copied {copied} Phase 1 intermediates")
    except Exception as exc:
        _say(f"Full-debug Phase 1 copy failed: {exc}")

    # Phase 2 — copy phase2_dir/debug/* if it exists.
    try:
        p2_src = phase2_dir / "debug"
        if p2_src.exists():
            p2_target = out_dir / "phase_2_debug"
            p2_target.mkdir(exist_ok=True)
            copied = 0
            for src in p2_src.rglob("*"):
                if not src.is_file():
                    continue
                rel = src.relative_to(p2_src)
                dest = p2_target / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
                copied += 1
            _say(f"Full-debug: copied {copied} Phase 2 debug files")
    except Exception as exc:
        _say(f"Full-debug Phase 2 copy failed: {exc}")


# ---------------------------------------------------------------------------
# Summary tile helper — identical layout to test_end_to_end.py
# ---------------------------------------------------------------------------

def _build_summary_page(panels: list[tuple[str, np.ndarray]],
                       tile_w: int = 600,
                       label_h: int = 32) -> np.ndarray:
    """Tile labelled images into a single summary PNG (3 per row)."""
    tiles: list[np.ndarray] = []
    for label, img in panels:
        h, w = img.shape[:2]
        scale = tile_w / float(w)
        tw, th = int(w * scale), int(h * scale)
        scaled = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        tile = np.full((th + label_h, tw, 3), 240, dtype=np.uint8)
        tile[label_h:, :] = scaled
        cv2.putText(tile, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (10, 10, 10), 1, cv2.LINE_AA)
        tiles.append(tile)

    cols = 3
    rows: list[np.ndarray] = []
    for i in range(0, len(tiles), cols):
        row_tiles = tiles[i:i + cols]
        max_h = max(t.shape[0] for t in row_tiles)
        padded = []
        for t in row_tiles:
            if t.shape[0] < max_h:
                pad = np.full((max_h - t.shape[0], t.shape[1], 3),
                              240, dtype=np.uint8)
                t = np.vstack([t, pad])
            padded.append(t)
        # equal widths since we forced tile_w
        rows.append(np.hstack(padded))
    max_w = max(r.shape[1] for r in rows)
    eq = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.full((r.shape[0], max_w - r.shape[1], 3),
                          240, dtype=np.uint8)
            r = np.hstack([r, pad])
        eq.append(r)
    return np.vstack(eq)
