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
from gui.image_loader import ensure_loadable_image
from gui.svg_export import export_vector_svg


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
) -> RunResult:
    """Run Phase 1 → Phase 2 → Phase 3 on a single image file.

    The output goes into a fresh timestamped subfolder under::

        data/outputs/end_to_end/<input_stem>/<YYYYMMDD-HHMMSS>/

    The same layout used by ``tools/test_end_to_end.py`` so existing
    diagnostics keep working unchanged.
    """
    cb = callbacks or RunCallbacks()
    started = time.time()

    def _log(line: str) -> None:
        if cb.on_log is not None:
            cb.on_log(line)

    def _stage(label: str, pct: int) -> None:
        _log(f"[{label}]")
        if cb.on_stage is not None:
            cb.on_stage(label, pct)

    def _cancel_check() -> bool:
        return bool(cb.is_cancelled and cb.is_cancelled())

    try:
        input_path = Path(input_path)
        if not input_path.exists():
            return RunResult(False, error=f"Input file not found: {input_path}")

        # ---- Output folder ----------------------------------------------
        stem = input_path.stem
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_dir = REPO / "data" / "outputs" / "end_to_end" / stem / stamp
        out_dir.mkdir(parents=True, exist_ok=True)
        _log(f"Output folder: {out_dir.relative_to(REPO)}")

        # =================================================================
        # Phase 1 — rectify
        # =================================================================
        _stage("Loading image / Detecting CDS border / Rectifying", 5)
        if _cancel_check():
            return RunResult(False, error="Cancelled before Phase 1")

        # If the input is a HEIC (iPhone photo) file, decode it via
        # pillow-heif and hand Phase 1 a PNG copy.  Other formats pass
        # through unchanged.
        load_path = ensure_loadable_image(input_path)
        if load_path != input_path:
            _log(f"Converted HEIC -> PNG: {load_path.name}")

        t0 = time.time()
        run_dir = run_phase1_pipeline(
            input_path=load_path,
            template_id=template_id,
            config_path=REPO / "configs" / "app" / "default.yaml",
        )
        _log(f"Phase 1 done in {time.time() - t0:.1f}s")

        scaled_path = run_dir / "scaled_design_area.png"
        rectified_path = run_dir / "rectified_outer.png"
        if not scaled_path.exists():
            return RunResult(False, error=f"Phase 1 missing {scaled_path}")

        # Show the Phase 1 preview immediately
        if cb.on_preview is not None:
            preview_img = rectified_path if rectified_path.exists() else scaled_path
            cb.on_preview(preview_img)

        # =================================================================
        # Phase 2 — clean grid
        # =================================================================
        _stage("Removing grid / Cleaning tracing", 35)
        if _cancel_check():
            return RunResult(False, error="Cancelled before Phase 2")

        phase2_dir = run_dir / "phase2"
        phase2_dir.mkdir(exist_ok=True)
        template = load_template_config(template_id)
        img = cv2.imread(str(scaled_path))
        debug = DebugWriter(phase2_dir / "debug", enabled=True)
        t0 = time.time()
        run_phase2_pipeline(img, template, run_dir=phase2_dir, debug=debug)
        cleaned_path = phase2_dir / "phase2_cleaned_raster.png"
        if not cleaned_path.exists():
            return RunResult(False, error=f"Phase 2 missing {cleaned_path}")
        _log(f"Phase 2 done in {time.time() - t0:.1f}s")

        if cb.on_preview is not None:
            cb.on_preview(cleaned_path)

        # =================================================================
        # Phase 3 — tool detect / gap fill / vectorise
        # =================================================================
        _stage("Detecting tools / Filling gaps / Vectorising", 55)
        if _cancel_check():
            return RunResult(False, error="Cancelled before Phase 3")

        cleaned_bgr = cv2.imread(str(cleaned_path))
        H, W = cleaned_bgr.shape[:2]
        gray = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
        trace = ((gray < 170).astype(np.uint8)) * 255

        t0 = time.time()
        tools = detect_tools(trace, group_dilate_radius=20,
                              min_tool_area_px=5000, verbose=False)
        _log(f"Detected {len(tools)} tool regions")

        # Per-tool vectorisation
        all_segments_per_loop: list[list] = []
        full_band = np.zeros((H, W), dtype=np.uint8)
        for tool_i, tool in enumerate(tools):
            if _cancel_check():
                return RunResult(False, error="Cancelled during vectorisation")
            pct = 55 + int(35 * (tool_i + 1) / max(1, len(tools)))
            if cb.on_stage is not None:
                cb.on_stage(
                    f"Vectorising tool {tool_i + 1}/{len(tools)}", pct,
                )
            segments, info = vectorise_tool(
                tool, padding_px=30,
                rdp_epsilon=4.5, tension=0.5,
                corner_angle_deg=60,
                bridge_max_dist_px=250,
                min_loop_abs_px=500, min_loop_rel=0.05,
                verbose=False,
            )
            all_segments_per_loop.extend(segments)
            bx, by = info.get("band_offset", (0, 0))
            band_crop = info.get("band_crop")
            if band_crop is not None:
                ch, cw = band_crop.shape[:2]
                full_band[by:by + ch, bx:bx + cw] = np.maximum(
                    full_band[by:by + ch, bx:bx + cw], band_crop,
                )
        n_beziers = sum(len(s) for s in all_segments_per_loop)
        _log(
            f"Phase 3 done in {time.time() - t0:.1f}s — "
            f"{len(tools)} tools, {len(all_segments_per_loop)} loops, "
            f"{n_beziers} Bezier segments"
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

        # Gap-filled visual
        fixed_visual = np.full((H, W, 3), 255, dtype=np.uint8)
        fixed_visual[full_band > 0] = (0, 0, 0)
        ok, fixed_png_bytes = cv2.imencode(".png", fixed_visual)
        if not ok:
            return RunResult(False, error="Failed to encode gap-fixed PNG")

        # ---- Production output set (only 3 files) -----------------------
        # 1) Rectified CDS PNG (cropped + to scale)
        # 2) Vector SVG with outer-border for scaling
        # 3) Debug summary page (5-panel diagnostic)
        p_rectified = out_dir / f"{stem}_rectified.png"
        p_svg       = out_dir / f"{stem}_vectors.svg"
        p_summary   = out_dir / f"{stem}_debug_summary.png"

        rectified_img = (cv2.imread(str(rectified_path))
                         if rectified_path.exists()
                         else cv2.imread(str(scaled_path)))
        save_image(p_rectified, rectified_img)

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

        _stage("Done", 100)
        elapsed = time.time() - started

        return RunResult(
            success=True,
            output_dir=out_dir,
            summary_path=p_summary,
            svg_path=p_svg,
            n_tools=len(tools),
            n_loops=len(all_segments_per_loop),
            n_beziers=n_beziers,
            elapsed_s=elapsed,
            output_paths=[p_rectified, p_svg, p_summary],
        )
    except Exception as e:
        _log(f"ERROR: {e}")
        _log(traceback.format_exc())
        return RunResult(False, error=str(e),
                         elapsed_s=time.time() - started)


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
