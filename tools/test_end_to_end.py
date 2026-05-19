"""End-to-end pipeline runner — Phase 1 → Phase 2 → Phase 3 (new batch).

Runs all three phases sequentially on a raw input image and produces a
single diagnostic page showing the key steps from every phase, so
end-to-end pipeline bugs can be spotted in one view.

Phase 1 (rectification): raw photo → cropped & rectified design area
Phase 2 (cleaning):       rectified raster → orange-grid removed, trace
                          candidate mask isolated
Phase 3 (new batch):      cleaned raster → per-tool detect, gap-fill,
                          vectorise to smooth-Bezier closed loops

Usage:
    python tools/test_end_to_end.py BlueColourTest00
    python tools/test_end_to_end.py BlueColourTest00 --skip-phase1  (reuse latest)
    python tools/test_end_to_end.py BlueColourTest00 --skip-phase2  (reuse latest)

Outputs go to data/outputs/end_to_end/<name>/
    01_phase1_rectified.png
    02_phase2_cleaned.png
    02_phase2_trace_mask.png
    03_phase3_tools_detected.png
    03_phase3_vector_preview.png
    03_phase3_with_image.svg          <- opens by default
    summary.png                        <- diagnostic page (all steps tiled)
"""
from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

import cv2
import numpy as np

from armourcore_cds.phase1.pipeline import run_phase1_pipeline
from armourcore_cds.phase2.pipeline import run_phase2_pipeline
from armourcore_cds.templates.registry import load_template_config
from armourcore_cds.utils.debug import DebugWriter
from armourcore_cds.utils.image_ops import save_image

# Bring in our new Phase 3 batch logic
from test_phase3_batch import detect_tools, vectorise_tool
from test_phase3_iso_vectorise import export_svg_with_layers


IMAGE_SUFFIXES = {".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
TEMPLATE_PATTERNS: list[tuple[str, str]] = [
    ("bluecolour", "cds_colour_test_260x350"),
    ("colourtest", "cds_colour_test_260x350"),
]
DEFAULT_TEMPLATE = "cds_regular_500x600"


def _template_for(stem: str) -> str:
    low = stem.lower()
    for pat, tid in TEMPLATE_PATTERNS:
        if pat in low:
            return tid
    return DEFAULT_TEMPLATE


def _find_input(name: str) -> Path | None:
    """Locate the raw input file by stem name."""
    for d in [REPO / "data" / "inputs" / "raw_images",
              REPO / "data" / "inputs" / "raw_pdfs"]:
        if not d.exists():
            continue
        for suf in IMAGE_SUFFIXES:
            p = d / f"{name}{suf}"
            if p.exists():
                return p
            p = d / f"{name}{suf.upper()}"
            if p.exists():
                return p
    return None


def _latest_phase1_run(stem: str) -> Path | None:
    runs_dir = REPO / "outputs" / "runs"
    if not runs_dir.exists():
        return None
    candidates = sorted(
        (d for d in runs_dir.iterdir()
         if d.is_dir() and d.name.endswith(stem)),
        reverse=True,
    )
    return candidates[0] if candidates else None


def _open_file(path: Path) -> None:
    try:
        if platform.system() == "Windows":
            subprocess.Popen(["start", "", str(path)], shell=True)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as exc:
        print(f"  [open] {exc}")


# =====================================================================
# Stitched diagnostic page builder
# =====================================================================

def _annotated(img: np.ndarray, label: str) -> np.ndarray:
    """Add a label bar to the top of an image."""
    h, w = img.shape[:2]
    bar = np.full((40, w, 3), 240, dtype=np.uint8)
    cv2.putText(bar, label, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return np.vstack([bar, img])


def _resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = target_h / h
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)


def build_summary_page(panels: list[tuple[str, np.ndarray]],
                        cols: int = 3) -> np.ndarray:
    """Tile annotated panels in a grid (default 3 columns)."""
    target_h = 700
    annotated = []
    for label, img in panels:
        if img is None:
            continue
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        ann = _annotated(_resize_to_height(img, target_h - 40), label)
        annotated.append(ann)
    if not annotated:
        return np.zeros((target_h, 100, 3), dtype=np.uint8)

    # Pad each panel to equal width within its row, equal height across all
    rows = []
    for start in range(0, len(annotated), cols):
        row_imgs = annotated[start:start + cols]
        max_h = max(a.shape[0] for a in row_imgs)
        row_padded = []
        for a in row_imgs:
            if a.shape[0] < max_h:
                pad = np.full(
                    (max_h - a.shape[0], a.shape[1], 3), 255, dtype=np.uint8
                )
                a = np.vstack([a, pad])
            row_padded.append(a)
        rows.append(np.hstack(row_padded))

    # Pad rows to equal width
    max_w = max(r.shape[1] for r in rows)
    rows_eq = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.full(
                (r.shape[0], max_w - r.shape[1], 3), 255, dtype=np.uint8
            )
            r = np.hstack([r, pad])
        rows_eq.append(r)
    return np.vstack(rows_eq)


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Input image stem, e.g. BlueColourTest00")
    parser.add_argument("--skip-phase1", action="store_true",
                        help="Reuse the latest existing Phase 1 run.")
    parser.add_argument("--skip-phase2", action="store_true",
                        help="Reuse the latest existing Phase 2 run.")
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--tag", default=None,
                        help="Optional short label appended to the timestamped "
                             "output folder, e.g. --tag option-b")
    args = parser.parse_args()

    # Each run goes into its own timestamped subfolder so previous iterations
    # remain available for visual comparison.  The newest folder is also
    # mirrored to ``_latest`` (a stable filename for quick reference).
    case_root = REPO / "data" / "outputs" / "end_to_end" / args.name
    case_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    folder = stamp if not args.tag else f"{stamp}_{args.tag}"
    out_dir = case_root / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output] this run: {out_dir.relative_to(REPO)}")

    print(f"\n{'=' * 70}\nEnd-to-end pipeline: {args.name}\n{'=' * 70}\n")

    # ------------------------------------------------------------------
    # Phase 1 — rectify
    # ------------------------------------------------------------------
    if args.skip_phase1 or args.skip_phase2:
        run_dir = _latest_phase1_run(args.name)
        if run_dir is None:
            print(f"ERROR: --skip-phase1/2 set but no existing Phase 1 run "
                  f"found for {args.name}")
            sys.exit(1)
        print(f"[Phase 1] reusing: {run_dir.name}")
    else:
        in_path = _find_input(args.name)
        if in_path is None:
            print(f"ERROR: input file not found for {args.name}")
            sys.exit(1)
        template_id = _template_for(args.name)
        print(f"[Phase 1] input:    {in_path}")
        print(f"[Phase 1] template: {template_id}")
        t0 = time.time()
        run_dir = run_phase1_pipeline(
            input_path=in_path,
            template_id=template_id,
            config_path=REPO / "configs" / "app" / "default.yaml",
        )
        print(f"[Phase 1] done in {time.time() - t0:.1f}s")
        print(f"[Phase 1] outputs: {run_dir}")

    scaled_path = run_dir / "scaled_design_area.png"
    rectified_path = run_dir / "rectified_outer.png"
    if not scaled_path.exists():
        print(f"ERROR: missing {scaled_path}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Phase 2 — clean grid
    # ------------------------------------------------------------------
    phase2_dir = run_dir / "phase2"
    cleaned_path = phase2_dir / "phase2_cleaned_raster.png"

    if args.skip_phase2 and cleaned_path.exists():
        print(f"[Phase 2] reusing: {phase2_dir}")
    else:
        template_id = _template_for(args.name)
        template = load_template_config(template_id)
        img = cv2.imread(str(scaled_path))
        phase2_dir.mkdir(exist_ok=True)
        debug = DebugWriter(phase2_dir / "debug", enabled=True)
        print(f"[Phase 2] running on {scaled_path.name}...")
        t0 = time.time()
        run_phase2_pipeline(img, template, run_dir=phase2_dir, debug=debug)
        print(f"[Phase 2] done in {time.time() - t0:.1f}s")
        print(f"[Phase 2] outputs: {phase2_dir}")

    if not cleaned_path.exists():
        print(f"ERROR: Phase 2 did not produce {cleaned_path}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Phase 3 (new batch)
    # ------------------------------------------------------------------
    print(f"[Phase 3] running NEW batch on {cleaned_path.name}...")
    cleaned_bgr = cv2.imread(str(cleaned_path))
    H, W = cleaned_bgr.shape[:2]
    gray = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
    trace = ((gray < 170).astype(np.uint8)) * 255

    t0 = time.time()
    tools = detect_tools(trace, group_dilate_radius=20,
                         min_tool_area_px=5000, verbose=True)
    print(f"[Phase 3] {len(tools)} tools detected")

    # Tool-detection diagnostic
    tools_vis = cleaned_bgr.copy()
    trace_px = trace > 0
    tools_vis[trace_px] = (
        tools_vis[trace_px].astype(np.int32) * 0.55
    ).clip(0, 255).astype(np.uint8)
    colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255),
              (255, 255, 100), (255, 100, 255), (100, 255, 255),
              (200, 150, 50),  (50, 200, 150),  (150, 50, 200)]
    for i, t in enumerate(tools):
        x, y, w, h = t["bbox"]
        color = colors[i % len(colors)]
        cv2.rectangle(tools_vis, (x, y), (x + w, y + h), color, 3)
        cv2.putText(tools_vis, f"#{i + 1}", (x + 8, y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

    all_segments_per_loop: list[list] = []
    full_band = np.zeros((H, W), dtype=np.uint8)
    for tool_i, tool in enumerate(tools):
        print(f"  Tool {tool_i + 1}/{len(tools)}:")
        segments, info = vectorise_tool(
            tool, padding_px=30,
            rdp_epsilon=4.5, tension=0.5,
            corner_angle_deg=60,
            bridge_max_dist_px=250,
            min_loop_abs_px=500, min_loop_rel=0.05,
            verbose=True,
        )
        all_segments_per_loop.extend(segments)
        bx, by = info.get("band_offset", (0, 0))
        band_crop = info.get("band_crop")
        if band_crop is not None:
            ch, cw = band_crop.shape[:2]
            full_band[by:by + ch, bx:bx + cw] = np.maximum(
                full_band[by:by + ch, bx:bx + cw], band_crop,
            )

    elapsed = time.time() - t0
    total_nodes = sum(len(s) for s in all_segments_per_loop)
    print(f"[Phase 3] done in {elapsed:.1f}s — "
          f"{len(tools)} tools, {len(all_segments_per_loop)} loops, "
          f"{total_nodes} Bezier segments")

    # ------------------------------------------------------------------
    # Phase 3 outputs
    # ------------------------------------------------------------------
    # Vector preview (red curves over original)
    vec_preview = cleaned_bgr.copy()
    vec_preview[trace_px] = (
        vec_preview[trace_px].astype(np.int32) * 0.55
    ).clip(0, 255).astype(np.uint8)
    for segments in all_segments_per_loop:
        sampled_parts: list[np.ndarray] = []
        for p1, cp1, cp2, p2 in segments:
            ts = np.linspace(0.0, 1.0, 24, endpoint=False)
            mt = 1.0 - ts
            seg = (
                mt[:, None] ** 3 * p1
                + 3 * mt[:, None] ** 2 * ts[:, None] * cp1
                + 3 * mt[:, None] * ts[:, None] ** 2 * cp2
                + ts[:, None] ** 3 * p2
            )
            sampled_parts.append(seg)
        if not sampled_parts:
            continue
        sampled = np.vstack(sampled_parts).round().astype(np.int32)
        cv2.polylines(vec_preview, [sampled], True,
                      (0, 0, 255), 2, cv2.LINE_AA)

    # Gap-fixed PNG (band on white background) for the layered SVG
    fixed_visual = np.full((H, W, 3), 255, dtype=np.uint8)
    fixed_visual[full_band > 0] = (0, 0, 0)
    ok, fixed_png_bytes = cv2.imencode(".png", fixed_visual)
    if not ok:
        raise RuntimeError("Failed to encode gap-fixed PNG")

    # ------------------------------------------------------------------
    # Save everything
    # ------------------------------------------------------------------
    p_p1     = out_dir / "01_phase1_rectified.png"
    p_p2     = out_dir / "02_phase2_cleaned.png"
    p_tools  = out_dir / "03_phase3_tools_detected.png"
    p_vec    = out_dir / "03_phase3_vector_preview.png"
    p_band   = out_dir / "03_phase3_gap_filled.png"
    p_svg    = out_dir / f"{args.name}_with_image.svg"
    p_summary = out_dir / "summary.png"

    # Copy Phase 1 + Phase 2 reference outputs
    rectified_img = cv2.imread(str(rectified_path)) if rectified_path.exists() \
        else cv2.imread(str(scaled_path))
    save_image(p_p1, rectified_img)
    save_image(p_p2, cleaned_bgr)
    save_image(p_tools, tools_vis)
    save_image(p_vec, vec_preview)
    save_image(p_band, fixed_visual)

    export_svg_with_layers(
        segments_per_loop=all_segments_per_loop,
        original_png_path=cleaned_path,
        gap_fixed_png_bytes=bytes(fixed_png_bytes),
        image_size=(H, W),
        output_path=p_svg,
    )

    # ------------------------------------------------------------------
    # Summary diagnostic page
    # ------------------------------------------------------------------
    summary = build_summary_page([
        ("Phase 1 - rectified", rectified_img),
        ("Phase 2 - cleaned",   cleaned_bgr),
        ("Phase 3 - tools detected", tools_vis),
        ("Phase 3 - gap filled",     fixed_visual),
        ("Phase 3 - vector preview", vec_preview),
    ])
    save_image(p_summary, summary)

    print(f"\nOutputs in: {out_dir}")
    print(f"  Summary diagnostic:  {p_summary.name}")
    print(f"  Layered SVG:         {p_svg.name}  <- opens by default")

    if not args.no_open:
        _open_file(p_summary)
        _open_file(p_svg)


if __name__ == "__main__":
    main()
