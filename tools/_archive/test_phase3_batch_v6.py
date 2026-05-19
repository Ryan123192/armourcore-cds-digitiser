"""Phase 3 BATCH tester — multiple tools per PNG.

Strategy
--------
1. Threshold the input → trace_binary
2. Heavy morphological dilate (group_dilate radius) to merge each tool's
   fragmented pieces while keeping different tools separate
3. Connected components on the dilated mask → one component per tool
4. For each detected tool:
     a. Extract that tool's trace pixels (mask out the other tools)
     b. Crop to the bounding box (with padding) for speed
     c. Run the v3 iso pipeline on the crop:
        - adaptive Method F → closed centerline
        - inside-edge contour(s) → Bezier loops
     d. Translate Bezier control points back to full-image coordinates
5. Combine all tools' loops into one final SVG + DXF + layered SVG

Imports the v3 building blocks from ``test_phase3_iso_vectorise`` so the
hard-won iso logic stays in one place — this script just orchestrates
the per-tool detection and the coordinate translation.

Usage:
    python tools/test_phase3_batch.py BatchToolUnfixed01
    python tools/test_phase3_batch.py BatchToolUnfixed01 --no-open
"""
from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np

# v3 building blocks
from test_phase3_iso_vectorise import (
    _avg_trace_thickness,
    _bezier_segments_smart,
    adaptive_method_F,
    render_uniform_band,
    find_inside_edge_contours,
    export_dxf_smooth,
    export_svg_smooth,
    export_svg_with_layers,
)
from armourcore_cds.utils.image_ops import save_image


# =====================================================================
# Tool detection — heavy dilate + connected components
# =====================================================================

def detect_tools(
    trace_binary: np.ndarray,
    group_dilate_radius: int = 40,
    min_tool_area_px: int = 5000,
    verbose: bool = True,
) -> list[dict]:
    """Group broken trace fragments by spatial proximity into individual
    tools.

    Each tool is a dict::

        {"trace": uint8 mask (full size, only this tool's pixels),
         "bbox":  (x, y, w, h),
         "area":  int,
         "label": int}

    Parameters
    ----------
    trace_binary : full-resolution binary trace mask.
    group_dilate_radius : how far apart fragments can be and still merge.
        Default 40 px (reach = 80 px).  Set high enough to bridge typical
        within-tool gaps, low enough to keep tools spaced further apart
        separated.
    min_tool_area_px : drop dilated components smaller than this — they
        are most likely noise specks rather than tools.
    """
    r = max(1, int(group_dilate_radius))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1),
    )
    dilated = cv2.dilate(trace_binary, kernel)
    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(
        (dilated > 0).astype(np.uint8), connectivity=8,
    )

    tools: list[dict] = []
    for c in range(1, n_lbl):
        x, y, w, h, area = stats[c]
        if area < min_tool_area_px:
            continue
        comp_mask = (lbl == c)
        # Trace pixels belonging to THIS tool only (mask out everything else)
        tool_trace = trace_binary.copy()
        tool_trace[~comp_mask] = 0
        tools.append({
            "trace": tool_trace,
            "bbox":  (int(x), int(y), int(w), int(h)),
            "area":  int(area),
            "label": int(c),
        })

    # Sort by bounding-box area descending (largest tool first)
    tools.sort(key=lambda t: -t["area"])

    if verbose:
        print(f"  Tool detection: dilate r={r}px -> {len(tools)} tools")
        for i, t in enumerate(tools):
            x, y, w, h = t["bbox"]
            print(f"    Tool {i + 1}: bbox=({x:4d},{y:4d}) {w}x{h}  "
                  f"area={t['area']:,}px")

    return tools


# =====================================================================
# Per-tool processing — v3 iso pipeline cropped + offset
# =====================================================================

def vectorise_tool(
    tool: dict,
    padding_px: int = 30,
    rdp_epsilon: float = 4.5,
    tension: float = 0.5,
    corner_angle_deg: float = 60.0,
    bridge_max_dist_px: int = 250,
    min_loop_abs_px: float = 500.0,
    min_loop_rel: float = 0.05,
    verbose: bool = True,
) -> tuple[list[list], dict]:
    """Run the v3 iso pipeline on one detected tool.

    Returns:
        (segments_per_loop, info)
        - segments_per_loop : list of Bezier-segment lists, in FULL IMAGE
          coordinates (already offset back from the cropped frame).
        - info : dict of metrics.
    """
    full_h, full_w = tool["trace"].shape
    x, y, w, h = tool["bbox"]
    x1 = max(0, x - padding_px)
    y1 = max(0, y - padding_px)
    x2 = min(full_w, x + w + padding_px)
    y2 = min(full_h, y + h + padding_px)

    crop = tool["trace"][y1:y2, x1:x2]
    crop = (crop > 0).astype(np.uint8) * 255

    if not np.any(crop > 0):
        return [], {"error": "empty crop"}

    avg_thick = _avg_trace_thickness(crop)
    band_thickness = max(2, int(round(avg_thick)))

    (centerline, n_bridges, kernel, n_ep, n_loops_during,
     bridges_mask) = adaptive_method_F(
        crop,
        band_thickness=band_thickness,
        bridge_max_dist_px=bridge_max_dist_px,
        min_loop_abs_px=min_loop_abs_px,
        min_loop_rel=min_loop_rel,
    )
    band = render_uniform_band(centerline, band_thickness)

    inner_contours = find_inside_edge_contours(
        band,
        min_area_abs_px=min_loop_abs_px,
        min_area_relative=min_loop_rel,
        bridge_mask=bridges_mask,
        verbose=False,
    )

    segments_per_loop: list[list] = []
    total_nodes = 0
    total_cusps = 0
    for c in inner_contours:
        simplified = cv2.approxPolyDP(c, rdp_epsilon, closed=True).reshape(-1, 2)
        if len(simplified) < 4:
            continue
        pts = simplified.astype(np.float64)
        bezier = _bezier_segments_smart(
            pts, tension=tension, alpha=0.5,
            corner_angle_deg=corner_angle_deg,
        )

        # Translate Bezier control points back to FULL IMAGE coordinates
        offset = np.array([float(x1), float(y1)])
        bezier_offset = []
        for p1, cp1, cp2, p2 in bezier:
            bezier_offset.append((
                p1 + offset, cp1 + offset, cp2 + offset, p2 + offset,
            ))
        segments_per_loop.append(bezier_offset)
        total_nodes += len(simplified)

    info = {
        "kernel":      kernel,
        "n_bridges":   n_bridges,
        "n_endpoints": n_ep,
        "n_loops":     len(segments_per_loop),
        "n_nodes":     total_nodes,
        "thickness":   avg_thick,
        "crop_size":   (y2 - y1, x2 - x1),
        "offset":      (x1, y1),
        "band_full_size": None,  # filled in by caller if it wants the band
    }

    if verbose:
        print(f"    kernel={kernel}px  bridges={n_bridges}  ep={n_ep}  "
              f"loops={info['n_loops']}  nodes={total_nodes}  "
              f"thick={avg_thick:.1f}px")

    # Also return the band image (in crop coords) for visualisation
    info["band_crop"] = band
    info["band_offset"] = (x1, y1)
    return segments_per_loop, info


# =====================================================================
# Main
# =====================================================================

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Image name (no .png), e.g. BatchToolUnfixed01")
    parser.add_argument("--threshold", type=int, default=170)
    parser.add_argument("--group-dilate", type=int, default=20,
                        dest="group_dilate",
                        help="Dilation radius for tool grouping (default: 20px). "
                             "Larger = more aggressive merging (may merge "
                             "closely-spaced tools).")
    parser.add_argument("--rdp", type=float, default=4.5)
    parser.add_argument("--tension", type=float, default=0.5)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    in_path = repo_root / "data" / "inputs"  / "phase03_testing" / f"{args.name}.png"
    out_dir = repo_root / "data" / "outputs" / "phase03_testing" / f"{args.name}_batch"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"ERROR: {in_path} not found")
        sys.exit(1)

    print(f"\n=== {args.name} (batch) ===\n")
    cleaned_bgr = cv2.imread(str(in_path))
    if cleaned_bgr is None:
        print("ERROR: cv2 could not read")
        sys.exit(1)
    H, W = cleaned_bgr.shape[:2]
    print(f"Image: {W}x{H}")

    # ------------------------------------------------------------------
    # Threshold
    # ------------------------------------------------------------------
    gray = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
    trace = ((gray < args.threshold).astype(np.uint8)) * 255
    print(f"Trace pixels: {int(np.count_nonzero(trace)):,}")

    # ------------------------------------------------------------------
    # Detect tools
    # ------------------------------------------------------------------
    tools = detect_tools(
        trace,
        group_dilate_radius=args.group_dilate,
        min_tool_area_px=5000,
        verbose=True,
    )
    if not tools:
        print("ERROR: no tools detected.  Try a different --group-dilate.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Process each tool
    # ------------------------------------------------------------------
    all_segments_per_loop: list[list] = []
    full_band = np.zeros((H, W), dtype=np.uint8)

    for i, tool in enumerate(tools):
        print(f"\n--- Tool {i + 1}/{len(tools)} ---")
        segments, info = vectorise_tool(
            tool,
            padding_px=30,
            rdp_epsilon=args.rdp,
            tension=args.tension,
            verbose=True,
        )
        all_segments_per_loop.extend(segments)

        # Paste the per-tool band into the full-image band
        bx, by = info.get("band_offset", (0, 0))
        band_crop = info.get("band_crop")
        if band_crop is not None:
            ch, cw = band_crop.shape[:2]
            full_band[by:by + ch, bx:bx + cw] = np.maximum(
                full_band[by:by + ch, bx:bx + cw], band_crop,
            )

    if not all_segments_per_loop:
        print("ERROR: no loops extracted from any tool.")
        sys.exit(1)

    total_nodes = sum(len(s) for s in all_segments_per_loop)
    print(f"\n=== TOTAL ===")
    print(f"  {len(tools)} tools detected")
    print(f"  {len(all_segments_per_loop)} closed loops vectorised")
    print(f"  {total_nodes} Bezier segments total")

    # ------------------------------------------------------------------
    # Build a "gap-fixed" PNG: white background with black union-of-bands
    # ------------------------------------------------------------------
    fixed_visual = np.full((H, W, 3), 255, dtype=np.uint8)
    fixed_visual[full_band > 0] = (0, 0, 0)
    ok, fixed_png_bytes = cv2.imencode(".png", fixed_visual)
    if not ok:
        raise RuntimeError("Failed to encode gap-fixed PNG")

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    p_dxf      = out_dir / f"{args.name}.dxf"
    p_svg      = out_dir / f"{args.name}.svg"
    p_overlay  = out_dir / f"{args.name}_with_image.svg"
    p_band     = out_dir / "uniform_band.png"
    p_preview  = out_dir / "vector_preview.png"

    save_image(p_band, fixed_visual)

    # Visual preview: original + faint trace + red Bezier
    vec_preview = cleaned_bgr.copy()
    trace_px = trace > 0
    vec_preview[trace_px] = (
        vec_preview[trace_px].astype(np.int32) * 0.55
    ).clip(0, 255).astype(np.uint8)
    for segments in all_segments_per_loop:
        sampled_pts: list[np.ndarray] = []
        for p1, cp1, cp2, p2 in segments:
            ts = np.linspace(0.0, 1.0, 24, endpoint=False)
            mt = 1.0 - ts
            seg = (
                mt[:, None] ** 3 * p1
                + 3 * mt[:, None] ** 2 * ts[:, None] * cp1
                + 3 * mt[:, None] * ts[:, None] ** 2 * cp2
                + ts[:, None] ** 3 * p2
            )
            sampled_pts.append(seg)
        if not sampled_pts:
            continue
        sampled = np.vstack(sampled_pts).round().astype(np.int32)
        cv2.polylines(
            vec_preview, [sampled], True,
            (0, 0, 255), 2, cv2.LINE_AA,
        )
    save_image(p_preview, vec_preview)

    export_dxf_smooth(all_segments_per_loop, p_dxf, image_height=H)
    export_svg_smooth(all_segments_per_loop, (H, W), p_svg)
    export_svg_with_layers(
        segments_per_loop=all_segments_per_loop,
        original_png_path=in_path,
        gap_fixed_png_bytes=bytes(fixed_png_bytes),
        image_size=(H, W),
        output_path=p_overlay,
    )

    print(f"\nOutputs in: {out_dir}")
    print(f"  DXF:      {p_dxf.name}")
    print(f"  SVG:      {p_svg.name}")
    print(f"  Layered:  {p_overlay.name}  <- opens by default")
    print(f"  Preview:  {p_preview.name}")
    print(f"  Band:     {p_band.name}")

    if not args.no_open:
        _open_file(p_overlay)


if __name__ == "__main__":
    main()
