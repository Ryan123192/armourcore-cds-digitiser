"""Phase 3 isolated-tool vectorisation — Method F + inside-edge vectorise.

Pipeline
--------
1. Load PNG, threshold → trace_binary
2. Apply Method F (hybrid: CLOSE k=25 + skeleton-endpoint Bezier bridges)
   → produces a CLOSED centerline of the tool outline
3. Render the closed centerline at uniform thickness (matches the trace
   thickness, so gap-filled sections look identical to original sections)
4. Find the INNER contour of this clean band — this is the "inside edge
   of the tool tracing", a single closed loop polygon
5. Simplify with RDP, save as SVG + DXF
6. Open the DXF (and a PNG preview) in default viewers

Why this gives a clean single loop
----------------------------------
After Method F, the centerline (medial axis + Bezier bridges) is a
single closed curve.  Dilating it uniformly yields a band of consistent
thickness.  cv2.findContours with RETR_TREE on this band returns the
outer boundary AND the interior hole — the hole's contour is the inside
edge of the customer's drawn line, which is exactly what we want as
the cuttable tool outline.

Usage:
    python tools/test_phase3_iso_vectorise.py IsoToolUnfixed01
    python tools/test_phase3_iso_vectorise.py IsoToolUnfixed01 --no-open
"""
from __future__ import annotations

import argparse
import base64
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import ezdxf
import numpy as np
from skimage.morphology import medial_axis

from armourcore_cds.utils.image_ops import save_image


# =====================================================================
# Smart cubic-Bezier conversion: centripetal Catmull-Rom + cusp corners
# =====================================================================
#
# Problem with uniform Catmull-Rom: handles have a fixed length scale
# regardless of local edge length, so at sharp corners the handles
# overshoot the polygon — visible as little "horns".
#
# Centripetal Catmull-Rom (alpha=0.5) parameterises by sqrt of edge
# length, so handles automatically shrink at densely-spaced or
# tight-corner regions.  This is the standard fix.
#
# We additionally detect sharp corners (turn angle > corner_angle_deg)
# and convert them to *cusp nodes* with zero-length handles on both
# sides — preserving any genuine sharp features (V-junctions, cut tips,
# etc.) instead of rounding them off.
# =====================================================================

def _bezier_segments_smart(
    pts: np.ndarray,
    tension: float = 0.5,
    alpha: float = 0.5,
    corner_angle_deg: float = 60.0,
) -> list:
    """Convert a closed polyline into cubic Bezier segments using
    centripetal Catmull-Rom interpolation with corner cusps.

    Parameters
    ----------
    pts : (N, 2) float64 closed polyline points (no repeated end)
    tension : 0..1 — multiplier on handle length (smaller = tighter hug)
    alpha : 0=uniform, 0.5=centripetal, 1.0=chordal
    corner_angle_deg : turns sharper than this become cusp nodes
    """
    n = len(pts)
    if n < 4:
        return []

    # ---- corner detection (turn angle at each node) ----
    cos_thresh = float(np.cos(np.radians(corner_angle_deg)))
    is_corner = np.zeros(n, dtype=bool)
    for i in range(n):
        p_prev = pts[(i - 1) % n]
        p_cur  = pts[i]
        p_next = pts[(i + 1) % n]
        v_in  = p_cur - p_prev
        v_out = p_next - p_cur
        n_in  = float(np.linalg.norm(v_in))
        n_out = float(np.linalg.norm(v_out))
        if n_in < 1e-9 or n_out < 1e-9:
            continue
        cos_a = float(np.dot(v_in, v_out) / (n_in * n_out))
        if cos_a < cos_thresh:
            is_corner[i] = True

    # ---- per-segment cubic Bezier from centripetal Catmull-Rom ----
    eps = 1e-6
    segments = []
    for i in range(n):
        p0 = pts[(i - 1) % n]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n]

        d01 = max(float(np.linalg.norm(p1 - p0)) ** alpha, eps)
        d12 = max(float(np.linalg.norm(p2 - p1)) ** alpha, eps)
        d23 = max(float(np.linalg.norm(p3 - p2)) ** alpha, eps)

        t0 = 0.0
        t1 = t0 + d01
        t2 = t1 + d12
        t3 = t2 + d23

        # Tangent vectors at p1 and p2 (centripetal Catmull-Rom)
        m1 = (t2 - t1) * (
            (p1 - p0) / (t1 - t0)
            - (p2 - p0) / (t2 - t0)
            + (p2 - p1) / (t2 - t1)
        )
        m2 = (t2 - t1) * (
            (p2 - p1) / (t2 - t1)
            - (p3 - p1) / (t3 - t1)
            + (p3 - p2) / (t3 - t2)
        )

        # Cusp at corners — zero out the handles that touch a corner node
        if is_corner[i]:
            m1 = np.zeros(2)
        if is_corner[(i + 1) % n]:
            m2 = np.zeros(2)

        cp1 = p1 + m1 * tension / 3.0
        cp2 = p2 - m2 * tension / 3.0
        segments.append((p1.copy(), cp1, cp2, p2.copy()))

    return segments


# =====================================================================
# Method F building blocks
# =====================================================================

def _avg_trace_thickness(trace_binary: np.ndarray) -> float:
    skel_bool, dist = medial_axis(trace_binary > 0, return_distance=True)
    th = dist[skel_bool] * 2.0
    th = th[th >= 2.0]
    if th.size == 0:
        return 4.0
    return float(np.clip(np.median(th), 2.0, 30.0))


def _skel_endpoints(skel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    skel_u8 = (skel > 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    kernel[1, 1] = 0
    nb = cv2.filter2D(skel_u8, cv2.CV_8U, kernel)
    ep_y, ep_x = np.where((skel_u8 > 0) & (nb == 1))
    return ep_x, ep_y


def _walk_back(skel: np.ndarray, sx: int, sy: int, n: int) -> list[tuple[int, int]]:
    H, W = skel.shape
    skel_bool = skel > 0
    path = [(sx, sy)]
    visited = {(sx, sy)}
    cx, cy = sx, sy
    for _ in range(n):
        found = False
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = cx + dx, cy + dy
                if (
                    (nx, ny) not in visited
                    and 0 <= nx < W and 0 <= ny < H
                    and skel_bool[ny, nx]
                ):
                    path.append((nx, ny))
                    visited.add((nx, ny))
                    cx, cy = nx, ny
                    found = True
                    break
            if found:
                break
        if not found:
            break
    return path


def method_F_closed_centerline(
    trace_binary: np.ndarray,
    base_k: int = 25,
    bridge_max_dist_px: int = 200,
    lookback_px: int = 15,
    facing_threshold: float = 0.20,
) -> tuple[np.ndarray, int]:
    """Method F → returns (closed_centerline_skeleton_uint8, n_bridges_added).

    Combines the medial axis of the CLOSED trace with Bezier-bridge
    completions for any remaining skeleton endpoints, producing a single
    closed 1-pixel-wide centerline.
    """
    # Stage 1: CLOSE + medial axis
    k = max(3, base_k | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(trace_binary, cv2.MORPH_CLOSE, kernel)
    skel_bool, _ = medial_axis(closed > 0, return_distance=True)
    skel = skel_bool.astype(np.uint8) * 255

    # Stage 2: pair remaining skeleton endpoints
    ep_x, ep_y = _skel_endpoints(skel)
    descriptors = []
    for ex, ey in zip(ep_x.tolist(), ep_y.tolist()):
        path = _walk_back(skel, int(ex), int(ey), lookback_px)
        if len(path) < 4:
            continue
        d = np.array(
            [path[0][0] - path[-1][0], path[0][1] - path[-1][1]],
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(d))
        if norm < 1e-9:
            continue
        descriptors.append({"pos": np.array([float(ex), float(ey)]), "dir": d / norm})

    candidates = []
    n = len(descriptors)
    for i in range(n):
        Ai  = descriptors[i]["pos"]
        dAi = descriptors[i]["dir"]
        for j in range(i + 1, n):
            Bj  = descriptors[j]["pos"]
            dBj = descriptors[j]["dir"]
            diff = Bj - Ai
            dist = float(np.linalg.norm(diff))
            if dist < 5 or dist > bridge_max_dist_px:
                continue
            AB = diff / dist
            sA = float(np.dot(dAi,  AB))
            sB = float(np.dot(-dBj, AB))
            if sA < facing_threshold or sB < facing_threshold:
                continue
            score = (sA + sB) / 2.0 / max(1.0, dist / 60.0)
            candidates.append((score, i, j))
    candidates.sort(key=lambda t: -t[0])

    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _s, i, j in candidates:
        if i in used or j in used:
            continue
        used.add(i); used.add(j)
        pairs.append((i, j))

    # Render bridges directly into the centerline as 1-px lines (no
    # antialiasing, so the centerline stays exactly 1px wide everywhere)
    centerline = skel.copy()
    for i, j in pairs:
        A  = descriptors[i]["pos"]
        B  = descriptors[j]["pos"]
        dA = descriptors[i]["dir"]
        dB = descriptors[j]["dir"]
        dist = float(np.linalg.norm(B - A))
        scale = dist / 3.0
        P0, P1, P2, P3 = A, A + scale * dA, B - scale * dB, B
        n_samp = max(20, int(dist))
        ts = np.linspace(0.0, 1.0, n_samp)
        mt = 1.0 - ts
        curve = (
            mt[:, None] ** 3 * P0
            + 3 * mt[:, None] ** 2 * ts[:, None] * P1
            + 3 * mt[:, None] * ts[:, None] ** 2 * P2
            + ts[:, None] ** 3 * P3
        ).astype(np.int32)
        cv2.polylines(centerline, [curve], False, 255, 1, cv2.LINE_8)

    return centerline, len(pairs)


# =====================================================================
# Uniform-thickness band rendering & inside-edge contour extraction
# =====================================================================

def render_uniform_band(skeleton: np.ndarray, thickness: int) -> np.ndarray:
    """Dilate the centerline so the result has uniform thickness everywhere
    — gap-filled sections look identical to original sections."""
    radius = max(1, int(round(thickness / 2.0)))
    k = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(skeleton, kernel)


def find_inside_edge_contours(
    band: np.ndarray,
    min_area_px: float = 500.0,
    verbose: bool = False,
) -> list[np.ndarray]:
    """Find ALL interior holes in a closed band (sorted largest-first),
    above a minimum area threshold.

    A multi-loop tool — e.g., a screwdriver drawn as separate handle and
    shaft outlines — will have one hole per loop.  Each hole's contour
    is the inside edge of one tool component.
    """
    contours, hierarchy = cv2.findContours(
        band, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    if not contours or hierarchy is None:
        return []

    if verbose:
        print(f"  cv2.findContours found {len(contours)} total contours")
        for i, c in enumerate(contours):
            parent = hierarchy[0][i][3]
            kind = "outer" if parent == -1 else f"hole(of {parent})"
            print(f"    [{i}] {kind}  pts={len(c)}  area={cv2.contourArea(c):.0f}px")

    inner: list[np.ndarray] = []
    for i, c in enumerate(contours):
        if hierarchy[0][i][3] == -1:
            continue  # outer contour, skip
        if cv2.contourArea(c) < min_area_px:
            continue  # too small — likely artefact
        inner.append(c)
    inner.sort(key=cv2.contourArea, reverse=True)
    return inner


# =====================================================================
# Vector exports
# =====================================================================

def _bezier_path_d(segments: list) -> str:
    """Build an SVG <path> 'd' attribute from cubic Bezier segments."""
    if not segments:
        return ""
    p0 = segments[0][0]
    parts = [f"M {p0[0]:.3f},{p0[1]:.3f}"]
    for _p1, cp1, cp2, p2 in segments:
        parts.append(
            f"C {cp1[0]:.3f},{cp1[1]:.3f} "
            f"{cp2[0]:.3f},{cp2[1]:.3f} "
            f"{p2[0]:.3f},{p2[1]:.3f}"
        )
    parts.append("Z")
    return " ".join(parts)


def export_dxf_smooth(
    segments_per_loop: list[list],
    output_path: Path,
    image_height: int,
) -> None:
    """Save one or more closed Bezier loops as DXF.  Each loop is a list
    of cubic-Bezier (P0, cp1, cp2, P3) tuples; each segment becomes a
    SPLINE entity with degree=3 and bezier knots.

    Y is flipped (image_height - y) for DXF's Y-up coordinate convention.
    """
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    bezier_knots = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]

    def flip(p):
        return (float(p[0]), float(image_height - p[1]))

    for loop_idx, segments in enumerate(segments_per_loop):
        for p1, cp1, cp2, p2 in segments:
            ctrl = [flip(p1), flip(cp1), flip(cp2), flip(p2)]
            spline = msp.add_open_spline(
                ctrl, degree=3, knots=bezier_knots,
            )
            spline.dxf.layer = f"TOOL_OUTLINE_{loop_idx + 1}"

    doc.saveas(str(output_path))


def export_svg_smooth(
    segments_per_loop: list[list],
    image_size: tuple[int, int],
    output_path: Path,
    stroke_width: float = 1.0,
) -> None:
    """Save one or more closed smooth-Bezier paths as plain SVG."""
    h, w = image_size
    paths_xml = []
    for i, segments in enumerate(segments_per_loop):
        d = _bezier_path_d(segments)
        if not d:
            continue
        paths_xml.append(
            f'  <path d="{d}" fill="none" stroke="red" '
            f'stroke-width="{stroke_width}" stroke-linejoin="round" '
            f'stroke-linecap="round" id="loop_{i + 1}"/>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {w} {h}" width="{w}" height="{h}">\n'
        + "\n".join(paths_xml) + "\n"
        + "</svg>\n"
    )
    output_path.write_text(svg)


def _png_bytes_b64(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode("ascii")


def export_svg_with_layers(
    segments_per_loop: list[list],
    original_png_path: Path,
    gap_fixed_png_bytes: bytes,
    image_size: tuple[int, int],
    output_path: Path,
    stroke_width: float = 1.0,
    fixed_opacity: float = 0.6,
) -> None:
    """Save self-contained SVG with three layers (bottom -> top):

      1. ``original_image``  - input PNG, hidden by default.
      2. ``gap_fixed_image`` - uniform-band PNG, visible at fixed_opacity.
      3. ``tool_outline``    - one or more smooth Bezier closed paths,
                                stroke centered, non-scaling.

    All bytes are inlined; the SVG is fully portable.
    """
    h, w = image_size
    orig_b64  = _png_bytes_b64(Path(original_png_path).read_bytes())
    fixed_b64 = _png_bytes_b64(gap_fixed_png_bytes)

    paths_xml = []
    for i, segments in enumerate(segments_per_loop):
        d = _bezier_path_d(segments)
        if not d:
            continue
        paths_xml.append(
            f'    <path d="{d}" fill="none" stroke="red" '
            f'stroke-width="{stroke_width}" '
            f'stroke-linejoin="round" stroke-linecap="round" '
            f'stroke-alignment="center" '
            f'vector-effect="non-scaling-stroke" id="loop_{i + 1}"/>'
        )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'viewBox="0 0 {w} {h}" width="{w}" height="{h}">\n'
        f'  <!-- Layer 1 (bottom): original input image, hidden -->\n'
        f'  <g id="original_image" visibility="hidden">\n'
        f'    <image xlink:href="data:image/png;base64,{orig_b64}" '
        f'x="0" y="0" width="{w}" height="{h}"/>\n'
        f'  </g>\n'
        f'  <!-- Layer 2 (middle): gap-fixed image, visible -->\n'
        f'  <g id="gap_fixed_image">\n'
        f'    <image xlink:href="data:image/png;base64,{fixed_b64}" '
        f'x="0" y="0" width="{w}" height="{h}" '
        f'opacity="{fixed_opacity}"/>\n'
        f'  </g>\n'
        f'  <!-- Layer 3 (top): one Bezier loop per tool component -->\n'
        f'  <g id="tool_outline">\n'
        + "\n".join(paths_xml) + "\n"
        + "  </g>\n"
        + "</svg>\n"
    )
    output_path.write_text(svg)


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
    parser.add_argument("name")
    parser.add_argument("--threshold", type=int, default=170)
    parser.add_argument("--rdp", type=float, default=4.5,
                        help="RDP epsilon (default: 4.5px).  Larger = fewer "
                             "nodes; centripetal Catmull-Rom handles smooth "
                             "out the curve so you don't need many.")
    parser.add_argument("--tension", type=float, default=0.5,
                        help="Catmull-Rom handle scale (default: 0.5).  "
                             "Smaller = tighter hug to the polygon.")
    parser.add_argument("--corner-angle", type=float, default=60.0,
                        dest="corner_angle",
                        help="Turns sharper than this (deg) become cusp "
                             "nodes (no horns).  Default: 60°.")
    parser.add_argument("--close-kernel", type=int, default=25, dest="close_kernel")
    parser.add_argument("--bridge-dist", type=int, default=200, dest="bridge_dist",
                        help="Method F max distance for skeleton-endpoint "
                             "Bezier bridges (default: 200px).  Raise for "
                             "tools with very long obliterated spans.")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    in_path = repo_root / "data" / "inputs"  / "phase03_testing" / f"{args.name}.png"
    out_dir = repo_root / "data" / "outputs" / "phase03_testing" / f"{args.name}_vector"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"ERROR: {in_path} not found")
        sys.exit(1)

    print(f"\n=== {args.name} — Method F + vectorise ===\n")

    cleaned_bgr = cv2.imread(str(in_path))
    if cleaned_bgr is None:
        print("ERROR: cv2 could not read image")
        sys.exit(1)
    H, W = cleaned_bgr.shape[:2]
    print(f"Size: {W}x{H}")

    # ------------------------------------------------------------------
    # 1. Threshold
    # ------------------------------------------------------------------
    gray = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
    trace = ((gray < args.threshold).astype(np.uint8)) * 255
    print(f"Trace pixels: {int(np.count_nonzero(trace)):,}")

    avg_thick = _avg_trace_thickness(trace)
    print(f"Avg trace thickness: {avg_thick:.1f}px")

    # ------------------------------------------------------------------
    # 2. Method F → closed centerline (1-pixel wide)
    # ------------------------------------------------------------------
    centerline, n_bridges = method_F_closed_centerline(
        trace, base_k=args.close_kernel,
        bridge_max_dist_px=args.bridge_dist,
    )
    print(f"Method F: bridge curves added = {n_bridges}")
    print(f"Centerline pixels: {int(np.count_nonzero(centerline)):,}")

    # Verify centerline is closed (no remaining endpoints)
    ep_x, _ = _skel_endpoints(centerline)
    print(f"Remaining centerline endpoints: {len(ep_x)} "
          f"(should be 0 for a fully closed loop)")

    # ------------------------------------------------------------------
    # 3. Uniform-thickness band
    # ------------------------------------------------------------------
    band_thickness = max(2, int(round(avg_thick)))
    band = render_uniform_band(centerline, band_thickness)
    print(f"Uniform band thickness: {band_thickness}px")

    # ------------------------------------------------------------------
    # 4. Inside-edge contours — one per closed loop in the band.  Multi-
    # piece tools (e.g., screwdriver = handle + shaft) yield multiple.
    # ------------------------------------------------------------------
    inner_contours = find_inside_edge_contours(band, min_area_px=500.0, verbose=True)
    if not inner_contours:
        print("ERROR: no interior holes detected in the band.  The centerline "
              "may not be closed enough.  Try a larger --close-kernel.")
        sys.exit(1)
    print(f"Inside-edge loops detected: {len(inner_contours)}")
    for i, c in enumerate(inner_contours):
        print(f"  Loop {i + 1}: {len(c)} raw points  area={cv2.contourArea(c):.0f}px")

    # ------------------------------------------------------------------
    # 5. Per loop: RDP → centripetal Catmull-Rom Bezier with corner cusps
    # ------------------------------------------------------------------
    segments_per_loop: list[list] = []
    total_nodes = 0
    total_corners = 0
    cos_thresh = float(np.cos(np.radians(args.corner_angle)))

    for i, contour in enumerate(inner_contours):
        simplified = cv2.approxPolyDP(
            contour, args.rdp, closed=True
        ).reshape(-1, 2)
        if len(simplified) < 4:
            print(f"  Loop {i + 1}: skipped (only {len(simplified)} nodes)")
            continue
        pts_f64 = simplified.astype(np.float64)

        bezier = _bezier_segments_smart(
            pts_f64,
            tension=args.tension,
            alpha=0.5,
            corner_angle_deg=args.corner_angle,
        )
        # Count corners on this loop
        n = len(pts_f64)
        loop_corners = 0
        for k in range(n):
            v_in  = pts_f64[k] - pts_f64[(k - 1) % n]
            v_out = pts_f64[(k + 1) % n] - pts_f64[k]
            ni = float(np.linalg.norm(v_in))
            no = float(np.linalg.norm(v_out))
            if ni < 1e-9 or no < 1e-9:
                continue
            if float(np.dot(v_in, v_out) / (ni * no)) < cos_thresh:
                loop_corners += 1
        total_nodes   += len(simplified)
        total_corners += loop_corners
        segments_per_loop.append(bezier)
        print(f"  Loop {i + 1}: {len(simplified)} RDP nodes, "
              f"{len(bezier)} Bezier segments, {loop_corners} cusps")

    print(f"Total: {total_nodes} nodes across {len(segments_per_loop)} loops, "
          f"{total_corners} cusp corners")

    # ------------------------------------------------------------------
    # Diagnostic renders
    # ------------------------------------------------------------------
    # (a) Uniform band over the original
    band_preview = cleaned_bgr.copy()
    trace_px = trace > 0
    band_only = (band > 0) & ~trace_px
    band_preview[trace_px] = (60, 60, 60)
    band_preview[band_only] = (0, 220, 60)

    # (b) Smooth-curve preview: one polyline per loop
    vec_preview = cleaned_bgr.copy()
    vec_preview[trace_px] = (
        vec_preview[trace_px].astype(np.int32) * 0.55
    ).clip(0, 255).astype(np.uint8)

    for segments in segments_per_loop:
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

    # ------------------------------------------------------------------
    # Build the "gap-fixed" reference PNG: black uniform band on white,
    # for use as the visible image layer in the SVG overlay.
    # ------------------------------------------------------------------
    fixed_visual = np.full((H, W, 3), 255, dtype=np.uint8)
    fixed_visual[band > 0] = (0, 0, 0)
    ok, fixed_png_bytes = cv2.imencode(".png", fixed_visual)
    if not ok:
        raise RuntimeError("Failed to encode gap-fixed PNG")

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    p_band     = out_dir / "uniform_band.png"
    p_fixed    = out_dir / "gap_fixed.png"
    p_preview  = out_dir / "vector_preview.png"
    p_skel     = out_dir / "closed_centerline.png"
    p_dxf      = out_dir / f"{args.name}.dxf"
    p_svg      = out_dir / f"{args.name}.svg"
    p_overlay  = out_dir / f"{args.name}_with_image.svg"

    save_image(p_band,    band_preview)
    save_image(p_preview, vec_preview)
    save_image(p_skel,    centerline)
    p_fixed.write_bytes(bytes(fixed_png_bytes))

    export_dxf_smooth(segments_per_loop, p_dxf, image_height=H)
    export_svg_smooth(segments_per_loop, (H, W), p_svg)
    export_svg_with_layers(
        segments_per_loop=segments_per_loop,
        original_png_path=in_path,
        gap_fixed_png_bytes=bytes(fixed_png_bytes),
        image_size=(H, W),
        output_path=p_overlay,
    )

    print(f"\nOutputs:")
    print(f"  Closed centerline: {p_skel}")
    print(f"  Uniform band:      {p_band}")
    print(f"  Gap-fixed PNG:     {p_fixed}")
    print(f"  Smooth preview:    {p_preview}")
    print(f"  DXF (smooth):      {p_dxf}")
    print(f"  SVG (smooth):      {p_svg}")
    print(f"  SVG (layered):     {p_overlay}  <- opens by default")

    if not args.no_open:
        _open_file(p_overlay)


if __name__ == "__main__":
    main()
