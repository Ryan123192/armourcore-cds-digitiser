"""Compare multiple adaptive-Method-F variants on the batch image.

Runs the SAME tool-detection pipeline for each variant, swapping only
the adaptive Method F selection logic.  Produces a labeled vector
preview for each variant so they can be eyeballed side-by-side.

Variants implemented:
  baseline — current production behaviour (v5 reverted, pre-culling)
  A        — bridge-count penalty in adaptive scoring (prefer kernels
             that need fewer bridges to close)
  C        — endpoint quality filter (drop endpoints whose distance-map
             value is large, i.e., sitting deep inside a close-thickened
             region rather than at a real trace edge)
  E        — same-side detection via medial-axis distance map (reject
             candidate pairs whose perpendicular offset across their
             tangent differs significantly — indicates cross-stub
             bridging)

Outputs go to data/outputs/phase03_testing/_experiments/<variant>/
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
from skimage.morphology import medial_axis

from test_phase3_iso_vectorise import (
    _avg_trace_thickness,
    _bezier_segments_smart,
    _chord_obstacle_score,
    _prune_skeleton_branches,
    _segments_intersect,
    _skel_endpoints,
    _walk_back,
    find_inside_edge_contours,
    render_uniform_band,
)
from test_phase3_batch import detect_tools


# ===================================================================
# Core Method F (replicated here so we can vary each variant cleanly)
# ===================================================================

def _method_F_core(
    trace_binary: np.ndarray,
    base_k: int,
    bridge_max_dist_px: int,
    *,
    # Variant hooks
    endpoint_quality_filter: bool = False,
    endpoint_max_distance: float = 6.0,
    same_side_filter: bool = False,
    same_side_max_perp: float = 10.0,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Single Method F call with optional variant hooks."""
    k = max(3, base_k | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(trace_binary, cv2.MORPH_CLOSE, kernel)
    skel_bool, distance_map = medial_axis(closed > 0, return_distance=True)
    skel = skel_bool.astype(np.uint8) * 255
    skel = _prune_skeleton_branches(skel, min_branch_px=6)

    # Endpoints
    ep_x, ep_y = _skel_endpoints(skel)
    descriptors: list[dict] = []
    for ex, ey in zip(ep_x.tolist(), ep_y.tolist()):
        path = _walk_back(skel, int(ex), int(ey), 15)
        if len(path) < 4:
            continue
        d = np.array(
            [path[0][0] - path[-1][0], path[0][1] - path[-1][1]],
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(d))
        if norm < 1e-9:
            continue

        # ---- Variant C: endpoint quality filter ----
        if endpoint_quality_filter:
            local_dist = float(distance_map[int(ey), int(ex)])
            if local_dist > endpoint_max_distance:
                continue

        descriptors.append({
            "pos":  np.array([float(ex), float(ey)]),
            "dir":  d / norm,
            "dist": float(distance_map[int(ey), int(ex)]),
        })

    # Same-component check
    n_comp_total, comp_labels = cv2.connectedComponents(skel, 8)
    n_real_comp = n_comp_total - 1
    allow_same_component = (n_real_comp <= 1)
    ep_comps = [
        int(comp_labels[int(d["pos"][1]), int(d["pos"][0])])
        if 0 <= int(d["pos"][1]) < skel.shape[0]
           and 0 <= int(d["pos"][0]) < skel.shape[1]
        else 0
        for d in descriptors
    ]

    # Candidates
    strong_facing_min = 0.40
    facing_threshold  = -0.05
    obstacle_threshold = 0.30
    n = len(descriptors)
    cands_strong = []
    cands_weak   = []
    for i in range(n):
        Ai  = descriptors[i]["pos"]
        dAi = descriptors[i]["dir"]
        for j in range(i + 1, n):
            if (not allow_same_component
                    and ep_comps[i] != 0
                    and ep_comps[i] == ep_comps[j]):
                continue
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
            if _chord_obstacle_score(trace_binary, Ai, Bj) > obstacle_threshold:
                continue

            # ---- Variant E: same-side detection ----
            if same_side_filter:
                # Perpendicular offset: each endpoint's tangent has a
                # perpendicular direction.  The midpoint of the chord
                # projected onto that perpendicular tells us "which side"
                # of the trace the bridge sits on.  Same-side pairs have
                # similar (small) perpendicular components on both
                # endpoints' frames.
                perp_A = np.array([-dAi[1], dAi[0]])
                perp_B = np.array([-dBj[1], dBj[0]])
                proj_A = float(np.dot(diff, perp_A))
                proj_B = float(np.dot(-diff, perp_B))
                # If both projections have the same sign and small
                # magnitude, the bridge is roughly co-linear (same-side).
                # If they have OPPOSITE signs, the bridge crosses from
                # one side of the trace to the other (cross-stub).
                if abs(proj_A + proj_B) > same_side_max_perp * 2:
                    # Cross-side bridge — REJECT
                    continue

            entry = (dist, i, j)
            if sA >= strong_facing_min and sB >= strong_facing_min:
                cands_strong.append(entry)
            else:
                cands_weak.append(entry)

    # Closest-first + non-crossing
    cands_strong.sort(key=lambda t: t[0])
    cands_weak.sort(key=lambda t: t[0])
    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    accepted_chords: list[tuple[np.ndarray, np.ndarray]] = []

    def _try(cs):
        for _d, i, j in cs:
            if i in used or j in used:
                continue
            A = descriptors[i]["pos"]
            B = descriptors[j]["pos"]
            crosses = False
            for (Ai, Bi) in accepted_chords:
                if _segments_intersect(A, B, Ai, Bi):
                    crosses = True
                    break
            if crosses:
                continue
            used.add(i); used.add(j)
            pairs.append((i, j))
            accepted_chords.append((A, B))

    _try(cands_strong)
    _try(cands_weak)

    # Render
    centerline = skel.copy()
    bridges_mask = np.zeros_like(skel)
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
        cv2.polylines(centerline,   [curve], False, 255, 1, cv2.LINE_8)
        cv2.polylines(bridges_mask, [curve], False, 255, 1, cv2.LINE_8)

    return centerline, len(pairs), bridges_mask


# ===================================================================
# Variant adaptive functions
# ===================================================================

def _adaptive_with_score(
    trace_binary: np.ndarray,
    band_thickness: int,
    score_fn,
    *,
    method_F_kwargs: dict | None = None,
    kernel_sizes: list[int] | None = None,
    bridge_max_dist_px: int = 250,
    min_loop_abs_px: float = 500.0,
    min_loop_rel: float = 0.05,
) -> tuple[np.ndarray, int, int, int, int, np.ndarray]:
    if kernel_sizes is None:
        kernel_sizes = [15, 20, 25, 30, 35, 40, 50, 60]
    method_F_kwargs = method_F_kwargs or {}
    H, W = trace_binary.shape[:2]
    image_area = float(H * W)
    bucket = max(image_area * 0.10, 1.0)

    results = []
    for k in kernel_sizes:
        cl, n_br, bm = _method_F_core(
            trace_binary, base_k=k, bridge_max_dist_px=bridge_max_dist_px,
            **method_F_kwargs,
        )
        ep_x, _ = _skel_endpoints(cl)
        n_ep = len(ep_x)
        band = render_uniform_band(cl, band_thickness)
        loops = find_inside_edge_contours(
            band,
            min_area_abs_px=min_loop_abs_px,
            min_area_relative=min_loop_rel,
            bridge_mask=bm,
            verbose=False,
        )
        n_loops = len(loops)
        total_area = sum(float(cv2.contourArea(c)) for c in loops)
        results.append({
            "k": k, "n_ep": n_ep, "n_loops": n_loops,
            "total_area": total_area, "bucket": int(total_area / bucket),
            "centerline": cl, "n_bridges": n_br,
            "bridges_mask": bm,
        })
        if n_ep == 0 and n_loops >= 1:
            break

    best = max(results, key=score_fn)
    return (
        best["centerline"], best["n_bridges"],
        best["k"], best["n_ep"], best["n_loops"],
        best["bridges_mask"],
    )


def adaptive_baseline(trace_binary, band_thickness, **kwargs):
    """Current production scoring."""
    return _adaptive_with_score(
        trace_binary, band_thickness,
        score_fn=lambda r: (r["n_loops"], r["bucket"], -r["k"], -r["n_ep"]),
        **kwargs,
    )


def adaptive_A_bridge_count(trace_binary, band_thickness, **kwargs):
    """A — bridge-count penalty in scoring."""
    return _adaptive_with_score(
        trace_binary, band_thickness,
        score_fn=lambda r: (
            r["n_loops"], r["bucket"], -r["n_bridges"], -r["k"], -r["n_ep"]
        ),
        **kwargs,
    )


def adaptive_C_endpoint_quality(trace_binary, band_thickness, **kwargs):
    """C — endpoint quality filter."""
    return _adaptive_with_score(
        trace_binary, band_thickness,
        score_fn=lambda r: (r["n_loops"], r["bucket"], -r["k"], -r["n_ep"]),
        method_F_kwargs={"endpoint_quality_filter": True,
                         "endpoint_max_distance": 6.0},
        **kwargs,
    )


def adaptive_E_same_side(trace_binary, band_thickness, **kwargs):
    """E — same-side detection via perp-offset check."""
    return _adaptive_with_score(
        trace_binary, band_thickness,
        score_fn=lambda r: (r["n_loops"], r["bucket"], -r["k"], -r["n_ep"]),
        method_F_kwargs={"same_side_filter": True,
                         "same_side_max_perp": 10.0},
        **kwargs,
    )


VARIANTS = {
    "baseline": adaptive_baseline,
    "A_bridge_count_penalty": adaptive_A_bridge_count,
    "C_endpoint_quality":     adaptive_C_endpoint_quality,
    "E_same_side":            adaptive_E_same_side,
}


# ===================================================================
# Batch processing for one variant
# ===================================================================

def run_variant(variant_name: str, adaptive_fn, image_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cleaned = cv2.imread(str(image_path))
    H, W = cleaned.shape[:2]
    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    trace = ((gray < 170).astype(np.uint8)) * 255

    tools = detect_tools(trace, group_dilate_radius=20,
                         min_tool_area_px=5000, verbose=False)

    vec_preview = cleaned.copy()
    trace_px = trace > 0
    vec_preview[trace_px] = (
        vec_preview[trace_px].astype(np.int32) * 0.55
    ).clip(0, 255).astype(np.uint8)

    n_total_loops = 0
    total_bridges = 0
    print(f"  [{variant_name}] processing {len(tools)} tools...")
    for tool_i, tool in enumerate(tools):
        x, y, w, h = tool["bbox"]; pad = 30
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(W, x + w + pad), min(H, y + h + pad)
        crop = tool["trace"][y1:y2, x1:x2]
        crop = (crop > 0).astype(np.uint8) * 255
        if not np.any(crop > 0):
            continue
        bt = max(2, int(round(_avg_trace_thickness(crop))))
        cl, n_br, kk, ep, _n_loops, bm = adaptive_fn(
            crop, bt,
            bridge_max_dist_px=250,
            min_loop_abs_px=500,
            min_loop_rel=0.05,
        )
        band = render_uniform_band(cl, bt)
        contours = find_inside_edge_contours(
            band, min_area_abs_px=500, min_area_relative=0.05,
            bridge_mask=bm, verbose=False,
        )
        total_bridges += n_br
        for c in contours:
            simp = cv2.approxPolyDP(c, 4.5, closed=True).reshape(-1, 2)
            if len(simp) < 4:
                continue
            pts = simp.astype(np.float64)
            bezier = _bezier_segments_smart(
                pts, tension=0.5, alpha=0.5, corner_angle_deg=60,
            )
            n_total_loops += 1
            # Sample + draw onto preview (offset to full image coords)
            sampled_parts = []
            for p1, cp1, cp2, p2 in bezier:
                ts = np.linspace(0, 1, 24, endpoint=False)
                mt = 1.0 - ts
                seg = (
                    mt[:, None] ** 3 * p1
                    + 3 * mt[:, None] ** 2 * ts[:, None] * cp1
                    + 3 * mt[:, None] * ts[:, None] ** 2 * cp2
                    + ts[:, None] ** 3 * p2
                )
                seg = seg + np.array([float(x1), float(y1)])
                sampled_parts.append(seg)
            if not sampled_parts:
                continue
            sampled = np.vstack(sampled_parts).round().astype(np.int32)
            cv2.polylines(
                vec_preview, [sampled], True, (0, 0, 255), 2, cv2.LINE_AA,
            )
        print(f"    Tool {tool_i + 1}: k={kk}  bridges={n_br}  ep={ep}  "
              f"loops_drawn={n_total_loops}")

    # Label the preview
    cv2.putText(
        vec_preview, variant_name,
        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (50, 50, 50), 3,
    )

    out_path = out_dir / f"{variant_name}.png"
    cv2.imwrite(str(out_path), vec_preview)
    print(f"    -> {out_path}  ({total_bridges} total bridges, "
          f"{n_total_loops} loops drawn)")


def main() -> None:
    repo = Path(__file__).parent.parent
    image_path = repo / "data" / "inputs" / "phase03_testing" / "BatchToolUnfixed01.png"
    out_root = repo / "data" / "outputs" / "phase03_testing" / "_experiments"
    out_root.mkdir(parents=True, exist_ok=True)

    print("\n=== Variant comparison on BatchToolUnfixed01 ===\n")
    for name, fn in VARIANTS.items():
        print(f"--- {name} ---")
        run_variant(name, fn, image_path, out_root)
        print()

    print(f"\nAll outputs in: {out_root}")


if __name__ == "__main__":
    main()
