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


def _segments_intersect(
    p1: np.ndarray, p2: np.ndarray,
    p3: np.ndarray, p4: np.ndarray,
) -> bool:
    """True iff line segment p1-p2 properly crosses segment p3-p4
    (touching only at endpoints does not count as crossing).
    """
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) - (B[1] - A[1]) * (C[0] - A[0])
    d1 = ccw(p3, p4, p1)
    d2 = ccw(p3, p4, p2)
    d3 = ccw(p1, p2, p3)
    d4 = ccw(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def _chord_obstacle_score(
    trace_binary: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    n_samples: int = 30,
) -> float:
    """Fraction of the inner 70 % of the A→B chord that hits trace pixels.

    Both endpoints are by definition on the trace (they are skeleton
    endpoints), so we ignore the first/last 15 % of the chord.  A high
    score means the straight chord passes THROUGH unrelated ink — that
    is, the bridge would cut across the tool's existing outline (a wrong
    pairing).  Low score means the chord travels through empty space (a
    real gap).
    """
    pts = np.linspace(A, B, n_samples)
    lo, hi = int(0.15 * n_samples), int(0.85 * n_samples)
    inner = pts[lo:hi]
    H, W = trace_binary.shape[:2]
    hits = 0
    for px, py in inner:
        ix, iy = int(px), int(py)
        if 0 <= ix < W and 0 <= iy < H and trace_binary[iy, ix] > 0:
            hits += 1
    return hits / max(1, len(inner))


def _prune_skeleton_branches(
    skel: np.ndarray,
    min_branch_px: int = 12,
    max_passes: int = 4,
) -> np.ndarray:
    """Remove short side-branches from a 1-px skeleton.

    A "short branch" is a path that starts at a degree-1 endpoint and
    reaches a junction (degree >= 3) in <= *min_branch_px* steps.  Those
    are typically noise spurs from rough edges in the underlying trace,
    NOT real gap endpoints.  Removing them prevents Method F from
    pairing a real gap endpoint with a phantom branch endpoint and
    producing the "bridge to a random spot on the skeleton" artefact.

    Iterative: removing one branch can expose a now-isolated short stub
    that is also worth pruning, so we repeat up to *max_passes* times
    or until nothing more is pruned.

    Endpoints on stubs longer than *min_branch_px*, or stubs that dead-
    end (no junction reached), are preserved — those are real gap edges.
    """
    if min_branch_px <= 0:
        return skel

    out = skel.copy()
    H, W = out.shape

    kernel = np.ones((3, 3), np.uint8)
    kernel[1, 1] = 0

    for _ in range(max_passes):
        sk8 = (out > 0).astype(np.uint8)
        nb = cv2.filter2D(sk8, cv2.CV_8U, kernel)
        endpoint_ys, endpoint_xs = np.where((sk8 > 0) & (nb == 1))
        if len(endpoint_xs) == 0:
            break

        to_remove: list[tuple[int, int]] = []
        skel_bool = out > 0

        for ex, ey in zip(endpoint_xs.tolist(), endpoint_ys.tolist()):
            # Walk along the branch.  Stop early if a junction is hit.
            cx, cy = int(ex), int(ey)
            visited: set[tuple[int, int]] = {(cx, cy)}
            path: list[tuple[int, int]] = [(cx, cy)]
            hit_junction = False
            for _step in range(min_branch_px + 1):
                # Look one step ahead
                next_xy = None
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
                            next_xy = (nx, ny)
                            break
                    if next_xy is not None:
                        break
                if next_xy is None:
                    break
                nx, ny = next_xy
                # If the next pixel is a junction (3+ neighbours), this
                # branch terminates HERE; mark length and stop.
                if int(nb[ny, nx]) >= 3:
                    hit_junction = True
                    break
                path.append(next_xy)
                visited.add(next_xy)
                cx, cy = nx, ny

            # Prune iff: we found a junction AND the branch was short.
            # Stubs that ran out into thin air (hit_junction=False) are
            # real gap edges — keep them.
            if hit_junction and len(path) <= min_branch_px:
                to_remove.extend(path)

        if not to_remove:
            break
        for px, py in to_remove:
            out[py, px] = 0

    return out


def method_F_closed_centerline(
    trace_binary: np.ndarray,
    base_k: int = 25,
    bridge_max_dist_px: int = 200,
    lookback_px: int = 15,
    facing_threshold: float = -0.05,
    obstacle_threshold: float = 0.30,
    prune_min_branch_px: int = 6,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Method F → returns (closed_centerline_skeleton_uint8, n_bridges_added).

    Combines the medial axis of the CLOSED trace with Bezier-bridge
    completions for any remaining skeleton endpoints, producing a single
    closed 1-pixel-wide centerline.

    *prune_min_branch_px* — short skeleton side-branches (junction → endpoint
    paths up to this length) are removed before endpoint pairing.  Prevents
    Method F from bridging a real gap to a phantom branch endpoint.
    """
    # Stage 1: CLOSE + medial axis
    k = max(3, base_k | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(trace_binary, cv2.MORPH_CLOSE, kernel)
    skel_bool, _ = medial_axis(closed > 0, return_distance=True)
    skel = skel_bool.astype(np.uint8) * 255

    # Stage 1b: prune skeleton noise branches.  Without this, the medial
    # axis at higher resolutions has phantom side-branches whose endpoints
    # would be paired by Method F, producing "bridge from a real gap to a
    # random point on the skeleton" artefacts.
    skel = _prune_skeleton_branches(skel, min_branch_px=prune_min_branch_px)

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

    # ------------------------------------------------------------------
    # Same-component check.
    #
    # The legitimate gap bridges always connect endpoints from DIFFERENT
    # skeleton components (the gap is what disconnects them).  Two
    # endpoints of the SAME stub bridging together would cut across
    # existing trace.
    #
    # Exception: if the whole pruned skeleton is a single connected
    # component (= one closed loop with one gap to bridge), the two
    # endpoints ARE supposed to pair, so we allow same-component pairs
    # in that case only.
    # ------------------------------------------------------------------
    n_comp_total, comp_labels = cv2.connectedComponents(skel, 8)
    n_real_comp = n_comp_total - 1
    allow_same_component = (n_real_comp <= 1)

    def _ep_comp(desc) -> int:
        px = int(desc["pos"][0])
        py = int(desc["pos"][1])
        if 0 <= px < skel.shape[1] and 0 <= py < skel.shape[0]:
            return int(comp_labels[py, px])
        return 0

    endpoint_comps = [_ep_comp(d) for d in descriptors]

    # ------------------------------------------------------------------
    # Two-pass matching
    #   Pass 1: STRONG-FACING candidates (sA, sB >= strong_facing_min).
    #           These are "obvious same-side gap" pairings — tangents
    #           align well with the bridge chord direction.  Get them
    #           first so they consume their endpoints before any weak
    #           candidates can grab them.
    #   Pass 2: remaining endpoints with normal (lax) facing_threshold.
    #           Handles perpendicular-tangent cases like 04's top-of-
    #           blade obliterated span.
    # Both passes use closest-first greedy + non-crossing constraint.
    # ------------------------------------------------------------------
    strong_facing_min = 0.40

    n = len(descriptors)
    candidates_strong = []
    candidates_weak   = []
    for i in range(n):
        Ai  = descriptors[i]["pos"]
        dAi = descriptors[i]["dir"]
        for j in range(i + 1, n):
            # Same-component rejection (multi-component case only)
            if (not allow_same_component
                    and endpoint_comps[i] != 0
                    and endpoint_comps[i] == endpoint_comps[j]):
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
            entry = (dist, i, j)
            if sA >= strong_facing_min and sB >= strong_facing_min:
                candidates_strong.append(entry)
            else:
                candidates_weak.append(entry)

    candidates_strong.sort(key=lambda t: t[0])
    candidates_weak.sort(key=lambda t: t[0])

    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    accepted_chords: list[tuple[np.ndarray, np.ndarray]] = []

    def _try_accept(candidates: list) -> None:
        for _dist, i, j in candidates:
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

    _try_accept(candidates_strong)   # pass 1
    _try_accept(candidates_weak)     # pass 2

    # Render bridges directly into the centerline as 1-px lines (no
    # antialiasing, so the centerline stays exactly 1px wide everywhere).
    # ALSO build a separate bridges_mask of just the bridge pixels —
    # used downstream by find_inside_edge_contours to recognise and
    # drop holes that are bounded by these auto-added bridges (= artefact
    # compartments).
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
        cv2.polylines(centerline,    [curve], False, 255, 1, cv2.LINE_8)
        cv2.polylines(bridges_mask,  [curve], False, 255, 1, cv2.LINE_8)

    return centerline, len(pairs), bridges_mask


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
    min_area_abs_px: float = 500.0,
    min_area_relative: float = 0.05,
    bridge_mask: np.ndarray | None = None,
    bridge_boundary_frac_max: float = 0.30,
    verbose: bool = False,
) -> list[np.ndarray]:
    """Find ALL interior holes in a closed band, sorted largest-first.

    Filters out junction artefacts using BOTH:
      * Absolute floor — drop loops smaller than *min_area_abs_px* (handles
        very small noise).
      * Relative floor — drop loops smaller than *min_area_relative* of
        the largest loop's area (handles junction artefacts that pop up
        between multi-loop pieces, which are typically <5% of the main
        component areas).

    A multi-loop tool — e.g., a screwdriver drawn as handle + stem — has
    one hole per loop, plus possible small holes at the intersection.
    The relative filter cleans those up.
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

    # Collect all interior holes
    holes: list[tuple[float, np.ndarray]] = []
    for i, c in enumerate(contours):
        if hierarchy[0][i][3] == -1:
            continue
        area = float(cv2.contourArea(c))
        holes.append((area, c))
    if not holes:
        return []

    # OPTIONAL: bridge-boundary filter.  When *bridge_mask* is provided
    # (= the pixels added by Method F's stage-2 Bezier bridges), we drop
    # any hole whose contour has a long CONTIGUOUS run of bridge pixels
    # — those compartments were cut off by a single long auto-added
    # bridge.
    #
    # Why max-contiguous-run rather than total fraction:
    #   * Legitimate handle/blade outlines may have many SHORT legit
    #     bridges (closing little gaps in the customer's trace).  Total
    #     bridge fraction can be 10-15 %, but each contiguous run is
    #     small (< 20 px).
    #   * Artefact compartments are bounded by ONE long bridge spanning
    #     the entire compartment, giving a single contiguous run of
    #     50+ pixels.
    H, W = band.shape[:2]
    if bridge_mask is not None and np.any(bridge_mask > 0):
        bridge_dilation = 5
        bk = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * bridge_dilation + 1, 2 * bridge_dilation + 1),
        )
        bridge_zone = cv2.dilate(bridge_mask, bk)

        filtered: list[tuple[float, np.ndarray]] = []
        for area, c in holes:
            pts = c.reshape(-1, 2)
            n = len(pts)
            on_bridge = np.zeros(n, dtype=np.uint8)
            for k_idx in range(n):
                ix, iy = int(pts[k_idx, 0]), int(pts[k_idx, 1])
                if 0 <= ix < W and 0 <= iy < H and bridge_zone[iy, ix] > 0:
                    on_bridge[k_idx] = 1

            # Longest contiguous run with wrap-around handling.  Doubling
            # the array and scanning [0..2n) lets a run that crosses the
            # boundary index 0/n-1 be measured correctly.  Cap at n.
            extended = np.concatenate([on_bridge, on_bridge])
            max_run = 0
            cur_run = 0
            for v in extended:
                if v:
                    cur_run += 1
                    if cur_run > max_run:
                        max_run = cur_run
                else:
                    cur_run = 0
            max_run = min(max_run, n)

            # Drop if the longest contiguous bridge segment exceeds
            # max(50 px, 15 % of perimeter)
            run_threshold = max(50, int(0.15 * n))
            if max_run > run_threshold:
                if verbose:
                    print(f"    DROP bridge-bounded hole area={area:.0f}px "
                          f"(longest contiguous bridge run = {max_run}px "
                          f"of {n}px perimeter, threshold={run_threshold}px)")
                continue
            filtered.append((area, c))
        holes = filtered

    if not holes:
        return []

    # Global filters: absolute floor + relative floor of the largest hole.
    holes.sort(key=lambda t: -t[0])
    largest_area = holes[0][0]
    abs_threshold = float(min_area_abs_px)
    rel_threshold = largest_area * min_area_relative

    kept: list[np.ndarray] = []
    for area, c in holes:
        if area < abs_threshold or area < rel_threshold:
            if verbose:
                print(f"    DROP loop area={area:.0f}px "
                      f"(abs_min={abs_threshold:.0f}, "
                      f"rel_min={rel_threshold:.0f})")
            continue
        kept.append(c)
    return kept


def adaptive_method_F(
    trace_binary: np.ndarray,
    band_thickness: int,
    kernel_sizes: list[int] | None = None,
    bridge_max_dist_px: int = 400,
    min_loop_abs_px: float = 500.0,
    min_loop_rel: float = 0.05,
) -> tuple[np.ndarray, int, int, int, int, np.ndarray]:
    """Adaptive Method F: try multiple close kernels and pick the best.

    The "best" combines two criteria:
      1. Maximum number of meaningful loops detected (after the relative
         filter)  — escalating the kernel often MERGES distinct loops
         which is worse than leaving them separate.
      2. Fewest skeleton endpoints  — measure of how completely the
         centerline is closed.
      3. Tiebreaker: smaller kernel (less shape distortion).

    Stops early on a "perfect" result (>=1 loop and 0 endpoints).

    Returns:
        (centerline, n_bridges, chosen_kernel, n_endpoints, n_loops)
    """
    if kernel_sizes is None:
        kernel_sizes = [15, 20, 25, 30, 35, 40, 50, 60]

    results = []
    for k in kernel_sizes:
        centerline, n_bridges, bridges_mask = method_F_closed_centerline(
            trace_binary, base_k=k, bridge_max_dist_px=bridge_max_dist_px,
        )
        ep_x, _ = _skel_endpoints(centerline)
        n_ep = len(ep_x)

        band = render_uniform_band(centerline, band_thickness)
        loops = find_inside_edge_contours(
            band,
            min_area_abs_px=min_loop_abs_px,
            min_area_relative=min_loop_rel,
            bridge_mask=bridges_mask,
            verbose=False,
        )
        n_loops = len(loops)
        total_area = sum(float(cv2.contourArea(c)) for c in loops)
        results.append({
            "k": k, "n_ep": n_ep, "n_loops": n_loops,
            "total_area": total_area,
            "centerline": centerline, "n_bridges": n_bridges,
            "bridges_mask": bridges_mask,
        })

        if n_ep == 0 and n_loops >= 1:
            break

    # Score (max → wins):
    #   (n_loops, area_bucket, -k, -n_ep)
    # ----------
    # 1. n_loops      — more pieces traced beats fewer.
    # 2. area_bucket  — total enclosed area in 10%-of-image-area buckets.
    #                   Catches the case where a small kernel "found 2 loops"
    #                   but one of them is just a tiny tip; a larger kernel
    #                   that found the full blade as the second loop has
    #                   massively more total area, so it wins.
    # 3. -k           — among comparable kernels, smaller distorts less.
    # 4. -n_ep        — fewer residual endpoints = cleaner closure.
    #
    # The bucket on area means small differences (10%) don't override the
    # smallest-kernel preference, but large differences (e.g., handle alone
    # vs handle+full-blade) do.
    H, W = trace_binary.shape[:2]
    image_area = float(H * W)
    bucket = max(image_area * 0.10, 1.0)

    def score(r):
        return (r["n_loops"], int(r["total_area"] / bucket),
                -r["k"], -r["n_ep"])

    best = max(results, key=score)
    return (
        best["centerline"], best["n_bridges"],
        best["k"], best["n_ep"], best["n_loops"],
        best["bridges_mask"],
    )


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
    parser.add_argument("--close-kernel", type=int, default=None, dest="close_kernel",
                        help="Force a specific MORPH_CLOSE kernel.  By default "
                             "the script searches across [15..100] and picks "
                             "the smallest one that fully closes the centerline.")
    parser.add_argument("--bridge-dist", type=int, default=250, dest="bridge_dist",
                        help="Method F max distance for skeleton-endpoint "
                             "Bezier bridges (default: 250px).")
    parser.add_argument("--min-loop-rel", type=float, default=0.05,
                        dest="min_loop_rel",
                        help="Drop inner loops smaller than this fraction of "
                             "the largest loop (default: 0.05 = 5%%) — cleans "
                             "up junction artefacts.")
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
    band_thickness = max(2, int(round(avg_thick)))

    # ------------------------------------------------------------------
    # 2. Method F → closed centerline (1-pixel wide)
    #    Adaptive: scan kernel sizes; prefer most loops, fewest endpoints,
    #    smallest kernel.  User can force via --close-kernel.
    # ------------------------------------------------------------------
    if args.close_kernel is not None:
        centerline, n_bridges, bridges_mask = method_F_closed_centerline(
            trace, base_k=args.close_kernel,
            bridge_max_dist_px=args.bridge_dist,
        )
        chosen_kernel = args.close_kernel
        ep_x, _ = _skel_endpoints(centerline)
        n_remaining = len(ep_x)
        print(f"Method F (forced kernel={chosen_kernel}): "
              f"{n_bridges} bridges, {n_remaining} remaining endpoints")
    else:
        (centerline, n_bridges, chosen_kernel, n_remaining, n_loops_found,
         bridges_mask) = adaptive_method_F(
                trace,
                band_thickness=band_thickness,
                bridge_max_dist_px=args.bridge_dist,
                min_loop_abs_px=500.0,
                min_loop_rel=args.min_loop_rel,
            )
        status = (
            f"fully closed ({n_loops_found} loops)"
            if n_remaining == 0
            else f"{n_remaining} endpoints remain ({n_loops_found} loops)"
        )
        print(f"Method F (adaptive: kernel={chosen_kernel}): "
              f"{n_bridges} bridges, {status}")
    print(f"Centerline pixels: {int(np.count_nonzero(centerline)):,}")

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
    inner_contours = find_inside_edge_contours(
        band,
        min_area_abs_px=500.0,
        min_area_relative=args.min_loop_rel,
        bridge_mask=bridges_mask,
        verbose=True,
    )
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
