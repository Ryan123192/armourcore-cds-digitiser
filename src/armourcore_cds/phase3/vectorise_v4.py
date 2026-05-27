"""Phase 3 - vectorise_v4: CLOSED-LOOPS ONLY + comprehensive logging.

Insight from user's reference image
===================================
The reference shows exactly 11 vectors - all CLOSED LOOPS, no open arcs.
Previous v3 produced 25-46 paths because it emitted open arcs from:
  - text fragments ("Boundary Line (Small Insert)")
  - major-grid-line residue
  - skeleton bridges between unrelated components
  - short branches off closed loops

Strategy: SUPPRESS OPEN ARCS ENTIRELY.

For each skeleton component:
  1. If naturally closed (no endpoints): keep as is.
  2. If has 2 endpoints close to each other: bridge them into a loop.
  3. Otherwise: drop.

This single change should drop ~80% of spurious paths.

In addition v4 emits a detailed JSON log per case so we can see at each
stage how many paths exist, why each was dropped, etc.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import medial_axis

from armourcore_cds.phase3.vectorise import (
    VectorPath, _rdp_simplify, _catmull_rom_to_bezier,
)
from armourcore_cds.phase3.vectorise_v3 import (
    remove_stray_satellites, _skel_neighbour_count,
    _prune_skeleton_spurs, chaikin_smooth,
)


def _angle_at_node(pts: np.ndarray, i: int) -> float:
    """Return interior angle (degrees) at node i of a closed polyline pts.
    180 = straight; 90 = right angle; 0 = u-turn."""
    n = len(pts)
    p_prev = pts[(i - 1) % n]
    p_cur = pts[i]
    p_next = pts[(i + 1) % n]
    v1 = p_prev - p_cur
    v2 = p_next - p_cur
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def _angle_aware_bezier(pts: np.ndarray,
                       sharp_angle_threshold_deg: float = 150.0,
                       tension: float = 1.0) -> list:
    """Build Bezier segments where SHARP corners (interior angle <
    threshold) get degenerate (straight-line) segments and SMOOTH
    transitions get standard Catmull-Rom curves.

    A 90° corner has interior angle ~90; a smooth point ~180.
    Default threshold 150° means anything within 30° of straight is
    smoothed; anything sharper is preserved as a corner.

    Returns list of (p1, cp1, cp2, p2) tuples like _catmull_rom_to_bezier.
    """
    n = len(pts)
    if n < 2:
        return []
    # Pre-compute angle at each node so we can detect SHARP NODES.
    is_sharp = [_angle_at_node(pts, i) < sharp_angle_threshold_deg
                for i in range(n)]

    segs: list[tuple] = []
    for i in range(n):
        p0 = pts[(i - 1) % n]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n]

        # If either ENDPOINT of this segment is a sharp corner, make the
        # control point near that endpoint degenerate (zero-length
        # tangent) so the segment becomes a straight line into/out of
        # the corner.  This preserves the corner without affecting
        # neighbouring smooth segments.
        if is_sharp[i]:
            cp1 = p1.copy()
        else:
            cp1 = p1 + tension * (p2 - p0) / 6.0

        if is_sharp[(i + 1) % n]:
            cp2 = p2.copy()
        else:
            cp2 = p2 - tension * (p3 - p1) / 6.0

        segs.append((p1.copy(), cp1, cp2, p2.copy()))
    return segs


@dataclass
class PathDiagnostic:
    """Per-path properties + outcome (KEPT / DROPPED) + reason."""
    index: int
    kind: str              # "natural_closed" or "bridged" or "open_arc"
    n_skel_pixels: int
    n_polyline_pts: int
    closed: bool
    bbox_xywh: tuple[int, int, int, int]
    bbox_w_mm: float
    bbox_h_mm: float
    area_px: float
    area_mm2: float
    perimeter_px: float
    compactness: float     # 4*pi*A / P^2
    outcome: str           # "kept", "dropped_open", "dropped_small", etc.
    drop_reason: str = ""


@dataclass
class StageReport:
    """Per-stage counts for the pipeline."""
    stage_input_mask_pixels: int = 0
    stage_after_stray_removal_pixels: int = 0
    stage_after_close_pixels: int = 0
    stage_skel_pixels: int = 0
    stage_skel_components: int = 0
    stage_natural_closed_loops: int = 0
    stage_open_arcs: int = 0
    stage_arcs_bridged: int = 0
    stage_arcs_dropped: int = 0
    final_paths: int = 0
    paths: list[PathDiagnostic] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "input_mask_px": self.stage_input_mask_pixels,
            "after_stray_removal_px": self.stage_after_stray_removal_pixels,
            "after_close_px": self.stage_after_close_pixels,
            "skel_px": self.stage_skel_pixels,
            "skel_components": self.stage_skel_components,
            "natural_closed_loops": self.stage_natural_closed_loops,
            "open_arcs": self.stage_open_arcs,
            "arcs_bridged_into_loops": self.stage_arcs_bridged,
            "arcs_dropped": self.stage_arcs_dropped,
            "final_paths": self.final_paths,
            "paths": [vars(p) for p in self.paths],
        }


# ---------------------------------------------------------------------------
# Skeleton walking - per-component
# ---------------------------------------------------------------------------

_N8 = [(-1, -1), (-1, 0), (-1, 1),
       (0, -1),           (0, 1),
       (1, -1),  (1, 0),  (1, 1)]


def _walk_from(skel: np.ndarray, start_yx: tuple[int, int],
               visited: np.ndarray) -> list[tuple[int, int]]:
    """Walk while we can find an unvisited 8-neighbour.  Iterative."""
    H, W = skel.shape
    path = [start_yx]
    visited[start_yx] = True
    cy, cx = start_yx
    while True:
        nxt = None
        for dy, dx in _N8:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < H and 0 <= nx < W and skel[ny, nx] and not visited[ny, nx]:
                nxt = (ny, nx)
                break
        if nxt is None:
            return path
        path.append(nxt)
        visited[nxt] = True
        cy, cx = nxt


def _polyline_length(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def _polyline_closed_area(pts: np.ndarray) -> tuple[float, float]:
    """Shoelace area + perimeter for a closed polyline (pts[0] == pts[-1])."""
    if len(pts) < 4:
        return 0.0, _polyline_length(pts)
    xs, ys = pts[:, 0], pts[:, 1]
    area = 0.5 * float(np.abs(np.dot(xs[:-1], ys[1:]) - np.dot(ys[:-1], xs[1:])))
    perim = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    return area, perim


# ---------------------------------------------------------------------------
# Per-component closed-loop extraction
# ---------------------------------------------------------------------------

def extract_closed_loops_per_component(
    skel: np.ndarray,
    bridge_max_gap_px: int,
    min_loop_perimeter_px: int,
) -> tuple[list[np.ndarray], int, int, int, int]:
    """Walk the skeleton component-by-component.  For each component:
      - if it has 0 endpoints -> natural closed loop, walk the cycle.
      - if it has 2 endpoints whose Euclidean distance <= bridge_max_gap:
           walk endpoint-to-endpoint then bridge -> closed loop.
      - otherwise drop.

    Returns (loops, n_natural, n_bridged, n_open_dropped, n_components).
    Loops are (N, 2) float64 in (x, y), guaranteed pts[0] == pts[-1].
    """
    skel_u8 = (skel > 0).astype(np.uint8)
    H, W = skel_u8.shape
    n_lbl, lbl = cv2.connectedComponents(skel_u8, connectivity=8)
    nb = _skel_neighbour_count(skel_u8)

    loops: list[np.ndarray] = []
    n_natural = 0
    n_bridged = 0
    n_dropped = 0

    for cid in range(1, n_lbl):
        comp = (lbl == cid)
        if comp.sum() < 10:
            continue
        ep_y, ep_x = np.where(comp & (nb == 1))
        n_endpoints = len(ep_x)

        visited = np.zeros_like(skel_u8, dtype=bool)
        visited[~comp] = True   # restrict walking to this component

        if n_endpoints == 0:
            # natural closed loop
            sy, sx = np.where(comp)
            path = _walk_from(skel_u8.astype(bool), (int(sy[0]), int(sx[0])),
                             visited)
            if len(path) < min_loop_perimeter_px:
                n_dropped += 1
                continue
            pts = np.array([(x, y) for (y, x) in path], dtype=np.float64)
            pts = np.vstack([pts, pts[0:1]])    # explicit close
            loops.append(pts)
            n_natural += 1

        elif n_endpoints == 2:
            sy_start, sx_start = int(ep_y[0]), int(ep_x[0])
            ey, ex = int(ep_y[1]), int(ep_x[1])
            gap = float(np.hypot(ex - sx_start, ey - sy_start))
            if gap > bridge_max_gap_px:
                n_dropped += 1
                continue
            path = _walk_from(skel_u8.astype(bool), (sy_start, sx_start),
                             visited)
            if len(path) < min_loop_perimeter_px:
                n_dropped += 1
                continue
            pts = np.array([(x, y) for (y, x) in path], dtype=np.float64)
            # Bridge: close the loop by straight-lining back to start
            pts = np.vstack([pts, pts[0:1]])
            loops.append(pts)
            n_bridged += 1

        else:
            # > 2 endpoints (junction): too messy, drop
            n_dropped += 1

    return loops, n_natural, n_bridged, n_dropped, n_lbl - 1


# ---------------------------------------------------------------------------
# Drop loops fully nested inside larger loops (avoid inner+outer duplicates)
# ---------------------------------------------------------------------------

def drop_nested_loops(loops: list[np.ndarray],
                     overlap_min: float = 0.85) -> list[np.ndarray]:
    """If loop A's bbox is >= overlap_min inside loop B's bbox AND A is
    smaller, drop A (it's the inner boundary of B's stroke)."""
    if len(loops) <= 1:
        return loops
    boxes = []
    for p in loops:
        xs, ys = p[:, 0], p[:, 1]
        boxes.append((float(xs.min()), float(ys.min()),
                     float(xs.max()), float(ys.max())))
    keep = [True] * len(loops)
    for i in range(len(loops)):
        if not keep[i]:
            continue
        xi0, yi0, xi1, yi1 = boxes[i]
        for j in range(len(loops)):
            if i == j or not keep[j]:
                continue
            xj0, yj0, xj1, yj1 = boxes[j]
            if (xj0 >= xi0 - 2 and yj0 >= yi0 - 2
                    and xj1 <= xi1 + 2 and yj1 <= yi1 + 2):
                area_i = (xi1 - xi0) * (yi1 - yi0)
                area_j = (xj1 - xj0) * (yj1 - yj0)
                if area_j < area_i * 0.95 and area_j > 0:
                    keep[j] = False
    return [loops[k] for k in range(len(loops)) if keep[k]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

ROUTE_PRESETS_V4: dict[str, dict] = {
    "pen": {
        "stray_proximity_px":     14,
        "pre_close_px":           11,
        "pre_close_wide_px":      21,
        "skel_prune_spurs":       25,
        "bridge_max_gap_px":      55,
        "min_loop_perimeter_px":  60,    # lowered to allow 5mm tracings
        "min_area_mm2":           25.0,  # 5mm x 5mm minimum
        "min_compactness":        0.02,
        "rdp_epsilon":            3.0,
        "smooth_iters":           0,
        "sharp_angle_threshold_deg": 150.0,
    },
    "pencil": {
        "stray_proximity_px":     18,
        "pre_close_px":           17,
        "skel_prune_spurs":       18,
        "bridge_max_gap_px":      80,
        "min_loop_perimeter_px":  60,
        "min_area_mm2":           25.0,
        "min_compactness":        0.02,
        "rdp_epsilon":            3.0,
        "smooth_iters":           0,
        "sharp_angle_threshold_deg": 150.0,
    },
}


def extract_vector_paths_closed_only(
    trace_mask: np.ndarray,
    design_width_mm: float,
    design_height_mm: float,
    route: str = "pen",
    save_stages_dir: Path | None = None,
    case_stem: str = "case",
    **overrides,
) -> tuple[list[VectorPath], StageReport, dict]:
    """Closed-loops-only extraction with full stage logging.

    Approach (revised after skeleton-strict approach gave 0 loops):
      1. Aggressive stray-satellite removal (kills text near outlines).
      2. Heavy morphological close (bridges all sub-stroke gaps so each
         shape outline becomes a solid filled blob).
      3. cv2.findContours with RETR_EXTERNAL -> one closed contour per
         remaining blob.  This is by construction always a closed loop.
      4. Filter by area / compactness / bbox dims.

    findContours guarantees closed loops + handles arbitrary topology
    that defeated the skeleton-per-component analysis.
    """
    cfg = ROUTE_PRESETS_V4.get(route, ROUTE_PRESETS_V4["pen"]).copy()
    cfg.update(overrides)
    report = StageReport()

    H, W = trace_mask.shape[:2]
    mm_per_px_x = design_width_mm / W
    mm_per_px_y = design_height_mm / H

    binary = (trace_mask > 0).astype(np.uint8) * 255
    report.stage_input_mask_pixels = int((binary > 0).sum())

    if save_stages_dir is not None:
        save_stages_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_stages_dir / f"{case_stem}_stage1_input.png"),
                   binary)

    # Stage 1: stray-satellite removal
    binary = remove_stray_satellites(
        binary, proximity_px=cfg["stray_proximity_px"])
    report.stage_after_stray_removal_pixels = int((binary > 0).sum())
    if save_stages_dir is not None:
        cv2.imwrite(str(save_stages_dir / f"{case_stem}_stage2_stray_removed.png"),
                   binary)

    # Stage 2: morphological close (gentle) - bridge sub-stroke gaps.
    # We do NOT global-floodfill here because if ANY shape has a tiny
    # outline gap the fill leaks across the whole paper, merging the
    # image into one mega-blob.  Per-shape floodfill happens later
    # safely within each bbox.
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (cfg["pre_close_px"], cfg["pre_close_px"]))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    report.stage_after_close_pixels = int((closed > 0).sum())

    # Optional second close pass at a larger kernel for shapes the small
    # kernel couldn't bridge.  Used for the multi-scale merge below.
    closed_wide = None
    wide_k = cfg.get("pre_close_wide_px", 0)
    if wide_k and wide_k > cfg["pre_close_px"]:
        k_w = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (wide_k, wide_k))
        closed_wide = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_w)
    if save_stages_dir is not None:
        cv2.imwrite(str(save_stages_dir / f"{case_stem}_stage3_closed.png"),
                   closed)

    # Stage 3 (debug only): keep skeleton image for compare-tile context
    if save_stages_dir is not None:
        skel = medial_axis(closed > 0).astype(np.uint8) * 255
        report.stage_skel_pixels = int((skel > 0).sum())
        cv2.imwrite(str(save_stages_dir / f"{case_stem}_stage4_skeleton.png"),
                   skel)

    # Stage 4: extract contours.  Use RETR_CCOMP so we get BOTH outer
    # boundaries AND holes.  For a stroked outline (1mm-thick ring), the
    # HOLE inside the stroke is the actual shape interior - tracing the
    # hole gives the true shape outline without the stroke-thickness
    # offset that RETR_EXTERNAL would add.  We prefer holes (children)
    # when they exist and only fall back to the outer contour when there
    # is no hole (i.e. the cleaning produced a solid filled blob).
    all_contours, hierarchy = cv2.findContours(
        closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    if hierarchy is not None and len(all_contours) > 0:
        hier = hierarchy[0]   # shape (N, 4): [next, prev, child, parent]
        # Map each parent index -> list of its hole indices
        children_of: dict[int, list[int]] = {}
        for i, h in enumerate(hier):
            parent = int(h[3])
            if parent >= 0:
                children_of.setdefault(parent, []).append(i)
        # Collect ALL contours that look like shape outlines.
        # Strategy:
        #   * For each parent that has NO children -> use the parent
        #     contour (it's a solid filled blob).
        #   * For each parent that HAS children -> each child (hole)
        #     is a separate shape interior.  Return ALL of them.
        #   * Don't bother with the parent contour in that case -
        #     it's the outer envelope around the cluster of shapes,
        #     not a tool shape itself.
        # This handles dense layouts where MORPH_CLOSE bridges across
        # shapes producing one big parent with many holes.
        chosen: list = []
        for i, h in enumerate(hier):
            if int(h[3]) >= 0:
                continue  # this is a hole; handled via its parent
            kids = children_of.get(i, [])
            if kids:
                # Each child = one shape interior.  Filter out very tiny
                # holes (likely text fragment interiors).
                for k in kids:
                    if cv2.contourArea(all_contours[k]) >= 200:
                        chosen.append(all_contours[k])
            else:
                chosen.append(all_contours[i])
        raw_contours = chosen
    else:
        raw_contours = list(all_contours)
    # (multi-scale wide-pass removed - the hole-preference approach via
    # RETR_CCOMP gives correct shape outlines without needing it.)
    report.stage_skel_components = len(raw_contours)

    # For each raw contour, check fill ratio.  If LOW (= ribbon around a
    # not-quite-closed stroke), do a local-bbox MORPH_CLOSE with bigger
    # kernels until it becomes a solid blob, then refind the external
    # contour within the bbox.  This rescues shapes whose stroke gaps
    # exceed the global pre_close kernel.
    n_natural = 0
    n_bridged = 0
    n_dropped = 0
    loops: list[np.ndarray] = []
    H, W = closed.shape

    image_area = H * W
    for c in raw_contours:
        if len(c) < 4:
            n_dropped += 1
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < 5 or bh < 5:
            n_dropped += 1
            continue
        bbox_area = bw * bh
        # Drop ribbons that span > 35% of the whole image - these are
        # almost always image-border artifacts, not real tool shapes.
        if bbox_area > image_area * 0.35:
            n_dropped += 1
            continue
        contour_area = float(cv2.contourArea(c))
        fill_ratio = contour_area / max(bbox_area, 1)

        if fill_ratio > 0.35:
            # Solid enough - take the contour as is
            pts = c.reshape(-1, 2).astype(np.float64)
            pts = np.vstack([pts, pts[0:1]])
            loops.append(pts)
            n_natural += 1
            continue

        # Ribbon: try local close inside contour's bbox + pad.  Pad scales
        # with kernel so we don't bleed into neighbours when k grows.
        orig_bw, orig_bh = bw, bh
        resolved = False
        for k_size in (17, 25, 35, 51, 71, 91):
            pad = max(6, k_size // 2 + 4)
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(W, x + bw + pad), min(H, y + bh + pad)
            sub = closed[y1:y2, x1:x2].copy()
            k_local = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                (k_size, k_size))
            sub_closed = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, k_local)
            sub_h, sub_w = sub_closed.shape
            ff = np.zeros((sub_h + 2, sub_w + 2), dtype=np.uint8)
            inv = cv2.bitwise_not(sub_closed)
            cv2.floodFill(inv, ff, (0, 0), 0)
            sub_solid = cv2.bitwise_or(sub_closed, inv)

            sub_contours, _ = cv2.findContours(
                sub_solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            picked = None
            for sc in sub_contours:
                sc_x, sc_y, sc_bw, sc_bh = cv2.boundingRect(sc)
                # Reject if rescued shape grew by more than 1 full kernel
                # in either dimension (would mean we bled into neighbours).
                if sc_bw > orig_bw + k_size or sc_bh > orig_bh + k_size:
                    continue
                if sc_bw * sc_bh < bbox_area * 0.5:
                    continue
                sc_fill = float(cv2.contourArea(sc)) / max(sc_bw * sc_bh, 1)
                # Reject only if essentially a perfect rectangle (bbox-fill
                # artefact).  0.98 leaves organic-but-densely-filled shapes
                # like the dome / rectangle alone.
                if sc_fill > 0.98:
                    continue
                if sc_fill > 0.35:
                    picked = sc
                    break
            if picked is not None:
                pts = picked.reshape(-1, 2).astype(np.float64)
                pts += np.array([x1, y1], dtype=np.float64)
                pts = np.vstack([pts, pts[0:1]])
                loops.append(pts)
                n_bridged += 1
                resolved = True
                break
        if not resolved:
            n_dropped += 1

    report.stage_natural_closed_loops = n_natural
    report.stage_arcs_bridged = n_bridged
    report.stage_arcs_dropped = n_dropped
    report.stage_open_arcs = n_bridged + n_dropped

    if save_stages_dir is not None:
        vis = np.full((H, W, 3), 255, dtype=np.uint8)
        for i, p in enumerate(loops):
            pts_int = p.astype(np.int32).reshape(-1, 1, 2)
            colour = ((i * 47) % 200 + 30,
                     (i * 91) % 200 + 30,
                     (i * 137) % 200 + 30)
            cv2.polylines(vis, [pts_int], isClosed=True, color=colour,
                         thickness=2)
        cv2.imwrite(str(save_stages_dir / f"{case_stem}_stage5_loops.png"),
                   vis)

    # Stage 5: drop nested loops (inner boundary of stroke) - but only
    # when both loops have similar bbox aspect (truly inner/outer of a
    # single stroked outline) so we don't kill siblings.
    # (Skipped here - RETR_EXTERNAL already removes children, and our
    # rescue logic doesn't produce overlapping shapes anymore.)

    # Stage 6: build VectorPaths + filter by area / compactness
    paths: list[VectorPath] = []
    for idx, pts in enumerate(loops):
        area_px, perim_px = _polyline_closed_area(pts)
        area_mm2 = area_px * mm_per_px_x * mm_per_px_y
        compactness = (4 * np.pi * area_px) / (perim_px * perim_px) if perim_px > 0 else 0
        xs, ys = pts[:, 0], pts[:, 1]
        x0, y0 = float(xs.min()), float(ys.min())
        bw, bh = float(xs.max() - x0), float(ys.max() - y0)
        bw_mm = bw * mm_per_px_x
        bh_mm = bh * mm_per_px_y

        diag = PathDiagnostic(
            index=idx, kind="closed",
            n_skel_pixels=len(pts), n_polyline_pts=len(pts),
            closed=True,
            bbox_xywh=(int(x0), int(y0), int(bw), int(bh)),
            bbox_w_mm=bw_mm, bbox_h_mm=bh_mm,
            area_px=area_px, area_mm2=area_mm2,
            perimeter_px=perim_px, compactness=compactness,
            outcome="kept",
        )

        if area_mm2 < cfg["min_area_mm2"]:
            diag.outcome = "dropped_small_area"
            diag.drop_reason = f"area_mm2={area_mm2:.1f} < {cfg['min_area_mm2']}"
            report.paths.append(diag)
            continue
        if compactness < cfg["min_compactness"]:
            diag.outcome = "dropped_slither"
            diag.drop_reason = f"compactness={compactness:.3f} < {cfg['min_compactness']}"
            report.paths.append(diag)
            continue

        # RDP simplify; SKIP Chaikin (was pre-rounding sharp corners).
        # Use angle-aware Bezier: sharp angles -> straight, smooth -> curve.
        contour = pts[:-1].astype(np.int32).reshape(-1, 1, 2)
        simp = _rdp_simplify(contour, cfg["rdp_epsilon"])
        if len(simp) < 3:
            diag.outcome = "dropped_rdp_too_few"
            report.paths.append(diag)
            continue
        sharp_thr = cfg.get("sharp_angle_threshold_deg", 150.0)
        segs = _angle_aware_bezier(simp, sharp_angle_threshold_deg=sharp_thr,
                                  tension=1.0)

        diag.n_polyline_pts = len(simp)
        report.paths.append(diag)
        paths.append(VectorPath(
            points=simp, bezier_segments=segs,
            area_px=area_px, perimeter_px=perim_px,
            bbox_xywh=(int(x0), int(y0), int(bw), int(bh)),
        ))

    paths.sort(key=lambda p: p.area_px, reverse=True)
    report.final_paths = len(paths)

    if save_stages_dir is not None:
        (save_stages_dir / f"{case_stem}_report.json").write_text(
            json.dumps(report.to_json(), indent=2), encoding="utf-8")

    return paths, report, cfg
