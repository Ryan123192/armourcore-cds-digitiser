"""Phase 3 isolated-tool gap-fill — multi-method comparison.

Runs several gap-fill strategies on a single isolated tool image and
writes a separate highlight per method so you can compare visually.

Methods
-------
A. close25     — MORPH_CLOSE k=25, then medial axis (current baseline)
B. close50     — same with k=50
C. close75     — same with k=75
D. close100    — same with k=100
E. iterative   — CLOSE k=15 → k=30 → k=50, each pass only adds new pixels
F. hybrid      — CLOSE k=25 + skeleton-endpoint ray-cast Bezier bridges
                 (catches medium-long gaps that CLOSE leaves open)
G. local_close — CLOSE k=25, then apply CLOSE k=80 in a local box around
                 each remaining skeleton endpoint (no global distortion)

Usage:
    python tools/test_phase3_iso_compare.py IsoToolUnfixed01
    python tools/test_phase3_iso_compare.py IsoToolUnfixed01 --no-open

Outputs (per method) to:
    data/outputs/phase03_testing/<name>_compare/<method>_highlight.png
"""
from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np
from skimage.morphology import medial_axis

from armourcore_cds.utils.image_ops import save_image


# =====================================================================
# Shared utilities
# =====================================================================

def _threshold(cleaned_bgr: np.ndarray, thr: int) -> np.ndarray:
    gray = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
    return ((gray < thr).astype(np.uint8)) * 255


def _avg_trace_thickness(trace_binary: np.ndarray) -> float:
    """Median diameter (= 2 * distance) along the medial axis of trace."""
    skel_bool, dist = medial_axis(trace_binary > 0, return_distance=True)
    th = dist[skel_bool] * 2.0
    th = th[th >= 2.0]
    if th.size == 0:
        return 4.0
    return float(np.clip(np.median(th), 2.0, 30.0))


def _render_skel_in_gaps_as_disks(
    skel: np.ndarray,
    trace_binary: np.ndarray,
    radius_px: int,
) -> np.ndarray:
    """Take skeleton pixels NOT in original trace; stamp a filled circle
    of given radius at each one — produces the gap-fill mask."""
    gap_skel = cv2.bitwise_and(skel, cv2.bitwise_not(trace_binary))
    out = np.zeros_like(trace_binary)
    ys, xs = np.where(gap_skel > 0)
    r = max(1, int(radius_px))
    for x, y in zip(xs.tolist(), ys.tolist()):
        cv2.circle(out, (int(x), int(y)), r, 255, -1)
    return cv2.bitwise_and(out, cv2.bitwise_not(trace_binary))


def _highlight(
    cleaned_bgr: np.ndarray,
    trace_binary: np.ndarray,
    skel: np.ndarray,
    completion: np.ndarray,
) -> np.ndarray:
    """Three-tone diagnostic overlay."""
    out = cleaned_bgr.copy()
    orig_px = trace_binary > 0
    out[orig_px] = (out[orig_px].astype(np.int32) * 0.55).clip(0, 255).astype(np.uint8)
    skel_only = (skel > 0) & (completion == 0)
    out[skel_only] = (230, 80, 0)
    out[completion > 0] = (0, 220, 60)
    return out


def _skel_endpoints(skel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (xs, ys) of skeleton pixels with exactly one 8-neighbour."""
    skel_u8 = (skel > 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    kernel[1, 1] = 0
    nb = cv2.filter2D(skel_u8, cv2.CV_8U, kernel)
    ep_y, ep_x = np.where((skel_u8 > 0) & (nb == 1))
    return ep_x, ep_y


def _walk_back(
    skel: np.ndarray, sx: int, sy: int, n_steps: int
) -> list[tuple[int, int]]:
    H, W = skel.shape
    skel_bool = skel > 0
    path = [(sx, sy)]
    visited = {(sx, sy)}
    cx, cy = sx, sy
    for _ in range(n_steps):
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


# =====================================================================
# Methods
# =====================================================================

def method_close(trace_binary: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single MORPH_CLOSE + medial axis."""
    k = max(3, k | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(trace_binary, cv2.MORPH_CLOSE, kernel)
    skel_bool, _ = medial_axis(closed > 0, return_distance=True)
    skel = skel_bool.astype(np.uint8) * 255

    radius = max(1, int(round(_avg_trace_thickness(trace_binary) / 2.0)))
    completion = _render_skel_in_gaps_as_disks(skel, trace_binary, radius)
    return completion, skel, closed


def method_iterative(
    trace_binary: np.ndarray, kernels: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply CLOSE with progressively larger kernels.  At each step we
    only ADD pixels that weren't there yet, preventing the largest kernel
    from over-distorting the small details."""
    closed = trace_binary.copy()
    for k in kernels:
        k = max(3, k | 1)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        new_closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kern)
        # Only allow pixels that the CLOSE adds — never drop pixels we already had
        closed = cv2.bitwise_or(closed, new_closed)
    skel_bool, _ = medial_axis(closed > 0, return_distance=True)
    skel = skel_bool.astype(np.uint8) * 255

    radius = max(1, int(round(_avg_trace_thickness(trace_binary) / 2.0)))
    completion = _render_skel_in_gaps_as_disks(skel, trace_binary, radius)
    return completion, skel, closed


def method_hybrid(
    trace_binary: np.ndarray,
    base_k: int = 25,
    bridge_max_dist_px: int = 200,
    lookback_px: int = 15,
    facing_threshold: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CLOSE for tiny gaps, then fill medium-long gaps via skeleton
    endpoint ray-casting + Bezier bridges.

    After CLOSE, any skeleton endpoints (degree-1 nodes) indicate gaps
    that survived.  We pair facing endpoints and connect them with a
    smooth curve.
    """
    # Step 1: initial CLOSE + medial axis
    completion, skel, closed = method_close(trace_binary, base_k)

    # Step 2: find skeleton endpoints (= unfilled gaps)
    ep_x, ep_y = _skel_endpoints(skel)
    if len(ep_x) < 2:
        return completion, skel, closed

    # Step 3: per-endpoint tangent direction
    descriptors = []
    for ex, ey in zip(ep_x.tolist(), ep_y.tolist()):
        path = _walk_back(skel, int(ex), int(ey), lookback_px)
        if len(path) < 4:
            continue
        d = np.array(
            [path[0][0] - path[-1][0], path[0][1] - path[-1][1]],
            dtype=np.float64,
        )
        n = float(np.linalg.norm(d))
        if n < 1e-9:
            continue
        descriptors.append({
            "pos": np.array([float(ex), float(ey)]),
            "dir": d / n,
        })
    if len(descriptors) < 2:
        return completion, skel, closed

    # Step 4: bidirectional ray-cast pairing.  Each endpoint's ray must
    # hit empty space then trace; partners must mutually face each other.
    n = len(descriptors)
    candidates = []
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

    # Step 5: render Bezier bridges
    avg_thick = _avg_trace_thickness(trace_binary)
    bridge_thickness = max(2, int(round(avg_thick)))
    H, W = trace_binary.shape
    bridge_canvas = np.zeros_like(trace_binary)

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
        cv2.polylines(
            bridge_canvas, [curve], False, 255, bridge_thickness, cv2.LINE_AA,
        )

    bridge_canvas = cv2.bitwise_and(bridge_canvas, cv2.bitwise_not(trace_binary))
    completion = cv2.bitwise_or(completion, bridge_canvas)
    return completion, skel, closed


def method_local_close(
    trace_binary: np.ndarray,
    base_k: int = 25,
    local_k: int = 80,
    box_pad_px: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CLOSE k=base globally, then for each remaining skeleton endpoint,
    apply a much larger CLOSE within a local bounding box around it.
    Avoids global distortion from a big kernel everywhere."""
    completion, skel, closed = method_close(trace_binary, base_k)

    ep_x, ep_y = _skel_endpoints(skel)
    if len(ep_x) == 0:
        return completion, skel, closed

    H, W = trace_binary.shape
    big_k = max(3, local_k | 1)
    big_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (big_k, big_k))

    closed_global = closed.copy()
    for ex, ey in zip(ep_x.tolist(), ep_y.tolist()):
        x1 = max(0, int(ex) - box_pad_px)
        y1 = max(0, int(ey) - box_pad_px)
        x2 = min(W, int(ex) + box_pad_px + 1)
        y2 = min(H, int(ey) + box_pad_px + 1)
        sub = closed_global[y1:y2, x1:x2]
        sub_closed = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, big_kernel)
        closed_global[y1:y2, x1:x2] = cv2.bitwise_or(sub, sub_closed)

    skel_bool, _ = medial_axis(closed_global > 0, return_distance=True)
    skel_new = skel_bool.astype(np.uint8) * 255
    radius = max(1, int(round(_avg_trace_thickness(trace_binary) / 2.0)))
    completion = _render_skel_in_gaps_as_disks(skel_new, trace_binary, radius)
    return completion, skel_new, closed_global


# =====================================================================
# Main
# =====================================================================

METHODS = [
    ("A_close25",     lambda t: method_close(t, 25)),
    ("B_close50",     lambda t: method_close(t, 50)),
    ("C_close75",     lambda t: method_close(t, 75)),
    ("D_close100",    lambda t: method_close(t, 100)),
    ("E_iterative",   lambda t: method_iterative(t, [15, 30, 50])),
    ("F_hybrid",      lambda t: method_hybrid(t, base_k=25, bridge_max_dist_px=200)),
    ("G_local_close", lambda t: method_local_close(t, base_k=25, local_k=80)),
]


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
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    in_path = repo_root / "data" / "inputs" / "phase03_testing" / f"{args.name}.png"
    out_dir = repo_root / "data" / "outputs" / "phase03_testing" / f"{args.name}_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"ERROR: {in_path} not found")
        sys.exit(1)

    cleaned_bgr = cv2.imread(str(in_path))
    H, W = cleaned_bgr.shape[:2]
    print(f"\n=== {args.name} ({W}x{H}) ===\n")

    trace = _threshold(cleaned_bgr, args.threshold)
    n_trace = int(np.count_nonzero(trace))
    avg_t = _avg_trace_thickness(trace)
    print(f"Trace pixels: {n_trace:,}   Avg thickness: {avg_t:.1f}px\n")

    for name, fn in METHODS:
        print(f"-- {name} --")
        completion, skel, closed = fn(trace)
        n_comp = int(np.count_nonzero(completion))
        ep_x, ep_y = _skel_endpoints(skel)
        print(f"   completion={n_comp:,}px   "
              f"remaining skel endpoints={len(ep_x)}")

        highlight = _highlight(cleaned_bgr, trace, skel, completion)
        out_path = out_dir / f"{name}_highlight.png"
        save_image(out_path, highlight)
        print(f"   -> {out_path.name}")

    print(f"\nAll outputs in: {out_dir}")
    if not args.no_open:
        _open_file(out_dir)


if __name__ == "__main__":
    main()
