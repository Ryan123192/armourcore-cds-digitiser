"""Rewrites _snap_to_outer_edge in boundary_detection.py with the averaged-profile approach."""
import pathlib

NEW_FUNC = (
    "def _snap_to_outer_edge(\n"
    "    quad: np.ndarray,\n"
    "    gray: np.ndarray,\n"
    "    *,\n"
    "    max_scan_px: int = 220,\n"
    "    dark_threshold: int = 130,\n"
    "    min_snap_px: int = 55,\n"
    "    thin_band_threshold: int = 20,\n"
    "    min_consistency: float = 0.60,\n"
    "    num_sample_pts: int = 11,\n"
    ") -> np.ndarray:\n"
    '    """Snap each side that sits on a thin interior line outward to the nearest\n'
    "    thick dark band (the actual outer border).\n"
    "\n"
    "    Algorithm per side\n"
    "    ------------------\n"
    "    1. Measure the dark-band width at the current detected position.\n"
    "       If already thick (>= thin_band_threshold px) the side is assumed to be\n"
    "       on the outer border -- skip it.  thin_band_threshold=20 keeps ~12 px\n"
    "       interior grid lines in the 'thin' category while outer borders\n"
    "       (typically >= 25 px at phone-camera resolution) are classified thick.\n"
    "    2. Otherwise (thin band = likely a grid line), build an *averaged* outward\n"
    "       darkness profile: for each of num_sample_pts positions along the side\n"
    "       scan outward up to max_scan_px pixels (never past the image boundary)\n"
    "       and accumulate a per-distance darkness fraction.\n"
    "    3. Search the averaged profile (starting at min_snap_px) for the first\n"
    "       CONTIGUOUS dark region with avg_dark >= min_consistency (default 0.60)\n"
    "       that spans >= 2 pixels AND ends clearly before the scan boundary.\n"
    "       A region that reaches the scan boundary is treated as continuous\n"
    "       background (e.g. a dark table beyond the sheet edge) and the snap is\n"
    "       suppressed.  Features in the last 15 % of the scan range are also\n"
    "       rejected as edge artefacts.\n"
    '    """\n'
    "    h, w = gray.shape[:2]\n"
    "    cx = float(np.mean(quad[:, 0]))\n"
    "    cy = float(np.mean(quad[:, 1]))\n"
    "\n"
    "    def _max_outward_dist(p: np.ndarray, nrm: np.ndarray) -> int:\n"
    '        """Pixels we can travel outward from p before leaving the image."""\n'
    "        limits: list[float] = []\n"
    "        if nrm[0] > 1e-6:\n"
    "            limits.append((w - 1 - float(p[0])) / nrm[0])\n"
    "        elif nrm[0] < -1e-6:\n"
    "            limits.append(float(p[0]) / (-nrm[0]))\n"
    "        if nrm[1] > 1e-6:\n"
    "            limits.append((h - 1 - float(p[1])) / nrm[1])\n"
    "        elif nrm[1] < -1e-6:\n"
    "            limits.append(float(p[1]) / (-nrm[1]))\n"
    "        return int(min(limits)) if limits else max_scan_px\n"
    "\n"
    "    side_offsets: list[float] = []\n"
    "\n"
    "    for i in range(4):\n"
    "        pa = quad[i].copy()\n"
    "        pb = quad[(i + 1) % 4].copy()\n"
    "        side_vec = pb - pa\n"
    "        side_len = float(np.linalg.norm(side_vec))\n"
    "        if side_len < 1e-6:\n"
    "            side_offsets.append(0.0)\n"
    "            continue\n"
    "\n"
    "        side_unit = side_vec / side_len\n"
    "        # Outward normal (away from centroid)\n"
    "        normal = np.array([-side_unit[1], side_unit[0]], dtype=np.float32)\n"
    "        mid = (pa + pb) / 2.0\n"
    "        if float(np.dot(normal, np.array([cx - mid[0], cy - mid[1]]))) > 0:\n"
    "            normal = -normal  # flip so normal points outward\n"
    "\n"
    "        # How far can we go outward before leaving the image?\n"
    "        mid_safe = _max_outward_dist(mid, normal)\n"
    "        if mid_safe < min_snap_px:\n"
    "            side_offsets.append(0.0)\n"
    "            continue\n"
    "\n"
    "        # --- Step 1: current band width check ---\n"
    "        bws = [\n"
    "            _current_band_width(gray, pa + a * side_vec, normal,\n"
    "                                dark_threshold=dark_threshold)\n"
    "            for a in np.linspace(0.15, 0.85, num_sample_pts)\n"
    "        ]\n"
    "        if float(np.median(bws)) >= thin_band_threshold:\n"
    "            # Already at the thick outer border -- no snap needed\n"
    "            side_offsets.append(0.0)\n"
    "            continue\n"
    "\n"
    "        # --- Step 2: build averaged outward darkness profile ---\n"
    "        # Scan is limited to image bounds so np.clip() cannot create phantom\n"
    "        # dark bands at image edge rows/columns.\n"
    "        effective_max = min(max_scan_px, mid_safe)\n"
    "        profile = np.zeros(effective_max + 1, dtype=np.float32)\n"
    "        n_valid = 0\n"
    "\n"
    "        for alpha in np.linspace(0.10, 0.90, num_sample_pts):\n"
    "            p = pa + alpha * side_vec\n"
    "            eff = min(effective_max, _max_outward_dist(p, normal))\n"
    "            if eff < min_snap_px:\n"
    "                continue\n"
    "            sr = np.arange(0, eff + 1, 1)\n"
    "            pxs = np.clip(np.round(p[0] + normal[0] * sr).astype(int), 0, w - 1)\n"
    "            pys = np.clip(np.round(p[1] + normal[1] * sr).astype(int), 0, h - 1)\n"
    "            profile[:len(sr)] += (gray[pys, pxs] < dark_threshold).astype(np.float32)\n"
    "            n_valid += 1\n"
    "\n"
    "        if n_valid == 0:\n"
    "            side_offsets.append(0.0)\n"
    "            continue\n"
    "\n"
    "        avg = profile / n_valid\n"
    "\n"
    "        # --- Step 3: find first consistent narrow dark band ---\n"
    "        # Accept the first region where avg_dark >= min_consistency for >= 2\n"
    "        # consecutive pixels, provided the region ENDS within the scan range\n"
    "        # (background/table produces a region that continues to the scan edge)\n"
    "        # and starts within the first 85 % of the scan range.\n"
    "        snap_dist = -1\n"
    "        in_region = False\n"
    "        region_start = -1\n"
    "        region_width = 0\n"
    "\n"
    "        for d in range(min_snap_px, effective_max + 1):\n"
    "            if avg[d] >= min_consistency:\n"
    "                if not in_region:\n"
    "                    in_region = True\n"
    "                    region_start = d\n"
    "                    region_width = 1\n"
    "                else:\n"
    "                    region_width += 1\n"
    "            else:\n"
    "                if in_region and region_width >= 2:\n"
    "                    if region_start <= effective_max * 0.85:\n"
    "                        snap_dist = region_start\n"
    "                    break  # stop at first qualifying (or rejected) region\n"
    "                in_region = False\n"
    "                region_width = 0\n"
    "        # If region extends to scan limit: background noise -- do NOT snap\n"
    "\n"
    "        side_offsets.append(float(snap_dist) if snap_dist >= min_snap_px else 0.0)\n"
    "\n"
    "    # Rebuild quad: shift each side by its offset, intersect adjacent pairs\n"
    "    moved_sides: list[tuple[np.ndarray, np.ndarray]] = []\n"
    "\n"
    "    for i in range(4):\n"
    "        pa = quad[i].copy()\n"
    "        pb = quad[(i + 1) % 4].copy()\n"
    "        side_vec = pb - pa\n"
    "        side_len = float(np.linalg.norm(side_vec))\n"
    "        side_unit = side_vec / max(side_len, 1e-6)\n"
    "        normal = np.array([-side_unit[1], side_unit[0]], dtype=np.float32)\n"
    "        mid = (pa + pb) / 2.0\n"
    "        if float(np.dot(normal, np.array([cx - mid[0], cy - mid[1]]))) > 0:\n"
    "            normal = -normal\n"
    "        off = side_offsets[i]\n"
    "        moved_sides.append((pa + normal * off, pb + normal * off))\n"
    "\n"
    "    refined_corners: list[np.ndarray] = []\n"
    "    for i in range(4):\n"
    "        prev_a, prev_b = moved_sides[(i - 1) % 4]\n"
    "        curr_a, curr_b = moved_sides[i]\n"
    "        pt = _line_intersect(prev_a, prev_b, curr_a, curr_b)\n"
    "        if pt is None:\n"
    "            pt = (prev_b + curr_a) / 2.0\n"
    "        refined_corners.append(pt)\n"
    "\n"
    "    refined = np.array(refined_corners, dtype=np.float32)\n"
    "\n"
    "    orig_area = _quad_area(quad)\n"
    "    refined_area = _quad_area(refined)\n"
    "    if orig_area > 0 and not (0.85 <= refined_area / orig_area <= 1.50):\n"
    "        return quad  # reject if snap changed geometry too drastically\n"
    "\n"
    "    refined[:, 0] = np.clip(refined[:, 0], -60, w + 60)\n"
    "    refined[:, 1] = np.clip(refined[:, 1], -60, h + 60)\n"
    "    return _order_quad_points(refined)\n"
)

path = pathlib.Path('src/armourcore_cds/phase1/boundary_detection.py')
content = path.read_text(encoding='utf-8')
lines = content.split('\n')

start_idx = None
end_idx = None
for i, line in enumerate(lines):
    if line.startswith('def _snap_to_outer_edge('):
        start_idx = i
    if start_idx is not None and i > start_idx and line.startswith('def ') and not line.startswith('def _snap_to_outer_edge('):
        end_idx = i
        break

print(f"Replacing lines {start_idx+1} to {end_idx} (0-indexed {start_idx} to {end_idx-1})")
print(f"Old function was {end_idx - start_idx} lines")

separator = (
    "\n\n"
    "# ---------------------------------------------------------------------------\n"
    "# Main scorer\n"
    "# ---------------------------------------------------------------------------\n"
    "\n"
)

before = '\n'.join(lines[:start_idx])
after_func_start = end_idx  # index of 'def _score_candidate(...)'
# The separator lines are end_idx-4 to end_idx-1 in 0-indexed
after = '\n'.join(lines[after_func_start:])

new_content = before + '\n' + NEW_FUNC + separator + after
path.write_text(new_content, encoding='utf-8')
print("Done.")

# Verify
content2 = path.read_text(encoding='utf-8')
lines2 = content2.split('\n')
for i, l in enumerate(lines2):
    if 'def _snap_to_outer_edge' in l:
        print(f"Function at line {i+1}")
    if 'def _score_candidate' in l:
        print(f"Next function at line {i+1}")
        break
print(f"Total lines: {len(lines2)}")
