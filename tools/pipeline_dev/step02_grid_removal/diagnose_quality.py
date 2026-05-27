"""Objective quality diagnostic for Phase 2 cleaned outputs.

Stops me from "looks better to me" self-assessment by computing
quantitative metrics + producing annotated overlays.

Metrics (lower = better unless noted)
=====================================
* grid_energy_db    -- FFT spectral energy at the 100-px grid frequency
                      (every grid pattern produces a spike here).  Pure
                      paper-only image: ~ -inf.  Heavily grid-laden: > 0.
* bg_dark_frac      -- fraction of "background" pixels (not within 20px
                      of a >= 200-px trace component) that are < 220 gray.
                      Pure paper: ~0%.  Dotty noise / grid residual: > 1%.
* noise_ratio       -- count of <100-px components / count of >=300-px
                      components.  Pure trace output: 0-1.  Noisy: > 5.
* big_components    (higher = better) number of >= 500-px ink components.
                      A clean output of 13 shapes -> ~13.
* closure_ratio     (higher = better) fraction of large contours whose
                      perimeter completes a closed loop (start ~ end).

Visuals
=======
For each cleaned.png we emit an `annotated.png` that shows:
* white background
* GREEN  -- large ink components (>= 500 px)        - the wins
* ORANGE -- mid components 100..500 px              - probably trace fragments
* RED    -- tiny components < 100 px                - noise
* MAGENTA -- pixels at the FFT-detected grid centres - residual grid
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

REPO = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

import cv2
import numpy as np

from armourcore_cds.phase1.marker_rectify_fast_v4 import (
    PAPER_W_MM, PAPER_H_MM,
)
from tools.pipeline_dev.corpus import make_run_dir


GRID_SPACING_MM = 10.0   # minor grid


@dataclass
class QualityMetrics:
    case: str
    method: str
    grid_energy_db: float
    bg_dark_frac_pct: float
    noise_ratio: float
    big_components: int
    closure_ratio_pct: float
    total_ink_pct: float

    def composite_score(self) -> float:
        """Single number.  Higher = better.

        Calibrated against user judgement (baseline best for pencil_flat,
        tmpl_grid_dilated best for pen cases, never reward 80+ components
        which signals grid dashes being counted as ink).
        """
        # Component-count goodness: peak at ~13 (expected shapes), penalise
        # both too few (missing traces) and too many (grid dashes counted).
        ideal_big = 13.0
        comp_score = -abs(self.big_components - ideal_big) * 1.5

        # Grid pattern is the killer.  Use exponential penalty above 10 dB.
        grid_penalty = max(0.0, self.grid_energy_db - 10) ** 1.5

        # Background cleanliness is critical.
        bg_penalty = self.bg_dark_frac_pct * 8.0

        # Noise: cap at 30 so it can't dominate.
        noise_penalty = min(self.noise_ratio, 30.0) * 0.4

        # Total-ink reward only if reasonable (1-6% is the trace zone).
        if 1.0 <= self.total_ink_pct <= 6.0:
            ink_bonus = 5.0
        else:
            ink_bonus = -abs(self.total_ink_pct - 3.0) * 1.5

        return comp_score - grid_penalty - bg_penalty - noise_penalty + ink_bonus


def _cleaned_to_binary(cleaned_bgr: np.ndarray) -> np.ndarray:
    """Return uint8 binary mask (255=ink, 0=paper) at full resolution."""
    gray = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
    # Generous threshold so we catch even faint pencil traces.
    return (gray < 200).astype(np.uint8) * 255


def measure_grid_energy_db(binary: np.ndarray,
                          spacing_px: float = 100.0,
                          tol_px: float = 6.0) -> float:
    """FFT-based estimate of how much periodic structure exists at the
    expected grid spacing.

    Approach:
      1. Compute 1-D row-sum and column-sum profiles.
      2. FFT both.
      3. Compute power at the frequency bin corresponding to spacing.
      4. Return ratio (in dB) of grid-frequency power to mean spectral power.
         A clean output has the spike absent (< -5 dB);
         a grid-laden output has it standing out (> 0 dB).
    """
    H, W = binary.shape
    row_proj = (binary > 0).sum(axis=1).astype(np.float32)
    col_proj = (binary > 0).sum(axis=0).astype(np.float32)
    row_proj -= row_proj.mean()
    col_proj -= col_proj.mean()
    fr = np.fft.rfft(row_proj)
    fc = np.fft.rfft(col_proj)
    pr = np.abs(fr) ** 2
    pc = np.abs(fc) ** 2
    target_freq_r = H / spacing_px
    target_freq_c = W / spacing_px
    tol_r = max(2, int(tol_px * H / spacing_px / spacing_px * 100))  # heuristic
    tol_r = 3
    tol_c = 3
    ir = int(round(target_freq_r))
    ic = int(round(target_freq_c))
    band_r = pr[max(0, ir - tol_r): ir + tol_r + 1].max() if ir < len(pr) else 0.0
    band_c = pc[max(0, ic - tol_c): ic + tol_c + 1].max() if ic < len(pc) else 0.0
    mean_r = pr.mean() + 1e-9
    mean_c = pc.mean() + 1e-9
    db_r = 10.0 * np.log10(band_r / mean_r + 1e-9)
    db_c = 10.0 * np.log10(band_c / mean_c + 1e-9)
    return float(max(db_r, db_c))


def measure_background_cleanliness(binary: np.ndarray,
                                  big_comp_thresh: int = 200,
                                  proximity_px: int = 20) -> float:
    """% of pixels that are 'background' (not near large components) but
    still classed as ink.  Lower = cleaner."""
    if not np.any(binary):
        return 0.0
    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    big = np.zeros_like(binary)
    for cid in range(1, n_lbl):
        if stats[cid, cv2.CC_STAT_AREA] >= big_comp_thresh:
            big[lbl == cid] = 255
    if not np.any(big):
        return 100.0 * (binary > 0).mean()  # everything is noise
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (proximity_px * 2 + 1, proximity_px * 2 + 1))
    big_neighbourhood = cv2.dilate(big, k, iterations=1)
    bg = big_neighbourhood == 0
    noise_in_bg = (binary > 0) & bg
    return 100.0 * float(noise_in_bg.sum()) / float(bg.sum() + 1e-9)


def measure_components(binary: np.ndarray) -> tuple[int, float, int, int]:
    """Return (big_components, noise_ratio, big_area_total, total_ink_pct)."""
    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    big = 0
    mid = 0
    small = 0
    for cid in range(1, n_lbl):
        a = stats[cid, cv2.CC_STAT_AREA]
        if a >= 500:
            big += 1
        elif a >= 100:
            mid += 1
        else:
            small += 1
    noise_ratio = float(small) / float(big + 1e-9)
    total_ink_pct = 100.0 * float((binary > 0).sum()) / float(binary.size)
    return big, noise_ratio, mid, total_ink_pct


def measure_closure(binary: np.ndarray) -> float:
    """% of large components whose bounding-box-perimeter to actual-perimeter
    ratio suggests a coherent closed shape (vs a fragmented line)."""
    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    coherent = 0
    big = 0
    for cid in range(1, n_lbl):
        a = stats[cid, cv2.CC_STAT_AREA]
        if a < 500:
            continue
        big += 1
        w = stats[cid, cv2.CC_STAT_WIDTH]
        h = stats[cid, cv2.CC_STAT_HEIGHT]
        bbox_area = w * h
        if bbox_area <= 0:
            continue
        # A coherent closed-shape outline fills only a small fraction of its
        # bounding box (perimeter, not interior).  Grid dashes fill > 30%.
        # Traces sit around 5-20%.
        fill_ratio = a / bbox_area
        if 0.03 <= fill_ratio <= 0.30:
            coherent += 1
    return 100.0 * coherent / max(big, 1)


def evaluate_case(cleaned_bgr: np.ndarray, case: str, method: str
                 ) -> QualityMetrics:
    binary = _cleaned_to_binary(cleaned_bgr)
    big, noise_ratio, _, total_pct = measure_components(binary)
    return QualityMetrics(
        case=case, method=method,
        grid_energy_db=measure_grid_energy_db(binary),
        bg_dark_frac_pct=measure_background_cleanliness(binary),
        noise_ratio=noise_ratio,
        big_components=big,
        closure_ratio_pct=measure_closure(binary),
        total_ink_pct=total_pct,
    )


def annotate(cleaned_bgr: np.ndarray) -> np.ndarray:
    """Annotated overlay: green=big, orange=mid, red=tiny, magenta=grid centres."""
    binary = _cleaned_to_binary(cleaned_bgr)
    H, W = binary.shape
    out = np.full((H, W, 3), 255, dtype=np.uint8)
    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    for cid in range(1, n_lbl):
        a = stats[cid, cv2.CC_STAT_AREA]
        mask = lbl == cid
        if a >= 500:
            colour = (40, 160, 40)   # green
        elif a >= 100:
            colour = (30, 130, 220)  # orange-ish (BGR)
        else:
            colour = (40, 40, 220)   # red
        out[mask] = colour
    # Magenta lines at template grid centres (1 px wide) for visual reference
    px_per_mm_x = W / PAPER_W_MM
    px_per_mm_y = H / PAPER_H_MM
    sx = GRID_SPACING_MM * px_per_mm_x
    sy = GRID_SPACING_MM * px_per_mm_y
    for i in range(int(W / sx) + 2):
        x = int(round(i * sx))
        if 0 <= x < W:
            cv2.line(out, (x, 0), (x, H - 1), (220, 80, 220), 1)
    for i in range(int(H / sy) + 2):
        y = int(round(i * sy))
        if 0 <= y < H:
            cv2.line(out, (0, y), (W - 1, y), (220, 80, 220), 1)
    return out


def main(run_folder: str):
    src = Path(run_folder)
    if not src.is_absolute():
        src = REPO / src
    if not src.exists():
        print(f"Folder not found: {src}")
        return

    diag_dir = make_run_dir("step02_grid_removal", "DIAG_" + src.name)
    print(f"Source: {src.relative_to(REPO)}")
    print(f"Diag:   {diag_dir.relative_to(REPO)}\n")

    methods = sorted([d for d in src.iterdir()
                      if d.is_dir() and d.name.startswith("M") and
                      d.name != "per_case_compare"])
    cases = sorted([p.parent.name for p in src.glob("*/*/cleaned.png")])
    cases = sorted(set(cases))

    rows: list[QualityMetrics] = []
    for method_dir in methods:
        method = method_dir.name
        for case in cases:
            cleaned_path = method_dir / case / "cleaned.png"
            if not cleaned_path.exists():
                continue
            cleaned = cv2.imread(str(cleaned_path))
            if cleaned is None:
                continue
            metrics = evaluate_case(cleaned, case, method)
            rows.append(metrics)
            ann = annotate(cleaned)
            out_dir = diag_dir / method / case
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / "annotated.png"), ann)

    # Print scorecard
    cases_in_order = sorted(set(r.case for r in rows))
    methods_in_order = sorted(set(r.method for r in rows))

    print("\n=== COMPOSITE SCORE per case (higher = better) ===")
    header = f"{'method':<32s}  " + "  ".join(f"{c[:16]:>16s}" for c in cases_in_order)
    print(header)
    print("-" * len(header))
    for m in methods_in_order:
        line = f"{m:<32s}  "
        for c in cases_in_order:
            r = next((r for r in rows if r.method == m and r.case == c), None)
            line += f"{r.composite_score():>16.1f}  " if r else f"{'-':>16s}  "
        print(line)

    print("\n=== DETAIL per (method,case) ===")
    print(
        f"{'method':<28s}  {'case':<22s}  "
        f"{'big':>4s}  {'closed%':>8s}  "
        f"{'noise/big':>10s}  {'gridE_dB':>10s}  {'bg_dark%':>10s}  "
        f"{'ink%':>8s}  {'score':>8s}"
    )
    print("-" * 130)
    for c in cases_in_order:
        for m in methods_in_order:
            r = next((r for r in rows if r.method == m and r.case == c), None)
            if r is None:
                continue
            print(
                f"{m:<28s}  {c:<22s}  "
                f"{r.big_components:>4d}  {r.closure_ratio_pct:>8.1f}  "
                f"{r.noise_ratio:>10.2f}  {r.grid_energy_db:>10.2f}  "
                f"{r.bg_dark_frac_pct:>10.3f}  {r.total_ink_pct:>8.3f}  "
                f"{r.composite_score():>8.1f}"
            )
        print("")

    # Save JSON
    (diag_dir / "metrics.json").write_text(
        json.dumps([asdict(r) for r in rows], indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote metrics.json + per-method annotated overlays to:\n  {diag_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default to latest v12 run
        runs = sorted((REPO / "data/outputs/pipeline_dev/step02_grid_removal")
                     .glob("*_flat_methods_v*"))
        if not runs:
            print("usage: diagnose_quality.py <run_folder>")
            sys.exit(1)
        target = str(runs[-1])
        print(f"No folder given; defaulting to latest: {target}\n")
        main(target)
    else:
        main(sys.argv[1])
