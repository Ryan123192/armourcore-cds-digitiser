"""Step 02 — Batch compare of grid-removal methods on all 12 cases.

User observation (after v2 run): v2 traded grid removal for tool-ink
preservation.  Going forward we need BOTH.  This batch tests several
combinations on all 12 corpus cases so we can pick the best.

Methods tested
==============

    M0_L11_baseline
        Production L11 (HSV chroma + hard clamp @ 160).  Reference.

    M1_lab_then_L11clamp
        Replace HSV with Lab Delta-a* mask (lighting-invariant), then
        run the rest of L11 unchanged -- paper-blend fill, paper-to-
        white, hard clamp at gray >= 160.

    M2_lab_strict_then_L11clamp
        Like M1 but adds a BRIGHTNESS FLOOR (Lab L >= 120) to the
        Lab mask.  Only catches LIGHT warm-shifted pixels (grid),
        not DARK warm-shifted pixels (pen ink with JPEG warmth).

    M3_lab_strict_plus_iterative
        Like M2 plus a SECOND Lab pass on the cleaned result -- any
        warm-shifted pixels that survived Stage 1 get wiped on
        Stage 2.

Outputs
=======

    data/outputs/pipeline_dev/step02_grid_removal/<stamp>_batch_compare/
        M0_L11_baseline/
            <stem>/
                cleaned.png
                trace_mask.png
                info.json
            summary_clean.png
        M1_lab_then_L11clamp/
            ...
        ...
        per_case_compare/
            <stem>.png   -- one image per case showing M0..M3 side-by-side

    python tools/pipeline_dev/step02_grid_removal/batch_compare.py
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

REPO = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

import cv2
import numpy as np

from armourcore_cds.phase1.marker_rectify_fast_v4 import (
    rectify_with_markers_fast_v4,
    PAPER_W_MM, PAPER_H_MM, DEFAULT_PX_PER_MM,
)
from armourcore_cds.phase2.trace_isolation import (
    isolate_trace_candidates, normalise_lighting, build_orange_mask,
    paper_blended_fill, normalise_paper_to_white, build_trace_mask,
)
from armourcore_cds.phase2.trace_isolation_v2 import (
    estimate_paper_lab, lab_delta_a_grid_mask,
)
from armourcore_cds.templates.registry import load_template_config
from tools.pipeline_dev.corpus import (
    discover_corpus, make_run_dir, label_image, grid_montage,
)


TEMPLATE_ID = "cds_colour_test_260x350"

# Per-image brightness floor (Lab L channel, 0-255 scale).  Pixels DARKER
# than this are NEVER classified as grid by the Lab mask, even if their
# a* shift exceeds the threshold.  Reason: tool ink can have small
# positive a* from JPEG warmth but is much darker than any grid line.
LAB_L_MIN_FOR_GRID = 120

# Hard clamp threshold — same as L11 production
WHITE_CLAMP_GRAY = 160


# ===========================================================================
# Helper variants
# ===========================================================================

def _lab_strict_mask(
    image_bgr: np.ndarray,
    paper_lab: tuple,
    *,
    delta_a_min: int = 6,
    orange_tolerance: int = 8,
    l_min: int | None = None,
) -> np.ndarray:
    """Lab Delta-a* mask with optional L brightness floor.

    When ``l_min`` is set, only pixels with Lab L >= l_min can be flagged
    as grid.  Excludes dark ink that has slight JPEG warmth.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
    L = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    _, ap, bp = paper_lab
    delta_a = a - ap
    delta_b = b - bp
    mask = (
        (delta_a >= delta_a_min)
        & ((delta_b - delta_a) <= orange_tolerance)
    )
    if l_min is not None:
        mask = mask & (L >= l_min)
    return mask.astype(np.uint8) * 255


def _apply_hard_clamp(image_bgr: np.ndarray,
                      threshold: int = WHITE_CLAMP_GRAY) -> np.ndarray:
    """L11's hard-white clamp: any pixel with gray >= threshold -> pure white."""
    out = image_bgr.copy()
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    out[gray >= threshold] = (255, 255, 255)
    return out


# ===========================================================================
# Method implementations
# ===========================================================================

def method_M0_L11_baseline(rect_bgr: np.ndarray, template) -> np.ndarray:
    """Production L11."""
    result = isolate_trace_candidates(rect_bgr, template)
    return result.cleaned_bgr


def method_M1_lab_then_L11clamp(rect_bgr: np.ndarray, template) -> np.ndarray:
    """Lab Δa* mask + L11's downstream stages (paper-blend, paper-to-white, hard clamp)."""
    norm = normalise_lighting(rect_bgr)
    paper_lab = estimate_paper_lab(cv2.cvtColor(norm, cv2.COLOR_BGR2LAB))
    grid_mask = _lab_strict_mask(norm, paper_lab, delta_a_min=6)
    filled = paper_blended_fill(norm, grid_mask)
    lifted = normalise_paper_to_white(filled)
    return _apply_hard_clamp(lifted, WHITE_CLAMP_GRAY)


def method_M2_lab_strict_then_L11clamp(
    rect_bgr: np.ndarray, template,
) -> np.ndarray:
    """Lab mask with brightness floor + L11 clamp.

    Brightness floor protects dark ink (pen / pencil) from being
    classified as grid even when JPEG warmth shifts their a* slightly.
    """
    norm = normalise_lighting(rect_bgr)
    paper_lab = estimate_paper_lab(cv2.cvtColor(norm, cv2.COLOR_BGR2LAB))
    grid_mask = _lab_strict_mask(
        norm, paper_lab,
        delta_a_min=6,
        l_min=LAB_L_MIN_FOR_GRID,
    )
    filled = paper_blended_fill(norm, grid_mask)
    lifted = normalise_paper_to_white(filled)
    return _apply_hard_clamp(lifted, WHITE_CLAMP_GRAY)


def method_M3_lab_strict_plus_iterative(
    rect_bgr: np.ndarray, template,
) -> np.ndarray:
    """M2 + a second Lab pass that catches any warm pixel surviving Stage 1."""
    norm = normalise_lighting(rect_bgr)
    paper_lab = estimate_paper_lab(cv2.cvtColor(norm, cv2.COLOR_BGR2LAB))

    # Stage 1 — strict Lab + brightness floor
    grid_mask = _lab_strict_mask(
        norm, paper_lab,
        delta_a_min=6,
        l_min=LAB_L_MIN_FOR_GRID,
    )
    filled = paper_blended_fill(norm, grid_mask)
    lifted = normalise_paper_to_white(filled)
    cleaned = _apply_hard_clamp(lifted, WHITE_CLAMP_GRAY)

    # Stage 2 — iterative Lab cleanup on the cleaned output.  Any pixel
    # that is STILL warm-shifted from paper (now-clean paper baseline)
    # is residual grid we missed.  Re-estimate the baseline because the
    # paper has been lifted.
    cleaned_lab = cv2.cvtColor(cleaned, cv2.COLOR_BGR2LAB)
    paper_lab2 = estimate_paper_lab(cleaned_lab)
    residual_mask = _lab_strict_mask(
        cleaned, paper_lab2,
        delta_a_min=4,                      # slightly looser on second pass
        l_min=LAB_L_MIN_FOR_GRID,
    )
    if np.any(residual_mask):
        cleaned[residual_mask > 0] = (255, 255, 255)
    return cleaned


METHODS: list[tuple[str, Callable]] = [
    ("M0_L11_baseline",                method_M0_L11_baseline),
    ("M1_lab_then_L11clamp",           method_M1_lab_then_L11clamp),
    ("M2_lab_strict_then_L11clamp",    method_M2_lab_strict_then_L11clamp),
    ("M3_lab_strict_plus_iterative",   method_M3_lab_strict_plus_iterative),
]


# ===========================================================================
# Runner
# ===========================================================================

def _make_case_compare_tile(
    rectified: np.ndarray,
    per_method_cleaned: dict[str, np.ndarray],
    case_stem: str,
) -> np.ndarray:
    """Tile rectified + each method's cleaned output side-by-side."""
    tiles = [label_image(rectified, "rectified", height_px=48, scale=0.9)]
    for name, cleaned in per_method_cleaned.items():
        tiles.append(
            label_image(cleaned, name, height_px=48, scale=0.9)
        )
    h_max = max(t.shape[0] for t in tiles)

    def _pad(img):
        if img.shape[0] == h_max:
            return img
        pad = np.full((h_max - img.shape[0], img.shape[1], 3),
                      240, dtype=np.uint8)
        return np.vstack([img, pad])

    row = np.hstack([_pad(t) for t in tiles])
    header = np.full((52, row.shape[1], 3), 25, dtype=np.uint8)
    cv2.putText(
        header, case_stem, (16, 36),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return np.vstack([header, row])


def main() -> None:
    corpus = discover_corpus()
    if not corpus:
        print("No corpus files found.")
        sys.exit(1)

    run_dir = make_run_dir("step02_grid_removal", "batch_compare")
    print(f"Run dir: {run_dir.relative_to(REPO)}\n")

    template = load_template_config(TEMPLATE_ID)

    # Per-method summary tiles (will be assembled at the end)
    per_method_summary_tiles: dict[str, list[np.ndarray]] = {
        name: [] for name, _ in METHODS
    }
    per_case_dir = run_dir / "per_case_compare"
    per_case_dir.mkdir(exist_ok=True)

    print(f"{'case':<28s}  " + "  ".join(
        f"{name[:13]:>13s}" for name, _ in METHODS) + "  status")
    print("-" * 120)

    summary_payload: list[dict] = []

    for case in corpus:
        case_info: dict = {"stem": case.stem}
        t_total = time.time()

        # ---- Phase 1: rectify once, reuse for every method ----
        try:
            rect_result = rectify_with_markers_fast_v4(
                cv2.imread(str(case.path)),
                paper_w_mm=PAPER_W_MM, paper_h_mm=PAPER_H_MM,
                px_per_mm=DEFAULT_PX_PER_MM,
            )
            rectified = rect_result.warped
        except Exception as exc:
            print(f"{case.stem:<28s}  Phase 1 failed: {exc}")
            case_info["phase1_status"] = "failed"
            case_info["phase1_error"] = str(exc)
            summary_payload.append(case_info)
            continue

        case_info["phase1_status"] = "ok"
        per_method_cleaned: dict[str, np.ndarray] = {}
        per_method_timings: dict[str, float] = {}

        # ---- Each method on the same rectified input ----
        for name, fn in METHODS:
            method_dir = run_dir / name / case.stem
            method_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            try:
                cleaned = fn(rectified, template)
                per_method_cleaned[name] = cleaned
                elapsed = time.time() - t0
                per_method_timings[name] = round(elapsed, 3)
                cv2.imwrite(str(method_dir / "cleaned.png"), cleaned)
                # Trace mask for diagnostics
                trace = build_trace_mask(cleaned)
                cv2.imwrite(str(method_dir / "trace_mask.png"), trace)
                (method_dir / "info.json").write_text(json.dumps({
                    "stem": case.stem,
                    "method": name,
                    "elapsed_s": per_method_timings[name],
                    "trace_mask_pct": round(
                        100.0 * (trace > 0).sum() / (trace.size or 1), 3,
                    ),
                }, indent=2), encoding="utf-8")
                # Add this method's cleaned to its summary
                per_method_summary_tiles[name].append(
                    label_image(cleaned, case.stem, height_px=48, scale=0.85)
                )
            except Exception as exc:
                per_method_cleaned[name] = np.full(
                    rectified.shape, 200, dtype=np.uint8,
                )
                per_method_timings[name] = float("nan")
                (method_dir / "info.json").write_text(json.dumps({
                    "stem": case.stem,
                    "method": name,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }, indent=2), encoding="utf-8")

        # ---- Per-case comparison tile (rectified + all 4 methods) ----
        case_tile = _make_case_compare_tile(
            rectified, per_method_cleaned, case.stem,
        )
        cv2.imwrite(str(per_case_dir / f"{case.stem}.png"), case_tile)

        case_info["timings"] = per_method_timings
        case_info["total_elapsed_s"] = round(time.time() - t_total, 3)
        summary_payload.append(case_info)

        line = f"{case.stem:<28s}"
        for name, _ in METHODS:
            line += f"  {per_method_timings[name]:>11.2f}s"
        line += "  ok"
        print(line)

    # ---- Per-method summary grids ----
    for name, tiles in per_method_summary_tiles.items():
        if not tiles:
            continue
        summary = grid_montage(tiles, cols=4, tile_max_dim=600)
        cv2.imwrite(str(run_dir / f"summary_{name}.png"), summary)

    # ---- Top-level summary JSON ----
    (run_dir / "summary.json").write_text(json.dumps({
        "step": "step02_grid_removal",
        "label": "batch_compare",
        "template_id": TEMPLATE_ID,
        "methods": [name for name, _ in METHODS],
        "lab_l_min_for_grid": LAB_L_MIN_FOR_GRID,
        "white_clamp_gray": WHITE_CLAMP_GRAY,
        "n_cases": len(corpus),
        "cases": summary_payload,
    }, indent=2), encoding="utf-8")

    print(f"\nFolder: {run_dir}")
    print("View these grids to compare methods at a glance:")
    for name, _ in METHODS:
        print(f"  summary_{name}.png")
    print("View per-case comparisons (rectified + all 4 methods):")
    print(f"  per_case_compare/<stem>.png")


if __name__ == "__main__":
    main()
