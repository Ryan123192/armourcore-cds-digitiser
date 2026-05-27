"""Step 02 — Grid removal: v2 (Lab Δa* + adaptive paper-relative clamp).

Same shape as ``run_baseline.py`` but calls
``isolate_trace_candidates_v2`` from
``src/armourcore_cds/phase2/trace_isolation_v2.py``.  Production L11
and its runner stay untouched for side-by-side comparison.

For each case writes:

    <stem>/
      rectified.png                -- Phase 1 output (input to Phase 2)
      stage0_lighting_norm.png     -- after CLAHE
      stage1_lab_mask_overlay.png  -- Lab Δa* mask overlaid red on input
      stage1b_after_blend.png      -- after paper-blend + paper-to-white
      stage2_clamp_mask.png        -- pixels the adaptive clamp wiped
      stage2_cleaned.png           -- FINAL cleaned image
      side_by_side.png             -- rectified | cleaned
      trace_mask.png               -- binary trace mask Phase 3 consumes
      info.json                    -- timings + mask coverage stats
                                      + the v2 parameters used

    summary_clean.png              -- 4 x 3 grid of all 12 cleaned outputs
    summary_side_by_side.png       -- rectified | cleaned pairs
    summary.json                   -- per-case stats

    python tools/pipeline_dev/step02_grid_removal/run_v2.py
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

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
from armourcore_cds.phase2.trace_isolation_v2 import (
    isolate_trace_candidates_v2,
    estimate_paper_lab, lab_delta_a_grid_mask, adaptive_paper_clamp,
    LAB_DELTA_A_MIN, LAB_ORANGE_TOLERANCE,
    ADAPTIVE_INK_GAP, ADAPTIVE_PAPER_SIGMA,
)
from armourcore_cds.phase2.trace_isolation import (
    normalise_lighting, paper_blended_fill, normalise_paper_to_white,
    build_trace_mask,
)
from armourcore_cds.templates.registry import load_template_config
from tools.pipeline_dev.corpus import (
    discover_corpus, make_run_dir, label_image, grid_montage,
)


TEMPLATE_ID = "cds_colour_test_260x350"


def _overlay_mask(base: np.ndarray, mask: np.ndarray,
                  colour=(0, 0, 255), alpha: float = 0.55) -> np.ndarray:
    out = base.copy()
    if not np.any(mask):
        return out
    paint = np.zeros_like(base)
    paint[:] = colour
    blend = cv2.addWeighted(out, 1.0 - alpha, paint, alpha, 0)
    return np.where(mask[..., None] > 0, blend, out)


def _stage_dump(rect_bgr: np.ndarray, out_dir: Path) -> dict:
    """Replay v2's stages with intermediate saves for diagnostics."""
    info: dict = {}

    # Step A — lighting normalisation
    norm = normalise_lighting(rect_bgr)
    cv2.imwrite(str(out_dir / "stage0_lighting_norm.png"), norm)

    # Step B — paper Lab baseline
    lab_uint8 = cv2.cvtColor(norm, cv2.COLOR_BGR2LAB)
    paper_lab = estimate_paper_lab(lab_uint8)
    info["paper_lab"] = [round(float(v), 1) for v in paper_lab]

    # Step C — Stage 1: Lab Δa* mask
    orange_mask = lab_delta_a_grid_mask(
        norm,
        paper_lab=paper_lab,
        delta_a_min=LAB_DELTA_A_MIN,
        orange_tolerance=LAB_ORANGE_TOLERANCE,
    )
    info["lab_mask_px"] = int((orange_mask > 0).sum())
    info["lab_mask_pct"] = round(
        100.0 * info["lab_mask_px"] / (orange_mask.size or 1), 2
    )
    cv2.imwrite(
        str(out_dir / "stage1_lab_mask_overlay.png"),
        _overlay_mask(norm, orange_mask, (0, 0, 255)),
    )

    # Step D — paper-blended fill + paper-to-white lift
    filled = paper_blended_fill(norm, orange_mask)
    lifted = normalise_paper_to_white(filled)
    cv2.imwrite(str(out_dir / "stage1b_after_blend.png"), lifted)

    # Step E — Stage 2: adaptive paper-relative clamp
    cleaned, clamp_mask = adaptive_paper_clamp(
        lifted,
        paper_sigma=ADAPTIVE_PAPER_SIGMA,
        ink_relative_gap=ADAPTIVE_INK_GAP,
    )
    info["clamp_mask_px"] = int((clamp_mask > 0).sum())
    info["clamp_mask_pct"] = round(
        100.0 * info["clamp_mask_px"] / (clamp_mask.size or 1), 2
    )
    cv2.imwrite(str(out_dir / "stage2_clamp_mask.png"), clamp_mask)
    cv2.imwrite(str(out_dir / "stage2_cleaned.png"), cleaned)

    # Step F — trace mask
    trace = build_trace_mask(cleaned)
    cv2.imwrite(str(out_dir / "trace_mask.png"), trace)
    info["trace_mask_px"] = int((trace > 0).sum())
    info["trace_mask_pct"] = round(
        100.0 * info["trace_mask_px"] / (trace.size or 1), 2
    )

    info["params"] = {
        "lab_delta_a_min": LAB_DELTA_A_MIN,
        "lab_orange_tolerance": LAB_ORANGE_TOLERANCE,
        "adaptive_ink_gap": ADAPTIVE_INK_GAP,
        "adaptive_paper_sigma": ADAPTIVE_PAPER_SIGMA,
    }
    return info


def _run_one(case_path: Path, out_subdir: Path) -> dict:
    out_subdir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(case_path))
    if img is None:
        return {"stem": case_path.stem, "status": "load_failed"}

    info: dict = {
        "stem": case_path.stem,
        "input_path": str(case_path),
    }
    t0 = time.time()

    # Phase 1
    try:
        rect_result = rectify_with_markers_fast_v4(
            img, paper_w_mm=PAPER_W_MM, paper_h_mm=PAPER_H_MM,
            px_per_mm=DEFAULT_PX_PER_MM,
        )
        rectified = rect_result.warped
        info["phase1_status"] = "ok"
    except Exception as exc:
        info["phase1_status"] = "failed"
        info["phase1_error"] = str(exc)
        info["elapsed_s"] = round(time.time() - t0, 3)
        return info
    info["phase1_elapsed_s"] = round(time.time() - t0, 3)
    cv2.imwrite(str(out_subdir / "rectified.png"), rectified)

    # Phase 2 v2 — diagnostic dump first
    t2 = time.time()
    try:
        stage_info = _stage_dump(rectified, out_subdir)
        info.update(stage_info)
        # Also call the official v2 entry point to sanity-check the
        # cleaned output matches the stage dump.
        template = load_template_config(TEMPLATE_ID)
        result = isolate_trace_candidates_v2(rectified, template)
        cv2.imwrite(
            str(out_subdir / "production_cleaned.png"),
            result.cleaned_bgr,
        )
        info["grid_colour"] = result.grid_colour
        info["phase2_status"] = "ok"
    except Exception as exc:
        info["phase2_status"] = "failed"
        info["phase2_error"] = str(exc)
        info["phase2_traceback"] = traceback.format_exc()
        info["elapsed_s"] = round(time.time() - t0, 3)
        return info
    info["phase2_elapsed_s"] = round(time.time() - t2, 3)
    info["elapsed_s"] = round(time.time() - t0, 3)

    # Side-by-side
    cleaned = cv2.imread(str(out_subdir / "stage2_cleaned.png"))
    side = _side_by_side(rectified, cleaned, case_path.stem)
    cv2.imwrite(str(out_subdir / "side_by_side.png"), side)

    (out_subdir / "info.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8",
    )
    return info


def _side_by_side(left, right, stem):
    h_max = max(left.shape[0], right.shape[0])

    def _pad_h(img, h):
        if img.shape[0] == h:
            return img
        pad = np.full((h - img.shape[0], img.shape[1], 3), 240, dtype=np.uint8)
        return np.vstack([img, pad])

    combo = np.hstack([_pad_h(left, h_max), _pad_h(right, h_max)])
    return label_image(
        combo,
        f"{stem}    LEFT: rectified         RIGHT: v2 cleaned",
        height_px=52, scale=1.0,
    )


def main() -> None:
    corpus = discover_corpus()
    if not corpus:
        print("No corpus files found.")
        sys.exit(1)

    run_dir = make_run_dir("step02_grid_removal", "v2_lab_local")
    print(f"Run dir: {run_dir.relative_to(REPO)}\n")
    print(f"v2 params:  delta_a>={LAB_DELTA_A_MIN}  "
          f"delta_b-delta_a<={LAB_ORANGE_TOLERANCE}  "
          f"ink_gap={ADAPTIVE_INK_GAP}  "
          f"paper_sigma={ADAPTIVE_PAPER_SIGMA}\n")
    print(f"{'case':<28s} {'p1':>6s}  {'p2':>6s}  "
          f"{'lab%':>6s}  {'clamp%':>6s}  {'trace%':>6s}  status")
    print("-" * 80)

    summaries = []
    side_tiles, clean_tiles = [], []
    for case in corpus:
        out_subdir = run_dir / case.stem
        info = _run_one(case.path, out_subdir)
        summaries.append({**info, "case_label": case.label,
                          "medium": case.medium, "paper": case.paper,
                          "index": case.index})
        p1 = info.get("phase1_elapsed_s", float("nan"))
        p2 = info.get("phase2_elapsed_s", float("nan"))
        lab_pct = info.get("lab_mask_pct", float("nan"))
        clamp_pct = info.get("clamp_mask_pct", float("nan"))
        trace_pct = info.get("trace_mask_pct", float("nan"))
        status = info.get("phase2_status", info.get("phase1_status", "?"))
        print(
            f"{case.stem:<28s} {p1:>5.2f}s  {p2:>5.2f}s  "
            f"{lab_pct:>5.2f}%  {clamp_pct:>5.2f}%  {trace_pct:>5.2f}%  "
            f"{status}"
        )
        side = cv2.imread(str(out_subdir / "side_by_side.png"))
        if side is not None:
            side_tiles.append(side)
        clean = cv2.imread(str(out_subdir / "stage2_cleaned.png"))
        if clean is not None:
            clean_tiles.append(
                label_image(clean, case.stem, height_px=48, scale=0.9)
            )

    if side_tiles:
        cv2.imwrite(
            str(run_dir / "summary_side_by_side.png"),
            grid_montage(side_tiles, cols=2, tile_max_dim=1100),
        )
    if clean_tiles:
        cv2.imwrite(
            str(run_dir / "summary_clean.png"),
            grid_montage(clean_tiles, cols=4, tile_max_dim=600),
        )

    summary_payload = {
        "step": "step02_grid_removal",
        "method": "v2_lab_local",
        "template_id": TEMPLATE_ID,
        "params": {
            "lab_delta_a_min": LAB_DELTA_A_MIN,
            "lab_orange_tolerance": LAB_ORANGE_TOLERANCE,
            "adaptive_ink_gap": ADAPTIVE_INK_GAP,
            "adaptive_paper_sigma": ADAPTIVE_PAPER_SIGMA,
        },
        "n_cases": len(summaries),
        "n_ok": sum(
            1 for s in summaries
            if s.get("phase1_status") == "ok"
            and s.get("phase2_status") == "ok"
        ),
        "cases": summaries,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2), encoding="utf-8",
    )
    print(f"\nFolder: {run_dir}")


if __name__ == "__main__":
    main()
