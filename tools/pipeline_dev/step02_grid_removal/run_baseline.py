"""Step 02 — Grid removal: baseline (current production L11 approach).

Reads the 12 BLUE_* source images, rectifies each with
``marker_rectify_fast_v4`` (Step 01 best, 12/12), and runs the
current production Phase 2 (``isolate_trace_candidates`` — L11
two-stage cleanup) on each rectified output.  Saves diagnostics so
we can see exactly which cases the existing logic handles well and
which need work.

For each case the run writes:

    <stem>/
      rectified.png              -- Phase 1 output (input to Phase 2)
      orange_mask_overlay.png    -- rectified with stage-1 HSV mask in red
      stage1_after_blend.png     -- after paper-blend + paper-to-white
      stage2_after_clamp.png     -- after hard-white clamp (= cleaned final)
      side_by_side.png           -- rectified | cleaned, with header
      trace_mask.png             -- final binary trace mask
      info.json                  -- timings + diagnostic counts

    summary_side_by_side.png     -- 4 x 3 grid of all 12 side-by-sides
    summary_clean.png            -- 4 x 3 grid of just the cleaned outputs
    summary.json                 -- per-case stats

    python tools/pipeline_dev/step02_grid_removal/run_baseline.py
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
from armourcore_cds.phase2.trace_isolation import (
    isolate_trace_candidates, normalise_lighting, build_orange_mask,
    paper_blended_fill, normalise_paper_to_white, build_trace_mask,
)
from armourcore_cds.templates.registry import load_template_config
from tools.pipeline_dev.corpus import (
    discover_corpus, make_run_dir, label_image, grid_montage,
)


# BLUE_* corpus uses this template (260 x 350 mm design area).  Phase 2's
# orange path doesn't actually use grid spacings from the template, but
# it expects a TemplateModel argument so the dataclass is loaded once
# and reused.
TEMPLATE_ID = "cds_colour_test_260x350"


def _overlay_mask(base: np.ndarray, mask: np.ndarray,
                  colour=(0, 0, 255), alpha: float = 0.55) -> np.ndarray:
    """Translucent red overlay of *mask* on *base*."""
    out = base.copy()
    if not np.any(mask):
        return out
    paint = np.zeros_like(base)
    paint[:] = colour
    blend = cv2.addWeighted(out, 1.0 - alpha, paint, alpha, 0)
    return np.where(mask[..., None] > 0, blend, out)


def _stage_dump(rect_bgr: np.ndarray, out_dir: Path) -> dict:
    """Reproduce the L11 stages with INTERMEDIATE saves for diagnostics.

    Mirrors the production code path inside ``isolate_trace_candidates``
    so we can save what each stage produced.  Returns a stats dict.
    """
    info: dict = {}

    # Step 1 — lighting normalisation
    norm = normalise_lighting(rect_bgr)
    cv2.imwrite(str(out_dir / "stage0_lighting_norm.png"), norm)

    # Step 2 — HSV orange detection (no chroma boost, sat>=25)
    orange_mask = build_orange_mask(norm, sat_min=25, chroma_boost=1.0)
    info["orange_mask_px"] = int((orange_mask > 0).sum())
    info["orange_mask_pct"] = round(
        100.0 * info["orange_mask_px"] / (orange_mask.size or 1), 2
    )
    cv2.imwrite(
        str(out_dir / "stage1a_orange_mask_overlay.png"),
        _overlay_mask(norm, orange_mask, (0, 0, 255)),
    )

    # Step 3 — paper-blended fill
    filled = paper_blended_fill(norm, orange_mask)
    # Step 4 — normalise paper to white
    lifted = normalise_paper_to_white(filled)
    cv2.imwrite(str(out_dir / "stage1b_after_blend.png"), lifted)

    # Step 5 — hard-white clamp
    WHITE_CLAMP_GRAY_THRESHOLD = 160
    gray = cv2.cvtColor(lifted, cv2.COLOR_BGR2GRAY)
    clamp_mask = gray >= WHITE_CLAMP_GRAY_THRESHOLD
    cleaned = lifted.copy()
    cleaned[clamp_mask] = (255, 255, 255)
    info["clamp_mask_px"] = int(clamp_mask.sum())
    info["clamp_mask_pct"] = round(
        100.0 * info["clamp_mask_px"] / (clamp_mask.size or 1), 2
    )
    cv2.imwrite(str(out_dir / "stage2_cleaned.png"), cleaned)

    # Step 6 — trace mask
    trace = build_trace_mask(cleaned)
    cv2.imwrite(str(out_dir / "trace_mask.png"), trace)
    info["trace_mask_px"] = int((trace > 0).sum())
    info["trace_mask_pct"] = round(
        100.0 * info["trace_mask_px"] / (trace.size or 1), 2
    )

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

    # --- Phase 1: rectify (v4) ---
    try:
        rect_result = rectify_with_markers_fast_v4(
            img, paper_w_mm=PAPER_W_MM, paper_h_mm=PAPER_H_MM,
            px_per_mm=DEFAULT_PX_PER_MM,
        )
        rectified = rect_result.warped
        info["phase1_status"] = "ok"
        info["rectified_size"] = list(rectified.shape[:2][::-1])
    except Exception as exc:
        info["phase1_status"] = "failed"
        info["phase1_error"] = str(exc)
        info["elapsed_s"] = round(time.time() - t0, 3)
        return info
    info["phase1_elapsed_s"] = round(time.time() - t0, 3)

    cv2.imwrite(str(out_subdir / "rectified.png"), rectified)

    # --- Phase 2: stage-by-stage dump for diagnostics ---
    t2 = time.time()
    try:
        template = load_template_config(TEMPLATE_ID)
        # First produce the diagnostic per-stage dump
        stage_info = _stage_dump(rectified, out_subdir)
        info.update(stage_info)

        # Also run the OFFICIAL production entry point to verify we get
        # the same final cleaned output (sanity-check).
        result = isolate_trace_candidates(rectified, template)
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

    # --- Side-by-side comparison ---
    cleaned_path = out_subdir / "stage2_cleaned.png"
    cleaned = cv2.imread(str(cleaned_path))
    side = _side_by_side(rectified, cleaned, case_path.stem)
    cv2.imwrite(str(out_subdir / "side_by_side.png"), side)

    (out_subdir / "info.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8",
    )
    return info


def _side_by_side(left: np.ndarray, right: np.ndarray,
                  stem: str) -> np.ndarray:
    """Stack two images with matched height and a stem header."""
    h_max = max(left.shape[0], right.shape[0])

    def _pad_to_height(img, h):
        if img.shape[0] == h:
            return img
        pad = np.full((h - img.shape[0], img.shape[1], 3), 240,
                      dtype=np.uint8)
        return np.vstack([img, pad])

    left_p = _pad_to_height(left, h_max)
    right_p = _pad_to_height(right, h_max)
    combo = np.hstack([left_p, right_p])
    return label_image(
        combo,
        f"{stem}    LEFT: rectified         RIGHT: cleaned",
        height_px=52, scale=1.0,
    )


def main() -> None:
    corpus = discover_corpus()
    if not corpus:
        print("No corpus files found.")
        sys.exit(1)

    run_dir = make_run_dir("step02_grid_removal", "baseline_L11")
    print(f"Run dir: {run_dir.relative_to(REPO)}\n")
    print(f"{'case':<28s} {'p1':>6s}  {'p2':>6s}  "
          f"{'orange%':>7s}  {'clamp%':>6s}  {'trace%':>6s}  status")
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
        o_pct = info.get("orange_mask_pct", float("nan"))
        c_pct = info.get("clamp_mask_pct", float("nan"))
        t_pct = info.get("trace_mask_pct", float("nan"))
        status = info.get("phase2_status",
                          info.get("phase1_status", "?"))
        print(
            f"{case.stem:<28s} {p1:>5.2f}s  {p2:>5.2f}s  "
            f"{o_pct:>6.2f}%  {c_pct:>5.2f}%  {t_pct:>5.2f}%  {status}"
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
        "method": "baseline_L11",
        "template_id": TEMPLATE_ID,
        "n_cases": len(summaries),
        "n_ok": sum(
            1 for s in summaries
            if s.get("phase1_status") == "ok"
            and s.get("phase2_status") == "ok"
        ),
        "n_failed": sum(
            1 for s in summaries
            if s.get("phase1_status") != "ok"
            or s.get("phase2_status") != "ok"
        ),
        "cases": summaries,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2), encoding="utf-8",
    )
    print(f"\nFolder: {run_dir}")


if __name__ == "__main__":
    main()
