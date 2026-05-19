"""Diagnostic: print candidate breakdown for a failing or suspicious image."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np
from armourcore_cds.phase1.boundary_detection import (
    detect_outer_border,
    _make_blue_mask, _make_blue_border_mask, _make_edge_mask,
    _build_blue_candidates, _dedupe_candidates, _clear_image_border,
    _quad_area, _quad_side_lengths, _occupancy_band_score,
    _score_candidate,
)


def diagnose(image_path: str, mode: str = "blue", aspect: float = 350 / 260,
             hex_border: str = "#00CAEC", fiducial_hex: str = "#FF0000"):
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Cannot read {image_path}")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    image_area = float(h * w)
    min_area_px = 0.02 * image_area
    border_px = max(4, int(round(min(h, w) * 0.008)))

    print(f"\n{'='*70}")
    print(f"File : {Path(image_path).name}  {w}x{h} ({w*h/1e6:.1f}MP)  mode={mode}")

    if mode == "blue":
        blue_mask = _make_blue_mask(img, [hex_border], hex_border)
        blue_mask = _clear_image_border(blue_mask, border_px)
        blue_pct = 100 * np.mean(blue_mask > 0)
        print(f"Blue mask coverage: {blue_pct:.1f}%")

        raw_candidates, masks = _build_blue_candidates(
            gray, img, border_px, min_area_px, [hex_border], hex_border)
        colour_mask = masks["colour_mask"]
        colour_border_mask = masks["colour_border_mask"]
    else:
        raw_candidates, masks = [], {}
        colour_mask = np.zeros((h, w), dtype=np.uint8)
        colour_border_mask = np.zeros((h, w), dtype=np.uint8)

    print(f"Raw candidates: {len(raw_candidates)}")
    candidates = _dedupe_candidates(raw_candidates, w, h)
    print(f"After dedupe:   {len(candidates)}")

    if not candidates:
        print("STOP: zero candidates after dedupe")
        return

    # Score all candidates
    scored = []
    for cand in candidates:
        sc = _score_candidate(
            cand, gray, img, image_area, aspect,
            colour_mask, colour_border_mask, mode,
            True, [fiducial_hex],
        )
        scored.append(sc)
    scored.sort(key=lambda x: x["score"], reverse=True)

    print(f"\n{'#':>3} {'src':<18} {'score':>6} {'ratio':>5} {'area':>5} "
          f"{'side':>5} {'wsid':>5} {'wcol':>5} {'mrkr':>5} {'plaus'}")
    print("-" * 80)
    for i, sc in enumerate(scored[:12]):
        flag = "OK" if sc["plausible"] else "  "
        print(f"{i:>3} {sc['source']:<18} {sc['score']:>6.2f} {sc['ratio_score']:>5.2f} "
              f"{sc['area_frac']:>5.2f} {sc['avg_side_score']:>5.2f} {sc['worst_side_score']:>5.2f} "
              f"{sc['worst_colour_side']:>5.2f} {sc['marker_support_score']:>5.2f}  {flag}")

    # Run detection normally
    print("\n--- Normal detection ---")
    try:
        result = detect_outer_border(
            img, expected_aspect_ratio=aspect,
            border_colour_mode=mode,
            outer_border_hex_candidates=[hex_border],
            blue_border_hex=hex_border,
            fiducial_hex_candidates=[fiducial_hex],
            use_shape_constraints=True,
        )
        d = result.diagnostics or {}
        print(f"  score={result.score:.2f}  conf={result.confidence}  src={d.get('candidate_source')}")
        print(f"  used_fallback={d.get('used_fallback')}  worst_colour_side={d.get('worst_colour_side',0):.3f}"
              f"  marker={d.get('marker_support_score',0):.3f}")
    except Exception as e:
        print(f"  FAIL: {e}")


if __name__ == "__main__":
    base = Path("data/inputs/raw_images")
    diagnose(str(base / "BlueColourTest01.JPG"), mode="blue")
    diagnose(str(base / "BlueColourTest02.JPG"), mode="blue")
    diagnose(str(base / "BlueColourTest04.png"), mode="blue")
