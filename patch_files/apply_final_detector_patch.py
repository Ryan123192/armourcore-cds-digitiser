from __future__ import annotations
from pathlib import Path

def replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise RuntimeError(f"Pattern not found:\n{old[:120]}")
    return text.replace(old, new, 1)

def main() -> int:
    root = Path(".")
    p = root / "src" / "armourcore_cds" / "phase1" / "boundary_detection.py"
    t = p.read_text(encoding="utf-8")

    # 1) Broaden orange mask toward the real printed terracotta border.
    t = replace_once(
        t,
        'lower2 = np.array([0, 20, 40], dtype=np.uint8)\n    upper2 = np.array([25, 180, 220], dtype=np.uint8)',
        'lower2 = np.array([0, 12, 35], dtype=np.uint8)\n    upper2 = np.array([30, 210, 235], dtype=np.uint8)'
    )

    # 2) In orange mode, tighten occupancy expectations so giant/page-like candidates lose.
    t = replace_once(
        t,
        'area_score = _occupancy_band_score(area_frac, lo=0.05, peak_lo=0.18, peak_hi=0.80, hi=0.92)\n            extent_score = _occupancy_band_score(bbox_frac, lo=0.08, peak_lo=0.20, peak_hi=0.82, hi=0.94)',
        'area_score = _occupancy_band_score(area_frac, lo=0.04, peak_lo=0.12, peak_hi=0.60, hi=0.86)\n            extent_score = _occupancy_band_score(bbox_frac, lo=0.06, peak_lo=0.14, peak_hi=0.64, hi=0.88)'
    )

    # 3) Stronger page-like penalty in orange mode.
    t = replace_once(
        t,
        'if area_frac > 0.72 or bbox_frac > 0.76:\n                page_like_penalty = max(area_frac - 0.72, 0.0) + max(bbox_frac - 0.76, 0.0)',
        'if area_frac > 0.58 or bbox_frac > 0.62:\n                page_like_penalty = 1.6 * max(area_frac - 0.58, 0.0) + 1.8 * max(bbox_frac - 0.62, 0.0)'
    )

    # 4) In orange mode, strongly prefer thick warm-colour borders and punish edge-only candidates.
    t = replace_once(
        t,
        'if requested_colour_mode == "orange":\n            total = (2.20 * ratio_score + 1.00 * area_score + 0.90 * extent_score + 1.30 * side_score + 0.70 * band_score + 2.40 * thickness_score + 2.00 * colour_score + 0.85 * marker_support_score + 0.60 * corner_score - 0.90 * outside_penalty - 1.60 * marker_inside_penalty - 2.80 * page_like_penalty)\n            if cand.get("source", "").startswith("edge") and colour_score < 0.12:\n                total -= 1.5',
        'if requested_colour_mode == "orange":\n            total = (2.20 * ratio_score + 0.95 * area_score + 0.80 * extent_score + 1.10 * side_score + 0.55 * band_score + 3.00 * thickness_score + 3.00 * colour_score + 1.20 * marker_support_score + 0.55 * corner_score - 0.90 * outside_penalty - 1.80 * marker_inside_penalty - 4.00 * page_like_penalty)\n            if cand.get("source", "").startswith("edge") and colour_score < 0.22:\n                total -= 3.0\n            if thickness_score < 0.22:\n                total -= 2.0\n            if colour_score < 0.16:\n                total -= 2.5'
    )

    # 5) In auto mode, prefer black unless orange is very clearly present.
    t = replace_once(
        t,
        'if orange_colour_score > black_colour_score + 0.08 and orange_thickness_score > black_thickness_score * 0.85:',
        'if orange_colour_score > black_colour_score + 0.18 and orange_thickness_score > max(0.22, black_thickness_score * 1.10):'
    )

    # 6) Fix kwarg mismatch in pipeline if still present.
    pipe = root / "src" / "armourcore_cds" / "phase1" / "pipeline.py"
    if pipe.exists():
        pt = pipe.read_text(encoding="utf-8")
        pt = pt.replace("colour_mode=border_colour_mode,", "border_colour_mode=border_colour_mode,")
        pipe.write_text(pt, encoding="utf-8")

    p.write_text(t, encoding="utf-8")
    print("Applied final orange/black detector retune.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())