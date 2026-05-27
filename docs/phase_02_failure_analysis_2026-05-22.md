# Phase 2 (Trace Isolation) — Failure Analysis on BLUE_* Corpus
**Run reference:** `data/outputs/pipeline_dev/step02_grid_removal/20260522-190807_baseline_L11/`
**Algorithm under test:** L11 two-stage cleanup (HSV chroma + paper-blend, then hard-white clamp at gray ≥ 160)
**Conducted by user inspection of `summary_clean.png` + per-case stages.**

---

## Per-case observations (user verbatim, condensed)

| Case | Lighting | Verdict | Failure mode |
|---|---|---|---|
| **PEN_CREASED_01** | mixed | decent | Creases create dark lines similar to pen ink → will be problem at vectorise. Lots of grid still visible because lighter V2 grid reads as "desaturated browny grey", not orange. Creases break grid lines into segments. |
| **PEN_CREASED_02** | sun warm | broken | Horizontal crease creates intense shadow just above it; any tool line in that shadow gets removed at the cleaned stage → broken vector. |
| **PEN_CREASED_03** | blue window tint | partial | Trace mask clearly highlights tools, BUT border + major grid + some minor grid REMAIN because the heavy blue tint shifts the orange out of HSV range. |
| **PEN_FLAT_01** | mixed | partial | Tool traces clean, BUT major grid lines + outside border remain. |
| **PEN_FLAT_02** | warm sun | over-eats | Shadow on middle-left gets detected as orange (warm sun on dark = "orange") → tool tracings removed inside that shadow. Major false positive. |
| **PEN_FLAT_03** | warm sun | over-eats | Same as above — dark tracing lines themselves get picked up as orange and removed. Minor grid NOT in orange mask but somehow disappears completely (hard-white clamp). |
| **PENCIL_CREASED_01** | blue tint | partial | Major + some minor grid remain. Irregular shadows from scrunched paper show as blue blotches in cleaned. Pencil is darker than blotches → still filterable downstream. |
| **PENCIL_CREASED_02** | warm sun | broken | Crease shadows show heavy in orange mask → tool tracings removed in those regions. |
| **PENCIL_CREASED_03** | warm sun | broken | Shadows show as blotches. Tool tracings removed when they intersect shadow. |
| **PENCIL_CREASED_04** | warm sun + heavy fold | broken | Vertical and horizontal fold creases show dark in cleaned because of intense shadow. |
| **PENCIL_CREASED_05** | mixed | close | Major grid + border remain. Shadows light. Pencil clear. |
| **PENCIL_FLAT_01** | flat | close | No major or minor grid, only some outer border. Pencil there but broken near major grid lines / text. Cleaning makes pencil thinner / spotty. |

---

## Failure modes grouped into root causes

### 1. Lighting cast shifts the orange out of detection
- **Blue window light** (PEN_CREASED_03, PEN_FLAT_01, PENCIL_CREASED_01,
  PENCIL_CREASED_05): the orange grid becomes brown-grey
  in the camera; HSV `sat ≥ 25, hue 0–40` no longer matches it, so the
  Stage-1 mask is essentially empty.  The grid then has to be wiped by
  the Stage-2 hard-white clamp, which works for the **lighter** grid
  lines but fails for the **major (darker) grid** and the outside border.

- **Warm sun light** (PEN_FLAT_02/03, PENCIL_CREASED_02/03/04):
  shadows on the paper take on a warm cast, so dark non-orange pixels
  (creases, fingers, fold-shadows, even tool ink itself) end up
  satisfying the orange criteria and get **removed**.  Tool tracings
  inside any shadow disappear.

> **Net effect:** the same HSV thresholds either under- or over-detect
> orange depending on the photo's white-balance.

### 2. Creases / fold shadows are dark enough to look like ink
- Crease lines are typically 4–10 px wide, gray 50–120 — the same range
  as pencil ink.
- The L11 Stage-2 clamp is gray-only and doesn't know about shape, so
  crease shadows pass through into the cleaned output.
- Downstream Phase 3 will then trace these crease lines as if they were
  tool outlines.

### 3. The hard-white clamp eats tool ink that sits in shadow
- A pencil line crossing a dark crease shadow has the same gray value as
  paper-in-shadow on either side.
- Phase 2 lifts paper to ~245 globally, but the shadow region stays
  darker than 160 — so it survives the clamp but the tool ink inside is
  indistinguishable from the surrounding shadow → continuity is lost.

### 4. HSV is brightness-coupled
- HSV saturation depends on Value.  A vivid orange in deep shadow has
  low saturation (close to grey), even though its hue is still orange.
- The L11 threshold `sat ≥ 25` therefore loses any grid that's slightly
  shaded.

### 5. The lighter V2 grid is just barely above 160
- The minor grid prints at gray ~200 *under good lighting*.  Under blue
  light it can drop into the 150–170 range — right at the clamp
  threshold.  Sometimes caught, sometimes not.

---

## Themes for the fix

| # | Theme | Direct cause it attacks |
|---|---|---|
| A | Replace HSV with **Lab a\* delta relative to a LOCAL paper baseline** | Issues 1 + 4 |
| B | **Shadow detection and lift** before colour detection | Issues 2 + 3 |
| C | **Per-region adaptive paper threshold** instead of global gray ≥ 160 | Issue 5 |
| D | **Geometric grid projection** as a safety net (we know mm/px from Phase 1) | Catches every grid line regardless of colour |

### Path A — "Like Phase 1 but for grid"
Phase 1 already proved that **Lab delta-a vs a locally-sampled paper
baseline** is lighting-invariant: it solved the same blue-light /
warm-sun problem on the marker corners.  The grid is the same physical
ink as the markers, so the same approach should work:

  1. Sample paper Lab baseline from a clean central patch (already
     done in Phase 1 — we can reuse the result).
  2. For each pixel: compute `delta_a = a − ap` and `delta_b − delta_a`.
  3. Threshold at `delta_a ≥ N` to find any "warm-shifted" pixel
     (catches the grid in both blue-light and warm-sun photos because
     the SAME shift formula applies to both).
  4. Anti-aliasing fringe handled by tiny morph close.

This is a much more robust replacement for Stage 1.  **Crucially it
does NOT trip on dark shadows** because shadows have low Lab-L but
neutral a\* (no warm shift relative to paper).

### Path B — Shadow detection and lift
Before colour detection runs:

  1. Compute a coarse "shadow map" = where is the L channel significantly
     darker than the rest of the paper?
  2. **Lift** the shadow regions: scale their L channel up so the
     paper inside the shadow reaches the same baseline as outside.
  3. Run the colour detection on the lifted image.
  4. Apply the cleaning to the *original* image (or to the lifted
     image — either way, tool ink in shadow is preserved because
     the lifted shadow now has the same paper baseline).

This is what the existing `normalise_lighting()` CLAHE tries to do, but
CLAHE is per-tile and doesn't know what "shadow" looks like.  An
explicit shadow detector + targeted lift should do a much better job.

### Path C — Adaptive paper threshold
Replace the fixed `gray ≥ 160 → white` clamp with:

  - per-region Otsu threshold, OR
  - subtract the local-paper estimate (Gaussian blur after dark-mark
    exclusion) and threshold the *residual*.

A pixel is "paper" if it's within X grey of its local paper baseline,
not if it's brighter than a global 160.  Same principle as A but for
the second stage.

### Path D — Geometric grid projection
After Phase 1 we know the rectified image is at `10 px/mm`.  The grid
is at known positions in mm.  So we can:

  1. Draw the expected grid as a binary mask in the rectified
     coordinate system.
  2. Dilate it slightly to allow for sub-pixel misalignment.
  3. Wipe that mask to paper-white directly.

This is **completely lighting-invariant** for the printed grid.  It
WON'T fix the shadow / crease issues, and it's brittle if the
rectification is slightly off — but it's a one-line safety net we can
add on top of A + C if needed.

---

## Recommended next step (single experiment)

**Build a v2 of `isolate_trace_candidates` that combines Path A + Path C.**

* Path A replaces Stage 1 (HSV) with **local-Lab delta-a**.
* Path C replaces Stage 2 (gray ≥ 160 clamp) with **local-paper-relative
  brightness threshold**.
* Path B (shadow lift) is then *optional* — A + C alone should solve
  most of the colour-cast cases.  We add B only if needed.
* Path D is held in reserve as a safety net.

Why this combo first:

1. Path A directly attacks the blue / warm-light failures (5 cases).
2. Path C directly attacks the shadow-clamps-ink issue (3 cases).
3. Together they retire ~10 of the 12 failure observations without
   requiring shape-based geometry or pattern projection.
4. Both are *additive* — we keep the L11 module in place and build a
   sibling `trace_isolation_v2.py` so we can A/B side-by-side and
   never lose the working baseline.

After A + C lands, re-run on the corpus and judge what's left.  The
remaining cases (crease lines being mistaken for ink, partial pencil
broken across shadows) likely need Path B (shadow detection) and
possibly geometric reasoning, which we'll attack as a second pass.
