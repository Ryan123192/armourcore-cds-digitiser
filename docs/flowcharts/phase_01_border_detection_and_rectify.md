# Phase 1 — Border Detection and Rectification

**Module:** `src/armourcore_cds/phase1/marker_rectify_fast_v4.py`
**Status:** Verified 12/12 on the BLUE_* test corpus (2026-05-22)
**Public entry point:** `rectify_with_markers_fast_v4(image_bgr, *, paper_w_mm, paper_h_mm, px_per_mm, ...)`

---

```
┌───────────────────────────────────────────────────────────────────────┐
│  INPUT: BGR image (numpy array, any resolution, any background)       │
└──────────────────────────────────┬────────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────────┐
│  STEP 1 — Estimate the paper's baseline Lab colour                    │
│  • Crop the central 10% patch of the image                            │
│  • Of those pixels, take only the top 25% brightest (skips creases)   │
│  • Median Lab of that set = (Lp, ap, bp) -- the "paper baseline"      │
└──────────────────────────────────┬────────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────────┐
│  STEP 2 — Build a lighting-invariant red-ink mask                     │
│  For every pixel, compute its Lab delta from the paper baseline:      │
│       delta_a = a_pixel - ap   (red-shift)                            │
│       delta_b = b_pixel - bp   (yellow-shift)                         │
│  Pixel is "red ink" if BOTH:                                          │
│       delta_a >= 12             (clearly red-shifted)                 │
│       delta_b - delta_a <= 6    (NOT orange/yellow → rejects wood,    │
│                                  warm paper edges, orange grid lines) │
│  Output: a binary mask of every red marker pixel in the image.        │
└──────────────────────────────────┬────────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────────┐
│  STEP 3 — Morphological clean-up                                      │
│  • CLOSE (18×18 rect, 2 iterations): fills the crosshair / circle     │
│    gaps inside each marker so each marker becomes one solid blob.     │
│  • OPEN  (15×15 rect, 1 iteration): erases the thin orange grid       │
│    stubs that connect to the markers (grid lines are too thin to      │
│    survive the open; marker bodies are wide enough to survive).       │
│  Result: a binary "marker core" mask -- ideally 4 clean square blobs. │
└──────────────────────────────────┬────────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────────┐
│  STEP 4 — Find every marker-shaped blob globally                      │
│  For each connected component in the core mask, accept it ONLY if:    │
│       3000 ≤ area ≤ 60000 px                                          │
│       minAreaRect aspect ratio ≤ 1.6   (square-ish, perspective ok)   │
│       solidity ≥ 0.65                  (rejects L-shapes from grid)   │
│       not touching the image edge                                     │
│  If < 4 survive, progressively relax the area floor to                │
│  [1500, 800, 400, 200] until ≥ 4 are found (or give up).              │
└──────────────────────────────────┬────────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────────┐
│  STEP 5 — Label the 4 markers by position                             │
│  • Compute the centroid of all surviving candidates.                  │
│  • Each candidate goes into a quadrant by its position relative       │
│    to the centroid: TL / TR / BR / BL.                                │
│  • Within each quadrant, pick the candidate with the largest area     │
│    (the real marker body, not a fragment).                            │
└──────────────────────────────────┬────────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────────┐
│  STEP 6 — Suspect-area check                                          │
│  • Compute the median area of the 4 chosen candidates.                │
│  • Any chosen candidate with area < 0.40 × median is flagged          │
│    SUSPECT (one of its corners was likely partly destroyed by         │
│    shadow / crease, leaving only a small fragment of red ink).        │
│  • Strong (non-suspect) candidates are refined first.                 │
└──────────────────────────────────┬────────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────────┐
│  STEP 7 — Refine strong markers (template + colour fallback)          │
│  For each strong candidate, crop a 220×220 px ROI around its centre   │
│  and try TWO methods in parallel:                                     │
│                                                                       │
│    a) tmpl_cnr  — multi-scale corner-aware template match.            │
│       Synthesises a black-ink template per corner (TL/TR/BR/BL)       │
│       with ruler ticks on the inward-facing edges.  Tries scales      │
│       0.85× → 1.25× the expected marker size, keeps best match.       │
│                                                                       │
│    b) ink_lab   — direct min-area-rect of all Lab-red pixels in       │
│       the ROI.  Pure colour, doesn't depend on shape.                 │
│                                                                       │
│  Whichever method scores lower (better) wins.  Score combines:        │
│       aspect ratio close to 1                                         │
│       centre close to ROI middle                                      │
│       size close to expected                                          │
└──────────────────────────────────┬────────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────────┐
│  STEP 8 — Geometric rescue for any suspect markers                    │
│  For each suspect candidate (e.g. shadowed TL on PCC01):              │
│  • Predict its true position from the 3 strong markers via            │
│    parallelogram closure:                                             │
│         target = adj_A + adj_B - opposite                             │
│         e.g. TL_predicted = TR + BL - BR                              │
│  • Re-run the template + ink_lab refine inside a LARGER 440×440 ROI   │
│    anchored on the predicted position (extra room to absorb           │
│    prediction error and still contain the full marker).               │
│  • The actual marker is now found at its true centre, not at the      │
│    offset fragment centre.                                            │
└──────────────────────────────────┬────────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────────┐
│  STEP 9 — Pick each marker's INNER corner                             │
│  Each refined marker is a rotated rectangle with 4 corner points.     │
│  • Compute the centre of all 4 marker bodies = "paper centre".        │
│  • For each marker, pick the corner of its rectangle CLOSEST to       │
│    the paper centre.  That's the corner of the design area, which     │
│    is where the printed border starts.                                │
│  Output: 4 (x, y) points -- the corners of the design quad.           │
└──────────────────────────────────┬────────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────────┐
│  STEP 10 — Perspective warp to known mm dimensions                    │
│  • Target raster size: paper_w_mm × px_per_mm  by                     │
│                          paper_h_mm × px_per_mm                       │
│    (e.g. 345 × 255 mm @ 10 px/mm = 3450 × 2550 px).                   │
│  • Build a homography from the 4 inner corners to the 4 target        │
│    rectangle corners with cv2.getPerspectiveTransform.                │
│  • Warp the input image with LANCZOS4 interpolation.                  │
└──────────────────────────────────┬────────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────────┐
│  OUTPUT: clean rectified raster at known px/mm scale                  │
│  Plus diagnostic data: 4 marker boxes, scores, rescue flag, etc.      │
└───────────────────────────────────────────────────────────────────────┘
```

## Key invariants the algorithm relies on

- The 4 red markers are always present at the corners of the printed border.
- The markers form a rectangle (with mild perspective allowed).
- The image has at least 30% of its area as paper-ish content (so the central-patch baseline estimate finds paper).
- Markers are 15 × 15 mm physically; their pixel size depends on shot distance.

## Default tuning constants (see top of `marker_rectify_fast_v4.py`)

| Constant | Value | Purpose |
|---|---|---|
| `DELTA_A_MIN` | 12 | Lab a* shift threshold to count as "red" |
| `ORANGE_TOLERANCE` | 6 | rejects orange/yellow (wood, grid) |
| `COARSE_CLOSE_KSIZE` | 18 | fills marker crosshair gaps |
| `COARSE_OPEN_KSIZE` | 15 | kills thin grid-line stubs |
| `CAND_MIN_AREA_PX` | 3000 | minimum candidate blob area (relaxes to 200 progressively) |
| `CAND_MAX_AREA_PX` | 60000 | maximum candidate blob area |
| `CAND_MAX_ASPECT` | 1.6 | max minAreaRect aspect ratio |
| `CAND_MIN_SOLIDITY` | 0.65 | min contour/hull-area ratio |
| `ROI_HALF` | 110 | refine ROI half-size for strong markers (220×220 ROI) |
| `SUSPECT_AREA_RATIO` | 0.40 | candidate flagged suspect if area < this × median |
| `SUSPECT_ROI_HALF` | 220 | rescue ROI half-size (440×440 ROI) |
| `PAPER_W_MM` / `PAPER_H_MM` | 345 / 255 | physical paper-design dimensions |
| `DEFAULT_PX_PER_MM` | 10 | output rectified scale (10 px = 1 mm) |
