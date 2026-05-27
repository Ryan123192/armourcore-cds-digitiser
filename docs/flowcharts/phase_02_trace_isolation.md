# Phase 2 — Trace Isolation (Grid Removal)

**Module:** `src/armourcore_cds/phase2/trace_isolation.py`
**Public entry point:** `isolate_trace_candidates(image_bgr, template, grid_colour="auto", ...)`
**Approach:** "L11" — two-stage cleanup (HSV chroma + paper-blend fill, then hard-white clamp)

---

```
┌───────────────────────────────────────────────────────────────────────┐
│  INPUT: rectified BGR image from Phase 1                              │
│  (already at known px/mm scale, design area filling the frame)        │
│  + template (used to pick orange vs black grid path; not for sizing)  │
└──────────────────────────────────┬────────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────────┐
│  STEP 1 — Lighting normalisation (CLAHE on LAB-L)                     │
│  Photo scans usually have one side darker (overhead-light fall-off,   │
│  hand shadow, page curl).  In dark regions the orange grid and        │
│  customer ink become indistinguishable.                               │
│                                                                       │
│  • Convert to LAB, split L (lightness) channel                        │
│  • Apply Contrast-Limited Adaptive Histogram Equalisation:            │
│       clip_limit = 2.5                                                │
│       tile_grid = 32 × 32                                             │
│  • Re-merge and convert back to BGR                                   │
│                                                                       │
│  Result: locally flattened brightness — dark regions brightened,      │
│  highlights tamed — without bleeding global colour casts.             │
└──────────────────────────────────┬────────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────────┐
│  STEP 2 — Auto-detect grid colour mode                                │
│  Cheap probe: count what fraction of pixels fall in the HSV orange    │
│  envelope on a small downscale of the image.                          │
│       coverage >= 0.5%  →  use the ORANGE path (L11)                  │
│       coverage <  0.5%  →  use the legacy BLACK path                  │
│  All current production / V2 templates take the ORANGE path.          │
└──────────────────────────────────┬────────────────────────────────────┘
                                   │
                          (ORANGE path)
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────────┐
│  STEP 3 — Stage 1: HSV chroma detection                               │
│  build_orange_mask() with:                                            │
│       hue        ∈ [0, 40]    (red through yellow-orange)             │
│       saturation ≥ 25         (no chroma boost; clean discriminator   │
│                                 between vivid orange and warm paper)  │
│       value      ≥ 10         (catches dark major grid in shadow)     │
│  + tiny 2-px morph close (anti-aliasing only, NOT bridging)           │
│  + drop components < 15 px (noise specks)                             │
│                                                                       │
│  Result: a binary mask of "strongly saturated orange" pixels —        │
│  the saturated MAJOR grid lines and orange text get caught.           │
└──────────────────────────────────┬────────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────────┐
│  STEP 4 — Stage 1: paper-blended fill                                 │
│  For every pixel marked as "orange" in Step 3:                        │
│  • Zero out those pixels (set weight = 0)                             │
│  • Gaussian-blur both the masked image AND the weight (sigma = 40)    │
│  • Per-pixel paper_bg = blurred_image / blurred_weight                │
│       (smoothed local paper colour, ignoring the removed pixels)      │
│  • Copy paper_bg into the removed-pixel locations of the original     │
│                                                                       │
│  Effect: orange pixels disappear under realistic local paper colour   │
│  (preserves any shading the photo has, instead of pasting pure white  │
│  which leaves an artificial "ghost grid").                            │
└──────────────────────────────────┬────────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────────┐
│  STEP 5 — Stage 1: normalise paper-to-white                           │
│  Photographed paper sits at gray 170-200; PDFs sit at 255.  Phase 3   │
│  uses a fixed gray-threshold ink detector, so the paper baseline      │
│  must be consistent.                                                  │
│                                                                       │
│  • Detect "dark mark" pixels: gray < 90  (assumed customer ink)       │
│  • Compute a local paper-only background estimate via Gaussian        │
│    blur (sigma=80, excluding the dark-mark pixels)                    │
│  • Per-pixel scale factor = 245 / local_bg                            │
│       clamped to [0.5, 3.0] to prevent extremes                       │
│  • Multiply the image by this scale (channel-wise)                    │
│                                                                       │
│  Effect: paper is lifted to a consistent ~245 grey value across the   │
│  entire image regardless of input source.  Ink stays proportionally   │
│  darker, so the downstream threshold still finds it.                  │
└──────────────────────────────────┬────────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────────┐
│  STEP 6 — Stage 2: hard-white clamp                                   │
│  After Stage 1 the page is mostly paper-white with a faint remaining  │
│  signal from the lighter MINOR grid (FFEBCC) and any anti-aliased     │
│  orange edges that the HSV mask was too strict to catch.              │
│                                                                       │
│  • Convert cleaned image to grayscale                                 │
│  • For every pixel with gray ≥ 160:  set to pure white (255,255,255)  │
│  • For every pixel with gray <  160: leave unchanged (it's ink)       │
│                                                                       │
│  Why this works on every grid colour we care about:                   │
│  • Customer ink (pencil / pen) sits at gray 50-140                    │
│  • Lightened orange grid + anti-aliasing sits at gray 170-210         │
│  • There is a clean ~30-grey gap between them after Stage 1           │
│  • Picking 160 stays safely below the lightest ink and above the      │
│    darkest paper                                                      │
└──────────────────────────────────┬────────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────────┐
│  STEP 7 — Build the trace-candidate mask                              │
│  build_trace_mask() runs on a downscaled copy (max 1500-px dim) of    │
│  the cleaned image and combines two detectors:                        │
│                                                                       │
│  a) ABSOLUTE :  gray < 170                                            │
│        catches solid pen / dark pencil regardless of background       │
│                                                                       │
│  b) RELATIVE :  Gaussian-blur the gray image (sigma=25); a pixel is   │
│                 "relatively dark" if local_bg - gray ≥ 10             │
│        catches light pencil that's only faintly darker than paper     │
│                                                                       │
│  union(a, b)  →  drop connected components < 8 px (noise / JPEG)      │
│              →  binary trace mask                                     │
└──────────────────────────────────┬────────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────────┐
│  OUTPUT: TraceIsolationResult                                         │
│  • cleaned_bgr        — the grid-removed BGR raster                   │
│  • orange_mask        — Stage-1 HSV detection (diagnostic)            │
│  • removal_mask       — what was filled in Stage 1 (same as orange)   │
│  • trace_candidate_mask — binary mask of customer ink (input to       │
│                            Phase 3)                                   │
│  • grid_colour        — "orange" or "black" (which path ran)          │
└───────────────────────────────────────────────────────────────────────┘
```

## Key invariants the algorithm relies on

- Phase 1 has rectified the input to a known px/mm scale.  Sigma values
  (40 px for paper-blend, 25 px for trace-mask local bg, 80 px for paper
  normalisation) are tuned for the rectified scale, not raw photos.
- Customer ink is consistently *darker* than any printed grid colour.
- The orange grid is in the HSV envelope `hue ∈ [0, 40], sat ≥ 25`
  for the saturated lines, and at gray ≥ 160 after Stage 1 for the
  faint lines.

## Default tuning constants

| Constant | Value | Stage |
|---|---|---|
| `normalise_lighting` `clip_limit` | 2.5 | Lighting |
| `normalise_lighting` `tile_grid` | 32 | Lighting |
| `ORANGE_HUE_LOW` / `ORANGE_HUE_HIGH` | 0 / 40 | HSV mask |
| HSV `sat_min` (production call) | 25 | HSV mask |
| HSV `chroma_boost` (production call) | 1.0 (off) | HSV mask |
| `ORANGE_VAL_MIN` | 10 | HSV mask |
| `paper_blended_fill` `bg_sigma` | 40 | Stage 1 fill |
| `normalise_paper_to_white` `target_paper_value` | 245 | Paper lift |
| `normalise_paper_to_white` `bg_sigma` | 80 | Paper lift |
| **`WHITE_CLAMP_GRAY_THRESHOLD`** | **160** | **Stage 2 clamp** |
| `build_trace_mask` `dark_abs_threshold` | 170 | Trace mask |
| `build_trace_mask` `rel_dark_sigma` | 25 | Trace mask |
| `build_trace_mask` `rel_dark_min` | 10 | Trace mask |
| `max_processing_dim` | 1500 | trace mask only |

## How this differs from earlier experiments

Several earlier exploratory paths still live in the module but are
unused in production:

- **Chroma-boosted HSV** (`chroma_boost=3.0`) — over-detected on V2
  inputs where tool ink had warm JPEG noise.  Replaced by
  no-boost + hard-white clamp.
- **LAB a\* primary** — too sparse for the new lighter V2 grid.
- **Curved-band geometry detection** — useful for catching missed
  grid pixels, but was over-eating tool ink at crossings.  Disabled.
- **Iterative residual cleanup** — superseded by the hard-white
  clamp which removes residuals in one pass.

The current L11 approach landed after extensive A/B testing on the
V2BlueColourTest corpus and is what `isolate_trace_candidates` runs
when `grid_colour == "orange"`.
