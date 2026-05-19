# Phase 3 isolated-tool vectorisation — WORKING METHODS

## v1 — single-loop tools (IsoToolUnfixed01)
First validated on the wishbone Y-tool: clean smooth-Bezier output with
sharp V-junction cusp and tight hug to the trace.

## v2 — multi-loop tools (IsoToolUnfixed02, 03, 04 — clean)
Extends v1 to find ALL inner contours (not just the largest), so
multi-piece tools like a screwdriver (handle + stem) get all loops
vectorised.  Validated on:
* 01 — wishbone (1 loop, defaults)
* 02 — wrench-like (1 loop, defaults)
* 03 — screwdriver handle + stem (4 loops detected: 2 main + 2 small
       junction artefacts; defaults)
* 04 — knife with long obliterated span (handle + blade, needs
       `--close-kernel 50 --bridge-dist 400`)

Known issues at v2 (to be addressed in v3):
* Junction artefacts (small spurious loops at handle/stem joins)
* Per-image kernel tuning required for 04

## v3 — all 4 iso cases clean, fully adaptive
All four iso cases produce clean smooth-Bezier output with NO manual
parameter tuning required.  Key additions over v2:

* **Adaptive close kernel** — sweeps `[15..60]`, scoring by
  `(n_loops, total_area_bucket, -k, -n_ep)`.  Smaller kernel preferred
  unless a larger one captures meaningfully more area (e.g., 04's blade).
* **Closest-first matching** for stage-2 Bezier bridges (so short legit
  bridges consume their endpoints before long wrong bridges).
* **Non-crossing constraint** — a bridge whose chord crosses an
  already-accepted bridge is rejected.  This blocks the diagonal artefact
  in 04.
* **Relaxed facing threshold (-0.05)** — allows perpendicular-tangent
  bridges (corners-of-blade case where vertical tangents need to be
  bridged horizontally).
* **Obstacle check on chord** — rejects bridges whose chord crosses
  unrelated trace ink.
* **Relative loop-area filter** (5 % of largest) — drops junction
  artefact loops automatically.
* **Bulk tester** (`test_phase3_iso_bulk.py`) for regression watching.

The scripts in this folder are working snapshots — do not edit in place;
copy out to a new file if iterating.

## Method F (HYBRID) — gap repair

```
Input: cleaned PNG of an isolated tool outline (with breaks/gaps).
```

1. **Threshold** to binary trace mask  (`gray < 170`)
2. **MORPH_CLOSE** with elliptical kernel k=25
   → bridges all *tiny* gaps via morphology
3. **Medial axis** (`skimage.morphology.medial_axis`) of the closed band
   → 1-pixel-wide centerline of the tool outline, with tiny gaps
   already merged
4. **Skeleton-endpoint detection** (degree-1 nodes)
   → these are precisely the **medium gaps** that survived CLOSE
5. **Tangent-paired Bezier bridges**:
   - Walk back 15 pixels along the skeleton from each endpoint to get
     a clean tangent direction
   - Pair facing endpoints (mutual dot-product > 0.20) sorted by
     `(facing_avg) / max(1, dist/60)`
   - Connect each pair with a 1-pixel-wide cubic Hermite Bezier curve
6. **Output**: closed centerline (1-px) — ready for re-rendering or
   vectorisation

**Why it works**: morphology nails the easy 95% of gaps cleanly; the
remaining handful of skeleton endpoints (typically 2-10) are easy to
pair with simple direction+distance scoring, which previously failed
because there were thousands of phantom endpoints to wade through.

## Vectorisation pipeline

```
Input: closed centerline from Method F.
```

1. **Render uniform-thickness band** by dilating the centerline by half
   the average trace thickness — gap-fills now exactly match original
   trace width
2. **Find inner contour** with `cv2.findContours(RETR_TREE)` and pick
   the largest hole (= inside edge of tool tracing)
3. **RDP simplification** with epsilon=4.5 px → ~30-40 polygon nodes
4. **Centripetal Catmull-Rom → cubic Bezier**:
   - Centripetal parameterisation (alpha=0.5) so handle length scales
     with edge length — no horns at sharp corners
   - **Corner-cusp detection** (turn > 60°) → those nodes get
     zero-length handles, preserving sharp features (V-junctions, tips)
   - **Tension 0.5** for tight hug to the polygon
5. **Export**:
   - DXF with one cubic SPLINE entity per Bezier segment
     (Y flipped for DXF coord convention)
   - SVG with single `<path>` using cubic `C` commands
   - Layered SVG with original PNG (hidden) + gap-fixed PNG (visible) +
     vector path on top — opens as multi-layer document in Affinity
     Designer

## Key parameters (validated defaults)

| Parameter | Value | Notes |
|---|---|---|
| `dark_threshold` | 170 | grayscale; pixels below = trace |
| `close_kernel` | 25 | MORPH_CLOSE ellipse, bridges tiny gaps |
| `bridge_max_dist_px` | 200 | max gap distance for Bezier bridges |
| `lookback_px` | 15 | skeleton walk for tangent estimation |
| `facing_threshold` | 0.20 | dot-product min for endpoint pairing |
| `rdp` | 4.5 | RDP simplification epsilon |
| `tension` | 0.5 | Catmull-Rom handle scale |
| `corner_angle_deg` | 60 | turns sharper become cusps |
| `alpha` | 0.5 | centripetal Catmull-Rom |

## What this does NOT yet handle

- **IsoToolUnfixed03** (intersecting two-loop tool: handle + stem) —
  the inside-edge contour-finding only returns one hole, so the second
  loop is missed.
- **IsoToolUnfixed04** (long obliterated span) — may need a larger
  closing kernel OR a longer bridge max distance.
- **BatchToolUnfixed01** (multiple tools per image) — needs per-tool
  isolation BEFORE this pipeline can run.
