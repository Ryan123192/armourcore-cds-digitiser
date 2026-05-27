# Phase 03 — Raster → Vector

Reference: `src/armourcore_cds/phase3/vectorise.py` (entry: `extract_vector_paths`)

```
                INPUT: trace_mask (uint8 binary, mask_h x mask_w)
                         |
                         v
        +--------------------------------------------+
        | (optional) repair_orange_gaps              |    medial-axis pre-step
        | medial-axis skeleton -> endpoint matching  |    that bridges grid
        | -> Bezier completion in orange-mask region |    crossings BEFORE
        +--------------------------------------------+    findContours runs
                         |
                         v
        +--------------------------------------------+
        | Pass 1 - GLOBAL CLOSE                      |
        | MORPH_CLOSE with kernel = gap_close_px     |    default 17px
        | bridges sub-grid-line gaps everywhere      |    (~2mm at 4.3px/mm)
        +--------------------------------------------+
                         |
                         v
        +--------------------------------------------+
        | cv2.findContours                           |
        | RETR_EXTERNAL  + CHAIN_APPROX_NONE         |    one contour per
        | (returns raw_contours)                     |    closed loop
        +--------------------------------------------+
                         |
                         v
            for contour in raw_contours:
                         |
        +----+-----------+----------+
        |    |area<min   |area>=min |
       drop  |           v          
             |    +-----------------------+
             |    | CIRCULARITY filter    |    4*pi*A / P^2
             |    | 4*pi*A/P^2 >= 0.05 ?  |    > 0.05 => closed loop
             |    +-----+-----+-----------+    < 0.05 => "ribbon"
             |          |     |
             |     YES  |     | NO (ribbon = double-sided open trace)
             |          v     v
             |   accept     +------------------------------+
             |              | Pass 2 - LOCAL RETRY         |
             |              | for k = gap_close+8 to       |
             |              |        max_gap_close (=51):  |
             |              |   crop bbox+pad              |
             |              |   MORPH_CLOSE(sub, k)        |
             |              |   findContours(sub)          |
             |              |   if circularity ok -> keep  |
             |              |   else try bigger k          |
             |              +------+-----------+-----------+
             |                     |           |
             |                resolved      dropped
             |                     |
             v                     v
          (next)                accept
                                   |
                                   v
                  +---------------------------------+
                  | RDP simplify (cv2.approxPolyDP) |   epsilon ~ 3px
                  +---------------------------------+
                                   |
                                   v
                  +---------------------------------+
                  | Catmull-Rom -> cubic Bezier     |   C1 smooth
                  +---------------------------------+
                                   |
                                   v
                          VectorPath dataclass
                                   |
                                   v
                  +---------------------------------+
                  | sort by area desc               |
                  +---------------------------------+
                                   |
        +--------------------------+----------------------+
        |                                                 |
        v                                                 v
+---------------+                              +---------------------+
| write_svg     |                              | render_vector_      |
| paths -> SVG  |                              | overlay -> PNG      |
| mm coordinates|                              | for visual check    |
+---------------+                              +---------------------+
```

## Where things go wrong (and where to fix)

| Symptom | Cause | Fix location |
|---|---|---|
| Tool shapes dropped as "ribbon" | gap between strokes wider than `max_gap_close_px` (default 51 ~6mm); pencil traces have larger gaps than pen | Increase gap_close_px / max_gap_close_px PER ROUTE (bigger for pencil) |
| Text characters become tiny paths | `min_area_px` default 200px is too small (~10mm² at 4.3px/mm) for 10mm-tracing baseline | Raise min_area_px to ~2000 (=100mm² = 10×10mm) |
| Thin slithers from grid dashes | Circularity 0.05 lets thin elongated shapes through if they happen to close into a sliver loop | Add bbox-aspect-ratio filter (drop min_dim/max_dim < 0.05 AND small area) |
| Boundary text "(Small Insert)" survives | Outline-fragment circularity can be > 0.05 because letters form small closed loops | Combine min-area + min-bbox-diagonal filters as a post-pass |
| Grid residual becomes random vector dashes | Per-pixel grid-dash blobs above 200px slip through | Tighten min_area_px (catches most); same fix as text |

## Tunable parameters

| Param | Current default | Suggested per-route |
|---|---|---|
| `gap_close_px` | 17 | pen: 11-15, pencil: 25-31 |
| `max_gap_close_px` | 51 | pen: 41, pencil: 81 |
| `min_area_px` | 200 (~10mm²) | bump to 2000 (~100mm² = 10×10mm) |
| `rdp_epsilon` | 3.0 | leave at 3.0 |
| `circularity_min` | 0.05 | leave at 0.05 (governs ribbon detection, not final accept) |

## Proposed new post-extraction filter

After `extract_vector_paths` returns the list, run:

```
filter_paths(paths, mask_shape, design_w_mm, design_h_mm,
             min_area_mm2 = 100.0,        # 10x10 mm tool tracing minimum
             min_bbox_diag_mm = 8.0,      # rejects most text characters
             max_aspect_ratio = 20.0,     # reject 20:1 slithers
             min_area_to_bbox = 0.01      # reject hollow-but-tiny outlines
            ) -> list[VectorPath]
```

This is **purely subtractive** — no shape data is altered, only paths
dropped.  Existing `extract_vector_paths` stays unchanged.
