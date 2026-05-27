# PRODUCTION SNAPSHOT - 2026-05-26 (updated)

FROZEN COPY of the working pipeline.  DO NOT MODIFY THESE FILES.

## Results

| Case                  | REAL | GRID | TEXT | SLIV | Verdict   |
|-----------------------|------|------|------|------|-----------|
| BLUE_PEN_FLAT_01      | 8    | 0    | 0    | 0    | GOOD      |
| BLUE_PEN_FLAT_02      | 10   | 0    | 0    | 0    | EXCELLENT |
| BLUE_PEN_FLAT_03      | 11   | 0    | 0    | 0    | EXCELLENT |
| BLUE_PENCIL_FLAT_01   | 12   | 0    | 0    | 0    | EXCELLENT |

3 of 4 cases at EXCELLENT (delta ≤ 2 from 11-shape target with zero
grid leaks).  img1 still missing 3 small shapes (the smallest tracings
that fall below min_area filters).

## Pipeline by route

### pen_standard (img3)
  rectify -> v14 adaptive -> grid_strip -> vectorise_v4 (RETR_CCOMP)

### pen_darkborder (img1, img2)
  rectify -> v14 -> strip_dark_border -> grid_strip_aggressive
          -> grid_strip -> strip_dashed_boundary_lines
          -> vectorise_v4

### pencil (img4)
  rectify -> v14 -> sauvola_adaptive(25, 0.15) -> grid_strip
          -> trace_mask(min_component=40)
          -> extract_vector_paths_pencil(dilate=17, close=11)

## Files in this snapshot

- `batch_phase3_v20.py`           - main driver
- `grid_strip.py`                  - template major+minor band removal
- `grid_strip_aggressive.py`       - dark-border + aggressive major-grid
- `line_strip.py`                  - Hough-based dashed boundary line removal
- `pencil_enhance_v2.py`           - Sauvola + Niblack + Frangi (pencil winner: sauvola)
- `vectorise_v4.py`                - RETR_CCOMP + hole-preference + bbox-bounded rescue
- `vectorise_pencil.py`            - dilate-first pencil extraction
- `path_classifier.py`             - REAL/GRID/TEXT/SLIV/NOISE diagnostic
