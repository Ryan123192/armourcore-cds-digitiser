# Large-Format CDS Templates

Print at scale on the large-format printer.  Same colour scheme as the
A3 in-house sheet; only the design-area dimensions change.

| Paper size | Outer paper | Design area      | Marker spacing  |
|------------|-------------|------------------|-----------------|
| A3         | 420 x 297   | 350 x 260 mm     | corners of design area |
| A2         | 594 x 420   | 540 x 380 mm     | corners of design area |
| A1         | 841 x 594   | 780 x 540 mm     | corners of design area |
| A0         | 1189 x 841  | 1130 x 780 mm    | corners of design area |

## Printing rules (apply to all sizes)

1. **Background**: pure white, 100 gsm+ matte
2. **Outer border** (`#00CAEC` cyan):
   - 4 mm stroke around the design area on all formats
3. **Corner markers** (`#FF0033` red):
   - 15 x 15 mm filled square WITH a black X inside (the X is what the
     pipeline detects)
   - Positioned with their INNER CORNER (facing page centre) coinciding
     with the design-area corner
4. **Background guide** (pick ONE):
   - **Option A (recommended)**: 1 mm orange dots (`#FF9C00`) on a 10 mm
     grid -- proven cleanest on V3 testing
   - **Option B**: 0.2 mm peach lines (`#FFEBCC`) on a 10 mm grid
   - **No major grid lines** -- avoid the 50 mm major lines entirely
5. **No boundary-line text** -- it leaks into vectors
6. **Title block**: bottom 30 mm, light gray text, ignored by pipeline

## SVG generator

To produce a new template at a specific size, edit
`tools/make_template_svg.py` (TODO if not present):
```
python tools/make_template_svg.py --width 594 --height 420 \
       --design-w 540 --design-h 380 \
       --grid dots --output configs/templates/CDS_A2.svg
```
Then print via Affinity / Inkscape.

## Why "no major grid lines" matters

Previous V1 template had 50 mm major grid lines.  These are wide enough
to survive colour-strip when they happen to lie under shadow / scanner
dim spots, then create cross-shaped artefacts in Phase 3.  The V3 dots
have zero risk of this because dots cannot form line-like structures.

## Pipeline support for any size

The `vectorise_gui.py` "Paper size" dropdown sets:
```python
PAPER_W_MM, PAPER_H_MM = chosen_size
```
which feeds the rectifier and all downstream steps.  Default A3
matches the existing test sheets; A2/A1/A0 are wired in but need
matching printed templates to test against.

## Scanning the large formats

| Paper size | Scanner needed                              |
|------------|---------------------------------------------|
| A3         | Office multi-function printer (most have A3)|
| A2         | Large-format scanner OR scan in 2 halves and stitch |
| A1, A0     | Large-format scanner                        |

Stitching workflow for A2-on-A3-scanner: NOT recommended -- introduces
seam artefacts the pipeline isn't tuned for.  Use a large-format
scanner if A2+ is needed.
