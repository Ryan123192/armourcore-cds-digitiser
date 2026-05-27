# In-House CDS Sheet — Design Spec

This document tells you exactly how to print the A3 tracing sheet so
the vectoriser hits img3-quality (11/11 EXCELLENT) reliably every time.

## Why a new sheet?

The current orange-grid CDS sheet works **only when lighting separates
orange from black** (img3 case).  On a phone photo with coloured light
cast, orange grid and black pen ink land at the same gray value and
become mathematically indistinguishable.  A controlled in-house sheet +
scanner removes 95% of that ambiguity.

## Print specs

| Property              | Value                                            |
|-----------------------|--------------------------------------------------|
| **Paper size**        | A3 (297 x 420 mm)                                |
| **Design area**       | 260 x 350 mm centred (matches existing template) |
| **Paper colour**      | Pure white 100 gsm matte                         |
| **Print process**     | Laser or inkjet at 300+ dpi                      |

## Required printed elements

### 1. Outer border (Phase 1 rectification anchor)
- **Colour**: cyan (`#00CAEC`) - matches existing detector
- **Stroke**: 4 mm wide rectangle around the design area
- **Position**: aligned to design area outer edge

### 2. Corner fiducials (Phase 1 marker detection)
- **Colour**: bright red (`#FF0000`)
- **Shape**: 12 mm filled circle in each corner of the design area
- **Position**: 8 mm inset from outer cyan border, one per corner

### 3. Grid (optional - recommended OFF for the first version)
**Option A (best): No grid at all.** Eliminates the entire class of
"grid line under tool ink" failures.  Trade-off: customer can't use
the grid to align shapes; they'd freehand or use a separate alignment
template.

**Option B: 5 mm dot grid.** Tiny dots (0.3 mm diameter) at every 5 mm
intersection, light gray (50% gray).  Provides alignment guide for the
customer, vectoriser ignores them as NOISE due to size filter.  Dots
don't form continuous lines, so they can't connect shapes together.

**Option C: Faint blue 10 mm grid lines.**  Pure blue (`#0066FF`),
0.2 mm strokes.  Easy to colour-filter out (no ambiguity with black
pen).  Risk: heavy lighting cast may still confuse it.

**Recommendation: Option A for v1, Option B if customer feedback wants
alignment aid.**

### 4. Title block / customer fields
- Bottom 30 mm of the sheet
- Light gray (40% gray) text
- Fields: Customer name, Date, Job ID
- Vectoriser ignores anything in the bottom 30 mm by cropping

### 5. NO boundary lines, NO "Boundary Line (Small Insert)" text
The dashed boundary lines on the existing sheet are the #1 cause of
failures on img1/img2.  Remove them entirely from the new sheet.

## Tracing instructions for staff

| Tool          | Acceptable                                  | NOT acceptable          |
|---------------|---------------------------------------------|-------------------------|
| **Pen**       | Black sharpie / fineliner, ~1 mm stroke     | Blue ballpoint, pencil  |
| **Pencil**    | 4B-6B graphite, dark and consistent stroke  | HB, hard pencil         |
| **Stroke**    | Single confident closed loop per tool       | Doubled-up parallel lines |
| **Min size**  | Each tool tracing >= 15 mm in shortest dim  | Smaller = will be filtered |
| **Spacing**   | >= 10 mm gap between tool tracings          | Touching = will merge   |

## Scanning instructions

| Setting        | Value                                                |
|----------------|------------------------------------------------------|
| Resolution     | **400 dpi**                                          |
| Mode           | **Colour** (NOT greyscale - phase 1 needs cyan/red)  |
| Output format  | PNG (lossless) or TIFF                               |
| Cropping       | Full sheet, no cropping                              |
| Adjustments    | NONE - no auto-rotate, no auto-contrast, no de-skew  |

Save as `<job_id>_<date>.png` in the watched input folder.

## Expected result on a well-prepared scan

| Metric                   | Target                          |
|--------------------------|---------------------------------|
| Phase 1 rectification    | 100% success                    |
| Phase 2 cleaning         | clean shapes, zero grid residue |
| Phase 3 closed loops     | 1 vector per drawn shape, 100% |
| Vector node count        | < 100 per shape (smooth)       |
| Total time per sheet     | < 5 seconds                    |
| Human review time        | < 30 seconds in the review GUI  |
