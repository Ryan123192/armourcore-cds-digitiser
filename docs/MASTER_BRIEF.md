# Master Brief — ArmourCore CDS Digitiser

## Project purpose
Build a robust, scalable system that converts customer-submitted CDS photos or scanned PDFs into usable 2D design inputs for ArmourCore.

## Current manual bottleneck
The slowest and most judgement-heavy step is turning a corrected CDS image into redraw-ready trace information for Affinity. This repo focuses on reducing manual cleanup before later vector and geometry stages.

## Primary project boundary
This project covers the path from:
- raw customer CDS image or scanned PDF
- to corrected raster
- to cleaned trace-preserved raster
- to later vector output suitable for Affinity import

It is **not** currently focused on:
- automatic tool naming
- automatic height assignment
- Fusion automation
- AI-first interpretation
- end-to-end design layout automation

## Immediate scope
1. Border/fiducial detection
2. Perspective rectification
3. Scaling to physical dimensions
4. Cropping to the designable region
5. Removing printed CDS elements
6. Suppressing glare, folds, shadows, and nuisance artefacts
7. Preserving customer-added marks for later stages

## Supported templates
- Regular CDS: 500 x 600 mm
- X-Large CDS: 500 x 900 mm
- Colour test mini grid: 260 x 350 mm

## Geometry truth
The authoritative scaling reference is the **inside-most edge of the thick outer border**.

## Output strategy
Every processing run should aim to produce:
- rectified raster
- debug intermediates
- run report JSON
- later: DXF/SVG outputs with ArmourCore-compatible layer naming

## Layer naming principle
Later vector output must align with the existing ArmourCore Affinity/Fusion conventions, including:
- `Outside Edge`
- `Badge Pockets`
- `Laser Labels`
- `Notes`
- `Reference & Scaling`
- numeric depth layers like `25mm`, `30mm`, `40mm`
- `THRU`

## Development philosophy
- deterministic classical CV first
- phase-based development
- resumable work blocks
- preserved debug outputs
- no premature vectorisation creep into Phase 1

## Phase order
1. Rectify and clean
2. Trace extraction
3. Vectorisation
4. Geometry rules
5. Optional design assistance tools

## Definition of “good enough” for Phase 1
Works reliably on stored project template examples and produces a correctly rectified, correctly scaled, visually clean raster that leaves customer-added marks available for later processing.

## Repo use model
During development:
- one input file
- one run output
- manual inspection

Later:
- batch processing
- staff-facing wrapper app
