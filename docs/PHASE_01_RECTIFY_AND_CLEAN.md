# Phase 01 — Rectify and Clean

## Goal
Convert one raw CDS image or scanned PDF into a correctly rectified, correctly scaled, cropped raster of the designable region.

## Input
- single image file (`.jpg`, `.jpeg`, `.png`, later possibly `.heic`)
- or single-page scanned PDF

## Output
Minimum:
- rectified image
- scaled image
- cropped design-area image
- debug images for each major stage
- `run_report.json`

Stretch goal:
- initial printed CDS suppression
- initial artefact reduction while preserving customer marks

## Phase 01 success criteria
- outer border detected
- perspective corrected accurately
- scale matches template physical dimensions
- crop corresponds to the inside-most edge of the thick border
- outputs are visually repeatable across stored benchmark images

## Current assumptions
- border is the primary geometry truth
- fiducials/reference marks are supporting cues or fallbacks
- customer backgrounds vary
- skew, fold lines, and glare are common
- customer-added dark marks should be preserved for now, not separated

## Suggested stage order
1. Load input
2. Convert PDFs to raster
3. Detect outer border candidate
4. Optionally detect fiducials/reference marks
5. Compute warp transform
6. Rectify to top-down view
7. Scale to template dimensions
8. Crop to inside-most border edge
9. Save debug pack
10. Save report JSON

## Explicit non-goals for this phase
- contour classification
- handwriting separation
- note/text recognition
- vector export
- geometry fitting

## First coding target
Make one stored benchmark image process cleanly end-to-end with clear debug images.
