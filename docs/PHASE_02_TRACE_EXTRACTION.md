# Phase 02 — Trace Extraction

## Goal
Start isolating customer-added trace content from the cleaned raster while preserving closed shapes and useful note content.

## Inputs
- Phase 01 cleaned raster
- template metadata
- optional masks from printed CDS suppression

## Outputs
- trace candidate mask
- contour candidate images
- optional note candidate mask
- debug overlays

## Near-term approach
Keep this stage conservative.
Preserve ambiguous dark content rather than aggressively deleting it.
