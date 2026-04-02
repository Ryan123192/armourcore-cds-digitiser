# ArmourCore CDS Digitiser

Structured development repo for converting raw customer CDS submissions into usable 2D design inputs for ArmourCore.

## Current focus
This repo is intentionally phase-based.

**Immediate milestone**
- Detect the CDS design border
- Rectify perspective
- Scale to known physical dimensions
- Crop to the designable area
- Preserve debug outputs for comparison

**Next milestone**
- Remove the printed CDS system
- Suppress fold lines, glare, shadows, and nuisance artefacts
- Preserve customer-added markings for later trace extraction

## Supported input types
- Phone image files
- Single-page scanned PDFs

## Intended longer-term flow
Raw CDS photo/PDF -> rectified/scaled raster -> cleaned trace-preserved raster -> trace extraction -> vectorisation -> geometry correction -> Affinity cleanup -> production workflow

## Repository layout
- `docs/` project truth, phase briefs, conventions, and session notes
- `configs/` application defaults and template definitions
- `data/` raw inputs, template assets, and benchmark sets
- `outputs/` timestamped run folders and approved baselines
- `src/armourcore_cds/` application source code
- `tools/` developer helper scripts
- `tests/` targeted regression tests

## Development principles
1. Keep the pipeline deterministic first.
2. Preserve debug outputs at every major stage.
3. Optimise for repeatability before “smartness”.
4. Do not let vectorisation creep into Phase 1 work.
5. Keep layer naming aligned with the existing ArmourCore Affinity/Fusion workflow.

## Quick start
```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python -m armourcore_cds.cli --help
```

## First recommended coding target
Implement Phase 1 single-file processing for:
- one raw image or scanned PDF in
- one rectified, scaled result out
- one debug folder
- one `run_report.json`

## Notes
This scaffold is designed so the backend can later be wrapped into a desktop app without major restructuring.
