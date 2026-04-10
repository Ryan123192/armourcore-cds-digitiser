# ArmourCore CDS Digitiser — Phase 01 Carryover Brief

## Purpose of this brief

This is a **full carryover brief for a new AI chat**. The current chat will not be continued.

The next chat should assume:
- the repo already exists
- multiple partial patches have already been applied
- the repo’s current state is the **source of truth**
- the next task is to **inspect the current repo zip directly**, identify all mismatches introduced by the recent detector/API changes, and produce **one clean Python apply-fix script** rather than continuing piecemeal edits

---

## Project context

Repo: **ArmourCore CDS Digitiser**

Goal of the tool:
- convert a customer-submitted CDS photo or scanned PDF into a clean, scaled, top-down raster suitable for later Affinity tracing / vectorisation / DXF workflows

Long-term pipeline:
- load image or scanned PDF
- detect CDS border
- rectify perspective
- scale correctly
- later: colour-aware filtering, artefact suppression, grid removal, trace isolation, vectorisation, geometry rules

This chat focused only on the **first working Phase 01 pass**.

---

## Phase 01 scope being implemented

Desired immediate workflow:

**single input → border detection → perspective correction → scale normalisation → debug outputs + run report**

Current intended architecture after the latest decisions:
- **detect**
- **rectify**
- **scale**
- no secondary crop pass

Important:
- We explicitly decided to **remove the secondary crop stage entirely**
- We do **not** want a two-step crop / border-refinement workflow
- We want the **first detected thick border quad** to be trusted as the geometry truth
- Calibration squares are support/confirmation only, not the crop truth

---

## Key design decisions agreed in this chat

### 1. Border truth
Primary geometry truth should be:
- the **inside edge of the thick black CDS border**

Not:
- the outer envelope of the red calibration squares
- not the customer-drawn insert rectangle
- not any second crop pass after rectification

### 2. Calibration squares
Calibration squares should be used only as:
- support
- confirmation
- scoring boost
- plausibility checking around corners

They should **not** define the final crop rectangle directly.

### 3. Crop policy
We explicitly decided:
- **no secondary crop at all**
- trust the first detected thick-border quad
- later, colour information will make candidate detection much more robust anyway

### 4. Hard-case behaviour
For difficult cases:
- detector should **not hard-fail immediately**
- instead it should return the **best candidate available**
- and mark it as **low confidence** in report/debug

This was explicitly chosen over fail-fast behaviour.

---

## What has already been achieved

### Working successes
The pipeline now works on at least some real camera images that resemble actual workflow inputs:
- real perspective distortion
- folds
- variable lighting
- some shadow and contrast mess

At least one regular-sheet test case and one x-large-sheet clean case successfully:
- loaded
- detected a border quad
- rectified perspective very well
- scaled correctly
- produced outputs usable in Affinity

There was a key milestone where the output, once placed in Affinity and set to the known physical width, aligned correctly. That strongly suggests:
- scale maths is basically correct
- any mismatch seen in Affinity is more about DPI interpretation/placement than geometry itself

### Important qualitative result
The rectification quality is already roughly equivalent to current manual workflow on some real files. That is a meaningful success.

---

## Template orientation decisions

We discovered the original template width/height assumptions were wrong for real usage.

They were corrected to landscape-oriented design areas where appropriate.

Expected template design-area orientations now intended to be:

- `cds_regular_500x600` → **600 x 500 mm**
- `cds_xlarge_500x900` → **900 x 500 mm**
- `cds_colour_test_260x350` → intended to be **350 x 260 mm** unless repo context proves otherwise

Note:
- Test expectations had originally assumed portrait values and needed updating afterward

---

## What changed conceptually in the detector

The detector started as mostly contour-first and brittle.

We then discussed and moved toward:
- single detected border quad
- border evidence over pure contour dominance
- calibration squares as support only
- low-confidence fallback rather than hard failure

### Intended detector philosophy now
Candidate scoring should combine:
- aspect ratio closeness to template
- area sanity
- side darkness
- side thickness consistency
- calibration-square support near corners
- penalty for wrapping around the outer calibration envelope
- tolerance for partial/interrupted border evidence

### Why
Because current real-world failures come from:
- tape over border
- folds breaking continuity
- one corner marker clipped
- shadows/glare
- border partly occluded
- detector latching to the outside of calibration squares rather than the thick black border

---

## What was observed in testing

### Clean / easier x-large case
A clean x-large case eventually succeeded after orientation fixes.
Observed behaviour:
- initial border detection could still sit around the outside of calibration squares
- but once inner-crop logic existed, it correctly found the thick black border
- after removing the secondary crop, the desired future direction became: detect the right border directly, once

### Regular-sheet Test 02
This became the main “good” case.

Observed:
- border detection was good
- rectification was strong
- scaled output aligned well in Affinity
- after removing secondary crop, the run report used:
  - `crop_mode: none_use_detected_border`

This is the current best reference success case.

### Regular-sheet Test 03
A good/easy file where:
- a tiny bit of a calibration square was clipped
- detector still completed
- but in some areas it drifted toward the outside of the red calibration squares rather than the thick black border

This means:
- detector is close
- but still not robust enough when corner evidence is incomplete

### X-large Test 01
This is the main hard failure case.

Characteristics:
- obstacles/tape over black border
- folds
- shadowy / contrasty lighting
- interrupted border evidence

Current issue:
- old detector hard-failed
- newer detector direction should continue with a low-confidence best candidate instead

---

## Important run-report observations

### Good regular test (`Test02.jpg`)
Run report showed:
- success
- `design_area_mm` = `600 x 500`
- `crop_mode` = `none_use_detected_border`
- scaling to `9449 x 7874 px` at ~400 dpi

This is consistent with correct scaling for 600 x 500 mm at 400 dpi.

### Slightly ambiguous regular test (`Test03.jpeg`)
Run report showed:
- success
- same `600 x 500` design area
- still successful rectification/scaling
- but visual debug indicated the border candidate could still drift toward the calibration envelope

### Earlier clean x-large case
Run report showed:
- `900 x 500` design area
- scaled output around `14173 x 7874 px` at ~400 dpi

This is consistent with correct scaling for 900 x 500 mm at 400 dpi.

---

## Major implementation changes already attempted

Several patch scripts / edits were already applied over the life of the chat.

The exact current repo state should be treated as authoritative.

### Broad categories of changes already introduced
- template orientation changes
- removal of secondary crop in pipeline
- new detector returning different field names/types
- attempt to move detector toward low-confidence best-candidate behaviour

### Critical issue now
The repo almost certainly contains **API mismatch propagation** across modules.

Example mismatches that definitely occurred:
- old code expected `ordered_corners`
- newer detector returns `ordered_corners_xy`
- old code expected NumPy arrays
- newer detector may return Python lists
- old pipeline/debug code expected preview images like `preview_edges`
- newer detector no longer exposes those fields
- helper functions such as polygon drawing and rectification still assumed NumPy array inputs and used `.astype(...)` directly

This created a chain of wiring errors.

---

## Specific mismatches that were seen

These concrete mismatches appeared during testing and are highly likely still relevant somewhere in the repo:

### 1. Field name mismatch
Old:
- `border.ordered_corners`

New detector:
- `border.ordered_corners_xy`

### 2. Type mismatch
New detector appears to return:
- `ordered_corners_xy` as a **plain Python list**

Downstream code often assumed:
- NumPy array
- `.astype(...)`
- `.shape`

This caused failures in:
- tests
- `draw_polygon`
- rectification helpers
- possibly other geometry utilities

### 3. Old debug fields missing
Old pipeline expected things like:
- `border.preview_edges`
- perhaps `border.preview_mask` / similar

New detector result no longer provided those.

### 4. Tests became stale
Tests that had to be updated manually included:
- template registry tests expecting old portrait dimensions
- boundary detection tests expecting:
  - `ordered_corners`
  - NumPy array `.shape`
instead of:
  - `ordered_corners_xy`
  - list-based assertions

---

## What the next chat must do

### Main task
The next chat should:
1. inspect the **current repo zip**
2. identify **all places** where old detector assumptions are still present
3. produce **one Python apply-fix script** that updates the repo cleanly in one pass

Do **not** continue with piecemeal manual fixes in chat.

### Goal of the repair pass
Bring the repo to a clean internally consistent state where:
- templates have the correct orientation
- workflow is:
  - detect
  - rectify
  - scale
- no secondary crop exists
- detector can return a best candidate with low-confidence status
- downstream modules accept the new detector result format
- tests reflect the new API and orientation expectations
- one hard file (`Test01`) can at least run to completion more often, rather than dying immediately

---

## What the next AI should inspect first

Please inspect the actual current repo contents before proposing fixes.

Likely key files to inspect:

### Configs
- `configs/templates/cds_regular_500x600.yaml`
- `configs/templates/cds_xlarge_500x900.yaml`
- `configs/templates/cds_colour_test_260x350.yaml`

### Core Phase 01 files
- `src/armourcore_cds/phase1/boundary_detection.py`
- `src/armourcore_cds/phase1/pipeline.py`
- `src/armourcore_cds/phase1/rectify.py`
- `src/armourcore_cds/phase1/scaling.py`
- `src/armourcore_cds/phase1/crop.py` (to confirm it is no longer used in default workflow)

### Utilities
- `src/armourcore_cds/utils/image_ops.py`
- any debug/report helper modules
- any type/model definitions used by border detection results

### Tests
- `tests/test_template_registry.py`
- `tests/test_boundary_detection.py`
- `tests/test_scaling.py`

### Also inspect
- `cli.py`
- template model / registry code
- run report generation structure

---

## Behaviour the next fix should aim for

### Detection
- prefer thick black border over calibration-envelope rectangle
- allow partial/interrupted border evidence
- support missing/clipped calibration corners
- continue with best candidate when confidence is low
- expose confidence in run report/debug

### Pipeline
- single-border workflow only
- no secondary crop logic in default path
- no references to removed debug fields
- no assumptions that corners are NumPy arrays
- immediately coerce corner collections to NumPy arrays where needed downstream

### Debug/reporting
Useful if available:
- chosen quad
- candidate count
- score/confidence
- low-confidence flag
- maybe reason labels if easy:
  - likely thick-border match
  - likely calibration-envelope drift
  - low-evidence fallback

### Tests
- update to current design-area orientations
- update to new border result API
- ensure tests do not assume NumPy arrays unless the public API is intentionally changed back to arrays

---

## Important style/process request for the next chat

The user explicitly wants:
- smaller test batches
- less repeated explanation
- more direct actionable steps
- repo-wide fix in one pass, not endless incremental wiring fixes

So the next chat should:
1. inspect repo zip
2. state files to fix
3. produce one repair script
4. give one compact test sequence:
   - test imports/package path
   - template registry test
   - scaling test
   - boundary detection test
   - one real image run

---

## Recommended immediate objective for next chat

**Scan the uploaded current repo zip, find all remaining old detector API assumptions, and generate one Python repair script that restores a clean Phase 01 pipeline with:**

- corrected template orientations
- single-border workflow
- low-confidence best-candidate detector behaviour
- consistent downstream handling of `ordered_corners_xy`
- no stale debug-field references
- updated tests

---

## Suggested opening prompt for the next chat

I am uploading the **current state** of the ArmourCore CDS Digitiser repo.  
Please inspect the repo directly and do not rely on earlier patch assumptions.  

We previously changed Phase 01 to:
- use landscape-oriented template dimensions where appropriate
- use a **single-border workflow**: detect → rectify → scale
- remove secondary crop entirely
- move the border detector toward best-candidate / low-confidence behaviour
- return `ordered_corners_xy`, which appears to now be a plain Python list

The repo is now inconsistent because several downstream files still assume the old detector API (`ordered_corners`, NumPy arrays, old preview debug fields, etc.).

Please:
1. inspect the current repo state
2. list the exact files that still need repairing
3. create **one Python apply-fix script** that updates the repo cleanly
4. update tests to match the corrected public API and template orientations
5. give me a short test sequence in small batches

The key hard case to keep in mind is `Test01.jpeg` on the x-large template:
- partly occluded border
- tape/folds/shadows
- previous contour-first detection failed
- we want best-candidate with low-confidence reporting rather than hard fail

