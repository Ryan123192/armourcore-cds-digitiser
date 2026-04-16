# ArmourCore CDS Digitiser — Phase 01 Carryover Brief (Part 2 Update)

## Purpose of this brief

This is an updated carryover brief for the next coding chat.

Use this together with the earlier Phase 01 carryover brief, but treat **this document as the latest truth** for:
- current repo behaviour
- what has been fixed already
- what is still broken
- what the next chat must do differently

The next chat should **read the repo deeply first**, then diagnose from the actual codebase rather than continuing patch-by-patch guesses.

---

## High-level project reminder

Repo: **ArmourCore CDS Digitiser**

Main long-term goal:
- convert customer CDS photos / scanned PDFs into clean, scaled, top-down raster inputs for later tracing, cleanup, vectorisation, and DXF workflows

Current real scope:
- **Phase 01 only**
- detect CDS outer border
- rectify perspective
- scale to known physical size
- save debug outputs and run report

Current intended Phase 01 workflow:
- **detect → rectify → scale**
- no secondary crop pass
- detected thick border is geometry truth

---

## What the package is currently capable of

As of this chat, the package can successfully do the following on valid **single-sheet black-border regular CDS** cases:

- load images and PDFs
- load template config
- detect a border candidate
- rectify perspective
- scale to target DPI / physical size
- write debug images
- write `run_report.json`

It is currently working reasonably well on multiple real black-border regular-sheet images, including:
- `Test02`
- `Test03` (before later orange-mode contamination)
- `Test04`
- `Test05`

Observed success characteristics:
- correct border quad selected
- high confidence
- scaled output dimensions correct
- overlap/occlusion by objects on border can still be tolerated in some cases

So the package is **not generally broken**.

The current broken/problem area is specifically:
- **colour border detection for the 260 × 350 colour test template**
- and interaction between newer colour scoring logic and the older black-border logic

---

## Key decisions already made

### 1. Single-border workflow
Still correct and still intended:
- detect one border
- rectify once
- scale once
- no secondary crop stage

### 2. Border truth
Current truth remains:
- the **inside edge of the thick outer border**

Not:
- calibration envelope
- internal user geometry
- second crop result

### 3. Calibration markers
They should be:
- support / confirmation cues
- scoring hints
- corner-location hints

They should **not** fully define the crop.

### 4. Low-confidence fallback
Still desired:
- if the detector is uncertain, choose best candidate with honest low-confidence reporting
- do not hard-fail immediately when a plausible candidate exists

---

## What was fixed successfully in this chat

### A. API wiring mismatch
There was a real mismatch where:
- detector returned `ordered_corners_xy`
- downstream code still assumed NumPy arrays / old field names

This was fixed enough for the black-border Phase 01 path to run again.

### B. Image-frame false positives
At one point the detector started selecting the literal photo boundary:
- `(0,0) → (w-1,0) → (w-1,h-1) → (0,h-1)`

This was diagnosed and corrected by:
- rejecting frame-like candidates
- clearing image borders before contour extraction
- changing area/extent from “bigger is always better” to a more plausible occupancy scoring

That was a real improvement.

### C. Hard-case misunderstanding: `Test01`
A major discovery:
- `Test01` is actually **two CDS templates taped together**

So it is **not** a valid single-template benchmark for the current detector. That case should be deferred to a future multi-sheet/composite mode.

### D. Black-border regular-sheet path
After the above fixes, these regular-sheet images worked well:
- `Test02`
- `Test04`
- `Test05`

This confirms the black-border baseline is usable.

---

## What was attempted next

After black-border regular sheets were working, development moved into:
- colour support
- colour-test benchmarking
- orange-border experimentation

This is where the current issues emerged.

---

## Colour test setup context

A special colour test sheet exists:
- A1 page containing four small CDS variants
- each small CDS is **350 × 260 mm**
- arranged in quadrants

Colour families:

- **Set 1 — Magenta**
  - outer border: `#C2185B`
  - major grid: `#E91E63`
  - standard grid: `#F8BBD0`

- **Set 2 — Cyan / teal**
  - outer border: `#007C91`
  - major grid: `#00ACC1`
  - standard grid: `#B2EBF2`

- **Set 3 — Orange**
  - outer border: `#D35400`
  - major grid: `#F39C12`
  - standard grid: `#FAD7A0`

- **Set 4 — Green**
  - outer border: `#2E7D32`
  - major grid: `#66BB6A`
  - standard grid: `#C8E6C9`

The corresponding PDF image confirmed the four-up arrangement:
- top-left = magenta
- top-right = cyan
- bottom-left = orange
- bottom-right = green

---

## What was learned from colour testing

### 1. The first batch benchmark harness was not trustworthy
A batch harness was built to test the colour quadrants, but it used rough ROI/quadrant searching rather than first rectifying the full A1 sheet.

Result:
- detections were inconsistent
- per-colour rankings were not trustworthy
- some apparent “successes” were clearly not correct quadrant detections
- summary metrics conflicted with visual inspection

So the numeric colour benchmark from that harness should **not** be treated as decision-grade evidence.

### 2. Visual observation still matters here
From actual photos:
- cyan and green often looked weak, greyed-out, or visually lost
- magenta and orange looked stronger
- orange looked especially promising and is also closer to ArmourCore brand fit

Practical decision reached:
- orange is the most sensible working colour candidate to keep pushing for now

But:
- this was **not** properly proven by the benchmark harness
- it was a pragmatic visual + design/brand decision

---

## Main current problem

The package now has **partial colour support**, but it is not behaving reliably.

### The exact current issue:
When trying to detect the **orange 350 × 260 CDS**:

- black mode tends to act like a page-edge / broad dark-boundary detector
- orange mode sometimes activates, but often still does not lock onto the **thick outer orange border**
- instead it may:
  - snap to the outer photographed page
  - snap to internal major grid lines
  - snap to internal rectangular geometry closer to the artwork bounding area
  - or fail to produce a useful candidate

### Most important recent evidence
A more isolated orange test image (`ColourTest3.jpg`) was run.

Results:
- **black mode** selected a very large page-like candidate
- **orange mode** selected a smaller but still wrong candidate
- the selected orange candidate aligned more with internal rectangular structure than the true thick outer border

That means:
- orange is being picked up **somewhat**
- but the detector is not distinguishing:
  - thick outer orange border
  - internal orange major grid / rectangle structure

This is currently the real blocker.

---

## Important deeper discovery about the printed orange

A close inspection of the photographed orange border showed:

- the real printed border is **not** a vivid digital orange
- it looks more like a muted **terracotta / reddish-brown**
- less saturated than the intended design colour
- darker and redder than clean `#D35400`
- thin border line plus surrounding white paper dilutes colour sampling

Implication:
- any colour matcher based too tightly on the design hex is likely wrong
- the real-world printed/captured colour space must be used instead

This is likely one real contributor to the poor orange score behaviour.

---

## Another important discovery

When the detector was run in orange mode on an existing black regular-sheet image (`Test03`), it made the result worse:
- border drifted toward the outside of the calibration markers / envelope
- confidence dropped
- it no longer behaved as cleanly as the earlier black-border baseline

Implication:
- orange-mode logic is contaminating the black-border path if used incorrectly
- colour mode should probably be **template-specific**
- regular black sheets should stay on black mode
- colour test sheet should stay on orange mode while being debugged

---

## What the code appears to be doing wrong conceptually

The next chat should verify all of this directly in the repo, but these are the likely root-cause buckets:

### 1. Candidate generation is still too generic
The system can generate candidates from:
- grayscale edges
- dark border signals
- some newer colour-related logic

But for the orange sheet it still seems willing to promote:
- page edges
- internal grid rectangles
- artwork-area rectangles

instead of the true thick outer border.

### 2. Orange support is too weak at the candidate level
Colour scoring may exist, but orange is still not dominating candidate generation enough.

It looks like:
- colour is helping score candidates
- but not reliably generating the right orange-border candidates
- and/or the wrong orange-like rectangular structures exist in the candidate pool

### 3. Thick-border distinction is not strong enough
For the orange small template, the detector seems to confuse:
- thick outer border
with
- internal orange grid or major rectangle structure

This suggests the scoring needs to differentiate:
- line thickness
- line continuity/coverage
- outermost plausible position
- relationship to fiducials/calibration markers

much more strongly.

### 4. Page-scale penalties are still not correct for colour mode
At multiple points, large page-like candidates have still been too competitive.

### 5. Auto-mode is not yet trustworthy
Auto-detection between black/orange is not the right focus right now.
It should be deferred until:
- black mode is stable
- orange mode is stable

---

## What should not be done next

The next chat should **not** keep guessing through tiny patch cycles without deeply reading the repo.

Specifically:
- do not keep rebalancing weights blindly
- do not keep patching around symptoms without reading the live code
- do not trust the earlier colour benchmark outputs as final truth
- do not keep using orange mode on black regular-sheet templates when validating black-border baseline
- do not treat `Test01` as a valid current benchmark (it is a double-sheet composite)

---

## What the next chat must do

### Main task
Do a **deep read of the updated repo** and identify the real root cause of orange-border failure.

The next chat should inspect at minimum:

- `src/armourcore_cds/phase1/boundary_detection.py`
- `src/armourcore_cds/phase1/pipeline.py`
- `src/armourcore_cds/phase1/rectify.py`
- `src/armourcore_cds/templates/models.py`
- template YAML files, especially:
  - `cds_regular_500x600`
  - `cds_xlarge_500x900`
  - `cds_colour_test_260x350`
- app config defaults
- any recent colour-mode logic
- any candidate-source / diagnostics fields in run reports

### The next chat should explicitly answer:
1. How are candidates currently generated?
2. Which candidate sources are producing the wrong orange detections?
3. How is orange currently scored?
4. Is the orange detection logic based on a realistic colour window for the real printed border?
5. Why is the detector preferring internal rectangular structure instead of the thick outer border?
6. How should candidate generation / scoring be restructured so this works robustly?

---

## Likely desired end-state for this phase

For now, the desired practical behaviour is:

### Black-border legacy sheets
- continue working as they already do
- black mode / baseline must remain stable

### Orange colour-test small template
- run in **forced orange mode**
- detect the **thick outer orange border**
- do **not** snap to:
  - page boundary
  - calibration envelope outside the border
  - major grid rectangle
  - internal artwork bounds

### Auto mode
- leave for later
- only revisit after black and orange modes work properly on their own

---

## Strong suggestions for the next implementation approach

These are suggestions only — the next chat should verify from code before patching.

### A. Separate modes properly
- `black` mode should be maintained for old templates
- `orange` mode should be maintained for colour test / future orange templates
- `auto` should not be the current focus

### B. Orange needs better candidate generation
Possible solution direction:
- generate candidates from a warm terracotta/orange mask in HSV/Lab
- not just from grayscale edges
- merge and deduplicate with edge candidates

### C. Score thickness / outermost-ness harder
The thick outer border should beat internal major grid lines.
So scoring probably needs stronger terms for:
- thicker border band
- outermost plausible rectangle
- corner marker relationship
- internal-marker-inside penalties if candidate sits too far inward

### D. Tune orange to the real printed border
Do **not** rely on ideal screen orange.
The real photographed border is more like muted terracotta / red-brown.

### E. Add better debug outputs before patching further
The next chat may want to add actual diagnostic outputs such as:
- orange mask preview
- black mask preview
- candidate-source overlays
- top-N candidate summary with source + score breakdown

That would probably reveal the root cause faster than another blind scoring patch.

---

## Useful recent evidence to carry forward

### Working black-border regular examples
- `Test02`
- `Test04`
- `Test05`
- `Test03` earlier, before later orange-mode contamination

### Invalid benchmark case
- `Test01` = two sheets taped together

### Current colour-test key image
- `ColourTest3.jpg`
  - isolated orange small template
  - current best image for debugging orange-border logic

### Orange mode failure pattern on `ColourTest3`
- detector snaps to internal rectangular structure / major grid region instead of thick outer border

### Black mode failure pattern on `ColourTest3`
- detector snaps to broader page-like boundary

### Printed orange reality
- muted terracotta / reddish-brown
- not vivid design orange

---

## What the next chat should ideally produce

1. A concise diagnosis of the real root cause from the repo itself  
2. A more principled fix plan  
3. One clean patch / diff / repair script  
4. A small, disciplined validation sequence:
   - black regular-sheet check
   - orange small-template check
   - maybe one synthetic/debug crop check if needed

---

## Final summary in one sentence

**The package’s black-border Phase 01 path is mostly working, but the current orange-border logic for the 350 × 260 colour test template is still selecting the wrong rectangular structures because candidate generation/scoring is not yet distinguishing the true thick outer orange border from page edges and internal orange grid geometry.**
