# ArmourCore CDS Digitiser — AI Starter Brief

Use this file as the opening context in a fresh AI chat before asking for code help.

## 1. Project overview

I am developing an internal ArmourCore tool called **ArmourCore CDS Digitiser**.

Its purpose is to convert a **customer-submitted CDS image or scanned PDF** into a **clean, properly scaled 2D traced design input** that can later be taken into Affinity Designer, refined if needed, and then eventually moved into Fusion 360 / downstream insert-building workflows.

This tool is intended to automate the early stages of the current manual workflow.

### Desired long-term pipeline

Raw customer CDS image/PDF  
→ detect CDS boundary  
→ correct perspective  
→ scale accurately  
→ remove printed CDS system and suppress artefacts  
→ isolate customer-added traces/markings  
→ vectorise into clean outlines  
→ organise vectors onto useful layers  
→ export DXF for Affinity / downstream workflow

## 2. Current development priority

The repo is intentionally broader than Phase 1, but **current code focus is only on Phase 1**.

### Phase 1 current objective

Take a **single raw customer image or single-page scanned PDF** and output a:

- correctly rectified raster
- correctly scaled raster
- cropped designable CDS area
- debug image sequence showing what happened

Then after that is stable:

- remove printed CDS border/grid system
- suppress fold lines, glare, shadows, stains, background junk
- preserve customer-added dark markings inside the border

### Important

Do **not** jump ahead into vectorisation, geometry inference, auto-arrangement, or UI work unless explicitly asked.

## 3. Real-world workflow context

ArmourCore sends customers a **Calibrated Design Sheet (CDS)**.

Customers place tools on it, trace around them, and send back:

- phone photos most of the time
- scanned PDFs sometimes
- rarely already-vectorised files, which are out of scope for this tool

The current manual process already has some semi-working perspective correction attempts, but the main bottleneck is still converting messy customer captures into something clean enough for tracing/vectorisation.

## 4. Supported CDS templates right now

The repo currently reserves configs for:

- **Regular CDS:** 500 × 600 mm
- **X-Large CDS:** 500 × 900 mm
- **Colour test mini grid:** 260 × 350 mm

The **primary geometry truth** is the **inside-most edge of the thick outer border**.

Border is the primary scaling/reference source. Fiducials/reference marks may later be used as checks or fallbacks.

## 5. Input assumptions

Current development assumptions:

- Start with **one file in, one run out**
- Support:
  - phone image files
  - single-page scanned PDFs
- Most real customer files are images
- Backgrounds vary
- Angles vary
- Elevation varies
- Common real-world issues:
  - perspective skew
  - fold lines
  - glare
- Customers may write notes, heights, labels, dimensions, or comments on the CDS
- For now, customer-added dark content should generally be **preserved**, not aggressively separated

## 6. Output assumptions

### Immediate output target
A high-quality raster that is:

- top-down corrected
- physically scaled
- cropped to the designable area
- suitable for later tracing work

### Later output target
DXF output with scale preserved and layers that match ArmourCore conventions.

## 7. Layer naming conventions to respect later

This workflow ultimately needs to align with existing ArmourCore Affinity / Fusion conventions.

Important layer names include:

### Core layers
- `Outside Edge`
- `Badge Pockets`
- `Laser Labels`
- `Notes`
- `Reference & Scaling`
- `Template Image`

### Depth layers
- `25mm`
- `30mm`
- `35mm`
- `40mm`
- other `<number>mm` layers as needed

### Through-cut layer
- `THRU`

Do not invent a completely different naming system unless explicitly asked.

## 8. Development philosophy

This project should be developed like ArmourCore Insert Builder:

- centralised repo docs
- structured file/folder system
- debug outputs at every major stage
- resumable milestone-based progress
- deterministic classical CV first
- vectorisation later
- geometry rules later
- AI only later if truly needed

### Important philosophy

Do **not** start with generic “AI solve everything” tracing ideas.

The preferred order is:

1. strong classical image-processing baseline
2. rough contour/vector output
3. rule-based geometry improvements
4. only later, optional AI support if still needed

## 9. Repo navigation

When helping with code, inspect these repo areas first:

### High-level docs
- `README.md`
- `docs/MASTER_BRIEF.md`
- `docs/PHASE_01_RECTIFY_AND_CLEAN.md`
- `docs/LAYER_NAMING_CONVENTIONS.md`
- `docs/TEMPLATE_REGISTRY.md`

### Template configs
- `configs/templates/cds_regular_500x600.yaml`
- `configs/templates/cds_xlarge_500x900.yaml`
- `configs/templates/cds_colour_test_260x350.yaml`

### Core code areas
- `src/armourcore_cds/phase1/`
- `src/armourcore_cds/templates/`
- `src/armourcore_cds/io/`
- `src/armourcore_cds/utils/`

### Output/debug expectations
- `outputs/runs/<timestamped_run>/`

## 10. Current repo structure intent

The repo is organised into phases:

- `phase1` = rectify / scale / crop / clean
- `phase2` = trace extraction
- `phase3` = vectorisation / DXF export
- `phase4` = geometry intelligence

There are also:

- template configs
- docs
- test scaffolding
- tools for running and benchmarking

## 11. What good help looks like in this project

When responding, please:

- stay tightly scoped to the requested phase/task
- preserve repo structure
- avoid giant monolithic scripts
- suggest modular code
- include where files should live
- explain what changed and why
- respect future extensibility
- keep debug outputs in mind
- do not silently change naming conventions or architecture

## 12. Current milestone status

At this stage, assume:

- repo scaffold and docs exist
- actual Phase 1 CV logic is still early / incomplete
- old prototype scripts exist only as references and should **not** be treated as architecture truth
- current immediate target is:

### Target
**single input → border detection → perspective correction → scale normalisation → crop → debug outputs**

Not vectorisation yet.

## 13. How to answer in this chat

When helping develop code for this repo:

1. briefly state what file(s) should be created or edited
2. explain the approach
3. provide code in a repo-friendly way
4. preserve modularity
5. mention assumptions
6. avoid scope creep

## 14. Example prompt to ask after pasting this brief

After pasting the above, ask something like:

> Please help me implement the first real Phase 1 pass for this repo. I want to load a single image or single-page PDF, detect the CDS outer border, rectify perspective, scale using the inside-most edge of the thick border, crop to the designable area, and save debug images plus a run report. Please tell me which files to create or edit, and keep the solution aligned to the existing repo structure.

## 15. Quick distinction

Use them like this:

- **`MASTER_BRIEF.md`** = long-term repo truth for humans inside the project
- **starter brief** = portable AI context pack for fresh chats
- **phase briefs** = deeper technical context for one development stage
