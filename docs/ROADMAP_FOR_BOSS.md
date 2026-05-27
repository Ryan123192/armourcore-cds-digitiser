# ArmourCore CDS Vectoriser - Roadmap

## TL;DR

A working in-house CDS vectoriser today.  A path to a "drop any photo,
get perfect vectors" product over the next 12 months, building on the
labelled data collected from day-to-day use.

## Stage 1 - SHIPPING NOW (this week)

**Scope**: in-house tracing using a controlled CDS sheet + scanner.

**What works**:
- Phase 1 rectification (corner-marker based): 100% reliable
- Phase 2 cleaning: removes grid, text, lighting artefacts
- Phase 3 vectorisation: smooth Bezier-curve closed loops
- Per-route adaptive pipeline: pen / dark-border-pen / pencil
- Self-diagnostic classifier: flags each path REAL / GRID / TEXT / SLIVER / NOISE
- One-shot CLI + simple review GUI
- SVG and DXF export

**What it replaces**: 5-15 minutes of manual node-by-node tracing
becomes ~30 seconds of glance-and-approve.

**Throughput estimate**: 100-150 sheets/day per workstation vs. ~30
sheets/day manual.

**Limitations**:
- Phone photos from variable lighting may fail (use the in-house scan
  workflow instead)
- Customer pencil shifts that produce doubled parallel lines get traced
  as two outlines per shape (review GUI lets staff merge in 1 click)

---

## Stage 2 - Q1 of overseas internship (~3 months in)

**Goal**: extend to handle any half-decent photo scan, not just the
in-house controlled sheet.

### 2a. Foundation model integration
Add Meta's SAM (Segment Anything) as a click-to-grab primitive in the
review GUI.  Even on bad inputs, staff can click on a missing shape
and SAM grabs it as a polygon.

**Effort**: ~2 weeks evenings/weekends.
**Impact**: review tool becomes useful for ANY input, not just
controlled scans.

### 2b. Training-data flywheel
Every sheet processed via the review tool saves a (raw image, final
vectors) pair to a labelled dataset.  Zero extra work for staff.

**Result by month 6**: ~500-1000 labelled examples ready for training.

---

## Stage 3 - Q2 internship (months 4-6)

**Goal**: train a CNN that handles the cases the rule-based pipeline
can't.

### 3a. Train a U-Net semantic segmenter
Per-pixel classification: paper / grid / pen / pencil / text.
Train on the collected dataset on a free Colab GPU.

**Effort**: ~3 weeks of weekend work.
**Impact**: lighting-invariant, ink-type-invariant pixel-level
understanding.  No more "orange grid = pen ink" failures.

### 3b. Hybrid pipeline
- Run rule-based pipeline (fast, free)
- If verdict isn't EXCELLENT, escalate to CNN
- Best of both: cheap on easy cases, robust on hard ones

---

## Stage 4 - Q3-Q4 internship (months 7-12)

**Goal**: end-to-end "drop any photo, get perfect vectors" with
geometric / symmetry polishing.

### 4a. Geometric polishing
After vectorisation, apply rules:
- Snap straight segments to true horizontal/vertical
- Detect circles / arcs and replace polyline with true circle / arc
- Detect symmetry axes and mirror to clean up asymmetric tracings
- Smooth corners with controlled radius based on tool type

### 4b. Multi-tool batch mode
- Camera mounted over a desk
- Trace -> press button -> next sheet
- Continuous processing pipeline
- Workshop floor can vectorise without a dedicated computer station

### 4c. Customer-facing capture
- Mobile app: customer photographs their tracing
- Uploads to a queue
- Reviewer at the office approves in 30s
- Customer gets SVG/DXF back same day

---

## Cost / time estimate by stage

| Stage | Duration            | Effort       | Investment needed         |
|-------|---------------------|--------------|---------------------------|
| 1     | NOW                 | Already done | None                      |
| 2     | 3 months            | Part-time    | 0                         |
| 3     | 3 months            | Part-time    | 0 (free Colab GPU)        |
| 4     | 6 months            | Part-time    | Possibly small ML server  |

Total: working in-house tool day 1, full automation by end of 13-month
internship.

## What this means for ArmourCore

**Today**: replaces a manual bottleneck with a one-click tool.  Frees
~6 hours/day of staff time across the design team.

**Year 1**: customers can capture their own tracings on a phone.
ArmourCore captures CDS data 10x more cheaply than competitors who
require sales-rep visits and manual digitisation.

**Year 2**: full self-service.  Customer designs and prototypes their
own protective covers with the app; ArmourCore quotes and produces.
Lead time from inquiry to quote drops from days to minutes.
