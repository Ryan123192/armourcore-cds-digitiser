# ArmourCore CDS Vectoriser - Status & Roadmap

## What it does today

Converts a scanned CDS tracing sheet into vector geometry (SVG) in
~5 seconds per page.  Replaces 5-15 minutes of manual node-by-node
tracing per sheet.

### Workflow
1. Staff trace tools on a printed CDS sheet using pen or pencil
2. Notes / dimensions / filter-out geometry get written in **blue pen**
3. Scan on the office or large-format scanner (200-400 dpi colour)
4. Open the GUI -> pick scan -> pick paper size -> click Vectorise
5. Save SVG -> open in Affinity / Illustrator / CAD

### Output quality (latest tests, May 2026)

| Test            | Result    | What worked                          |
|-----------------|-----------|--------------------------------------|
| V1 pen          | EXCELLENT | major+minor grid                     |
| V2 pen          | GOOD      | minor grid only                      |
| V3 pen (dots)   | EXCELLENT | dots grid - the recommended template |
| V1 pencil       | EXCELLENT |                                      |
| V2 pencil       | EXCELLENT |                                      |
| V3 pencil       | EXCELLENT |                                      |
| Pen + blue notes| EXCELLENT | blue notes erased, tools preserved   |
| Pen + scribbles | GOOD      | 10 tools traced, scribbles ignored   |

**Best results:** V3 (dots) template + black sharpie / 4B pencil.
Both ink types proven to hit production quality.

### Throughput estimate

- Manual tracing per sheet: ~10 minutes
- Vectoriser + review per sheet: ~30 seconds
- **20x faster** per sheet
- 100 sheets/day per workstation vs ~30 sheets/day manual

## What's in the box

- `vectorise_gui.py` -- GUI for daily use (Tkinter, no install)
- `vectorise_cli.py` -- command-line tool for batch processing
- `batch_inhouse.py` -- runs all template variants for QA
- 6 colour-coded diagnostic panels per run (corners | rectified |
  cleaned | trace | classified | final) so staff can spot bad scans
  before they reach Affinity

## Roadmap (12 months)

| Stage | Duration | Cost  | What changes                          |
|-------|----------|-------|---------------------------------------|
| 1 - NOW | shipped | 0    | Controlled in-house workflow         |
| 2 - Q1  | 3 mo    | 0    | SAM click-to-grab in review GUI      |
| 3 - Q2  | 3 mo    | $0   | Train a custom CNN from collected data |
| 4 - Q3-4| 6 mo    | small | End-to-end "any photo, any time"     |

### Why the staged approach
Stage 1 ships value today against a manageable input class (controlled
scans).  Each stage relaxes the input constraints while keeping the
previous stage as a safe fallback.  By end of internship the tool
should handle phone photos taken anywhere with any lighting.

## Data flywheel

Every sheet processed via the review GUI saves a `(raw scan, final
vectors)` pair.  After 3-6 months of normal use this becomes a
**labelled training set worth thousands of dollars** if outsourced.
Use it to train Stage 3's CNN at zero acquisition cost.

## Risks honestly stated

1. **Off-template inputs** (free-form paper, no markers) will fail
   the rectifier today.  Stage 2 SAM integration mitigates this.
2. **Pencil under bad lighting** can be too faint -- use 6B for safety.
3. **Customer-shift double lines** in pencil produce two outlines per
   shape.  Stage 4 collapses them automatically.

## Ask

- Approve printing of V3 dots template at A2/A1/A0 sizes for production
- Allocate ~1 day/week during internship for continued development
- Schedule mid-internship check-in to demo Stage 2 (SAM integration)
