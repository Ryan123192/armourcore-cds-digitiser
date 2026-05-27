# In-House Tracing Workflow

End-to-end procedure for staff using the ArmourCore vectoriser on the
new in-house CDS sheet.  Replaces 5-15 minute manual node-by-node
tracing with ~30 seconds of guided review per sheet.

## Workflow steps

### 1. Print
Print the in-house CDS sheet (see `IN_HOUSE_CDS_DESIGN_SPEC.md`) on
A3 white paper at the colour laser printer.

### 2. Trace
- Use a black sharpie / fineliner (~1 mm stroke) for pen jobs, OR
  4B-6B graphite pencil for pencil jobs
- Trace each tool as a SINGLE CONFIDENT CLOSED LOOP
- Don't double back over your own line - one stroke per shape
- Keep tool tracings at least 10 mm apart from each other

### 3. Scan
- Place sheet on the large-format scanner
- Settings: **400 dpi, Colour, PNG**
- No auto-rotate, no de-skew, no contrast adjustment
- Save as `<job_id>_<date>.png` in `\\fileshare\cds_inbox\`

### 4. Vectorise
At the office Windows PC:

```
> vectorise <path_to_scan.png>
```

(or double-click the desktop shortcut and pick the file)

The pipeline runs in 3-5 seconds and creates `<scan>_out/` next to
the scan containing:

- `vectors.svg`     - the production SVG (REAL paths only)
- `vectors_all.svg` - includes everything (for review)
- `overlay.png`     - visual of what was extracted
- `annotated.png`   - colour-coded by classification
- `report.json`     - machine-readable verdict + counts

### 5. Review

Open `overlay.png` in any image viewer.

- Looks correct? Open `vectors.svg` in Affinity and you're done.
- Missing a shape? Open the GUI review tool (see below).
- Extra junk? Open the GUI review tool and click-to-delete.

### 6. GUI review (when needed)

```
> vectorise-review <path_to_scan_out_folder>
```

Shows the rectified + overlay panels side-by-side, lets you:
- Click a vector to select it -> Delete key to remove
- Right-click on missing-shape area -> SAM-prompt to add
- File > Export when done -> writes `vectors_clean.svg`

## What "good" looks like

| Verdict      | Meaning                                                      |
|--------------|--------------------------------------------------------------|
| EXCELLENT    | Use the SVG directly, no review needed                       |
| GOOD         | Open overlay.png to confirm, usually 30s of glancing         |
| GRID_LEAK    | Run GUI review; will need 1-3 click-deletes                  |
| POOR         | Something off with the scan - re-scan or fall back to manual |

## Troubleshooting

| Symptom                            | Fix                                         |
|------------------------------------|---------------------------------------------|
| Phase 1 says "marker not found"    | Check all 4 red corner dots are visible    |
| Verdict POOR on every sheet        | Check scanner is in COLOUR mode             |
| Pencil too faint, missing shapes   | Use darker pencil (6B), press harder        |
| Shapes merging into one            | Trace with at least 10 mm gaps             |
| Doubled-up outline per shape       | Use one confident stroke, not back-and-forth |

## Time savings vs. manual tracing

| Method                       | Time per sheet | Notes                       |
|------------------------------|----------------|-----------------------------|
| Manual node-by-node tracing  | 5-15 min       | The current baseline        |
| Vectoriser + EXCELLENT output| ~5 seconds     | Pipeline does it all        |
| Vectoriser + GUI review      | ~30 seconds    | Maybe 1-3 clicks            |
| Vectoriser + manual fallback | 2-3 min        | When scan quality is poor   |

Typical mix on production-quality scans: 70% EXCELLENT, 25% review,
5% fallback.  Average per sheet ~20 seconds.
