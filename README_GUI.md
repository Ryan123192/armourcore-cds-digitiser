# ArmourCore CDS Vectoriser — GUI

A thin PyQt6 wrapper around the existing image-processing pipeline.

## Quick start (Windows)

1. Make sure Python 3.10+ is installed.
2. Install PyQt6 once:

   ```
   pip install PyQt6
   ```

3. Double-click `Launch_ArmourCore_Vectoriser.bat`.

If the window opens, you're set.

## Quick start (any OS)

```
python Launch_ArmourCore_Vectoriser.py
```

## Using the GUI

1. Click **Browse...** and pick a CDS photo, scan, or PDF.
2. The template type is auto-detected from the filename, but you can
   override it in the dropdown.
3. Click **Start Vectorising**.
4. Watch the stage label, progress bar, and live preview update.
5. When it finishes, the success dialog shows tool / loop counts and
   the output folder path.
6. Click **Open Output Folder** to inspect:
   - `01_phase1_rectified.png`
   - `02_phase2_cleaned.png`
   - `03_phase3_tools_detected.png`
   - `03_phase3_gap_filled.png`
   - `03_phase3_vector_preview.png`
   - `<input>_with_image.svg` (open in any browser / Affinity)
   - `summary.png` (side-by-side diagnostic page)

Output goes to:

```
data/outputs/end_to_end/<input_stem>/<YYYYMMDD-HHMMSS>/
```

(same layout as the existing `tools/test_end_to_end.py` script — runs
from both are interleaved by timestamp and never overwrite each other).

## What this is NOT (yet)

This first version intentionally defers:

- Affinity Publisher `.afpub` file generation
- Right-click "Open with ArmourCore CDS Vectoriser" folder menu
- Installer / `.exe` packaging
- Order-folder QS-number auto-detection
- Multi-page / multi-file batch processing

These are tracked in
`D:/Downloads/ArmourCore_CDS_Vectoriser_GUI_Packaging_Brief.md` and can
be layered on later without disturbing the rest of the app.

## Reverting

Everything the GUI adds lives in:

- `gui/`                                 — the GUI code
- `Launch_ArmourCore_Vectoriser.py`      — root launcher
- `Launch_ArmourCore_Vectoriser.bat`     — Windows double-click launcher
- `README_GUI.md`                        — this file

Delete all four and the project is back to its previous state.  No
existing code under `src/`, `tools/`, `data/`, `configs/`, or
`outputs/` was modified.
