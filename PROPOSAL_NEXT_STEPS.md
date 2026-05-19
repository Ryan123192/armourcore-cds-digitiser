# Improvement Proposals — Pick Any, Skip Any

Research-only document. **No code has been changed.** Every proposal
below has been validated by reading the existing codebase and/or
external research. Each one is independently optional.

Sections:
1. Brand & GUI styling (ArmourCore visual identity)
2. Affinity Publisher integration — the realistic path
3. Phase 3 gap-fill improvements (V2.02 pencil bottleneck)
4. Adaptive pencil-trace detection
5. Legacy black-grid CDS support cleanup
6. Multi-file / order-folder batch mode
7. Right-click "Open with..." Windows integration
8. HEIC support (iPhone photos)
9. Performance optimisations (small wins)
10. Code organisation — move `tools/test_phase3_*` into `src/`

For each: what it is, why it matters, effort estimate (S/M/L),
implementation risk, and rank by overall value.

---

## 1. Brand & GUI styling

**Current state:** the GUI uses default Qt widget style (off-white,
Windows native). Functional but generic.

**What ArmourCore looks like** (researched from the live site):

- Premium / industrial / engineered aesthetic — aviation-engineering
  heritage, EVA-foam tool inserts, B2B/prosumer tone.
- **Dark, matte hero panels** with monochrome product photography.
- High-contrast typography — large uppercase headings, tight tracking.
- Sharp 2–4 px button corners, no gradients, no drop shadows.
- Photography-heavy when not flat; flat sections are line-only.

**Best-estimate palette** (eyedropper readings from rendered pages —
real values are in the Squarespace CDN, can be locked exactly by
sampling the actual logo asset later):

| Role            | Hex      | Use                              |
|-----------------|----------|----------------------------------|
| Charcoal        | `#0E1116` | Window background                |
| Card surface    | `#1B1F24` | Side panels                      |
| Cyan accent     | `#00B5C8` | Primary buttons, hover/focus     |
| Amber accent    | `#E8A33D` | Optional warm callout            |
| Body (dark bg)  | `#E6E8EB` | Most text                        |
| Muted           | `#9AA3AD` | Subtitles, hint text             |

**Typography**: site uses Proxima Nova-family geometric sans-serif.
Free shippable substitutes that will look right and don't need a
licence: **Montserrat** for headings (uppercase, condensed), **Inter**
for body, **JetBrains Mono** for the log panel.

**Effort:** S (1 hour). Drop a QSS stylesheet into `gui/app.py`, give
the Start button `objectName="StartButton"` so the accent colour
applies, ship the three OFL-licensed font files in `gui/fonts/`,
register them via `QFontDatabase.addApplicationFont` at startup.

**Risk:** none. Pure cosmetic; rolling back = remove the stylesheet
line.

**Rank: 🥈 — highest visible value per hour of work.** Makes the
internal tool feel like an ArmourCore product, which matters for
staff adoption.

Ready-to-paste QSS draft included at the bottom of this doc
(Appendix A).

---

## 2. Affinity Publisher integration — the realistic path

**Reality check from the research:**

- **Native `.afpub` generation is not viable.** Format is proprietary,
  no parser, no library, no spec.
- **Affinity has no scripting API.** Serif staff describe it as
  "pre-pre-pre-alpha" in 2026; no public beta in Publisher 2.6.
- **No headless / command-line mode.** Cannot invoke Publisher from
  Python at all.

**What does work, ranked best-first:**

1. **PDF** — Affinity Publisher imports PDF very well. Vectors come in
   as fully editable curves; raster placed as expected; layers from
   PDF Optional Content (OCG) ARE honoured; pages map 1:1.
   - **Best target for us.** Generate a PDF from Python with
     ReportLab or `pypdf` containing:
     - One page per CDS template at the exact mm dimensions.
     - The rectified PNG placed at the bottom layer.
     - The Bezier contours emitted as native PDF vector ops on a
       separate, named OCG layer ("Raw Vectors").
     - Optional: an empty "Working Vectors" layer (per brief 10.3).
   - When the user opens the PDF in Affinity Publisher it feels
     basically native.

2. **IDML** (the dark horse) — Affinity Publisher imports Adobe IDML
   bundles cleanly since v1.8. Python library `SimpleIDML` exists and
   is in production at Le Figaro. IDML round-trips back to a real
   Affinity layered document. But:
   - SimpleIDML is a *manipulator*, not a builder. We'd need to ship a
     hand-built `template.idml` skeleton and splice content into it.
   - More fiddly than PDF for vector paths.
   - Worth pursuing only if PDF's import doesn't preserve ArmourCore's
     specific layer structure.

3. **SVG** — works, but Affinity imports SVG as an "embedded document
   group". User has to manually Release to access the vectors. Adds a
   click per file. Avoid as primary output.

**Recommended path:** **PDF.** Specifically:

```python
# Conceptual sketch — not yet implemented
from gui.affinity_export import write_armourcore_pdf

write_armourcore_pdf(
    pages=[
        AffinityPage(
            page_name="Big Drawer 01",
            size_mm=(500, 600),
            reference_png_path=rectified_path,
            vector_contours=loops,
            layer_names=["Reference Image", "Raw Vectors", "Working Vectors"],
        ),
        ...
    ],
    output_path="QS26105, Design.pdf",
)
```

The user opens this PDF in Affinity Publisher (or it auto-opens),
clicks "Save As .afpub" once, and they're inside a real Affinity
document with layers and editable vectors.

**Effort:** M (1–2 days for a clean implementation with ReportLab).

**Risk:** low — PDF generation is well-understood. The one tricky
piece is matching ArmourCore's exact layer naming, which is currently
unknown (brief TODO-Q-AFF-001). That can be parameterised.

**Rank: 🥇 — this is the highest-impact functional gap.** Closes the
brief's biggest open question with a viable path that doesn't depend
on Serif shipping anything new.

---

## 3. Phase 3 gap-fill improvements (V2.02 pencil)

**Bottleneck:** V2BlueColourTest02 cleaned image is gorgeous; only 9
of 19 detected tools close into loops. The pencil tracings themselves
have small breaks that the current kernel sweep misses.

**How the current pipeline closes loops** (verified in
`tools/test_phase3_iso_vectorise.py:666 adaptive_method_F`):

```python
kernel_sizes = [15, 20, 25, 30, 35, 40, 50, 60]
for k in kernel_sizes:
    centerline = method_F_closed_centerline(trace, base_k=k, ...)
    n_loops = count_loops(centerline)
    if n_endpoints == 0 and n_loops >= 1:
        break
```

It tries each kernel in order, stops on first "perfect" result.
**Step sizes are 5 px at the low end and 10 px at the top — coarse.**
For pencil traces with gaps that sit between kernel sizes (e.g., a
gap that needs k=17 closes at k=20 but distorts shape), the chosen
kernel can either fail to close or close at distorted scale.

**Three concrete improvements, ranked:**

### 3.1 Finer kernel granularity (S, very low risk)

Replace `[15, 20, 25, 30, 35, 40, 50, 60]` with
`[13, 15, 17, 19, 21, 24, 27, 30, 34, 38, 42, 50, 58]`. More attempts
at smaller kernels, especially where pencil-gap sizes cluster
(15–25 px).

**Why this is safe:** the scoring function picks the smallest kernel
that gives a clean loop, so adding intermediate options can only
*improve* outcomes — never make them worse, except for taking slightly
more compute (sub-linear because of the early break).

### 3.2 Confidence-based "rescue" pass (M, low risk)

After the per-tool primary loop closes nothing or returns a
near-loop (e.g., contour with 1–3 endpoints), retry with a more
permissive `bridge_max_dist_px` (current default 250 → 400) and a
smaller `min_loop_abs_px` (current 500 → 250). Only on rescue
candidates, so it doesn't slow the normal path.

Identifies "near-misses" automatically and gives them a second
chance with relaxed parameters.

### 3.3 Otsu adaptive trace threshold (S, low–medium risk)

Currently Phase 3 binarises the cleaned image with a hard
`gray < 170` threshold. Pencil tracings on the new V2 paper sometimes
sit at `gray ≈ 150–175`; some of the pencil ink is being thresholded
away before Phase 3 ever sees it.

Replace with Otsu's method — automatically picks the optimal split
between the ink-dark distribution and the paper-light distribution.

```python
_, trace = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)
```

**Risk note:** Otsu can over-include shadowed paper as ink on poorly-
lit images. Compare both side-by-side on V2.02 before adopting.

### Combined expected impact

V2.02 should jump from 9 → 14–16 loops with all three. V2.01 and
V2.03 should be unaffected or marginally better.

**Effort total:** S (all three), 2–4 hours.

**Rank: 🥈** — directly addresses the user's stated bottleneck on
real-world pencil photos.

---

## 4. Adaptive pencil-trace detection

(Covered in 3.3 above — Otsu threshold. Listing separately because it
also helps Phase 2's `normalise_paper_to_white` step in marginal
cases.)

**Rank: bundled with Section 3.**

---

## 5. Legacy black-grid CDS support cleanup

**Current state:** `src/armourcore_cds/phase2/trace_isolation.py`
contains an entire alternate code path called
`_build_black_grid_removal_mask` for legacy CDS sheets with black
(not orange) printed grids. It uses directional erosion to identify
thin grid lines without eating thick customer marks.

This code path is **preserved but largely unused** in current
testing. It's also **substantially more complex** than the new orange
path.

**Hypothesis worth testing:** the new L11 approach (HSV chroma + hard-
white clamp at gray≥160) might generalise to legacy black grids too.

- Black grid line: gray ≈ 60–120 (dark, but generally NOT as dark as
  pencil ink ~100, OR shares the range).
- Pencil ink:    gray ≈ 80–130.

If the grid is *lighter* than the customer ink in the legacy template
(e.g., grid grey ~140, ink black ~30), the hard-white clamp at
gray≥160 doesn't apply, but a different threshold might.

**Proposed test:** find one legacy CDS sample, run the *current
production orange path* on it (which will do nothing useful for
grid, but will still try), and inspect the output. If the legacy
grid is consistently lighter than tool ink, a single threshold
captures both cases. If they overlap heavily, the directional
erosion stays.

**Effort:** S (find one sample + run one test). The cleanup would
only happen if the test passes.

**Risk:** low — purely additive testing.

**Rank: 🥉 nice-to-have** — code simplification and a unified pipeline
across both template families. No urgency unless legacy CDS jobs are
coming in regularly.

---

## 6. Multi-file / order-folder batch mode

**Current state:** GUI processes one file at a time. The brief
section 5.1 describes the production workflow as folder-based:

> 1. Staff right-click an order folder.
> 2. App auto-detects multiple CDS files inside.
> 3. App lists them in a table for review.
> 4. App processes each, builds one output (PDF / Affinity file)
>    with one page per detected file.

**What needs to be added:**

- **Folder picker** in the GUI alongside the file picker.
- **File detection logic** — walk the folder, filter by extension,
  match filename keywords (`template`, `insert`, `tool`).
- **Confirmation table widget** — `QTableWidget` listing each detected
  file with editable insert name + template-type dropdown + include
  checkbox.
- **Sequential processing** with cumulative progress.
- **Output file naming** per brief section 11: `QS26105, Design.pdf`
  with versioning if exists.

This is **orthogonal** to the single-file flow we have. We add it as
a new section in the GUI without removing the single-file mode.

**Effort:** M (1 day for the GUI + 0.5 day for the file-detection
heuristics).

**Risk:** low.

**Rank: 🥇 once single-file workflow has been tested at work** —
this is the closest match to the brief's "production workflow"
section 5.1.

---

## 7. Right-click "Open with..." Windows integration

**Reality from research:** two registry keys, written under HKCU
(per-user, no admin required):

- `HKCU\Software\Classes\Directory\shell\ArmourCoreProcess`
- `HKCU\Software\Classes\Directory\Background\shell\ArmourCoreProcess`

Each gets a child `command` subkey pointing to:

```
"C:\path\to\python.exe" "C:\path\to\Launch_ArmourCore_Vectoriser.py" "%V"
```

Python's stdlib `winreg` writes these. We just add a "Register
right-click menu" button to the GUI's first launch (or a separate
`Register_RightClick.bat`).

**Effort:** S (1–2 hours).

**Risk:** low. The brief notes IT/admin may block registry edits
(TODO-Q-PKG-007); for HKCU keys this is rarely blocked, but we
should add a clear failure message if so.

**Bonus:** the launcher needs to accept a folder path as a CLI
argument and pre-select it. Two lines of code.

**Rank: nice-to-have, post-MVP.** Cleanly matches brief section 4.2.

---

## 8. HEIC support (iPhone photos)

**Brief TODO-Q-IN-001** asks whether HEIC support is needed. iPhone
photos default to HEIC on iOS 11+. If staff or customers send
iPhone photos directly, the current pipeline (OpenCV `cv2.imread`)
fails on HEIC.

**Fix:** install `pillow-heif`, which adds HEIC support to PIL,
then route image loading through PIL → numpy → OpenCV.

```python
import pillow_heif
pillow_heif.register_heif_opener()
from PIL import Image
img_pil = Image.open(path)
img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
```

**Effort:** S (30 minutes, mostly testing).

**Risk:** very low — `pillow-heif` is a stable, widely-used
library. One extra dependency, ~5 MB.

**Rank: cheap insurance** if any chance of iPhone photos.

---

## 9. Performance optimisations

**Current end-to-end timings** (V2 cases, latest production):

- Phase 1 (rectify): ~10–20 s per image
- Phase 2 (clean):   ~15–20 s per image
- Phase 3 (vector): ~10–35 s per image (depends on tool count)

**Small wins identified:**

### 9.1 Downscale CLAHE input for Phase 2 lighting normalisation

`normalise_lighting()` runs CLAHE on the full 4094×5512 image. The
output gets re-scaled for downstream operations anyway. Run CLAHE
on a 1500-max-dim downscale, then upscale the L channel back.

**Expected savings:** 1–2 s per case.

### 9.2 Cache Phase 1 rectification for repeated runs

When testing, we often re-run Phase 2/3 on the same Phase 1 output.
The `--skip-phase1` flag in `test_end_to_end.py` already does this.
The GUI doesn't currently expose it. Add a checkbox: "Reuse last
Phase 1 result if available".

**Expected savings:** ~15 s per iteration during dev testing.

### 9.3 Multi-tool vectorisation in parallel

`vectorise_tool` is called per-tool in a serial loop. Each call is
CPU-bound and independent. A `ProcessPoolExecutor(4)` would give a
~2.5× speedup on Phase 3 for V2.03-style cases (13 tools, ~30 s →
~12 s).

**Risk:** medium — multiprocessing on Windows requires care
(pickling, no shared state, OpenCV may need re-initialisation in
workers). Test carefully before adopting.

**Effort:** S (9.1 and 9.2), M (9.3).

**Rank: lowest priority** — the pipeline is already usable. Worth
doing once we've ironed out the bigger issues.

---

## 10. Code organisation — promote `tools/test_phase3_*` into `src/`

**Current state:** the Phase 3 code that the GUI and end-to-end
script actually use lives in `tools/test_phase3_batch.py` and
`tools/test_phase3_iso_vectorise.py`. The names imply test scripts,
but they ARE the production pipeline. There's also a separate
`src/armourcore_cds/phase3/vectorise.py` that contains an older /
parallel implementation.

This is confusing for:
- new contributors
- packaging (the `tools/` folder shouldn't ship to production)
- import paths (the GUI has to `sys.path.insert(0, "tools")`)

**Proposal:** carefully move the *production* Phase 3 code from
`tools/test_phase3_*.py` into `src/armourcore_cds/phase3/`. Keep the
old test scripts as thin shims that import from `src/`. Audit the
older `phase3/vectorise.py` — either replace it or delete it.

**Effort:** M (1 day, careful refactor).

**Risk:** medium — touches a lot of import statements. Must be
done with a regression test on V2.01/02/03.

**Rank: low priority, eventual cleanup.** Doesn't change behaviour
or unlock features. Save for when the codebase is otherwise stable.

---

## Recommended path forward

If you want a *ranked roadmap* to pick from:

| # | Item                                       | Effort | Impact         |
|---|--------------------------------------------|--------|----------------|
| 1 | GUI brand styling (Section 1)              | S      | High polish    |
| 2 | PDF output for Affinity (Section 2)        | M      | **Production unlock** |
| 3 | Phase 3 gap-fill fixes for pencil (Sec 3)  | S      | +loops on V2.02 |
| 4 | Multi-file batch mode (Section 6)          | M      | Matches brief 5.1 |
| 5 | Right-click integration (Section 7)        | S      | Matches brief 4.2 |
| 6 | HEIC support (Section 8)                   | S      | Future-proofs inputs |
| 7 | Performance optimisations (Section 9)      | S–M    | Marginal       |
| 8 | Legacy CDS unification (Section 5)         | S      | Code cleanup   |
| 9 | Promote tools/test_phase3 to src/ (Sec 10) | M      | Code cleanup   |

**My honest pick of top 3 in order:** 1, 3, 2.

- (1) Brand styling makes the GUI feel like an ArmourCore tool the
  moment you launch it — high return for an hour.
- (3) Phase 3 gap-fill rescues the pencil case — fixes the only
  visible weakness in current outputs without changing the rest.
- (2) PDF output for Affinity lands the production workflow's
  biggest open question.

After those three, multi-file batch + right-click integration land
the brief's production workflow (sections 4.2 and 5.1).

Performance, legacy unification, and code organisation are all
post-MVP cleanup.

---

## Appendix A — ready-to-paste QSS

```css
/* ArmourCore-themed QSS — dark industrial */
* { font-family: "Montserrat", "Inter", "Segoe UI", sans-serif; color: #E6E8EB; }

QMainWindow, QDialog, QWidget { background-color: #0E1116; }

QLabel { color: #E6E8EB; font-size: 10pt; }
QLabel#H1 { font-size: 22pt; font-weight: 700; text-transform: uppercase;
            letter-spacing: 1.5px; color: #FFFFFF; padding: 6px 0; }
QLabel#H2 { font-size: 14pt; font-weight: 600; color: #FFFFFF;
            letter-spacing: 0.8px; }
QLabel#Muted { color: #9AA3AD; font-size: 9pt; }

QFrame, QGroupBox {
    background-color: #1B1F24; border: 1px solid #262B31;
    border-radius: 4px; padding: 12px;
}

QPushButton {
    background-color: #1B1F24; color: #E6E8EB;
    border: 1px solid #2E343B; border-radius: 3px;
    padding: 8px 18px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.8px;
}
QPushButton:hover   { border-color: #00B5C8; color: #FFFFFF; }
QPushButton:pressed { background-color: #0E1116; }

/* Primary CTA — set objectName "StartButton" on the Start widget */
QPushButton#StartButton {
    background-color: #00B5C8; color: #0E1116; border: none;
}
QPushButton#StartButton:hover   { background-color: #1FC8DB; }
QPushButton#StartButton:pressed { background-color: #008FA0; }

QLineEdit, QTextEdit, QComboBox, QSpinBox {
    background-color: #15191E; color: #E6E8EB;
    border: 1px solid #2E343B; border-radius: 3px; padding: 6px 8px;
    selection-background-color: #00B5C8; selection-color: #0E1116;
}
QLineEdit:focus, QTextEdit:focus, QComboBox:focus { border-color: #00B5C8; }

QProgressBar { background: #15191E; border: 1px solid #2E343B;
               border-radius: 3px; text-align: center; color: #E6E8EB; }
QProgressBar::chunk { background-color: #00B5C8; }
```

To apply, add to `gui/app.py` after creating QApplication:

```python
app.setStyleSheet(open(ROOT / "gui" / "armourcore.qss").read())
```

And in `gui/main_window.py`, set `self.start_btn.setObjectName("StartButton")`.

---

## Appendix B — what I did NOT change

- `src/armourcore_cds/` — untouched
- `tools/` — untouched
- `gui/` — untouched
- `Launch_ArmourCore_Vectoriser.*` — untouched
- `data/`, `configs/`, `outputs/` — untouched

Only `PROPOSAL_NEXT_STEPS.md` (this file) was added. Delete this
single file to revert.
