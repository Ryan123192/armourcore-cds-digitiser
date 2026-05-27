# Improvement Proposals v2 — Post-Work-Computer Testing

Updated after the 2026-05-20 work-computer testing session.
**Scope:** new-CDS workflow only (legacy CDS deferred per instruction).

This document only proposes work — no code is being changed by it.
The minor fixes from the handoff document have already been applied
(see "Applied today" at the bottom).

---

## A. CDS colour-exploration tool  ⭐ NEW

You mentioned wanting visual exploration of CDS colour combinations
that balance **aesthetic** with **computer-vision friendliness**.
A small standalone GUI / script could make this very fast to iterate on:

### Concept

A `tools/test_cds_colour_explorer.py` script with a small Qt window
that lets you:

1. Pick (or type hex) the minor-grid colour, major-grid colour, and
   text colour.
2. **Live-render** a synthetic CDS preview at print scale (e.g., a
   200×200 mm patch with grid lines + sample text).
3. Apply the **exact production pipeline's detection logic** (`L11`
   HSV + hard-white clamp at gray≥160) to that synthetic render and
   visualise:
   - what would be caught as grid (red overlay)
   - what would remain after cleaning (gray overlay)
   - a **CV-friendliness score** (% of grid pixels caught, % of dark
     ink pixels left intact)
4. Side-by-side panel showing the **aesthetic preview** at typical
   viewing scale.
5. "Save palette" button writes a tiny JSON of the chosen colours
   so they can be referenced in a print spec sheet later.

### What this saves you

Instead of printing a sample → photographing → running the full
pipeline → judging → reprinting (a 30-minute cycle), you iterate in
seconds. The synthetic render isn't a perfect substitute for real
photo conditions, but it's close enough to *eliminate* obvious bad
choices before you waste a print.

### Implementation effort

S–M (4–6 hours). Re-uses existing pipeline functions:
- `build_orange_mask` from `phase2/trace_isolation.py`
- The hard-white-clamp logic from the same module

Lives entirely in `tools/` — no production-code changes.

**Rank: 🥇** — directly supports your stated need this week, and a
better palette pays back forever in CV reliability.

---

## B. Per-tool vectorisation diagnostics + timeout (ISSUE-019)

The biggest production-blocker on the work PC: the pipeline sits at
`vectorising tool 1 of x` for ages when `x` is large.

### Investigation tier (cheap — apply first)

Add **per-tool logging** so we can SEE which tool is hanging:

```
[Tool  3/47]  bbox=(412,180,820,400)  area=58 400 px  fg=12 880 px
[Tool  3/47]  start  components=18  contours_before=44  contours_after=12
[Tool  3/47]  done in 1.3 s  loops=2  bridges=4  k=15
```

In *minimal* debug mode, only emit summary stats; in *full* mode,
dump `tool_NNN_crop.png`, `tool_NNN_mask.png`, `tool_NNN_stats.json`
into a `phase_3/` subfolder.

### Safety tier (medium effort — apply if needed)

Add a per-tool **timeout** of e.g. 60 seconds. If a single tool
exceeds the timeout, log a warning, mark that tool as `timed_out`
in the run report, and continue to the next. This rescues a batch
from one pathological input.

Implementation note: the cleanest path is `multiprocessing.Pool` with
a per-task timeout. This DOES require carefully passing OpenCV
arrays across processes (no shared state); the existing
`vectorise_tool` function would need a small wrapper.

Lower-risk alternative: run vectorisation in a daemon thread inside
the worker and check elapsed time; can't truly kill a runaway
NumPy/OpenCV loop but at least signals the issue.

### Effort

S (logging only), M (timeout via multiprocessing).
**Rank: 🥇** — critical for production usability.

---

## C. Local working cache for network inputs (ISSUE-012)

Paired timing proves IO is the cost:
- Blue CDS, Phase 1 local: **6 s**
- Blue CDS, Phase 1 network: **19.7 s**

### Proposal

Add `gui/local_cache.py`:

```python
def stage_input_locally(src_path: Path) -> Path:
    """If src is on a network/UNC path, copy to %TEMP%/armourcore/<stem>
    and return the local copy.  Otherwise return src unchanged.

    The copy is named with a hash of (src_path, src_mtime) so repeated
    runs of the same unchanged source reuse the cached copy instead
    of re-copying."""
```

Detection of "is network": check if path starts with `\\` or matches
a mapped drive that's network-backed (`win32api.GetDriveType`).

Same idea for output: write intermediate debug artefacts locally,
then copy ONLY the final 3-file output set back to the user-chosen
network folder at the end.

### Effort

S–M (1–2 days including testing).
**Rank: 🥈** — meaningfully faster on work-PC + network-drive combo.

---

## D. Output folder restructure + debug modes (ISSUE-013)

Handoff describes the current output as messy:
> Many PNG variants are saved... results being saved to
> `data/outputs/endtoend` AS WELL AS an `outputs` folder.

### Proposed structure (per file processed)

```
<user-chosen-output-root>/
  <stem>/
    <YYYYMMDD-HHMMSS>/
      <stem>_rectified.png              (always)
      <stem>_vectors.svg                (always, if Phase 3 succeeds)
      <stem>_debug_summary.png          (debug ≥ minimal)
      run_report.json                   (always — see schema in handoff §9)
      debug/                            (only if debug = full)
        phase_1/...
        phase_2/...
        phase_3/
          tool_001_crop.png
          tool_001_mask.png
          tool_001_stats.json
          ...
```

GUI gets a dropdown: **Debug output: Off / Minimal / Full**.

Off  = rectified + vectors + run_report only.
Minimal = above + debug_summary.png.
Full = above + per-phase debug folders.

### Effort

S. Most of the structure is already producible — just need to gate
the writes on debug mode.

**Rank: 🥈** — high housekeeping value, low risk.

---

## E. Memory & timing instrumentation (ISSUE-011)

Memory hit 26.8 GB on the work PC. Before we can fix this, we need
to know **where** memory is going.

### Proposal

Add `gui/profile.py`:

```python
import psutil, time, os
class StageProfiler:
    def __init__(self): self.records = []
    def snapshot(self, label):
        proc = psutil.Process(os.getpid())
        rss_mb = proc.memory_info().rss / 1e6
        self.records.append({
            "stage": label,
            "wall_time": time.time(),
            "rss_mb": rss_mb,
        })
    def report(self) -> dict: ...
```

Pipe these snapshots into `run_report.json` so every run produces a
memory-by-stage timeline. After 5–10 runs we'll know which stage
spikes worst on the work PC and can target it.

### Effort

S. Pure additive instrumentation, no behaviour change.

**Rank: 🥈** — necessary precondition for any real memory work.

---

## F. Text / noise pre-vectorisation filter (ISSUE-018)

Handoff says customer handwritten text fragments are creating tiny
junk vectors and likely slowing the per-tool vectorisation pass.

### Proposal — *physical-size* filter (not just pixel count)

Now that Phase 1 rectifies to a known px/mm scale, "min component
size" can be expressed in millimetres and applied confidently across
templates and resolutions:

```python
PX_PER_MM = image_width_px / design_area_mm.width
MIN_COMPONENT_AREA_MM2 = 30   # configurable

min_area_px = MIN_COMPONENT_AREA_MM2 * PX_PER_MM ** 2
# ... filter connected components in trace mask
```

Plus an **aspect-ratio filter** for text-like fragments: components
with extremely elongated shape AND small bounding-box width usually
aren't tools.

Lives in the GUI/pipeline_runner layer between Phase 2 and Phase 3
so the existing Phase 3 code stays untouched.

### Effort

M (1 day, with care to not eat real tools).

**Rank: 🥉** — biggest win for messy customer photos; may need
tuning per template type.

---

## G. Mid-vectorisation cancellation (ISSUE-015)

Cancel currently waits for the current file to complete. On a stuck
tool, that means forever.

### Proposal — multiprocess each file's Phase 3

Run Phase 3 (vectorisation only) in a `multiprocessing.Process`
spawned from the worker thread. The cancel button then calls
`process.terminate()` for a hard kill, while:
- the rectified PNG is already on disk (ISSUE-006 fix already shipped)
- the run_report.json records `cancelled_by_user`

The process boundary also helps with the memory issue (E) — the
huge Phase 3 arrays get freed by OS process exit rather than by
Python GC.

### Effort

M. The Phase 3 entry function (`detect_tools` + per-tool loop in
`pipeline_runner.run_pipeline`) needs to be split out as a standalone
callable that's safe to pickle.

**Rank: 🥈** — pairs naturally with B's per-tool timeout.

---

## H. Auto-orientation (ISSUE-001) — only if portrait shoots appear

The handoff lists this as high-priority but it's also listed under
legacy/old workflows. For the **future new CDS**, customers shooting
on phones usually get landscape (the CDS is wider than tall).

**Recommend:** add a tiny diagnostic NOW — log the input image's
orientation EXIF tag and a "looks portrait?" guess based on aspect
ratio. Don't auto-rotate yet. After 1–2 weeks of real-world inputs
we'll know if it's needed.

**Effort:** XS (10 minutes for the logging).

**Rank: 🥉** — premature without data.

---

## I. Auto template-size detection (ISSUE-005)

Same logic as H: log the detected inside-border aspect ratio as a
diagnostic for every run, then assess whether auto-detection is
needed. The detection itself (compare aspect ratio against
0.8333 vs 0.5556) is trivial — the risk is misclassifying.

**Effort:** XS for the diagnostic, S for actual auto-classification
with a confidence threshold.

**Rank: 🥉** — defer until data shows real misclassification.

---

## Suggested order

If I were prioritising for the rest of today:

1. **A. CDS colour-exploration tool** — your stated need this week,
   high leverage for picking the next print run's palette.
2. **B-investigation. Per-tool diagnostic logging** — cheap, tells
   us what's actually happening on the work PC's hangs.
3. **D. Output folder restructure + debug modes** — improves
   testing hygiene immediately.
4. **E. Memory + timing instrumentation** — precondition for any
   later perf work.
5. **C. Local working cache** — biggest single perf win confirmed
   by timing data.
6. **G + B-timeout. Mid-vectorisation cancel via subprocess** —
   structural fix for the hangs.
7. **F. Text / noise pre-filter** — quality fix once memory and
   hangs are under control.

H and I stay parked.

---

## Applied today (informational, no action needed)

These were the small fixes from the handoff that were judged safe
to apply immediately:

| Issue | Fix | Files touched |
|---|---|---|
| ISSUE-006 | Rectified PNG saved into final folder *right after Phase 1*. Survives any later hang/crash. | `gui/pipeline_runner.py`, `gui/worker.py`, `gui/main_window.py` |
| ISSUE-007 | Drag-and-drop files / folders from File Explorer onto the GUI. | `gui/main_window.py` |
| ISSUE-008 | "Output folder: [path] [Change...]" picker in the left panel. Output goes wherever the user chose. | `gui/main_window.py`, `gui/pipeline_runner.py` (new `output_root` arg) |
| ISSUE-009 | Last-used input and output folders remembered via `QSettings` ("ArmourCore" / "CDS Vectoriser"). | `gui/main_window.py` |
| ISSUE-010 | Preview is now loaded via cv2 → downscale to 1400 px → QImage. Bypasses Qt's 256 MB allocation limit. | `gui/main_window.py` |
| ISSUE-014 | "Open SVG" button removed. "Open Output Folder" is now the single output-side button. | `gui/main_window.py` |

Nothing under `src/` or `tools/` was modified. All changes live in
`gui/`. Deleting `gui/` reverts to command-line-only behaviour.

A smoke test confirms the GUI launches cleanly with the new layout.
