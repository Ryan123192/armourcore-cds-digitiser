# ArmourCore CDS Digitiser — Phase 01 Colour Border Debug Summary

## Purpose

This summary captures the main findings, code changes, tests, failures, and current direction from the recent Phase 01 debugging work on coloured border detection.

---

## Core objective

Get **Phase 01** reliably detecting the CDS outer border for:
- legacy **black-border** sheets
- newer **coloured-border** sheets

with the border truth defined as:
- **the inside edge of the thick outer border**, not the outside edge.

The reason for introducing colour was to make later grid suppression and customer-marking isolation easier for downstream vectorisation.

---

## Big-picture findings

### 1. The original coloured-border architecture was structurally wrong
The earlier implementation was too loose and mixed together:
- page/geometry detection
- border colour detection
- grid/text colour detection
- template mode resolution

The main issue was that colour handling was being blended into generic candidate generation/scoring instead of being cleanly controlled by template/mode.

### 2. Template mode must be decided up front
A major improvement was making border mode explicit and template-driven:
- legacy sheets should stay in **black** mode
- colour test templates should stay in **orange** mode initially, then later **blue** mode
- auto mode should not be trusted until each explicit mode is stable

### 3. Outer border colour candidates must only contain the border colour
A major bug was identified where `outer_border_hex_candidates` included colours belonging to:
- major grid
- minor grid
- text / other printed structure

This polluted candidate generation badly.

This was the reason digital ideal cases were failing until the set was restricted.

### 4. Clean digital cases and photographed cases fail for different reasons
- Clean digital/PDF cases became easy once the border-only candidate colour was isolated.
- Real photographed cases remained unstable because colour evidence weakens under:
  - lighting shift
  - glare
  - white balance
  - perspective
  - background confusion

### 5. The current remaining issue is not “can it see blue?”
The system can now detect blue borders in some good real images.
The real remaining issue is:
- **one or more sides/corners drifting**, especially at bad angles
- and the refinement logic not correcting the bad side robustly enough.

### 6. The dimensional truth is still critical
Even when the border is found, the correct geometry is:
- **inside edge of thick border**

not:
- outside edge of thick border.

This required a dedicated refinement stage.

---

## Sequence of changes attempted

## A. Structural border-mode overhaul
Patch file:
- `apply_phase1_border_mode_overhaul.py`

Main intent:
- add template-level border mode
- resolve mode up front in pipeline
- separate black/orange logic more clearly
- fix template schema / colour hint flow

Outcome:
- applied successfully
- `pytest` passed
- black legacy sheet path behaved well again
- orange path improved structurally but still selected wrong orange structures in some cases

Key finding:
- architecture improved, but orange-border logic still too entangled with internal orange page structure.

---

## B. Orange border debugging
Initial orange border used:
- `#D35400`

Important discovery:
- ideal digital orange cases failed largely because `outer_border_hex_candidates` incorrectly included non-border colours.
- once restricted to only the actual outer border hex, digital cases started working properly.

Key conclusion:
- `outer_border_hex_candidates` must be **outer-border-only**.

Real-photo issue remained:
- photographed orange border did not consistently survive lighting and perspective.

---

## C. Decision to switch border colour from orange to blue
Chosen border colour:
- **blue outer border** = `#00CAEC`

Fiducials retained as:
- **red** = `#FF0000`

Intent:
- keep border colour separated from grid/text and from user pencil/pen marks
- reduce overlap with warm/orange page content

Patch file:
- `apply_phase1_blue_border_patch.py`

Main intent:
- add blue border mode end-to-end
- support a blue range around `#00CAEC`
- keep fiducials separate from the border colour path

Outcome:
- ideal digital version worked
- a good real client-like image worked after removing blue background interference
- a bad-angle real image still failed or performed poorly

Key finding:
- pure colour-led candidate generation was still too brittle.

---

## D. Geometry-first candidate generation
Patch file:
- `apply_phase1_geometry_first_patch.py`

Main intent:
- stop using colour as the primary generator of candidate rectangles
- go back toward the earlier black-border style process:
  - geometry first
  - colour second
- separate masks by role:
  - border colour
  - fiducials
  - grid/text
- dump diagnostics on failure

Outcome:
- geometry-first behaviour was more sensible
- but bad-angle cases still selected candidates with one bad corner/side
- colour and side averaging were still too forgiving

Key finding:
- the general direction was right, but ranking/refinement still needed work.

---

## E. Inside-edge and scoring patch
Patch file:
- `apply_phase1_inside_edge_and_scoring_patch.py`

Main intent:
- move from the outside of thick border toward the inside edge
- remove or weaken unhelpful scoring terms
- improve scoring logic based on discussion

Changes targeted:
- remove `shape_score` from ranking
- remove `extent_score`
- remove separate `band_score`
- remove grid penalty
- remove source bonus
- make ratio use rectified estimate rather than rough averaging
- strengthen side scoring
- use all 4 corners in corner scoring
- add stricter blue range logic
- add inside-edge refinement

Outcome:
- inside-edge detection improved
- but bad-angle `Test02` still had a bad top-left corner
- the top-left side was not corrected enough by refinement

Run report evidence showed:
- winning source was still often `edges`
- `marker_support_score` was often `0.0`
- all `marker_corner_scores` were `0.0`
- inside-edge offsets on one or more sides were weak or zero

Key finding:
- refinement was still too weak and too local
- fiducial logic was not contributing meaningfully
- one bad corner was still not punished hard enough.

---

## F. Rectified multi-pass refinement attempt
Patch file:
- `apply_phase1_rectified_refine_patch.py`

Main intent:
- try a multi-pass coarse-to-fine refinement flow:
  1. coarse geometry-first quad
  2. rough rectify with padding
  3. refine border in rectified space
  4. final rectify

Reasoning:
- avoid a generic second crop
- do border-constrained refinement in rectified space instead

Outcome:
- this made things worse
- patch was rolled back

Rollback command used:
```bash
cp "src/armourcore_cds/phase1/boundary_detection.py.bak_phase1_rectified_refine_patch" "src/armourcore_cds/phase1/boundary_detection.py"
```

Key finding:
- the idea may still be valid conceptually, but the specific implementation was not good enough and should not be treated as current direction without redesign.

---

## Scoring logic discussion and recommended corrections

The following score terms were reviewed in detail.

### Ratio score
Issue raised:
- using average side lengths was not robust enough

Recommended direction:
- compare **rectified width:height** against expected ratio
- keep this score

### Shape score
Issue raised:
- real photos under perspective should not be penalised just because the quad is not rectangle-like in image space

Recommended direction:
- remove from ranking
- at most keep as a cheap reject-only sanity check

### Area score
Issue raised:
- useful only as a weak plausibility bound

Recommended direction:
- keep with low weight
- allow broad framing variation

### Extent score
Issue raised:
- too similar to area score and can work against cropped/far-away images

Recommended direction:
- remove

### Side score
Issue raised:
- this is one of the genuinely useful metrics

Recommended direction:
- keep and make it a main term
- ensure tolerance to:
  - creases
  - tool overlap
  - pen/pencil crossing
  - small local gaps

### Band score
Issue raised:
- overlaps too much with side support

Recommended direction:
- merge into side logic or remove as a separate ranking term

### Corner score
Issue raised:
- averaging the best 3 corners can hide one terrible corner

Recommended direction:
- use all 4 corners
- weight the worst corner more strongly

### Colour score
Issue raised:
- should be blue-only in blue mode
- strict colour should still be a range, not an exact pixel match

Recommended direction:
- use a **strict blue range** and a **looser blue range**, both centred around the intended border colour
- do not use orange logic in blue mode

### Grid penalty
Issue raised:
- the grid is adjacent to the border, so penalising nearby grid can hurt the correct border

Recommended direction:
- remove the grid-side penalty

### Marker support score
Issue raised:
- current logic often returns zero even when corners are visually close to fiducials

Recommended direction:
- redesign marker logic
- use fiducials as:
  - guard rails
  - corner validators
  - corner refiners
- not just a weak bonus term

### Source bonus
Issue raised:
- if scoring works correctly, source bias should not be needed

Recommended direction:
- remove

---

## Most important observed failure mode

In the problematic bad-angle blue-border case:
- three sides/corners were visually good
- top-left was wrong
- selected candidate still scored highly enough to win
- inside-edge refinement corrected some sides but not the bad top-left side

Interpretation:
- the pipeline is still too forgiving of a single bad side/corner
- global averaging is masking local failure
- marker logic is currently not functioning well enough to stop corner drift

---

## Current best understanding of the right direction

### Keep
- geometry-first candidate generation
- colour as a ranking/refinement aid, not the primary generator
- explicit template border mode
- border-only colour candidates
- border truth = inside edge of thick line

### Remove or reduce
- shape score in main ranking
- extent score
- redundant band score
- grid penalty near border
- source bonus
- any broad candidate colour list that includes non-border colours

### Improve next
- marker detection / scoring / corner validation
- side-specific rather than globally averaged colour and side scoring
- worst-corner-sensitive ranking
- inside-edge refinement robustness
- failure diagnostics

---

## Current practical notes

### Current border colour setup
- border: blue `#00CAEC`
- fiducials: red `#FF0000`

### Important rule
`outer_border_hex_candidates` should only contain the actual border colour for the active template.

Example:
```yaml
outer_border_hex_candidates:
  - "#00CAEC"
```

Do **not** include:
- grid colours
- text colours
- helper colours

---

## Rollback note

The last rectified-refinement patch was rolled back.

If needed again, the restore command was:

```bash
cp "src/armourcore_cds/phase1/boundary_detection.py.bak_phase1_rectified_refine_patch" "src/armourcore_cds/phase1/boundary_detection.py"
```

---

## Recommended next conversation starting point

For the next debugging pass, start from this framing:

1. **Geometry-first candidate generation stays**
2. **Blue border colour is only used to rank/refine, not to generate rectangles**
3. **Outer border candidate colour list must remain border-only**
4. **The output must snap to the inside edge of the thick border**
5. **The main current blocker is a single-side / single-corner drift problem**
6. **Marker logic is probably wrong or too weak, because it keeps returning zero even when corners are visually close to fiducials**
7. **Global averaging is still hiding local failures**

---

## One-sentence status summary

Phase 01 is much improved structurally and can now detect clean digital and some real coloured-border sheets, but bad-angle photographed cases still fail because one wrong side/corner can survive scoring and the current marker + inside-edge refinement logic is not yet strong enough to force the quad onto the true inside edge of the border.
