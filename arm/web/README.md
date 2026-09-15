# The S01 arm in the browser

Draw a figure and the arm copies it. The experience agent of `agent/brain.py` ported to
JavaScript, the arm of `arm/env.py` and its planner alongside it, and one page that shows the
body and the whole brain while it learns. The engine is the same numerics as the Python agent
on the cadence CPU backend, and it carries the records world head of the current config
(`RecordsHead`: the sparse-expansion world model, cerebellum-like and dentate-like) beside the
settled regions; a parity harness replays the recorded fixture through it and compares every
decision, prediction, the learned parameters and the records.

## Files

| file | what |
| --- | --- |
| `../../web/engine.js` | `ExperienceAgent` (one stream) built from a `cadence-experience-web/1` snapshot: `step(moment)`, `predictBatch(observations, actions, goal, {observed})`, `snapshotState()`, `setPlanner(fn)`, the `onSettleStep`, `onPhase`, `onLearn` and `onRecords` callbacks; `Mulberry32` with `batch` and `normals`; `RecordsHead`; `normalizeMoment`; `encodeObservation` on the agent. Float64 everywhere. |
| `../../web/records_view.js` | `RecordsView` (the records cortex on a Canvas2D: the granule raster, the writes, the habituated reading) and `readsHTML` (the per-field record reads as HTML rows). Shared by both pages. |
| `../../web/parity.mjs` | `node web/parity.mjs runs/arm/parity`: the parity check against the Python fixture. |
| `../../web/brain_scan.js` | The standard renderer, a verbatim copy of `cadence/src/cadence/brain_scan.js` (never edited here). |
| `planner.js` | `compose` and `ModelPlanner` from `arm/brain.py`: the bounded beam search through the model's imagined consequences, stable sort. `lastImagined` keeps the hand positions the search imagined for every torque, which the page draws as the fan at the hand. |
| `arm.js` | The arm physics from `env.py` (`Arm`, `TORQUES`, `FIELDS`, `wrap`, `movingPath`) with a Mulberry32 random stream, `requestTruncation()` so a page can move to a new target at an episode boundary, and the joint angles after every body step (`trace`) for drawing the hand's path at the body's own clock. The drawn figures live here too: `workspaceOf`, `clipToRing`, `resamplePolyline`, `smoothPolyline`, `figurePath` (strokes to evenly spaced points with the pen up between strokes), `distanceToDrawing`, `CopyPath` (the path as the episode's moving target) and `exampleFigure`. |
| `page.js`, `index.html`, `style.css` | The page: the drawing surface, the arm view, the brain beside it, the controls and the brain selector. |
| `build_page.py` | Inlines the snapshot, the renderer, the engine, the records view, the planner, the arm and the page into one `index.html` that works from `file://` and on GitHub Pages; `--checkpoint` copies other snapshots of the same life beside the page and writes the `checkpoints.json` manifest the brain selector reads. |
| `check_page.py` | Headless Chromium (SwiftShader) check: the layout at 1280 by 800 and at 390 by 844, a figure drawn through pointer events, two copies of it measured against a declared bound, the rest of the controls, screenshots, and no page or console error. |
| `life_supplement.py` | Regenerates the snapshot's life records in float64 (the pending decision with its records code, and the full replay ring with warm potentials): the fallback for an export from before `web.py` wrote `pending` and `ring.v`/`ring.a`. |
| `../tools/parity_fixture.py` | Writes the fixture: `snapshot.json`, `records_f64.json`, `stream.jsonl`, `final.json`. |

## Build and run

```sh
cd /Users/muellerberndt/Projects/oph-meta/cadence-examples
PY=/Users/muellerberndt/Projects/oph-meta/cadence/.venv/bin/python

# the fixture (a few seconds): 300 babbling decisions, the snapshot, then 120 recorded moments
# at exploration rate 0.5; writes the float64 records sidecar next to the snapshot
$PY arm/tools/parity_fixture.py --out runs/arm/parity

# the parity check (about 6 s)
/opt/homebrew/bin/node web/parity.mjs runs/arm/parity

# the self-contained page (4.3 MB): open it from the file system or serve it anywhere
$PY arm/web/build_page.py \
    --snapshot runs/arm/parity/snapshot.json --out runs/arm/web/index.html

# the headless check (one to three minutes): the layout, a drawn figure copied twice, the
# controls, the screenshots
/Users/muellerberndt/Projects/oph-meta/cadence-artist/.venv/bin/python \
    arm/web/check_page.py runs/arm/web/index.html
```

Any snapshot `agent/web.py` exports for the arm can be built into the page. A snapshot
without `extra.planner` or `extra.arm` takes both from `--config` (the stage config), so the
page always knows its planner and its body.

To let a visitor compare points of the same arm's life, name them on the build:

```sh
$PY arm/web/build_page.py \
    --snapshot runs/arm/final/snapshot.json --out runs/arm/web/index.html \
    --snapshot-label "the final brain" \
    --checkpoint "after 500 babbling decisions=runs/arm/ck/babble500.json" \
    --checkpoint "after 5,000 babbling decisions=runs/arm/ck/babble5000.json" \
    --checkpoint "after reaching training=runs/arm/ck/reaching.json"
```

Each named snapshot is copied to `checkpoints/<id>.json` beside the page and listed in
`checkpoints.json` with its label, file and size. The page fetches that manifest by relative
URL and shows the brain selector when it is there; the inlined snapshot stays the default, so
a page opened from `file://`, where a page cannot fetch its neighbours, still works and offers
the one brain it carries. Choosing another brain rebuilds the agent, the scan and the records view in
place and copies the figure on the surface again, so the same drawing can be compared across
the life.

For development the sources stay separate: serve the repository root over http
(`python -m http.server 8000`) and open `http://localhost:8000/arm/web/index.html`; `page.js`
then fetches the snapshot named by `<body data-snapshot>`. The Python side changes; regenerate
the fixture, rerun the parity check and rebuild the page in that order.

## What the page shows

**Draw.** The drawing surface is the arm's workspace: the reachable ring is drawn faintly, and
a visitor draws inside it with a mouse, a pen or a finger, in one stroke or several. A moment
after the pointer is released the strokes become the target path: clipped to the ring,
resampled to even spacing, smoothed a little, and joined by straight travel segments where the
pen lifts between strokes. The path is then played as a moving target, the way `moving_path`
produces one in `env.py`: the target waits on the first point until the hand comes within 0.06
of it (at most six seconds), then advances one point per decision, which is a steady 0.08,
0.12 or 0.18 arm lengths per second. The planner's velocity objective chases it.

The hand leaves ink in its own colour over the faint drawing, sampled at the body's 50 Hz
clock, dotted where the pen is up. The tracking error above it is the mean distance between
the hand and the target over the moving part of the copy, in arm lengths (the arm's full reach
is one), with the mean distance from the ink to the drawing under it; the chips below list
every copy so far. Copy again replays the same figure, and because the brain keeps learning
while it copies, the second copy of a figure is usually closer than the first. Clear empties
the surface, and four example figures (a circle, a star, a spiral, a letter S) are there for
visitors who do not draw. The page opens with the examples copying themselves one after
another until the visitor touches the surface.

Around the hand the page draws what the planner imagined: for each of the nine torque pairs,
the hand positions the records predicted over the three decisions the torque would be held,
drawn at three times their size with the chosen one bright, because one decision moves the
hand by a few hundredths of an arm length. The executed torque shows as an arc at each joint. Nothing here is
scripted: every torque comes from `ExperienceAgent.step` through the planner, and the copy is
only as good as the model the arm has learned.

**Reach targets.** The second tab is the earlier page: a static target with its success
radius, moving targets (line, circle, mixed; the paths of `env.py`), a click to place a target,
the predicted displacement against the one that happened (drawn at five times their size),
lengthening the upper link by 15 percent so the model has to adapt, and restoring the body. A
new target or a new target mode ends the episode at the next event as a time limit would, so
the decision awaiting its outcome is fed back first.

**The whole brain**, beside the drawing at the same size. The scan through `BrainScan`: every
neuron coloured by its region (sensory, goal, efference, workspace, context, dynamics,
prediction, motor: the colours of the snapshot's atlas) and every synapse, animated per
settling step of every phase the settled regions run (free, the free phase after learning, the
bootstrap value at a time limit); the phase label names the phase. Beside the scan, its own
labelled region, the records cortex (cerebellum-like / dentate-like): the granule raster, all
32,000 cells in a compact grid with the 160 active cells of the executed reading's plain code
lit by their code value (blue) and the valued code's cells in pink (the arm has no reward
record, so no field reads it); imagined reads (the planner's batches of nine torques) as brief
grey flickers on the cells they touch; the writes as flashes on the active cells when the
outcome arrives, scaled by the record rate; under the raster the reading the expansion sees as
bars from a midline with the ports separated by ticks and named (sensors, missing flags, goal,
action) and the running norm of every pathway; and under that the per-field record reads, the
predicted next displacement change, velocity change and angle change on their sensor ranges,
with the largest error of the last write per field. Every code, imagined read and write is
queued with the settle frames and played in order.

Below the brain: the legend of every region, the EEG montage of every region, the dopamine bar
(the critic's clipped TD error per event, signed), the ledger (the last phase's steps and
residual, the parameter version, rejected updates, record writes, real transitions and
imagined reads, the exploration rate) and the body counters. The body clock plays the copy at
the arm's own rate of ten decisions per second by default, and ×0.1 slows it until every
settling step is visible; the brain playback keeps pace with the body by itself. Exploration ε
is the probability of a uniformly random torque instead of the planner's choice and starts at
zero, as the copier evaluation of `run.py` measures it.

The layout is one measure: the stage height that the arm canvas and the brain share. At 1280 by
800 both sit inside the first screen side by side with the controls and readouts around them;
narrower screens stack the arm above the brain, and at phone width the records cortex moves
under the scan.

## The records head in the engine

`brain.py`'s `RecordsHead` is ported as `RecordsHead` in `engine.js`, with the same
Mulberry32 draws (`Mulberry32.batch(n)`: the draw at index j is the hash of
`(state + j * 0x6D2B79F5) mod 2^32`, bit-identical to n calls of `random`; `normals(n)`: the
Box-Muller pairs of `batch(2 * ceil(n / 2))`, r cos then r sin), so the expansion `W` and
the bias `b` are regenerated from the seed on the page: the snapshot carries only the
config, the seed, the reading port indices, the input pathways (index arrays into the
reading: observation, goal, action, recall, context, empty ones dropped) with their running
norms, the habituation mean (float64), the records tables (float32 by default; the tables
are the bulk of the export) and the write count. `code`: the reading minus each unit's
running mean when habituation is on (the mean follows the executed reading only), then two
codes from the one expansion (`_winners`: `h = x @ W + b` summed over the inputs in index
order, the `active` largest entries kept and rectified with a min-heap over one pass, the row
normalised with numpy's pairwise sum over the whole row): the plain code of the habituated
reading, read by the consequence records, and the valued code, in which every pathway is
divided by its running norm plus 1e-3 (the norm follows the executed reading only, at
`block_rate`), read by the reward and terminal records (`VALUED_SOURCES`); without
`normalize_blocks` or without pathways the valued code is the plain one. `predict`: each
field's records read through its code's active cells in index order, the positive part
normalised for a categorical field (uniform when nothing is read), the sensor range for a
continuous one; `learn`: the delta rule `R[g] += rate * (code[g] * err)` on the active rows
of the field's code, the reward rate for reward and terminal fields. Codes are kept sparse
(`{index, values}` pairs `{plain, valued}`): a read or a write touches only the active rows.
The transaction takes the records path exactly as `Agent.step_batch` does: the executed
reading is coded in `_decide` after the planner's imagined reads (the mean and the norms
adapt there and only there), the outcome is written into the pending code's records in
`_repair` (no ring, no settle, `records_written` in the learning report), and `predict` and
`predictBatch` are record reads (`ledger.imagined` counts them; `predictBatch` takes an
`observed` flag map per row, as the Python signature does). `snapshotState()` carries the
mean, the pathway norms, every table and the write count; `parameters()` counts the records.

Two things differ from Python at rounding level and are visible in the parity numbers: V8's
`Math.log`, `Math.cos` and `Math.sin` differ from the libm numpy calls by one ulp in about
7 percent of the draws (measured on 200,778 normals: max 8.9e-16, no other difference), so
`W` and `b` agree to 1e-16; and the products `x @ W` and `code @ R` are BLAS reductions in
Python, summed in index order here. The winner-take-all selection is discrete, so these
differences could flip a cell only at an exact tie of the k-th largest entries, which the
fixtures do not contain.

## The parity check

```sh
/opt/homebrew/bin/node web/parity.mjs runs/arm/parity

# the same stream from the snapshot's float32 tables (the page's tables)
/opt/homebrew/bin/node web/parity.mjs runs/arm/parity --no-sidecar

# the pending record and the ring from a float64 supplement (life_supplement.py)
$PY arm/web/life_supplement.py \
    --snapshot runs/arm/parity/snapshot.json --out runs/arm/web/life_supplement.json
/opt/homebrew/bin/node web/parity.mjs runs/arm/parity \
    --supplement runs/arm/web/life_supplement.json
```

The harness loads `snapshot.json`, the float64 records sidecar `records_f64.json` when its
efficacy digest matches the snapshot (`--no-sidecar` keeps the float32 tables), installs the
S01 planner from `snapshot.extra.planner`, replays the 120 moments of `stream.jsonl` and
reports: decisions matching (action, controller), the largest prediction difference,
parameter-version and efficacy-sha256 agreement per line, the record-write and
imagined-read counts per line, and the final differences of efficacy, bias, world velocity,
critic, context trace, the records mean, the pathway norms, every records table and the
generator states against `final.json`. It exits non-zero if any decision differs, a count differs, or a
difference exceeds the tolerances recorded in `final.json` (records: 1e-10). The planner's
record of what it imagined (`lastImagined`) is read by the page only; the search and its
decisions are unchanged, and the harness reports `PARITY OK`.

Results on the fixture of 2026-09-15 (32,000 granules, 160 active, 47 reading neurons in 3
pathways, 3 fields, 192,000 records; 120 moments, 119 decisions; 2 to 6 s):

| tables | decisions | prediction | records mean, pathway norms | records tables (dd_hand, d_velocity, d_angles) | writes, imagined | efficacy, bias, world velocity | critic w / b | context trace | rng, versions, phases |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sidecar (float64) | 119/119 | 3.3e-16 | 0, 0 | 1.4e-16 (8.3e-17, 1.4e-16, 1.4e-16) | 1254 / 1782, exact on every line | 0 | 6.5e-19 / 0 | 2.2e-16 | exact |
| snapshot (float32) | 119/119 | 1.1e-9 | 0, 0 | 7.2e-9 | exact | 0 | 6.5e-19 / 0 | 2.2e-16 | exact |

The float64 run reports `PARITY OK`. The float32 run reports `PARITY FAILED` on the records
tolerance alone: the page's tables are rounded at the 1e-8 level and every write moves them
from the rounded values, so the final tables sit 7e-9 from Python's; every decision and count
still agrees. The per-line efficacy digest agrees byte for byte on every line in this fixture:
with the records head the world repair moves no synapse, and the planner's decisions carry no
motor eligibility, so the efficacy stays at the snapshot's values (parameter version 0
throughout). The `--supplement` run prints the same numbers as the sidecar run; the supplement
carries the pending decision's code and the ring is empty under the records head.

## The headless check

`window.__page` exposes `{scan, agent, arm, planner, records, queue, stats, copy, log(), step(),
run(n), pause(), play(), selectCheckpoint(id), toClient(x, y), ready}`; `run(n)` computes n
events and shows them at once, which is how `check_page.py` drives the page after drawing on it
with real pointer events. The check loads the page with its demo off
(`window.__ARM_PAGE__ = {autoplay: false}`), so the measurement repeats.

The tracking bound is judged only when the brain has lived at least `--tracking-from`
decisions (2,000 by default). A brain with a few hundred babbled decisions has no world model
worth planning through and copies nothing: the arm swings around the workspace and the error
is larger than standing still. Everything else the check measures is judged on every snapshot.

Two pages of 2026-09-15, both drawn with the same figure eight through pointer events, both
with zero page and console errors:

| the check | the parity fixture, 299 decisions lived | a snapshot babbled for 5,000 decisions |
| --- | --- | --- |
| copy 1 of the drawn figure, 162 decisions with the target moving | mean tracking error 0.171, ink 0.046 from the drawing | 0.065, ink 0.010 |
| copy 2 of the same figure | 0.073, ink 0.009, largest 0.110 | 0.066, ink 0.008, largest 0.096 |
| a hand that never moved, over the same two copies | 0.229 and 0.214 | 0.213 and 0.212 |
| the bound of 0.12 on the last copy | reported, not judged (299 decisions) | judged, passed |
| the page's readout against the log recomputed in Python | equal to 1e-15 | equal to 1e-15 |
| two strokes drawn as one figure | 2 strokes, 156 points, 52 of them travel with the pen up | the same |
| Clear | the surface goes idle | the same |
| the brain selector | no manifest beside the page, so the one brain | switches to the 300-decision brain over http and copies a circle with it |
| the layout at 1280 by 800 | the arm canvas 372 by 372 at (32, 197), the brain scan 538 by 374 at (444, 193), the records cortex at (1000, 288), the readouts ending at 777: side by side inside the viewport, nothing scrolling sideways | the same |
| the layout at 390 by 844 | the arm canvas at the top, the scan below it, nothing scrolling sideways | the same |
| seconds per event, headless SwiftShader drawing the whole page | 0.04 | 0.04 |

A brain this young copies chaotically. Moving the drawn path by one pixel (a label in the
page changed width, so the canvas moved sixteen pixels) took the same two copies of the parity
fixture from 0.386 and 0.240 to the 0.171 and 0.073 in the table, while the trained column
repeated to every digit. That is what the bound's experience threshold is for.

## Notes

- The page continues the snapshot's life: the body is restored from `extra.next_moment` (the
  moment the Python agent was fed next; angles from their sines and cosines, velocities, the
  previous step from the measured deltas, the target from the goal marker; the success hold
  is not observable and starts at zero), that moment is fed back first, so the pending
  decision receives its outcome (its code was exported with it) and the page's first decision
  is the fixture's (decision 300, planner). A snapshot without such a moment starts a new
  episode on the page's own arm after `agent.abandon(0)`, the evaluation-copy rule of
  `Agent.abandon`. The page's records tables are the snapshot's float32 tables, widened.
- One copy is one episode. It begins with the body where the last one left it (the angles stay,
  the velocities start at zero, as every episode does) and ends when the target has rested on
  the last point for one second, which the page reaches by asking the arm to truncate, the way
  a time limit would. The episode's horizon covers the longest approach, the path and that
  rest.
- A snapshot without a records head takes the settled path as before (the predict settle, the
  world repair with the replay ring, imagined settles); the records panel stays hidden.
- All modules use one-line `import` statements and `export const/function/class`
  declarations only; `build_page.py` strips those, checks that no two modules declare the
  same top-level name, and concatenates them into one module script.
