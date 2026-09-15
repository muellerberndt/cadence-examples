# The S01 arm in the browser

The experience agent of `agent/brain.py` ported to JavaScript, the arm
of `arm/env.py` and its planner alongside it, and one page that
shows the body and the whole brain while it learns. The engine is the same numerics as the
Python agent on the cadence CPU backend, and it carries the records world head of the
current config (`RecordsHead`: the sparse-expansion world model, cerebellum-like and
dentate-like) beside the settled regions; a parity harness replays the recorded fixture
through it and compares every decision, prediction, the learned parameters and the records.

## Files

| file | what |
| --- | --- |
| `../../web/engine.js` | `ExperienceAgent` (one stream) built from a `cadence-experience-web/1` snapshot: `step(moment)`, `predictBatch(observations, actions, goal, {observed})`, `snapshotState()`, `setPlanner(fn)`, the `onSettleStep`, `onPhase`, `onLearn` and `onRecords` callbacks; `Mulberry32` with `batch` and `normals`; `RecordsHead`; `normalizeMoment`; `encodeObservation` on the agent. Float64 everywhere. |
| `../../web/records_view.js` | `RecordsView` (the records cortex on a Canvas2D: the granule raster, the writes, the habituated reading) and `readsHTML` (the per-field record reads as HTML rows). Shared by both pages. |
| `../../web/parity.mjs` | `node web/parity.mjs runs/arm/parity`: the parity check against the Python fixture. |
| `../../web/brain_scan.js` | The standard renderer, a verbatim copy of `cadence/src/cadence/brain_scan.js` (never edited here). |
| `planner.js` | `compose` and `ModelPlanner` from `s01_arm/brain.py`: the bounded beam search through the model's imagined consequences, stable sort. |
| `arm.js` | The arm physics from `env.py` (`Arm`, `TORQUES`, `FIELDS`, `wrap`, `movingPath`) with a Mulberry32 random stream and `requestTruncation()` so a page can move to a new target at an episode boundary. |
| `page.js`, `index.html`, `style.css` | The page. |
| `build_page.py` | Inlines the snapshot, the renderer, the engine, the records view, the planner, the arm and the page into one `index.html` that works from `file://` and on GitHub Pages. |
| `check_page.py` | Headless Chromium (SwiftShader) check: loads the built page, runs events through `window.__page.step()`, saves screenshots, fails on any page or console error. |
| `life_supplement.py` | Regenerates the snapshot's life records in float64 (the pending decision with its records code, and the full replay ring with warm potentials): the fallback for an export from before `web.py` wrote `pending` and `ring.v`/`ring.a`. |
| `../tools/parity_fixture.py` | Writes the fixture: `snapshot.json`, `records_f64.json`, `stream.jsonl`, `final.json`. |

## Build and run

```sh
cd /Users/muellerberndt/Projects/oph-meta/cadence-paper
PY=/Users/muellerberndt/Projects/oph-meta/cadence/.venv/bin/python

# the fixture (a few seconds): 300 babbling decisions, the snapshot, then 120 recorded moments
# at exploration rate 0.5; writes the float64 records sidecar next to the snapshot
$PY arm/tools/parity_fixture.py --out runs/arm/parity

# the parity check (about 2 s)
/opt/homebrew/bin/node web/parity.mjs runs/arm/parity

# the self-contained page (3.8 MB): open it from the file system or serve it anywhere
$PY arm/web/build_page.py \
    --snapshot runs/arm/parity/snapshot.json --out runs/arm/web/index.html

# the headless check: zero console errors, 50 events, screenshots next to the page
/Users/muellerberndt/Projects/oph-meta/cadence-artist/.venv/bin/python \
    arm/web/check_page.py runs/arm/web/index.html --events 50
```

For development the sources stay separate: serve the repository root over http
(`python -m http.server 8000`) and open
`http://localhost:8000/arm/web/index.html`; `page.js` then fetches
the snapshot named by `<body data-snapshot>`. The Python side changes; regenerate the fixture,
rerun the parity check and rebuild the page in that order.

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
difference exceeds the tolerances recorded in `final.json` (records: 1e-10).

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

## What the page shows

Left, the body: the two links, the hand, the target with its success radius, the hand
trail, and from the hand the displacement the model predicted for the executed decision
(teal) against the one that happened (orange), drawn at five times their size. Below it
the counters (episode, tick, events, decisions, episodes ended, successes, distance, decisions
inside the radius, last reward, controller, prediction error, target mode, link lengths,
compute time per event) and the body controls: a new target, static or moving targets
(line, circle, mixed; the paths of `env.py`), lengthening the upper link by 15% (the model
must adapt), restoring the body. A new target or a new target mode ends the episode at the
next event as a time limit would, so the decision awaiting its outcome is fed back first.

Right, the whole brain. The scan through `BrainScan`: every neuron coloured by its region
(sensory, goal, efference, workspace, context, dynamics, prediction, motor: the colours of the
snapshot's atlas) and every synapse, animated per settling step of every phase the settled
regions run (free, the free phase after learning, the bootstrap value at a time limit); the
phase label names the phase and, for a batch, the row shown. Beside the scan, its own
labelled and coloured region, the records cortex (cerebellum-like / dentate-like; the legend
lists it with the atlas regions): the granule raster, all 32,000 cells in a compact grid with
the 160 active cells of the executed reading's plain code lit by their code value (blue) and
the valued code's cells in pink (the arm has no reward record, so no field reads it); imagined
reads (the planner's batches of nine torques, held for three decisions) as brief grey
flickers on the cells they touch; the writes as flashes on the active cells when the outcome
arrives, scaled by the record rate (a reward record would flash the valued code's cells
pink); under the raster the reading the expansion sees as bars from a midline with the ports
separated by ticks and named (sensors, missing flags, goal, action; habituation is off in
this config, so the reading is the raw drive) and the running norm of every pathway
(observation, goal, action) the valued code divides by; and under that the per-field record
reads, the predicted next displacement, velocity change and angle change on their sensor
ranges, with the largest error of the last write per field. Every code, imagined read and write is queued
with the settle frames and played in order. Below: the EEG montage of every region, the
dopamine bar (the critic's clipped TD error per event, signed), and the ledger (the last
phase's steps and residual, rejected updates, unconverged phases, parameter version, record
writes, real transitions, imagined reads, phase counts, the exploration rate). The
exploration slider sets ε, the probability of a uniformly random torque instead of the
planner's choice. Learning never stops while it plays; there is no scripted success path.

`window.__page` exposes `{scan, agent, arm, records, queue, stats, step(), ready}`; `step()`
computes one event and shows it at once, which is how `check_page.py` drives the page. The
check of 2026-09-15 (50 events, then the controls) ran with zero page or console errors and
wrote `page_start.png`, `page_mid.png`, `page_after.png`, `brain_after.png`, `arm_after.png`
and `records_after.png` next to the page; after 50 events the records cortex had shown 50
codes, 837 imagined reads and 150 field writes over 160 active cells.

## Notes

- The page continues the snapshot's life: the body is restored from `extra.next_moment` (the
  moment the Python agent was fed next; angles from their sines and cosines, velocities, the
  previous step from the measured deltas, the target from the goal marker; the success hold
  is not observable and starts at zero), that moment is fed back first, so the pending
  decision receives its outcome (its code was exported with it) and the page's first decision
  is the fixture's (decision 300, planner). A snapshot without such a moment starts a new
  episode on the page's own arm after `agent.abandon(0)`, the evaluation-copy rule of
  `Agent.abandon`. The page's records tables are the snapshot's float32 tables, widened.
- A snapshot without a records head takes the settled path as before (the predict settle, the
  world repair with the replay ring, imagined settles); the records panel stays hidden.
- All modules use one-line `import` statements and `export const/function/class`
  declarations only; `build_page.py` strips those, checks that no two modules declare the
  same top-level name, and concatenates them into one module script.
