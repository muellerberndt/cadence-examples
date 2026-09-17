# S04: the artist draws what it is shown

The arm of S01 holds a pen over a 32 by 32 canvas. One life scribbles, learns what its
strokes leave on the canvas, and then draws requested figures: single segments, two
connected segments, polygons, curves and compositions it has never seen, replanning from
what the canvas shows after every stroke. A longer upper link and an offset canvas change
the body and the world during a drawing; the artist keeps drawing and learns the change.
Every decision and its outcome go through the S00 experience contract.

## Supplied and learned

Supplied: the arm body of S01 with its own goal switched off, the canvas with stroke
rasterisation at one pixel width, the pen state as one more action bit (nine torques by
pen up or down), the two aligned views of the target and the canvas, the discrepancy
measure the reward and the evaluation use (symmetric Chamfer distance normalised by the
canvas diagonal, with foreground F1 within one pixel reported beside it), the reward as
the change in measured discrepancy less an action cost, the target families and their
splits, and the two-level search: a slow level that chooses a stroke intention (direction,
distance, pen state) by imagining its marks, and a fast level that holds each candidate
torque for a few decisions and follows the intention through the learned body.

Learned: the records of the world head (a records cortex over the body senses, the 8 by 8
ink window around the pen, the target window, the intention and the action, with fields
for the displacement change, the velocity change, the angle change and the marks the pen
leaves), and the synapses of the settled regions, from empty records and random synapses.
There is no replay ring: each outcome is written once into the records its reading touched.

Controls: the same architecture frozen from birth, random scribbling, the supplied path
oracle, an online MLP with a replay ring behind the same search, corrupted dynamics,
shuffled action pairing, replanning removed, and the canvas readback removed.

## Files

| file | contents |
|---|---|
| `env.py` | the canvas world over the S01 arm: rasterisation, pen, windows, discrepancy and F1, families and splits, reward |
| `brain.py` | the graph over the S00 agent with its records head, the intention search and the torque search, `ArtistLife` |
| `controls.py` | the random scribbler, the path oracle and the online MLP behind the same search |
| `run.py` | the progression (scribble, segments, strokes with replanning, compositions, perturbations), held-out suites, controls, receipt, checkpoints |
| `verify.py` | recomputes every gate from the event log, checks digests, source hashes and the seed schedule, fails closed |
| `config.json` | budgets, gates and the recorded assumptions |
| `tests/` | rasterisation, pen-up travel, window alignment, discrepancy and F1 on known canvases, splits, reward, the graph contract, imagination writing nothing, copy isolation |

## Commands

```bash
PY=../cadence/.venv/bin/python
$PY -m pytest artist/tests -q
$PY artist/run.py --seeds 0 --pilot --out runs/artist/pilot        # a tenth of every budget, about ten minutes
$PY artist/run.py --seeds 10 11 12 13 14 --workers 5 --out runs/artist/acceptance   # about two hours, 9 GB
$PY artist/verify.py runs/artist/acceptance
```

## The page

`artist/web/` builds one self-contained HTML file. A visitor draws a figure on the canvas with
a mouse, a pen or a finger, or picks one of the target families, and the artist draws it stroke
by stroke: it commits a stroke intention, follows it with torques, lifts the pen between
strokes and looks at its own canvas again every second decision. The whole brain settles beside
the canvas at the same size, with the records cortex showing the reading, the active cells, the
grey flickers of the reads the search makes and the orange flash of every write. A brain
selector carries the same life at five points, and a HUD reads out the discrepancy and the F1
of the drawing in progress.

The engine is the shared port of `agent/brain.py` with its records head (`web/engine.js`); the
renderer and the records cortex are the shared `web/brain_scan.js` and `web/records_view.js`;
the fast level's leaf objective is `arm/web/planner.js`, the port of the S01 planner that
`brain.py` imports. `artist/web/artist.js` is the port of this stage.

| file | contents |
|---|---|
| `web/artist.js` | the canvas world of `env.py` (rasterisation, the pen, the two pen-centred windows, the Chamfer discrepancy and the foreground F1, the target families), the planning objective `Sheet`, the body, and the intention search, the torque search and the life of `brain.py`. Float64 throughout, numpy's pairwise sums, numpy's round-half-to-even, Mulberry32 where Python seeds a numpy generator |
| `web/page.js`, `web/index.html`, `web/style.css` | the page: the canvas and the arm holding the pen, the target and canvas thumbnails the brain reads, the controls, the HUD, the brain beside it and the receipt of the run |
| `web/build_page.py` | `--run` a run directory or `--snapshot` one export; inlines the snapshot, the renderer, the engine, the records view, the S01 planner, the artist and the page into one HTML that works from `file://` and on GitHub Pages, copies one checkpoint per phase beside it and writes the `checkpoints.json` manifest the brain selector reads |
| `web/check_page.py` | headless Chromium (SwiftShader): the layout at 1280 by 800 and at 390 by 844, a figure drawn through pointer events and drawn twice by the artist, the measures recomputed against the page's readout, every family, Clear, the brain selector, the decision budget, the screenshots, and no page or console error |
| `web/parity.mjs` | replays the recorded fixture through `artist.js` and holds every moment, intention, decision, prediction and record table to the Python numbers |
| `tools/parity_fixture.py` | writes the fixture: `snapshot.json`, `records_f64.json`, `drawings.json`, `stream.jsonl`, `final.json` |

### Build and check

```bash
PY=../cadence/.venv/bin/python
NODE=/opt/homebrew/bin/node

# the page: the final brain inlined, four earlier brains beside it (about a minute)
$PY artist/web/build_page.py --run runs/artist/pilot --out runs/artist/web/index.html

# the fixture (about ten seconds) and the parity check (about ten seconds)
$PY artist/tools/parity_fixture.py \
    --checkpoint runs/artist/pilot/artist_seed0_scribble.npz --out runs/artist/parity
$NODE artist/web/parity.mjs runs/artist/parity

# the headless check (about five minutes), with the screenshots
../cadence-artist/.venv/bin/python artist/web/check_page.py runs/artist/web/index.html
```

The committed page is the acceptance run built into `artist/index.html` with
`artist/checkpoints.json` beside it:

```bash
$PY artist/web/build_page.py --run runs/artist/acceptance --seed 10 --out artist/index.html
../cadence-artist/.venv/bin/python artist/web/check_page.py artist/index.html --out runs/artist/web
$PY tools/checkpoint_assets.py prepare artist <release>   # prints the gh release upload line
```

Any run directory works. The checkpoints are the `artist_seed<seed>*.npz` files the receipt
lists under `artifacts.checkpoints`, their sha256 checked against it; one is taken per phase
(after the scribbling, after the segments, after the strokes, after the compositions, the final
brain) and `--all-checkpoints` keeps every file. The final brain is inlined, the other four are
written to `checkpoints/<id>.json` beside the page and listed in `checkpoints.json`, which is
the manifest format the other stages use and `tools/checkpoint_assets.py` turns into release
assets. Choosing another brain rebuilds the agent, the scan and the records view in place and
draws the same figure again with it. The receipt's numbers for that
seed travel in the snapshot and the page prints them under the controls.

`--snapshot runs/artist/parity/snapshot.json` inlines one export instead, which is what the
parity fixture writes. For development the sources stay separate: serve the repository root
over http and open `http://localhost:8000/artist/web/index.html`, where `page.js` fetches the
snapshot named by `<body data-snapshot>`.

### Parity

The harness loads the snapshot, rebuilds the life, replays the recorded drawings from their
targets and body poses through its own canvas world, and compares every step with Python: the
moment the world produces, the intention the slow level commits, the action and the controller
exactly, and the predictions, the policy, the objective and the record tables to the fixture's
tolerances. On the fixture of 2026-09-16 (the 2,000-decision brain of the pilot, 2 drawings,
160 decisions, 32,000 granules, 160 active, 238 reading neurons, 2,240,000 records) every
decision, intention and moment agrees and the largest relative differences are 2.2e-16 on the
moments, 5.6e-16 on the reads, 5.8e-16 on the intentions, 4.4e-16 on the objective and 1.7e-16
on the records tables: `PARITY HELD`. `--no-sidecar` reads the page's own float32 tables
instead, where every decision, intention and moment still agrees and the tables sit 7.6e-9 from
Python's, above the 1e-9 tolerance, because the export rounds them at the 1e-8 level and every
write moves them from the rounded values.

### Two functions live in the stage rather than in the shared modules

`CanvasArm` in `web/artist.js` is the S01 body rather than `Arm` of `arm/web/arm.js`. That port
wraps an angle with two moduli, which costs an ulp of 2π per body step against Python's
`(angle + π) % (2π) − π`; 210 body steps of drift move a pen pixel and part two intention
searches that agree to 1e-13. Everything else is the same body, and the joint state now moves
by additions, multiplications and one modulus, so it stays bit-identical to Python's.

`artistReadsHTML` in `web/page.js` is `readsHTML` of `web/records_view.js` with one change: a
prediction field wider than eight values is drawn as the square patch it is, since the ink head
predicts 64 cells and a track per cell does not fit the panel.

### What the check measures

Two pages of 2026-09-16, both drawn with the same open V of 29 target pixels through pointer
events, both with zero page and console errors: the run page of the pilot at seed 0 (the final
brain, 5,198 decisions lived, four brains beside it) and the snapshot page of the parity fixture
(the brain after the scribbling, 2,000 decisions lived, no manifest and no receipt).

| the check | the final brain | the brain after scribbling |
| --- | --- | --- |
| drawing 1 of the figure | 183 decisions, F1 0.885, Chamfer 0.0098, 24 pixels inked | 179 decisions, F1 0.916, Chamfer 0.0093, 39 pixels inked |
| drawing 2 of the same figure | 197 decisions, F1 0.879, Chamfer 0.0163, 32 pixels inked | 166 decisions, F1 0.967, Chamfer 0.0099, 31 pixels inked |
| the bar, the largest F1 gate the receipt reports | 0.80, judged, passed | 0.70 without a receipt, reported only (2,000 decisions) |
| the page's readout against the canvas recomputed in Python | equal to 1e-9 | the same |
| two strokes drawn as one figure | 6 strokes left on the canvas | 5 strokes |
| every target family | all five start a drawing | the same |
| Clear | the page goes idle | the same |
| the brain selector | switches to the 2,000-decision brain and draws a segment with it | no manifest beside the page, so the one brain |
| the layout at 1280 by 800 | the canvas 344 by 344 at (32, 180), the brain scan 566 by 346 at (416, 180), the records cortex at (1000, 275), the canvas column ending at 767 and the brain column at 749: side by side inside the viewport, nothing scrolling sideways | the same, the canvas column ending at 726 without the receipt line |
| the layout at 390 by 844 | the canvas at the top, the scan below it, nothing scrolling sideways | the same |
| seconds per decision, headless SwiftShader drawing the whole page | 0.09 to 0.11 | 0.11 |

The final brain draws this figure with precision 1.00 and recall 0.79, the 2,000-decision brain
with precision 0.97 and recall 0.97: the older brain marks fewer pixels and every one of them
lands on the target, the younger one covers the figure and pays for it in precision. The two
sit on either side of the receipt's held-out F1 of 0.94, which is the mean over 20 held-out
drawings of five families rather than one figure a visitor drew.

The F1 bar is judged only when the brain has lived at least `--judge-from` decisions (3,000 by
default), so a page built early in a life reports its F1 without being held to a bar its ink
model has not been shown to reach. The decision budget (`--max-decision-ms`, 400 ms) is judged
on every page: a decision has 100 ms in Python and the page runs the same two-level search in
JavaScript.

## Assumptions recorded in config.json

The canvas spans 0.7 arm lengths centred half an arm length above the shoulder, so every
target pixel lies between 0.15 and 0.92 of the reach and off the arm's fold. The ink head
predicts the 8 by 8 window around the pen at canvas resolution, since one decision moves
the pen about 1.4 pixels and a coarser window cannot resolve a one-pixel stroke. The
intention travels in the goal port, so the executed reading and the imagined readings
carry the same fields. The action's neurons are driven at three times the input gain, which
separates the pen states in the code. The planning objective caps each target pixel's
reach at 1.5 pixels and a drawing finishes when the discrepancy falls under 0.010.

## Evaluation copies

Held-out drawings run on an adapting copy of the checkpoint with exploration off: the copy
keeps writing its records while it draws, which is the ongoing mode. The lesions (corrupted
dynamics, shuffled pairing) are read on the first ten drawings, since an adapting copy
heals a lesion within a few dozen drawings; the readback control removes the canvas
observation after the perturbation and must lose.

## Receipt

`receipt.json` is the acceptance receipt of seeds 10 to 14: every predicate with its value
and threshold, the source hashes of this directory and of the library, the sha256 of every
event log and checkpoint, the learning curve, and the digest of the receipt itself.
`verify.py` recomputes the gates from the event logs and fails closed on an incomplete run.
