# The S02 world in the browser

Ask for an object and the agent goes to the place it remembers. The remembered world of
`world/` ported to JavaScript: the world and its curriculum (`env.py`, `tasks.py`), the four
stores and the best-first search with the `Brain` wrapper (`brain.py`), on the experience
agent of `../../web/engine.js` with its records world head, and one page that shows the world
and the whole brain side by side (the settled regions in the scan, the records cortex beside
it) while the visitor asks for objects, moves them, teaches words, runs the cue task and locks
the doors.
Two parity harnesses compare the port with recorded Python fixtures: the S01 harness for the
engine itself, and `parity_s02.mjs` for the brain (every decision, the stores, the recall
port, the planner's counters and plans, the learned parameters, the records) and for the
world (every moment and the private state after every event).

## Files

| file | what |
| --- | --- |
| `../../web/engine.js` | `ExperienceAgent`, with the records head (`RecordsHead`; see `../../arm/web/README.md`), the `recallDrive` hook of `Agent.recall_drive` (`setRecallDrive(fn, last)`) and `predictBatch(observations, actions, goal, {observed})` taking one flag map per imagined reading, as `Agent.predict_batch` does. |
| `../../web/records_view.js` | `RecordsView` and `readsHTML`: the records cortex on the page. |
| `world.js` | `World` (the map in the same edge order, `act`, `observation` with the eighteen fields including the displacement of the last action `dx`, `dy`, `reset`, `shortestPath`, the privileged accessors, `state()` and `World.fromState`), `Curriculum` (`explore`, `remember`, `request`, `moveObject`, `cue`, `armCue`, `hideCue`, `word`, `nameRequest`, `lockDoors`, `farCandidates`), `requestDeadline`, `goalVector`, `manhattan`, the constants. Reward rules and the joint-attention word rule are data (`{kind, ...}`). Random numbers from a Mulberry32 stream: the page's life is its own. |
| `stores.js` | `PlaceStore`, `MapStore` (cell to its four passages with a known flag: the witnessed map), `CueStore`, `WordStore` on `DeltaSynapses` (cadence `FastSynapses`, rule "delta") and `ConsolidatingSynapses` (cadence `SynapticMemory`), element by element in numpy's evaluation order. |
| `planner.js` | `compose` (the position composes from the predicted displacement; a known next cell takes its passages from the map store), `observeFrom`, `stateKey`, `IMAGINED_OBSERVED`, `WorldPlanner` (a binary heap ordered by (f, g, id) as heapq orders the Python tuples, the memo per visible state, depth-0 marker and action, the same batch composition of every `predictBatch`, pruning, the closed set and best-cost table, the model-based fallback with the Mulberry32 tie-break, `greedyReward`) and `Brain` (`witness` into the four stores, the recall read, the exploration target, plan reuse, `step`, `setEpsilon`, `counters`, `stores`, `bookkeeping`). |
| `parity_s02.mjs` | `node world/web/parity_s02.mjs runs/world/parity`: the S02 parity check. |
| `../tools/parity_fixture.py` | Writes the fixture: `snapshot.json`, `records_f64.json`, `stream.jsonl`, `world.jsonl`, `final.json`. |
| `page.js`, `index.html`, `style.css` | The page. |
| `build_page.py` | Inlines the snapshot, the renderer, the engine, the records view, the world, the stores, the planner and the page into one `index.html`. `--run` exports every checkpoint `run.py` saved for one seed, inlines the final brain and writes the `checkpoints.json` manifest the brain selector reads. |
| `check_page.py` | Headless Chromium (SwiftShader): the layout at 1280 by 800 and at 390 by 844, a request the agent answers, an object dragged behind its back, a word taught and asked by, the cue task, the doors, run.py's phases, the brain selector, the time an event takes, screenshots, and no page or console error. |

## Build and run

```sh
cd /Users/muellerberndt/Projects/oph-meta/cadence-examples
PY=/Users/muellerberndt/Projects/oph-meta/cadence/.venv/bin/python

# the fixtures (about 15 s each): 320 explore steps at rate 1, the snapshot at the episode
# boundary, then one explore episode, two remembered requests at delay 8, two cue episodes at
# delay 8 and one word episode through run.py's own run_explore, run_remember, run_cue and
# run_words; writes the float64 records sidecar next to each snapshot
$PY world/tools/parity_fixture.py --out runs/world/parity
$PY world/tools/parity_fixture.py --out runs/world/parity_gains --graph-override \
    '{"graph": {"field_gains": {"x": 1.5, "y": 1.5, "here_object": 2.0, "word": 0.5}, "action_gain": 3.0},
      "records": {"pathways": "fields", "fan_in": 3}}'

# the parity checks (about 5 s each; the arm fixture too, because the engine is shared)
/opt/homebrew/bin/node web/parity.mjs runs/arm/parity
/opt/homebrew/bin/node world/web/parity_s02.mjs runs/world/parity
/opt/homebrew/bin/node world/web/parity_s02.mjs runs/world/parity_gains

# the page a visitor sees: the final brain of a run inlined, its checkpoints beside it
$PY world/web/build_page.py --run runs/world/dev --out runs/world/web/index.html

# the headless check (two to four minutes): the layout, a request, a drag, a word, the cue
# task, the doors, run.py's phases, the brain selector, the screenshots
/Users/muellerberndt/Projects/oph-meta/cadence-artist/.venv/bin/python \
    world/web/check_page.py runs/world/web/index.html
```

`--run` takes any run directory: the checkpoints are the `world_seed<seed>*.npz` files the
receipt lists under `artifacts.checkpoints` (explore at a tenth, a half and all of the explore
budget, then after the remembered requests, after the cue task, after the doors locked, and
the final brain), in the order the run saved them, each sha256 checked against the receipt.
`--seed` picks the seed (the first completed one by default). The world of that life comes
back from its own seed, so every brain of the run meets the map it learned, and the numbers of
the run's receipt travel in the snapshot and appear in the page's last paragraph. `Agent.save`
holds the brain alone, so a brain loaded from the selector starts with empty stores and fills
them by wandering. One exported snapshot can be inlined instead (`--snapshot`, what the parity
fixture writes), and `--checkpoint "label=path"` names snapshots directly.

For development the sources stay separate: serve the repository root over http
(`python -m http.server 8000`) and open
`http://localhost:8000/world/web/index.html`; `page.js` then fetches
the snapshot named by `<body data-snapshot>`.

The fixture's world uses the stage config's map (5 doors, 10 walls) with a horizon of 32
(`--horizon`) so the stream's explore episode stays short; `--babble` sets the explore steps
before the snapshot, `--remember`, `--cue`, `--words` the stream's episodes, `--max-nodes`
bounds the search (the value travels in `extra.planner`). The Python side changes; regenerate
the fixture, rerun both checks and rebuild the page in that order.

## The fixture

`parity_fixture.py` drives run.py's loop functions with a recording `Life` (one stream line
per moment the brain is fed: the moment, the exploration rate in effect, the decision, the
parameter version, the efficacy digest, the learning report, the planner counters, whether
the decision searched or reused a plan, the plan, the stores, the recall read, the
record-write and imagined-read counts) and a recording `World` (one `world.jsonl` line per
reset or act: the privileged operations the curriculum applied before it, the event, the
moment it produced, the private state after, including the last displacement). The reward
rules are identified from their closures (`_reach_reward`: target; `arm_cue`: paid room),
the word rule from its defaults (object, word); an unknown rule stops the tool. The
snapshot's `extra` block carries `config` (the stage config), `planner` (budget, seed,
generator state, fallbacks, pruned), `brain` (visited, inspected, plan, plan cells, plan key,
searches, reuses, expansions), `stores` (the strengths, masses and write counts of the place,
cue, word and map stores in the shape the JS stores load), `recall` (the last read and the
gain), `world` (the state dump), `curriculum` (the cue rule) and `steps_before`; the
snapshot's `records` block carries the head (see the S01 README), `final.json` the head in
float64 after the stream.

`--graph-override` merges settings over the stage config, so a fixture can carry what the
stage config leaves at its default and the port is checked on those paths too. It takes a
mapping of section to settings (`{"graph": {...}, "records": {...}}`) or a flat mapping of
graph settings. The second fixture, `runs/world/parity_gains`, uses
`graph.field_gains {x: 1.5, y: 1.5, here_object: 2.0, word: 0.5}` (a factor on the input gain
per observation field, the field's missing flag keeping the plain gain), `graph.action_gain
3.0` (the drive of the chosen action's neurons), `records.pathways "fields"` (one input
pathway per observation field with its flag instead of one for the whole observation port: 21
pathways here, not 4) and `records.fan_in 3` (each cell reads three of those pathways, drawn
at birth from the same generator after the task groups, and every input outside the pathways;
its column of the expansion is zeroed elsewhere and rescaled by the inputs it keeps, which
leaves 86% of the expansion at zero).

## The parity check

The brain section builds the agent (with the float64 sidecar `records_f64.json` when its
efficacy digest matches; `--no-sidecar` keeps the snapshot's float32 tables) and the `Brain`
from the snapshot and replays `stream.jsonl`: `brain.setEpsilon(line.epsilon)`, then
`brain.step(line.moment)`. It reports decisions matching (action and controller; ids and the
search's expansions must agree too), the largest prediction, probability and learning-report
differences, the parameter version and the efficacy digest per line, the planner counters
and the plan per line, the largest store and recall-port differences per line, the
record-write and imagined-read counts per line, and the final differences of efficacy, bias,
world velocity, critic, context trace, stores, recall, the records mean, the pathway norms
and every records table against `final.json`, with the generator states, the ring cursors,
the counters, the write counts and the ledger phase counts exact. The world section loads `extra.world`,
applies every operation and event of `world.jsonl` through the World API and compares the
moment each event produced and the state after it; every `set_deadline` is checked against
`requestDeadline` (so `shortestPath` too) and every request's start cell against
`farCandidates`.

Results on the fixture of 2026-09-15 (16,000 granules, 80 active, 165 reading neurons in 4
pathways, 11 fields, 656,000 records; 124 stream moments, 124 world events; about 3 s):

| tables | decisions | of which | prediction | records mean, pathway norms | records tables | writes, imagined | stores, recall | efficacy, bias, world velocity | critic w / b | context trace | counters, plans, rng, ring, ledger | world |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sidecar (float64) | 113/113 | 17 searches (14 fallbacks), 6 plan reuses, 2 reward-greedy cue decisions, 88 exploration | 1.1e-15 | 0, 0 | 3.3e-16 (largest: next_dx and next_passage_north; reward 2.2e-16) | 4763 / 211, exact on every line | 0 | 0 | 2.2e-18 / 0 | 0 | exact | 124/124 moments and states identical |
| snapshot (float32) | 113/113 | the same | 7.6e-9 | 0, 0 | 2.6e-8 | exact | 0 | 0 | 2.2e-18 / 0 | 0 | exact | 124/124 |

The float64 run reports `PARITY OK`; the float32 run fails the records tolerance (1e-10)
alone, as on S01: the page's tables start rounded at 1e-8 and every write moves them from
there, while every decision, plan, store and count still agrees. The efficacy digest agrees
byte for byte on every line: with the records head the world repair moves no synapse and the
search's decisions carry no motor eligibility, so the efficacy stays at the snapshot's values.
The stores agree exactly: on one-hot keys every store operation is elementwise in a fixed
order. The stream now holds plan reuse (6 of 113 decisions): with the witnessed map supplying
the passages of known cells, the search's imagined path is followed by the world more often.

On `runs/world/parity_gains` (the same stream through the field gains, the action gain, the
21 field pathways and fan-in 3; 125 moments, 125 world events) the check reports `PARITY OK`
as well: 114/114 decisions, prediction 8.9e-16, records tables 2.2e-16, stores and recall 0,
the planner counters, the generator states, the write counts and the ledger phases exact,
125/125 moments and states identical.

## What the page shows

The layout is the arm page's: one stage measure (`--stage` in `style.css`) for both views, the
world square on the left and the whole brain beside it, both inside a 1280 by 800 screen, and
one column with the world first below 1120 px.

**The world.** Six by six cells: walls (grey), doors (orange shut, green dashed open), the
four objects in the life's colours (letters A to D), the agent (white disc) with what it
carries, the neighbourhood the last inspect revealed (green cells), the cue task's rooms and
junction, the cell that was asked for (pink ring, privileged for the viewer), the cell the
search aimed at (teal dashed ring: the remembered place or the exploration target), the
imagined path of the last search (teal, orange for a reused plan), the executed action, and
the remembered place of every object as a dashed diamond from the place store. Over the canvas
the readout names the task and its number: the places the agent remembers while it wanders,
the decisions left of the deadline during a request, the wander step of the cue task, the
teaching step of a word; under it the last seven tasks with their outcome.

**What the visitor does.** *Ask for A, B, C or D* starts a new episode of the same life from a
cell at least four steps away, with the goal naming the object and a deadline of twice the
shortest path plus two: the agent answers from its place store, or fails when the record is
missing or stale. *Dragging an object* to another cell (pointer down on it, up on the cell, or
a tap to pick up and a tap to place) changes the world while the record stays, which is the
correction the stage tests: the agent keeps the old place until it stands there and sees the
cell empty, which erases the record, or sees the object elsewhere, which overwrites it.
*Wander* is free exploration at rate one, where the sightings come from; the objects stay
where they are. *Teach a word* shows a word while its object is in view for 24 steps, then
asks by that word alone. *Cue task* shows one of the two cue words, wanders the chosen delay
with it hidden, and takes the two-choice decision at the junction; it clears the world of
objects the way the stage does and puts them back in their cells afterwards. *Lock doors* makes interact
stop opening doors, mid-life, and unlocks them again. The `run.py` selector runs one of the
stage's own phases instead, in the order run.py runs them, with page-sized budgets
(`PAGE_BUDGET` in `page.js`): the whole curriculum, explore, remember at delays 8, 16 and 32,
the corrections with the move seen and unseen, cue at delays 8 and 32, the locked doors, and
the words. A task starts at the next event; a wander closes its episode first
(`close_episode`, as a time limit would).

**The stores**, under the world: the place record of each object (the cell it remembers, a
tick when it matches the world, a mark when it does not), the cells of the witnessed map, the
words that name an object, the cue held and the word shown now. The recall port beside the
ledger shows what those stores drive into the recall neurons (the place read, the cue, the
word read).

**The brain.** The scan through `BrainScan` as on the arm page: every neuron coloured by its
region (sensory, goal, efference, workspace, context, recall, dynamics, prediction, motor) and
every synapse, animated per settling step of every phase the settled regions run (free, the
free phase after learning, the bootstrap value at a time limit), with the synapses flashing
when an update moves them. Beside it the records cortex, its own labelled region: the raster
of all 16,000 cells with the 80 active cells of the executed reading's plain code lit in blue
(the consequence records read it) and the 80 cells of its valued code in pink (each pathway
divided by its running norm; the reward record reads it), the search's imagined reads as grey
flickers, and the writes as flashes on those cells when the outcome arrives (orange at the
consequence rate 0.3, pink for the reward record at rate 1.0); under it the habituated reading
(the 165 reading neurons minus each unit's running mean: sensors, missing flags, goal, action,
recall, with the running norm of the four pathways) and the per-field record reads (the
predicted displacement, passages, cell contents, carried object and outcome as class bars with
the largest class and its probability, the reward on its range, with the largest error of the
last write). The montage, the dopamine bar, the ledger and the world's own counters follow.
`Speed` sets settling steps per animation frame, `Imagine` how many imagined settles are
drawn, `Exploration ε` the rate of a random action during a request (a wander always explores
at one).

**The brain selector** appears when a `checkpoints.json` manifest sits beside the page. It
loads another point of the same life, rebuilds the agent, the scan, the records view and the
stores in place, and keeps the world as the visitor left it. A checkpoint carries no stores,
so the brain that arrives remembers nothing until it wanders.

`window.__page` exposes `{scan, agent, brain, world, cur, records, queue, stats, task, step(),
run(n), runTask(n), ask(k), move(k, cell), play(), pause(), selectCheckpoint(id), toClient(cell),
cellAt(x, y), ready}`; `run(n)` computes and shows n events, `runTask(n)` runs until the
current task records its outcome, which is how `check_page.py` drives the page.

The check of 2026-09-15 on the page built from `runs/world/page` (seed 0 of
`S02-development-20260915T113712Z`, 38,191 decisions lived) passed with zero page and console
errors: both canvases inside the 1280 by 800 viewport and both panels above the fold, 40
wandering events until the agent had seen two objects, the request for C answered in 4
decisions from the remembered cell (2,4), the drag to (0,0) leaving the record at (2,4) and
the next request failing at its deadline, a word taught and answered in 5 decisions, the cue
task at delay 16 answered, the doors locked and unlocked, run.py's remember-8 phase entered,
the brain selector loading the 1,000-step brain and returning, 16 ms per event on average
(26 ms in the slowest batch, before the browser has warmed up; the bound is 100 ms), and the
world above the brain at 390 px. It wrote `page_start.png`, `page_wander.png`, `page_ask.png`,
`page_after.png`, `page_full.png`, `world_after.png`, `brain_after.png`, `records_after.png`
and `page_phone.png` next to the page.

## Notes

- All modules use one-line `import` statements and `export const/function/class`
  declarations only; `build_page.py` strips those, checks that no two modules declare the
  same top-level name, and concatenates them into one module script.
- `parity_s02.mjs` imports the records helpers from `../../web/parity.mjs`, which runs its
  own `main` only when it is the entry point.
- A snapshot without a records head takes the settled path as before (the predict settle, the
  world repair with the replay ring, imagined settles); the records panel stays hidden.
- `../../web/engine.js` is shared with the arm page, so a change there is checked on three
  fixtures: `runs/arm/parity`, `runs/world/parity` and `runs/world/parity_gains`. It carries
  `GraphSpec.field_gains` and `GraphSpec.action_gain` (both no-ops by default),
  `RecordsConfig.pathways` (`"ports"` or `"fields"`, which decides how `Agent.__init__` builds
  the input pathways when the snapshot lists none) and the `fan_in` mask of
  `cadence/src/cadence/records.py`, drawn cell by cell with the same Mulberry32 stream as
  Python, after the offsets and the task groups.
- `World.continueFrom({event, episode})` carries the world's counters forward, never back: a
  brain loaded from the selector has seen thousands of events, and the next moment of the
  running world has to come after them.
- `checkpoints.json` (`cadence-world-checkpoints/1`) lists the inlined brain and every
  checkpoint with its label, file, size, decisions lived and source. A page opened from
  `file://` cannot fetch its neighbours, so the selector stays hidden there and the page runs
  on the one brain it carries.
