# The S02 world in the browser

The remembered world of `experiments/experience/s02_world` ported to JavaScript: the world and
its curriculum (`env.py`, `tasks.py`), the four stores and the best-first search with the
`Brain` wrapper (`brain.py`), on the experience agent of `../../web/engine.js` with its
records world head, and one page that shows the world, the stores and the whole brain
(the settled regions in the scan, the records cortex beside it) while the curriculum runs.
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
| `build_page.py` | Inlines the snapshot, the renderer, the engine, the records view, the world, the stores, the planner and the page into one `index.html`. |
| `check_page.py` | Headless Chromium (SwiftShader): steps events through `window.__page.step()`, saves screenshots, fails on any page or console error. |

## Build and run

```sh
cd /Users/muellerberndt/Projects/oph-meta/cadence-paper
PY=/Users/muellerberndt/Projects/oph-meta/cadence/.venv/bin/python

# the fixture (about 5 s): 320 explore steps at rate 1, the snapshot at the episode boundary,
# then one explore episode, two remembered requests at delay 8, two cue episodes at delay 8
# and one word episode through run.py's own run_explore, run_remember, run_cue and run_words;
# writes the float64 records sidecar next to the snapshot
$PY world/tools/parity_fixture.py --out runs/world/parity

# the two parity checks (about 2 s each)
/opt/homebrew/bin/node web/parity.mjs runs/arm/parity
/opt/homebrew/bin/node world/web/parity_s02.mjs runs/world/parity

# the self-contained page (10.9 MB, the snapshot inlined with float32 records tables)
$PY world/web/build_page.py \
    --snapshot runs/world/parity/snapshot.json --out runs/world/web/index.html

# the headless check: zero console errors, 40 events in the remember-8 phase, then the curriculum
/Users/muellerberndt/Projects/oph-meta/cadence-artist/.venv/bin/python \
    world/web/check_page.py runs/world/web/index.html --events 40
```

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

## What the page shows

Left, the world at 6 by 6: cells, walls (grey), doors (orange closed, green dashed open), the
four objects with the life's colours (letters A to D), the agent (white disc) with what it
carries, the neighbourhood the last inspect revealed (green cells), the cue task's rooms and
junction, the request's true target (pink ring, privileged for the viewer), the target the
search aimed at (teal dashed ring: the remembered place or the exploration target), the
imagined path of the last search (teal; orange for a reused plan), the executed action, the
remembered place of every object (dashed diamonds from the PlaceStore) and, in the caption,
the tick, the deadline left, the word shown, the cue held and the door rule. Above it the
phase label (phase, episode, wander or teach step, close, request with its target and the
decisions left, the two-choice decision with the paid room) and the running success per
phase; below it the counters (episode, tick, events, decisions, episodes ended, goal, action,
controller, expansions of the search, plan, last reward, consequences predicted correctly,
deadline, word shown, doors, exploration rate, compute time) and the phase selector.

The stores: the place records (remembered cell, known flag, the true cell with a mark, with
the map store's known cells and writes), the cue held, the word table (word to object
strengths with the referent the store names) and the recall port's drive (place read, cue,
word read: what enters the recall neurons).

Right, the whole brain. The scan through `BrainScan` as on the S01 page: every neuron
coloured by its region (sensory, goal, efference, workspace, context, recall, dynamics,
prediction, motor) and every synapse, animated per settling step of every phase the settled
regions run (free, the free phase after learning, the bootstrap value at a time limit).
Beside it the records cortex (cerebellum-like / dentate-like), its own labelled and coloured
region: the granule raster of all 16,000 cells with the 80 active cells of the executed
reading's plain code lit in blue (the consequence records read it) and the 80 cells of its
valued code in pink (each pathway divided by its running norm; the reward record reads it),
the search's imagined reads as grey flickers, the writes as flashes on the active cells when
the outcome arrives (orange on the plain code's cells at the consequence rate 0.3, pink on
the valued code's cells for the reward record at rate 1.0); the habituated reading under it
(the 165 reading neurons minus each unit's running mean: sensors, missing flags, goal,
action, recall, separated by ticks and named under the strip, with the running norm of the
four pathways observation, goal, action and recall); and the per-field record reads: the
predicted next displacement, passages, cell contents, carried object and outcome as class
bars with the largest class and its probability, the reward on its range, with the largest
error of the last write per field.
The EEG montage, the dopamine bar and the ledger (record writes, imagined reads, the searches,
reuses and fallbacks) follow. The speed control sets settle steps per animation frame; the
imagination control applies to settled imagination only (a snapshot without a records head),
since imagined consequences are record reads here.

The curriculum runs as run.py runs it, with page-sized budgets (`PAGE_BUDGET` in `page.js`:
100 explore steps, 5 remembered requests per delay, 3 corrections per kind, 6 cue episodes
per delay, 4 door trials before and after locking, 4 word episodes with 24 teaching steps):
explore; remember at delays 8, 16, 32; visible and invisible moves; cue at delays 8 and 32
(the decision at the life's own exploration rate); remember at delay 8, the doors locked,
remember at delay 8; words. Every wander ends with a truncated closing moment
(`close_episode`) and every request, name request and cue decision is a new episode of the
same life. The sequence repeats; a new cycle sets the door rule back to toggle. The phase
selector runs one phase repeatedly instead; a change takes effect at the next episode
boundary. The page continues the fixture's map (`extra.world`), with the stage config's
horizon and its own random stream from there; the snapshot is taken at an episode boundary,
so the page starts with a fresh episode and no decision awaits its outcome.

`window.__page` exposes `{scan, agent, brain, world, cur, records, queue, stats, step(), ready}`;
`step()` computes one event and shows it at once, which is how `check_page.py` drives the
page. The check of 2026-09-15 (40 events in the remember-8 phase, then the controls and six
curriculum events) ran with zero page or console errors and wrote `page_start.png`,
`page_mid.png`, `page_after.png`, `brain_after.png`, `world_after.png`, `stores_after.png`,
`records_after.png` and `page_phase.png` next to the page; after 40 events the records
cortex had shown 36 codes, 236 imagined reads and 396 field writes over 80 active cells.

## Notes

- All modules use one-line `import` statements and `export const/function/class`
  declarations only; `build_page.py` strips those, checks that no two modules declare the
  same top-level name, and concatenates them into one module script.
- `parity_s02.mjs` imports the records helpers from `../../web/parity.mjs`, which runs its
  own `main` only when it is the entry point.
- A snapshot without a records head takes the settled path as before (the predict settle, the
  world repair with the replay ring, imagined settles); the records panel stays hidden.
