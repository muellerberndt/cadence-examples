# The browser engine

A settling brain of the library, in a page, with the library's arithmetic: `brain.js` settles a
sparse connectome (the rate model, the free phase, the nudged phase), `learner.js` runs the
one-stream actor-critic on it, `export.py` writes what the engine reads from a `cadence.Brain`,
and `parity.mjs` holds the two to each other. The fruit fly runs 60,000 neurons and 1.2 million
synapse classes on it in a worker at about 6 ms per step; the worm's temporal patch is a
different model and carries its own engine.

This page is the guide for building an example of your own: the payload, the settle loop, the
viewer, the lessons, the parity test, and the conventions every example page follows.

## The payload

`export.py` turns a `cadence.Brain` into one JSON file:

```python
from engine.export import write_payload, settle_cases, record_lessons

write_payload(brain, "web/data/brain.json")     # or members=sub.members for a sub-net of a larger brain
```

| field | what it holds |
| --- | --- |
| `n`, `edges`, `synapses` | neurons, synapse classes, synapses |
| `model` | `dt`, `slope`, `threshold`, `gain`, `stimulus_amplitude`, `leak` (adaptation is not settled in the browser) |
| `populations` | every named set of neurons, by name |
| `arrays.row_ptr`, `arrays.pre` | the synapses by receiving neuron (CSR), senders in the library's order |
| `arrays.weight` | the weight the library settles with: gain, count, sign and any per-neuron gain folded in |
| `arrays.count`, `arrays.sign` | per synapse class (the sign as int8 when integral, `sign_dtype` says) |
| `arrays.log_gain` | one number per neuron, a gain per cell class the connectome's dictionary declares (`Brain(log_gain=...)`); left out when every neuron's is zero. The engine composes each synapse's factor `gain * count * exp(log_gain[pre])` from it |
| `arrays.gain_pre` | that factor per synapse, written only when a count exceeds `uint16` |
| `arrays.efficacy_index`, `arrays.efficacy_value` | the synapses whose efficacy is off the connectome's sign, as index and value (a seam started naive, a checkpoint); `arrays.efficacy` holds the whole array instead when more than a third differ; both left out while every efficacy is its sign |
| `arrays.bias` | the constant input per neuron |
| `arrays.members` | this brain's indices in a larger one, when it is a sub-net |

Arrays are base64 of the raw bytes (`Int32`, `Float64`, `Uint16`, `Int8`); `decodeArray` reads
them. Keep the synapse order the library uses (by receiving neuron, then sender; `Connectome`
sorts them), because the learner's plastic set, a checkpoint's efficacies and the parity
recording all name synapses by that index.

## Settling in a page

```js
import { SettlingBrain } from "./brain.js";
const brain = new SettlingBrain(await (await fetch("./data/brain.json")).json());
brain.clearStimuli();
brain.stimulate("orn:decaying_fruit:left", 0.8);   // a level in [0, 1] on a named population, times the amplitude
for (let k = 0; k < 12; k++) brain.step();         // one step per declared slice of simulated time
brain.mean("mbon:MBON11:right"); brain.activeCount(0.5); brain.s;   // readouts, the active count, every activation
brain.reset();                                     // the state back to rest; the stimuli stay until clearStimuli()
```

Run it in a worker and send a batch of steps per message, with a rule that a slow worker never
builds a backlog (drop the simulated time you cannot settle; the fly caps the backlog at ten
slices). The activation vector `brain.s` (copied to a `Float32Array` and transferred) feeds the
viewer; `brain.members` maps it onto the atlas of the whole brain when the page settles a
sub-net. Cost: one step is one pass over the synapses; 1.2 million classes take about 6 ms in
Chrome on a laptop.

## The viewer

`brain_scan.js` from the library draws every neuron where it sits (the anatomical mode on an
atlas with `positions3`) or wrapped into a stylised brain, with the activity live and the recent
change glowing; [pages.md](https://github.com/muellerberndt/cadence/blob/main/docs/pages.md)
documents the atlas export and every option. Copy the file into `web/`. Budget it first: it is
the frame cost of a page. On a 1440 by 900 window at pixel ratio 2, 200,000 ribbons with 150,000
particles gave 30 fps and 60,000 with 40,000 gave 60; the bloom pass and two extra scene views
cost nothing measurable. Options `hot`, `cool`, `labelCount` and `dpr` set the palette, the
labels of a small card and the pixel ratio cap (v4.2).

## Learning in a page

```js
import { ActorCriticLearner } from "./learner.js";
const learner = new ActorCriticLearner(brain, {
  outputs: ["mbon:MBON11:right", "mbon:MBON05:left"],   // one neuron each, in action order
  actions: [0, 1],                                       // what each output stands for
  plastic: { pre: ["kc"], post: ["mbon"] },              // or "all", or an array of synapse indices
  critic: "kc",
  beta: 0.1, temperature: 0.3, nudgedSteps: 10, tolerance: 1e-3,
  gamma: 0.95, lam: 0.9, eta: 10, etaCritic: 0.5, cap: 3, dopamineCap: 1,
});
// at a decision: the live state is the free phase
const d = learner.act();                 // { choice, action, p, value }; two nudged phases keep the eligibility
// ... the world moves; the page keeps stepping the brain under the new senses ...
const lesson = learner.learn(reward, done);   // dopamine from the reward and the live state's value; the plastic synapses move
lesson.delta; lesson.saturation; lesson.trace; lesson.changed;
```

The order the library keeps, and the engine with it: `act` reads the live state as the free
phase and runs the two nudged phases on copies; the page then puts the next observation on
the brain and keeps stepping (that state is the bootstrap value); `learn` moves the synapses;
the brain keeps stepping under the new weights before the next `act` (the library's `act`
refreshes its cached state under the updated weights the same way). A finished stream resets
the brain first.

The constants are the library's (`LearnerConfig` and `ActorCriticConfig`); take them from the
receipted experiment, not from another example. Two readings say whether a lesson can still
move anything: `saturation` (the fraction of outputs within 0.02 of 0 or 1, where a nudge has
no slope) and `trace` (the mean absolute eligibility of the plastic synapses); saturation near 1
with the trace near 0 is the latch described in the library's
[learning from reward](https://github.com/muellerberndt/cadence/blob/main/docs/reward.md). The
temperature is relative to the outputs' activation range: 0.05 for outputs near rest, 0.3 for
outputs that sit at 0.7 to 1.0 under their drive. `load(edges, efficacies)` puts a receipted
checkpoint's efficacies on the brain (`tools/export_learned.py` in the fly maps a checkpoint
onto the page's sub-net by whole-brain synapse), `reset()` returns to the measured signs.

An outcome handed to `learn` with no decision pending teaches nothing; the learner counts it in
`dropped` (in `stats()`), and a page shows that count, because the fly's blows were lost exactly
this way for a day: the search had been closed at the landing, the blow found nothing to credit,
and the screen still counted the blow.

## Parity

Every example keeps two recordings of the library beside its payload and replays them through
the engine in CI:

```python
settle_cases(brain, {"loom": {"vis:LC4": 1.0}}, ["gf", "dn:landing"], steps=40, out="tests/parity_cases.json")
record_lessons(learner, critic, drives, rewards, dones, config=ActorCriticConfig(...), out="tests/lessons.json")
```

```
node engine/parity.mjs web/data/brain.json tests/parity_cases.json tests/lessons.json
```

The settle cases are per-step population means under named stimuli; the lessons are one
stream's recorded uniform draws, actions, dopamine, and the plastic efficacies and critic at
the end. Both must agree below 1e-9 (the fly's settle agrees to 4e-16). `make_fixture.py`
builds a 30-neuron brain, its cases and a 24-decision recording and is what CI runs; read it as
the worked example of `export.py`. An example that copies `brain.js` and `learner.js` into its
`web/` folder registers the copies in `check_copies.mjs`, so a fix to the engine reaches every
page.

## The page

- Three views and nothing else: the brain viewer, the body in close-up at all times, and what
  the animal senses (the fly's compound-eye view). Everything else is one status line and one
  toolbar; the instruments panel is hidden until asked for.
- One sentence for the visitor, permanent, saying what to do and what the brain learns from it.
- Test at device pixel ratio 2. Three.js's `setViewport` and `setScissor` take CSS pixels and
  apply the ratio themselves; passing device pixels scales every view by two on a Retina screen.
- Cap the page's pixel ratio (the fly uses 1.25 and `?dpr=2`); the viewer takes its own `dpr`.
- The README carries the card ([Contributing](../README.md#contributing)), the live link, a
  screenshot and the run-it-locally recipe; the page's credit line links the source folder.
- The page deploys as a copy of `web/` (minus dev pages) into the site's
  `cadence-examples/<name>/` folder, with a 1200 by 630 social card under 200 kB.

The page contract, from the fly's three visitor reports of one day:

- One owner per fact. What the visitor can change (a fruit's place, the sugar) lives in one
  object and every other layer reads it there; a copy taken at construction is the fly landing
  on a fruit that was moved an hour ago.
- Every wait is bounded and every modal state is visible. A brain reply that never comes gets a
  watchdog; a hover for a decision lands after a few seconds anyway; a sit has a longest; a
  paused simulation says so on screen (the space key froze the fly in silence).
- Every outcome finds its decision. A search stays open until its outcome; an outcome with no
  decision pending is counted (`dropped`) and shown, not swallowed.
- One decision per episode, credited to that decision; a decision every few hundred
  milliseconds along an approach mixes the eligibility of both actions.
- A scenario test of the page's promise, scripted against the real page in a headless browser
  (the fly's `tools/reversal_scenario.py`: sugar on one fruit, the sugar moved, blows at the old
  one, the other found, in both directions), runs before every deploy. A page ships what its
  sentence promises, and the test is the receipt.

## A checklist for a new example

1. The brain: a `cadence.Brain`, its populations named, the gain selected by a protocol with
   shuffled controls, the receipt written before the page.
2. `write_payload`, `settle_cases`; the page settles and its readouts match the library.
3. The body and the senses as declared dictionaries; what the brain cannot carry, supplied and
   said so.
4. The lessons: `preflight` under the situations the page decides in, with no warnings, before
   the readout, the plastic set and the constants are fixed from the experiment;
   `record_lessons`; the parity replay.
5. The three views, the sentence, the card, the scenario test of the promise, the deploy, the
   video tour.
