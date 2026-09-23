# Patch World

Soft bodies live on a torus that conserves its mass under one moving sun. A body is a graph of
point masses and springs. Some springs are muscles. Every node keeps more of a step along its
head-ward axis than against it or across it, and that grip is all that makes a shape crawl.
Each being inherits its shape, its grips, a clock, its drives and the slow weights of a small
Cadence brain. It is born with three nodes, grows the rest one unit of mass at a time, learns
during its life, splits when it has stored enough, and dies when it runs out. Every birth
mutates the genome: nodes move, limbs are added or mirrored across the body, muscles come and go,
and the brain gains, loses, resizes or splits a cortex. Energy and death select. The world has no
fitness function.

`web/rules.html` states every rule with its numbers. The page draws the world in 3D and opens any being you click: its body with the muscles at
work, its reading, both halves of its brain. `Inspect` puts a chase camera on the body beside a
live picture of the whole brain. Every neuron is a sphere sized by its activity, every weight a
line that shines with the signal it carries this tick, the record cells a grid with the
reading's code lit, and the muscles a column of motor neurons between the two halves. Every
group, cortex and port carries its name.

[![A world in its first day: soft bodies on the torus, the world's counters and lineages beside it](screenshot.png)](https://floatingpragma.io/cadence-examples/patchworld/)

Live page: [floatingpragma.io/cadence-examples/patchworld](https://floatingpragma.io/cadence-examples/patchworld/).
The same page runs from this directory; see [Run it locally](#run-it-locally).

## Card

- **Name:** Patch World
- **Author:** Bernhard Mueller, Pragma Research
- **Description:** Soft bodies of point masses, springs and muscles live on a torus that conserves its mass under one moving sun. Each being carries two record patches, a policy and a model, built from inherited cortices; it is born with three nodes, grows, learns during its life, splits and dies. Every birth mutates body and brain. Energy and death select, and the world has no fitness function.
- **Cadence version:** 0.12.0. `ref/make_fixture.py` produces the parity fixture from the library at that release (first checked at commit `02fec624`).
- **Hardware for initial training:** CPU only, one node process on an Apple M4 laptop. The founders come from a gait search outside the world (`sim/gaitlab.js`) and their model is taught by observation on their own stream. A world of 20,000 ticks with up to 500 beings takes 13 minutes, 20 with the night and the sleep gene on.
- **Cadence features showcased:** `cadence.RecordPatchNet` twice per being, a policy and a model; cortices as masked blocks of context channels with their own slowest timescale, which is what evolves; adjoint steps over windows and record writes of each moment's residual during a life; planning by gradient repair of the motor under the model with the record read held fixed; `RecordPatchNet.sleep` as a gene; every read, channel, cell and write priced in mass; a JavaScript twin held to the library at 1e-15 on observation, windows, backtracking and planning.
- **Problems encountered during development:**
  - All-zero slow weights are a saddle from which only the bias learns, and a model that predicts the next reading learns nothing beyond persistence. The model predicts the change of the reading and starts from taught weights.
  - `observe` cannot learn a window an actor has advanced through, so the twin carries a window interface (`step`, `teach`, `flush`) for an actor that acts every tick.
  - A planner over a record-augmented model exploits the records off the data; on the lab course the score fell from 89 to between 24 and 34. Replays hold the record read fixed.
  - A policy that learns every repair one-shot drifts off its gait. The policy learns a repair only when what followed beat the model's prediction for the unrepaired motor, and records stay out of the policy.
  - In the first sweeps carriers of records died 5 to 30 percent younger in every world, and planners died younger too: the per-tick gradient plan could not cash in the pressure to learn. Cortices, body mutation and priced structure came out of that.
  - A body bitten to zero energy paid its price in the same tick and left one unit of mass at its death. Conservation checks over short runs miss rare leaks; the leaking seed was bisected tick by tick, and every receipt made before the fix was rerun.
  - The library changed the scaling of the record reading while the twin was being built; the committed fixture kept passing and a fresh one failed 24 of 86 cases. The fixture is regenerated before anything is published.
  - At the population cap the oldest beings split first, a selection bias. Candidates are shuffled, and food limits the population instead of the cap.
  - The sleep gene loses in every run: the awake learner has the world at every tick, and the sleeper trades its actions for second-hand presentations of what it holds.
  - Under the prices the measured worlds keep small brains; a second cortex that pays for itself is the open question.
  - Headless Chromium renders the page slowly, so every number comes from the node probe and the browser gives the screenshot.
- **Hosted at:** https://floatingpragma.io/cadence-examples/patchworld/
- **Receipts and checks:** `receipts/*.jsonl` (worlds of 20,000 ticks, base and sleep), `sim/founder.json`, `sim/parity.js` against `ref/fixture.json`, `tests/physics.test.js`, `web/build.py --check`. `node patchworld/sim/parity.js patchworld/ref/fixture.json && node patchworld/tests/physics.test.js` from the repository root.
- **Data and rights:** No external data. Every world is generated from its seed.
- **Work in progress:** a world in which a second cortex pays for itself; planning over whole-gait alternatives with the policy in the loop; per-patch record-cell genes.

## What this example shows

- **Evolution of bodies and wiring.** Shape, muscles and each patch's list of cortices with their reading masks mutate at every birth. Energy and death select; the world has no fitness function.
- **Learning in one life.** Each being takes adjoint steps and writes records as it lives. Records are never inherited.
- **Planning through its own model.** A being with a planning horizon repairs its motor on a private path under the model it learned, toward its drives, and executes the first repaired motor.
- **Direct motor control.** The policy patch drives eight muscle ports and the mouth of a soft body that crawls by grip alone.
- **Drives.** Energy and pain are among the senses, and the inherited drives say what the planner steers toward.
- **Computation has a price.** Every unit read, channel, record cell, write and planning replay costs mass, so a brain has to pay for itself.
- **Sleep is a closed-corpus mechanism.** A gene lets a being keep the day's windows and dream them
  through its records in the dark, moving its slow weights on the dreams (the library's `sleep`).
  Under selection the gene loses in every run: the awake learner has the world at every tick, the
  sleeper trades its actions for second-hand presentations of what it holds.

## The brain is two record patches

Both are `cadence.RecordPatchNet`: a gated linear context, a slow readout, and a store of
records inside the patch that holds what the slow readout got wrong at each reading.

- **Reading**: twelve senses (the scent of food to the left against the right, ahead and under
  the head; rock and another body ahead; energy, pain, light, grip; speed, turn, size), one
  strain per muscle port, the clock's sine and cosine, and the previous motor.
- **Policy**: reads the reading and proposes the motor: eight muscle ports and the mouth.
- **Model**: reads the reading and this tick's motor and predicts how every unit of the reading
  and the energy will change. It carries the records.
- **Cortices**: each patch is a list of cortices. A cortex is a block of context channels with a
  mask over the groups of the reading it may read and its own slowest timescale. The mask zeroes
  rows of the input weights and survives learning, so the whole brain stays one patch with one
  energy, and who reads what is what evolves.
- **Learning in a life**: every eight ticks each patch takes one adjoint step over the window at
  its inherited rate and writes each moment's residual into the record cells that moment read.
- **Planning**: a being whose horizon gene is above zero repairs the motor ports of a private
  path under its model, toward its drives, and executes the first repaired motor. The policy
  learns a repair one tick later, and only if what actually followed beat what the model had
  predicted for the unrepaired proposal.

`sim/patch.js` is the JavaScript twin of the library class, with a window interface for an actor
that acts every tick (`step`, `teach`, `flush`) and the planner. `ref/record_plan.py` is the same
planner on the library class.

## Supplied, inherited, learned

- **Supplied**: the physics, the senses, the prices, and the founders. The founders come from
  `sim/gaitlab.js`, a search outside the world over body graphs and the policy's motor rows for a
  body that crawls and turns toward food, as an adult and as a newborn; their model was then
  taught by observation on their own stream. The world never sees that fitness again.
- **Inherited**: the body graph and grips, the clock, the split threshold, the drives, the motor
  noise, the planning horizon and radius, the size of the records, the learning rates, the list
  of cortices of each patch, and every slow weight.
- **Learned in one life**: the records, and the slow weights at the inherited rate. Records are
  never inherited.
- **Priced**, each as the chance of losing one unit of mass per tick: a base rate, every node,
  muscle work, every unit read, every channel, every record cell, every write, every planning
  replay, every bite.

## Evidence

- `node sim/parity.js`: the twin against `ref/fixture.json`, which `ref/make_fixture.py` produces
  from the library at the release the examples pin, 0.12.0 (first checked at commit
  `02fec624648d421e02ecb00f52f3d3072e9fe9ae`). Observation, the window
  interface, backtracking admission, learning without writing, and planning with feedback, with
  additive feedback, with bounds per port and with the record read held fixed agree to a relative
  difference of 1e-15.
- `node tests/physics.test.js`: mass is conserved at every tick through eating, growth, bites,
  deaths and births; a two-node body with one muscle and a symmetric grip stays where it is and
  crawls with a ratchet grip (the scallop theorem); the same seed replays the same world.
- `sim/founder.json`: the founders' model predicts the next reading on a held-out stream with a
  mean squared error of 0.0099, against 0.0179 for predicting no change and 0.151 for the
  running mean.
- `receipts/sleep_*.jsonl`: the night and the sleep gene over 20,000 ticks, two seeds, three worlds
  (`sim/summarize_sleep.py receipts`): founders that all sleep are down to 6 and 1 percent of the
  living by tick 5,000 and 3 and 1 percent at the end; a gene that mutates on in an awake world
  stays between 2 and 6 percent; sleepers' late-life model error is 0.056 to 0.058 against 0.035
  to 0.049 for beings that learn by day, and they live no longer. The page runs with the night
  and the gene on, half the founders sleeping, so the loss can be watched.
- `receipts/`: two base worlds of 20,000 ticks on this code, before the night. The population's mean speed rises
  from 0.040 to 0.071 and from 0.042 to 0.063 cells per tick, with mass drift zero in both.

## Run it locally

From the root of this repository, with Python 3 and node installed. The page is one static
file, `web/index.html`, built from `web/page.html` with the sources inlined:

```bash
python -m http.server -d patchworld/web 8804     # then open http://127.0.0.1:8804 (the rules: /rules.html)
```

To check the browser brain against the library, the physics, and the build:

```bash
python -m pip install "cadence-net==0.12.0"
python patchworld/ref/make_fixture.py --out /tmp/fixture.json && node patchworld/sim/parity.js /tmp/fixture.json
node patchworld/tests/physics.test.js
python patchworld/web/build.py --check           # web/index.html is web/page.html with the sources inlined
```

A world without a browser, and why its winner won:

```bash
node patchworld/sim/probe.js '{}' 20000 3 '{"chronicle":"run.chronicle.json"}' > run.jsonl
python patchworld/sim/why.py run.chronicle.json
```

The page's `Export chronicle` button writes the same file: every lineage every hundred ticks
(births, deaths by cause, kills, meals, speed, body and brain), the events, genome samples, and
the full genomes of the largest lineages. `sim/why.py` names the winner, when it rose, what it
did differently per being over its rise, how its genome differs from the founder's and its
rivals', and says so when a sweep looks like drift.

`?lite` drops shadows and the population, `?pop=N` sets the cap, `?rate=max` runs the simulation
as fast as the page allows, `?seed=N` fixes the world.

## Limits

The founders are found by a search outside the world. The planner is a gradient repair of one to
four ticks of motor, which a periodic gait gives little to work with. A channel, a read and a
record cell are priced, and the founders' reflex already feeds a body, so the worlds measured so
far keep small brains. The body is a planar mass-spring model with friction and no inertia.

## Build on it

Fork it and build a world of your own: change a price, add a rule, find the world in which a second cortex pays for itself.
