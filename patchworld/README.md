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

## What this example shows

- **Evolution of bodies and wiring.** Shape, muscles and each patch's list of cortices with their reading masks mutate at every birth. Energy and death select; the world has no fitness function.
- **Learning in one life.** Each being takes adjoint steps and writes records as it lives. Records are never inherited.
- **Planning through its own model.** A being with a planning horizon repairs its motor on a private path under the model it learned, toward its drives, and executes the first repaired motor.
- **Direct motor control.** The policy patch drives eight muscle ports and the mouth of a soft body that crawls by grip alone.
- **Drives.** Energy and pain are among the senses, and the inherited drives say what the planner steers toward.
- **Computation has a price.** Every unit read, channel, record cell, write and planning replay costs mass, so a brain has to pay for itself.

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
  from the library at commit `02fec624648d421e02ecb00f52f3d3072e9fe9ae`. Observation, the window
  interface, backtracking admission, learning without writing, and planning with feedback, with
  additive feedback, with bounds per port and with the record read held fixed agree to a relative
  difference of 1e-15.
- `node tests/physics.test.js`: mass is conserved at every tick through eating, growth, bites,
  deaths and births; a two-node body with one muscle and a symmetric grip stays where it is and
  crawls with a ratchet grip (the scallop theorem); the same seed replays the same world.
- `sim/founder.json`: the founders' model predicts the next reading on a held-out stream with a
  mean squared error of 0.0099, against 0.0179 for predicting no change and 0.151 for the
  running mean.
- `receipts/`: two base worlds of 20,000 ticks on this code. The population's mean speed rises
  from 0.040 to 0.071 and from 0.042 to 0.063 cells per tick, with mass drift zero in both.

## Run it

```bash
python -m pip install "cadence-net @ git+https://github.com/muellerberndt/cadence.git@02fec624648d421e02ecb00f52f3d3072e9fe9ae"
python ref/make_fixture.py --out /tmp/fixture.json && node sim/parity.js /tmp/fixture.json
node tests/physics.test.js
python web/build.py --check                 # web/index.html is web/page.html with the sources inlined
python -m http.server -d web 8804           # then open http://127.0.0.1:8804 (the rules: /rules.html)

node sim/probe.js '{}' 20000 3 '{"chronicle":"run.chronicle.json"}' > run.jsonl   # a world, headless
python sim/why.py run.chronicle.json        # why one lineage took the world
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
