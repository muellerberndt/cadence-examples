# The worm

> Version-pinned historical example. For the public demo collection, see
> [Cadence demos](https://floatingpragma.io/demos/). This guide describes the
> model and evidence preserved in this directory.

The nervous system of the hermaphrodite *C. elegans*, 302 neurons wired as measured, as one
Cadence brain. It lives on a plate, smells, eats, gets hurt, and learns during its
life what the smells around it predict. The page draws the whole nervous system inside the
transparent body at every zoom, in two tones. Structure is cold: the body, the cords, the wiring
and every neuron at rest are pale blue on navy. Activity is hot: a firing neuron glows from deep
red through orange to white, each synapse releases pulses in its transmitter's colour, and the
nerve cords heat up where the cells along them fire. Around each cell lies a cloud of the
transmitter it is receiving most of, welling up as it is released and draining slowly, so food
floods the head with dopamine. A lesson is the one green: the connections
that changed, then a ring arriving at each cell hop by hop from the command neurons. A second
view shows the head at scale, straightened: the nerve ring, the ganglia around it, the
sensory and command neurons by name, and what each cell is by the tint and shape of its patch
(sensory, interneuron, command, motor, pharyngeal). Heat follows the logarithm of activity, because a signal
fades by orders of magnitude as it spreads.

[![The worm on its plate, the nervous system drawn inside the body, a lesson arriving from food](screenshot.png)](#run-it-locally)

Run the preserved page locally; see [Run it locally](#run-it-locally).

## Card

- **Name:** The worm
- **Author:** Bernhard Mueller
- **Description:** The 302 neurons of the hermaphrodite *C. elegans*, wired as measured, as one Cadence brain. The worm lives on a plate, smells, eats, gets hurt, and learns during its life what the smells around it predict. The page draws the whole nervous system inside the crawling body.
- **Cadence version:** 0.50.0. The newborn brain in `web/data/brain.json` is exported under that release, and CI rebuilds it under the same pin.
- **Hardware for initial training:** None. The twelve reflex lessons before birth run inside `worm/tools/export_web.py` in the library's pure-Python engine, 55 seconds of processor time on an Apple M4 laptop, CPU only. Everything else is learned in the browser while the worm lives.
- **Cadence features showcased:** `Cortex` and `Brain` used as the library documents them: one single-patch `column` per neuron, so the measured connectome is exactly the declared wiring and nothing else can grow; `step` retaining live activity tick by tick; lessons as `observe_batch(source="estimate")` with output targets, qualification and all-or-nothing admission; `settle` for pure probes; a browser port of the same reference repair, held to the library by a parity test.
- **Problems encountered during development:**
  - Cadence 0.50 has no way to install inherited synaptic strengths: a fresh brain's weights are the seed's, and a checkpoint with edited weights is rejected. The connectome therefore supplies which connections exist; every weight starts at the library's seeded value and is learned. Synapse counts and transmitter signs are drawn on the page but not imposed on the brain.
  - Wiring every synapse from the previous tick's states left lessons no path into the sensory chain: a batch row's context is fixed, and the specification is explicit that batch rows carry no delayed credit, so twelve reflex lessons taught the command drives' biases instead of the wiring. The neurons are now declared in order of synaptic distance from the senses; synapses running down that order are live state connections inside one joint settlement, synapses running back up it arrive from the previous tick, and a lesson's correction reaches back through the live chain.
  - Reflex lessons that only ever asked for reversing still taught "always reverse": the drives' biases absorbed the target. Each prenatal batch now also asks for rest on its quiet ticks, which a bias cannot satisfy at the same time, and lessons of the life restate the drives' own free predictions on the ticks they do not correct.
  - The first body rewound a trail of past head positions when reversing. Repeated reversals in an irritant field used the trail up and the worm shrank to a point, and omega turns and wall bounces turned the head in one step and folded the body. The body is a centreline of exactly one body length whose leading end bends at a bounded curvature, and a turn rotates the direction of travel directly.
  - The same seed leads a different life on another machine, because the last digit of a float changes a decision. The CI runner's life found a wall-clamp kink that never occurred locally. Coverage therefore rests on a brain-free fuzz of 300 bodies that is identical on every machine, and the wall rule turns an end inward before it reaches the edge.
  - Float exports differ between machines at the last digits through libm's tanh, so a byte-for-byte gate on `brain.json` failed. `check_exports.py` compares exactly where nothing is rounded and to 1e-6 elsewhere.
  - The network has no spontaneous activity. Away from every smell it goes cold, and the page shows that instead of adding noise.
  - A 1.7 MB social card made X fall back to a card without an image. The card is a 1200 by 630 JPEG of about 100 kB.
- **Hosted at:** Local reproduction; see [Run](#run-it-locally). The public demo collection is [Cadence demos](https://floatingpragma.io/demos/).
- **Receipts and checks:** `tests/parity.mjs` (the browser engine against the library along lived ticks and lessons, worst relative difference 4.8e-8), `tests/body.mjs` (the body's invariants under stress), `tests/test_learning_accounting.py` and `tests/learning-accounting.mjs` (lesson custody), `tools/check_exports.py` (the committed exports against a rebuild). `node worm/tests/parity.mjs && node worm/tests/body.mjs` from the repository root.
- **Data and rights:** The connectome and the cell classes come from openworm's c302 and ConnectomeToolbox (MIT); the soma positions from Kaiser and Hilgetag (2006). Sources, hashes and citations are in `data/SOURCES.md`.
- **Work in progress:** a conditioning receipt with paired, unpaired, frozen and lesioned controls across seeds; a body the motor neurons drive.

## Layout and version

The 0.50 worm uses **ordinary state-coupled settlement**: each neuron is a
single-patch column, forward connections read live patch states, and feedback
connections read states retained from the previous tick. The command drives
join that same solve. The layout declares no error-reading observers.

Flat input-only columns and recursive observers are also supported Cadence
design patterns. All three use the same patch rule, repair and qualification;
recursion adds state-and-exact-error connections. Choose connections that fit
the available signals and compare behavior and cost. See the
[layout guide](https://github.com/muellerberndt/cadence/blob/main/docs/VARIANTS.md)
and [performance guide](https://github.com/muellerberndt/cadence/blob/main/docs/PERFORMANCE.md).
Earlier worm revisions used `PartitionedTemporalPatchNet`; their receipts refer
to that versioned model.

## What this example shows

- **Learning from experience in one life.** Apart from twelve reflex lessons before birth, nothing is trained in advance. What a smell means is learned from the food or the pain it came before, while the worm lives on the plate.
- **Simple affect.** Food and pain are the only outcomes. They decide when a lesson happens and which command group it favours; bacteria reach the dopaminergic neurons and pain the ASH nociceptors.
- **Neural command readouts with supplied movement.** Forward/reverse readouts modulate an engineered crawling, reversal, steering and wall controller. The motor neurons do not drive muscles.
- **Structure as a constraint.** Every neuron is one declared patch and its declared inputs are exactly its synapses. Learning changes the weight of a connection that exists and cannot add one.
- **The library used as documented.** `step`, `observe_batch` and `settle` as the reference describes them: joint settlement, qualification against the configured tolerance, a parameter anchor on every lesson, and refusal that changes nothing.

## The brain is the connectome

- Neurons: the 302 of `data/c302_A_Full.net.nml`, placed at their measured soma positions
  (`data/SOURCES.md`). Each is one single-patch `column` of one `Cortex`.
- Connections: a neuron's declared inputs are exactly its presynaptic partners — every chemical
  synapse, every gap junction in both directions, and its own state a tick ago as a persistence
  term. Nothing else can grow. The neurons are declared in order of synaptic distance from the
  sensory cells: a synapse running down that order is a live state connection, read inside the
  tick's one joint settlement, and a synapse running back up it arrives through the previous
  tick's settled states. Signals flow from the senses toward the commands within a tick;
  feedback takes a tick.
- Weights: Cadence 0.50 installs no inherited parameters, so at birth every weight is the
  library's seeded value, scaled by the incoming connection count. The measured synapse counts
  and transmitter signs are drawn on the page but not imposed on the brain; what a connection
  does is learned.
- Senses: smell A on AWA, smell B on AWC, bacteria on the dopaminergic CEP, ADE and PDE,
  pain on the ASH nociceptors — each sense one declared input read by exactly those cells.
- Readouts: two single-patch columns, `forward drive` reading AVB and PVC live, `reverse drive`
  reading AVA, AVD and AVE live, exposed as the brain's two outputs. Their weights are learned
  like any others.
- Behaviour: when backward drive wins, the worm reverses and then makes an omega turn (a
  pirouette). It does so more often while forward drive falls, and it bends toward the side of
  its head swing on which forward drive rose, so smells steer it only through what the brain
  makes of them.
- Body: a centreline of exactly one body length. Whichever end leads lays the track and the
  rest of the body follows it: the head when crawling, the tail when reversing. The leading end
  can only change direction at a bounded curvature (10 rad/mm, a 0.1 mm radius), so an omega
  turn is a curl of the head that the body follows through, mostly toward the ventral side.
  Near the plate's edge either end must head inward, the more so the nearer the edge, so the
  edge is met with a turn, not a bounce; a nose that reaches it all the same backs off. The body wave advances with distance
  travelled (one wave per 0.7 mm). The motor neurons are part of the brain and carry activity,
  but nothing in the body reads them: the body reads the command interneurons only, and the
  wave and the manoeuvres are the body's own.

## How it learns, by the letter

The brain is used exactly as the library documents it:

1. Every tick, `sense` feeds the previous settled states back in with the current senses and
   commits one `step`. The settled command drives move the body, and the committed tick joins
   a stretch of the last eight, together with what the drives freely did.
2. When food or pain arrives, the stretch becomes one `observe_batch(source="estimate")`: on
   the last four ticks the drive that should have led there is asked to reach 0.8 and its
   rival to rest, and every earlier tick keeps the drives' own free prediction as its target,
   so it carries no correction — and the drives' biases cannot absorb the outcome.
3. The whole proposal must qualify: projected stationarity at most the configured tolerance
   (1e-4), with every parameter anchored to its pre-lesson value. A refused solve changes
   nothing, and is counted and shown. The live state is preserved; life continues.
4. Before birth the worm receives twelve such lessons in which pain drives the reverse group
   and quiet ticks are asked for rest: the withdrawal reflex it hatches with, and the
   discrimination that makes it a reflex rather than a habit. What smells mean is not
   supplied; it is learned from what the smells come before.

A treat on the page is food arriving; a poke is pain. Both are lessons like any other.

## The page

`web/` runs the same brain in the browser. `web/brain.js` is the cadence 0.50.0 reference
repair, ported operation for operation for graphs without residual readback (exact summation
included), with the same worm loop on top; `tests/parity.mjs` binds it to the library along
lived ticks and three lessons (worst relative difference 4.8e-8). `tests/body.mjs` checks after
every physics step that the body is one body length, inside the plate, and nowhere bent tighter
than it can bend: in six lives under stress (irritants crowded round the worm, a plate carpeted
with them, a corner, random pokes and treats), and in 300 bodies without a brain, started
against edges and in corners, reversing at random with their headings kicked (1.2 million
steps). The lives a brain leads differ between machines in the last digit of a float; the 300
do not. It also checks that the browser body lays the same track as the Python reference (to
1e-15 mm on three scripted crawls with reversals, omega turns and wall turns).

`web/card.jpg` is the page's social card, drawn by the page's own renderer:
`node worm/tools/make_card.mjs` (needs Google Chrome) photographs `tools/card.html` into it.
The page's canonical address and card address are written in `web/index.html`; change both
when the page moves.

`?seed=N` chooses the plate. `worm/` (`brain.py`, `connectome.py`, `life.py`, `world.py`,
`rng.py`) is the Python reference of the brain loop, the world and the life that the page ports.

## Run it locally

From the root of this repository, with Python 3 and node installed. The page is static and
needs no build step:

```bash
python -m http.server -d worm/web 8801     # then open http://127.0.0.1:8801
```

To rebuild the page's data from the sources and check it against the library:

```bash
python -m pip install "cadence-net==0.50.0" numpy scipy pytest
python worm/tools/build_connectome.py      # data/connectome.json from the sources
python worm/tools/export_web.py            # the newborn brain, web data and parity cases
python worm/tools/check_exports.py         # identical to the committed files where nothing is rounded, to 1e-6 elsewhere
node worm/tests/parity.mjs                 # the browser engine against the library
node worm/tests/body.mjs                   # the body's invariants under stress
python -m pytest -q worm/tests/test_learning_accounting.py
node worm/tests/learning-accounting.mjs    # lesson custody on both sides
```

## Supplied and learned

- Supplied: which connections exist and which act within a tick, which neurons sense what,
  which neurons the drives read, the body mechanics, the moments at which lessons happen, the
  targets a lesson constructs, the qualification tolerance.
- Learned: the weight of every connection and the bias of every patch, from seeded values,
  through twelve reflex lessons and then from the life.

## Limits

- Birth weights are not the measured synapse strengths: Cadence 0.50 has no documented way to
  install parameters, so synapse counts and transmitter signs inform the drawing, not the brain.
- The declaration order is a modelling choice: synapses running toward the commands act within
  a tick, synapses running back act across ticks. The connectome fixes which connections exist,
  not their timing.
- Neurons are rate units, not spiking cells. The body is a follow-the-leader curve, not a
  muscle model: the motor neurons do not drive it, and the brain has no oscillator or stretch
  feedback from which a body wave could arise.
- Food and pain are the only outcomes; a smell learns only by coming before one of them. A
  lesson's targets are constructed teaching values, declared `source="estimate"`, not observed
  witnesses.
- A receipt of conditioning with paired, unpaired, frozen and lesioned controls across seeds is
  not yet in this directory. Neural parity and visible weight changes do not establish robust
  conditioning or learned motor control.
- The displayed transmitter colors/clouds are illustrations driven by the model, not a
  reconstructed receptor, concentration or release-kinetics simulation.

## Build on it

Fork it and raise your own worm: other senses, other outcomes, a lesioned connectome, a body the motor neurons drive.
