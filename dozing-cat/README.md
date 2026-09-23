# The dozing cat

A cat dozes on a sill on one cheap moment of its `cadence.BeliefPatch` per decision. A laser dot
where the belief expected nothing is a surprise; a governor, a settling `cadence.Brain` of
fourteen neurons whose every synapse is a gene, reads that surprise off the brain's own readback
port and returns the mode: doze (the habit holds the paw), chase (ten pushes imagined in the
belief's private imagination, the best executed), or learn (the belief observes its executed
window and the habit is refitted). The page runs the whole brain in the browser with the
library's arithmetic, draws it live, and lets your mouse be the laser.

[![The sill with the cat chasing the laser dot, and beside it the whole brain drawn in three dimensions with its regions lit](screenshot.png)](https://floatingpragma.io/cadence-examples/dozing-cat/)

Live page: [floatingpragma.io/cadence-examples/dozing-cat](https://floatingpragma.io/cadence-examples/dozing-cat/).
The same page runs from this directory; see [Run it locally](#run-it-locally).

## Card

- **Name:** The dozing cat
- **Author:** Bernhard Mueller
- **Description:** One belief patch watches a sill through a coarse retina and predicts how the laser dot and its own paw will move; a settling patch of fourteen neurons reads the belief's surprise, residual and mode and decides whether the cat dozes, chases or learns. The page runs the belief, the governor, the imagination and the learning in the browser, with the cat's whole brain drawn beside the sill.
- **Cadence version:** 0.13.0, from the library's checkout at commit `f06eab06` (the belief patch's boundary state and per-unit repair step landed after the 0.13.0 release; the checks pin that commit). The receipts name the commits their numbers were produced with.
- **Hardware for initial training:** CPU only, one Apple M4 laptop shared with two other jobs. Pretraining the belief on 120 streams of 400 decisions took 61 s in one process. The four-arm lives (four governors, five seeds, 4,000 decisions each) took about six minutes on a pool of three; the two evolutions with their random-search controls and held-out scoring took 495 s and 388 s (evolution) plus 959 s and 585 s (random search) on the same pool. Wall times were measured under load and are upper bounds; the moment counts in the receipts are exact.
- **Cadence features showcased:** `cadence.BeliefPatch` with a `StructuredPort` (a learned transition under action, evidence repair by iteration, the record store read inside the repair, `assimilate`, `imagine` and `observe` with a boundary state, `snapshot` and `restore` for a rollback); `cadence.Brain` with `NeuronModel` and `Connectome.from_synapses` as a settling governor whose mode is its equilibrium (`equilibrate` with a step budget and a tolerance); `cadence.evolve` with `cadence.genes` over the governor's 77 genes and the threshold rule's 12; the readback port as the thing a metacognitive patch reads; a browser twin of all of it held to the library by a parity test.
- **Problems encountered during development:**
  - The first belief predicted the change of every retina cell and explained 0.005 of it after 60 epochs: a dot's appearance is a jump of tens of standardized units and the loss taught it with fifty times the weight of the motion. The belief predicts the fovea's readout (centroid, mass) and the paw instead, five channels.
  - A velocity the belief had to infer from two consecutive centroids through the recurrence was learned to 0.008 after 65 epochs. The fovea now reports the optic flow as a reading, and the motion is explained 0.58.
  - Filling the record store with the routine corpus raised the quiet moments' error from 0.0009 to 0.41: an appearance's evidence is a quiet field, so the store wrote the jump's residual into the cells every quiet moment activates. The store is left empty; it reads exactly zero, and the page skips the address of a read it does not draw.
  - A slow average of the raw surprise fired the learn unit after every dot (68 learn calls in a life). The slow readback is the average of the log-compressed ratio, so a spike moves it a twentieth of a unit and a changed law a whole one.
  - With the imagine unit feeding itself 0.6, the quiet after a catch kept the cat awake half the time it had no dot; at 0.4 the self-excitation holds a chase and not a quiet field. Selection then set it to zero.
  - Candidates centred on the habit's push, as an earlier room did, could not leave home once a refit had made the habit a strong drift. The candidates are rest, the eight directions and the habit's own push.
  - A cat led by a slow dot closes to about a tenth of the sill and rests there: a full push held for six imagined decisions costs more than the imagined gain, so a slowly moving dot is caught when it comes to the paw or the paw's drift crosses it. Scheduled dots, which walk at 0.015 per decision and reflect, are caught 0.96 of the time.
  - The cat's page needed a laser that feels instant. The dot is drawn at the pointer's place at every animation frame and the brain reads the latest place once per decision; a click toggles the laser exactly once and a drag never does.
- **Hosted at:** https://floatingpragma.io/cadence-examples/dozing-cat/
- **Receipts and checks:** `receipts/cat_lives_hand_set.json` and `receipts/cat_lives_evolved.json` (the four arms and the evolved genomes, five lives each), `receipts/cat_evolution_patch.json` and `receipts/cat_evolution_threshold.json` (six generations, the random-search control, the held-out lives), `receipts/cat_timing.json` (the moment counts of every piece), `pretrained/cat_belief.json` (the pretraining and its held-out grades), `tests/parity.mjs` (the browser brain against a recording of the library's, worst relative difference 4.8e-12 on the committed recording, which is rounded to twelve significant digits, and 7.5e-15 on a full-precision one), `tests/chaos.mjs` (the sill's invariants under random paths), `verify.py`. `python dozing-cat/verify.py && node dozing-cat/tests/parity.mjs && node dozing-cat/tests/chaos.mjs` from the repository root.
- **Data and rights:** Nothing external. The sill, its dots and the pretraining corpus are generated by `cat.py` from their seeds; the pretrained belief is `pretrained/cat_belief.npz` (58 kB) and may be redistributed with the rest under the repository's licence.
- **Work in progress:** a sill on which the changed law costs catches, so that learning is priced in and the evolved governor learns; the imagination's horizon, spread and candidate set as genes; a per-moment weight on the loss and a write mask for the store, so the store can be filled without drowning in appearances; a life whose learn calls fire under the mouse.

## What this example shows

- **A brain that reads its own surprise.** The belief's one-step read against what it then read is one number per decision. A governor of fourteen neurons reads it over its baseline, its slow average, the repair residual and the current mode, settles, and its settled motor state is the mode. Nothing in the governor is a threshold written by hand; every synapse is a gene, and the genome the page runs is the one that won on held-out lives.
- **Habit, imagination and learning as modes of one patch.** Dozing is one moment of the belief patch per decision. Chasing is sixty: ten candidate pushes held for six decisions in the belief's private imagination, which writes nothing. Learning is `observe` on the executed window from the belief that was live at its start, kept only when the window's loss fell by a tenth, then a refit of the habit in imagination.
- **The cost is counted.** Every moment of the patch is priced, the governor's own settle included (its 23 synapses times about 20 steps are a two-hundredth of a moment). The fitness is the catch rate minus the priced compute minus the priced surprise, and selection ran against a random search of the same size.
- **A control and two brains without a governor.** Hand-set thresholds over the same readback, a brain that never wakes, and one that is always awake bracket the governor on the page's selector.
- **The whole brain, live in the browser.** The belief patch, the governor, the imagination and the learning run in the page with the library's arithmetic, and a parity test replays recorded decisions of the Python brain through the same code.

## Layout

| file | what it is |
| --- | --- |
| `cat.py` | the experiment: the sill, the readings, the belief patch, the habit, the two governors and the two controls, the metrics, the fitness, evolution, the receipts |
| `pretrained/cat_belief.npz`, `.json` | the pretrained belief the page and every life start from, and its pretraining record with the held-out grades and the fitted habit |
| `receipts/` | the lives, the evolutions and the timing (see the card; each is the experiment's receipt with the paths of the machine it was made on replaced by paths relative to this directory and the digest re-sealed, the original digest kept inside) |
| `export.py` | writes `web/data/brain.json` (the belief, the genomes, the constants) and copies the library's viewer to `web/brain_scan.js`, recording its commit |
| `web/` | the page: `cat.js` (the sill, the readings, the habit), `belief.js` (the belief patch: moment, imagination, learning), `governor.js` (the settling patch), `life.js` (the life: readback, modes, the executed window, the ledger), `connectome.js` (the brain as the viewer draws it), `page.js`, `index.html`, `style.css`, `brain_scan.js` (the library's viewer) |
| `tests/record_parity.py` | records the Python brain's decisions along a scripted dot path, with two forced learn calls, into `tests/parity_cases.json` |
| `tests/parity.mjs` | replays that recording through the browser brain and holds it to the library |
| `tests/chaos.mjs` | random dot paths and scheduled dots under every brain: the sill's invariants and the catch rule |
| `tools/check_page.py` | the headless check of the page (Playwright): console errors, overflow, the laser, a catch, the brain drawing |
| `verify.py` | recomputes the receipts, the numbers this README states, and the export the page runs |

## Run it locally

From the root of this repository, with Python 3, node and the library at the commit the checks pin. The page is
static and runs the brain in `web/data/brain.json`:

```bash
python -m http.server -d dozing-cat/web 8805     # then open http://127.0.0.1:8805
```

To check the page's brain and the receipts beside it against the library:

```bash
python -m pip install "cadence-net @ git+https://github.com/muellerberndt/cadence@f06eab065803a61e92fc2f578f24e91eaee48e79"
python dozing-cat/verify.py                                                   # the receipts, the README's numbers, the export
node dozing-cat/tests/parity.mjs                                              # the browser brain against the committed recording
node dozing-cat/tests/chaos.mjs                                               # the sill under chaos
python dozing-cat/export.py --out /tmp/dozing-cat --no-viewer                 # a rebuild of the export on this machine,
python dozing-cat/tests/record_parity.py --data /tmp/dozing-cat/brain.json --out /tmp/dozing-cat/parity_cases.json
node dozing-cat/tests/parity.mjs --data /tmp/dozing-cat/brain.json --cases /tmp/dozing-cat/parity_cases.json   # and a fresh recording
```

## Reproduce the experiment

    python dozing-cat/cat.py pretrain --episodes 120 --epochs 60 --empty-store --out dozing-cat/pretrained/cat_belief.npz
    python dozing-cat/cat.py lives --out dozing-cat/receipts/cat_lives_hand_set.json                                    # the four arms, hand-set genomes
    python dozing-cat/cat.py evolve --which patch --out dozing-cat/receipts/cat_evolution_patch.json
    python dozing-cat/cat.py evolve --which threshold --out dozing-cat/receipts/cat_evolution_threshold.json
    python dozing-cat/cat.py lives --arms patch,threshold --patch-genome dozing-cat/receipts/cat_evolution_patch.json --threshold-genome dozing-cat/receipts/cat_evolution_threshold.json --out dozing-cat/receipts/cat_lives_evolved.json
    python dozing-cat/cat.py timing --out dozing-cat/receipts/cat_timing.json
    python dozing-cat/export.py

The sill, the dots and the corpus are drawn from their seeds; a life meets the same dots on every machine, and the
pretrained belief differs between machines in the last digits of its floats.

## What was measured

The sill is the unit box. The paw moves 0.05 per decision at a full push. A dot appears 40 to 140 decisions after the
previous one vanished, walks, curves or jitters at 0.015 per decision and reflects at the walls; the paw within 0.08
of it for three decisions is a catch, an uncaught dot vanishes after 150 decisions as a miss. At decision 2,400 of
4,000 the dot's law changes for good: gravity. The brain reads a six-by-six retina of Gaussian bumps over the dot, the
fovea's readout (centroid, optic flow, mass) and its own paw: 43 readings, never the dot's law.

The belief is one `BeliefPatch` of 32 units with a dense port of 43 readings to 24 encoded units, two actions, two
repair iterations at damping 0.5, a store of 512 cells left empty, and five outputs (the change of the dot's centroid,
of its mass and of the paw, in units of their motion): 6,853 parameters, 96,488 multiply-accumulates per moment.
Pretrained on 120 streams of 400 decisions of random and pursuing pushes, held out it explains 0.94 of the paw's change
and 0.58 of the dot's motion; its open-loop imagination explains 0.39 of the centroid's change eight decisions ahead;
its routine surprise (the mean over the five channels of the squared standardized error) has a median of 0.016, the
baseline the governors start from.

**The four arms at the hand-set genomes** (`receipts/cat_lives_hand_set.json`, five seeds, means):

| measure | governor patch | hand-set thresholds | always awake | never wakes |
|---|---|---|---|---|
| catch rate | 0.977 | 0.931 | 0.962 | 0.249 |
| catches, misses per life | 26.4, 0.6 | 24.8, 1.8 | 25.6, 1.0 | 4.0, 12.0 |
| awake share | 0.365 | 0.271 | 1.000 | 0.000 |
| moments per decision | 37.3 | 26.2 | 71.3 | 1.0 |
| detection of the changed law, decisions | 99.6 | 158.0 | 159.0 | none |
| fitness | 0.782 | 0.783 | 0.631 | 0.167 |

**Selection** (`receipts/cat_evolution_patch.json`, `receipts/cat_evolution_threshold.json`): six generations of twelve
with the four best as parents, `cadence.evolve` with `cadence.genes` at mutation rate 0.3, one training life per genome,
against a random search of 72 genomes drawn uniformly over the same space; the three genomes scored on five held-out lives.

| genome | held-out fitness | catch rate | moments per decision | awake share |
|---|---|---|---|---|
| governor patch, hand-set | 0.782 | 0.977 | 37.3 | 0.365 |
| governor patch, evolved | **0.853** | 0.961 | 15.3 | 0.238 |
| governor patch, best of random search | 0.642 | 0.857 | 42.1 | 0.205 |
| thresholds, hand-set | 0.783 | 0.931 | 26.2 | 0.271 |
| thresholds, evolved | **0.795** | 0.909 | 17.5 | 0.269 |
| thresholds, best of random search | 0.788 | 0.952 | 29.4 | 0.283 |

The patch lineage climbed from 0.824 to 0.906 on its training life and its winner beats the hand-set wiring on the
held-out lives and the best of 72 random draws. It moved 65 of its 77 genes: the fast surprise drives the imagine unit
harder (1.0 to 1.85), the imagine unit's self-excitation is gone, the cortex the hand-set wiring left at zero carries
the slow surprise and the current mode. The result wakes within a decision of a dot (wake latency 1.0), sleeps as
soon as the dot is gone (awake 0.80 with a dot on the sill, 0.03 without; 14 false wakes per life against 5.4 for the
thresholds) and never learns: zero learn calls on every held-out seed (`receipts/cat_lives_evolved.json`). The
threshold lineage found nothing beyond its first generation. Selection switched learning off in both lineages at these
prices: under gravity the dot stays slower than the paw, every governed arm catches every dot after the change with or
without learning, and the priced surprise a kept update saves is a fortieth of what the observe passes and the refit
cost.

**The falsifier.** The settling governor does no better than hand-set thresholds at matched compute: at the hand-set
genomes the two tie on the fitness (0.782 against 0.783; the patch catches more at more compute). After selection over
each genome's own space at an equal budget of 72 lives, the patch wins on every part: fitness 0.853 against 0.795,
catch rate 0.961 against 0.909, 15.3 against 17.5 moments per decision, awake share 0.238 against 0.269. The
advantage is what 77 genes let selection do (a cortex, self-excitation, a learn unit with its own bias) and 12 do not.

**Timing** (`receipts/cat_timing.json`, the moment counts exact, the wall times measured under load): one moment of
the patch is 96,488 multiply-accumulates; one imagination call is 60 moments; one settle of the governor patch is 23
synapses over about 20 steps, 0.005 moments; one learn call is 1,824 moments (six observe passes over 96 decisions and
the check).

## What the browser brain does differently

The page runs the belief patch's moment, its imagination, its learning (the adjoint over the executed window, the
validity check, the rollback and the habit's refit in imagination), the governor's settle, the threshold rule and both
controls in plain JavaScript in the library's order of operations; the parity test replays recorded decisions of the
Python brain through that code and holds beliefs, imagination costs, modes, actions, readbacks, governor activations
and settle steps to a relative difference under 1e-9 (4.8e-12 achieved on the committed recording). What differs:

- The decisions run at 25 per second on a fixed timer; the experiment's lives run as fast as the machine allows and
  the demo it came from ran at the same 25.
- The random starts of a habit refit are drawn from the page's generator (Mulberry32), so a refit on the page lands
  on genes of its own; the parity test feeds the refit the starts the library drew and reproduces its genes.
- Scheduled dots (the driven check's `?script=1`) are drawn from the page's generator; under the mouse the sill has no
  randomness and is the experiment's sill exactly.
- The wall-clock milliseconds per decision are the browser's.
- The empty record store reads exactly zero, and the page skips the address of a read it does not draw; the code
  path stays, and a store with records in it is read in full.

## Open

- Learning fires rarely under the mouse and never under the evolved governor; a sill on which the changed law costs
  catches would price it in.
- The imagination's horizon, spread and candidate set, the learning window, rate and passes, and the readback's form
  are constants here and belong in a genome.
- The store is empty because filling it drowned the belief in appearances; a per-moment weight on the loss and a
  write mask would let it hold the routine.

## Build on it

Fork this directory and give the cat a new sill: a dot with a law that costs catches, a second sense, a paw with
inertia. The world, the belief, the governor, the life and the page are separate files, so one can be replaced while
the rest keeps measuring; `cat.py lives` and `cat.py evolve` score any governor you write against the same random
search, and the parity test tells you whether your page still runs what your Python runs.
