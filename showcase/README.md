# Cadence live systems

Six separate browser examples expose observer-like software patches: bounded
local state, ports, readback, retained records and feedback. Each page puts the
actual circuit beside the task at the top on desktop; mobile stacks the panels.

## Launch any demo

From the repository root, run `python serve.py eye-arm`, `python serve.py mouse`,
`python serve.py worm`, `python serve.py fly`, `python serve.py memory`, or `python serve.py connect-four`.
Python 3.11+ is enough to serve them. No account, packages or GPU are required.
The launcher opens the example's own directory URL on an available local port.
Use `--no-browser` to print the address. Open `/` for the gallery.

## A two-minute demonstration

1. **Eye & arm:** clear the left pad and draw a simple mark. The eye reads its
   pixels; the arm raises the pencil, moves its joints and lowers it to draw.
   Disable **Pencil motors** after **Copy again**: no ink appears. Restore them,
   then disturb a joint to watch feedback correct the pose.
2. **Mouse:** teach the fourth task, perform it, then change a corridor. Task
   memory retains the lesson while spatial and motor state update. Disable
   **Motor neurons** to stop the body. New maze retains the task lessons.
3. **Worm:** place food and draw a wall. Odor, chemical-circuit activity and
   directional motor neurons determine the next body step. Open a passage to
   restore access to sealed-off food. Switch to Circuit to inspect the supplied
   chemical network and its MLP comparator.
4. **Forager:** watch nectar contact update memory, then change the nectar.
   Motor populations turn and propel the body toward selected flowers.
5. **Memory:** replace an observed value and compare retention against online
   MLP updates on the same observations.

6. **Connect Four:** select **Watch before moving**, play a column, and inspect
   the predicted replies before executing the preferred move. Disable the
   self-monitor to cap the normal search at four plies.

## Ready to run and watch learning

| Example | Initial state | What changes |
|---|---|---|
| Eye & arm | Supplied geometry and pixel/motor circuit; a sample drawing | Retinal, error, motor and proprioceptive activity; no weight training |
| Mouse | Three cue/destination demonstrations | New task associations persist in browser storage; field and motor state follow the body |
| Worm | Public chemical graph plus an engineered directional motor circuit | Odor and body/circuit state; fixed weights |
| Forager | Working sensor/motor loop, blank nectar memory | Contact reveals nectar and changes associative weights |
| Connect Four | Supplied game rules, threat evaluator and monitor | Hypothetical boards, option values and budget readback; no weight training |
| Memory | Blank record store and randomly initialized MLP | Both update on each demonstrated key/value pair |

The memory example isolates a subsystem; it has no invented body. The embodied
examples implement the complete **task controller**, with supplied encodings,
attention/readout rules, wiring and physical bodies. They do not reconstruct
complete biological brains, learned perception or learned anatomy.

## Why Cadence fits each task

| Example | Mechanism and supported advantage |
|---|---|
| Eye & arm | Visual error and joint coordination exchange local feedback; motor outputs actuate the body. Readback repairs disturbances. This is a feedback/ablation comparison, not a trained-MLP comparison. |
| Mouse | A learned task record selects a destination; a spatial equilibrium updates routes and motor state follows positional error. BFS and dictionary lookup are strong conventional controls and also work. |
| Worm | The chemical graph is reused after stimulation or lesions. Its equilibrium agrees with the numerical reference; a trained MLP is faster but approximate. The habitat adds engineered motor control and is separately tested. |
| Forager | One residual write revises a contacted flower's value. Both learners share the same motor design; live trajectories contain different experiences, so nectar totals are illustrative. |
| Connect Four | Isolated futures and activity readback control search depth. The same evaluator wins more scheduled games with lookahead; ordinary minimax can also do this. |
| Memory | Distinct-key records can be replaced locally with exact retention. Dictionary storage is exact too; overlapping keys interfere and can favor the tested MLP. |

## What the brain actually contains

- **Arm:** 24 × 24 retinal units read pixel darkness. An explicit attention rule
  chooses an unvisited visible target. Six sensory/readback units carry target,
  pen position and height; three error units, two joint-coordination units and
  six antagonistic motor units complete the controller. Motor output moves
  shoulder, elbow and pencil height. Paper contact creates ink. The controller
  receives pixels, never the user's stroke coordinates.
- **Mouse:** task memory → destination lookup → supplied occupancy map/spatial
  field → positional readout → antagonistic directional motors → body position.
- **Worm:** adjacent odor → public chemical network and directional odor readback
  → motor population → body step → contact consumption.
- **Forager:** visible flower cue/bearing → nectar memory and target selection →
  turn/propulsion motor units → body → contact reward and memory write.

Circuit seams are weighted local connections. An explicit application readout
(such as selecting a visual target) is documented as a readout, not drawn as an
invented synapse. Physics includes joint kinematics, inertia/contact and collision
bounds. Motor ablations test whether the neural output really causes movement.

## Circuit sizes and biological references

| Demo | Allocated owners | Declared seams | Learned entries |
|---|---|---|---|
| Eye & arm | 593: 576 retina + 17 controller | 20 | 0 |
| Mouse, seed 13 | 265; 132 unmasked initially | 278 | 32 |
| Worm habitat | 309: 297 chemical + 12 directional | 3,612 | 0 |
| Worm circuit probe | 297 | 3,604 | 0 |
| Forager, Cadence agent | 18: 12 memory + 6 sensory/motor | 36 | 32 |
| Changing memory | 12 | 32 | 32 |

The mouse allocates 247 spatial slots (133 initially masked walls), 12 memory
ports and six sensor/motor owners. Its initial seams are 242 spatial, 32 memory
and four motor. Maze edits change the counts. Retina owners have independent
sensory drives; they do not add a dense all-to-all matrix. These counts exclude
body/environment variables, attention records and comparison MLPs.

For orientation, adult hermaphrodite C. elegans has 302 neurons, including 20
pharyngeal neurons ([WormAtlas](https://www.wormatlas.org/hermaphrodite/nervous/mainframe.htm)).
Software state coordinates are not biologically equivalent neurons or measures
of animal intelligence. No whole-animal equivalence is claimed.

## Read the brain view

Regions group the actual task components by function. The circuit appears beside
the body at the top of each page. Hover or tap an owner to see its numerical value.

- **Behavior badge:** green = positive outcome; blue = seeking/moving; amber =
  correction; gray = idle/paused. These name observed events, not measured feelings.
- **Activity / input / repair:** amber = positive, blue = negative. Values are
  dimensionless and scaled within a region; replay scales remain fixed.
- **Violet repair trails:** one second of display history. They are not neural
  memory. **Plasticity** separately highlights learned weight changes.
- **Following repairs:** reconstructs sampled component trajectories from captured
  inputs, weights and initial state. It expands early iterations and compresses
  later ones into 1.6 seconds. Rapid updates are coalesced. Components can have
  different iteration counts; shorter trajectories hold their final state.
  This view is not a synchronous recording of every body tick.
- **Replay repair:** inspect a captured update at a selected iteration rate.
  Stateful motor circuits start from their retained potentials, not from zero.
- **Release input:** removes drives in an isolated copy and shows recurrent decay.
  It does not change the body or its actual memory.
- **Live:** returns to current samples; **Pause view** freezes only the viewer.
  Body controls are independent. Reduced-motion preferences disable autoplay.

The current motor circuits retain graded potentials between control ticks, so
transient neural state really is carried during behavior. Associative weights
and visited-target records provide different forms of retained information.
Neither transient motor state nor the isolated release probe demonstrates a
learned working-memory task. The mouse's spatial field is recalculated after map
or goal changes. Memory-only circuits have no recurrent-decay probe.

## Brain colors and neurotransmitters

The viewer colors **model state, input drive, state changes or learned weights**.
It does not model dopamine, serotonin, glutamate or GABA concentrations. Numerical
values are dimensionless model units, not blood oxygenation, tracer binding or
chemical concentration. Signed weights do not identify a neurotransmitter.

Brain imaging uses different measurements:

- Structural MRI provides anatomical contrast. Functional MRI commonly maps
  changes related to blood oxygenation as an indirect activity signal; a colored
  activation map is not a neurotransmitter map.
  [NIH MRI introduction](https://www.nibib.nih.gov/science-education/science-topics/magnetic-resonance-imaging-mri)
- PET uses a chosen radiotracer to investigate particular molecular targets,
  including neurotransmitter receptors and transporters. Appropriate tracer
  experiments and models can infer changes in transmitter release; this is not
  a direct, simultaneous live measurement of every neurotransmitter.
  [NIMH PET research](https://www.nimh.nih.gov/research/research-conducted-at-nimh/research-areas/clinics-and-labs/mib/sprs/section-on-pet-radiopharmaceutical-sciences),
  [dopamine-release experiment](https://pubmed.ncbi.nlm.nih.gov/18442926/)
- MR spectroscopy can estimate regional chemical pools, such as glutamate.
  A regional concentration estimate is distinct from moment-to-moment synaptic
  release. [Combined PET/MRS study](https://www.nature.com/articles/s41398-021-01515-3)

An application that explicitly computes a reward prediction error could expose
that scalar as a **modulatory learning signal**. Calling it a dopamine
concentration would require a biochemical model and calibration. The current
showcase plots only quantities its controllers actually compute.

## Food and walls in the worm habitat

Food drives leaky diffusion in a 31 × 21 grid. Walls block odor flux and movement.
The worm reads adjacent odor values; their maximum drives the public AWA/AWC
sensory ports. Chemical motor activity enables four directional error channels.
Eight antagonistic motor units encode their response, and a readout selects a
positive direction. No path or remote food coordinates enter this controller.

Contact removes food and recomputes the field. Smell ablation, chemical-motor
ablation, directional-motor ablation and a sealed wall all stop food acquisition
in the scheduled tests. Paint with mouse/touch; focus the habitat and use arrows
plus Space/Enter for keyboard editing. Body cells cannot be painted into walls.

Real C. elegans feeds using its pharynx ([WormBook](https://www.ncbi.nlm.nih.gov/books/NBK116080/)).
Contact consumption here does not simulate pharyngeal pumping or digestion.
The 297-cell fixture is not the full 302-neuron nervous system; gap junctions are
excluded and chemical weights/dynamics are imposed.

## Results and comparison contract

See the [per-example advantage contracts](../ADVANTAGES.md), including measured
memory runtime and all scheduled Connect Four outcomes. Each page separates
learned records, supplied rules, feedback effects and conventional controls.


Current task-controller evidence is in each embodied example's `evidence.json`,
produced by `tools/nervous_system_benchmark.mjs`. All controller sources are bound
by hashes, all scheduled conditions are retained, and CI reruns the producer.

- **Arm:** square, flower and two separated marks, six conditions each, 6,000
  control steps. Intact and disturbed-feedback cases cover all retinal targets.
  With pose feedback disabled after the same disturbance, coverage is about
  32–42%. Silencing joint motors gives zero joint displacement. Silencing pencil
  motors while raised gives zero ink, as does removing visual input. Coverage
  means target distance under 0.024 normalized units; it does not measure artistic
  fidelity. User drawings can be harder than these fixtures.
- **Mouse:** 12 seeds × intact, moved goal and motor ablation. The 24 reachable
  navigation trials finish without collision; motor ablations produce no moves.
- **Worm:** intact habitat consumes two patches in 41 moves. Smell, either motor
  population, and sealed-wall controls acquire no food.
- **Forager:** a fixed 4,000-step run produces 25 contacts; motor ablation prevents
  displacement and contact. This is an actuator test, not a learner comparison.

The shared [evidence.json](evidence.json) retains the unchanged matched memory
and chemical-circuit benchmarks: seeds 7–9, 128 memory writes per stream with all
seen-key queries, and three held-out chemical-circuit conditions. The memory MLP
gets 1/10/100 updates on the same sample; correlated keys can favor it. The worm
MLP gets 2,048 labels and 1,200 training steps per candidate; it is faster per
query but approximate under interventions. These are bounded comparisons, not
universal speed or energy claims.

[composite_evidence.json](composite_evidence.json) and
[habitat_evidence.json](habitat_evidence.json) retain the original spatial,
four-owner arm and habitat reference experiments. They validate those reference
kernels; the expanded browser controllers are measured in the new task receipts.

## Reproduce

Serving requires only Python. Reproduction also uses Node and the pinned Cadence
dependencies:

```bash
python -m pip install -r requirements-reproduce.txt pytest playwright
python showcase/verify.py
node tools/nervous_system_benchmark.mjs
node connect-four/benchmark.mjs
node memory/history_benchmark.mjs
python -m pytest -q tests
python tools/showcase_pages.py
python tools/build_showcase.py
```

The test suite independently checks motor dynamics against Python Cadence from
retained state and under lesions, replay endpoints, motor/visual ablations,
freehand drawing, image upload, task persistence, separate URLs and responsive
layout. `build_showcase.py` generates six pages plus the gallery from authored
shells. Models and views remain in their example folders; shared code is in
`showcase/`.

To rerun the older comparison training and reference body trials:

```bash
python showcase/fetch_worm.py
python showcase/benchmark.py --seeds 3 --steps 1200 --trials 128
python showcase/build_composites.py
node showcase/habitat_benchmark.mjs
```

## Biological sources and data attribution

The worm chemical edge table and cell descriptions are from
[OpenWorm ConnectomeToolbox](https://github.com/openworm/ConnectomeToolbox), pinned
in `fetch_worm.py`; its [MIT notice](OPENWORM_LICENSE.txt) is included. The neuron
list comes from [c302](https://github.com/openworm/c302) at the pinned revision.
Source URLs and digests are retained in `worm.json`.

Anatomical context: [Cook et al. (2019)](https://www.nature.com/articles/s41586-019-1352-7).
Body-feedback motivation: [Wen et al. (2012)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3508473/).
For the distinction between a circuit response model and a whole behaving fly,
see [Shiu et al. (2024)](https://www.nature.com/articles/s41586-024-07763-9).
These papers motivate the examples; their biological validation does not transfer
to the imposed rules or simplified bodies used here.
