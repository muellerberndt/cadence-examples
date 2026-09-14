# Cadence live systems

Five official examples expose Cadence brains: graded neurons with bounded
local state, ports, readback, retained records and feedback. Four run in the
browser. The composer studio runs locally with its pretrained model. Each page puts
the actual circuit beside the task at the top on desktop; mobile stacks the panels.

## Launch any demo

From the repository root, run `python serve.py eye-arm`, `python serve.py worm`,
`python serve.py fly`, or `python serve.py connect-four`.
Python 3.11+ is enough to serve them. No account, packages or GPU are required.
`python serve.py composer` starts the composer studio once its requirements and the
maestro-1 model are installed ([composer/README.md](composer/README.md)).
The launcher opens the example's own directory URL on an available local port.
Use `--no-browser` to print the address. Open `/` for the gallery.

## A two-minute demonstration

1. **Eye & arm:** clear the left pad and draw a simple mark. The eye reads its
   pixels; the arm raises the pencil, moves its joints and lowers it to draw.
   Disable **Pencil motors** after **Copy again**: no ink appears. Restore them,
   then disturb a joint to watch feedback correct the pose.
2. **Worm:** place food and draw a wall. Odor, chemical-circuit activity and
   directional motor neurons determine the next body step. Open a passage to
   restore access to sealed-off food. Switch to Circuit to inspect the supplied
   chemical network and its MLP comparator.
3. **Composer:** choose a mood and compose. Watch candidate continuations
   compete in the piano roll, then compare the draft with the edited final piece
   while every neuron replays in sync with playback.
4. **Forager:** watch nectar contact update memory, then change the nectar.
   Motor populations turn and propel the body toward selected flowers.
5. **Connect Four:** select **Watch before moving**, play a column, and inspect
   the predicted replies before executing the preferred move. Disable the
   self-monitor to cap the normal search at four plies.

## Ready to run and watch learning

| Example | Initial state | What changes |
|---|---|---|
| Eye & arm | Supplied geometry and pixel/motor circuit; a sample drawing | Retinal, error, motor and proprioceptive activity; no weight training |
| Worm | Public chemical graph plus an engineered directional motor circuit | Odor and body/circuit state; fixed weights |
| Composer | Pretrained maestro-1 musician | Working memory, form record and intention state per event; the draft is revised, trained synapses stay fixed while composing |
| Forager | Working sensor/motor loop, blank nectar memory | Contact reveals nectar and changes associative weights |
| Connect Four | Supplied game rules, threat evaluator and monitor | Hypothetical boards, option values and budget readback; no weight training |

The embodied examples implement the complete **task controller**, with supplied encodings,
attention/readout rules, connectomes and physical bodies. They do not reconstruct
complete biological brains, learned perception or learned anatomy.

## Why Cadence fits each task

| Example | Mechanism and supported advantage |
|---|---|
| Eye & arm | Visual error and joint coordination exchange local feedback; motor outputs actuate the body. Readback repairs disturbances. This is a feedback/ablation comparison, not a trained-MLP comparison. |
| Worm | The chemical graph is reused after stimulation or lesions. Its equilibrium agrees with the numerical reference; a trained MLP is faster but approximate. The habitat adds engineered motor control and is separately tested. |
| Composer | Hearing, working memory, form record and intention settle in one brain trained by local contrasts. It imagines continuations through its own predictions and edits its weakest passage. A GRU trained with backpropagation through time predicts held-out events better ([model card](composer/MODEL_CARD.md)). |
| Forager | One residual write revises a contacted flower's value. Both learners share the same motor design; live trajectories contain different experiences, so nectar totals are illustrative. |
| Connect Four | Isolated futures and activity readback control search depth. The same evaluator wins more scheduled games with lookahead; ordinary minimax can also do this. |

## One joint state per task

Different functions share a common equilibrium through their synapses. The
controllers assemble their regions **before** settling, then read the resulting
motor or decision state. The circuit panel reports global and regional equation
errors. [Task connectomes and reproducible tests](COUPLED_BRAINS.md) describe every
connection, boundary adapter and causal control. Self-consistency under current
input is distinct from globally optimal task performance.

## What the brain actually contains

- **Arm:** 48 × 48 pixels are thresholded into sparse dark-sample patches.
  Each has reference, actual-ink readback and missing-mark neurons. An explicit
  attention rule consumes missing-mark activity and follows connected dark
  pixels, keeping the pencil down on strokes and lifting across blank space.
  Six sensory/readback units carry target, pen position and height; three
  position/height error units, two joint-coordination units and six opposing
  motor units complete the controller. Contact writes the same 192 × 192 ink
  raster that the eye reads and the view displays. Erased marks reactivate
  repair. The controller receives pixels, never the user's stroke coordinates.
- **Worm:** adjacent odor → public chemical network and directional odor readback
  → motor population → body step → contact consumption.
- **Composer:** the last 16 note events, pitch intervals, chroma and beat → melody,
  harmony, rhythm, timbre and form cortices → phrase cortex with working memory and a
  bar-keyed form record → intention over pitch, duration, attack, family and velocity.
  [MUSICIAN.md](composer/MUSICIAN.md) gives every region and synapse block.
- **Forager:** visible flower cue/bearing → nectar memory and target selection →
  turn/propulsion motor units → body → contact reward and memory write.

Circuit synapses are weighted local connections. An explicit application readout
(such as selecting a visual target) is documented as a readout, not drawn as an
invented synapse. Physics includes joint kinematics, inertia/contact and collision
bounds. Motor ablations test whether the neural output really causes movement.

## Circuit sizes and biological references

| Demo | Allocated neurons | Declared synapses | Mutable memory values |
|---|---|---|---|
| Eye & arm | 3N + 17 for N dark samples: reference, ink, missing marks + controller | up to 2N + 28 | 0 |
| Worm habitat | 309: 297 chemical + 12 directional | up to 4,108 | 0 |
| Worm circuit probe | 297 | 3,604 | 0 |
| Forager, Cadence agent | 18: 12 memory + 6 sensory/motor | up to 44 | 32 persistent + 32 transient |
| Connect Four | 19: 6 evaluator + 7 candidates + 6 monitor | 39 | 0 |
| Composer, maestro-1 | 17,855 in 14 regions | 68,570,458 directed | 52,849,100 trained parameters |

Retina neurons have independent
sensory drives; they do not add a dense all-to-all matrix. A memory contact carries a persistent component and a transient residual; 64 stored scalars do not mean 64 anatomical synapses. These counts exclude
body/environment variables, attention records and comparison MLPs.

For orientation, adult hermaphrodite C. elegans has 302 neurons, including 20
pharyngeal neurons ([WormAtlas](https://www.wormatlas.org/hermaphrodite/nervous/mainframe.htm)).
A Cadence neuron is a graded rate unit; it is not a biologically equivalent cell,
and these counts are not measures of animal intelligence. No whole-animal equivalence is claimed.

## Read the brain view

The circuit appears beside the body at the top of each page as one integrated brain,
drawn by the standard Cadence brain scan (`shared/brain_scan.js`, copied from the
cadence library by `tools/sync_brain_scan.py`). Every declared neuron and directed
synapse is mapped; there is no representative-neuron or strongest-synapse subset.
Regions are the task components; they are placed next to the regions they connect to,
image-like regions keep their grid, and inside a region each neuron sits among the
neurons it talks to. Region colour follows its role: blue vision, green sensory and
cue, violet memory, teal association, orange motor, rose value. Labels identify the
exact component; colour expresses design intent, not independently established
neural specialization. The tissue's brightness is the selected signal on its region
scale, the hot glow (violet, magenta, orange, white) is the change that just happened,
synapses light up when their source changed, and particles travel along synapses in
proportion to the message sent, so a settling reads as a wave. Dense synapses overlap
when zoomed out. Wheel/pinch zooms, dragging pans, **Fit whole brain** resets the
camera and **Expand** opens the map full screen. Hover or tap a neuron to inspect its
value.

WebGL2 retains the full graph on the GPU and caches its static connection image.
Actual changes supply the activity overlay. A Canvas fallback draws the neurons when
WebGL2 is unavailable. Rendering work and artificial replay time are excluded from
numerical efficiency comparisons.

- **Behavior badge:** green = positive outcome; blue = seeking/moving; amber =
  correction; gray = idle/paused. These name observed events, not measured feelings.
- **Activity / input / activity change / equation mismatch:** amber = positive, blue = negative. Values are
  dimensionless and scaled within a region; replay scales remain fixed.
  Change and mismatch colors use logarithmic magnitude above a 1e-8 noise floor,
  making small repairs visible alongside large ones. Numerical readouts are unscaled.
- **Violet activity-change trails:** one second of display history. They are not neural
  memory. **Plasticity** separately highlights learned weight changes.
- **Following settling:** reconstructs one captured joint trajectory from its
  actual input, weights and initial state. Long runs expand the early propagation
  and compress the convergence tail within three seconds;
  rapid body updates are coalesced, retaining the largest waiting detuning.
  The timeline and arrow buttons inspect **every iteration** of the captured
  run without skipping. Live playback is sampled, not every body tick.
- **Repair counters:** show neurons changing by more than 1e-8 and directed
  messages changing by more than 1e-8 at the displayed iteration. The full map
  still includes quiet neurons and synapses. These thresholds suppress numerical noise.
- **Joint equilibrium:** reports the current displayed potential-equation error
  and the final error, so a replay shows the actual reduction in disagreement.
  Expand controls for regional endpoint errors.
  A small residual does not guarantee the best possible action.
- **Replay settling:** inspect a captured update at a selected iteration rate.
  Stateful motor circuits start from their retained potentials, not from zero.
- **Release input:** removes drives in an isolated copy and shows recurrent decay.
  It does not change the body or its actual memory.
- **Population waves:** green is signed mean activation; violet is RMS neuron-equation
  mismatch. Every unmasked neuron contributes. Live history retains the last 180
  displayed samples across updates; manual replay plots that captured solve's
  iterations. Distinct input changes and isolated game futures are separate
  computations, not one biological time series. Oscillations appear only when
  the values produce them; a settled circuit is allowed to become quiet.
- **Moving synapse signals:** a packet marks a changed outgoing message on an actual
  synapse. Time-expanded motion illustrates transport, not physical propagation speed.
- **Slow thought:** in the four body demos, hold the actuator command while its
  captured settling sequence is inspected. Movement follows the last iteration.
  The selected replay rate controls presentation time; solver iteration counts
  do not measure human thought or prove that harder semantic tasks take longer.
  Editing the world cancels a pending command. Connect Four already provides its
  own search/preview phase.
- **Live:** returns to current samples; **Pause view** freezes only the viewer.
  Body controls are independent, except **Pause view** also holds a pending slow-thought
  command. Reduced-motion preferences disable autoplay.

The current motor circuits retain graded potentials between control ticks, so
transient neural state really is carried during behavior. Associative weights
and visited-target records provide different forms of retained information.
Neither transient motor state nor the isolated release probe demonstrates a
learned working-memory task.

Connect Four retains **every actual leaf-value evaluation** within the 80,000-node
search budget. Live playback samples this history; the extra future slider selects
any recorded evaluation, and the iteration controls inspect its two-step graded
readout. During this explicitly labeled isolated phase, six value-cortex neurons
update while the other thirteen retain the last shared decision. Only the five
evaluator synapses carry its changing messages; the whole 19-neuron, 39-synapse map
remains visible. Completed depths supply actual joint value/candidate/monitor
repairs. Cache hits and terminal checks do not fabricate neural evaluations.
Telemetry does not change search choices or budgets; the tests compare both paths.
Playback may continue after the worker finishes and is labeled recorded thought.

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
examples plot only quantities its controllers actually compute.

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

See the [per-example advantage contracts](ADVANTAGES.md), including measured
memory runtime and all scheduled Connect Four outcomes. Each page separates
learned records, supplied rules, feedback effects and conventional controls.


Current task-controller evidence is in each embodied example's `evidence.json`,
produced by `tools/nervous_system_benchmark.mjs`. All controller sources are bound
by hashes, all scheduled conditions are retained, and CI reruns the producer.

- **Arm:** square, flower and two separated marks, six conditions each, 6,000
  control steps. Intact and disturbed-feedback cases must ink over 95% of
  reference samples. Silencing joint motors gives zero joint displacement;
  silencing pencil motors while raised gives zero ink, as does removing visual
  input. Coverage reads actual ink at reference samples.
  Separate `eye-arm/brain.test.mjs` regressions inspect ink
  between samples, connected components, erased-ink repair and causal
  missing-mark ablations. Sampling can lose fine detail; unwanted ink cannot be
  erased by the pencil, and sampled coverage does not certify image fidelity.
- **Worm:** intact habitat consumes two patches in 41 moves. Smell, either motor
  population, and sealed-wall controls acquire no food.
- **Forager:** a fixed 4,000-step run produces 26 contacts; motor ablation prevents
  displacement and contact. This is an actuator test, not a learner comparison.
  An additional [adaptation suite](evidence/adaptation_evidence.json) uses six new
  layouts and an unannounced nectar reversal. Aging-observation revisits improve
  recall from 40/48 to 47/48, with lower nectar collection before the reversal.
  The same suite checks drawing coverage on a diagonal, cross and circle.
  Run `node tools/adaptation_benchmark.mjs` to reproduce every row.

The shared [evidence.json](evidence/evidence.json) retains the matched memory
and chemical-circuit benchmarks: seeds 7–9, 128 memory writes per stream with all
seen-key queries, and three held-out chemical-circuit conditions. The memory MLP
gets 1/10/100 updates on the same sample; correlated keys can favor it. The worm
MLP gets 2,048 labels and 1,200 training steps per candidate; it is faster per
query but approximate under interventions. These are bounded comparisons, not
universal speed or energy claims.

[composite_evidence.json](evidence/composite_evidence.json) and
[habitat_evidence.json](worm/habitat_evidence.json) retain the original spatial,
four-neuron arm and habitat reference experiments. They validate those reference
kernels; the expanded browser controllers are measured in the new task receipts.

## Reproduce

Serving requires only Python. Reproduction also uses Node and the pinned Cadence
dependencies:

```bash
python -m pip install -r requirements-reproduce.txt pytest playwright
python tools/verify.py
node tools/nervous_system_benchmark.mjs
node tools/coupled_brain_benchmark.mjs
node connect-four/benchmark.mjs
node benchmarks/memory/history_benchmark.mjs
python -m pytest -q tests
python tools/showcase_pages.py
python tools/build_showcase.py
```

The test suite independently checks motor dynamics against Python Cadence from
retained state and under lesions, replay endpoints, motor/visual ablations,
freehand drawing, image upload, separate URLs and responsive
layout. `build_showcase.py` generates four browser pages plus the gallery from authored
shells. Models and views remain in their example folders; shared browser code is in
`shared/`, cross-example receipts in `evidence/` and producers in `tools/`.

To rerun the older comparison training and reference body trials:

```bash
python tools/fetch_worm.py
python tools/benchmark.py --seeds 3 --steps 1200 --trials 128
python tools/build_composites.py
node tools/habitat_benchmark.mjs
```

## Biological sources and data attribution

The worm chemical edge table and cell descriptions are from
[OpenWorm ConnectomeToolbox](https://github.com/openworm/ConnectomeToolbox), pinned
in `tools/fetch_worm.py`; its [MIT notice](worm/OPENWORM_LICENSE.txt) is included. The neuron
list comes from [c302](https://github.com/openworm/c302) at the pinned revision.
Source URLs and digests are retained in `worm/worm.json`.

Anatomical context: [Cook et al. (2019)](https://www.nature.com/articles/s41586-019-1352-7).
Body-feedback motivation: [Wen et al. (2012)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3508473/).
For the distinction between a circuit response model and a whole behaving fly,
see [Shiu et al. (2024)](https://www.nature.com/articles/s41586-024-07763-9).
These papers motivate the examples; their biological validation does not transfer
to the imposed rules or simplified bodies used here.
