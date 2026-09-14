# Cadence live systems

Five interactive experiments make state, feedback and learning visible. Run
`python serve.py` from the repository root, then open the printed address. The
browser needs no Python packages, account, GPU or remote model service. Open the
HTML through the local server so the browser can load the model files.

## Launch any demo

From the repository root:

```bash
python serve.py mouse
python serve.py eye-arm
python serve.py fly
python serve.py worm
python serve.py memory
```

Each command starts its website on an available local port and opens the browser.
Use `--no-browser` to print the URL instead. `python serve.py` defaults to mouse.

## A two-minute demonstration

1. **Teachable mouse:** select **New task**, demonstrate **Flag**, then choose
   **Perform task**. The mouse recalls its new goal and navigates there. Choose
   **New maze** to transfer the same task to a different layout. Revise its
   destination, then perform an earlier task to check retention.
2. **Eye & arm:** watch a flower outline emerge, then **Disturb a joint**. Visual
   error and motor correction settle together and repair the next movement.
   Repeat with **Feedback off**. Upload an image to supply your own outline.
3. **Embodied forager:** let the agents encounter flowers, then **Change nectar**.
   New encounters revise their preferences. Drag flowers to change the body loop.
4. **C. elegans habitat:** choose **Food** and click to place a patch. Choose
   **Wall** and drag a barrier; use **Eraser** to open a passage. Turn **Smell off**
   to interrupt cue-driven movement. Switch to **Circuit** to stimulate sensory
   groups, remove neurons and compare against the trained MLP.
5. **Changing memory:** teach a key, replace its value, and query other keys.
   Increase key similarity to expose interference rather than hiding it.

Task demonstrations persist in this browser's local storage. The mouse's first
three cue/destination pairs are supplied demonstrations; the fourth starts
untaught. A named cue is an explicit key, not natural-language understanding.
The worm habitat retains edits while you switch views or tabs; a page reload
starts it fresh. The arm, forager and memory demos reset when reopened. No experience is
uploaded. The five websites above are the current public examples.

## Ready to run and watch learning

All five demos ship with the assets needed to run immediately in the browser.
They do not all use trained checkpoints: some learn from your observations,
while others execute a supplied mechanism. No installation of Cadence or model
training is needed for the browser experience.

| Demo | Starting state | What you can observe |
|---|---|---|
| Teachable mouse | Three supplied task demonstrations and a working navigation rule; additional lessons reload from this browser | **Teach task** performs a residual memory write. **Perform task** uses the recalled goal; **New maze** tests transfer. Revise a cue and check earlier tasks. |
| Eye & arm | Working visual/motor controller, supplied arm geometry and built-in outlines | **Disturb a joint** and toggle **Feedback** to watch correction. This changes activity and movement, not learned weights. Image upload supplies new edge targets. |
| Fly-inspired forager | Working body and steering; fresh nectar memory and a randomly initialized MLP | Encounters reveal nectar and update each learner live. **Change nectar** tests revision; **Pause** freezes the scene. |
| C. elegans habitat and circuit | Public chemical topology, supplied body/sensory rules and trained MLP circuit comparator | Paint food and walls, watch contact consumption, disable smell, then inspect circuit stimulation and lesions. No weight training occurs in this view. |
| Changing memory | Blank fast memory and a randomly initialized MLP | **Teach once** exposes individual updates; **Run stream** automates observations and updates accuracy. **Clear** restarts. |

Open `python serve.py worm` to go directly to C. elegans. Its graph contains
297 participating annotated neurons and 3,604 chemical edges. Its habitat wraps
the circuit in an engineered body; it is not a validated whole-animal model. The forager is fly-inspired,
with supplied planar sensors and movement.

## Why Cadence fits each task

| Demo | Useful Cadence property | Advantage and relevant alternatives |
|---|---|---|
| Mouse | Task records remain separate from the current spatial settlement | One residual write changes a destination without erasing other distinct cues; the map field adapts without fitting a new policy. A frozen route fails changed routes, while BFS and dictionary lookup also handle their respective parts. This does not establish a trained-MLP advantage. |
| Eye & arm | Visual-error and motor-correction owners settle jointly using current pose readback | Continuous error correction repairs disturbed movement. The measured comparison enables or disables readback in the same controller. Classical feedback and recurrent neural controllers can also do this; the experiment does not compare trained MLPs. |
| Fly-inspired forager | An encounter can revise a bounded record immediately inside the body loop | Residual writes make new nectar observations available for the next choice. The MLP also learns online with a selectable update budget. Live agents see different streams, so their nectar totals are illustrative, not a matched performance claim. |
| C. elegans circuit | Known local interactions are reused under changed drives or lesions | Settlement remains accurate against the reference without surrogate refitting. The bundled MLP has larger intervention errors but faster per-query inference. Conventional graph recurrence also reuses the supplied mechanism. |
| Changing memory | A residual write directly repairs an incorrect stored value | Exact distinct-key recall with 32 mutable entries exceeds the tested MLP update budgets on identical streams. Dictionary lookup is also exact. Strongly overlapping keys can favor the MLP; this is explicit record storage, not a general learning benchmark. |

Every browser tab includes its own starting-state, observation and advantage
explanation. Expand **Inside the feedback loop** for the component breakdown,
then inspect the measured comparison and its methods below it.

## What each brain contains

| Demonstration | Regions and feedback | Learned, supplied, and measured |
|---|---|---|
| Mouse | Task memory → visual/map ports → recurrent place field → motor/body → position readback | Task associations are learned by residual writes. The complete maze, local movement rules and spatial-field construction are supplied. Tests cover new tasks, revision, retention, new layouts and changed goals/corridors. |
| Eye & arm | Visual-error owners ↔ motor-correction owners → joints → visual/proprioceptive readback | Four owners settle jointly through the arm Jacobian. Image edge extraction and arm geometry are supplied. A disturbed-body comparison isolates feedback. This controller does not learn anatomy or artistic style. |
| Fly-inspired forager | Visual flower ports → learned nectar memory → target selection → steering/body → contact reward | Nectar is revealed only on contact. Residual memory and online MLP collect their own live experiences. The planar body, sensor encoding, steering and exploration schedule are supplied. |
| Worm | Local food cue → sensory drive → recurrent chemical graph → motor-gated body step → contact | Public anatomical topology; imposed weights, tanh transfer, gradient heading, diffusion and body rules. The separate circuit view compares a trained MLP and 32-step recurrence. |
| Memory | Key → readback → observed value → residual write | Eight explicit keys and four values; 32 mutable fast-memory entries. Same-stream MLP controls receive 1, 10 or 100 SGD updates per observation. A dictionary also solves exact-key storage. |

These are bounded observer-like software patches with local state, declared
ports, readback, records and feedback. Composing them creates inspectable systems:
we can distinguish a remembered goal from a spatial state, motor correction, and
body observation. The two-way visual/motor circuit is a joint settlement; the
mouse and forager connect mechanisms through repeated environmental feedback.

## Read the brain view

Every website has the same MRI-inspired circuit viewer. The two-hemisphere
silhouette is a visual metaphor; the nodes and connections come from the actual
software circuit. No additional neurons or oscillations are drawn to imply
capacity the model does not have.

| Demo | Owners shown | State and plasticity |
|---|---|---|
| Mouse | 247 spatial slots, including masked walls, plus 12 task-memory ports | Spatial activity and 32 plastic associative entries; the current task cue and recalled destination are visible separately |
| Eye & arm | 2 visual-error and 2 motor-correction owners | Reciprocal repair; Jacobian-dependent coupling changes are geometry, not learned plasticity |
| C. elegans | 297 participating annotated neurons | Actual chemical edges, sensory drive and motor activity; fixed supplied weights |
| Forager | 8 key ports and 4 value ports | The Cadence agent's actual 32 memory entries, updated on nectar contact |
| Changing memory | 8 key ports and 4 value ports | Current key/value readout and observed residual writes |

- **Activity** shows live model values. Brightness is normalized within each
  region; input-release probes keep the captured scales fixed so fading remains
  visible. The numerical scale is printed and hovering reveals owner values.
  The trace plots peak recurrent activity, excluding the mouse’s separate
  associative ports; memory-only views plot peak port activity.
- **Repair** shows changes between live observations, or between iterations
  during replay. Amber increases; blue decreases. A model may converge smoothly,
  ring, or remain quiet. The renderer adds no oscillation.
- **Plasticity** shows learned weight strength and highlights recent changes
  for 1.5 seconds. The counter counts observed weight-change events, not training
  steps. Fixed circuit and geometry weights are never counted as learning.
- **Replay repair** recomputes the captured settlement from its original zero
  state with the same weights, drive, mask, integration step and iteration count.
  It displays those numerical iterations slowly. Tests compare the final replay
  against the actual mouse, arm and worm states.
- **Release input** starts an isolated copy from the captured recurrent state
  and removes its drive. The resulting decay is a probe of transient
  reverberation. It does not feed back into the body, and these demos do not use
  that diagnostic copy as behavioral short-term memory. A stable attracting
  equilibrium does not provide durable memory by itself.
- **Live** returns to the current model; **Pause view** freezes only the circuit
  display. Body controls remain independent. Memory-only circuits disable
  recurrent replay, because retained associative weights are a different kind
  of memory from fading neural activity.

The existing capacity is exposed faithfully. Drawing needs four coupled error
owners in this supplied controller; adding untrained nodes would not make that
controller more capable. GPU training is not required for these diagnostics.

## Food and walls in the worm habitat

Real C. elegans feeds on bacteria using its pharynx
([WormBook: feeding](https://www.ncbi.nlm.nih.gov/books/NBK116080/)). The browser
implements contact consumption of a food patch, not pharyngeal pumping or
metabolism. There is no learned locomotion policy.

- Food clamps a leaky diffusion field on a 31 × 21 habitat. Walls block movement
  and flux. The field is recomputed when food or walls change.
- The sensory adapter reads only adjacent open-cell food-cue values. Their
  maximum drives the AWA/AWC ports in the public circuit; the same recurrent
  kernel as the circuit view settles those inputs.
- A supplied heading rule follows the neighboring gradient. Mean motor-owner
  activity gates each body step. Removing all motor owners or disabling smell
  stops movement. The controller receives no path or remote food coordinates.
- Contact removes a patch and updates the field. Sealed-off food has no usable
  cue in the worm's region. Open a passage to restore access. This diffusion
  environment makes navigation tractable; it is not evidence of learned planning.
- Mouse and touch dragging paint continuous strokes. Pause while editing if you
  want time to build a maze. Walls cannot be painted through the displayed body.
  For keyboard editing, focus the habitat, move with arrow keys, and press Space
  or Enter to apply the selected tool.

[Habitat evidence](habitat_evidence.json) is reproduced by
`node showcase/habitat_benchmark.mjs`. It covers food consumption, barriers,
opening a passage, cue and motor ablations, additional food after completion,
and protected body cells. These are behavior checks of supplied rules. The MLP
comparison underneath is the separate circuit-response experiment.

## Results and comparison contract

The page computes its tables from [evidence.json](evidence.json) and
[composite_evidence.json](composite_evidence.json). They retain all scheduled
conditions, source hashes, parameters and model/control boundaries. Local browser
latency is labeled separately from recorded experiment timings.

- **Memory:** 128 writes per stream, 8 keys, 4 values, seeds 7–9. Query every seen
  key after every write. Compare distinct keys and pairwise cosines 0.5 and 0.9.
  Cadence performs one residual write; the 420-parameter MLP uses 1/10/100 SGD
  updates on the identical sample, at a declared fixed learning rate. These are
  bounded online-learning budgets, not exhaustive optimization of MLP training.
  Distinct-key recall is exact for both Cadence and dictionary lookup. Similar
  keys interfere; at cosine 0.9 the one-step MLP outperforms this fast memory.
- **Worm:** seeds 7–9; 2,048 training and 256 validation samples; 64/128 hidden-owner
  MLP candidates, 1,200 Adam steps each, selected only on validation error. Test
  128 queries per condition: intact, new mixtures and up to 64 lesions. Both
  methods know the fixed graph; the MLP receives drive, mask and an exact first
  propagation. The supplied local dynamical prior is explicit. Cadence and the
  independent converged sparse solver agree near numerical precision. A tied
  recurrence also solves the task and is itself an unrolled feedforward graph.
  The MLP is approximate but faster per query; costs include separate training
  and label-generation timings. No energy-efficiency or universal speedup claim.
- **Mouse:** 12 seeds × 3 conditions. Navigation uses a complete supplied maze,
  not hidden-maze exploration. A frozen route and BFS replanner are controls.
  Edited-corridor cases choose a route cell whose removal preserves reachability;
  the receipt records the cell, including `-1` if no such edit is possible.
  Collision checks apply to every simulated body step. Task memory is additionally
  tested for first teaching, replacement, retention and transfer to three mazes.
- **Arm:** flower, leaf and spiral; joint disturbance at step 60/120/240; 4,000
  control steps. Compare the same controller with actual pose feedback enabled
  or disabled. Coverage counts target points within 0.022 normalized workspace
  units of deposited ink. It is not a perceptual/artistic quality score or an
  MLP comparison. Uploaded images are reduced to up to 320 edge targets and may
  be harder or take longer than these built-in outlines.

The full-data worm fixture contains 297 participating annotated neurons and
3,604 chemical edges after canonicalizing zero-padded names, intersecting with
the pinned c302 neuron list, removing self-edges and excluding non-neural targets.
It is not the complete 302-neuron nervous system. Gap junctions are excluded.
The fly and mouse bodies are engineered visual simulations, not validated
whole-animal reconstructions. This public implementation contains no private
whole-fly fixtures, research models, or training assets.

## Reproduce

Use Python 3.11+, Node 20+ and the dependencies documented in the repository root.
The browser kernels are small ES modules; Node is needed only for reproduction
and tests, not to serve the demos.

```bash
python -m pip install -r requirements-reproduce.txt
python -m pip install torch pytest
python showcase/fetch_worm.py
python showcase/benchmark.py --seeds 3 --steps 1200 --trials 128
python showcase/build_composites.py
python showcase/verify.py
node showcase/habitat_benchmark.mjs
python -m pytest -q tests/test_showcase.py
python tools/showcase_pages.py
```

`fetch_worm.py` downloads pinned public files and checks their SHA-256 digests.
`benchmark.py` writes the measured comparisons and exports the model from the
first scheduled seed, not the best test seed. `build_composites.py` executes the
body trials and checks the spatial and motor kernels against Python Cadence.
`verify.py` checks source binding, body-trial arithmetic and benchmark contracts;
it does not rerun the full training process. Browser tests check live interactions,
persistence, upload, mobile layout and numerical parity. Rebuilding the two hub
copies uses `python tools/build_showcase.py`.

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
