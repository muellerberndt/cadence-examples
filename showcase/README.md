# Cadence live systems

Five interactive experiments make state, feedback and learning visible. Run
`python serve.py` from the repository root, then open the printed address. The
browser needs no Python packages, account, GPU or remote model service. Open the
HTML through the local server so the browser can load the model files.

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
4. **Worm circuit:** stimulate different sensory groups, remove neurons, and
   compare the new settlement with the trained MLP approximation.
5. **Changing memory:** teach a key, replace its value, and query other keys.
   Increase key similarity to expose interference rather than hiding it.

Task demonstrations persist in this browser's local storage. The mouse's first
three cue/destination pairs are supplied demonstrations; the fourth starts
untaught. A named cue is an explicit key, not natural-language understanding.
The other demos reset their transient state when reopened. No experience is
uploaded. Old games and tutorials remain at [the tutorial hub](../tutorials.html).

## What each brain contains

| Demonstration | Regions and feedback | Learned, supplied, and measured |
|---|---|---|
| Mouse | Task memory → visual/map ports → recurrent place field → motor/body → position readback | Task associations are learned by residual writes. The complete maze, local movement rules and spatial-field construction are supplied. Tests cover new tasks, revision, retention, new layouts and changed goals/corridors. |
| Eye & arm | Visual-error owners ↔ motor-correction owners → joints → visual/proprioceptive readback | Four owners settle jointly through the arm Jacobian. Image edge extraction and arm geometry are supplied. A disturbed-body comparison isolates feedback. This controller does not learn anatomy or artistic style. |
| Fly-inspired forager | Visual flower ports → learned nectar memory → target selection → steering/body → contact reward | Nectar is revealed only on contact. Residual memory and online MLP collect their own live experiences. The planar body, sensor encoding, steering and exploration schedule are supplied. |
| Worm | Sensory drive → recurrent chemical graph → whole-state/motor readout | Public anatomical topology; imposed positive normalized weights and tanh transfer. No training between interventions. A trained MLP surrogate and 32-step recurrence provide controls. |
| Memory | Key → readback → observed value → residual write | Eight explicit keys and four values; 32 mutable fast-memory entries. Same-stream MLP controls receive 1, 10 or 100 SGD updates per observation. A dictionary also solves exact-key storage. |

These are bounded observer-like software patches with local state, declared
ports, readback, records and feedback. Composing them creates inspectable systems:
we can distinguish a remembered goal from a spatial state, motor correction, and
body observation. The two-way visual/motor circuit is a joint settlement; the
mouse and forager connect mechanisms through repeated environmental feedback.

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
