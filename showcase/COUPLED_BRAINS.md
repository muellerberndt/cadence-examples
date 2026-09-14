# Different regions, one equilibrium

Every task brain contains bounded owners with local state, readback and repair.
Regions label their function; their seams make them part of one joint state.
For each fixed observation, all connected regions settle together before the
body acts or the strategy controller requests another search depth.

The public Cadence library provides
[`couple(regions, bridges)`](https://github.com/muellerberndt/cadence/blob/main/docs/brains.md#different-regions-one-equilibrium).
The browser's small `settleTogether` implementation assembles the same sparse
owner graph and applies Cadence's existing graded rule. It does not concatenate
separately solved circuits. The tests independently replay all six controllers
with Python Cadence.

| Brain | Labeled regions sharing the current phase | Boundary supplied by the application |
|---|---|---|
| Eye & arm | Retina → target/proprioception ↔ visual error ↔ joint coordination ↔ opposing motor units | Pixels, target attention, current pose and Jacobian geometry; motor output moves joints and raises/lowers the pencil. The attended retinal sample enters the target ports through explicit seams. |
| Mouse | Task cue → retained association → spatial field → position error ↔ directional motors | Occupancy map and goal-port routing. Neighbor selection reads the field; motor output determines motion. A taught destination drives the selected goal through its recalled association. |
| Worm habitat | Chemical sensory input ↔ interneurons / chemical motor output → directional odor readback ↔ body motors | Local odor diffusion and directional gradients. Chemical output arrives through explicit seams at directional owners; no remote food coordinates or path enter the body controller. |
| Forager | Flower cue → nectar memory → approach error ↔ turn/propulsion motors | Visible bearing, target selection and body bounds. Recalled nectar also contributes to approach drive. Contact writes memory for the next phase. |
| Changing memory | Cue owners → associative seams → recall owners | Observed cue/value lessons. A local residual write changes the stored weights; fixed weights then publish recall through graded owners. No motor system is needed. |
| Connect Four | Threat features → value ↔ candidates ↔ own-activity readback → self-monitor | Exact game rules, feature extraction and bounded search in isolated board copies. Candidate activity selects a legal action; monitor output decides whether to deepen search. |

Arrows summarize connectivity, not separate evaluation stages. Some seams are
one-way; feedback exists where the task uses it. A directed readout can share a
fixed point without requiring an invented reverse connection.

## What settles

With the displayed gains, each unmasked owner uses:

```text
activation = tanh(potential)
error = incoming weighted activity + fixed drive - potential
potential += dt * error
```

An iteration reads the previous **whole-brain** state. The equation residual is
the largest absolute error over every owner; masked potentials must be zero.
The viewer shows this global error, the requested tolerance and the number of
nonzero links crossing region boundaries. Expand the circuit controls for each
region's error. Hover the equilibrium line for the same breakdown.

When the body moves, an observed lesson changes a weight, or a search finishes
another depth, the boundary changes and the next phase begins. Body positions,
attention bookkeeping, learned records and imagined worlds are different state
objects with different roles. Mutually exclusive futures must stay isolated;
the decision brain combines their *scores*, not incompatible board states.

The strategy circuit's shared offsets preserve the ordering of candidate scores.
Its feedback controls the further-work gate; it does not invent a better search
algorithm or prove optimal play. Likewise, a motor equilibrium needs useful
geometry and a working body readout to accomplish its task.

## Self-consistency and task quality

A small residual certifies the fixed-point equations at this state. It does not
prove stability, uniqueness, biological realism or a globally optimal action.
Strong feedback may prevent convergence. The finite iteration budget is explicit;
a budget-limited result is labeled as such rather than called settled.

Retained potentials warm-start the next phase. The release probe removes drives
in an isolated copy so fading activity can be inspected. Fully converging a
contractive circuit under identical input erases initial-state differences;
durable lessons here reside in associative weights, not unexplained reverberation.
Memory readout uses `tanh` of the linear association; inverse decoding recovers
the displayed value and its argmax. Exact-record and timing benchmarks isolate
the underlying linear record operation, excluding the extra display/control solve.

## Tests you can run

```bash
node tools/coupled_brain_benchmark.mjs
python -m pytest -q tests/test_coupled_brains.py
node tools/nervous_system_benchmark.mjs
node connect-four/benchmark.mjs
```

The [joint receipt](coupled_evidence.json) binds all six sources and records
whole-circuit residual bounds, replay accuracy and affected regions when links
between functions are cut. Python checks the joint trajectories independently.
The body tests measure navigation, contact, drawing and actuator ablations;
strategy tests check legal play, budget control and scheduled opponents.
A pleasing animation alone is not an equilibrium or performance test.

Older [comparison receipts](../ADVANTAGES.md) cover their specified reference
kernels and observation streams. Their timing figures are not measurements of
the expanded coupled controllers or the browser visualization.
