# Eye, brain and drawing arm

From the repository root: `python serve.py eye-arm`. The launcher opens
`/eye-arm/`; use `/` to browse the other examples.

Press **Clear** directly below the drawing area to empty both your picture and
the arm’s copy. Draw a mark, then release to let the arm copy it. Releasing keeps
your strokes; **Clear** starts again, including while paused or inspecting a thought.
Erase a patch of the copy to watch the arm repair it. Use Joint motors, Pencil motors and Eye on to interrupt the control path. Disturb a joint to test pose feedback.

## Control path

Draw on the left pad or upload a dark line image. The eye samples 48 × 48 pixels. Every dark sample has a reference neuron, an actual-ink readback neuron and a missing-mark neuron. Their local comparison drives a supplied attention rule through connected dark pixels. The pencil stays down along these routes and lifts before moving to a separate stroke. Target and proprioceptive input drive three position/height error units, two joint-coordination units and six motor units. Shoulder, elbow and pencil height change only through motor output.

For N dark samples, the circuit has 3N + 17 neurons and up to 2N + 28 synapses. Geometry, raster connectivity, target attention and weights are supplied. Retained neural potentials are transient state; this is not trained image recognition.

The displayed 192 × 192 ink raster is also the physical readback source. Only pencil contact deposits ink. Completion requires ink at every reference sample, and erasing a patch reactivates missing-mark neurons without resetting the arm. The metric is sampled mark coverage, not exact image similarity: sampling can lose fine detail, and this pencil cannot erase unwanted marks.

The circuit at the top shows actual state, activity changes and retained information.
Supplied readout and body rules are documented rather than shown as extra neurons.
See the [shared viewer guide](../METHODS.md#read-the-brain-view).

## A shared equilibrium

The labeled regions participate in one connected solve for the current input.
Local readback and settling change the same joint state; the action readout uses
that state. The top circuit panel shows the global equation error and can replay
the actual cross-region cascade. Lessons change records between phases.
[Task connectomes, boundaries and tests](../COUPLED_BRAINS.md) explain the
connections. Self-consistency is not a guarantee of the globally best behavior.

## Reproduce

Run `node --test eye-arm/brain.test.mjs` for continuous-stroke, separate-mark, erased-ink repair and causal-ablation regressions. The tests inspect the actual ink bitmap, including points between retinal samples.

Run `node tools/nervous_system_benchmark.mjs` from the repository root.
[evidence.json](evidence.json) contains all scheduled task trials and source hashes.

See the [advantage contract](../ADVANTAGES.md) for the disturbance and motor
controls, measured target coverage, and the conventional feedback comparison.
