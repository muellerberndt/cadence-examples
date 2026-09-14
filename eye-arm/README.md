# Eye, brain and drawing arm

From the repository root: `python serve.py eye-arm`. The launcher opens
`/eye-arm/`; use `/` to browse the other examples.

Clear pad → draw a mark → watch the copy. Use Joint motors, Pencil motors and Eye on to interrupt the control path. Disturb a joint to test pose feedback.

## Control path

Draw on the left pad or upload a dark line image. The eye samples 24 × 24 pixels. A visible target and proprioceptive input drive three error units, two joint-coordination units and six motor units. Shoulder, elbow and pencil height change only through motor output.

593 owners and up to 28 seams. Geometry, target attention and weights are supplied. Retained neural potentials are transient state; this is not trained image recognition.

The circuit at the top shows actual state, repairs and retained information.
Supplied readout and body rules are documented rather than shown as extra neurons.
See the [shared viewer guide](../showcase/README.md#read-the-brain-view).

## A shared equilibrium

The labeled regions participate in one connected solve for the current input.
Local readback and repair change the same joint state; the action readout uses
that state. The top circuit panel shows the global equation error and can replay
the actual cross-region cascade. Lessons change records between phases.
[Task wiring, boundaries and tests](../showcase/COUPLED_BRAINS.md) explain the
connections. Self-consistency is not a guarantee of the globally best behavior.

## Reproduce

Run `node tools/nervous_system_benchmark.mjs` from the repository root.
[evidence.json](evidence.json) contains all scheduled task trials and source hashes.

See the [advantage contract](../ADVANTAGES.md) for the disturbance and motor
controls, measured target coverage, and the conventional feedback comparison.
