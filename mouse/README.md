# Teachable mouse

From the repository root: `python serve.py mouse`. The launcher opens
`/mouse/`; use `/` to browse the other examples.

Teach a fourth task, change a corridor, move the goal, or disable Motor neurons. Lessons persist in this browser.

## Control path

Task cues recall destinations from associative memory. A supplied visual map defines a recurrent spatial field. Position errors drive four directional motor units, which move the body.

265 allocated owners, 132 initially unmasked, 285 seams and 32 learned entries in seed 13. Map and readout rules are supplied.

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

See the [advantage contract](../ADVANTAGES.md) for navigation and motor-ablation
results, with BFS and dictionary lookup as strong conventional controls.
