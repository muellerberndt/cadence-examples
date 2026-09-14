# Worm habitat

From the repository root: `python serve.py worm`. The launcher opens
`/worm/`; use `/` to browse the other examples.

Paint food or walls, erase a passage, disable smell, or inspect and lesion the chemical circuit. Keyboard: focus the habitat, arrows move the cursor, Space paints.

## Control path

Adjacent odor drives the 297-cell public chemical graph. Four directional readback units and eight antagonistic motor units convert local differences into body steps, enabled by chemical motor activity.

309 owners and up to 4,108 seams in the habitat. The circuit-only probe has 297/3,604. This is an engineered body controller, not full worm physiology.

The circuit at the top shows actual state, repairs and retained information.
Supplied readout and body rules are documented rather than shown as extra neurons.
See the [shared viewer guide](../METHODS.md#read-the-brain-view).

## A shared equilibrium

The labeled regions participate in one connected solve for the current input.
Local readback and repair change the same joint state; the action readout uses
that state. The top circuit panel shows the global equation error and can replay
the actual cross-region cascade. Lessons change records between phases.
[Task wiring, boundaries and tests](../COUPLED_BRAINS.md) explain the
connections. Self-consistency is not a guarantee of the globally best behavior.

## Reproduce

Run `node tools/nervous_system_benchmark.mjs` from the repository root.
[evidence.json](evidence.json) contains all scheduled task trials and source hashes.

See the [advantage contract](../ADVANTAGES.md) for food/lesion controls and the
separate chemical-circuit comparison against an approximate MLP surrogate.
