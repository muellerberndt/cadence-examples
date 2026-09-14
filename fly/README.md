# Embodied forager

From the repository root: `python serve.py fly`. The launcher opens
`/fly/`; use `/` to browse the other examples.

Move flowers and change nectar. Each contact updates memory. Live agents collect different experiences; use the memory page for matched comparisons.

## Control path

Flower cues query learned nectar memory. A supplied target-selection rule chooses a flower; bearing and approach signals drive turn and propulsion motor units. Nectar becomes available only on contact.

18 owners, up to 44 seams and 32 learned memory entries. A simplified planar body, not a reconstructed fly connectome.

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

See the [advantage contract](../ADVANTAGES.md) for actuator results and why live
nectar totals are not a matched learning comparison. The memory example provides
that controlled observation stream.
