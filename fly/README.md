# Embodied forager

From the repository root: `python serve.py fly`. The launcher opens
`/fly/`; use `/` to browse the other examples.

Move flowers and change nectar. Each contact updates transient and persistent synaptic weights. Repetition consolidates nectar expectations; the absolute difference between observed and predicted nectar supplies salience. Live agents collect different experiences; use the matched-stream benchmark below for comparisons.

## Control path

Flower cues query learned nectar memory. A supplied target-selection rule chooses a flower; bearing and approach signals drive turn and propulsion motor units. Nectar becomes available only on contact.

The selection rule lets certainty fade with time since the last contact. Neglected
flowers become worth revisiting, even if they used to be poor. The rule uses eight
contact timestamps alongside the learned nectar expectations; it cannot inspect
the current nectar. Both live agents use this same rule. This is supplied
exploration, not a learned curiosity circuit or an extra population of neurons.

18 neurons, up to 44 synapses and 32 persistent memory weights plus 32 transient residuals. A simplified planar body, not a reconstructed fly connectome.

The circuit at the top shows actual state, activity changes and retained information.
Supplied readout and body rules are documented rather than shown as extra neurons.
See the [shared viewer guide](../METHODS.md#read-the-brain-view).

## A shared equilibrium

The labeled regions participate in one connected solve for the current input.
Local readback and settling change the same joint state; the action readout uses
that state. The top circuit panel shows the global equation error and can replay
the actual cross-region cascade. Observed lessons update synaptic strengths within the ongoing interaction loop. Repetition strengthens their persistent component. Neuronal settling and plasticity have different timescales.
[Task connectomes, boundaries and tests](../COUPLED_BRAINS.md) explain the
connections. Self-consistency is not a guarantee of the globally best behavior.

## Reproduce

Run `node tools/nervous_system_benchmark.mjs` from the repository root.
[evidence.json](evidence.json) contains all scheduled task trials and source hashes.

Run `node tools/adaptation_benchmark.mjs` for six additional layouts with an
unannounced nectar reversal halfway through each 8,000-tick run. The exploration
parameters were selected on three different development layouts. The complete
[evaluation receipt](../evidence/adaptation_evidence.json) retains both controllers:

| Across six evaluation layouts | Previous selection | Aging-observation revisits |
|---|---:|---:|
| Correct nectar memories after reversal | 40/48 | 47/48 |
| Nectar before reversal | 508 | 445 |
| Nectar after reversal | 373 | 382 |

Revisiting improves adaptation here at the cost of exploiting known food sooner.
It does not improve nectar on every layout. These rows compare exploration rules
with identical Cadence memory and motor circuits, not Cadence against an MLP.

See the [advantage contract](../ADVANTAGES.md) for actuator results and why live
nectar totals are not a matched learning comparison. The [memory benchmark](../benchmarks/memory/README.md) provides
that controlled observation stream.
