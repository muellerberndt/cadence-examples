# Changing memory

From the repository root: `python serve.py memory`. The launcher opens
`/memory/`; use `/` to browse the other examples.

Teach a key/value, replace the value, query earlier keys and compare the online MLP on the same observation stream.

## Control path

This page isolates the associative-memory mechanism used by the embodied examples. It has no body and no motor neurons.

12 ports, 32 memory entries. One residual write replaces a distinct-key record. Similar keys can interfere.

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

The matched learning comparison is in [shared evidence](../evidence/evidence.json).
Run `python tools/verify.py` to check the pinned producer and arithmetic.

For measured local processing time, run `node memory/benchmark.mjs`. The bundled
[runtime receipt](evidence.json) reports 100% accuracy on a distinct-key overwrite
stream at 0.447 ms median per 128-write/996-query stream, versus 1.360, 3.179 and
22.635 ms for the MLP's 1/10/100-update settings. That is about 3×/7×/50× lower
time on the recorded Apple M4/Node runtime. It is not a general speed or energy
claim; see the [full comparison contract](../ADVANTAGES.md#measured-processing-time).

## Same cue, a newly taught meaning

Teach a distinct key one value, then teach that same key a different value. The
current query is identical; the retained lesson is what allows a new answer.
`node memory/history_benchmark.mjs` tests this in a balanced 128-query schedule:
Cadence and conventional last-value lookup score 100%; a frozen query-only MLP
scores 25%. **Any fixed deterministic function of the same current cues is
bounded by 25%**, even if chosen optimally in hindsight.

The [receipt](history_evidence.json) retains all queries and predictions.
This boundary assumes no retained activations, weight updates, history/context
tokens, clock or external store. A transformer with lesson history can use that
information too. The [derivation and controls](../ADVANTAGES.md#a-precise-limit-of-frozen-inference-without-history)
explain exactly what is impossible under the restriction.

The runtime receipt measures the underlying linear residual-memory operation.
It excludes the newer graded readout, the coupled body controllers and the web
visualization. Their additional computation is not covered by the recorded speed
ratios. See [the joint-brain design](../COUPLED_BRAINS.md).
