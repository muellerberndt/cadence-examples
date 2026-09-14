# Changing memory

From the repository root: `python serve.py memory`. The launcher opens
`/memory/`; use `/` to browse the other examples.

Teach a key/value, replace the value, query earlier keys and compare the online MLP on the same observation stream.

## Control path

This page isolates the associative-memory mechanism used by the embodied examples. It has no body and no motor neurons.

12 ports, 32 memory entries. One residual write replaces a distinct-key record. Similar keys can interfere.

The circuit at the top shows actual state, repairs and retained information.
Supplied readout and body rules are documented rather than shown as extra neurons.
See the [shared viewer guide](../showcase/README.md#read-the-brain-view).

## Reproduce

The matched learning comparison is in [shared evidence](../showcase/evidence.json).
Run `python showcase/verify.py` to check the pinned producer and arithmetic.

For measured local processing time, run `node memory/benchmark.mjs`. The bundled
[runtime receipt](evidence.json) reports 100% accuracy on a distinct-key overwrite
stream at 0.435 ms median per 128-write/996-query stream, versus 1.338, 3.170 and
22.430 ms for the MLP's 1/10/100-update settings. That is about 3×/7×/52× lower
time on the recorded Apple M4/Node runtime. It is not a general speed or energy
claim; see the [full comparison contract](../ADVANTAGES.md#measured-processing-time).
