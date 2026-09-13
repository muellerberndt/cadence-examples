# 05 · Updating memory

An association changes. Should yesterday's repetitions outvote today's correction?
Additive Hebbian memory keeps adding evidence; residual memory first reads its own
answer and writes only what it got wrong.

```python
import numpy as np
import cadence as cd

memory = cd.FastSeams(np.arange(2), np.arange(2, 4), rule="delta")
key = np.array([[1., 0.]])
memory.observe(key, np.array([[1., 0.]]))
memory.observe(key, np.array([[0., 1.]]))  # the association changed
print(memory.recall(key))                # [[0., 1.]]
```

Run `python demo.py` for a twenty-line example. The residual update is the standard
normalised delta/LMS rule. Cadence gives it an explicit place alongside slow learned
seams: bounded memory state, key/value ports, readback, a local repair, and a receipt.
This is the observer-like self-reading structure. A memory write does not train the
slow network and does not require equilibrium propagation.

## The small comparison

```bash
# From cadence-examples:
python -m pip install -r requirements.txt torch
cd 05_memory
python train.py --steps 800 --seeds 3 --trials 100 --output local_receipt.json
python train.py --verify local_receipt.json
```

This writes a fresh result using the current supported library. To verify the saved
`receipt.json`, use the separate environment installed from
[`requirements-reproduce.txt`](../requirements-reproduce.txt), as shown in the
[reproduction instructions](../README.md#reproduction), then run
`python tools/verify_receipts.py 05_memory` from the repository root. The saved
measurements bind that original dependency, not later library fixes. The
[original receipt and producer](history/2026-09-13-before-source-clarification/)
remain available with their historical source revision.

Eight keys, eight possible values. First write every key once, then write random
revisions. Query every key's latest value after 8, 32 or 128 writes. All five methods
receive **the same key/value vectors, explicit pair boundaries, order and test episodes**:

- Residual `FastSeams`: a normalised read, then `W += key ⊗ (value − readback)`.
- Additive `FastSeams`: `W += key ⊗ value` with no correction.
- Dictionary: exact vector keys and latest-value replacement.
- Exact attention: retain every pair, match the query key and read the latest match.
- Trained transformer: two encoder layers, width 32, four heads, 17,960 parameters;
  explicit key/value tokens and query tokens, with sinusoidal positions.

The transformer learns on 8-, 16- and 32-write streams. The run selects between its
half-budget and full-budget checkpoints using separate validation streams. Test streams
are separate, shared across all arms, and read after selection. The 128-write condition
checks length extrapolation; it is not a matched training-distribution comparison.

The experiment varies key overlap too. At overlap 0, keys are orthogonal; at overlap
0.5, every pair has cosine similarity 0.5. A residual write fits its current key but can
disturb other, correlated keys. The receipt therefore records both **all-key accuracy**
and the accuracy of the **most recently written key**. The eight-write condition shows
first-write performance; longer streams show revisions. None of the streams replay a
complete clean target table after the revisions.

Two extra controls write every key and every value exactly once, at key cosines 0.5
and 0.9. The 0.9 condition also changes the key distribution seen by the transformer.
Balanced values remove the popularity advantage of common values in additive memory.
This first-pass case can favor additive memory over residual writes: correcting each
new association can disturb an older one before it is ever revisited.

## Reading the result

Mean all-key accuracy across all recorded seeds:

<!-- memory-results -->
| writes | key cosine | residual | additive | transformer | exact lookup |
|---|---|---|---|---|---|
| 8 | 0 | 1.000 | 1.000 | 1.000 | 1.000 |
| 32 | 0 | 1.000 | 0.444 | 0.982 | 1.000 |
| 128 | 0 | 1.000 | 0.260 | 0.145 | 1.000 |
| 8 | 0.5 | 0.765 | 0.640 | 1.000 | 1.000 |
| 32 | 0.5 | 0.712 | 0.329 | 0.978 | 1.000 |
| 128 | 0.5 | 0.706 | 0.209 | 0.148 | 1.000 |
| 8, balanced values | 0.5 | 1.000 | 1.000 | 1.000 | 1.000 |
| 8, balanced values | 0.9 | 0.125 | 1.000 | 0.983 | 1.000 |
<!-- /memory-results -->

The 128-write rows are length extrapolation; the transformer trains through 32 writes.
The full receipt also records each seed and the most recently written key's accuracy.

With no offline training, residual memory reaches **100% after 128 orthogonal-key
writes**, versus **26.0% for additive memory and 14.5% for this trained transformer**.
The transformer reaches 98.2% at its trained length of 32 writes; it learned the task.
Both exact retrieval controls reach 100% throughout.

The counter-results matter equally. At cosine 0.5 and 32 writes, the transformer
reaches 97.8% while residual memory reaches 71.2%. For a first pass of balanced values
at cosine 0.9, additive memory is perfect and residual memory falls to 12.5%.
The most recent residual write is still correct: the failure is interference with
older associations. A simple rule has a useful operating range, not universal optimality.

`receipt.json` contains each seed, prediction, target, condition, training budget and
wall-clock measurement. Its manifest binds this script and Cadence's memory implementation.
The generated table in the repository README reports the orthogonal 128-write condition;
inspect the correlated-key rows before choosing a write rule.

The memory contains 9×8 float64 entries, or **576 bytes per stream**, at every tested
length. This counts seam payload only. Exact attention keeps a growing key/value
history; a dictionary only needs the latest value per key and is also bounded. The
script's dictionary is a Python implementation, so its object overhead is not included
in the dense-array payload comparison. Inference timings are CPU wall times for all
writes and reads; transformer encoding and inference are timed together, and its
training cost is separate. These are small batch timings, not a hardware throughput claim.

Exact lookup solves the symbolic task without training. The useful result is that a
simple bounded residual memory can preserve revisions that additive memory loses,
and can generalise beyond one trained transformer's context lengths. Correlated-key
interference is a real limitation. This is not evidence that patch nets outperform all
transformers, that delta learning is new, or that language modelling is solved.
