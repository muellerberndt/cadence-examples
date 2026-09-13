# 02 · Recall

A small associative memory with explicit key and value owners. Each write adds a
Hebbian outer product to its seams; a query clamps a key and reads the resulting value
activation. The current demonstration uses 128 distinct one-hot keys. This gives the
memory a known address for every key; a Python dictionary solves the same task exactly.

From the repository root, after activating your Python environment:

```bash
python -m pip install -r requirements.txt torch
cd 02_recall
python train.py
python train.py --verify receipt.json
```

The checked-in receipt is a **historical measurement**. Check its preserved provenance
from the repository root with `python tools/verify_receipts.py 02_recall`. Its Cadence
dependency was not source-bound; it does not certify a later library implementation.

## What is computed

There are 128 key owners and 128 value owners, with 32,768 directed overlaps. A write
adds one unit in both directions between the active key and value owners. Two steps
from zero are enough for the key to activate and deliver its contribution to the values.
This is an unnormalised linear associative read followed by the activation function.
It is **not softmax attention** or the modern Hopfield update. Additional settlement
steps do not improve accuracy on this example's recorded task.

This is an observer-like self-reading system: bounded owners hold local state, seams
act as ports, a clamped query causes readback, and settlement applies local feedback.
The receipt records the experiment. No slow learning or equilibrium detuning is used.

## Historical comparison and limits

The receipt reports 100% recall for both the two-step read and longer settlement on
100 episodes at each of 4, 8, 16, 32, 48, 64, 96 and 128 pairs, across three seeds.
Each episode has distinct keys, so it never asks the memory to replace an association.
The dense allocation is fixed at 128×128; this is not unbounded context or memory.
The historical reference-conformance check used an empty memory, so it did not check
the written seams. The current script runs that check after representative writes.

A two-layer transformer trained for 5,000 Adam steps reached 30% at four pairs and 1%
at 128. This is a weak baseline for this protocol: it had to learn the key/value parser
that the memory receives explicitly, was evaluated on different random episodes, and
its learned positional embeddings past its training length were untrained. The result
does not establish a limitation of transformers or attention. A dictionary and an
exact key-matching attention reader both achieve perfect accuracy here without training.

## Try the page

Open `index.html`, write a context and click a key. Each key has a dedicated owner;
the bars display value readback. Repeated writes **accumulate**, although the context
labels show the most recently requested association. This can expose stale-memory
errors; the historical receipt tests distinct keys only.

[05 · Updating memory](../05_memory/) tests changing associations with the residual
write rule, an additive control, exact retrieval, and a trained attention model on
identical episodes. It also measures interference between correlated keys.
