# One circuit, new interventions

A declared feedback circuit can answer a new stimulation, ablation or rewiring
without fitting another input/output model. Cadence reaches a checked numerical
equilibrium on every held-out slice here with **zero training examples**. The
trained MLP sees the complete circuit too, but its finite training budget leaves
measurable prediction error. This is useful when the local mechanism is known
and you need to explore many interventions.

[demo.py](demo.py) is the whole public demonstration: four owners, four seams,
two stimulation ports and an ablation. Each owner holds local state, reads its
incoming messages and repairs its state. The independent equation check and
[receipt](receipt.json) make that readback auditable.

```bash
# From cadence-examples:
python -m pip install -r requirements.txt torch
cd 06_interventions
python demo.py
python train.py --steps 2000 --seeds 3 --trials 128 --output local_receipt.json
python train.py --verify local_receipt.json
```

This writes a fresh result with the supported Cadence revision and optional Numba
acceleration. To verify the saved `receipt.json`, use the separate environment
installed from [`requirements-reproduce.txt`](../requirements-reproduce.txt), as
shown in the [reproduction instructions](../README.md#reproduction), then run
`python tools/verify_receipts.py 06_interventions` from the repository root. Saved
measurements bind that original dependency; compatibility checks with newer code
do not refresh them. There is no cloud or GPU requirement. The
[original receipt and producer](history/2026-09-13-before-mask-fix/) remain
available with their historical source revision.

The larger comparison uses 16-owner directed circuits with equation
`s = mask * tanh(s @ W + drive)`. We supply the wiring and local rule; this is
**executing known dynamics, not learning physics or modeling an animal**.
Every incoming column of `abs(W)` sums to at most 0.75. Because `tanh` is
1-Lipschitz, the update has a unique fixed point. The maximum owner-equation
residual bounds the maximum state error by `residual / (1 - 0.75)`. Cadence's
stopping tolerance is checked against that equation independently.

Every arm receives identical weights, drives and binary ablation masks. The MLP
also gets an exact first propagation, a direct-drive skip, `tanh` and the hard
output mask. Its two hidden layers have width 128 or 256 (57,616 or 147,984
parameters). Both candidates train for 2,000 Adam steps on 8,192 examples over
32 circuits. Selection uses the half/final checkpoints on 512 independent
examples over eight separate validation circuits. This is a bounded fit,
not evidence of the best achievable MLP. Targets come from 128 independently
implemented recurrence steps; generating these labels is charged separately.

Evaluation has three seeds and 128 queries per seed in each slice. `new_drives`
uses training circuits with fresh drives. The other slices use 16 previously
unseen circuits: ordinary drives, up to three ablations instead of up to one,
the joint stress of drive standard deviation 2 instead of 0.6 plus up to three
ablations, and sequences of 16 small drive changes. Every owner is an eligible
stimulation and readout port; the output is
the complete state. No test slice chooses a checkpoint.

<!-- intervention-results -->
Mean squared error, averaged over the three seeds:

| slice | Cadence | warm | MLP | one propagation | 8 steps | 32 steps | Newton |
|---|---|---|---|---|---|---|---|
| new_drives | 7.70e-24 | 7.95e-24 | 4.21e-03 | 4.43e-03 | 2.06e-06 | 4.60e-14 | 8.47e-29 |
| new_graphs | 4.76e-24 | 4.52e-24 | 4.49e-03 | 4.04e-03 | 1.85e-06 | 9.76e-15 | 1.20e-28 |
| more_ablations | 3.58e-24 | 3.85e-24 | 4.35e-03 | 4.03e-03 | 1.42e-06 | 6.99e-15 | 1.77e-28 |
| strong_drive | 2.38e-24 | 2.48e-24 | 8.17e-03 | 6.80e-03 | 9.01e-07 | 5.43e-15 | 3.46e-28 |
| small_changes | 8.11e-24 | 7.44e-24 | 4.74e-03 | 4.49e-03 | 4.31e-06 | 3.30e-15 | 8.93e-30 |

Across both Cadence paths, maximum independent residual **7.48e-12**, giving a maximum state-error bound of **2.99e-11**.

Mean settling steps:

| slice | cold | warm |
|---|---|---|
| new_drives | 38.0 | 38.6 |
| new_graphs | 37.2 | 38.7 |
| more_ablations | 34.5 | 36.0 |
| strong_drive | 30.8 | 32.5 |
| small_changes | 39.6 | 28.8 |

Mean CPU microseconds per query across all five slices:

| arm | sequential | within a batch of 128 |
|---|---|---|
| Cadence | 26.5 | not measured |
| warm | 27.7 | not measured |
| MLP | 32.8 | 5.2 |
| one propagation | 4.1 | 0.3 |
| 8 steps | 13.0 | 1.0 |
| 32 steps | 48.1 | 4.0 |
| Newton | 43.8 | not measured |

Across the complete run, both MLP candidates and their checkpoint selection took 11.34 s, plus 0.54 s to prepare training/validation inputs and labels. All Cadence cold/warm graph bindings took 0.030 s, plus 0.293 s to initialize kernels. These costs are additional to the query timings.
<!-- /intervention-results -->

The MLP slightly improves on one propagation for new drives on familiar
circuits, and loses to that simple control on the other four slices in this
run. Cold and warm Cadence both meet the residual contract. Reusing state saves
iterations on the small-change sequences; on unrelated queries it costs
slightly more iterations. State reuse is a workload-dependent option.

The **direct Newton solver and tied recurrence also need zero training** and
solve the same problem. In particular, the 32-step control is already very
accurate. Unrolling it produces a feedforward computational graph with shared
weights. The result supports carrying a known mechanism into new contexts; it
does not establish a general separation between recurrence and feedforward
computation, or an advantage in learning an unknown mechanism.

All predictions use float64. Sequential timings include per-query preparation
and graph lookup after bindings are built; batched timings amortize work across
128 queries and must be compared separately. Binding and initialization costs
are recorded, including any first-use JIT. MLP training and synthetic-label
preparation are also recorded separately. Cadence uses a fused Numba CPU kernel
in this receipt; the unrolled control uses a Python loop over NumPy operations.
These timings describe implementations on this machine, not inherent
architecture speed limits. The fixed-depth controls trade precision for speed.

The receipt contains every prediction, residual, seed, evaluation-input hash,
candidate validation score, step count and measured cost. Verification binds
the benchmark and library sources, regenerates evaluation inputs, checks the
equilibrium targets, recomputes metrics and replays the fixed-depth controls.
It does not replay MLP training or certify timing observations. Positive-threshold
agreement at 0.2 is included as a secondary diagnostic; it ignores negative
activations and should not replace the residual and MSE.
Residual arithmetic checks allow small absolute libm roundoff across platforms;
the independent solver residual contract remains 1e-9.
