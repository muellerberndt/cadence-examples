# Cadence examples

Small runnable examples for [Cadence](https://github.com/muellerberndt/cadence).
Patch nets are observer-like self-reading systems: local owners hold state, declared
seams carry information across their boundaries, queries cause readback, and local
feedback repairs state or memory. Scripts produce evidence receipts.

Start with [updating memory](05_memory/): a few lines add fast residual writes to a
fixed-size memory. The benchmark compares changing associations with additive writes,
exact retrieval and a trained transformer. The exact algorithm is included because
explicit symbolic keys make this a lookup task; beating one trained model does not
establish a general advantage over transformers.

[Circuit interventions](06_interventions/) shows the complementary idea: retain
a known local mechanism, then change drives, wiring or ablations without training
an input/output surrogate. Its comparison includes a trained MLP, direct solver
and an explicitly unrolled recurrence with the same complete circuit information.

<!-- ladder -->
| # | example | what it shows |
|---|---|---|
| 01 | [digits](01_digits/) | 8×8 scikit-learn digits; 64 input owners, a hidden layer, 10 output owners; receipt: 0.962 ± 0.003 held-out in 20 epochs; a smaller MLP (2,410 parameters) 0.967 in 50; [play it](https://claude.ai/code/artifact/0f6136f7-79ca-4b0e-9cef-65fd70fc6618) |
| 02 | [recall](02_recall/) | distinct one-hot keys written into 128-key Hebbian memory; receipt: historical distinct-key recall 1.00; dictionary also solves this lookup task; original transformer comparison has parser and position disadvantages; [play it](https://claude.ai/code/artifact/ff0e3f63-b674-494e-8c0f-a99d845d678b) |
| 03 | [Connect Four](03_connect_four/) | self-play positions labelled by a depth-4 search; the net imitates the search and plays with no lookahead; receipt: agrees with a depth-4 search on 0.529 of positions (MLP 0.533); 98-0-2 vs random, 2-1-97 vs depth 2; [play it](https://claude.ai/code/artifact/a75ef805-c396-4c9d-b64d-6ca5fe60badc) |
| 04 | [Pong](04_pong/) | a paddle that learned from pixels and reward: the nudge's target is the action taken, its strength the action's advantage, each seam's step read from its own history; receipt: reward-trained paddle returns 87.8% of balls against 92.9% for backprop REINFORCE on the same rollout budget; the same net taught the tracker's moves 96.0%; [play it](https://claude.ai/code/artifact/4b3fe725-e687-4acb-a29c-5f2eb9a69e04) |
| 05 | [updating memory](05_memory/) | a fixed-size memory corrects its own readback when an association changes; receipt: 128 writes, orthogonal keys: residual 1.000, additive 0.260, dictionary 1.000, trained transformer 0.145; correlated-key results in the receipt |
| 06 | [circuit interventions](06_interventions/) | owners read incoming messages and repair their local state as drives, wiring and ablations change; receipt: declared feedback circuits answer new interventions without training; maximum residual 7.4e-12, trained MLP mean MSE 0.0052; direct and tied-recurrence controls also solve the task |
<!-- /ladder -->

Examples 01–04 retain **historical receipts**. Their numbers describe the pinned source
versions, and those receipts did not bind the Cadence dependency. The Connect Four
receipt predates a board/reflection split fix; the recall transformer comparison has
parser and positional-embedding disadvantages. Each tutorial explains its limits.
The earlier experiments remain available at [tag v0.5.0](https://github.com/muellerberndt/cadence-examples/tree/v0.5.0).

## Run

```bash
pip install -r requirements.txt torch
cd 05_memory
python train.py --steps 800 --seeds 3
python train.py --verify receipt.json
```

These examples require the repository version of Cadence containing
`FastSeams(rule="delta")`; while working with sibling checkouts, install with
`pip install -e ../cadence` from this repository's root.

The numbered tutorials explain each data source and run budget. Existing trained
pages need no retraining: `python serve.py` opens the local hub. Publicly hosted copies
are historical and are not updated by editing this checkout.

## Reproduction

- `python tools/verify_receipts.py` verifies current source digests or explicitly
  reports pinned historical provenance. This is an integrity check, not a rerun.
- `python tools/smoke.py` runs small budgets in isolated temporary copies. It leaves
  modified source files and existing nets untouched.
- `python -m pytest -q tests` checks reflection isolation and benchmark controls.
- `python tools/ladder.py` regenerates the table and hub cards from receipts.
- `python tools/pages.py` checks the playable pages and their Python/JavaScript parity.

[How a patch net learns](HOW_IT_LEARNS.md) explains the slower equilibrium-propagation
learner used by the supervised and reward examples. Fast residual memory is a separate
mechanism; it does not require that learner or an equilibrium solve for each write.
