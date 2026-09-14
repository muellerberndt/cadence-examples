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
| 01 | [digits](01_digits/) | 8×8 scikit-learn digits; 64 input owners, a hidden layer, 10 output owners; receipt: 0.962 ± 0.003 held-out in 20 epochs; a smaller MLP (2,410 parameters) 0.967 in 50; [browser page](01_digits/index.html) |
| 02 | [recall](02_recall/) | residual fast seams learn on each write and correct old associations; receipt: residual writes: 100.0% correct after replacing repeatedly written associations; exact dictionary also succeeds; [browser page](02_recall/index.html) |
| 03 | [Connect Four](03_connect_four/) | imitate a teacher, play, and revisit mistakes; receipt: with depth-4 lookahead: 100.0% wins vs random, 56.0% vs depth 2; search alone 51.0%; raw policy 1.0%; [browser page](03_connect_four/index.html) |
| 04 | [Pong](04_pong/) | a paddle sees one frame and keeps a fading trace of its activity; receipt: one frame plus a fading neural trace; imitation then practice: 94.5% points won vs skill-0.7 tracker, 94.9% balls returned; stronger opponent results in the receipt; [browser page](04_pong/index.html) |
| 05 | [updating memory](05_memory/) | a fixed-size memory corrects its own readback when an association changes; receipt: 128 writes, orthogonal keys: residual 1.000, additive 0.260, dictionary 1.000, trained transformer 0.145; correlated-key results in the receipt |
| 06 | [circuit interventions](06_interventions/) | owners read incoming messages and repair their local state as drives, wiring and ablations change; receipt: declared feedback circuits answer new interventions without training; maximum residual 7.4e-12, trained MLP mean MSE 0.0052; direct and tied-recurrence controls also solve the task |
<!-- /ladder -->

All six examples run on the pinned current core. Their receipts bind the producer
and Cadence implementation; historical versions remain in Git history.

| example | mechanism used |
|---|---|
| digits | local free/nudged learning; capacity selected on validation images |
| recall | residual fast writes that correct a previous association |
| Connect Four | imitation, play, corrective teaching; optional explicit lookahead |
| Pong | one incoming frame, carried `Afterglow`, imitation, practice and rehearsal |
| updating memory | residual writes with additive, retrieval and learned controls |
| circuit interventions | feedback settlement, warm state and independent residual checks |

For the game-learning sequence and persistent browser lessons, see
[Training a player](TRAINING.md). Temporal memory is used where observations omit
history; static images and fully visible boards do not need an extra frame buffer.

## Run

Clone this repository and run these commands from its root. The four browser pages
already contain their trained nets and need only Python, with no packages installed:

```bash
python serve.py
```

Use `python serve.py pong` to open one game, or
`python serve.py --port 0 --no-browser` to print an available local URL. Stop the
server with Ctrl-C. Publicly hosted copies are historical and are not updated by
editing this checkout.

For the two short Python demos, use Python 3.11 or newer and Git. The default
dependency file installs the supported Cadence revision with the latest fixes:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python 05_memory/demo.py
python 06_interventions/demo.py
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. The numbered
tutorials explain the full training runs; those write new receipts and, for examples
01, 03 and 04, trained nets. Examples 04–06 use PyTorch for their trained
baselines: `python -m pip install torch`.

To develop against a sibling Cadence checkout, install it with
`python -m pip install -e '../cadence[fast]'`. Smoke runs can test that version;
the saved receipts still certify their original source bytes.

## Reproduction

Install `requirements-reproduce.txt` to check the recorded measurements with the
same supported core revision, then run `python tools/verify_receipts.py`. This checks
source integrity and recorded arithmetic, not a rerun of training. CI also runs small
fresh experiments, performance checks, and Python/JavaScript parity checks.

In your ordinary development environment:

- `python tools/smoke.py` runs small budgets in isolated temporary copies. It leaves
  modified source files and existing nets untouched. Pass a directory such as
  `05_memory` to run one example. Install PyTorch first to run all six.
- `python -m pip install pytest`, then `python -m pytest -q tests`, checks reflection
  isolation and benchmark controls.
- `python tools/ladder.py` regenerates the table and hub cards from receipts.
- For browser checks, run `python -m pip install playwright` and
  `python -m playwright install chromium`, then `python tools/pages.py`.
  This checks the playable pages and their Python/JavaScript parity.

[How a patch net learns](HOW_IT_LEARNS.md) explains the slower equilibrium-propagation
learner used by the supervised and reward examples. Fast residual memory is a separate
mechanism; it does not require that learner or an equilibrium solve for each write.
