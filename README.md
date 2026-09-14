# Cadence examples

Small runnable examples for [Cadence](https://github.com/muellerberndt/cadence).
Patch nets are observer-like self-reading systems: local owners hold state, declared
seams carry information across their boundaries, queries cause readback, and local
feedback repairs state or memory. Scripts produce evidence receipts.

## Live composite brains

![The teachable mouse: task memory, a spatial field and a moving body](showcase/preview.png)

Run `python serve.py` and open the printed address. The new default experience
puts embodiment, continual task learning and visible feedback first:

| Demo | Try this | What it makes visible |
|---|---|---|
| [Teachable mouse](showcase/README.md) | Teach a new destination, run the task, then generate another maze | Task memory, a recurrent spatial field and a moving body; new lessons retain earlier cue mappings |
| [Eye & arm](showcase/README.md) | Present an outline or upload an image; disturb a joint while it draws | Visual-error and motor-correction regions settle together and correct actual body readback |
| [Fly-inspired forager](showcase/README.md) | Change nectar or move flowers during flight | Each encounter updates memory and influences later choices |
| [Worm circuit](showcase/README.md) | Stimulate or remove neurons | A supplied model on public chemical wiring answers interventions without another training run |
| [Changing memory](showcase/README.md) | Teach a key, replace its value, increase key overlap | Residual writes, online MLP update budgets, retention and interference |

The [two-minute demonstration guide](showcase/README.md#a-two-minute-demonstration)
explains the interactions. The browser displays measured comparisons beside the
live systems. The mouse and fly use simplified bodies; the arm has supplied
geometry and an edge-extraction adapter. These examples distinguish learned
records from supplied mechanisms and include strong conventional controls.

```bash
python serve.py              # the teachable mouse and all five live demos
python serve.py eye-arm      # start with the drawing arm
python serve.py fly          # start with the forager
```

### Supporting tutorials

The original six examples remain runnable, with their receipts and history.
The [tutorial hub](tutorials.html) provides the previous learning ladder.

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

Clone this repository and run these commands from its root. The live showcase
and original browser pages include their model data and need only Python to serve,
with no packages installed:

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
- `python tools/ladder.py` regenerates supporting tutorial tables and cards.
- `python tools/build_showcase.py` rebuilds the default and publication hubs.
- `python showcase/verify.py` checks the new source-bound evidence packages.
- `python tools/showcase_pages.py` checks all five new demos in a real browser.
  See [showcase reproduction](showcase/README.md#reproduce) for the experiment commands.
- For browser checks, run `python -m pip install playwright` and
  `python -m playwright install chromium`, then `python tools/pages.py`.
  This checks the playable pages and their Python/JavaScript parity.

[How a patch net learns](HOW_IT_LEARNS.md) explains the slower equilibrium-propagation
learner used by the supervised and reward examples. Fast residual memory is a separate
mechanism; it does not require that learner or an equilibrium solve for each write.
