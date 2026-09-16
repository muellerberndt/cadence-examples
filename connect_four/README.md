# S03: Connect Four

One life learns what a dropped stone does, which windows of four cells are completed lines and
what positions are worth, from the games it plays, and chooses its moves by searching over what
it learned. The brain receives moments and returns decisions: no board object, no step function,
no terminal oracle, no minimax labels and no correct-move labels. `tests/test_env.py` reads
`brain.py` and fails when it imports the evaluator, the controls, the opponents or the engine.

A decision spans the candidate's move and the opponent's reply. The board after the candidate's
move is the feedback of that decision and stays inside the brain for repair; the reply moment
carries the feedback to the agent. Every observed transition is stored with its mover, so the
candidate's action is never blamed for the opponent's move.

## The brain

Two record cortices (`cadence.Records`) with local receptive fields whose records are shared
across positions, the way a cortical column repeats over space.

- **Drop records** read one column: its cells from the bottom as one-hot (empty, the side to
  move, the other side), and whether the column is the chosen one. The field `landing` has one
  class per row and one for none. The seven column readings of a board are coded as one batch,
  so every observed move of either side teaches every column. The imagined next board is the
  board plus the mover's stone at each predicted landing: the change is predicted and the board
  is composed.
- **Line records** read one window of `connect` cells along a row, a column or a diagonal,
  relative to the side that just moved. The field `complete` says whether the window is a
  completed line of that side. A board after which the game went on has no completed window; a
  won board has one among the windows through the new stone, and the write goes to the candidate
  windows whose records already rate them highest. The valued field `value` holds the outcome of
  finished games for the side that just moved (win 1, draw 0.5, loss 0), written when the
  outcome arrives. A board's value is the average over its windows.
- **The search** is negamax with alpha-beta pruning and iterative deepening over imagined
  boards: depth 2 within 256 imagined transitions per decision, and depth 4 within 1,024 once
  the drop records' exact validity over the brain's own last 500 real transitions reaches the
  validity gate. Leaves are scored by the line records: a completed window of the mover is a
  win, a full board without one is a draw, and every other leaf takes the value records. A
  proven result ends the deepening. A validator rejects an imagined board that is not the old
  board plus one stone of the mover in an empty cell of the chosen column; that branch keeps the
  value of the position it came from, `planner.invalid` counts it, and the board is left as it
  was composed. Gravity belongs to the drop records, so the validator does not check it.

Learning is online from one stream. There is no replay ring and no minibatch: a reading touches
few records and the witnessed outcome is written into exactly those. The memo
`../../cadence-paper/research/RECORD_PRINCIPLE.md` states the principle and its receipts.

## Supplied and learned

Supplied: the world, its opponents and its rules, all outside the brain; the legality mask of
real moments and the legal columns of an imagined board (its top cell is empty); the categorical
cell encoding relative to the candidate with the mover and phase fields and the goal port; the
layout of the receptive fields (the cells of each column, the windows of `connect` cells, a
reading seen by one side); the fixed random expansions of the cortices with their winner-take-all
inhibition, write rates and the multiple-instance write of a won board; the composition of an
imagined board and the validator; the terminal classes composed from the line records (a win is a
completed window, a draw is a full board without one); the negamax search with its budgets; the
one-step snapshot policy; and the S00 event transaction with its small settled regions (a
workspace of 16 neurons, a dynamics region of 8, and the critic).

Learned: the landing row per column reading, from every observed move of either side; which
window patterns are completed lines, from witnessed continuations and wins; the outcome of
finished games per window pattern for the side that just moved; and the S00 critic over the
workspace, which is reported and which the planner does not read.

## Controls and interventions

- **No planning**: the snapshot policy of the same records, which imagines each legal move one
  step ahead and takes the best learned value. It is the no-planning control of the planning
  gain and the snapshot opponent of the mixture.
- **Corrupted dynamics**: a frozen copy whose drop records are permuted across code cells, with
  the search and everything else unchanged.
- **Born frozen**: the identical architecture with learning off from birth, run through a
  quarter of the exploration games.
- **Supplied-rules controls**: random, the one-ply opponent, and minimax at 256 and 1,024 nodes
  (alpha-beta with terminal values only, iterative deepening under the node budget). A
  controls-only run calibrates the evaluator without a candidate.

## Measurements

The pilot runs seed 0 for 120 games with 100 paired games per opponent; the development run
runs seeds 0 and 1 for the declared 1,200 games with 1,000 paired games per opponent, 22 minutes
per life on two workers. Every value below is recomputed from the event logs by `verify.py`.

| gate | threshold | pilot | development |
|---|---|---|---|
| next board validity | 0.99 | 1.00 on 10,000 held-out moves | 1.00, changed cell 1.00 |
| terminal prediction | 0.99 | 1.00 over 2,000 boards | 1.00, every class 1.00 |
| tactical success | 0.95 | 1.00 on 1,000 probes | 1.00, every stratum 1.00 |
| paired against random | 0.85 | 0.99 | 0.997, seed floor 0.995 |
| paired against one-ply | 0.60 | 0.875 | 0.968, seed floor 0.961 |
| planning gain over the snapshot policy | 0.10 | 0.745 | 0.393 |
| tactical loss under corrupted dynamics | 0.15 | 0.888 | 0.873 |
| loss on old opponents after adaptation | 0.05 | 0.00 | 0.0015 |
| p95 decision latency | 1,000 ms | 28.6 ms | 34.5 ms (p50 11.9 ms, max 654 ms) |
| games of one life | the declared budget | 120 | 1,200 |

In the development run the candidate scores 0.873 and 0.881 against the supplied-rules minimax
at 256 nodes, and 0.867 and 0.941 against it after the adaptation games, which are played against
a mixture that is half that opponent. The controls on the same 1,000 probes and 1,000 paired
games: the no-planning snapshot policy solves 0.62 and 0.68 of the tactical suite and scores
0.541 and 0.608 against one-ply; the corrupted copy solves 0.138 and 0.117 with next-board
validity 0.004 and 0.044; the born-frozen life solves 0.178 and 0.143 and scores about 0.49
against random and 0.03 against one-ply. The controls-only run calibrates the evaluator: minimax
at 256 and at 1,024 nodes solves the tactical suite completely, random solves 0.163 of it,
one-ply scores 0.96 against random, and random against random sits at 0.54. Against one-ply,
minimax at 1,024 nodes scores 0.87 and minimax at 256 nodes scores 0.705, at node budgets
matched to the candidate's 1,024 and 256 imagined transitions.

Games to threshold, measured on the development stream with 200 tactical probes, 2,000 held-out
moves, 400 terminal boards and 60 paired games per opponent:

| games | next board | terminal | tactical | random | one-ply | planning gain |
|---|---|---|---|---|---|---|
| 40 | 1.00 | 1.00 | 1.00 | 1.00 | 0.942 | 0.659 |
| 100 | 1.00 | 1.00 | 1.00 | 0.967 | 0.808 | 0.758 |
| 200 | 1.00 | 1.00 | 1.00 | 0.983 | 0.950 | 0.433 |
| 400 | 1.00 | 1.00 | 1.00 | 1.00 | 0.967 | 0.459 |
| 800 | 1.00 | 1.00 | 1.00 | 1.00 | 0.967 | 0.592 |

The search extends to depth 4 once validity reaches the gate, which happens during the
exploration games. Over the pilot's 848 searches, 586 completed depth 4 within 1,024 imagined
transitions, 42 stopped after depth 3, 146 after depth 2 and 74 after depth 1. A proven result
ends the deepening, so a win in one stops it at depth 1 and a position where every move loses at
once stops it at depth 2; the remaining early stops are searches whose next iteration would have
passed the budget. The validator rejected 53 of 351,638 imagined boards. The learned state is 14,000 drop records
and 6,000 line records; the settled regions are 187 neurons with 3,960 directed synapses, and
their free phase settles in 48 steps on average with no rejected update and no replay write.

## Commands

```bash
PY=../cadence/.venv/bin/python
$PY -m pytest connect_four/tests -q
$PY connect_four/run.py --seeds 0 --pilot --out runs/connect_four/pilot
$PY connect_four/run.py --seeds 0 1 --workers 2 --out runs/connect_four/dev
$PY connect_four/run.py --controls-only --seeds 0 --out runs/connect_four/controls
$PY connect_four/run.py --seeds 10 11 12 13 14 --workers 2 --out runs/connect_four/acceptance
$PY connect_four/verify.py runs/connect_four/acceptance
```

Run them from the repository root. Each life is one process of small numpy work; set
`OMP_NUM_THREADS=1` when several seeds share a machine.

## Files

- `env.py`: the rules and the world that emits moments; `opponents.py`: random, one-ply,
  snapshots and their mixture; `controls.py`: the supplied-rules policies.
- `evaluate.py`: the independent evaluator, its held-out suites and the paired harness.
- `brain.py`: the record cortices, the imagination with its validator, the planner and the
  `Brain` the runner drives; `brain_interface.md` states the contract.
- `run.py`, `verify.py`: the life runner and the verifier, which replays every logged game and
  judges every logged probe with its own rule checker.
- `config.json`: the game, the budgets, the suites, the gates and the brain's own sections;
  `tests/`: the engine, the evaluator and the brain.

## The page

`web/` holds the public page: a board a visitor plays on, with the whole brain beside it.
`web/brain.js` is the port of this brain to the browser (the two record cortices, the
imagination with its validator, the negamax search within its budget, and the writes after
every move and every finished game), `web/game.js` the port of the rules and the world,
`web/page.js` the page itself. The S00 part runs through the shared engine in `web/engine.js`,
so `web/brain_scan.js` draws every settling step while the visitor plays and `web/records_view.js`
draws the cells of the chosen column's reading and of the windows through the new stone. The
brain keeps learning from the game being played, and a switch freezes it.

```bash
$PY connect_four/web/build_page.py --run runs/connect_four/acceptance --seed 10
$PY connect_four/tools/parity_fixture.py --run runs/connect_four/acceptance --seed 10 \
    --games 32 --out runs/connect_four/parity
node connect_four/web/parity.mjs runs/connect_four/parity
$PY connect_four/web/check_page.py connect_four/index.html
```

`build_page.py` writes `index.html` with the last checkpoint inlined, `checkpoints/<id>.json`
for every other checkpoint of that life, `cortices.json` (the fixed cells both cortices were
drawn with, shared by every checkpoint) and `checkpoints.json` (the manifest and the receipt's
numbers). The parity harness replays a recorded fixture of 32 games through the browser brain
and holds every column, every predicted landing, every imagined next board and every counter to
the Python numbers exactly, and the record tables to 1e-9. `check_page.py` opens the built page
in headless Chromium, plays a game at every checkpoint with random legal moves, and fails on a
page error, a layout that does not fit, a brain view that does not render or an answer slower
than 100 ms.

## The page's brain and the bench

The brain on the page is seed 0 of `config_solver.json`: the same life with Pascal Pons'
perfect solver in the curriculum at graded strengths (`bench/pons.py`; the solver plays the
perfect column with probability 0.3 or 0.7 in the mixture and 0.7 or 1.0 in the adaptation,
else a random one). Its records play at eight plies within 32,768 imagined transitions; the
search is supplied, so the records are unchanged. `bench/measure.py` plays paired games
against named opponents and scores every candidate move with the solver: optimal, a slip
(worse, the theoretical result kept) or a blunder (the result changes), the blunder rate over
the positions not already lost. `bench/setup_external.sh` fetches and builds the solver with
its opening book and alpha-zero-general under the git-ignored `external/`;
`bench/alphazero/` is that repository's Connect Four in PyTorch with a trainer and an
opponent; `opponents.make_opponent` names every opponent (`solver_070`,
`alphazero=<checkpoint>@<sims>`, ...). The model card on the page carries the numbers; the
receipts are in `bench/receipts/`.

```bash
bash connect_four/bench/setup_external.sh
$PY connect_four/run.py --config connect_four/config_solver.json --seeds 0 1 --workers 2 --out runs/connect_four/solver_dev
$PY connect_four/bench/measure.py --brain runs/connect_four/solver_dev/brain_seed0_games001200 --games 100 \
    --depth 8 --budget 32768 --opponents one_ply minimax_1024 solver_070 --out runs/connect_four/bench/solver_dev_seed0
$PY connect_four/bench/alphazero/train.py --out runs/connect_four/alphazero/recipe --iters 32 --episodes 100 --sims 25
```

`config_solver_boot.json` adds search bootstrapping (the searched root values are written
into the value records) and `config_solver_support.json` a value cortex whose reading carries
the support of each window cell; both are work in progress.

## Budgets, checkpoints and assumptions

One life is 200 exploration games, 800 mixture games and 200 adaptation games, which is 1.2
percent of the plan's cap of 100,000 games per seed. The drop records reach exact held-out
validity after 20 to 50 random games and the line records classify every window pattern after
about 10, and every gate already passes at 40 games on the development curve. The declared budget
is thirty times that point, so the pilot, which divides the game budgets, the paired games and the
snapshot interval by ten, keeps a threefold margin. The diagnostic suites keep their size in the
pilot.

Checkpoints are written after 100, 400 and 1,000 games and at the end of every phase, named by
the game count: `brain_seed<seed>_games<NNNNNN>.agent.npz` holds the agent snapshot
(`agent.save`) and `brain_seed<seed>_games<NNNNNN>.records.npz` holds every record cortex's
configuration, fixed projection and offsets, tables and running statistics, the layout of the
receptive fields, the planner's settings, statistics and generator, and the validity record that
decides the search depth. The pair is about 580 KB and rebuilds the brain completely:
`Brain.load` returns a brain that answers and plays identically, and `tests/test_brain.py` checks
that the rebuilt life continues the saved one move for move. The receipt lists every checkpoint
file with its sha256 and its size.

The remaining assumptions of the stage are in `config.json`: the observation encoding, the
decision and feedback timing, the curriculum, the held-out suites, the paired harness and the
latency rule.
