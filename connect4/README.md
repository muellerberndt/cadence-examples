# Connect Four

One `cadence.RecordPatchNet` reads a position right after a stone has landed, as the side that
placed it sees it, and says how the game ends for that side. A supplied search knows the rules,
imagines moves and reads the patch where it stops looking. The page plays it in the browser
with the same arithmetic as the library and draws every sampled read.

## What this example shows

- **Planning.** The brain chooses a move by imagining boards and reading a learned value where it stops looking. A line it can follow to the end of the game is proven and outranks anything it reads.
- **Learning by watching.** The value comes from games a perfect player played out, seen with how they ended. No score was given.
- **Slow parameters and records have different jobs.** The school trains the slow parameters and leaves the record store empty, because writing millions of positions into it drowns it (measured on three runs). The records are for the games played afterwards; that use is work in progress.
- **Play judged move by move.** A perfect solver grades every move, by the stage of the game, beside a control that searches with no value patch.

## Layout

| file | what it is |
| --- | --- |
| `game.py` | the rules on two 49-bit boards, and the 84-unit reading (a position and its mirror image are one reading) |
| `school.py` | games to learn from: set-up positions played out to the end by Pascal Pons' perfect solver |
| `patch.py` | the value patch: one `RecordPatchNet`, every reading a path of one moment from rest |
| `train.py` | the school: slow parameters only, records left empty, checkpoint chosen on a validation split |
| `deploy.py` | the brain as deployed: empty record store, reading statistics settled through documented calls |
| `brain.py` | the search: iterative deepening, exact wins from the rules, forced blocks, one value per reading |
| `bench/` | the solver bridge, the AlphaZero baseline, `measure.py` (every move graded by the solver) |
| `web/` | the page, its engine (`patch.js`, `brain.js`), parity checks and the arena that produces the receipt |
| `brain/v1.npz` | the deployed patch the page's `brain.json` is exported from |
| `receipts/record_address.json` | what two moves that differ by one stone share in the record store, and what a write to one does to the other, plain and anchored (`tools/record_address.py`, reproduced from this directory alone) |
| `receipts/records_drown.json` | three runs schooled with every position written into the records: the slow readout alone against the value with the record read (`tools/records_drown.py`) |
| `receipts/control_rules_only.json` | the same opponents against the search alone, with no value patch: what the patch adds is the difference |
| `verify.py` | replays every game of the receipt with the rules, recounts its table, and checks that the receipt is bound to the page's brain and engine and that the page's brain is the export of `brain/v1.npz` |
| `tools/make_card.py` | draws the social card with the page itself, mid-thought |

## Reproduce

    python connect4/verify.py                              # the receipt, its games, the page's brain
    connect4/bench/setup_external.sh                       # Pons' solver, its book, alpha-zero-general
    python connect4/school.py --games 50000 --seed 1 --out runs/connect4/school/s1.npz
    python connect4/train.py --school runs/connect4/school/s1.npz --heldout runs/connect4/school/test.npz --out runs/connect4/patch/v1
    python connect4/deploy.py --patch runs/connect4/patch/v1/patch.npz --school runs/connect4/school/s1.npz --out runs/connect4/deployed/v1.npz
    python connect4/web/export.py --patch runs/connect4/deployed/v1.npz --out runs/connect4/web_v1
    node connect4/web/parity.mjs runs/connect4/web_v1      # the page reads what cadence reads
    node connect4/web/arena.mjs --brain runs/connect4/web_v1 --opponents one_ply solver_070 solver_100 --out receipt.json

School files are regenerated from their seed (the same seed gives the same file), so they are
not stored.

## What was measured

The receipt beside the page (`web/receipt.json`) holds every game and every graded move of the
build that is deployed; the page's card renders its table from that file and plays with the
search settings recorded in it.

Findings of the build, each with the script that measured it, are collected for the library's
documentation: bulk writing drowns the record store (on three runs the slow readout alone names the winner 3 points more often on positions it never saw and 2 points more often on positions it witnessed;
`receipts/records_drown.json`), so the school leaves the records empty; a row
of `imagine` is not bitwise invariant to the size of its batch, so the search reads each
distinct position once; record addresses are a similarity kernel, so a write also moves the
sibling moves unless they are anchored at their current values (`receipts/record_address.json`: siblings share 51 percent of their active cells; a plain write moves them 0.29 where the written move goes 0.55, an anchored write 0.04 where it goes 0.71).

A second regime was measured and not deployed (`receipts/sleep-2026-09-22/`, `train_sleep.py`):
records by day, slow parameters by night. A day writes every school position once at slow
rate zero; a night is `RecordPatchNet.sleep` on the day's cues, so the slow readout learns
from the store's own dreams with the school closed. One night takes the readout from chance
to 0.755 (4,096 cells), 0.768 (16,384) and 0.772 (65,536) on the held-out sign, against the
school's 0.785 from the same 263 updates on the outcomes themselves; at every matched update
count the school is one to three points ahead, and the dawn readout follows what the store
read at bedtime. An averaging store (`record_averaging`, floor 0.02) is not drowned by a day
of 67,399 writes: it reads unseen positions at 0.778 in 4,096 cells and lifts the schooled
readout from 0.792 to 0.806 when written on top of it, where the last-writers store lowers
it to 0.765. The deployed brain stays the schooled one.

## Open

- The page does not yet learn from the games played on it. The write path and its
  no-degradation test come first.
- From 16 to 24 stones the search does not always reach the end of the game within its reads;
  a dedicated proof search for the late game is the next step.
- `verify.py` replays the receipt's games and binds it to the deployed files. The solver's grade of each move is stored as
  rates per stage of the game, not move by move, so the blunder rates are not recomputed without the solver.

## Build on it

Fork this directory and make a stronger player. The school, the training, the search, the arena and the page are
separate files, so one of them can be replaced while the rest keeps measuring. A proof search for the late game, a
wider school of early positions and learning from the games played on the page are the open ends.
