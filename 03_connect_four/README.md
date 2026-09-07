# 03 · Connect Four

A patch net learns to play Connect Four from positions a shallow search has labelled, and
then plays you in the browser, settling in the page for every move. The page is the point
of this rung: you can watch seven output owners come to rest and see which one wins.

```bash
pip install "cadence-net>=0.2" scikit-learn
python dataset.py                  # about 2 minutes: self-play positions labelled by a depth-4 search
python train.py                    # about 15 minutes: selection, training, matches, baselines, receipt
python build_page.py               # embeds net.json into index.html; open it in a browser
python train.py --verify receipt.json
```

## What the net is

The board reaches the net as 84 input owners: one plane of the mover's discs and one of
the opponent's, 6×7 each, clamped at 1 where a disc sits. Hidden owners sit between them
and 7 output owners, one per column. `cadence.layered(84, hidden, 7, density=1.0)` ties
every hidden ↔ output pair into one seam, so a nudge at the outputs reaches the hidden
owners. To move, the net settles under the clamp and plays the most active output among
the legal columns. There is no search and no lookahead in the net; the search was its
teacher.

## Where the positions come from

`dataset.py` plays games between two depth-2 searchers that pick a random column a quarter
of the time, which gives varied, plausible positions rather than a few opening lines. Each
non-terminal position is labelled with the column a depth-4 alpha-beta search prefers for
the side to move (a threat-count heuristic at the leaves, centre first on ties). Positions
are deduplicated; the training rows are mirrored left-right, the test positions are not.
The dataset's digest is in the receipt, and `--verify` checks the file on disk against it.

## How it learns and how it is judged

The same free/nudged rule as the digits example: a free settlement, a nudged settlement
toward the teacher's column and one away from it, and every seam moving on its own
two-phase difference. Selection is by agreement with the teacher on a validation split of
the training rows; the test positions are read once.

Agreement with the teacher is a hard target: on a sample of the positions 84% have a
unique best move at depth 4, so a perfect imitator would score above 0.9. What matters
more is play. The net plays 100 games against each of four opponents, alternating who
starts: a random mover, and searches of depth 1, 2, and 4 (the teacher itself). An MLP of
the selected size, trained by Adam on the same rows, plays the same matches.

## The numbers

From `receipt.json`: 107,637 positions from 8,000 games (193,746
training rows after mirroring, 10,764 held-out positions), validation hidden 64: 0.498, hidden 128: 0.518,
100 games per opponent from random two-ply openings, one laptop core.

| model | parameters | epochs | training | agreement | vs random | vs depth 1 | vs depth 2 | vs depth 4 |
|---|---|---|---|---|---|---|---|---|
| patch net 84-128-7, free/nudged rule | 11,888 | 15 | 1181 s | 0.527 | 91-0-9 | 25-1-74 | 2-0-98 | 9-2-89 |
| MLP 84-128-7, Adam | 11,783 | 15 | 11 s | 0.533 | 93-0-7 | 29-4-67 | 8-8-84 | 2-4-94 |
| MLP 84-128-7, Adam | 11,783 | 50 | 40 s | 0.557 | 94-0-6 | 28-8-64 | 4-4-92 | 8-1-91 |

Records are wins-draws-losses for the net. The shuffled-label control, the same net
trained for five epochs on permuted labels, agrees with the teacher on 0.147 of
positions, which is chance for seven columns.

Read it plainly. The patch net and the MLP learn the same amount from these positions:
about half the teacher's moves, and the MLP with three times the epochs gains three
points. Both beat a random mover and both lose to a two-ply search, because a policy
that agrees with its teacher half the time misses a forced block often enough to lose
almost every game against an opponent that never does. That is what imitation of a
shallow search on a hundred thousand positions buys a one-hidden-layer net, whichever
rule trains it; a stronger player needs search at play time or far more positions, and
this example deliberately has neither. Wall-clock is the usual factor: the settlements
cost a hundred times the MLP's passes.

## The page

`index.html` is self-contained: the dense overlap matrix and biases are embedded, and the
settlement runs in JavaScript, owner by owner, with the same rule. The bars above the
board replay the output owners' activations step by step; the strip on the right shows
the hidden owners at rest. Nothing is precomputed. A published copy is linked from the
top-level README.
