# 03 · Connect Four

A patch net learns to play Connect Four from positions a shallow search has labelled, and
then plays you in the browser, settling in the page for every move. The page is the point
of this rung: you can watch seven output owners come to rest and see which one wins. Read
[How a patch net learns](../HOW_IT_LEARNS.md) first for the mechanism; this page is about
turning a game into something the rule can learn, and being honest about what it learned.

```bash
pip install "cadence-net>=0.7" scikit-learn
python dataset.py                  # a few minutes: self-play positions labelled by a depth-4 search
python train.py                    # under an hour: selection, training, matches, baselines, receipt
python build_page.py               # embeds net.json into index.html; open it in a browser
python train.py --verify receipt.json
```

## 1. Where the positions come from

There are no labelled Connect Four positions to download, so `dataset.py` makes them:

1. **Games.** Two players who search two plies ahead play each other, and a quarter of
   the time either plays a random legal column instead. The randomness matters: two
   deterministic searchers would play the same game every time. 8,000 games.
2. **Positions.** Every non-terminal position of every game, from the side to move's
   point of view, deduplicated: 107,637 of them.
3. **Labels.** For each position, the column a depth-4 alpha-beta search prefers for the
   side to move. The search sees wins and losses four plies out and scores everything
   else by counting two- and three-in-a-window threats, centre first on ties. It is the
   teacher, and the net can learn at most what the teacher knows.

`connect4.py` is the game on bitboards: a position is two integers, one per side, and a
four-in-a-row test is four shifts. The dataset's digest goes into the receipt, and
`--verify` checks the file on disk against it.

## 2. From a board to a clamp

The board reaches the net as 84 input owners: one plane of the mover's discs and one of
the opponent's, 6 × 7 each, clamped at 1 where a disc sits and 0 elsewhere. Because the
planes are "mine" and "theirs" rather than "red" and "yellow", one net plays both colours.
The 7 output owners are the columns. Training rows are mirrored left-right (and the label
with them), which doubles the data at no cost; test positions are not mirrored.

## 3. The net and a move

`cadence.layered(84, hidden, 7, density=1.0)` with hidden 64 or 128: every board owner
reaches every hidden owner, every hidden owner reaches every column owner, and each
hidden↔column pair is one seam. To move, the net settles under the board's clamp to a
tolerance of 10⁻⁴ (about 30 steps), the full columns are masked, and the most active
remaining column owner is played. There is no search in the net and no lookahead; the
search was its teacher.

## 4. How it learns

This is classification with seven classes, and the mechanism is exactly section 3 of
[How a patch net learns](../HOW_IT_LEARNS.md): for a batch of 64 positions, a free
settlement, a settlement nudged toward the teacher's column (`β = 0.1`, `T = 0.1`) and one
nudged away, and every seam moving on the difference of its own two endpoint products.
Fifteen epochs over 193,746 rows, `η` from 3 decaying by 0.8 per epoch.

Why imitation and not self-play reinforcement learning? Because a search that already
plays well is available and cheap, so the question this rung asks is "can the rule absorb
a teacher's policy into a net that then plays without searching", which is answerable in
an hour and measurable exactly. Learning a game from the win/loss signal alone is the
harder problem and it is what Pong (rung 04) does in a smaller setting.

## 5. Selection, matches, and the control

The hidden size is chosen by agreement with the teacher on a validation split of the
training rows; the test positions are read once. Then the net plays 100 games against
each of four opponents, alternating who starts, from random two-ply openings (the net and
the searches are deterministic, so without the random opening every game on the same
side would be the same game): a random mover, and searches of depth 1, 2, and 4 (the
teacher itself). An MLP of the selected size trained by Adam on the same rows plays the
same matches. A net trained for five epochs on shuffled labels is the control.

## 6. The numbers

From `receipt.json`: 107,637 positions from 8,000 games (193,746 training rows after
mirroring, 10,764 held-out positions), validation hidden 64: 0.498, hidden 128: 0.518,
100 games per opponent from random two-ply openings, one laptop core.

| model | parameters | epochs | training | agreement | vs random | vs depth 1 | vs depth 2 | vs depth 4 |
|---|---|---|---|---|---|---|---|---|
| patch net 84-128-7, free/nudged rule | 11,888 | 15 | 1181 s | 0.527 | 91-0-9 | 25-1-74 | 2-0-98 | 9-2-89 |
| MLP 84-128-7, Adam | 11,783 | 15 | 11 s | 0.533 | 93-0-7 | 29-4-67 | 8-8-84 | 2-4-94 |
| MLP 84-128-7, Adam | 11,783 | 50 | 32 s | 0.557 | 94-0-6 | 30-3-67 | 5-2-93 | 9-0-91 |

Records are wins-draws-losses for the net. The shuffled-label control agrees with the
teacher on 0.147 of positions, which is chance for seven columns. Agreement is a hard
target: on a sample of positions 84% have a unique best move at depth 4, so a perfect
imitator would score above 0.9.

Read it plainly. The patch net and the MLP learn the same amount from these positions:
about half the teacher's moves, and the MLP with three times the epochs gains three
points. Both beat a random mover and both lose to a two-ply search, because a policy that
agrees with its teacher half the time misses a forced block often enough to lose almost
every game against an opponent that never does. That is what imitation of a shallow
search on a hundred thousand positions buys a one-hidden-layer net, whichever rule trains
it; a stronger player needs search at play time or far more positions, and this example
deliberately has neither. Wall-clock is the usual factor: the settlements cost a hundred
times the MLP's passes.

So when you play the page: it will block some threats and miss others, it will take an
open win most of the time, and a patient human beats it. What you are watching is a net
that learned, locally, to reproduce half of a search's judgement.

## 7. What is different from the MLP

The two policies have the same shape, the same numbers, the same data, and the same
agreement. The MLP's move is one pass; the patch net's move is a settlement you can watch
form on the page, bar by bar, and its hidden owners at rest carry the influence of the
column owners as well as the board. Section 6 of [How a patch net learns](../HOW_IT_LEARNS.md)
has the comparison.

## 8. The page

`index.html` is self-contained: the dense overlap matrix and biases are embedded, and the
settlement runs in JavaScript, owner by owner, with the same rule. The bars above the
board replay the output owners' activations step by step; the strip on the right shows
the hidden owners at rest. Nothing is precomputed. A published copy is linked from the
[hub](https://claude.ai/code/artifact/14644daf-1a2f-47b3-8c2c-f896c5ca3c60).

## 9. Things to try

- `dataset.py --games 20000` for more positions; watch agreement and the depth-2 record.
- `LABEL_DEPTH = 6` in `dataset.py` for a stronger teacher (slower to label).
- Add a one-ply tactical check at play time (take a win, block a loss) and see how the
  records change; then note that the net is no longer the whole player.
