# 03 · Connect Four

A patch net imitates a teacher, plays games, and revisits teacher examples where
its actions disagree. In the browser it can also compare possible futures using
four-ply search. Toggle **Think four moves ahead** to compare that player with its
raw learned policy.

```bash
# From the repository root
python -m pip install -r requirements.txt
python serve.py connect-four
python serve.py connect-four --learn  # save lessons and resume learning locally
```

The board supplies two 42-cell planes: the current player's discs and the
opponent's. Hidden and output owners communicate through reciprocal seams;
free/nudged local updates learn seven action scores. A fully visible board needs
no temporal image buffer. Illegal moves are masked.

## Learning and deliberation

Training compares 64 and 128 hidden owners on validation data, then trains the
selected size. A depth-4 search labels the demonstrations. Equivalent boards and
their reflections stay together across splits, preventing mirrored test leakage.
The practice stage plays 30 games against a depth-2 opponent, asks the teacher
about disagreements, and rehearses earlier training examples. Those games start
from the empty board; repeated deterministic paths limit their diversity.
Held-out boards and their reflections are excluded from corrective teaching.

Deliberation is an architectural pattern. The browser's supplied game simulator
branches legal moves; a supplied threat heuristic evaluates futures. Learned
scores break ties between equally valued moves. This is a strong playable
composition, but its strength cannot be attributed entirely to the neural policy.
The receipt reports three separate players: raw policy, policy plus search, and
search alone, with identical search budgets for the last two.

Cadence also provides a tested example with a **learned terminal evaluator**:
[Comparing possible futures](https://github.com/muellerberndt/cadence/blob/main/docs/deliberation.md).
Neither example claims to learn the transition model.

With `--learn`, completed human games provide new positions for teacher correction
and rehearsal. Episodes and checkpoints persist under `runs/learning/`. The server
checks retention on initial rehearsal examples before accepting an update. See
[Training a player](../TRAINING.md) for the complete lifecycle and its limits.

## Measured play

<!-- game-results -->
Wins / draws / losses, 100 games per opponent:

| player | random | depth 2 | depth 4 |
|---|---|---|---|
| raw policy | 95 / 0 / 5 | 1 / 3 / 96 | 5 / 5 / 90 |
| policy + depth-4 search | 100 / 0 / 0 | 56 / 9 / 35 | 57 / 9 / 34 |
| depth-4 search alone | 100 / 0 / 0 | 51 / 11 / 38 | 45 / 21 / 34 |
<!-- /game-results -->

## Reproduce

```bash
cd 03_connect_four
python dataset.py
python train.py --output receipt.json
python ../tools/rehearsal.py
python build_page.py
```

The full run takes tens of minutes on a laptop. It writes the model, learner
checkpoint and a source-bound receipt. The shared rehearsal helper extracts
training-only anchors for later browser lessons. Results against random,
depth-1, depth-2 and depth-4 opponents include wins, losses and draws, alternating
who starts after random two-ply openings. The root [results table](../README.md) is generated from the receipt.
MLPs receive the same labelled board dataset; their teacher agreement and raw play
strength are recorded separately from the planning players. The patch net also
receives the practice corrections described above; that extra teaching is not
matched in the MLP runs.
