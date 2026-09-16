# Connect Four: what we found

The brain on the page learned Connect Four from the games it played, with no rules, no engine
and no labelled moves, and beats an AlphaZero trained with alpha-zero-general's published recipe
in 0.945 and 0.990 of 100 paired games at 25 and 100 simulations per move. This is what the work
that got it there taught us, in the order of what mattered most. Every number is from a receipt
in `bench/receipts/` or `receipt_solver.json`; the methods are in `README.md`.

## 1. The horizon is the largest lever

The brain plans exactly inside its search and badly at its edge. Raising the search from four
plies within 1,024 imagined transitions to eight within 32,768, with the learned records
unchanged, moved the paired score against minimax at 1,024 nodes from 0.875 to 0.955, against
the perfect solver played at 70% from 0.725 to 0.890, and against AlphaZero from 0.730 and 0.800
to 0.945 and 0.990. Blunders in positions not already lost fell from 17% to 27% of moves to 10% to
18%. The browser pays about 40 ms per move for it.

## 2. The demo lost to what it cannot see, not to what it cannot do

The lost game that started this was a double threat prepared three opponent moves ahead. The
brain never misses a win or a block in one; reconstructed from the screenshot, it blocks or wins
in every position where that is possible. It lost because at the edge of its search every leaf
looked the same: the seven moves of a typical middle-game position were valued within 0.02 of
each other.

## 3. Why the evaluation is flat, and what is not the cause

The value table itself is informative: three in a row with an empty cell reads 0.67, the
opponent's three reads 0.30, a completed line 0.96. The flatness comes from combining them. A
board's value is the mean over its 69 windows, and about 65 of them sit near 0.5 on any board, so
a position with two playable threats, a forced win, reads 0.54. Summing the evidence at read time
instead of averaging it does not repair this on the same table (against minimax 0.855 to 0.890,
against the solver at 70% 0.730 to 0.660; a steeper sum is worse on both). The deeper reason is
the key: one window of four cells cannot say whether its empty cell is playable, nor that two
windows share a winning cell. What decides Connect Four is the relation between threats, and no
per-window value can carry it.

## 4. More games on the same table buy nothing

The training curve of the solver curriculum is flat after about 200 games. A 120-game pilot
blundered in 12% of its moves against one-ply; the 1,200-game life of the same configuration
blundered in 19% (seed 0) and 34% (seed 1). A table of 81 window patterns fills in a few dozen
games; after that, more games re-average the same numbers. Capacity has to come before data.

## 5. A graded perfect opponent teaches more than a strong one

Pascal Pons' solver as an opponent, playing the perfect column 30% or 70% of the time and a
random one otherwise, halved the blunder rate against every opponent at a tenth of the games:
0.118 against 0.279 (one-ply), 0.149 against 0.238 (minimax at 1,024 nodes), 0.191 against 0.328
(the solver at 50%). Perfect play alone would make every game a loss and write zero into every
value, which is why the strengths are graded. The solver is an opponent and nothing else; the
brain sees its moves and the outcomes, never its scores.

## 6. Writing the search back into the values helps only when the search is deep

Writing every searched root value into the value records (bootstrapping) made the depth-8 brain
better on every opponent (1.000, 1.000 and 0.850 against one-ply, minimax and the solver at 70%,
against 1.000, 0.940 and 0.785 without it) and made the depth-4 brain worse (blunders against
one-ply 0.226 against 0.118). A bootstrapped target is only as good as the search that produced
it.

## 7. A wider reading needs its own cortex, and more games

Adding to each window's reading whether every cell rests on a stone or the floor, so a playable
threat reads differently from a floating one, first broke the brain: the evidence that a
window is a completed line spread over sixteen variants and seven of the twelve gates failed.
Keeping the completed-line field on the plain reading and giving the value its own cortex over
the 1,296 supported patterns passes every gate. At 120 games it is not better than the plain
table, which is the expected face of sixteen times the capacity with the same data; the
full-length lives are work in progress.

## 8. A deterministic opponent turns 100 games into one

AlphaZero at temperature 0 with one search tree kept across games plays the same game every
time, so the first head-to-head (1.000 for one seed, 0.210 for the other) was one game pair
replayed fifty times. Sampling its first three stones from its visit counts, seeded per pair,
and rebuilding its tree every game gives games that differ; the numbers above come from that.

## 9. Blunder rates must be conditioned on positions not already lost

Against the perfect solver almost every position is lost, and in a lost position no move can
change the result, so a raw blunder rate of 4% there says nothing. The rate reported everywhere
here is over the positions where a blunder was possible.

## 10. AlphaZero at the published recipe is a weak opponent, and the claim has to say so

alpha-zero-general's Connect Four with its published recipe, 32 iterations of 100 self-play games
at 25 simulations per move, scores 0.738 against one-ply and 0.688 against minimax at 1,024
nodes; that minimax scores 0.375 and 0.260 against it. There is no public pretrained Connect
Four AlphaZero, so "beats AlphaZero" means this model at this budget. A stronger one, 120
iterations at 50 simulations, is training as the held-out opponent.

## 11. Two seeds are not a measurement

The same curriculum gave 0.875 and 0.675 against minimax at 1,024 nodes for seeds 0 and 1. The
stage's acceptance runs five seeds and gates on the minimum; nothing here is claimed from one.

## 12. A perfect solver is a cheap grader

Pons' solver with its opening book scores 541 positions of random play in 0.4 s, so every move of
every game can be graded optimal, slip or blunder. Handing it a board with the wrong side to move
hangs it: the parity of the stone count is part of the position.

## 13. Receipts bind source bytes

The stage receipt hashes `brain.py`, `opponents.py` and `run.py`; changing any of them, even
behind a switch that is off, invalidates the receipt and blocks the published gallery until the
acceptance runs again against the new bytes. That is the point of the receipt.
