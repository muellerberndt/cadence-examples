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
18%. On 200 trap positions whose only sound column is refuted between four and more than ten
plies later (`bench/lookahead.py`, stratified by that horizon), the brain keeps the result in
89.5% at eight plies and 87.5% at four, against 57.0% for minimax at 1,024 nodes and 50.5% for
one-ply: the records carry some of what the search cannot reach. The browser pays about 40 ms
per move for the deeper search.

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

## 14. The evaluator was the disease, and threats are the cure

The brain that lost to a human on the page carried records that already knew more than its
search could read: with the same records, frozen, at the same depth of eight plies, reading the
threats standing on a board off the learned complete field (a window one cell short of a line,
its empty cell resting on a stone or the floor) took the paired score against the solver at 70%
from 0.825 to 0.950, the blunders over not-lost moves against one-ply from 11.8% to 7.2%, and the
200 traps from 0.895 to 0.915 with the rows that fall to each side counted too. Inside the search
the same reading decides children before they are expanded, a child the other side can complete
a line on and a child on which the mover holds two reachable threats, and the search sees two
plies further for the same budget. The rule that turns the threats into a value is supplied, like
the search; the threats are read off the records.

## 15. A proven result must not share a tie band with a heuristic read

Root values within 0.01 of the best count as equal and the column nearest the middle is played,
which is what keeps the centre when the opening reads are flat (after 3,3,3,3 only 3 wins, and the
records read 0.488 against 0.487). A proven win of 1.0 and a threat-decided leaf of 0.99 fell
within that band, and the middle column threw a won endgame at ply 30. The band now applies only
when nothing is proven. Two constants collided the same way: a proven result ten plies deep at
1 - 0.001 * 10 equalled the 0.99 of a decided leaf and stopped the deepening; the bonus per ply
is 1e-4 now.

## 16. Proofs remember, but they cannot learn an opening

A memory of every position a search proved, read back at every visit, lengthened the losses to
the perfect solver from 16 plies to 26 to 34 and did not remove them: proving an opening move lost
needs every alternative refuted, and that is exponential. What the proof memory is good for is
the end of the game and the lines the brain has already lost once.

## 17. Watching perfect games is worth more than playing them

On every board as its mover saw it, the brain counts the columns played by the side that went on
to win. Played as the second player against the perfect solver for 150 games, that memory took
the brain, first against the perfect solver, from 1 win in 30 to 1 in 30 (the lines diverge).
Watched: 300 games of the solver against itself, at 0.1 s a game, took it to 14 wins, 4 draws and
2 losses in 20. The tutored form, where the watched first player plays the memory's column
where it has one and the perfect column where it has none so that every game extends the
brain's own lines, took it to 29 of 30 with the full memory and 30 of 30 with the page's cut of
306,313 boards after 195,000 games, two rounds. The brain never sees a score: it sees boards and
who won.

## 18. What the school buys on each side

Thirty-game sets at the page's settings, ten plies within 131,072 imagined transitions and
sixteen within 1,048,576 from eighteen stones on. Moving first: 30-0-0 against the perfect solver,
30-0-0 at 90%, 30-0-0 at 70%. Moving second, where the first player wins with perfect play:
0-0-30 against the full solver, 17-3-10 at 90%, 29-1-0 at 70%, 29-1-0 at 50%. A first player who
plays the perfect column nine times in ten is far beyond human play; the losses to it come from
positions off the remembered lines, where the opening reads are still flat.

## 19. The memory travels beside the page

467,501 boards with the columns winners played and 60,000 proven positions make 8.9 MB packed five
cells to a byte, 3.1 MB gzipped, and install in about a second; the page's cut of 306,313 boards
and 40,000 proofs, chosen by fewest stones, plays as well as the whole. The inlined checkpoint
stays 1.5 MB and the memory is fetched after the page is up, from the release beside the
checkpoint snapshots.

## 20. The page is measured in the browser, and the browser found two things

The numbers of finding 18 came from the page's brain run through Node. Played on the deployed
page in headless Chromium, the opponent's columns clicked on the board and every brain move graded
by the solver, 170 games found two defects. First, the page kept learning from the games played on
it, and a session of games against the solver rewrote enough of the value records that the brain
threw three won positions it plays correctly when fresh: once against AlphaZero, once against the
solver at 90%, once at 70%. The page now freezes the records and the winners' memory as loaded and
keeps only what its searches prove. On the same seeds after the change: AlphaZero, both models,
both sides, 40-0-0; the solver moving second at 70% 18-1-1 (it was 17-2-1). Second, the page took
moves before its 6.6 MB memory was installed; it now waits for it.

## 21. What the school does not reach yet

The remaining thrown positions are reproduced by a fresh brain and none is in the whole school
memory of 866,451 boards. Along the brain's own second-player games the memory covers every
position through the ninth stone, 80% at the thirteenth, 65% at the fifteenth and seventeenth and a
third at the twenty-first, and the throws fall exactly where it thins: positions where one column
wins, every other loses, and the records read the columns within a few hundredths of each other.
Searching deeper does not recover them in a browser's time: at eighteen plies, up to a minute a
move, four of eight. A school of 200,000 games with an imperfect first player, and 60,000 with an
imperfect second player, is running to push the coverage along those lines.

## 22. A bigger memory makes the first player perfect and leaves the second player's tail

Watching 260,000 more games (five schools with the first player perfect at 60% to 90% of its
moves, two with the second player at 70% and 85%) took the merged memory from 866,451 boards to
2,189,619. A page cut of 1,007,760 boards (every board seen in two games or more, then by fewest
stones) with 72,380 proven positions, 19 MB and 6.5 MB compressed, played on the page against the
same seeds: moving first 30-0-0 against the perfect solver with every move perfect, 20-0-0 at 90%,
no blunder against either AlphaZero; moving second 19-0-1 at 70%, and at 90% over 60 games 28-7-25
against 32-7-21 for the smaller memory, a difference inside the spread of three seeds (8-4-8 to
15-2-3 for one build), with its extra throws made identically by the smaller memory. Along the
brain's second-player games the coverage past the thirteenth stone rose by three to seven points
for two and a half times the boards: watched games open the positions an imperfect first player
reaches too rarely for memory to close the tail.
