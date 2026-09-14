# Connect Four: compare futures before acting

From the repository root, run `python serve.py connect-four`. A browser opens
the standalone `/connect-four/` website; no installation or GPU is needed.

Play a column as coral; Cadence replies in mint. Turn on **Watch before moving**
to hold its reply while inspecting candidate continuations with the future-ply
slider. **Play Cadence’s move** executes its preferred candidate. The real board
stays unchanged during inspection. The activity circuit remains beside the board
on desktop and stacks below on mobile.

## Its task brain

1. The 42 board ports record observed cells. A supplied readout counts threats,
   pairs and center occupancy.
2. Five feature owners feed one graded value owner. The browser evaluates its
   exact two-step formula inside search; tests compare it with Python Cadence.
3. A bounded alpha-beta search copies legal boards, alternates players, checks
   terminal outcomes exactly and retains only completed search depths. Candidate
   records show seven predicted action values.
4. Six monitor owners read changes in those candidate values, score ambiguity
   and budget pressure. Their output can extend the normal four-ply search to
   six plies, subject to an 80,000-node budget. Toggle **Self-monitor** to test
   its causal role. More thought uses more computation.

There are **61 displayed ports and 11 weighted seams**. Supplied board records
and candidate scores are ports, not hidden biological neurons. Search, feature
extraction and rules are explicit application operations. No weights train during
this game; imagined outcomes are not presented as observed rewards. The search
runs in a browser worker so interaction and circuit inspection remain responsive.

The optional core [`cadence.brains`](https://github.com/muellerberndt/cadence/blob/main/docs/brains.md)
provides generic isolated-future comparison, sensor/motor wiring and the same
activity-monitor design. This page specializes the generic branching pattern to
game search with pruning and iterative deepening. It is deliberately wired
deliberation; recurrence alone does not guarantee planning.

## What has been tested

Run `node connect-four/benchmark.mjs`. The [receipt](evidence.json) retains every
scheduled game, position count and source hash:

- Chosen moves agree with independent exhaustive three-ply evaluation on 20
  fixed-seed positions.
- Adaptive search wins **8/8** games against its one-ply control.
- Against fixed four-ply search it wins **5/8** and loses **3/8**. The same
  evaluator is used on both sides, across four openings with sides swapped.
- On the empty board, monitoring extends depth from 4 to 6 and visited positions
  from 800 to 9,192. This verifies allocation of extra work, not a speed benefit.

Conventional minimax with the same evaluator can implement this behavior, and a
transformer can also be connected to search. This is a small strategy controller,
not solved Connect Four or evidence of superiority over trained game engines.
The monitor demonstrates self-reading control, not subjective experience or a
validated consciousness mechanism. See the [comparison contracts](../ADVANTAGES.md).
