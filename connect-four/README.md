# Connect Four: compare futures before acting

From the repository root, run `python serve.py connect-four`. A browser opens
the standalone `/connect-four/` website; no installation or GPU is needed.

Play a column as coral; Cadence replies in mint. Turn on **Watch before moving**
to hold its reply while inspecting candidate continuations with the future-ply
slider. **Play Cadence’s move** executes its preferred candidate. The real board
stays unchanged during inspection. The activity circuit remains beside the board
on desktop and stacks below on mobile.

## Thought continues between turns

**Think between turns** is on by default. While you choose, the reasoner considers
possible human moves and its own replies. You can play immediately; hypothetical
stones never enter the real board. The shared value/choice/monitor circuit updates
when a search depth completes. Once the depth or node budget is reached, its state
remains available while computation waits for a new observation.

The worker yields every 128 search events, so a human move cancels obsolete work
at the next slice. Exact results keyed by board, side to move and remaining depth
survive across moves in a cache capped at 20,000 entries. A new game clears that
cache and the monitor. Turning background thought off stops between-turn work;
Cadence still thinks when it must play. Changing depth or monitoring starts a new
search under those controls.

**Reused positions** counts exact cache hits in the current search. The position
counter reports current-search work; hover over it for the entire game's count,
including canceled and background thought. Background candidates are scored for
the human side to move; during Cadence's turn they are scored for Cadence.

On the empty board followed by a human center move, pondering visits **9,192**
positions. Reuse then needs **12,630** positions versus **13,955** for a fresh
response, with the same depth, candidate scores and choice. Total work with
pondering is **21,822**, so this example demonstrates preparing a response earlier,
not a reduction in total computation. The [receipt](evidence.json) records both costs.
This cache holds computed hypotheses; it is separate from learned synaptic memory.

The [runtime tests](pondering.test.mjs) cover bounded slices, board isolation,
cancellation, cache limits, score equivalence and an actual worker receiving a
replacement observation mid-search. Browser tests verify default pondering, usable
human controls, pausing, cache reuse and no speculative board moves.

## Its task brain

1. The 42 board cells record the observed environment. A supplied readout counts threats,
   pairs and center occupancy.
2. Five feature neurons feed one graded value neuron. The browser evaluates its
   exact two-step formula inside search; tests compare it with Python Cadence.
3. A bounded alpha-beta search copies legal boards, alternates players, checks
   terminal outcomes exactly and retains only completed search depths. Candidate
   records show seven predicted action values.
4. Six monitor neurons read changes in those candidate values, score ambiguity
   and budget pressure. Their output can extend the normal four-ply search to
   six plies, subject to an 80,000-node budget. Toggle **Self-monitor** to test
   its causal role. More thought uses more computation.

There are **19 circuit neurons and 39 weighted synapses**. The six evaluator,
seven candidate and six monitor neurons settle jointly; their state selects the
move and further-work request. The board remains an external observation. Search, feature
extraction and rules are explicit application operations. No weights train during
this game; imagined outcomes are not presented as observed rewards. The search
runs in a browser worker so interaction and circuit inspection remain responsive.

The optional core [`cadence.circuits`](https://github.com/muellerberndt/cadence/blob/main/docs/patterns.md#with-a-supplied-world-model)
provides resumable `Deliberator` ticks as well as synchronous `imagine`, sensor/motor reflex arcs and the same
activity-monitor design. This page specializes the generic branching pattern to
game search with pruning and iterative deepening. It is deliberately wired
deliberation; recurrence alone does not guarantee planning.

## One shared decision state

The value, candidate and monitor regions exchange activity changes in one graph. Their
joint endpoint supplies the move and extra-search request. The display labels
these functions and reports their common equation residual. Each imagined board
remains an isolated world; only candidate scores enter the shared decision.
[Connectomes and tests](../COUPLED_BRAINS.md) explain the separation. A fixed
point certifies self-consistency, not a globally optimal strategy.

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
