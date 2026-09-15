# S03 brain interface

`run.py` imports `Brain` from `brain.py` and drives it through the calls below. The brain
receives moments and nothing else: no board object, no step function, no terminal oracle.
`tests/test_env.py` parses `brain.py` and fails when it imports `evaluate`, `controls`,
`opponents`, or the engine class `Board` (by name or attribute). The encoding helpers of
`env.py` (`GameConfig`, `fields`, `cells_of`, `observation`, `goal_vector`, `CELL`,
`MOVER`, `TURN`) are interface knowledge and may be imported.

## Construction

```python
Brain(config: dict, seed: int, *, learning: bool = True)
```

`config` is the whole `config.json`. The runner reads `game`, `planning`, `training`,
`budget`, `evaluation`, `mixture`, `adaptation_mixture`, `controls`, `gates`, `seeds`; the
brain adds and reads its own sections (graph, world head, actor, records, replay, planner
parameters). `config["game"]` sizes the ports: `rows * cols` cell fields of width 3, the
`mover` and `phase` fields of width 2, a goal of width 3, `cols` actions. Tests may build the
brain with `config["debug_game"]` (4 by 4, connect 3: fields `c00..c15`, 4 actions).

`seed` is the brain seed. Derive every generator from `seed_for(STAGE, split, seed,
environment_id, stream_id)` (`..s00.life`); never share a generator with the world.
`learning=False` is the born-frozen control: the same architecture, random parameters,
no update from birth.

Optional class attributes `SUPPLIED: list[str]` and `LEARNED: list[str]` name the brain's
supplied and learned components; the runner copies them into the receipt.

## Moments

Every moment is an S00 `Moment` (`..s00.life`). Observation fields (all observed flags
True):

| field | width | meaning |
|---|---|---|
| `c00`..`c41` | 3 | one cell each, row-major from the bottom-left (`index = row * cols + col`, row 0 at the bottom): empty, self, opponent; relative to the candidate whichever side it plays |
| `mover` | 2 | whose move produced this board: candidate, opponent (the opening board of a game the candidate opens counts as the opponent's) |
| `phase` | 2 | the candidate has no stone on the board yet, or it has moved |

`goal` has width 3: `[candidate to move, opponent to move, candidate opened]`. The first
two are one-hot for the side that moves next by alternation, on terminal boards too, so
the goal never says whether the game is over. The third is 1 through a game the candidate
opened. On real moments the goal is redundant with `mover`; during imagination the brain
sets it to the side whose move it imagines.

`action_mask` has one entry per column; True where a stone can be dropped.

One decision spans the candidate's move and the opponent's reply. In one game the brain
sees, in this order and with event ids increasing by one:

1. The decision moment: the opening board, or the board after the opponent's reply.
   `feedback_for is None`, `reward_known False`, the legal columns in the mask. `step`
   returns a `Decision` whose `action` is a legal column and whose `decision_id` the
   brain chose (the S00 Agent's `next_decision_id`).
2. The intermediate moment, right after the candidate's move: `feedback_for` names the
   decision, `executed` the column, `reward_known True`, `reward` 0, or +1 / 0 with
   `terminated True` when the move won or filled the board. The mask is all False:
   `step` returns `None`.
3. The reply moment: the next decision moment, or, when the opponent's reply ended the
   game, a terminal moment with `feedback_for None`, `terminated True`, `reward` -1 (loss)
   or 0 (draw) and `reward_known False` (the S00 contract ties a known reward to an
   executed decision). `step` returns `None` on terminal moments.

Evaluation probes start an episode at a saved position (`World.start_from`): a decision
moment with `tick 0`, then the intermediate moment, then, unless the move ended the game,
a truncated moment (`truncated True`, `final_observation` set, mask all False) that closes
the episode; `step` returns `None` on it. A new game is a new `episode_id`; event ids
increase over the whole life; ticks count moments of an episode.

The S00 `Agent` refuses a continuing moment without a legal action and a moment without
feedback while a decision is pending, so the brain cannot pass the intermediate moment to
the Agent as it is. One way that satisfies the Agent's validation: keep the intermediate
board for world-model repair (the candidate's own transition), and hand the Agent the reply
moment merged with the decision's feedback,
`Moment(**{**reply.__dict__, "feedback_for": id, "executed": column, "reward": outcome,
"reward_known": True})`, where `outcome` is the reply moment's reward; when the intermediate
moment is terminal, hand the Agent that moment itself. Store each observed transition with
its mover so the candidate's action is not blamed for the opponent's move.

## Methods the runner calls

```python
step(moment: Moment) -> Decision | None
```
A `Decision` on moments with legal columns, `None` otherwise. The runner times this call on
decision moments; p95 must stay at or below one second (`gates.latency_p95_ms`) at the
declared budget. `Decision.budget` may carry `{"expansions": n}` (imagined transitions of
this decision); the runner sums it. `Decision.controller` is logged by its first letter.

```python
frozen() -> Brain
```
An isolated read-only copy: learning off, no shared mutable state with the live brain,
fresh event cursors (the S00 `Agent.frozen()` attaches its stream afresh), and every method
of this interface working on the copy. The live brain keeps learning while copies are
evaluated; a copy is used for thousands of probe episodes and paired games.

```python
set_epsilon(epsilon: float) -> None
```
The exploration rate of decisions: 1.0 during the random-legal-games phase, the
`training.epsilon` during the mixtures, 0.0 before checkpoints and on copies.

```python
snapshot_policy() -> Callable[[np.ndarray, np.ndarray], int]
```
A saved policy of the current parameters: `policy(cells, legal) -> column`, where `cells`
is the board as the mover sees it (0 empty, 1 the mover's stone, 2 the other's; the mover
is the policy's own side) and `legal` the mask; it returns a legal column. It decides
without search or imagined transitions (the actor's or one-step readout), it never learns,
and it does not change when the live brain does. It is both the snapshot opponent of the
mixture and the no-planning control of the planning-gain gate.

```python
predict_board(observation, column, *, goal=None) -> np.ndarray
```
The imagined next board: an integer array of `rows * cols` values in {0, 1, 2}, relative
to the candidate, after the side that moves next (the other side of `mover`, equivalently
`goal[:2]`) drops a stone in `column`. Judged exactly on 10,000 held-out moves stratified by
column height and mover; changed-cell accuracy is reported separately.

```python
predict_terminal(observation, *, goal=None) -> np.ndarray
```
Scores over `TERMINAL_CLASSES = ("nonterminal", "win", "draw", "loss")` from the
candidate's view; the argmax is judged on 2,000 boards, balanced accuracy over the four
classes. Targets during learning must come from actual outcomes.

```python
corrupt_dynamics() -> None
```
Called on a frozen copy: corrupt the learned transition model or its action mapping while
the search and everything else stay. The tactical suite must fall by at least
`gates.corrupted_dynamics_loss`.

```python
agent  # the S00 Agent: the runner reads .ledger.to_dict(), .parameters(), .save(path)
planner  # search statistics read with getattr: nodes, searches, invalid, depth, budget
```

## Budgets and rules

- Planning starts at depth 2 with 256 model transitions per decision and extends to depth
  4 with 1,024 only when rollout validity supports it (`config["planning"]`). Search reads
  learned transitions and terminal estimates; an imagined board the brain's own validator
  rejects terminates that branch with uncertainty and is counted (`planner.invalid`); the
  validator may not repair it into the correct board.
- No correct-move labels, no minimax move labels, no engine calls. The legality mask is
  interface knowledge.
- If the actor chooses moves, use the legal-mask policy and eligible reward updates; if the
  planner chooses, learn world and value from real outcomes and report planner moves as
  planner moves (`Decision.controller`).
- Weight changed-cell targets: predicting no change scores well on unchanged cells.
- Checkpoints (`agent.save`) are taken at transaction boundaries (between games).

## The runner's schedule per seed

explore_games random legal games (epsilon 1, random opponent), then mixture_games against
the mixture (a snapshot when the mixture starts and every snapshot_every games), checkpoint,
then on frozen copies: the validity and terminal gates, the tactical suite, paired games
against random, one-ply and minimax-256, the no-planning copy (the snapshot policy) on the
tactical suite and the paired games, the corrupted copy on the tactical suite and the
validity gate; then adaptation_games against the changed mixture without a reset, a second
checkpoint, and the paired games again on a frozen copy. `--pilot` divides the game budgets
by ten.
