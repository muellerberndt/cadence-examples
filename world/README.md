# S02: a world that must be remembered

A 6 by 6 world with walls, doors, four movable objects, colours and words, permuted per
life. The agent sees only its cell and, after an inspect, the four adjacent cells. One
life learns the consequences of its actions, remembers where it saw an object once,
corrects that memory when the object moves in view, holds a cue across a delay, adapts
to doors that stop opening, and grounds words in objects through ambiguous scenes.

A request is a new episode of the same life: after the wander the agent finds itself in
a distant cell and the goal names the object; the deadline is twice the shortest path
plus two. The stores persist across episodes, the context trace does not, so only a
remembered place is reachable in time. The cue decision is the plan's two-choice
diagnostic in a new episode at the junction with the cue hidden: north or west, one of
them paid according to the cue seen before the wander.

## Supplied and learned

Supplied: the world, its tasks and reward rules, the categorical sensor encoding with one
missingness flag per field (the plan's sixteen fields plus the displacement of the last
action), four stores with declared categorical keys and their write rules (object to place
with a known flag; cell to its passages, the witnessed map; the last word seen, held until
the next; word to referent co-occurrence), an A* search over cells through the learned
model's imagined consequences toward a remembered or exploration target with at most 128
expansions, and the settling schedule with its guards. The cue decision is one-step reward
greed over the model's reward prediction.

Learned: the records of the world head (a records cortex: the reading minus its running
mean, a fixed sparse expansion with lateral inhibition, delta-rule records for eleven
prediction fields: the displacement, the four passages, the object and colour in the cell,
the carried object, the outcome, and the reward), the critic, and the contents of the four
stores, which are witnessed evidence. There is no replay ring: a reading touches few records
and the outcome is written into exactly those. The research memo
`../../../research/RECORD_PRINCIPLE.md` states the principle and its receipts.

Controls: erased place records, erased word associations, shuffled action pairing, a
reward-shifted cue pairing, the identical architecture frozen from birth, random actions,
the privileged exact planner, and a tabular model behind the same stores and search.

## Files

- `env.py`: the world; private state stays behind underscores; `state` and
  `shortest_path` serve tasks, baselines and evaluation only.
- `tasks.py`: the curriculum; `brain.py`: the agent, stores, imagined consequences and
  search; `baselines.py`: the controls outside the candidate's imports.
- `run.py`, `verify.py`: the life runner and the independent verifier.
- `tests/`: reachability and per-life permutation, visibility, doors, carrying,
  truncation, deterministic replay, the request episode, the two-choice cue decision.

## Commands

```bash
PY=../cadence/.venv/bin/python
$PY -m pytest world/tests -q
$PY world/run.py --seeds 0 1 --pilot --out runs/world/pilot
$PY world/run.py --seeds 10 11 12 13 14 --out runs/world/acceptance
$PY world/verify.py runs/world/acceptance
```
