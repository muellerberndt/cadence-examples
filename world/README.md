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
greed over the model's reward prediction. The reading's gains are declared in
`config.json`: the action's neurons at 4.0 and the object-here and carrying fields at
twice the input gain, so readings that differ there share fewer active cells and the
consequence records written under one goal or door rule interfere less with the others.

Learned: the records of the world head (a records cortex: the reading minus its running
mean, a fixed sparse expansion with winner-take-all inhibition, delta-rule records for
eleven prediction fields: the displacement, the four passages, the object and colour in
the cell, the carried object, the outcome, and the reward; the reward records read a
valued code with an equal say for every input pathway and one group of cells per goal, so
a reward earned under one goal is written into cells no other goal reads), the critic, and
the contents of the four stores, which are witnessed evidence. There is no replay ring: a
reading touches few records and the outcome is written into exactly those. The library
page [records](https://github.com/muellerberndt/cadence/blob/main/docs/memory.md#records)
states the mechanism.

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

From the repository root, with a Python that has `cadence-net` installed:

```bash
python -m pytest world/tests -q
python world/run.py --seeds 0 1 --pilot --out runs/world/pilot
python world/run.py --seeds 10 11 12 13 14 --out runs/world/acceptance
python world/verify.py runs/world/acceptance
```

## Receipt

`receipt.json` is the acceptance receipt of seeds 10 to 14: every predicate with its value
and threshold, the source hashes of this directory and of the library, the sha256 of every
event log and checkpoint, and the digest of the receipt itself. `verify.py` recomputes the
request, correction and grounding rates from the event logs and fails closed on an
incomplete run.

## The page

`web/` continues a life of a saved brain in the browser: the world beside the whole brain, the
visitor asking for objects, dragging one to another cell, teaching a word, running the cue
task and locking the doors, and every decision computed by the ported agent while the scan
settles. `python world/web/build_page.py --run runs/world/dev --out runs/world/web/index.html`
inlines the run's final brain and writes the checkpoints of that seed beside the page, so the
brain selector reaches the same life after 1,000, 5,000 and 10,000 explore steps, after the
remembered requests, after the cue task and after the doors locked. `web/README.md` states
what the page shows and how it is checked.
