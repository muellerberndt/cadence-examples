# Comparisons: the same agent with a conventional world model

The candidate of each stage learns its world model as records over a sparse code, and it
learns from one stream with no replay ring. The research memo
`../../cadence-paper/research/RECORD_PRINCIPLE.md` states what that is supposed to buy and
names the falsifier: a conventional learner that reaches the record learner's held-out
numbers on the same stream at the same number of real transitions. These two runners are
that measurement. Each one runs the stage's life with the world model replaced by an
online multilayer perceptron or an online transformer, under two replay protocols, and
writes one receipt.

## What stays the stage's

Everything except the world model. The arm, the world, the sensors and their bounds, the
tasks and the reward rules, the planner (the beam search of S01 and the A\* search over
imagined consequences of S02), the four declared stores of S02 with the witnessing that
writes them, the curriculum, the budgets, the seeds, the held-out targets, the moving
paths and the held-out suites are taken from the stage packages and its `config.json` at
run time. S02 also borrows the candidate's own scoring, so a consequence counts as correct
under the same rule the stage's receipt uses.

The network sees the reading the candidate's world head sees: every observation field
scaled by its declared bounds, an unobserved value read as zero, the executed action, and
on S02 one missing flag per field, the goal and the reads of the stores. On S01 the goal
is left out, which is what the arm's own MLP baseline does. Imagined observations carry
the reads of the real moment, the way the candidate's imagination carries its recall.

## The four arms

| arm | world model | replay ring |
|---|---|---|
| `mlp` | tanh perceptron, Adam | off |
| `mlp_replay` | tanh perceptron, Adam | one stored transition per witnessed one |
| `transformer` | two pre-norm layers, width 64, four heads, Adam | off |
| `transformer_replay` | the same transformer | one stored transition per witnessed one |

The ring holds 4096 raw transitions and draws uniformly from the transitions stored before
the current one, which is the protocol of `arm.brain.ModelController`. The candidate runs
with the ring off, as `arm/config.json` and `world/config.json` record. A replayed
transition is one more update of the same shape, so an arm with the ring makes two updates
per witnessed transition.

The perceptron with one hidden layer, no flags, no goal and no store reads is
`arm.brain.OnlineMLP`: `comparisons/tests/test_networks.py` runs both on the same stream
and holds them to the same numbers. The transformer carries one token per observation
field, one for the goal, one for the action and one for the stores' reads. A token is a
linear embedding of the field's entries plus a learned field embedding. Each prediction
field is read from the pooled token by its own linear row, and that readout starts at
zero, so the first predictions are uniform classes and mid-range values.

## What is measured

S01, per arm: held-out reaching success on the stage's own held-out targets, the held-out
and babbling prediction error of each predicted field relative to predicting zero, the
copier tracking error on fresh moving paths, success after every 1000 real decisions, and
wall-clock per decision. Held-out reaching and the copier each run on their own copy of
the reaching checkpoint, and both copies keep learning with exploration off, which is the
protocol the stage's own evaluation uses. The learning curve evaluates a copy with
learning off, so the probe never teaches the arm it measures.

S02, per arm: held-out consequence accuracy on the five gated fields, request success at
delay 32, visible correction, the cue at delays 8 and 32, grounding, door recovery from
the life, and wall-clock per decision. The held-out suites run on a copy with learning
off, in a world with the same map and the same rules, which is the stage's held-out
protocol.

## Commands

From the repository root, with the core's interpreter:

```bash
PY=../cadence/.venv/bin/python
$PY -m pytest comparisons/tests -q
$PY -m ruff check comparisons

$PY comparisons/arm.py --seeds 0 --pilot --workers 2 --out runs/comparisons/arm/pilot
$PY comparisons/world.py --seeds 0 --pilot --workers 2 --out runs/comparisons/world/pilot

$PY comparisons/arm.py --seeds 10 11 12 13 14 --workers 2 --out runs/comparisons/arm/acceptance
$PY comparisons/world.py --seeds 10 11 12 13 14 --workers 2 --out runs/comparisons/world/acceptance
```

`--pilot` divides the stage's budgets by ten, as the stage's own pilot does, and divides
the curve interval with them. `--arms` runs a subset, for example `--arms mlp
transformer`. `--workers` is the number of processes; each one runs torch on one thread.
Each run writes `summary_seed<seed>.json` with every arm of that seed and one
`receipt.json`. No event log is written, so a summary and a receipt are all a run leaves.

## Reading the receipt next to the stage's

The comparison receipt is the stage's receipt with the arms in place of the seeds. Open it
next to `runs/arm/acceptance/receipt.json` or `runs/world/acceptance/receipt.json` and
read the same metric names:

- `metrics.heldout_success` of S01 and `metrics.request_success_32`,
  `metrics.consequence_accuracy`, `metrics.cue_delay8`, `metrics.grounding` and
  `metrics.door_recovery` of S02 carry the candidate's numbers in the stage receipt and
  one entry per arm here, each with `per_seed` and `mean`.
- The stage receipt's `controls.online_mlp` and `controls.online_mlp_replay` are the arm
  stage's own perceptron controls. The `mlp` arm here is that learner on that stream, and
  the two agree to the last digit: on the pilot seed both report a babbling error of
  4.805685e-05 on the hand acceleration and 0.00 held-out success. The `mlp_replay` arm
  drifts from `controls.online_mlp_replay`, because the stage's control draws its replay
  index from the generator that also draws its exploration, so a replayed transition moves
  the actions that follow. The ring here draws from its own generator, which holds all
  four arms on one stream.
- `sources.stage_under_test.stage_files` holds the hashes of the stage files that ran.
  They match the stage receipt's `sources.stage_files` when both runs used the same stage
  code. When they differ, the two receipts do not describe the same stage.
- `acceptance.predicates` holds one predicate, `runs_complete`. A comparison states
  numbers and gates nothing, so the stage's own gates stay where they are.
- `networks` holds every network's configuration and its parameter count, and
  `resources.latency_ms` the wall-clock per decision of each arm. The candidate's
  `resources.latency_ms` in the stage receipt covers its controls as well, so the two
  latencies describe different populations of decisions.

## Files

- `networks.py`: the two world models behind one interface, the reading they encode, the
  replay ring and the per-field loss.
- `common.py`: the arms, the job pool, the timings and the summaries.
- `arm.py`, `world.py`: the two runners.
- `config.json`: the network configurations, the reading of each stage, the curve of S01
  and the assumptions the run records.
- `tests/test_networks.py`: shapes, one step of learning, the ring, the batched
  imagination of the transformer, and the perceptron against the arm's own baseline.
