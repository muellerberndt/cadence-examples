# S01: learn an arm, then copy motion

A two-link planar arm learns its own body from motor babbling, reaches targets with a
supplied search over its learned model, tracks fresh moving targets, adapts to a longer
upper link without any reset and returns to its original body. Every decision and its
outcome go through the S00 experience contract.

## Supplied and learned

Supplied: the arm dynamics and the forward kinematics that produce the sensors and the
rendering, the reward rule, the sensor encoding with its fixed bounds, the planning
objective (a clipped velocity field toward the target, evaluated through the learned
model's predicted displacement), the beam search that imagines every torque pair in one
batched settle and holds each candidate for three decisions, the replay ring schedule,
and the settling schedule with its rejection guards.

Learned from random synapses: the workspace and dynamics synapses and the three prediction
heads (hand acceleration, joint velocity change, joint angle change). No inverse map,
Jacobian or target-action pair enters the brain. The Jacobian-transpose PD controller and
the online MLP forward model are baselines outside the candidate's imports; the MLP
receives the same planner and the same replay budget.

## Files

- `env.py`: the arm, its sensors, the moving-target paths and the analytic Jacobian used
  by baselines and tests only.
- `brain.py`: the agent configuration, imagined-consequence composition, the planner, and
  the three baselines.
- `run.py`, `verify.py`: the life runner with its controls and predicates, and the
  independent verifier that recomputes success and prediction gain from the event logs.
- `tools/parity_fixture.py`: a snapshot and a recorded stream for the browser engine.
- `web/`: the page that continues a life in the browser and shows the whole brain.
- `tests/`: integrator replay, observation alignment, clipping flags, target split,
  success and truncation, kinematics and paths.

## Commands

From `cadence-paper`, with the core's interpreter:

```bash
PY=../cadence/.venv/bin/python
$PY -m pytest arm/tests -q
$PY arm/run.py --seeds 0 1 --pilot --out runs/arm/pilot
$PY arm/run.py --seeds 10 11 12 13 14 --out runs/arm/acceptance
$PY arm/verify.py runs/arm/acceptance
```

## Assumptions recorded in config.json

Decisions at 10 Hz (a torque pair is held for five 50 Hz body steps); the sensors report
the measured displacement, its change, the velocity change and the angle change over the
last decision; the prediction targets are the next decision's values of those sensors;
the displacement bound is 0.12 arm lengths; a bounded ring replays three stored
transitions per real transition, because the arm's stream is so autocorrelated that every
online learner, the MLP included, diverges without it.

## Evaluation copies

Each checkpoint is evaluated twice on the same held-out targets. A read-only copy reports
the competence at the moment of the checkpoint (`heldout_success_frozen` in the receipt).
A copy that keeps learning with exploration off, one held-out target after another, is
the gated measure (`heldout_success`): the world head keeps repairing itself in the
agent's one operating mode. Both copies are isolated checkpoint copies; the saved parent
life is never modified, as the plan requires. The MLP baseline is evaluated the same way. Lesions (shuffled pairing, corrupted
dynamics) are compared on the first fifty held-out targets of adapting copies, before
relearning can mask the damage; the born-frozen control is read-only by definition. The
copier tracking runs on an adapting copy as well.
