# S01: learn an arm, then copy motion

A two-link planar arm learns its own body from motor babbling, reaches targets with a
supplied search over its learned model, tracks fresh moving targets, adapts to a longer
upper link without any reset and returns to its original body. Every decision and its
outcome go through the S00 experience contract.

## Supplied and learned

Supplied: the arm dynamics and the forward kinematics that produce the sensors and the
rendering, the reward rule, the sensor encoding with its fixed bounds, the planning
objective (a clipped velocity field toward the target, evaluated through the learned
model's predicted displacement), the beam search that imagines every torque pair through
the records and holds each candidate for three decisions, and the settling schedule of the
settled regions with its guards.

Learned from empty records and random synapses: the records of the world head (a records
cortex over the sensory, goal and action neurons: the reading, a fixed sparse expansion
with winner-take-all inhibition, delta-rule records for the three prediction fields: hand
acceleration, joint velocity change, joint angle change) and the settled regions' synapses.
No inverse map, Jacobian or target-action pair enters the brain. There is no replay ring:
a reading touches few records and the outcome is written into exactly those. The
Jacobian-transpose PD controller and the online MLP forward model are baselines outside the
candidate's imports; the MLP runs twice, under the candidate's replay policy (none) and
with its own ring of 4,096 transitions, so the comparison is read both ways.

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

From the repository root, with a Python that has `cadence-net` installed:

```bash
python -m pytest arm/tests -q
python arm/run.py --seeds 0 1 --pilot --out runs/arm/pilot
python arm/run.py --seeds 10 11 12 13 14 --out runs/arm/acceptance
python arm/verify.py runs/arm/acceptance
```

## Assumptions recorded in config.json

Decisions at 10 Hz (a torque pair is held for five 50 Hz body steps); the sensors report
the measured displacement, its change, the velocity change and the angle change over the
last decision; the prediction targets are the next decision's values of those sensors;
the displacement bound is 0.12 arm lengths; the world head is a records cortex of 32,000
cells with 160 active per reading, records at rate 0.1; no replay ring.

## Evaluation copies

Each checkpoint is evaluated twice on the same held-out targets. A read-only copy reports
the competence at the moment of the checkpoint (`heldout_success_frozen` in the receipt).
A copy that keeps learning with exploration off, one held-out target after another, is
the gated measure (`heldout_success`): the world head keeps writing its records in the
agent's one operating mode. Both copies are isolated checkpoint copies; the saved parent
life is never modified. The MLP baselines are evaluated the same way. Lesions (shuffled
pairing, corrupted dynamics: the dynamics synapses and the records permuted) are compared
on the first fifty held-out targets of adapting copies, before relearning can mask the
damage; the born-frozen control is read-only by definition. The copier tracking runs on an
adapting copy as well.

## Receipt

`receipt.json` is the acceptance receipt of seeds 10 to 14: every predicate with its value
and threshold, the source hashes of this directory and of the library, the sha256 of every
event log and checkpoint, the learning curve (`metrics.curve`: after every
`curve_interval` decisions of the life, a read-only copy's success on the first
`curve_targets` held-out targets) and the digest of the receipt itself. `verify.py` recomputes the
success rates and the prediction gain from the event logs and fails closed on an
incomplete run. The page `web/` continues a life of the acceptance brain in the browser.
