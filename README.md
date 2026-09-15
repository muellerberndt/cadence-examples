# Cadence examples

Three applications that learn through one stream of experience: observe, remember, predict,
act, learn from the outcome. Each entry links its page, its acceptance receipt and the
verifier that recomputes the receipt from the event logs. The library is
[cadence](https://github.com/muellerberndt/cadence); its README explains the principle.

## Arm

A two-link arm learns its own body from motor babbling, reaches targets through a search
over the consequences it has recorded, copies what a visitor draws, adapts to a longer
link without a reset and returns to its original body.

- Page: https://floatingpragma.io/cadence-examples/arm/ (draw a figure; the arm copies it while
  the whole brain runs beside it; a checkpoint selector loads intermediate states of the life).
- Receipt: [arm/receipt.json](arm/receipt.json), five acceptance seeds. Measured: held-out
  reaching {{ARM_REACHING}} (gate 0.9, each seed at least {{ARM_REACHING_MIN}}); copier
  tracking error {{ARM_COPIER}} (bar {{ARM_COPIER_BAR}}); recovery after a longer link
  {{ARM_RECOVER}} and retention on the original body {{ARM_RETAIN}}; the online MLP on the
  same stream reaches {{ARM_MLP_NO_REPLAY}} without a replay ring and {{ARM_MLP_REPLAY}}
  with its own ring; the Jacobian PD controller {{ARM_PD}}; random torques {{ARM_RANDOM}}.
  Learning curve of a read-only copy on forty held-out targets: {{ARM_CURVE_3000}} after
  3,000 decisions of the life, {{ARM_CURVE_6000}} after 6,000, {{ARM_CURVE_10000}} after 10,000.
- Compared ([comparisons/arm/receipt.json](comparisons/arm/receipt.json), the same life
  with the world model replaced, five seeds): held-out reaching with an online MLP
  {{CMP_ARM_MLP}} without a replay ring and {{CMP_ARM_MLP_RING}} with one, with an online
  transformer {{CMP_ARM_TRANSFORMER}} without a ring and {{CMP_ARM_TRANSFORMER_RING}} with
  one; copier tracking error {{CMP_ARM_MLP_COPIER}} (MLP), {{CMP_ARM_MLP_RING_COPIER}} (MLP
  with ring), {{CMP_ARM_TRANSFORMER_COPIER}} (transformer), {{CMP_ARM_TRANSFORMER_RING_COPIER}}
  (transformer with ring). The records brain reaches {{ARM_REACHING}} and {{ARM_COPIER}} from
  one pass over the same stream with no ring.
- Supplied: the arm dynamics and kinematics, the reward rule, the sensor encoding, the
  planning objective and the beam search over recorded consequences, the settling schedule.
- Learned: the records of the world head (hand acceleration, joint velocity change, joint
  angle change) and the settled regions' synapses, from empty records and random synapses.
- Controls: the Jacobian-transpose controller, the online MLP with and without a ring, the
  architecture frozen from birth, shuffled action pairing, corrupted dynamics.
- Details: [arm/README.md](arm/README.md).

## World

In a 6 by 6 world seen one cell at a time, one life learns the consequences of its
actions, remembers where it saw an object, corrects that memory when the object moves,
holds a cue across a delay, adapts to doors that stop opening and grounds words in objects.

- Page: https://floatingpragma.io/cadence-examples/world/ (the world, the four stores, the
  records cortex and the whole brain, phase by phase).
- Receipt: [world/receipt.json](world/receipt.json), five acceptance seeds. Measured:
  requests fulfilled {{WORLD_REQUESTS}} with memory and {{WORLD_ERASED}} with the memory
  erased (gate: a loss of at least 30 points); visible correction {{WORLD_CORRECTION}}; cue
  choice {{WORLD_CUE}} at delay 8 with the reward-shifted control at {{WORLD_CUE_SHIFT}};
  grounding {{WORLD_GROUNDING}}; door recovery {{WORLD_DOORS}}; shuffled action pairing
  {{WORLD_SHUFFLED}}; random moves {{WORLD_RANDOM}}; the privileged planner {{WORLD_PRIVILEGED}}.
- Compared ([comparisons/world/receipt.json](comparisons/world/receipt.json), the same life
  with the world model replaced, five seeds): consequence accuracy {{CMP_WORLD_MLP}} (MLP),
  {{CMP_WORLD_MLP_RING}} (MLP with ring), {{CMP_WORLD_TRANSFORMER}} (transformer),
  {{CMP_WORLD_TRANSFORMER_RING}} (transformer with ring) against {{WORLD_CONSEQUENCES}} for
  the records brain; cue choice at delay 8: {{CMP_WORLD_MLP_CUE}}, {{CMP_WORLD_MLP_RING_CUE}},
  {{CMP_WORLD_TRANSFORMER_CUE}}, {{CMP_WORLD_TRANSFORMER_RING_CUE}} against {{WORLD_CUE}}.
  The requests, corrections, doors and grounding pass with every world model, since the
  declared stores and the search carry them.
- Supplied: the world, its tasks and reward rules, the sensor encoding with missing flags,
  four stores with declared keys, the A* search over recorded consequences, the settling
  schedule.
- Learned: the records of the world head (eleven fields), the critic, the contents of the
  stores.
- Controls: erased places, erased words, shuffled pairing, reward-shifted cue, born frozen,
  random, the privileged planner, a tabular model behind the same stores and search.
- Details: [world/README.md](world/README.md).

## Connect Four

One life learns what a dropped stone does, which windows of four cells are completed lines
and what positions are worth, from the games it plays, and chooses its moves by searching
over what it learned: no board object, no terminal oracle, no minimax labels.

- Page: https://floatingpragma.io/cadence-examples/connect_four/ (play against the brain
  while it keeps learning from the game; the columns it imagined and the values it read are
  shown beside the whole brain; a checkpoint selector loads the brain after 100, 400 and
  1,000 games and at every phase end).
- Receipt: [connect_four/receipt.json](connect_four/receipt.json), five acceptance seeds.
  Measured: exact next boards on held-out moves {{C4_VALIDITY}}; terminal prediction
  {{C4_TERMINAL}}; tactical suite {{C4_TACTICAL}} (win in one, block in one); paired score
  against random {{C4_RANDOM}} and against one-ply {{C4_ONE_PLY}} (each seed at least
  {{C4_ONE_PLY_MIN}}); planning gain over the same records without search {{C4_PLANNING}};
  loss with corrupted dynamics {{C4_CORRUPTED}}; loss on the old opponents after the
  adaptation games {{C4_CONTINUAL}}; decision latency p95 {{C4_LATENCY}} ms.
- Supplied: the rules as the world, the move legality, the negamax search over imagined
  boards within a budget of imagined transitions, the settling schedule.
- Learned: the drop records (where a stone lands), the line records (which windows are
  completed lines, what a position is worth), from empty records.
- Controls: the same records frozen, corrupted dynamics, random, one-ply and minimax
  opponents at fixed budgets.
- Details: [connect_four/README.md](connect_four/README.md).

## Run them

Python 3.11 or later and the library at the commit the receipts ran against:

```bash
python -m pip install "cadence-net @ git+https://github.com/muellerberndt/cadence.git@{{CADENCE_COMMIT}}"
python -m pytest -q -m "not slow" agent/tests arm/tests world/tests connect_four/tests
python arm/run.py --seeds 0 --pilot --out runs/arm/pilot
python world/run.py --seeds 0 --pilot --out runs/world/pilot
python connect_four/run.py --seeds 0 --pilot --out runs/connect_four/pilot
python tools/verify_gallery.py
```

`--pilot` divides the budgets by ten; the acceptance runs use seeds 10 to 14 at full budget
(`arm/README.md`, `world/README.md`). `comparisons/arm.py` and `comparisons/world.py` run
the same lives with the world model replaced by an online MLP or transformer, with and
without a replay ring ([comparisons/README.md](comparisons/README.md)).
`tools/verify_gallery.py` checks each receipt's digest, predicates, seed schedule, source
hashes and library commit. The browser engine is
checked against the Python agent decision by decision (`web/parity.mjs`,
`world/web/parity_s02.mjs`).

## What every example passes

- Its stage gates on frozen held-out suites, on every scheduled seed, with every control.
- Behaviour that changes through experience and survives the retention test; a born-frozen
  agent and a causally broken learner fail the relevant test.
- Reproduction from random initialisation and empty records by the commands above.
- One controller for the page and the headless runner: the page continues the same life.
- Receipts with source hashes, event logs and checkpoints named by their sha256.

## The principle

A patch keeps records of what followed each reading it took part in; its prediction is the
sum of the records the reading touches, weighted by activity; the witnessed outcome is
written into exactly those records. A sparse code touches few, so one stream suffices and
no replay ring is needed. The settled regions complete partial readings, carry context
across a delay and hold the policy. The library page
[records](https://github.com/muellerberndt/cadence/blob/main/docs/memory.md#records) states
the mechanism in full.

## Layout

`agent/`: the experience agent shared by every example (the event transaction, the records
head, receipts, the web export). `arm/`, `world/`, `connect_four/`: one directory per example with `run.py`,
`verify.py`, `config.json`, tests, tools and `web/`. `web/`: the browser engine, the
parity harness, the records view and the library's brain renderer. `tools/verify_gallery.py`:
the gallery check the workflows run.
