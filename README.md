# Cadence examples

Five applications that learn through one stream of experience: observe, remember, predict,
act, learn from the outcome. Each entry links its page and its acceptance receipt; the four
that pass their gates link the verifier that recomputes the receipt from the event logs, and
the composer is published with its gates open. The library is
[cadence](https://github.com/muellerberndt/cadence); its README explains the principle.

## Arm

A two-link arm learns its own body from motor babbling, reaches targets through a search
over the consequences it has recorded, copies what a visitor draws, adapts to a longer
link without a reset and returns to its original body.

- Page: https://floatingpragma.io/cadence-examples/arm/ (draw a figure; the arm copies it while
  the whole brain runs beside it; a checkpoint selector loads intermediate states of the life).
- Receipt: [arm/receipt.json](arm/receipt.json), five acceptance seeds. Measured: held-out
  reaching 99% (gate 0.9, each seed at least 98%); copier
  tracking error 0.052 (bar 0.113); recovery after a longer link
  99% and retention on the original body 99%; the online MLP on the
  same stream reaches 3% without a replay ring and 99%
  with its own ring; the Jacobian PD controller 94%; random torques 0%.
  Learning curve of a read-only copy on forty held-out targets: 22% after
  3,000 decisions of the life, 60% after 6,000, 87% after 10,000.
- Compared ([comparisons/arm/receipt.json](comparisons/arm/receipt.json), the same life
  with the world model replaced, five seeds): held-out reaching with an online MLP
  2% without a replay ring and 98% with one, with an online
  transformer 40% without a ring and 100% with
  one; copier tracking error 0.702 (MLP), 0.102 (MLP
  with ring), 0.596 (transformer), 0.058
  (transformer with ring). The records brain reaches 99% and 0.052 from
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
  requests fulfilled 100% with memory and 18% with the memory
  erased (gate: a loss of at least 30 points); visible correction 100%; cue
  choice 100% at delay 8 with the reward-shifted control at 51%;
  grounding 100%; door recovery 100%; shuffled action pairing
  3%; random moves 1%; the privileged planner 100%.
- Compared ([comparisons/world/receipt.json](comparisons/world/receipt.json), the same life
  with the world model replaced, five seeds): consequence accuracy 95% (MLP),
  95% (MLP with ring), 93% (transformer),
  95% (transformer with ring) against 98% for
  the records brain; cue choice at delay 8: 59%, 49%,
  52%, 48% against 100%.
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

## Artist

The arm of the first example holds a pen over a canvas. One life scribbles, learns what its
strokes leave, and then draws requested figures it has never seen, replanning from what its
canvas shows after every stroke; a longer link and an offset canvas change the body and the
world mid-drawing and it keeps drawing.

- Page: https://floatingpragma.io/cadence-examples/artist/ (draw a figure; the artist draws
  it stroke by stroke with the whole brain live beside it; a checkpoint selector loads the
  life after scribbling, after the segments, after the strokes and at the end).
- Receipt: [artist/receipt.json](artist/receipt.json), five acceptance seeds. Measured:
  held-out foreground F1 0.93 (gate 0.80) at a Chamfer distance of 0.014
  (gate 0.04), the weakest family 0.91 (gate 0.70); after a longer link
  0.93 and back on the original body 0.93; the same architecture frozen
  from birth 0.00, random scribbling 0.08, the supplied path oracle
  0.64, an online MLP with a replay ring behind the same search 0.80;
  decision latency p95 75 ms (gate 90).
- Supplied: the arm body, the canvas and its stroke rasterisation, the discrepancy measure,
  the reward rule, the target families and splits, the two-level search over the recorded
  consequences (a stroke intention, then the torques that follow it).
- Learned: the records of the world head (the body consequences and the marks the pen
  leaves in the window around it) and the settled regions' synapses, from empty records.
- Controls: born frozen, random scribbling, the path oracle, the online MLP with a ring,
  corrupted dynamics, shuffled action pairing, replanning removed, canvas readback removed.
- Details: [artist/README.md](artist/README.md).

## Connect Four

One life learns what a dropped stone does, which windows of four cells are completed lines
and what positions are worth, from the games it plays, and chooses its moves by searching
over what it learned: no board object, no terminal oracle, no minimax labels.

- Page: https://floatingpragma.io/cadence-examples/connect_four/ (play against the brain
  while it keeps learning from the game; the columns it imagined and the values it read are
  shown beside the whole brain; a checkpoint selector loads the brain after 100, 400 and
  1,000 games and at every phase end).
- Receipt: [connect_four/receipt.json](connect_four/receipt.json), five acceptance seeds.
  Measured: exact next boards on held-out moves 100%; terminal prediction
  100%; tactical suite 100% (win in one, block in one); paired score
  against random 0.995 and against one-ply 0.957 (each seed at least
  0.925); planning gain over the same records without search 0.61;
  loss with corrupted dynamics 0.83; loss on the old opponents after the
  adaptation games -0.003; decision latency p95 19 ms.
- Supplied: the rules as the world, the move legality, the negamax search over imagined
  boards within a budget of imagined transitions, the settling schedule.
- Learned: the drop records (where a stone lands), the line records (which windows are
  completed lines, what a position is worth), from empty records.
- Controls: the same records frozen, corrupted dynamics, random, one-ply and minimax
  opponents at fixed budgets.
- Details: [connect_four/README.md](connect_four/README.md).

## Composer

The composer hears a two-bar phrase on a three-voice SID instrument and plays it back, then
plays on from a phrase it heard half of. One life learns the MOS 8580 chip by playing it: a
gesture is 28 registers over 541 motor neurons, the world model is a records cortex holding
what a gesture does to the sound, and the answer is a search over imagined hearings.
Published with the composing gates open.

- Page: https://floatingpragma.io/cadence-examples/composer/ (seven public-domain pieces: the
  demonstration, the brain's imitation and its continuation as the emulated chip played them,
  with the whole brain's recorded settling and the records cortex's reads and writes replayed
  in step with the audio).
- Receipt: [composer/receipt.json](composer/receipt.json), five acceptance seeds on the
  held-out split of the private corpus, 8 of 13 gated
  predicates passed. Measured: pitch within a semitone 0.505 (gate 0.85); onset
  F1 0.443 (gate 0.85); continuation surprise 1.635 nats per
  event (gate 1.608); imitation margin over the procedural composer 0.469
  and over shuffled pairing 0.487 (gate 0.20 each); margin over the same
  brain frozen from birth -0.033 (gate 0.20); renderer faults
  1 (gate 0); decision latency p95 19.4 ms (gate 20).
  Open, without a gate: composing under an instruction 0.274 (wanted
  0.80), theme recall 0.013 (wanted 0.85), revision 0.125
  (wanted 0.80).
- The born-frozen margin states that the search and the note reading carry the imitation in
  this run; the sixth round scores candidates by the imagined hearing, and the page is
  rebuilt from that run when it lands.
- Supplied: the instrument (pyresidfp 0.17.0, MOS 8580, pinned file by file), the hearing,
  the corpus and its splits, the objectives, the search over imagined hearings, the settling
  schedule.
- Learned: the records of the world head (the next change of every hearing field under a
  gesture) and the settled regions' synapses, from empty records.
- Controls: born frozen, the procedural composer, shuffled pairing, a scrambled request.
- Stage sources: in the research tree the receipt hashes; they move into `composer/` when the
  stage passes its gates. The page replays a recorded life, since the chip has no browser
  port yet.
- Details: [composer/README.md](composer/README.md).

## Run them

Python 3.11 or later and the library at the commit the receipts ran against:

```bash
python -m pip install "cadence-net @ git+https://github.com/muellerberndt/cadence.git@21a120311e9817e817499318f8125896cd19534d"
python -m pytest -q -m "not slow" agent/tests arm/tests world/tests connect_four/tests artist/tests
python arm/run.py --seeds 0 --pilot --out runs/arm/pilot
python world/run.py --seeds 0 --pilot --out runs/world/pilot
python connect_four/run.py --seeds 0 --pilot --out runs/connect_four/pilot
python artist/run.py --seeds 0 --pilot --out runs/artist/pilot
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

- Its stage gates on frozen held-out suites, on every scheduled seed, with every control;
  the composer is the exception, published with its gates open and its failed predicates
  stated on its page and in its README.
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
head, receipts, the web export). `arm/`, `world/`, `connect_four/`, `artist/`: one directory per example with `run.py`,
`verify.py`, `config.json`, tests, tools and `web/`. `composer/`: the page, its bundle
manifest, its receipt and `web/` (the stage sources follow when its gates pass). `web/`: the
browser engine, the parity harness, the records view and the library's brain renderer.
`tools/verify_gallery.py`: the gallery check the workflows run.
