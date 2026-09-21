# Cadence examples

Worked examples for [Cadence](https://github.com/muellerberndt/cadence). Each example is one
directory with its own README, its page, the receipts behind every number it states, and a check
that recomputes them.

| Example | What it shows | Evidence | Receipt |
|---|---|---|---|
| [worm](worm/) | The complete nervous system of *C. elegans*, 302 neurons wired as measured, as one TemporalPatchNet. It learns during its life what the smells around it predict, from food, from pain, and from the treats and pokes you give it; every neuron, synapse and lesson is drawn as it happens | the browser brain reproduces the library's TemporalPatchNet to a relative difference of 8.5e-12 on lessons it lives through | [parity](worm/tests/parity.mjs) |
| [connect_four](connect_four/) | One life learns what a dropped stone does, which windows of four cells are completed lines and what positions are worth, from the games it plays, and chooses its moves by searching over what it learned | tactical suite 100%; paired score against random 0.995 and against one-ply 0.962; decision latency p95 52 ms | [receipt](connect_four/receipt.json) |
| [amen](amen/) | A jungle composer in one patch, running in the browser: from silence it computes sixteen bars of drums, bass and texture while the page shows its brain, then plays them | next drum slice 0.84 to 0.86 on held-out tracks (most frequent slice 0.03 to 0.05), next bass note 0.54 to 0.64 (repeat previous 0.14 to 0.48), texture error 0.085 to 0.090 (repeat previous 0.099 to 0.117); browser engine equal to the library on 128 half-beats | [receipt](amen/runs/record-composer-v10/receipt.json) |

## The worm

The worm's brain is its connectome: every chemical synapse and gap junction becomes one permitted
connection of a `PartitionedTemporalPatchNet`, weighted at birth by synapse count and signed by
transmitter. Smells reach the olfactory neurons AWA and AWC, bacteria the dopaminergic CEP, ADE
and PDE, pain the ASH nociceptors. The body reads the command interneurons directly: AVB and PVC
drive it forward, AVA, AVD and AVE drive it backward. When food or pain arrives, the worm relives
the moments that led there and learns, by centred equilibrium detuning, what its command neurons
should have been doing on the way. Details, sources and limits: [worm/README.md](worm/README.md).

## Connect Four

One life learns what a dropped stone does, which windows of four cells are completed lines
and what positions are worth, from the games it plays, and chooses its moves by searching
over what it learned: no board object, no terminal oracle, no minimax labels.

- Page: https://floatingpragma.io/cadence-examples/connect_four/ (play against the brain
  while it keeps learning from the game, with the whole brain live beside the board). The
  page's brain is seed 10 of the five-seed acceptance run, trained against Pascal Pons'
  perfect solver at graded strengths and then schooled against it: it reads the threats on a
  board off its records, remembers what its searches proved and the columns winners played,
  and searches its records as deep as its budget of imagined boards reaches. Played on the page in a browser: moving
  first it beat the perfect solver in 40 of 40 games with every move perfect; it beat both
  AlphaZero models in every game, first and second; moving second it beat the solver at 70% in 29 of 30. Its model card states the training, the schooling, the
  bench and what it cannot do; bench receipts: [connect_four/bench/receipts](connect_four/bench/receipts).
- Receipt of the stage: [connect_four/receipt.json](connect_four/receipt.json), five acceptance seeds.
  Measured: exact next boards on held-out moves 100%; terminal prediction
  100%; tactical suite 100% (win in one, block in one); paired score
  against random 0.995 and against one-ply 0.962 (each seed at least
  0.925); planning gain over the same records without search 0.65;
  loss with corrupted dynamics 0.84; loss on the old opponents after the
  adaptation games -0.004; decision latency p95 52 ms.
- Supplied: the rules as the world, the move legality, the negamax search over imagined
  boards within a budget of imagined transitions, the settling schedule.
- Learned: the drop records (where a stone lands), the line records (which windows are
  completed lines, what a position is worth), from empty records.
- Controls: the same records frozen, corrupted dynamics, random, one-ply and minimax
  opponents at fixed budgets.
- Details: [connect_four/README.md](connect_four/README.md); what the work taught us:
  [connect_four/FINDINGS.md](connect_four/FINDINGS.md).

## The composer

One record patch learned sixteen jungle tracks as events per half-beat. The page ships the
trained brain: it starts from silence, hears each half-beat it plays, computes a track in the
browser and plays it with its activity in time with the sound. Details, receipts and what it
does not do: [amen/README.md](amen/README.md).

## Run them

Each example pins the Cadence commit its evidence was produced with.

```bash
# worm: cadence 3d655c84b131388c4ef05d945947fd3e9e786d45
python -m pip install "cadence-net @ git+https://github.com/muellerberndt/cadence.git@3d655c84b131388c4ef05d945947fd3e9e786d45" scipy
python worm/tools/export_web.py && node worm/tests/parity.mjs
python -m http.server -d worm/web 8801        # then open http://127.0.0.1:8801

# connect_four: cadence 21a120311e9817e817499318f8125896cd19534d
python -m pip install "cadence-net @ git+https://github.com/muellerberndt/cadence.git@21a120311e9817e817499318f8125896cd19534d" pytest
python -m pytest -q -m "not slow" agent/tests connect_four/tests

# amen: the receipt names its library commit; verify.py also runs the browser parity under node
python amen/verify.py
```

`agent/`, `web/` and `conftest.py` are the actor, the browser engine and the test import path that
Connect Four is built on.
