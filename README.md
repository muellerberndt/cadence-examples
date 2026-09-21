# Cadence examples

Worked examples for [Cadence](https://github.com/muellerberndt/cadence). Each example is one
directory with its own README, its page, the receipts behind every number it states, and a check
that recomputes them.

| Example | What it shows | Evidence | Receipt |
|---|---|---|---|
| [worm](worm/) | The complete nervous system of *C. elegans*, 302 neurons wired as measured, as one TemporalPatchNet. It learns during its life what the smells around it predict, from food, from pain, and from the treats and pokes you give it; every neuron, synapse and lesson is drawn as it happens | the browser brain reproduces the library's TemporalPatchNet to a relative difference of 8.5e-12 on lessons it lives through | [parity](worm/tests/parity.mjs) |
| [connect_four](connect_four/) | One life learns what a dropped stone does, which windows of four cells are completed lines and what positions are worth, from the games it plays, and chooses its moves by searching over what it learned | tactical suite 100%; paired score against random 0.995 and against one-ply 0.962; decision latency p95 52 ms | [receipt](connect_four/receipt.json) |
| [connect4](connect4/) | Connect Four on one RecordPatchNet. The patch reads a position and says how the game ends for the side that placed the last stone; it learned that by watching a perfect player play out set-up positions, and a supplied search reads it. The page draws the reads as they happen: two retinas, 256 context channels, 4,096 record cells | 40-0-0 against AlphaZero at 25 simulations and 35-0-5 at 100; 36-0-4 against the perfect solver playing 70% of its moves; moving first against the perfect solver 6-1-13; every move graded by the solver. The browser engine selects the library's record cells and agrees with its values to 1e-15, and searches as the Python reference does | [receipt](connect4/web/receipt.json), [check](connect4/verify.py) |
| [amen](amen/) | A jungle composer in one patch, running in the browser: from silence it computes sixteen bars of drums, bass and texture while the page shows its brain, then plays them | next drum slice 0.84 to 0.86 on held-out tracks (most frequent slice 0.03 to 0.05), next bass note 0.54 to 0.64 (repeat previous 0.14 to 0.48), texture error 0.085 to 0.090 (repeat previous 0.099 to 0.117); browser engine equal to the library on 128 half-beats | [receipt](amen/runs/record-composer-v10/receipt.json) |
| [patchworld](patchworld/) | Soft bodies evolve on a world that conserves its mass. Shapes, muscles and brains built from cortices of one RecordPatchNet mutate at every birth, each being learns during its life and can plan its motor through its own model, and the page opens any brain with every neuron, connection and record cell drawn as it works | the browser brain reproduces the library's RecordPatchNet to a relative difference of 1e-15 on observation, windows, backtracking and planning; mass is conserved at every tick | [parity](patchworld/sim/parity.js) |

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

## Connect Four on the record patch

One `RecordPatchNet` reads a position right after a stone has landed, as the side that placed it
sees it, and says how the game ends for that side. It learned that by watching Pascal Pons'
perfect solver play out set-up positions to the end: it saw the games and how they ended, and
the solver's scores were never recorded. Its slow parameters carry what it learned; its record
store is left empty, because writing millions of positions into it drowns it (the slow readout
alone then scores about 3.5 points higher, on three runs). A supplied search knows the rules,
imagines moves and reads the patch where it stops looking; a line it can follow to the end of
the game is proven. Measured on the build the page plays, 40 paired games against each
opponent with every move graded by the solver: what the search alone achieves, what the patch
adds, and where it still errs are in [connect4/README.md](connect4/README.md). It does not yet
learn from the games played on its page.

## The composer

One record patch learned sixteen jungle tracks as events per half-beat. The page ships the
trained brain: it starts from silence, hears each half-beat it plays, computes a track in the
browser and plays it with its activity in time with the sound. Details, receipts and what it
does not do: [amen/README.md](amen/README.md).

## Patch World

A torus of soil, food, rock and mud under one moving sun, with mass conserved at every tick.
A body is a graph of point masses, springs and muscles that crawls by grip alone; it is born
with three nodes and grows the rest. Its brain is two record patches: a policy that proposes the
motor and a model that predicts what the motor will change, each a list of cortices that read
their own part of the senses. Every birth mutates shape, limbs, muscles and cortices; energy and
death select, and the world has no fitness function. Click any body to open its brain; `Inspect`
draws every neuron, weight, record cell and motor neuron live beside the moving body. The page
exports a chronicle of every lineage, and `patchworld/sim/why.py` reads it and says why one
lineage took the world. Details, evidence and limits: [patchworld/README.md](patchworld/README.md).

## Run them

Each example pins the Cadence commit its evidence was produced with.

```bash
# worm: cadence 3d655c84b131388c4ef05d945947fd3e9e786d45
python -m pip install "cadence-net @ git+https://github.com/muellerberndt/cadence.git@3d655c84b131388c4ef05d945947fd3e9e786d45" scipy
python worm/tools/export_web.py && node worm/tests/parity.mjs && node worm/tests/body.mjs
python -m http.server -d worm/web 8801        # then open http://127.0.0.1:8801

# connect_four: cadence 21a120311e9817e817499318f8125896cd19534d
python -m pip install "cadence-net @ git+https://github.com/muellerberndt/cadence.git@21a120311e9817e817499318f8125896cd19534d" pytest
python -m pytest -q -m "not slow" agent/tests connect_four/tests

# connect4: cadence 02fec624648d421e02ecb00f52f3d3072e9fe9ae
python -m pip install "cadence-net @ git+https://github.com/muellerberndt/cadence.git@02fec624648d421e02ecb00f52f3d3072e9fe9ae"
python connect4/verify.py
python connect4/web/export.py --patch connect4/brain/v1.npz --out /tmp/connect4 && node connect4/web/parity.mjs /tmp/connect4
python -m http.server -d connect4/web 8805     # then open http://127.0.0.1:8805

# amen: the receipt names its library commit; verify.py also runs the browser parity under node
python amen/verify.py

# patchworld: cadence 02fec624648d421e02ecb00f52f3d3072e9fe9ae
python -m pip install "cadence-net @ git+https://github.com/muellerberndt/cadence.git@02fec624648d421e02ecb00f52f3d3072e9fe9ae"
python patchworld/ref/make_fixture.py --out /tmp/patchworld_fixture.json && node patchworld/sim/parity.js /tmp/patchworld_fixture.json
node patchworld/tests/physics.test.js
python -m http.server -d patchworld/web 8804  # then open http://127.0.0.1:8804
```

`agent/`, `web/` and `conftest.py` are the actor, the browser engine and the test import path that
Connect Four is built on.
