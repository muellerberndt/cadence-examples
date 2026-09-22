# Cadence examples

Four worked examples for [Cadence](https://github.com/muellerberndt/cadence). Each is one directory
with its own README, its page, the receipts behind every number it states, and a check that
recomputes them. Each page runs its brain in the browser with the arithmetic of the library, and a
parity test holds the two together.

Each example shows a different side of the same architecture.

| Example | What it shows | The brain | Evidence | Check |
|---|---|---|---|---|
| [worm](worm/) | Learning from experience in one life, simple affect (food and pain), direct motor control, a measured connectome as the only wiring | The 302 neurons of *C. elegans* as one `PartitionedTemporalPatchNet` masked by the connectome | the browser brain reproduces the library to a relative difference of 8.5e-12 on lessons it lives through; the body holds its invariants under stress | [parity](worm/tests/parity.mjs), [body](worm/tests/body.mjs) |
| [amen](amen/) | Creation: from silence it computes sixteen bars of drums, bass and texture, hearing each half-beat it plays | One `RecordPatchNet`: 128 context channels, 8,192 record cells | on held-out tracks, next drum slice 0.84 to 0.86 (most frequent slice 0.03 to 0.05), next bass note 0.54 to 0.64 (repeat previous 0.14 to 0.48), texture error 0.085 to 0.090 (repeat previous 0.099 to 0.117); browser engine equal to the library on 128 half-beats | [receipt](amen/runs/record-composer-v10/receipt.json), [verify](amen/verify.py) |
| [patchworld](patchworld/) | Evolution of bodies and wiring, learning in one life, planning through a learned model, muscles driven directly, drives, computation priced in mass | Two `RecordPatchNet`s per being, a policy and a model, each a list of inherited cortices | the browser brain reproduces the library to 1e-15 on observation, windows, backtracking and planning; mass is conserved at every tick; in two worlds of 20,000 ticks mean speed rises from 0.040 to 0.071 and from 0.042 to 0.063 cells per tick | [parity](patchworld/sim/parity.js), [physics](patchworld/tests/physics.test.js) |
| [connect4](connect4/) | Planning: a search over imagined boards reads a value learned by watching a perfect player, and every move is graded by that player | One `RecordPatchNet`: 256 context channels, 4,096 record cells left empty by the school | 40-0-0 against AlphaZero at 25 simulations and 35-0-5 at 100; 36-0-4 against the perfect solver playing 70% of its moves, where the search alone scores 28-2-10; moving first against the perfect solver 6-1-13, the search alone 0-0-20; the browser engine selects the library's record cells and agrees with its values to 1e-15 | [receipt](connect4/web/receipt.json), [verify](connect4/verify.py) |

## Build on them

These examples are starting points. Fork the repository, change a rule, swap a sense, grow a different body, school a
stronger player, train a composer on other music, and see what the same few operations can be made to do. Every brain
is short enough to read in an evening, every page runs from a static folder, and every check runs from one command, so
a change shows at once whether it kept the numbers or broke them. The library is
[Cadence](https://github.com/muellerberndt/cadence); issues and pull requests are welcome in both repositories.

## The worm

The worm's brain is its connectome: every chemical synapse and gap junction becomes one permitted
connection of a `PartitionedTemporalPatchNet`, weighted at birth by synapse count and signed by
transmitter. Smells reach the olfactory neurons AWA and AWC, bacteria the dopaminergic CEP, ADE
and PDE, pain the ASH nociceptors. The body reads the command interneurons directly: AVB and PVC
drive it forward, AVA, AVD and AVE drive it backward. When food or pain arrives, the worm relives
the moments that led there and learns, by centred equilibrium detuning, what its command neurons
should have been doing on the way. Details, sources and limits: [worm/README.md](worm/README.md).

## Connect Four

One `RecordPatchNet` reads a position right after a stone has landed, as the side that placed it
sees it, and says how the game ends for that side. It learned that by watching Pascal Pons'
perfect solver play out set-up positions to the end: it saw the games and how they ended, and
the solver's scores were never recorded. Its slow parameters carry what it learned; its record
store is left empty, because writing millions of positions into it drowns it (on three runs the slow readout alone names the winner 3 points more often on positions it never saw and 2 points more often on positions it witnessed:
[receipt](connect4/receipts/records_drown.json)). A supplied search knows the rules,
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

Every example is verified against the current Cadence release, 0.12.0: its receipts are replayed, its exports rebuilt and its browser engine checked against the library at that release. A receipt names the commit its own numbers were produced with.

```bash
python -m pip install "cadence-net==0.12.0" scipy

# worm
python worm/tools/export_web.py && node worm/tests/parity.mjs && node worm/tests/body.mjs
python -m http.server -d worm/web 8801        # then open http://127.0.0.1:8801

# connect4
python connect4/verify.py
python connect4/web/export.py --patch connect4/brain/v1.npz --out /tmp/connect4 && node connect4/web/parity.mjs /tmp/connect4
python -m http.server -d connect4/web 8805     # then open http://127.0.0.1:8805

# amen: the receipt names its library commit; verify.py also runs the browser parity under node
python amen/verify.py

# patchworld
python patchworld/ref/make_fixture.py --out /tmp/patchworld_fixture.json && node patchworld/sim/parity.js /tmp/patchworld_fixture.json
node patchworld/tests/physics.test.js
python -m http.server -d patchworld/web 8804  # then open http://127.0.0.1:8804
```

Made with ♥ by [Pragma Research](https://floatingpragma.io).
