# Cadence examples: version-pinned research archive

Six worked examples of [Cadence](https://github.com/muellerberndt/cadence),
**a deep real-time brain that learns from experience**. This archived repository
preserves their source, trained models and reproducibility checks. Each example
uses its own library version and mechanism; use its documented pin to reproduce it.

**For the public demo collection, visit [Cadence demos](https://floatingpragma.io/demos/)
and [cadence-demos on GitHub](https://github.com/muellerberndt/cadence-demos).**
You can [play AMEN](https://floatingpragma.io/demos/amen/) or
[explore Patch World](https://floatingpragma.io/demos/patch-world/) in your browser.
Those published demos have their own source and version records.

[Cadence overview](https://floatingpragma.io/cadence/) ·
[Research preprint](https://philpapers.org/rec/MUECAP-2) ·
[Pragma Research](https://floatingpragma.io/)

## Explore the archive

The examples make observer-like computation visible: bounded local state,
explicit ports, readback and feedback from experience. Their biological wiring,
supplied controllers and learning rules differ. A result belongs to the model,
task and version identified by its evidence.

| Example and guide | What to inspect | Library model | Evidence and checks |
| --- | --- | --- | --- |
| [Worm](worm/README.md) | A 302-neuron *C. elegans* connectome supplying the wiring, with learning from food and pain in a simulated body | Cadence 0.50.0; state-coupled columns and feedback across ticks | [Browser parity](worm/tests/parity.mjs), [body invariants](worm/tests/body.mjs), [learning accounting](worm/tests/test_learning_accounting.py) |
| [AMEN](amen/README.md) | A trained record patch generates a jungle track before audio playback | 0.11-trained `RecordPatchNet`; fixed browser weights and archived-run parity | [Receipt](amen/runs/record-composer-v12/receipt.json), [verifier](amen/verify.py) |
| [Patch World](patchworld/README.md) | Soft bodies evolve and learn in a world whose rules conserve mass | 0.12.0; policy and model record patches | [World recordings](patchworld/receipts/), [parity](patchworld/sim/parity.js), [physics checks](patchworld/tests/physics.test.js) |
| [Connect Four](connect4/README.md) | A learned position evaluator guides a supplied game-tree search | 0.12.0; record patch trained from perfect-player games | [Game receipt](connect4/web/receipt.json), [verifier](connect4/verify.py) |
| [Dozing cat](dozing-cat/README.md) | A settling governor reads a belief patch's surprise and selects habit, imagination or learning | Commit `f06eab06`, after 0.13.0; belief patch and separate governor | [Receipts](dozing-cat/receipts/), [parity](dozing-cat/tests/parity.mjs), [verifier](dozing-cat/verify.py) |
| [Fly in the Matrix](fly-matrix/README.md) | A retained fruit-fly connectome chooses approach or avoidance goals; a supplied controller executes flight | 0.17.0; recurrent neuron-rate network with external actor-critic learning signals | [Causal learning receipt](fly-matrix/receipts/goal_learning_causal_2026-09-27.json), [browser learning receipt](fly-matrix/receipts/release_0_17_local_learning.json), [reproduction guide](fly-matrix/README.md#run) |

These are inspectable simulations and software experiments. They do not establish
biological fidelity, general robot capability or a matched advantage over transformers.
Each guide explains its measurements and scope.

## Run and reproduce

The six browser applications are preserved under each example's `web/` folder.
Serve one locally, for example from the repository root:

```bash
python3 -m http.server 8800 --directory amen/web
# Open http://127.0.0.1:8800/
```

Substitute `worm/web`, `patchworld/web`, `connect4/web`, `dozing-cat/web` or
`fly-matrix/web`. These pages include browser engines; serving them does not
require installing the Python Cadence package. Rebuilding exports and running
library parity checks require the example's pinned environment.

Use a separate virtual environment for each library pin. The example guides
contain reproduction commands, and [CI](.github/workflows/ci.yml) records the
exact releases, commits and checks. Some checks require external baseline tools;
follow their setup instructions before reproducing the corresponding results.

The supporting [browser engine](engine/README.md) uses the legacy 0.17 model.
The [viewer](viewer/README.md) and [three local quickstarts](quickstart/README.md)
are checked against the exact 0.18.0 source pinned in CI.

## Build with Cadence

For an application using the supported library, start with
[cadence-demos](https://github.com/muellerberndt/cadence-demos), the
[layout guide](https://github.com/muellerberndt/cadence/blob/main/docs/VARIANTS.md)
and the [performance guide](https://github.com/muellerberndt/cadence/blob/main/docs/PERFORMANCE.md).
Flat input-only, ordinary state-coupled and recursive observer layouts share the
settlement and qualification procedure. The legacy record-patch examples here
use different mechanisms and should be read with their version notes.

## License and assets

Repository code is [MIT licensed](LICENSE). Each example's **Data and rights**
entry identifies the terms and provenance of its training data and imported
assets. The [AMEN card](amen/README.md#card) records unverified redistribution
rights for its audio kit; the code license does not settle those permissions.

Built by [Pragma Research](https://floatingpragma.io/).
