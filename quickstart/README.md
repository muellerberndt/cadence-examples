# The quickstart brains in the browser

Three tiny demos, each the library quickstart's own computation behind a local page with
the whole brain in the viewer, its activity moment by moment, how far each neuron still
moves as the brain settles, the synapses' last change, the numbers that define the
equilibrium, and the learning curves. Nothing is hosted and nothing is trained ahead of
time; a run takes seconds to a minute.

These are the legacy quickstart models checked against Cadence 0.18.0: `stream`
uses a gated record patch, `decide` a recurrent settling neuron network, and
`body` a temporal patch with consequence learning and planning. Use the exact
release source pinned in CI to run them:

```bash
cd cadence-examples
python -m pip install "cadence-net @ git+https://github.com/muellerberndt/cadence@318a48c510eddfafa0cd299451446bf644aa9287" pytest
python -m quickstart.demo stream     # a record patch learns a stream, remembers in one shot, and sleeps
python -m quickstart.demo decide     # a settling brain decides
python -m quickstart.demo body       # a temporal patch learns a consequence and plans
python -m pytest -q quickstart       # the demos' tests
```

For a new application on Cadence 0.50.0, choose flat input-only, ordinary
state-coupled or recursive state-and-error wiring. All three use the same patch
rule, settlement and qualification checks. Current construction examples are in
the [layout guide](https://github.com/muellerberndt/cadence/blob/main/docs/VARIANTS.md);
the [performance guide](https://github.com/muellerberndt/cadence/blob/main/docs/PERFORMANCE.md)
explains their capability and cost tradeoffs. The six worked examples in the
parent directory each document their own model and library pin.
