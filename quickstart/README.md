# The quickstart brains in the browser

Three tiny demos, each the library quickstart's own computation behind a local page with
the whole brain in the viewer, its activity moment by moment, how far each neuron still
moves as the brain settles, the synapses' last change, the numbers that define the
equilibrium, and the learning curves. Nothing is hosted and nothing is trained ahead of
time; a run takes seconds to a minute.

```bash
cd cadence-examples
python -m quickstart.demo stream     # a record patch learns a stream, remembers in one shot, and sleeps
python -m quickstart.demo decide     # a settling brain decides
python -m quickstart.demo body       # a temporal patch learns a consequence and plans
python -m pytest -q quickstart       # the demos' tests
```

The interpreter needs the library installed (`pip install cadence-net` or the editable
checkout). These are the smallest examples; the six worked examples in the parent
directory are the canonical ones.
