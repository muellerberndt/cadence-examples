# The viewer

The whole-brain viewer every page of these examples draws with: `atlas.py` lays a
connectome out (regions by a force layout of the region graph, neurons by a spectral
embedding of their synapses, sheets by declared shapes or supplied coordinates) and writes
the `cadence.atlas/v1` payload; `brain_scan.js` draws it in WebGL2 with a Canvas2D
fallback (tissue in region colours whose brightness is the activation, a hot glow where
neurons change, synapses lit by their source, an EEG-style montage per region);
`brain_scan_page.html` is the self-contained replay page `Atlas.page` writes.

## Setup and migration

The viewer is source code in this repository, not a module installed by
`pip install cadence-net`. Install Cadence and clone the examples alongside your
application, then run from the examples repository root:

```bash
python -m pip install cadence-net
git clone https://github.com/muellerberndt/cadence-examples.git
cd cadence-examples
```

Cadence 0.18 removes `cadence.atlas`, its top-level atlas exports and the
`cadence-demo` executable. Import the viewer from this checkout; applications in
another directory must add its absolute path to `sys.path` or `PYTHONPATH`.
The numerical examples here also work with Cadence 0.17. The browser quickstarts
now run with `python -m quickstart.demo stream`, `decide` or `body`; see
[the quickstart instructions](../quickstart/README.md).

```python
import cadence as cd
from viewer.atlas import atlas_of, build_atlas, brain_scan_script

brain = cd.Brain(cd.layered(4, 8, 2, seed=0), cd.NeuronModel())
atlas = atlas_of(brain, shapes={"input": (2, 2)})
script = brain_scan_script()                           # the viewer, served as a classic script
```

The [complete viewer reference](reference.md) documents atlas fields, layout,
recorded frames, self-contained pages, renderer methods and display options.
Generic brain lobes, particle motion and heat are visual conventions; the viewer
does not establish anatomy, biological transmission timing or solver convergence.

The library ships the brains alone; pages take the viewer from here. `test_atlas.py` holds
the viewer's tests (`python -m pytest -q viewer quickstart` from the repository root).
