# The viewer

The whole-brain viewer every page of these examples draws with: `atlas.py` lays a
connectome out (regions by a force layout of the region graph, neurons by a spectral
embedding of their synapses, sheets by declared shapes or supplied coordinates) and writes
the `cadence.atlas/v1` payload; `brain_scan.js` draws it in WebGL2 with a Canvas2D
fallback (tissue in region colours whose brightness is the activation, a hot glow where
neurons change, synapses lit by their source, an EEG-style montage per region);
`brain_scan_page.html` is the self-contained replay page `Atlas.page` writes.

```python
import sys; sys.path.insert(0, "<path to cadence-examples>")
from viewer.atlas import atlas_of, build_atlas, brain_scan_script

atlas = atlas_of(brain, shapes={"retina": (32, 32)})   # a cadence.Brain
script = brain_scan_script()                           # the viewer, served as a classic script
```

The library ships the brains alone; pages take the viewer from here. `test_atlas.py` holds
the viewer's tests (`python -m pytest -q viewer` from the repository root).
