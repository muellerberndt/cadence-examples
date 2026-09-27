# The brain in a browser page

Every page that shows a Cadence brain shows the whole brain: every neuron a point coloured
by its region, every synapse a line, laid out so that connected neurons sit near each other,
with the activity and the change of every settling step animated live and the traces of
every region below. The component lives in `cadence-examples/viewer`; the numerical library supplies
connectomes and recorded states. Rendering budgets can limit displayed edges or particles.
Particles and generic lobes are visual conventions, not measured transmission delays or anatomy.

Run the Python examples from the examples repository root with Cadence installed;
see [setup and migration](README.md).

## The atlas

`build_atlas` lays a connectome out; `atlas_of(brain)` uses the brain's effective weights.
Regions are the connectome's populations (coarsest first; a `regions=` map overrides them).
Their role and colour follow their name: vision, sensory, memory, association, motor, value
(`roles=` overrides). Regions are placed by a force layout of the region graph, so strongly
connected regions sit next to each other. Inside a region, neurons are placed by their
synapses (iterated neighbour averaging, a spectral-style embedding), by a declared sheet
shape (`shapes={"retina": (rows, cols)}` gives a grid), or by supplied coordinates
(`positions={"cortex": array}`), or every neuron by one shared anatomical frame
(`positions={"*": array}` with a row per neuron, kept as given). Everything is deterministic
under `seed`.

```python
import numpy as np
import cadence as cd
from viewer.atlas import atlas_of, brain_scan_script

connectome = cd.layered(4, 8, 2, density=0.5, seed=0)
neuron_model = cd.NeuronModel(gain=0.02)
brain = cd.Brain(connectome, neuron_model)
drive = np.zeros((1, connectome.n))
drive[:, :4] = 0.1
atlas = atlas_of(brain, shapes={"input": (2, 2)})
print(atlas.summary())          # neurons, synapses, regions with roles
payload = atlas.to_json()       # positions, regions, every synapse, base64 arrays
```

`atlas.subsample_edges(limit, seed=None)` keeps at most `limit` synapses, drawn with
probability proportional to their absolute weight, when a page cannot draw them all.

## The renderer

`brain_scan_script()` is the source of `brain_scan.js`, the patch-net visualizer every
Cadence page shares: an ES module with no dependencies. Inline it in a page (drop the
`export` keywords) or serve it as a file. It has two styles, `style: "scan"` (the default,
the traditional view) and `style: "brain"` (the net wrapped into a transparent brain), the
same API in both, and a page switches between them at any time.

```javascript
const scan = new BrainScan(canvas, ATLAS, { labels: labelsDiv, strip: stripCanvas, style: "brain" });
scan.setStyle("scan");       // switch the style; the state, the camera and the traces carry over
scan.reset(activation);      // a new stimulus: the next step measures change from here
scan.step(activation);       // after every settling step: activation per neuron ({draw: false} defers the frame)
scan.set(activation);        // show a state without measuring change
scan.show(activation, heat, { level });  // a page's own signals: messages, glow, brightness
scan.setWeights(weights);    // the synapses learned; scan.setVisible(mask) hides lesioned neurons
scan.draw(now);              // every animation frame between steps: the messages travel
scan.onhover = (hit) => ...; // {neuron, region, activation, change, heat, potential}
scan.fit(); scan.snapshot(); // {renderer, neurons, synapses, steps, zoom, allEdgesSubmitted}
scan.screen(i); scan.toScreen(x, y);     // CSS pixels, for a page's own overlay
playFrames(scan, FRAMES, { fps: 30 });   // replay quantised recorded steps
```

### The scan style

`style: "scan"` draws tissue in each region's colour whose brightness is the activation
(the field of every neuron, so a region reads as one glowing organ); a hot glow where neurons
changed in the last steps, on a scan colour map from violet through magenta and orange to
white, fading with `heatDecay`; synapses that light up when their presynaptic neuron just
changed; particles travelling along synapses in proportion to the message sent, as a schematic display of supplied activity changes; region labels; and a montage strip
with one EEG-style row per region (its change as a line, its activity as a fill) and the
whole brain on top. `mode` selects the brightness: `activity`, `potential` or `change`.
Brains with more synapses than `particleBudget` (300,000) draw particles for a uniform
sample of them; every synapse is still rasterised. Scroll zooms, drag pans, hover inspects.
WebGL2 draws it; without it, a Canvas2D fallback draws the neurons.

### The brain style

`style: "brain"` wraps the same net into the volume of one stylised animal brain, in three
dimensions: two smooth lobed hemispheres, a cerebellum behind and below, a short stem, one
generic brain for every net, drawn as a translucent shell with a soft rim light, viewed from
the side at a slight angle, turning slowly on its own. Every region is assigned a lobe by
its role and name: sensory regions (retina, senses, evidence) at the back in the occipital
area; memory regions (records, context) deep and low as the hippocampus; association
regions (belief, expectation, prediction) across the parietal and frontal cortex; motor
regions in the frontal strip; a governor, steering, monitor or readback patch at the front
as the prefrontal area; anything else in the temporal lobe. Regions of one lobe are tiled
inside it, and each region keeps the atlas's own arrangement (connected neurons near each
other) on the lobe's two long axes, the third axis scattered, so `setAtlas` keeps working
and a brain built at run time lands in the same stylized layout. `lobeOf(region)` and
`brainLayout(atlas)` are exported for a page's own overlays.

Neurons are glowing somata: brightness is the activation, a halo the heat, the colour the
region; the busiest region reads as a dense field of small cells. Every synapse is a cubic
Bezier ribbon in three dimensions from the presynaptic soma to the postsynaptic one, bowing
outward from the brain's centre (deterministic per synapse from the atlas seed), thin and
tapering with a bouton at its end. At rest the ribbons are very dark (`restAlpha`, 0.025),
a faint web. They light up in real time: on a synapse whose presynaptic neuron sends a
message, a glow runs along the path and a bright pulse with a tail travels along it, the
brightness by message times weight (relative to the 98th percentile of the weights) and
fading within a couple of seconds after the last step, so a settling reads as activity
sweeping from the sensory lobe at the back through belief and records to the motor strip at
the front, along the true synapses. The strongest synapses come first under `lineBudget`
(the web, the glow and the boutons) and `particleBudget` (the pulses). Everything is drawn
additively into a floating-point scene with a bloom pass and a depth fog, so the far side
of the brain recedes. The 3D positions, the bow of every synapse and the shell mesh are
uploaded once per atlas or weights change; the vertex shader evaluates the curves, and a
frame rebuilds nothing.

Drag rotates the brain (shift-drag or a right-button drag pans), scroll zooms, and `fit()`
frames the whole shell again. `snapshot().view` is the current yaw and pitch; `screen(i)`,
`screenAll()` and `inspect` project the 3D positions. Options: `restAlpha` for the resting
web, `bloom` (0.5) for the bloom's gain, `shell: false` hides the shell, `spin: false` stops
the self-rotation and `spinRate` (0.12 radians per second) sets its speed. `mode`,
`particles`, `edges` and the budgets mean the same as in the scan style; in either style
a page feeds `setWeights` with the weights or with their last change, and the ribbons or
lines follow.

The brain style also draws a measured anatomy. An atlas that carries `positions3`, one soma
position per neuron in three dimensions, centred and scaled so the longest extent spans the
unit ball, is drawn at those positions instead of the generic lobes: no shell unless the page
sets `shell: true`, the labels hung from each region's members, the point sizes from
`spacing3` (the distance to a neuron's neighbours from the local density in three
dimensions, computed when the payload lacks it), a frontal resting view, and `fit()` framing
the projected bounding boxes. `layoutAtlas({positions3: {'*': array}})` builds such an atlas
in the browser from an array of `[x, y, z]` or a flat array of 3n floats and derives the
scan's own square from the (x, y) projection; `fitPositions3`, `spacing3Of` and
`anatomyLayout` are exported for a page's own overlays. Payloads without the keys draw as
before.

The renderer draws the connectome and the states the caller supplies; it does not qualify their convergence. The example pages draw the records
cortex beside the scan with their own component, `records_view.js`: the granule raster with
the active cells of the executed reading, imagined reads, writes scaled by the record rate,
the habituated reading and the per-field reads.

`layoutAtlas({ n, pre, post, weight, groups, shapes, positions, roles, labels, seed })` is the
same layout in the browser, for pages that build their brains at run time: `groups` names a
region per neuron, `shapes` declares sheets, `positions` supplies coordinates per region or,
under the key `"*"`, one shared anatomical frame for every neuron. `scan.setAtlas(atlas)`
loads a new brain into the same canvas.

## One call for a page

`atlas.page(*, frames=None, brain=None, title="Cadence brain scan", note=..., inputs=None, limit=2000)`
returns a self-contained HTML page: the atlas, the renderer, and either recorded frames or
a live brain, or both. `note` is the text under the title; `inputs` names the neurons that
the live page's `Detune` drives, by default the first population whose name reads as
vision or sensory, or else the first eight neurons.

```python
records = []
with cd.record_settlements(records.append, label="probe"):
    brain.settle_batch(drive, steps=60, tolerance=None)
frames = atlas.frames_from_record(records[0])            # eight bits per neuron and step
html = atlas.page(frames=frames, brain=brain, title="A brain settles")
```

With `brain=`, the page embeds the dense weights and settles in the browser: `Detune` draws
a new stimulus on the input region and the brain settles into its new equilibrium step by
step, at the chosen rate, with the certificate's error bound in the status line when the
brain is certified. The dense matrix is `n * n` numbers, so the live option is capped at
2,000 neurons (`limit=` raises it). Larger brains use recorded frames or their page's own
engine. Write the returned `html` with `Path("brain.html").write_text(html, encoding="utf-8")`.

## A page with its own engine

A sparse engine for large brains, with the nudged phase and a one-stream actor-critic, and a
parity harness that holds it to the library, lives in
[cadence-examples/engine](https://github.com/muellerberndt/cadence-examples/tree/main/engine)
with a guide for building a browser example. The fruit fly uses this rate-model engine;
the worm temporal model has its own engine. The dense recipe below is the small-brain version.

For settling without adaptation, a page can store the effective weight matrix, the bias
vector and the neuron model's activation, and settle the brain locally, then feed every
step to the scan. `Brain.dense()` returns the synapse matrix with every effective drive
folded in (`gain`, `count`, `efficacy`, `log_gain`).

```python
payload = {
    "n": connectome.n,
    "W": brain.dense().ravel().tolist(),   # W[pre, post], row-major
    "bias": brain.bias.tolist(),
    "neuron": {"slope": neuron_model.slope, "threshold": neuron_model.threshold,
               "leak": neuron_model.leak, "dt": neuron_model.dt,
               "stimulus": neuron_model.stimulus_amplitude, "rest": neuron_model.rest_emission},
}
```

```javascript
const n = payload.n, W = payload.W, BIAS = payload.bias, NEURON = payload.neuron;
function act(v) {
  const r = 1 / (1 + Math.exp(-NEURON.slope * (v - NEURON.threshold))) - NEURON.rest;
  return r > 0 ? r / (1 - NEURON.rest) : NEURON.leak * r / NEURON.rest;
}
function settle(drive, scan) {              // drive: one number per neuron, the stimulus
  const v = new Float64Array(n), s = new Float64Array(n), synapticInput = new Float64Array(n);
  scan.reset(s);
  for (let t = 0; t < 100; t++) {
    synapticInput.fill(0);
    for (let i = 0; i < n; i++) { const si = s[i]; if (si === 0) continue;
      for (let j = 0; j < n; j++) synapticInput[j] += si * W[i * n + j]; }
    let moved = 0;
    for (let j = 0; j < n; j++) {
      v[j] += NEURON.dt * (-v[j] + synapticInput[j] + drive[j] + BIAS[j]);
      const ns = act(v[j]); moved = Math.max(moved, Math.abs(ns - s[j])); s[j] = ns;
    }
    scan.step(s);
    if (moved < 1e-4) break;
  }
  return s;
}
```

Compare its outputs against `brain.settle` on the CPU backend with `steps=100` and
`tolerance=1e-4` on representative drives. JavaScript uses double-precision numbers, but
operation ordering can change rounding. An activation stopping tolerance alone does not
certify an equilibrium; a [certificate](https://github.com/muellerberndt/cadence/blob/main/docs/certificate.md)
requires its stated contraction assumptions, and only then can a page show an error
bound from the last movement. Otherwise independently evaluate the full fixed-input
equation defect before authorizing outputs. Keep a website's numerical kernel separate from its
rendering so the same recorded observations can check it against Python.

## Python API (`viewer.atlas`)

- `build_atlas(connectome, weights=None, *, regions=None, shapes=None, positions=None, roles=None, seed=0, iterations=24) -> Atlas`:
  one layout of a whole connectome. `weights` are effective synaptic weights in connectome
  order (`brain.weights`); without them the contact counts stand in. `regions` partitions
  the neurons by name, by default the connectome's populations, coarsest first, each neuron
  in one region and the rest in `other`. `shapes` declares sheets `(rows, cols[, channels])`
  per region name and places them on a grid; `positions` supplies `(count, 2)` coordinates
  per region name, fitted into the region's place, or under `"*"` one `(n, 2)` frame for
  every neuron, scaled into the square; `roles` overrides the role a region's name
  suggests; `iterations` counts the neighbour-averaging passes that place the remaining
  neurons by their synapses. Deterministic under `seed`.
- `atlas_of(brain, **options) -> Atlas`: `build_atlas(brain.connectome, brain.weights, **options)`.
- `Atlas`: `n`, `positions` (`(n, 2)`, both axes in `[-1, 1]`), `region_index`, `regions`
  (layout records of `viewer.atlas.Region` with name, role, colour, centre, extent and
  size, distinct from the `cadence.regions.Region` of a genome), `pre`, `post`, `weight`,
  `seed` and `extras`; property `synapses`. `region_of(neuron)`; `summary()` (neurons,
  synapses, and each region's name, role and size); `to_dict()` and `to_json()` (the
  `cadence.atlas/v1` payload with base64 arrays); `subsample_edges(limit, seed=None)` (at
  most `limit` synapses, drawn with probability proportional to absolute weight);
  `frames(activation, potential=None)` (recorded `(steps, n)` settling steps quantised to
  eight bits per neuron and step); `frames_from_record(record, row=0)` (the frames of one
  batch row of a `SettlementRecord`);
  `page(*, frames=None, brain=None, title="Cadence brain scan", note=..., inputs=None, limit=2000) -> str`
  (a self-contained HTML page; `note` is the text under the title, `inputs` the neurons the
  live page's `Detune` drives, and `limit` the largest live brain, whose dense weight matrix
  the page embeds; a live brain with adaptation raises `ValueError`).
- `brain_scan_script() -> str`: the source of the shipped renderer `brain_scan.js`.
- `viewer.atlas.role_of(name, roles=None)`: the role a region name declares, by an explicit
  map or by its wording; `viewer.atlas.PALETTE` maps each role to its colour. See
  [renderer options above](#the-renderer).

