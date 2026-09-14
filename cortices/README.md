# Cortex catalogue

A small set of basic cortices for [cadence](https://github.com/muellerberndt/cadence)
brains. Each builder in [`cortices.py`](cortices.py) returns a cadence `Region` with named
input and output populations, plus the auxiliary object where one is needed (a `Trace`, a
`FastSynapses` record). `assemble_brain` develops a list of them into one connectome and
returns the mask of plastic neurons and the resting bias vector. Everything uses the public
cadence API: `cadence.regions`, `cadence.genome`, `cd.Trace`, `cd.FastSynapses`,
`cd.Learner`, `cd.Brain`.

A designed catalogue is one way to start a brain. The core principles of cadence are local
repair, checked convergence and equilibrium detuning. Minimal design plus plasticity is the
research direction; the catalogue gives a first layout that the learner, the genome and the
records then change.

What is supplied and what is learned: the layout, the population sizes, the sign and
density of each projection, the tonic level, the trace decay and the clock keys are
supplied. The efficacy of every plastic synapse, the bias of every free neuron and the
content of every record are learned.

## The cortices

| Builder | Returns | Clamped | Auxiliary |
|---|---|---|---|
| `sensory_sheet(n)` | one neuron per feature | all | |
| `visual_sheet(h, w, features, field)` | retina `input` and feature maps `output` | `input` | |
| `association(n, tonic=0.5)` | a blank cortex with a resting bias | none | |
| `motor(actions, lateral=-0.5)` | one neuron per action, population `actions` | none | |
| `working_memory(held)` | a prefrontal population sized to `held` | all | `Trace` |
| `clocked_record(keys, values)` | populations `clock` and `recall` | both | `FastSynapses` |
| `conditioning(classes)` | one-hot context classes | all | `condition(...)` projections |
| `assemble_brain(regions, projections, seed)` | connectome, plastic-neuron mask, bias | | |

### sensory_sheet

What it is: a clamped population, one neuron per feature. The environment writes the
feature vector into it as stimulus.

When to use it: any sense that already arrives as a vector.

Project from it with `reciprocal=False`. A reciprocal projection would write the brain's
state back into the sense.

Check: after settling, the sheet's activation follows the stimulus and nothing else. Zero
the stimulus and the downstream activity must change.

### visual_sheet

What it is: `visual_cortex` from cadence. Population `input` holds one neuron per pixel,
row by row, and is clamped to the picture. Population `output` holds `features` maps whose
neurons each see one `field` by `field` window. The receptive fields start random and are
learned.

When to use it: pictures, where neighbouring pixels belong together.

Check: mask one feature map and the held-out accuracy must drop; a sheet of the same size
with a full projection instead of local fields is the control.

### association

What it is: a blank cortex of `n` neurons with a resting bias of `tonic` on each.

The finding behind the tonic level: the learning neuron model is a rectified sigmoid that
is exactly zero at rest. With symmetric random projections about half of a stage receives
negative net input, sits at rest and carries no contrast, so the learning rule cannot move
it. A resting bias of about half a unit on the free neurons puts nearly every neuron on the
slope. In the catalogue's own check, a level of 0 left 30 percent of an association cortex
on the slope and gave erratic held-out accuracy; 0.5 put 96 to 99 percent on the slope and
gave the best accuracy across eight seeds and two widths; 1.0 put every neuron on the slope
but lost accuracy on the wider cortex. The bias applies to every free neuron, motor neurons
included: with the motor cortex left at rest the same task fell to chance. The learner moves
the bias from there.

`tonic_bias(connectome, populations, level)` returns the bias vector by hand: `level` on
every neuron, zero on the listed clamped populations.

When to use it: whenever senses must combine before a choice.

Check: the fraction of neurons with activation above a small threshold after a free
settling run (`live` in the tests); with the bias at zero the fraction is about a third.

### motor

What it is: `motor_cortex` with lateral inhibition, population `actions`.

When to use it: one discrete choice per moment, read as the most active neuron
(`Learner.predict`) or as a softmax over the actions. For several independent choices per
moment lay the actions out slot by slot and pass `slots=` to the learner.

Check: cut the lateral synapses and the choices must become less decisive; compare the
softmax margin.

### working_memory

What it is: a prefrontal population with one neuron per neuron of the region it holds, and
the `Trace` of that region's activity that is written into it as stimulus before every
settling run. The population is clamped. `projection()` is the one-way bundle from it back
into the held region. `trace(connectome)` builds the trace once the connectome exists.

The trace decays as `c <- decay * c + (1 - decay) * h` once per real moment, so it fades
over about `1 / (1 - decay)` moments: 0.5 keeps the last two, 0.85 the last several, 0.98
about the last fifty. A settled cortex is small, so `amplitude` is set high.

When to use it: a cue matters after it has disappeared.

Check: the same brain with the trace amplitude at zero. The catalogue test runs a delayed
response over five seeds and requires the memory to beat that control in the majority.

### clocked_record

What it is: a `FastSynapses` delta-rule record addressed by a clock or position key.
Population `clock` holds one neuron per beat or position and is driven one-hot by the
environment or by a rhythm region. Population `recall` holds one neuron per value feature;
`store(connectome).stimulate(drive)` writes the record's read into it as stimulus. Read
the beat before writing the new item there: the read is then the item from one period
earlier.

When to use it: the answer is what happened `n` beats ago or at a position.

Check: recall against lag or span, with a control that has no clock-keyed store.

### conditioning

What it is: a clamped population of one-hot context classes: a mode, a mood, a task.
`condition(context, regions)` projects it one way into every free population of the listed
regions, so the same brain settles differently under each class.

This is supplied conditioning, not understanding. The brain is told which situation it is
in; it does not find out.

Check: hold the class fixed and the brain must lose the task that needed it; swap classes
between two cues and the choices must swap.

### assemble_brain

What it is: `Genome` and `develop` behind one call. It returns the connectome, a
`plastic_neurons` mask that is False on every clamped population, and the bias vector:
`tonic` on free neurons, a cortex's own `tonic` where it set one, zero on clamped
populations. Pass the mask as `plastic_neurons=` to the learner and the bias as `bias=` to
the brain. `stimulus(connectome, {population: values})` writes named populations into a
drive.

Check: every region is one contiguous block, the blocks are disjoint and cover the
connectome; the bias is zero exactly on the clamped populations.

## Example: a delayed response

Senses, association, working memory and a choice. One of two cues is shown, two moments
of waiting follow, then a go signal. The right action names the cue, which is gone by the
time the go signal arrives. Run from a clone of cadence-examples with the root on the path
(`PYTHONPATH=. python example.py`).

```python
import numpy as np
import cadence as cd
from cortices import assemble_brain, association, motor, sensory_sheet, stimulus, working_memory

sense = sensory_sheet(3)  # cue A, cue B, go
cortex = association(24)
memory = working_memory(cortex)
choice = motor(2)
connectome, plastic, bias = assemble_brain(
    [sense, memory.region, cortex, choice],
    [
        cd.Projection("sensory", "association", reciprocal=False),
        memory.projection(),
        cd.Projection("association", "motor"),
    ],
    seed=0,
)
brain = cd.Brain(connectome, cd.learning_neuron_model(dt=1.0), bias=bias)
learner = cd.Learner(brain, list(connectome.populations["motor/actions"]), plastic_neurons=plastic)
trace = memory.trace(connectome)
frames = np.eye(3)

def episode(cues, learn):
    trace.reset(len(cues))
    for frame in [frames[cues], frames[[2] * len(cues)], frames[[2] * len(cues)]]:
        state = learner.free(trace.stimulate(stimulus(connectome, {"sensory": frame})))
        trace.update(state)  # once per real moment
    go = trace.stimulate(stimulus(connectome, {"sensory": frames[[2] * len(cues)]}))
    if learn:
        learner.step(go, cues)
        return None
    return learner.accuracy(go, cues)

rng = np.random.default_rng(0)
for _ in range(300):
    episode(rng.integers(0, 2, size=16), learn=True)
print("held-out accuracy:", episode(rng.integers(0, 2, size=200), learn=False))
```

This prints `held-out accuracy: 1.0` for seed 0. Chance is 0.5, and the same brain with
`working_memory(cortex, amplitude=0.0)` stays near chance. Not every seed reaches 1.0; the
test in `tests/test_cortices.py` states the majority condition.

## Example: a record addressed by time

Four items cycle through a period of three beats. At every beat the brain reads what
happened at this beat one period ago; then the new item is written there. The read enters
the `recall` population as stimulus and the association cortex settles with it.

```python
import numpy as np
import cadence as cd
from cortices import assemble_brain, association, clocked_record, stimulus

record = clocked_record(keys=3, values=4)
connectome, plastic, bias = assemble_brain(
    [record.region, association(8)],
    [cd.Projection("record", "association", reciprocal=False)],
)
brain = cd.Brain(connectome, cd.learning_neuron_model(dt=1.0), bias=bias)
store = record.store(connectome)
items = np.eye(4)
sequence = [0, 2, 1, 3, 3, 0, 2, 1, 1]
hits = []
for t, item in enumerate(sequence):
    beat = record.beats(t)
    drive = store.stimulate(stimulus(connectome, {"record/clock": beat}))  # recall as stimulus
    state = brain.settle_batch(drive, steps=40)
    recalled = state.activation[0, list(connectome.populations["record/recall"])]
    if t >= 3:
        hits.append(int(recalled.argmax()) == sequence[t - 3])
    store.observe(beat, items[[item]])  # write after the read
print("recall from one period earlier:", all(hits), len(hits))
```

This prints `recall from one period earlier: True 6`. The record is per stream: call
`store.reset(batch)` at episode boundaries. Records addressed by correlated keys drift; a
one-hot clock has orthogonal keys and does not.

## Tests

```bash
python -m pytest -q tests/test_cortices.py
```

The tests check that regions are contiguous and disjoint after assembly, that the tonic
bias is zero on clamped and at level on free neurons, that working memory carries a cue
across a delay better than the same brain without it (five seeds, majority), that the
clocked record returns the item from one period earlier, and that both examples above run.
