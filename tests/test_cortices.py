"""The cortex catalogue: layout, tonic bias, working memory, clocked records, README examples."""

import io
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import cadence as cd
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cortices import (  # noqa: E402
    assemble_brain,
    association,
    clamped_populations,
    clocked_record,
    condition,
    conditioning,
    motor,
    sensory_sheet,
    stimulus,
    tonic_bias,
    visual_sheet,
    working_memory,
)


def catalogue(seed=0, tonic=0.5):
    sense = sensory_sheet(6)
    vision = visual_sheet(5, 5, features=2, field=3)
    context = conditioning(3)
    record = clocked_record(3, 2)
    cortex = association(10, tonic=0.3)
    memory = working_memory(cortex)
    choice = motor(4)
    regions = [sense, vision, context, record.region, memory.region, cortex, choice]
    projections = [
        cd.Projection("sensory", "association", reciprocal=False),
        cd.Projection("visual", "association"),
        cd.Projection("record", "association", reciprocal=False),
        memory.projection(),
        cd.Projection("association", "motor"),
        *condition(context, regions),
    ]
    return regions, assemble_brain(regions, projections, seed=seed, tonic=tonic)


def test_regions_are_contiguous_disjoint_and_cover_the_connectome():
    regions, (connectome, plastic, bias) = catalogue()
    populations = connectome.populations
    seen = []
    for region in regions:
        block = populations[region.name]
        assert block == tuple(range(block[0], block[0] + len(block)))
        assert len(block) == region.size
        seen.extend(block)
        if region.circuit is not None:
            for name in region.circuit.populations:
                assert set(populations[f"{region.name}/{name}"]) <= set(block)
    assert sorted(seen) == list(range(connectome.n))
    assert plastic.shape == bias.shape == (connectome.n,)


def test_tonic_bias_is_zero_on_clamped_and_at_level_on_free_neurons():
    regions, (connectome, plastic, bias) = catalogue(tonic=0.7)
    populations = connectome.populations
    clamped = clamped_populations(regions)
    assert set(clamped) == {
        "sensory", "visual/input", "context", "record/clock", "record/recall", "prefrontal",
    }
    clamped_neurons = sorted(i for name in clamped for i in populations[name])
    assert np.all(bias[clamped_neurons] == 0.0)
    assert not plastic[clamped_neurons].any()
    free = np.ones(connectome.n, bool)
    free[clamped_neurons] = False
    assert plastic[free].all()
    assert np.all(bias[list(populations["association"])] == 0.3)  # the cortex set its own level
    assert np.all(bias[list(populations["motor"])] == 0.7)
    assert np.all(bias[list(populations["visual/output"])] == 0.7)
    by_hand = tonic_bias(connectome, clamped, 0.7)
    by_hand[list(populations["association"])] = 0.3
    assert np.array_equal(by_hand, bias)


def test_conditioning_projects_one_way_into_free_populations_only():
    regions, _ = catalogue()
    context = next(r for r in regions if r.name == "context")
    ends = {(p.pre, p.post, p.reciprocal) for p in condition(context, regions)}
    assert ends == {
        ("context", "visual/output", False),
        ("context", "association", False),
        ("context", "motor/actions", False),
    }


def test_tonic_bias_puts_the_association_on_the_slope():
    sense, cortex, choice = sensory_sheet(8), association(32), motor(2)
    projections = [cd.Projection("sensory", "association", reciprocal=False), cd.Projection("association", "motor")]
    live = {}
    for level in (0.0, 0.5):
        connectome, plastic, bias = assemble_brain([sense, association(32, tonic=level), choice], projections)
        brain = cd.Brain(connectome, cd.learning_neuron_model(dt=1.0), bias=bias)
        x = np.random.default_rng(0).uniform(0, 1, size=(32, 8))
        state = brain.settle_batch(stimulus(connectome, {"sensory": x}), steps=100)
        h = state.activation[:, list(connectome.populations["association"])]
        live[level] = float((h > 0.05).mean())
    assert live[0.0] < 0.5
    assert live[0.5] > 0.9


def delayed_response(seed, amplitude, delay=2, episodes=300):
    sense, cortex, choice = sensory_sheet(3), association(24), motor(2)
    memory = working_memory(cortex, amplitude=amplitude)
    connectome, plastic, bias = assemble_brain(
        [sense, memory.region, cortex, choice],
        [
            cd.Projection("sensory", "association", reciprocal=False),
            memory.projection(),
            cd.Projection("association", "motor"),
        ],
        seed=seed,
    )
    brain = cd.Brain(connectome, cd.learning_neuron_model(dt=1.0), bias=bias)
    learner = cd.Learner(brain, list(connectome.populations["motor/actions"]), plastic_neurons=plastic)
    trace = memory.trace(connectome)
    frames = np.eye(3)

    def episode(cues, learn):
        trace.reset(len(cues))
        for frame in [frames[cues]] + [frames[[2] * len(cues)]] * delay:
            state = learner.free(trace.stimulate(stimulus(connectome, {"sensory": frame})))
            trace.update(state)
        go = trace.stimulate(stimulus(connectome, {"sensory": frames[[2] * len(cues)]}))
        if learn:
            learner.step(go, cues)
            return None
        return learner.accuracy(go, cues)

    rng = np.random.default_rng(seed)
    for _ in range(episodes):
        episode(rng.integers(0, 2, size=16), True)
    return episode(rng.integers(0, 2, size=200), False)


def test_working_memory_carries_a_cue_across_a_delay_better_than_no_memory():
    seeds = range(5)
    with_memory = [delayed_response(s, amplitude=8.0) for s in seeds]
    without = [delayed_response(s, amplitude=0.0) for s in seeds]
    wins = sum(a > b + 0.2 for a, b in zip(with_memory, without))
    assert max(without) < 0.7, without
    assert wins >= 3, (with_memory, without)


def test_clocked_record_returns_the_item_from_one_period_earlier():
    record = clocked_record(keys=3, values=4)
    store = record.store()
    items = np.eye(4)
    rng = np.random.default_rng(0)
    sequence = rng.integers(0, 4, size=30)
    hits = []
    for t, item in enumerate(sequence):
        beat = record.beats(t)
        if t >= 3:
            hits.append(int(store.recall(beat).argmax()) == sequence[t - 3])
        store.observe(beat, items[[item]])
    assert len(hits) == 27 and all(hits)
    assert store.recall(record.beats(0)).shape == (1, 4)
    assert record.beats([0, 1, 2, 3]).tolist() == np.eye(3)[[0, 1, 2, 0]].tolist()


def test_clocked_record_recall_enters_an_assembled_brain_as_stimulus():
    record = clocked_record(keys=2, values=3)
    connectome, plastic, bias = assemble_brain(
        [record.region, association(4)], [cd.Projection("record", "association", reciprocal=False)]
    )
    store = record.store(connectome)
    recall = list(connectome.populations["record/recall"])
    beat = record.beats(0)
    store.observe(beat, np.eye(3)[[1]])
    drive = store.stimulate(stimulus(connectome, {"record/clock": beat}))
    assert drive[0, recall].argmax() == 1 and drive[0, recall].max() > 0.9
    assert not plastic[recall].any() and np.all(bias[recall] == 0.0)


def readme_examples():
    text = (ROOT / "cortices" / "README.md").read_text(encoding="ascii")
    return re.findall(r"```python\n(.*?)```", text, flags=re.S)


@pytest.mark.parametrize("index", [0, 1])
def test_readme_examples_run(index):
    examples = readme_examples()
    assert len(examples) == 2
    out = io.StringIO()
    with redirect_stdout(out):
        exec(compile(examples[index], f"README example {index}", "exec"), {"__name__": "__readme__"})
    printed = out.getvalue()
    if index == 0:
        assert printed.startswith("held-out accuracy:")
        assert float(printed.split(":")[1]) > 0.9
    else:
        assert printed.strip() == "recall from one period earlier: True 6"


def test_every_catalogue_file_is_ascii():
    for name in ("cortices/cortices.py", "cortices/__init__.py", "cortices/README.md", "tests/test_cortices.py"):
        (ROOT / name).read_text(encoding="ascii")
