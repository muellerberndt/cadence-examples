"""Contracts of the musician brain: causal senses, one joint connectome with contiguous
regions, isolated futures, stream resets at piece boundaries, review snapshots and custody."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from composer.musician import (
    CYCLE,
    DELTAS,
    DURATIONS,
    EVENT,
    SENSE,
    SIZES,
    WINDOW,
    Design,
    Musician,
    Senses,
    build,
    encode_events,
    encode_mood,
    encode_sense,
    events_from_tokens,
    mood_classes,
    write_score,
)

TINY = Design(cortex=16, phrase=16, embedding=4, seed=3)


def random_tokens(rng, count):
    return np.column_stack(
        [rng.integers(0, size, count) for size in SIZES]
    ).astype(np.int64)


def test_regions_are_contiguous_disjoint_and_cover_the_brain():
    m = build(TINY)
    w = m.brain.connectome
    covered = np.zeros(w.n, bool)
    for name, ids in w.populations.items():
        ids = np.asarray(ids)
        assert np.array_equal(ids, np.arange(ids[0], ids[0] + len(ids))), name
        assert not covered[ids].any()
        covered[ids] = True
    assert covered.all()
    assert m.brain.layout.pairs <= 256 and m.brain._blocked
    # the shared ear: one embedding table across the window positions
    tie = m.learner.tie_groups
    tied = tie[tie >= 0]
    assert len(np.unique(tied)) == EVENT * TINY.embedding
    assert m.learner.parameters() < w.synapses


def test_senses_are_causal_and_clocked():
    s = Senses(total_steps=64)
    first = s.raw()
    assert first[0] == 0 and first[1] == 0 and first[2] == 0 and first[3:].sum() == 0
    s.observe([36, 3, 0, 0, 5])  # pitch 60, one beat, no gap, keys
    row = s.raw()
    assert row[0] == 0 and row[15 + 0 * 12 + 0] == 255  # C sounding in the keys family
    assert row[3 + 0] == 255  # chroma C
    s.observe([40, 3, 4, 1, 5])  # a beat later, strings E
    row = s.raw()
    assert row[0] == 4 and row[15 + 12 + 4] == 255 and row[15 + 0] == 0  # C released
    for _ in range(20):
        s.observe([40, 3, 12, 1, 5])
    assert s.raw()[1] == (s.step // 16) % CYCLE and s.raw()[2] == 3


def test_encoders_are_one_hot_where_declared():
    rng = np.random.default_rng(0)
    context = np.stack([random_tokens(rng, WINDOW) for _ in range(3)])
    x = encode_events(context)
    assert x.shape == (3, WINDOW * EVENT) and (x.sum(1) == WINDOW * len(SIZES)).all()
    raw = np.zeros((2, 111), np.uint8)
    raw[:, 0] = [3, 15]
    raw[:, 1] = [1, 15]
    raw[:, 2] = [0, 3]
    sense = encode_sense(raw)
    assert sense.shape == (2, SENSE) and (sense[:, :36].sum(1) == 3).all()
    mood = encode_mood(np.array([[1, 2, 0, 3, 2, 2, 1]]))
    assert mood.sum() == 7 and mood[0, 1] == 1


def test_mood_classes_measure_the_score():
    calm = [(k * 8, 8, 60 + 2 * (k % 3), 80, 0) for k in range(40)]
    busy = [(k, 1, 84 + (k % 5), 40 + (k % 7) * 12, k % 4) for k in range(40)]
    a, b = mood_classes(calm, 70, 0, 0), mood_classes(busy, 160, 0, 1)
    assert list(a) == [0, 0, 0, 0, 1, 0, 0]
    assert b[0] == 1 and b[1] == 2 and b[2] == 2 and b[3] > 0 and b[4] == 2 and b[5] == 2


def test_stream_reset_rows_and_working_memory_carry():
    m = build(TINY)
    rng = np.random.default_rng(1)
    state = m.fresh(2)
    context = np.stack([random_tokens(rng, WINDOW) for _ in range(2)])
    raw = np.zeros((2, 111), np.uint8)
    mood = np.zeros((2, 7), np.uint8)
    _, free = m.predict(context, raw, mood, state)
    m.advance(state, free, raw)
    assert np.abs(state.trace.trace).sum() > 0 and state.memory.writes == 2
    drive_carried = m.drive(context, raw, mood, state)
    assert np.abs(drive_carried[:, m.populations["prefrontal"]]).sum() > 0
    state.reset_rows([0])
    assert np.abs(state.trace.trace[0]).sum() == 0 and np.abs(state.trace.trace[1]).sum() > 0
    assert np.abs(state.memory.strength[0]).sum() == 0
    assert np.abs(np.asarray(state.warm.activation)[0]).sum() == 0


def test_futures_are_isolated_and_never_write_the_live_state():
    m = build(TINY)
    rng = np.random.default_rng(2)
    tokens = random_tokens(rng, WINDOW).tolist()
    senses = Senses(256)
    for t in tokens:
        senses.observe(t)
    state = m.fresh(1)
    trace_before = state.trace.trace.copy()
    weights_before = m.brain.efficacy.copy()
    out = m.imagine(tokens, senses, np.zeros(7, int), futures=3, horizon=4, rng=rng)
    assert len(out["tokens"]) == 3 and all(len(t) == 4 for t in out["tokens"])
    assert len({tuple(map(tuple, t)) for t in out["tokens"]}) > 1  # sampled alternatives differ
    np.testing.assert_array_equal(trace_before, state.trace.trace)
    np.testing.assert_array_equal(weights_before, m.brain.efficacy)
    assert senses.step == sum(DELTAS[t[2]] for t in tokens)
    assert all(0 <= s for row in out["surprise"] for s in row)
    limited = m.imagine(tokens, senses, np.zeros(7, int), futures=2, horizon=40, rng=rng, stop_at=senses.step + 8)
    assert all(s.step <= senses.step + 8 for s in limited["senses"])


def test_review_measures_every_event_and_snapshots_each_bar():
    m = build(TINY)
    rng = np.random.default_rng(3)
    tokens = random_tokens(rng, 48)
    tokens[:, 2] = 4  # one beat apart: twelve bars
    report = m.review(tokens.tolist(), np.zeros(7, int), total_steps=48 * 4)
    assert len(report["surprise"]) == 48 - WINDOW
    assert set(report["snapshots"]) == set(report["bars"].tolist())
    snapshot, senses, length = report["snapshots"][int(report["bars"][0])]
    assert length == WINDOW and senses.step == 0  # the prime is heard with the clock rewound
    assert snapshot.trace.trace.shape[0] == 1


def test_save_load_and_score(tmp_path):
    m = build(TINY)
    path = m.save(tmp_path / "musician.npz")
    back = Musician.load(path)
    assert back.design == TINY
    np.testing.assert_array_equal(back.brain.efficacy, m.brain.efficacy)
    rng = np.random.default_rng(4)
    tokens = random_tokens(rng, 20)
    events = events_from_tokens(tokens.tolist(), key=2)
    assert len(events) == 20 and all(0 <= e[2] <= 127 for e in events)
    assert events[3][0] == int(DELTAS[tokens[:4, 2]].sum())
    assert events[3][1] == int(DURATIONS[tokens[3, 1]])
    import mido

    file = write_score(events, tmp_path / "score.mid", bpm=100)
    midi = mido.MidiFile(file)
    assert sum(1 for t in midi.tracks for msg in t if msg.type == "note_on" and msg.velocity) == 20


def test_replay_reproduces_the_free_phase_iteration_by_iteration():
    m = build(TINY)
    rng = np.random.default_rng(5)
    tokens = random_tokens(rng, WINDOW + 6).tolist()
    senses = Senses(128)
    for t in tokens[:WINDOW]:
        senses.observe(t)
    mood = np.zeros(7, int)
    # the reference: the same events through predict/advance
    state = m.fresh(1)
    reference = []
    history = [list(t) for t in tokens[:WINDOW]]
    probe = _copy(senses)
    for token in tokens[WINDOW:]:
        raw = probe.raw()[None, :]
        _, free = m.predict(np.array([history[-WINDOW:]]), raw, mood[None, :], state)
        reference.append(np.array(free.activation[0]))
        m.advance(state, free, raw)
        history.append(list(token))
        probe.observe(token)
    seen = {}
    last = {}

    def observer(k, iteration, activation, repair, mismatch, drive):
        seen[k] = iteration
        last[k] = activation.copy()
        assert activation.shape == (m.n,) and mismatch.shape == (m.n,) and repair.shape == (m.n,)

    _, _, surprise = m.replay(tokens[:WINDOW], senses, m.fresh(1), tokens[WINDOW:], mood, observer)
    assert list(seen.values()) == [m.learner.config.free_steps] * 6
    for k, ref in enumerate(reference):
        np.testing.assert_allclose(last[k], ref, atol=1e-9)
    assert len(surprise) == 6


def _copy(senses):
    out = Senses(senses.total)
    out.chroma = senses.chroma.copy()
    out.held = list(senses.held)
    out.step = senses.step
    return out


def test_version_two_design_round_trips_and_carries_two_memories(tmp_path):
    design = Design(cortex=16, phrase=16, embedding=4, seed=5, version=2, form=8, tonic=0.7, eta=0.003)
    m = build(design)
    w = m.brain.connectome
    for name in ("interval", "form", "piece"):
        assert name in w.populations
    assert np.all(m.brain.bias[list(w.populations["form"])] == 0.7)
    assert np.all(m.brain.bias[list(w.populations["ear"])] == 0.0)
    assert m.learner.config.eta == 0.003
    state = m.fresh(2)
    assert state.piece is not None and state.piece.decay == design.piece_decay
    rng = np.random.default_rng(6)
    context = np.stack([random_tokens(rng, WINDOW) for _ in range(2)])
    raw = np.zeros((2, 111), np.uint8)
    _, free = m.predict(context, raw, np.zeros((2, 7), int), state)
    m.advance(state, free, raw)
    assert np.abs(state.piece.trace).sum() > 0
    drive = m.drive(context, raw, np.zeros((2, 7), int), state)
    assert np.abs(drive[:, m.populations["piece"]]).sum() > 0
    assert drive[:, m.populations["interval"]].sum() == 2 * 8 * 2.5
    path = m.save(tmp_path / "v2.npz")
    back = Musician.load(path)
    assert back.design == design and "form" in back.populations
