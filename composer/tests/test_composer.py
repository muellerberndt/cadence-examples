"""Behavioral contracts: no invented telemetry, no silent token shortcut, isolated futures."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from composer.brain import Design, build, drives, settle_checked
from composer.compose import Composer
from composer.encoding import music_features
from composer.music import critique, parse_prompt, write_midi
from composer.telemetry import capture


def test_style_brief_and_exact_minute(tmp_path):
    import mido

    for text in ["sad piano", "heroic orchestral theme", "calm baroque"]:
        b = parse_prompt(text)
        events = [{"pitch": 60, "step": 0, "duration": 4, "chord": 0}]
        p = tmp_path / "piece.mid"
        write_midi(events, b, p)
        assert abs(mido.MidiFile(p).length - 60) < 0.0001
    assert (
        parse_prompt("John Williams inspired orchestral theme").instrument
        == "orchestra"
    )
    with pytest.raises(ValueError):
        parse_prompt("")


def test_exact_populations_and_equation_error():
    b = build(Design(8, 4, 4))
    history = np.array([[[24, 3, 0, 12]] * 4])
    extra = np.array([music_features(history[0], 0, 0)])
    d = drives(b, history, extra)
    observation = capture(b, d, steps=32, include_release=False)
    final = b.brain.settle_batch(d, steps=32, tolerance=0)
    np.testing.assert_allclose(
        observation["equation_error"][-1], b.brain.residual(d, final)[0], atol=1e-12
    )
    for name, ids in b.brain.connectome.populations.items():
        np.testing.assert_allclose(
            observation["populations"][-1][name]["mean"],
            final.activation[0, list(ids)].mean(),
            atol=1e-12,
        )
    assert (
        sum(r["neurons"] for r in observation["regions"].values())
        == b.brain.connectome.n
    )


def test_actual_weight_learning_and_isolated_candidate_rehearsal(tmp_path):
    brain = build(Design(8, 4, 4))
    file = tmp_path / "brain.npz"
    brain.save(file)
    c = Composer(file)
    before = c.learner.brain.efficacy.copy()
    previous = [[24, 3, 0, 12]] * 4
    selected, _history, record, _drive = c.phrase(
        parse_prompt("calm piano"), previous, 0, 1, np.random.default_rng(7), variants=3
    )
    assert len(record["candidates"]) == 3
    assert selected == record["candidates"][record["winner"]]["notes"]
    assert record["winner"] == int(
        np.argmax([r["critique"]["score"] for r in record["candidates"]])
    )
    np.testing.assert_array_equal(before, c.learner.brain.efficacy)
    assert previous == [[24, 3, 0, 12]] * 4
    assert (
        selected[0]["step"] == 0
        and selected[-1]["step"] + selected[-1]["duration"] == 16
    )
    assert all(
        r["residual"] <= r["tolerance"] or r["steps"] == 512 for r in record["settling"]
    )
    context = np.array([previous])
    extra = np.array([music_features(previous, 0, 0)])
    c.learner.step(drives(c.learner, context, extra), np.array([[28, 3, 0, 12]]))
    assert np.max(np.abs(before - c.learner.brain.efficacy)) > 0


def test_critic_penalizes_note_collapse():
    brief = parse_prompt("calm piano")
    stagnant = [
        {"pitch": 60, "step": i * 4, "duration": 4, "chord": 0} for i in range(16)
    ]
    moving = [
        {**e, "pitch": [60, 64, 67, 64, 62, 64, 65, 67][i % 8]}
        for i, e in enumerate(stagnant)
    ]
    assert critique(moving, brief)["score"] > critique(stagnant, brief)["score"]


def test_one_trial_motif_memory_changes_shared_equilibrium():
    from composer.motif import MotifBrain

    b = MotifBrain(build(Design(8, 4, 4)))
    n = b.brain.connectome.n
    d = np.zeros((1, n))
    d[:, b.cues[0]] = 2.5
    before, _ = settle_checked(b, d)
    b.remember([60, 62, 64, 65, 67, 64])
    after, receipt = settle_checked(b, d)
    assert receipt["converged"]
    assert after.activation[0, b.recall[24]] > before.activation[0, b.recall[24]] + 0.5
    assert (
        after.activation[0, b.output_index[24]]
        > before.activation[0, b.output_index[24]] + 0.3
    )
    assert b.memory.writes == 6
    assert len(b.brain.connectome.populations) == (
        10 if b.intuition.tables is not None else 7
    )


def test_phrase_density_is_independent_of_location():
    b = parse_prompt("calm piano")
    events = [
        {"pitch": p, "step": i * 4, "duration": 4, "chord": 0}
        for i, p in enumerate([60, 64, 67, 62, 65, 64, 62, 60])
    ]
    later = [{**e, "step": e["step"] + 128} for e in events]
    assert critique(events, b) == critique(later, b)


def test_render_uses_committed_harmony_and_breathing(tmp_path):
    import mido

    b = parse_prompt("calm piano")
    events = [{"pitch": 66, "step": 0, "duration": 8, "chord": 6, "gate": 0.5}]
    file = tmp_path / "score.mid"
    write_midi(events, b, file)
    midi = mido.MidiFile(file)
    tick = 0
    accompaniment = []
    for m in midi.tracks[2]:
        tick += m.time
        if m.type == "note_on" and m.velocity and tick < 1920:
            accompaniment.append(m.note % 12)
    assert set(accompaniment) == {6, 10, 1}
    end = sum(m.time for m in midi.tracks[1] if m.type != "end_of_track")
    assert (
        end == 480
    )  # eight-sixteenth interval, half-gate: a real rest before the next onset.


def test_motif_cue_releases_after_the_opening():
    from composer.motif import MotifBrain

    b = MotifBrain(build(Design(8, 4, 4)))
    b.remember([60, 62, 64, 65, 67, 64])
    drive = np.zeros((1, b.brain.connectome.n))
    b.cue(drive, 6)
    assert not drive.any()


def test_abstract_interval_expectations_ignore_absolute_key():
    from composer.intuition import Intuition

    memory = Intuition(Path("/nonexistent/expectations.npz"))
    memory.tables = {"intervals": np.arange(625).reshape(25, 25)}
    notes = [{"pitch": p} for p in [60, 62, 65, 64, 67, 65]]
    shifted = [{"pitch": n["pitch"] + 7} for n in notes]
    assert memory.evaluate(notes) == memory.evaluate(shifted)


def test_preference_learns_relative_patterns_without_an_exact_score():
    from composer.intuition import Intuition

    b = parse_prompt("calm piano")

    def memory():
        m = Intuition(Path("/nonexistent/expectations.npz"))
        m.tables = {
            "chords": np.ones((2, 24, 24)),
            "rhythm": np.ones((3, 4, 12, 12)),
            "intervals": np.ones((25, 25)),
        }
        return m

    notes = [
        {
            "pitch": p,
            "step": i * 4,
            "duration": 4,
            "chord": 0,
            "token": [p - 36, 3, 0, 12],
        }
        for i, p in enumerate([60, 62, 64, 66, 68, 70])
    ]
    shifted = [
        {**e, "pitch": e["pitch"] + 7, "token": [e["token"][0] + 7, *e["token"][1:]]}
        for e in notes
    ]
    a, c = memory(), memory()
    before = a.interval_probability(2)[14]
    a.teach(notes, b, 1)
    c.teach(shifted, b, 1)
    assert a.interval_probability(2)[14] > before
    for name in a.tables:
        np.testing.assert_array_equal(a.tables[name], c.tables[name])
    d = memory()
    d.teach(notes, b, -1)
    assert d.interval_probability(2)[14] < before


def test_corpus_expectation_has_a_causal_path_to_note_intention():
    from composer.intuition import Intuition
    from composer.motif import MotifBrain

    memory = Intuition(Path("/nonexistent/expectations.npz"))
    memory.tables = {"chords": np.ones((2, 24, 24)), "rhythm": np.ones((3, 4, 12, 12))}
    memory.tables["chords"][0, 0, 7] = 10000
    b = MotifBrain(build(Design(8, 4, 4)), memory)
    d = np.zeros((1, b.brain.connectome.n))
    d[0, b.expectation_cues[0]] = 2.5
    intact, _ = settle_checked(b, d)
    mask = np.ones(b.brain.connectome.n)
    mask[list(b.brain.connectome.populations["harmonic_expectation"])] = 0
    lesioned = b.brain.settle_batch(d, mask=mask, steps=512, tolerance=0)
    # The learned dominant expectation changes the actual chord-intention neuron.
    chord_neuron = b.output_index[61 + 12 + 7]
    assert (
        intact.activation[0, chord_neuron] > lesioned.activation[0, chord_neuron] + 0.01
    )


def test_motif_critic_uses_phrase_openings_not_a_chance_window():
    brief = parse_prompt("calm piano")
    pitches = [60, 62, 64, 65, 67, 70, 65, 60, 62, 64, 65, 67]
    events = [
        {"pitch": p, "step": i * 4, "duration": 4, "chord": 0}
        for i, p in enumerate(pitches)
    ]
    assert critique(events, brief)["motif_contour_similarity"] == 0
    # An external theme is compared with this phrase's opening, not its best later fragment.
    reverse = [
        {**e, "pitch": p}
        for e, p in zip(events, [70, 68, 66, 65, 63, 60, 62, 64, 65, 67, 70, 72])
    ]
    assert critique(reverse, brief, motif=pitches[:5])["motif_contour_similarity"] == 0


def test_all_bad_candidates_can_produce_negative_valence(tmp_path):
    file = tmp_path / "brain.npz"
    build(Design(8, 4, 4)).save(file)
    composer = Composer(file)
    composer.appraise([4.4, 4.5, 4.6])
    worse = composer.appraise([1.0, 1.2, 1.3])
    assert worse["valence"] < 0
    assert worse["selected_quality"] < worse["expected_quality"]
