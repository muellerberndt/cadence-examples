"""Multitrack examples preserve simultaneous notes, duration and causal heard-state inputs."""

import sys
from pathlib import Path

import mido
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from composer.brain import Design, build, settle_checked
from composer.ensemble import INPUTS, encode, family
from composer.telemetry import Recorder
from composer.topology import topology
from tools import prepare_ensemble


def score(path, altered=False):
    midi = mido.MidiFile(ticks_per_beat=480)
    for program, channel in [(48, 0), (60, 1)]:
        track = mido.MidiTrack(
            [mido.Message("program_change", program=program, channel=channel)]
        )
        events = []
        for i in range(40):
            pitch = 60 + (i % 4) * 2 + (12 if channel else 0)
            if altered and i == 8 and channel == 0:
                pitch += 1
            events.extend([(i * 480, True, pitch), (i * 480 + 720, False, pitch)])
        previous = 0
        for tick, on, pitch in sorted(events, key=lambda e: (e[0], e[1])):
            track.append(
                mido.Message(
                    "note_on" if on else "note_off",
                    note=pitch,
                    velocity=80 if on else 0,
                    channel=channel,
                    time=tick - previous,
                )
            )
            previous = tick
        midi.tracks.append(track)
    midi.save(path)


def test_polyphony_held_notes_and_causal_inputs(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    path = tmp_path / "data/example.mid"
    monkeypatch.setattr(prepare_ensemble, "ROOT", tmp_path)
    # Key and mood are supplied conditioning, held equal for this causal test.
    monkeypatch.setattr(prepare_ensemble, "tonal_center", lambda *args: (0, 0))
    meta = {"mid": "example.mid", "composer_name": "fixture"}
    score(path)
    context, extra, labels, _ = prepare_ensemble.parse((meta, 1000))
    assert set(labels[:, 3]) == {1, 2}
    assert (labels[:, 2] == 0).any()  # simultaneous cross-instrument attacks survive
    assert np.count_nonzero(extra[:, 17:].reshape(-1, 8, 12)[:, 1:3]) > 0
    assert set(labels[:, 4]) == {5}
    assert encode(context, extra).shape == (len(labels), INPUTS)
    score(path, altered=True)
    other_context, other_extra, other_labels, _ = prepare_ensemble.parse((meta, 1000))
    changed = np.flatnonzero(np.any(labels != other_labels, axis=1))[0]
    np.testing.assert_array_equal(context[: changed + 1], other_context[: changed + 1])
    np.testing.assert_array_equal(extra[: changed + 1], other_extra[: changed + 1])
    assert family(0, 9) == 7


def test_full_topology_and_actual_warm_state_recorder():
    learner = build(Design(8, 4, 4))
    brain = learner.brain
    graph = topology(brain)
    file = (
        Path(__file__).resolve().parents[1] / "runs/topology" / Path(graph["url"]).name
    )
    edges = np.fromfile(file, dtype="<f4").reshape(-1, 3)
    order = np.asarray(graph["neuron_ids"])
    inverse = np.argsort(order)
    np.testing.assert_array_equal(edges[:, 0], inverse[brain.connectome.pre])
    np.testing.assert_array_equal(edges[:, 1], inverse[brain.connectome.post])
    np.testing.assert_allclose(edges[:, 2], brain.weights, rtol=1e-7)
    assert len(edges) == brain.connectome.synapses and len(order) == brain.connectome.n
    first = np.ones((1, brain.connectome.n)) * 0.1
    warm, _ = settle_checked(learner, first)
    old = warm.activation.copy()
    next_drive = first.copy()
    next_drive[0, 0] = 2
    recorder = Recorder(learner, next_drive)
    final, receipt = settle_checked(
        learner, next_drive, warm=warm, observer=recorder.observe
    )
    np.testing.assert_allclose(recorder.frames[0]["activation"], old[0, order])
    np.testing.assert_allclose(
        recorder.frames[-1]["activation"], final.activation[0, order]
    )
    np.testing.assert_allclose(
        warm.activation, old
    )  # caller-owned candidate state stays intact
    assert receipt["converged"]
    assert (
        recorder.finish(packed=True)["encoded_frames"]["shape"][2] == brain.connectome.n
    )
