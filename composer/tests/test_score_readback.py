"""Playing a score drives real neurons but cannot train on the playback."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataclasses import asdict

from composer.brain import Design, build
from composer.compose import Composer
from composer.music import parse_prompt


def test_readback_is_causal_and_does_not_write_weights(tmp_path):
    path = tmp_path / "brain.npz"
    build(Design(8, 4, 4)).save(path)
    c = Composer(path)
    events = [
        {
            "step": i * 4,
            "duration": 4,
            "pitch": p,
            "chord": 0,
            "token": [p - 36, 3, 0, 12],
        }
        for i, p in enumerate([60, 64, 67, 72, 65, 62, 60, 64])
    ]
    report = {
        "id": "test",
        "brief": asdict(parse_prompt("calm piano")),
        "events": events,
        "draft": events,
    }
    weights = c.performer.brain.weights.copy()
    trace = c.hear(report, 1)["trace"]
    assert trace["origin"].startswith("MIDI score readback")
    assert trace["topology"]["neurons"] == c.performer.brain.connectome.n
    first = c.hearing_state.activation.copy()
    changed = {
        **report,
        "id": "changed",
        "events": [
            e if e["step"] <= 4 else {**e, "token": [0, 3, 0, 12]} for e in events
        ],
    }
    c.hear(changed, 1)
    np.testing.assert_array_equal(first, c.hearing_state.activation)
    c.hear(report, 8)
    assert np.max(np.abs(first - c.hearing_state.activation)) > 0.01
    released = c.release()
    assert released["released"] and released["origin"].endswith("MIDI score readback")
    np.testing.assert_array_equal(weights, c.performer.brain.weights)
