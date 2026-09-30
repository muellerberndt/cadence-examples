"""Bounded adapter regressions; synthetic admissions exercise lesson custody."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worm.brain import WormBrain


class FakeBrain:
    """The admission surface of a cadence Brain, without a solve."""

    def __init__(self, accept=True):
        self.weights = (0.0, 0.0)
        self.biases = (0.0,)
        self.accept = accept
        self.calls = []

    def observe_batch(self, rows, *, source):
        self.calls.append((rows, source))
        if self.accept:
            self.weights = (1.0, -0.5)
            self.biases = (0.25,)
            return {"accepted": True, "reason": "qualified", "event_id": 3,
                    "energy": 0.5, "stationarity": 1e-7, "sweeps": 9}
        return {"accepted": False, "reason": "line_search", "event_id": None}


def worm(*, frozen=False, accept=True, ticks=3):
    b = WormBrain.__new__(WormBrain)
    b.p = {"teach": 2, "teach_level": 0.8, "window": 4}
    b.readouts = ["forward", "reverse"]
    b.frozen = frozen
    b.brain = FakeBrain(accept)
    b.stretch = [({"~X": (0.1 * t,)}, {"forward": (0.1 * t,), "reverse": (0.1 * t + 0.05,)})
                 for t in range(ticks)]
    b.lessons = b.rejected = b.refusals = 0
    return b


def test_accepted_lesson_reports_retained_change_and_counts_once():
    b = worm()
    r = b.learn("food")
    assert r.updated and r.reason == "qualified" and r.rows == 3 and r.event_id == 3
    assert b.lessons == 1 and b.rejected == 0 and b.stretch == []
    assert r.applied == {"weights": (1.0, -0.5), "biases": (0.25,)}
    (rows, source), = b.brain.calls
    assert source == "estimate"
    # the early tick restates the brain's own free prediction; the last two are corrected
    assert rows[0][1] == {"forward": (0.0,), "reverse": (0.05,)}
    for _, targets in rows[1:]:
        assert targets == {"forward": (0.8,), "reverse": (0.0,)}


def test_pain_and_joint_outcomes_build_their_targets():
    b = worm()
    b.learn("pain")
    (rows, _), = b.brain.calls
    assert rows[-1][1] == {"forward": (0.0,), "reverse": (0.8,)}
    b = worm()
    b.learn("food", "pain")
    (rows, _), = b.brain.calls
    assert rows[-1][1] == {"forward": (0.8,), "reverse": (0.8,)}   # no rival to rest


def test_refused_solve_never_claims_applied_learning():
    b = worm(accept=False)
    r = b.learn("food")
    assert not r.updated and r.reason == "line_search" and r.applied is None
    assert b.lessons == 0 and b.rejected == 1 and b.stretch == []
    assert b.brain.weights == (0.0, 0.0) and b.brain.biases == (0.0,)


def test_frozen_worm_spends_the_stretch_without_a_solve():
    b = worm(frozen=True)
    r = b.learn("pain")
    assert not r.updated and r.reason == "frozen" and r.applied is None
    assert b.brain.calls == [] and b.rejected == 1 and b.stretch == []


def test_outcome_without_experience_is_rejected():
    b = worm(ticks=0)
    r = b.learn("food")
    assert not r.updated and r.reason == "no_experience"
    assert b.brain.calls == [] and b.rejected == 1
