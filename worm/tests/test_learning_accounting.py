"""Bounded adapter regressions; synthetic proposals exercise post-learning custody."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worm.brain import WormBrain


class Net:
    def __init__(self, accept=True, value=2.0):
        self.values = {k: np.zeros((1, 1)) for k in ('A', 'B', 'C')}
        self.updates = 7
        self.accept, self.value = accept, value

    def parameters(self):
        return {k: v.copy() for k, v in self.values.items()}

    def set_parameters(self, values):
        self.values = {k: v.copy() for k, v in values.items()}

    def reset(self):
        pass

    def observe(self, u, y, **kw):
        updated = self.accept and kw['rate'] > 0
        if updated:
            self.values = {k: v + self.value for k, v in self.values.items()}
            self.updates += 1
        return SimpleNamespace(updated=updated, reason='updated' if updated else 'no_step',
            delta={k: np.ones((1, 1)) for k in self.values},
            free=SimpleNamespace(hidden=np.zeros((1, 2, 1))), plus=None, minus=None)


def worm(*, accept=True, frozen=False, value=2.0):
    b = WormBrain.__new__(WormBrain)
    b.net = Net(accept, value)
    b.p = dict(stability=.97, beta=.01, rate=1., window=2, prenatal_lessons=2, teach_level=.8)
    b.frozen = frozen
    b.stretch = [np.ones(1)]
    b.lessons = b.rejected = b.halvings = 0
    b.inputs, b.outputs = ['pain'], ['forward', 'reverse']
    b.growth = lambda A=None: float(np.max(np.abs(b.net.values['A'] if A is None else A)))
    return b


def observe(b):
    return b._observe('food', np.ones((1, 2, 1)), np.zeros((1, 2, 2)))


def test_scaled_commit_reports_actual_change_and_counts_once():
    b = worm()
    r = observe(b)
    assert r.updated and r.reason == 'updated_after_growth_filter'
    assert r.halvings == 2 and b.lessons == 1 and b.rejected == 0 and b.net.updates == 8
    for key in ('A', 'B', 'C'):
        np.testing.assert_array_equal(r.applied[key], [[.5]])
        np.testing.assert_array_equal(r.delta[key], [[1.]])  # proposal gradient is not applied change
    assert b.stretch == []


def test_full_filter_rollback_is_rejected_and_restores_update_counter():
    b = worm()
    b.growth = lambda A=None: 0 if not np.any(A) else 1
    r = observe(b)
    assert not r.updated and r.reason == 'growth_filter_rejected'
    assert r.halvings == 12 and b.lessons == 0 and b.rejected == 1 and b.net.updates == 7
    for key in ('A', 'B', 'C'):
        np.testing.assert_array_equal(b.net.values[key], [[0.]])
        np.testing.assert_array_equal(r.applied[key], [[0.]])


def test_nonfinite_growth_fails_closed():
    b = worm()
    b.growth = lambda A=None: float('nan')
    r = observe(b)
    assert not r.updated and r.halvings == 12 and b.net.updates == 7
    assert all(not np.any(v) for v in b.net.parameters().values())


def test_frozen_and_solver_rejection_never_claim_applied_learning():
    for b in (worm(frozen=True), worm(accept=False)):
        r = observe(b)
        assert not r.updated and r.applied is None
        assert b.lessons == 0 and b.rejected == 1 and b.net.updates == 7


def test_birth_admission_counts_final_weights_after_filter():
    b = worm()
    b.growth = lambda A=None: 0 if not np.any(A) else 1
    assert b.born() == [False, False]
    assert b.net.updates == 7 and all(not np.any(v) for v in b.net.parameters().values())


def test_single_start_growth_is_not_a_stability_certificate():
    b = WormBrain.__new__(WormBrain)
    b.H = 2
    A = np.array([[4., -2.], [-2., 1.]])  # the fixed start [1,2] misses its eigenvalue 5
    assert WormBrain.growth(b, A) == 0
    assert max(abs(np.linalg.eigvals(A))) == 5
