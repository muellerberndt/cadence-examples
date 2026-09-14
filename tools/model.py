"""Small public mechanisms behind the browser labs. No private research dependencies."""

from __future__ import annotations

import json
from pathlib import Path

import cadence as cd
import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
WORM = json.loads((ROOT / "worm/worm.json").read_text())
N = len(WORM["names"])
EDGES = np.asarray(WORM["edges"])
MATRIX = csr_matrix(
    (EDGES[:, 2], (EDGES[:, 1].astype(int), EDGES[:, 0].astype(int))), shape=(N, N)
)
NEURON_MODEL = cd.NeuronModel(
    gain=1, slope=2, threshold=0, leak=1, dt=1, stimulus_amplitude=1
)


def worm_brain():
    connectome = cd.Connectome.from_synapses(
        N, pre=EDGES[:, 0].astype(int), post=EDGES[:, 1].astype(int), sign=EDGES[:, 2]
    )
    return cd.Brain(connectome, NEURON_MODEL)


def reference(drive, mask, steps=200):
    state = np.zeros_like(drive)
    for _ in range(steps):
        state = mask * np.tanh(MATRIX.dot(state.T).T + drive)
    return state


def residual(state, drive, mask):
    return np.max(
        np.abs(state - mask * np.tanh(MATRIX.dot(state.T).T + drive)), axis=-1
    )


def samples(seed, count, lesions=2, mixed=True):
    rng = np.random.default_rng(seed)
    drive, mask = np.zeros((count, N)), np.ones((count, N))
    for i in range(count):
        strengths = rng.uniform(0, 2, 4)
        if not mixed:
            strengths[np.arange(4) != rng.integers(4)] = 0
        for strength, neurons in zip(strengths, WORM["stimuli"].values(), strict=True):
            drive[i, neurons] = strength
        mask[i, rng.choice(N, rng.integers(lesions + 1), replace=False)] = 0
    return drive, mask


def fast_memory(keys=8, values=4, rule="delta"):
    return cd.FastSynapses(np.arange(keys), np.arange(keys, keys + values), rule=rule)


class OnlineMLP:
    """Matched revealed samples; one-hidden-layer SGD baseline also executed in JS."""

    def __init__(self, seed=7, keys=8, hidden=32, values=4):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, 1 / np.sqrt(keys), (keys, hidden))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0, 1 / np.sqrt(hidden), (hidden, values))
        self.b2 = np.zeros(values)

    def predict(self, key):
        return np.tanh(key @ self.w1 + self.b1) @ self.w2 + self.b2

    def observe(self, key, value, steps=1, rate=0.15):
        for _ in range(steps):
            h = np.tanh(key @ self.w1 + self.b1)
            error = (h @ self.w2 + self.b2 - value) / len(value)
            delta = (self.w2 @ error) * (1 - h * h)
            self.w2 -= rate * np.outer(h, error)
            self.b2 -= rate * error
            self.w1 -= rate * np.outer(key, delta)
            self.b1 -= rate * delta

    def export(self):
        return {k: getattr(self, k).tolist() for k in ("w1", "b1", "w2", "b2")}


def key_bank(correlation=0.0):
    # Unit vectors with exact off-diagonal cosine correlation.
    return np.linalg.cholesky(
        (1 - correlation) * np.eye(8) + correlation * np.ones((8, 8))
    )
