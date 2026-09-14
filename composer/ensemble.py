"""Polyphonic event brain: instruments, simultaneous attacks and held notes survive encoding."""

import json
from dataclasses import dataclass
from pathlib import Path

import cadence as cd
import numpy as np
from cadence.genome import Genome, Projection, Region, develop

HISTORY = 8
DELTAS = np.array([0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64])
SIZES = (
    73,
    12,
    13,
    8,
    8,
)  # pitch24..96, sounding duration, time to attack, instrument family, velocity
OFFSETS = np.cumsum((0,) + SIZES[:-1])
EVENTS = sum(SIZES)
EXTRA_RAW = 5 + 12 + 96
EXTRA = 2 + 3 + 3 + 16 + 16 + 12 + 96
INPUTS = HISTORY * EVENTS + EXTRA
FAMILIES = (
    "keys",
    "strings",
    "brass",
    "woodwind",
    "bass",
    "plucked",
    "other",
    "percussion",
)
PROGRAMS = (0, 48, 60, 73, 32, 24, 80, 0)


def family(program, channel):
    if channel == 9:
        return 7
    if 24 <= program < 32:
        return 5
    if 32 <= program < 40:
        return 4
    if 40 <= program < 56:
        return 1
    if 56 <= program < 64:
        return 2
    if 64 <= program < 80:
        return 3
    return 0 if program < 24 else 6


def encode(context, extra):
    x = np.zeros((len(context), INPUTS), np.float32)
    rows = np.arange(len(x))
    for h in range(HISTORY):
        for k, offset in enumerate(OFFSETS):
            x[rows, h * EVENTS + offset + context[:, h, k]] = 1
    start = HISTORY * EVENTS
    for col, width in enumerate([2, 3, 3, 16, 16]):
        x[rows, start + extra[:, col].astype(int)] = 1
        start += width
    x[:, start:] = extra[:, 5:] / 255
    return x


@dataclass
class Design:
    size: int = 256
    seed: int = 31


def build(design=None, *, backend="cpu", device=None):
    d = design or Design()
    parts = (
        Region("heard_events", INPUTS),
        Region("harmonic_assembly", d.size),
        Region("instrument_assembly", d.size // 2),
        Region("rhythm_assembly", d.size // 2),
        Region("ensemble_intention", EVENTS),
    )
    projections = []
    for name in ["harmonic_assembly", "instrument_assembly", "rhythm_assembly"]:
        projections += [
            Projection("heard_events", name, scale=0.6, reciprocal=False),
            Projection(name, "ensemble_intention", scale=0.6),
        ]
    projections += [
        Projection("heard_events", "ensemble_intention", scale=0.5, reciprocal=False),
        Projection("harmonic_assembly", "instrument_assembly", density=0.2, scale=0.2),
        Projection("rhythm_assembly", "instrument_assembly", density=0.2, scale=0.2),
    ]
    connectome = develop(
        Genome(parts, tuple(projections), label="polyphonic-ensemble"), seed=d.seed
    )
    brain = cd.Brain(
        connectome,
        cd.learning_neuron_model(dt=1, leak=0.1),
        backend=backend,
        device=device,
        dense_limit=8192,
        precision="float32" if backend == "torch" else None,
    )
    plastic = np.ones(connectome.n, bool)
    plastic[list(connectome.populations["heard_events"])] = False
    return cd.Learner(
        brain,
        connectome.populations["ensemble_intention"],
        cd.LearnerConfig(
            beta=0.3,
            eta=0.16,
            eta_bias=0.016,
            free_steps=32,
            nudged_steps=16,
            centered=True,
            tolerance=0,
            temperature=0.2,
            normalize=0.98,
            momentum=0.8,
            normalize_floor=0.01,
            decay=1e-6,
        ),
        plastic_neurons=plastic,
        slots=SIZES,
    )


def drives(brain, context, extra):
    x = encode(context, extra)
    out = np.zeros((len(x), brain.brain.connectome.n), np.float32)
    out[:, :INPUTS] = x * 2.5
    return out


class Dataset:
    """Memory-map shards so four GPU workers share the operating system's file cache."""

    def __init__(self, path, split):
        path = Path(path)
        self.manifest = json.loads((path / "manifest.json").read_text())
        self.shards = [
            {
                k: np.load(path / split / s[k], mmap_mode="r")
                for k in ["context", "extra", "labels"]
            }
            for s in self.manifest["shards"][split]
        ]
        self.ends = np.cumsum([len(s["labels"]) for s in self.shards])
        self.count = int(self.ends[-1])

    def rows(self, ids):
        ids = np.asarray(ids)
        selected = np.searchsorted(self.ends, ids, side="right")
        result = {
            "context": np.empty((len(ids), HISTORY, 5), np.uint8),
            "extra": np.empty((len(ids), EXTRA_RAW), np.uint8),
            "labels": np.empty((len(ids), 5), np.uint8),
        }
        for shard in np.unique(selected):
            mask = selected == shard
            ix = ids[mask] - (self.ends[shard - 1] if shard else 0)
            for k, value in result.items():
                value[mask] = self.shards[shard][k][ix]
        return result

    def sample(self, rng, count):
        return self.rows(rng.integers(self.count, size=count))
