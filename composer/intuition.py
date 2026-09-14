"""Aggregate musical expectations, deliberately without a store of complete pieces.

These are small, smoothed association tables learned from training families.
They supply phrase planning and readback drives to the musical patch graph;
they are not a claim of learned beauty or a replacement for the trained net.
"""

from itertools import pairwise
from pathlib import Path

import numpy as np

from .encoding import DURATIONS

ROOT = Path(__file__).resolve().parents[1]


class Intuition:
    def __init__(self, path=None):
        self.path = Path(path or ROOT / "checkpoints/intuition/music.npz")
        self.tables = dict(np.load(self.path)) if self.path.exists() else None

    @staticmethod
    def probability(counts, smoothing=0.5):
        c = np.asarray(counts, float) + smoothing
        return c / c.sum(axis=-1, keepdims=True)

    def chord_probability(self, mode, previous):
        if self.tables is None:
            return np.ones(24) / 24
        return self.probability(self.tables["chords"][mode, previous, :24])

    def rhythm_probability(self, arousal, beat, previous):
        if self.tables is None:
            return np.ones(len(DURATIONS)) / len(DURATIONS)
        return self.probability(self.tables["rhythm"][arousal, beat % 4, previous])

    def interval_probability(self, previous):
        if self.tables is None:
            return np.ones(25) / 25
        return self.probability(
            self.tables["intervals"][np.clip(previous, -12, 12) + 12]
        )

    def evaluate(self, events):
        """Surprise in learned relationships, not a score for copying exact notes."""
        if len(events) < 3 or self.tables is None:
            return {"interval_surprise": 0.0, "extreme_surprise_rate": 0.0}
        intervals = np.diff([e["pitch"] for e in events])
        surprise = np.array(
            [
                -np.log(self.interval_probability(a)[np.clip(b, -12, 12) + 12])
                for a, b in pairwise(intervals)
            ]
        )
        return {
            "interval_surprise": float(surprise.mean()),
            "extreme_surprise_rate": float((surprise > 5).mean()),
        }

    def teach(self, events, brief, rating, *, strength=0.06):
        """A local signed association update. Absolute melodies never enter this memory."""
        if self.tables is None:
            raise ValueError("Train the musical relationship memory first.")
        if rating not in (-1, 1):
            raise ValueError("A preference must be +1 or -1.")
        observed = {k: np.zeros_like(v) for k, v in self.tables.items()}
        chords = []
        last_bar = None
        for e in events:
            bar = e["step"] // 16
            if bar != last_bar:
                chords.append(min(23, e["chord"]))
                last_bar = bar
        for a, b in pairwise(chords):
            observed["chords"][brief.mode, a, b] += 1
        for a, b in pairwise(events):
            observed["rhythm"][
                brief.arousal, b["step"] % 4, a["token"][1], b["token"][1]
            ] += 1
        intervals = np.clip(np.diff([e["pitch"] for e in events]), -12, 12) + 12
        for a, b in pairwise(intervals):
            observed["intervals"][a, b] += 1
        for name, visits in observed.items():
            # Keep each conditioning row as a bounded prior plus one preference experience.
            old = self.probability(self.tables[name])
            row = visits.sum(axis=-1, keepdims=True)
            update = visits / np.maximum(1, row)
            touched = row > 0
            new = np.maximum(1e-5, old + rating * strength * (update - old))
            new /= new.sum(axis=-1, keepdims=True)
            # Store counts whose smoothed readout is exactly the new probability.
            self.tables[name] = np.where(touched, new * 1e6 - 0.5, self.tables[name])

    def retention(self, validation):
        losses = []
        details = {}
        for name, observations in validation.items():
            prob = self.probability(self.tables[name])
            nll = float(
                -(observations * np.log(prob)).sum() / max(1, observations.sum())
            )
            details[name] = nll
            losses.append(nll)
        return {"mean_nll": float(np.mean(losses)), "relationships": details}

    def save(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.path, **self.tables)
