"""The brain's value patch: one ``cadence.RecordPatchNet`` that reads a position and says how
the game ends for the side that just placed a stone.

A board is its own history, so every reading is a path of one moment that starts from rest:
``imagine`` is called with a zero boundary and ``observe`` after ``reset``. The patch is used
through the library's documented calls only. Its slow parameters learn the value of positions
in general; its records hold what the slow readout still gets wrong at the readings it has
witnessed, written in one shot, which is how a game just played changes the next one.

The value of a position is the outcome for the side that placed the last stone, discounted by
the plies the game still lasts: ``z * GAMMA ** n`` with ``z`` 1 for a win, 0 for a draw and -1
for a loss. A win that is near reads higher than a win that is far.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from cadence import RecordPatchNet

from .game import READING, readings

GAMMA = 0.99


def value_target(z: np.ndarray, n: np.ndarray) -> np.ndarray:
    return np.asarray(z, dtype=float) * GAMMA ** np.asarray(n, dtype=float)


class ValuePatch:
    def __init__(self, hidden: int = 256, cells: int = 4096, active: int = 32, seed: int = 0,
                 record_rate: float = 0.5, record_averaging: bool = False, net: RecordPatchNet | None = None) -> None:
        self.net = net or RecordPatchNet(READING, hidden, 1, seed=seed, cells=cells, active=active,
                                         record_rate=record_rate, record_averaging=record_averaging, slowest=2.0)
        self.evaluations = 0

    @property
    def hidden(self) -> int:
        return self.net.hidden

    def values(self, keys: Iterable[tuple[int, int]], *, parts: bool = False):
        """The values of positions given by their keys, read privately: nothing changes.
        ``parts`` also returns what the slow readout says alone, without the record read."""
        x = readings(keys)
        self.evaluations += len(x)
        path = self.net.imagine(x[:, None, :], state=np.zeros((len(x), self.net.hidden)))
        if parts:
            return path.output[:, 0, 0], path.slow_output[:, 0, 0]
        return path.output[:, 0, 0]

    def learn(self, keys: Iterable[tuple[int, int]], targets: np.ndarray, *, rate: float, write: bool = True):
        """Witness positions with how their games ended: one slow step and one write each."""
        x = readings(keys)
        self.net.reset()
        return self.net.observe(x[:, None, :], np.asarray(targets, dtype=float)[:, None, None], rate=rate, write=write)

    def save(self, path: str | Path) -> Path:
        return self.net.save(path)

    @classmethod
    def load(cls, path: str | Path) -> ValuePatch:
        return cls(net=RecordPatchNet.load(path))
