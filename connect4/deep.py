"""The deep value patch: one cadence 0.50 brain whose observers read observers, settled
once per position, answering how the game ends for the side that just placed a stone.

The layout is a ladder. A column of ``layers[0]`` patches reads the 84 units of a board.
Each later level is an observer of the level below: it reads that level's live states and
its exact prediction errors, and the board as well, and its feedback joins the same joint
settlement. A small head observes every level at once and the single output patch of the
head is the value. Depth here is observation depth, the library's own recursion, not a
stack of completed layer answers.

Learning is witness admission: ``observe_batch`` fixes the outcome at the output and
jointly repairs every level's private activity and the shared parameters under one anchor
(cadence 0.50, ``docs/SPECIFICATION.md``). The patch never calls ``step`` or single
``observe``, so its live state stays at rest and every read starts from the zero state.
Reads go through ``engine.FreeSettleTwin``, a batched NumPy twin of the library's free
query, rebuilt after every admission and bound to ``Brain.settle`` by parity receipts.

The value of a position is the outcome for the side that placed the last stone,
discounted by the plies the game still lasts: ``z * GAMMA ** n`` with ``z`` 1 a win, 0 a
draw, -1 a loss. Teaching clamps sit at ``SCALE`` times that, comfortably inside the
``tanh`` prediction range and the state bound as the library's bootstrap guide asks;
``values`` divides the settled output by ``SCALE`` on the way back out.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from cadence import Brain, Cortex

from .game import READING, readings

FORMAT = "cadence-examples.connect4.deep/2"
GAMMA = 0.99
SCALE = 0.6
INPUT_SCALE = 0.6


def value_target(z: np.ndarray, n: np.ndarray) -> np.ndarray:
    """The undiscounted-by-SCALE value in [-1, 1]; ``learn`` scales it for the clamp."""
    return np.asarray(z, dtype=float) * GAMMA ** np.asarray(n, dtype=float)


def build(layers: Iterable[int], head: int, *, seed: int = 0, parameter_prior: float = 0.1,
          initial_scale: float = 0.3, settle_budget: int = 2048, tolerance: float = 1e-6,
          device: str = "python", board_to_all: bool = True) -> Brain:
    """A ladder of observers over one board-reading column, and a head over every level."""
    layers = [int(w) for w in layers]
    layout = Cortex(seed=seed, parameter_prior=parameter_prior, initial_scale=initial_scale,
                    settle_budget=settle_budget, tolerance=tolerance, device=device)
    board = layout.input("board", shape=(READING,))
    levels = [layout.column("level_0", patches=layers[0], inputs=board)]
    for k, width in enumerate(layers[1:], start=1):
        extra = (board,) if board_to_all else ()
        levels.append(layout.observer(f"level_{k}", patches=width, inputs=extra, observes=levels[-1]))
    head_pop = layout.observer("head", patches=int(head), inputs=(board,) if board_to_all else (),
                               observes=tuple(levels))
    layout.output("value", shape=(1,), reads=head_pop, indices=(0,))
    return layout.build()


class DeepValuePatch:
    """The observer-ladder brain behind the same two calls the search and school use."""

    def __init__(self, layers: Iterable[int] = (96, 48, 24), head: int = 4, *, seed: int = 0,
                 parameter_prior: float = 0.1, initial_scale: float = 0.3,
                 settle_budget: int = 2048, tolerance: float = 1e-6, device: str = "python",
                 board_to_all: bool = True, brain: Brain | None = None,
                 metadata: dict | None = None) -> None:
        if brain is not None:
            self.brain = brain
            self.metadata = dict(metadata or {})
        else:
            self.brain = build(layers, head, seed=seed, parameter_prior=parameter_prior,
                               initial_scale=initial_scale, settle_budget=settle_budget,
                               tolerance=tolerance, device=device, board_to_all=board_to_all)
            self.metadata = {"layers": [int(w) for w in layers], "head": int(head),
                             "board_to_all": bool(board_to_all), "gamma": GAMMA,
                             "scale": SCALE, "input_scale": INPUT_SCALE}
        self.evaluations = 0
        self.unqualified_reads = 0
        self._twin = None

    # ---------------------------------------------------------------- wiring

    @property
    def layers(self) -> list[int]:
        return list(self.metadata["layers"]) + [self.metadata["head"]]

    @property
    def output_index(self) -> int:
        """The global patch index the ``value`` output reads: patch 0 of the head."""
        for population in self.brain.inspect()["populations"]:
            if population["name"] == "head":
                return int(population["indices"][0])
        raise RuntimeError("no head population")

    def twin(self):
        from .engine import FreeSettleTwin

        if self._twin is None:
            self._twin = FreeSettleTwin(self.brain, self.output_index)
        return self._twin

    @staticmethod
    def boards(keys: Iterable[tuple[int, int]]) -> np.ndarray:
        return INPUT_SCALE * readings(keys)

    # ----------------------------------------------------------------- reads

    def values(self, keys: Iterable[tuple[int, int]]) -> np.ndarray:
        """The values of positions given by their keys, each settled from rest. A row the
        settle cannot qualify within the configured budget is retried once with four
        times the sweeps; a row that still does not qualify is an error, as for the
        record patch's failed read."""
        u = self.boards(keys)
        self.evaluations += len(u)
        twin = self.twin()
        result = twin.settle(u)
        bad = np.flatnonzero(~result["qualified"])
        if len(bad):
            retry = twin.settle(u[bad], budget=4 * twin.budget)
            result["outputs"][bad] = retry["outputs"]
            still = bad[~retry["qualified"]]
            if len(still):
                self.unqualified_reads += len(still)
                raise RuntimeError(f"{len(still)} of {len(u)} readings failed qualification")
        return result["outputs"] / SCALE

    # -------------------------------------------------------------- learning

    def learn(self, keys: Iterable[tuple[int, int]], targets: np.ndarray, *,
              budget: int | None = None) -> dict:
        """Witness positions with how their games ended: one joint batch admission.
        ``targets`` are in [-1, 1]; the clamp is ``SCALE`` times them. Returns the
        library's result; nothing is retained when ``accepted`` is false."""
        u = self.boards(keys)
        clamps = SCALE * np.asarray(targets, dtype=float)
        examples = [({"board": row.tolist()}, {"value": [float(t)]})
                    for row, t in zip(u, clamps, strict=True)]
        result = self.brain.observe_batch(examples, budget=budget)
        if result["accepted"]:
            self._twin = None
        return result

    # ------------------------------------------------------------ continuation

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps({"format": FORMAT, **self.metadata,
                                    "snapshot": self.brain.snapshot()}))
        return path

    @classmethod
    def load(cls, path: str | Path, *, device: str | None = None) -> DeepValuePatch:
        """``device`` moves the continuation to another execution device, for reading a
        checkpoint from a machine whose training device this one does not have."""
        data = json.loads(Path(path).read_text())
        if data.get("format") != FORMAT:
            raise ValueError(f"{path}: not a {FORMAT} checkpoint")
        snapshot = data.pop("snapshot")
        data.pop("format")
        return cls(brain=Brain.from_snapshot(snapshot, device=device), metadata=data)
