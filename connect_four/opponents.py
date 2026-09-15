"""The candidate's opponents: random, tactical one-ply, saved snapshots, and their mixture.

Every opponent is a callable ``(cells, legal) -> column`` where ``cells`` is the board as
the opponent sees it (0 empty, 1 its own stone, 2 the candidate's) and ``legal`` the mask
of playable columns; it has a ``name`` for the logs and its own generator. Opponents read
the rules through the engine; the candidate never imports this module.
"""

from __future__ import annotations

import numpy as np

from .env import Board, GameConfig, Opponent


class RandomOpponent:
    """A uniformly random legal column."""

    name = "random"

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def __call__(self, cells: np.ndarray, legal: np.ndarray) -> int:
        cols = np.flatnonzero(legal)
        return int(cols[self.rng.integers(len(cols))])


class OnePlyOpponent:
    """Wins at once when it can, else blocks a column where the other side would win at
    once, else plays a random legal column. Ties among wins or blocks are drawn at random."""

    name = "one_ply"

    def __init__(self, seed: int, config: GameConfig | None = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.config = config or GameConfig()

    def _pick(self, cols: list[int]) -> int:
        return int(cols[self.rng.integers(len(cols))])

    def __call__(self, cells: np.ndarray, legal: np.ndarray) -> int:
        board = Board.from_relative(cells, self.config)
        cols = [int(c) for c in np.flatnonzero(legal)]
        wins = [c for c in cols if board.would_connect(c, 0)]
        if wins:
            return self._pick(wins)
        blocks = [c for c in cols if board.would_connect(c, 1)]
        if blocks:
            return self._pick(blocks)
        return self._pick(cols)


class SnapshotOpponent:
    """Plays a saved policy callable (a snapshot of the candidate, or any control policy)."""

    def __init__(self, policy: Opponent, name: str) -> None:
        self.policy = policy
        self.name = name

    def __call__(self, cells: np.ndarray, legal: np.ndarray) -> int:
        column = int(self.policy(np.asarray(cells), np.asarray(legal, bool)))
        if not legal[column]:
            raise ValueError(f"snapshot {self.name} chose the illegal column {column}")
        return column


class OpponentMixture:
    """One component per game, drawn by weight. ``policies`` maps a component name to its
    opponent; the reserved component ``snapshots`` draws uniformly from the snapshot pool
    (the last ``pool`` snapshots added). While the pool is empty its weight is redistributed
    over the other components. Draws are counted per component for the report."""

    def __init__(self, weights: dict[str, float], policies: dict[str, Opponent], seed: int, *, pool: int = 8) -> None:
        if any(w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
            raise ValueError("weights must be nonnegative with a positive sum")
        for name in weights:
            if name != "snapshots" and name not in policies:
                raise ValueError(f"no policy for component {name!r}")
        self.weights = dict(weights)
        self.policies = dict(policies)
        self.rng = np.random.default_rng(seed)
        self.pool = int(pool)
        self.snapshots: list[SnapshotOpponent] = []
        self.draws: dict[str, int] = dict.fromkeys(weights, 0)
        self.added = 0

    def add_snapshot(self, policy: Opponent, name: str | None = None) -> SnapshotOpponent:
        snapshot = SnapshotOpponent(policy, name or f"snapshot_{self.added}")
        self.added += 1
        self.snapshots.append(snapshot)
        del self.snapshots[: max(0, len(self.snapshots) - self.pool)]
        return snapshot

    def new_game(self) -> tuple[str, Opponent]:
        """(component name, opponent) for the next game."""
        names = [n for n in self.weights if self.weights[n] > 0 and (n != "snapshots" or self.snapshots)]
        weights = np.array([self.weights[n] for n in names], float)
        name = names[int(self.rng.choice(len(names), p=weights / weights.sum()))]
        self.draws[name] += 1
        if name == "snapshots":
            return name, self.snapshots[int(self.rng.integers(len(self.snapshots)))]
        return name, self.policies[name]


DEFAULT_WEIGHTS = {"random": 0.4, "one_ply": 0.3, "snapshots": 0.3}
