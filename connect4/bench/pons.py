"""Pascal Pons' perfect solver as a scorer and as an opponent.

``Solver`` runs ``external/pons/c4bridge`` (built by ``bench/setup_external.sh``) as a
persistent process and answers the exact score of every column of a board: positive when the
side to move wins with perfect play (larger is sooner), negative when it loses, 0 a draw,
``INVALID`` for an unplayable column. ``SolverOpponent`` plays a score-maximal column with
probability ``strength`` and a random legal column otherwise, so a life can be trained against
graded perfect play. ``move_quality`` classifies a chosen column against the scores: optimal
(a score-maximal column), a slip (a worse column that keeps the theoretical result) or a
blunder (the result changes: a won position becomes drawn or lost, a drawn one lost).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

INVALID = -1000
HERE = Path(__file__).resolve().parent
BRIDGE = HERE.parent / "external" / "pons" / "c4bridge"
BOOK = HERE.parent / "external" / "pons" / "7x6.book"


def board_line(cells: np.ndarray) -> str:
    """The bridge's line for a board as the side to move sees it (0 empty, 1 own, 2 other)."""
    cells = np.asarray(cells).reshape(-1)
    if cells.size != 42:
        raise ValueError("the solver reads the 6 by 7 board only")
    return "".join(".xo"[int(v)] for v in cells)


class Solver:
    def __init__(self, bridge: Path = BRIDGE, book: Path | None = BOOK) -> None:
        if not Path(bridge).exists():
            raise FileNotFoundError(f"{bridge}: run connect4/bench/setup_external.sh")
        args = [str(bridge)] + ([str(book)] if book and Path(book).exists() else [])
        self.process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.calls = 0
        self.nodes = 0
        self.cache: dict[str, np.ndarray] = {}

    def scores(self, cells: np.ndarray) -> np.ndarray:
        """The seven column scores of the board for the side to move."""
        line = board_line(cells)
        hit = self.cache.get(line)
        if hit is not None:
            return hit
        assert self.process.stdin and self.process.stdout
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()
        answer = self.process.stdout.readline().split()
        if len(answer) != 8:
            raise RuntimeError(f"solver answered {answer!r}")
        out = np.array([int(v) for v in answer[:7]])
        self.calls += 1
        self.nodes = int(answer[7])
        if len(self.cache) < 200_000:
            self.cache[line] = out
        return out

    def close(self) -> None:
        if self.process.stdin:
            self.process.stdin.close()
        self.process.wait(timeout=5)


def move_quality(scores: np.ndarray, column: int) -> str:
    """'optimal', 'slip' or 'blunder' for ``column`` under the column ``scores``."""
    playable = scores[scores != INVALID]
    best = int(playable.max())
    chosen = int(scores[column])
    if chosen == INVALID:
        raise ValueError("an unplayable column")
    if chosen == best:
        return "optimal"
    return "slip" if np.sign(chosen) == np.sign(best) else "blunder"


class SolverOpponent:
    """Perfect play with probability ``strength``, else a random legal column. Ties among the
    score-maximal columns break at random. ``name`` carries the strength for the logs."""

    def __init__(self, strength: float, seed: int, solver: Solver | None = None) -> None:
        self.strength = float(strength)
        self.rng = np.random.default_rng(seed)
        self.solver = solver or Solver()
        self.name = f"solver_{round(100 * self.strength):03d}"

    def __call__(self, cells: np.ndarray, legal: np.ndarray) -> int:
        cols = np.flatnonzero(legal)
        if self.rng.random() >= self.strength:
            return int(cols[self.rng.integers(len(cols))])
        scores = self.solver.scores(cells)
        playable = [int(c) for c in cols if scores[c] != INVALID]
        best = max(scores[c] for c in playable)
        top = [c for c in playable if scores[c] == best]
        return int(top[self.rng.integers(len(top))])
