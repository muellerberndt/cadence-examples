"""Who the brain plays: random, tactical one-ply, a rules-only search, the perfect solver at a
given strength, and an AlphaZero checkpoint.

Every opponent is a callable ``(cells, legal) -> column``: ``cells`` are the 42 cells
row-major from the bottom-left as the opponent sees them (0 empty, 1 its own stone, 2 the
other side's) and ``legal`` the mask of playable columns. ``play`` runs one game between two
of them and returns its columns.
"""

from __future__ import annotations

import numpy as np

from .game import CELLS, H1, WIDTH, Position


def position_of(cells: np.ndarray) -> Position:
    """The position whose side to move sees ``cells``."""
    cells = np.asarray(cells).reshape(-1)
    mover = mask = 0
    for k in range(CELLS):
        if cells[k]:
            bit = 1 << ((k % WIDTH) * H1 + k // WIDTH)
            mask |= bit
            if cells[k] == 1:
                mover |= bit
    return Position(mover, mask, int(np.count_nonzero(cells)))


class RandomOpponent:
    name = "random"

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def __call__(self, cells: np.ndarray, legal: np.ndarray) -> int:
        return int(self.rng.choice(np.flatnonzero(legal)))


class OnePlyOpponent:
    """Wins at once when it can, else blocks a line the other side would complete at once,
    else plays a random legal column."""

    name = "one_ply"

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def __call__(self, cells: np.ndarray, legal: np.ndarray) -> int:
        position = position_of(cells)
        columns = position.legal()
        wins = [c for c in columns if position.wins(c)]
        other = Position(position.mover ^ position.mask, position.mask, position.plies)
        blocks = [c for c in columns if other.wins(c)]
        return int(self.rng.choice(wins or blocks or columns))


class MinimaxOpponent:
    """Alpha-beta over the rules alone, deepened until ``nodes`` positions are visited; a
    position the search cannot finish is worth nothing. The control for what search without a
    value patch achieves."""

    def __init__(self, nodes: int, seed: int) -> None:
        self.nodes, self.rng, self.name = int(nodes), np.random.default_rng(seed), f"minimax_{nodes}"

    def _search(self, position: Position, depth: int, alpha: float, beta: float) -> float:
        self.visited += 1
        if self.visited > self.nodes:
            raise TimeoutError
        for c in position.legal():
            if position.wins(c):
                return 1.0 + depth / 100
        if depth == 0 or position.plies >= 41:
            return 0.0
        best = -2.0
        for c in position.legal():
            best = max(best, -self._search(position.play(c), depth - 1, -beta, -alpha))
            alpha = max(alpha, best)
            if alpha >= beta:
                break
        return best

    def __call__(self, cells: np.ndarray, legal: np.ndarray) -> int:
        position = position_of(cells)
        columns = position.legal()
        self.visited, chosen = 0, [columns[0]]
        for depth in range(1, 43 - position.plies):
            try:
                values = {c: (2.0 if position.wins(c) else -self._search(position.play(c), depth - 1, -2.0, 2.0)) for c in columns}
            except TimeoutError:
                break
            top = max(values.values())
            chosen = [c for c in columns if values[c] == top]
        return int(self.rng.choice(chosen))


def make_opponent(name: str, seed: int, shared: dict | None = None):
    """``random``, ``one_ply``, ``minimax_<nodes>``, ``solver_<percent>`` (Pascal Pons' solver
    plays a best column with that probability and a random one otherwise) or
    ``alphazero=<checkpoint>@<simulations>``. ``shared`` keeps one solver process and one
    AlphaZero net per checkpoint across the opponents of a bench."""
    shared = shared if shared is not None else {}
    if name == "random":
        return RandomOpponent(seed)
    if name == "one_ply":
        return OnePlyOpponent(seed)
    if name.startswith("minimax_"):
        return MinimaxOpponent(int(name.split("_")[1]), seed)
    if name.startswith("solver_"):
        from .bench.pons import Solver, SolverOpponent

        if shared.get("solver") is None:
            shared["solver"] = Solver()
        return SolverOpponent(int(name.split("_")[1]) / 100.0, seed, shared["solver"])
    if name.startswith("alphazero="):
        from .bench.alphazero.opponent import AlphaZeroOpponent

        path, _, sims = name[len("alphazero="):].partition("@")
        key = ("alphazero", path, int(sims or 50))
        if key not in shared:
            shared[key] = AlphaZeroOpponent(path, sims=int(sims or 50), seed=seed)
        else:
            shared[key].reseed(seed)
        return shared[key]
    raise ValueError(f"unknown opponent {name!r}")


def play(first, second, *, opening: list[int] | None = None) -> tuple[list[int], int]:
    """One game. Returns its columns and the winner: 0 the first player, 1 the second, -1 a draw."""
    position, columns = Position(), []
    for column in opening or []:
        position = position.play(column)
        columns.append(int(column))
    players = (first, second)
    while True:
        legal = np.array([position.playable(c) for c in range(WIDTH)])
        column = int(players[position.plies % 2](position.cells(), legal))
        if not legal[column]:
            raise ValueError(f"{getattr(players[position.plies % 2], 'name', 'a player')} chose the full column {column}")
        position = position.play(column)
        columns.append(column)
        if position.lost:
            return columns, (position.plies - 1) % 2
        if position.full:
            return columns, -1
