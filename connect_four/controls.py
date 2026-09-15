"""Supplied-rules controls, outside the candidate's imports.

``Policy`` is the one-shot policy interface: ``(cells, legal) -> column`` from the board as
the mover sees it, with no access to a step function or a terminal oracle beyond what the
policy itself brings. The candidate's ``snapshot_policy`` (its decision without search)
satisfies it, as do the opponents. ``RandomPolicy`` plays random legal columns.
``MinimaxPolicy`` is the supplied-rules planning control: alpha-beta negamax over the
exact rules with terminal values only (a win, a draw, a loss; every other leaf scores 0),
iterative deepening under a node budget, one node per board evaluated. Matched budgets
of 256 and 1,024 nodes mirror the candidate's planning budgets.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .env import Board, GameConfig

WIN = 1000


class Policy(Protocol):
    name: str

    def __call__(self, cells: np.ndarray, legal: np.ndarray) -> int: ...


class RandomPolicy:
    name = "random_policy"

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def __call__(self, cells: np.ndarray, legal: np.ndarray) -> int:
        cols = np.flatnonzero(legal)
        return int(cols[self.rng.integers(len(cols))])


class BudgetExhausted(Exception):
    """Raised inside the search when the node budget is spent."""


class MinimaxPolicy:
    """Alpha-beta negamax with iterative deepening under ``budget`` nodes per move. Root
    moves are searched with the full window so exact ties are known; a tie is broken at
    random (seeded). A forced result stops the deepening. Statistics: nodes over all calls,
    calls, and the completed depth of every call."""

    def __init__(self, budget: int, seed: int, config: GameConfig | None = None, max_depth: int | None = None) -> None:
        self.budget = int(budget)
        self.name = f"minimax_{self.budget}"
        self.rng = np.random.default_rng(seed)
        self.config = config or GameConfig()
        self.max_depth = int(max_depth) if max_depth is not None else self.config.cells
        self.nodes = 0
        self.calls = 0
        self.depths: list[int] = []
        self._nodes = 0

    def _order(self, board: Board) -> list[int]:
        center = (self.config.cols - 1) / 2
        return sorted(board.legal_columns(), key=lambda c: (abs(c - center), c))

    def _negamax(self, board: Board, depth: int, alpha: int, beta: int, ply: int) -> int:
        if self._nodes >= self.budget:
            raise BudgetExhausted  # the budget is spent: this board is neither evaluated nor counted
        self._nodes += 1
        if board.winner is not None:
            return -(WIN - ply)  # the previous mover connected: the side to move has lost
        if board.full or depth == 0:
            return 0
        best = -WIN
        for col in self._order(board):
            board.play(col)
            try:
                value = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            finally:
                board.undo()
            best = max(best, value)
            alpha = max(alpha, best)
            if alpha >= beta:
                break
        return best

    def __call__(self, cells: np.ndarray, legal: np.ndarray) -> int:
        board = Board.from_relative(cells, self.config)
        order = [c for c in self._order(board) if legal[c]]
        if not order:
            raise ValueError("no legal column")
        self._nodes = 0
        best, completed = list(order), 0
        remaining = self.config.cells - sum(1 for v in board.cells if v)
        for depth in range(1, min(self.max_depth, remaining) + 1):
            values: dict[int, int] = {}
            try:
                for col in order:
                    board.play(col)
                    try:
                        values[col] = -self._negamax(board, depth - 1, -WIN, WIN, 1)
                    finally:
                        board.undo()
            except BudgetExhausted:
                break
            top = max(values.values())
            best, completed = [c for c in order if values[c] == top], depth
            if abs(top) > WIN - self.config.cells - 1:
                break  # a forced result: deeper search cannot change the choice
        self.nodes += self._nodes
        self.calls += 1
        self.depths.append(completed)
        return int(best[self.rng.integers(len(best))])

    def stats(self) -> dict[str, float]:
        return {"nodes": self.nodes, "calls": self.calls, "mean_depth": float(np.mean(self.depths)) if self.depths else 0.0, "budget": self.budget}
