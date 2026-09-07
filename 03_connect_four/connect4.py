"""Connect Four on bitboards: rules, a heuristic alpha-beta search, and the mover's-eye encoding.

Columns are 0..6, rows 0..5 from the bottom. Each side's discs are one integer with bit
``col * 7 + row`` set, the standard seven-bit-column layout that makes the four-in-a-row
tests four shifts each. ``Position`` is immutable; ``play`` returns a new one.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

COLS, ROWS = 7, 6
COLUMN_ORDER = (3, 2, 4, 1, 5, 0, 6)  # centre first: better pruning and a sane tie-break
BOTTOM = sum(1 << (c * 7) for c in range(COLS))
FULL = sum(((1 << ROWS) - 1) << (c * 7) for c in range(COLS))
TOP_ROW = sum(1 << (c * 7 + ROWS - 1) for c in range(COLS))


def _windows() -> list[int]:
    out = []
    for c in range(COLS):
        for r in range(ROWS):
            for dc, dr in ((1, 0), (0, 1), (1, 1), (1, -1)):
                cells = [(c + k * dc, r + k * dr) for k in range(4)]
                if all(0 <= cc < COLS and 0 <= rr < ROWS for cc, rr in cells):
                    out.append(sum(1 << (cc * 7 + rr) for cc, rr in cells))
    return out


WINDOWS = _windows()  # the 69 lines of four
CENTRE = sum(1 << (3 * 7 + r) for r in range(ROWS))


def has_four(discs: int) -> bool:
    for shift in (1, 7, 6, 8):  # vertical, horizontal, the two diagonals
        m = discs & (discs >> shift)
        if m & (m >> (2 * shift)):
            return True
    return False


@dataclass(frozen=True, slots=True)
class Position:
    mover: int = 0  # discs of the side to move
    other: int = 0  # discs of the side that just moved
    plies: int = 0

    @property
    def occupied(self) -> int:
        return self.mover | self.other

    def legal(self) -> list[int]:
        occ = self.occupied
        return [c for c in range(COLS) if not occ & (1 << (c * 7 + ROWS - 1))]

    def play(self, col: int) -> Position:
        occ = self.occupied
        cell = (occ + (1 << (col * 7))) & ~occ & (((1 << ROWS) - 1) << (col * 7))
        if not cell:
            raise ValueError(f"column {col} is full")
        return Position(mover=self.other, other=self.mover | cell, plies=self.plies + 1)

    def just_won(self) -> bool:
        """Did the side that just moved (``other``) complete four?"""
        return has_four(self.other)

    def full(self) -> bool:
        return self.occupied == FULL

    def terminal(self) -> bool:
        return self.just_won() or self.full()

    def winning_moves(self) -> list[int]:
        return [c for c in self.legal() if self.play(c).just_won()]

    def planes(self) -> np.ndarray:
        """Two 6x7 planes from the mover's side: my discs, their discs; flattened to 84."""
        mine = np.array([(self.mover >> (c * 7 + r)) & 1 for r in range(ROWS) for c in range(COLS)])
        theirs = np.array(
            [(self.other >> (c * 7 + r)) & 1 for r in range(ROWS) for c in range(COLS)]
        )
        return np.concatenate([mine, theirs]).astype(float)

    def mirrored(self) -> Position:
        def flip(discs: int) -> int:
            out = 0
            for c in range(COLS):
                out |= ((discs >> (c * 7)) & 0x7F) << ((COLS - 1 - c) * 7)
            return out

        return Position(flip(self.mover), flip(self.other), self.plies)

    def grid(self) -> list[list[int]]:
        """Rows top to bottom, 0 empty, 1 mover, 2 other; for printing and pages."""
        return [
            [
                1 if (self.mover >> (c * 7 + r)) & 1 else 2 if (self.other >> (c * 7 + r)) & 1 else 0
                for c in range(COLS)
            ]
            for r in reversed(range(ROWS))
        ]


def popcount(x: int) -> int:
    return bin(x).count("1")


def evaluate(pos: Position) -> float:
    """Heuristic value for the mover: threats in the 69 windows plus centre control."""
    score = 0.0
    mine, theirs = pos.mover, pos.other
    for w in WINDOWS:
        a, b = popcount(mine & w), popcount(theirs & w)
        if a and b:
            continue
        if a == 3:
            score += 5.0
        elif a == 2:
            score += 1.0
        if b == 3:
            score -= 5.0
        elif b == 2:
            score -= 1.0
    score += 0.5 * (popcount(mine & CENTRE) - popcount(theirs & CENTRE))
    return score


WIN = 1000.0


def negamax(pos: Position, depth: int, alpha: float, beta: float) -> float:
    if pos.just_won():
        return -(WIN + depth)  # the side to move has just lost; sooner is worse
    if pos.full():
        return 0.0
    if depth == 0:
        return evaluate(pos)
    best = -np.inf
    for col in COLUMN_ORDER:
        if pos.occupied & (1 << (col * 7 + ROWS - 1)):
            continue
        value = -negamax(pos.play(col), depth - 1, -beta, -alpha)
        if value > best:
            best = value
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best


def search(pos: Position, depth: int) -> tuple[int, dict[int, float]]:
    """Best column for the mover at ``depth`` plies, and the value of every legal column."""
    values: dict[int, float] = {}
    for col in COLUMN_ORDER:
        if pos.occupied & (1 << (col * 7 + ROWS - 1)):
            continue
        values[col] = -negamax(pos.play(col), depth - 1, -np.inf, np.inf)
    best = max(COLUMN_ORDER, key=lambda c: (values.get(c, -np.inf), -abs(c - 3)))
    return best, values


@lru_cache(maxsize=None)
def _mirror_col(col: int) -> int:
    return COLS - 1 - col


def mirror_col(col: int) -> int:
    return _mirror_col(col)
