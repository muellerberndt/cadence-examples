"""The rules of Connect Four on two 49-bit boards, and what the brain reads of a position.

A position is two integers: ``mover``, the stones of the side to move, and ``mask``, every
stone. Bit ``column * 7 + row`` is the cell in ``column`` (0 the leftmost) and ``row`` (0 the
floor); the seventh bit of a column stays empty, so a line never wraps around an edge.

The brain reads a position right after a stone has landed, as the side that placed it sees
it: 84 units, the 42 cells of that side's stones and the 42 cells of the other side's, each
row-major from the bottom-left cell. A position and its mirror image are one reading: of the
two, the one whose pair of boards is smaller is read.
"""

from __future__ import annotations

import numpy as np

WIDTH, HEIGHT = 7, 6
H1 = HEIGHT + 1
CELLS = WIDTH * HEIGHT
READING = 2 * CELLS
BOTTOM = tuple(1 << (c * H1) for c in range(WIDTH))
TOP = tuple(1 << (c * H1 + HEIGHT - 1) for c in range(WIDTH))
COLUMN = tuple(((1 << HEIGHT) - 1) << (c * H1) for c in range(WIDTH))
FULL = sum(COLUMN)
ORDER = (3, 2, 4, 1, 5, 0, 6)  # the centre first: the order a search tries columns in
# bit of the cell at row-major index k (row * 7 + column), for unpacking a board into units
_BIT_OF_CELL = np.array([(k % WIDTH) * H1 + k // WIDTH for k in range(CELLS)])


def won(stones: int) -> bool:
    """Whether ``stones`` holds four in a line: vertical, horizontal or on either diagonal."""
    for shift in (1, H1, H1 - 1, H1 + 1):
        pairs = stones & (stones >> shift)
        if pairs & (pairs >> (2 * shift)):
            return True
    return False


def mirrored(stones: int) -> int:
    """The board with its columns reversed."""
    out = 0
    for c in range(WIDTH):
        out |= ((stones >> (c * H1)) & 0x7F) << ((WIDTH - 1 - c) * H1)
    return out


class Position:
    """A position with the side to move. ``play`` returns the position after a column."""

    __slots__ = ("mover", "mask", "plies")

    def __init__(self, mover: int = 0, mask: int = 0, plies: int = 0) -> None:
        self.mover, self.mask, self.plies = mover, mask, plies

    @classmethod
    def after(cls, columns) -> Position:
        position = cls()
        for column in columns:
            position = position.play(int(column))
        return position

    def playable(self, column: int) -> bool:
        return not self.mask & TOP[column]

    def legal(self) -> list[int]:
        return [c for c in ORDER if not self.mask & TOP[c]]

    def play(self, column: int) -> Position:
        """The position after the side to move drops a stone into ``column``."""
        return Position(self.mover ^ self.mask, self.mask | (self.mask + BOTTOM[column]), self.plies + 1)

    def wins(self, column: int) -> bool:
        """Whether the side to move completes a line by playing ``column``."""
        stone = (self.mask + BOTTOM[column]) & COLUMN[column]
        return won(self.mover | stone)

    @property
    def placed(self) -> int:
        """The stones of the side that moved last."""
        return self.mover ^ self.mask

    @property
    def lost(self) -> bool:
        """Whether the side that moved last has completed a line."""
        return won(self.mover ^ self.mask)

    @property
    def full(self) -> bool:
        return self.mask == FULL

    @property
    def key(self) -> tuple[int, int]:
        """The position up to its mirror image, as the side that moved last sees it."""
        own, other = self.mover ^ self.mask, self.mover
        return min((own, other), (mirrored(own), mirrored(other)))

    def cells(self) -> np.ndarray:
        """The 42 cells row-major from the bottom-left as the side to move sees them:
        0 empty, 1 its own stone, 2 the other side's (what the solver and the opponents read)."""
        own = _unpack(np.array([self.mover], dtype=np.uint64))[0]
        other = _unpack(np.array([self.mover ^ self.mask], dtype=np.uint64))[0]
        return (own + 2 * other).astype(np.int64)

    def __repr__(self) -> str:
        cells = self.cells().reshape(HEIGHT, WIDTH)[::-1]
        return "\n".join("".join(".xo"[v] for v in row) for row in cells)


def _unpack(boards: np.ndarray) -> np.ndarray:
    """The 42 cells of each 64-bit board as 0/1 units, row-major from the bottom-left."""
    bits = np.unpackbits(boards.astype("<u8").view(np.uint8).reshape(-1, 8), axis=1, bitorder="little")
    return bits[:, _BIT_OF_CELL]


def readings(keys) -> np.ndarray:
    """The ``(n, 84)`` readings of positions given by their ``key``."""
    pairs = np.array(list(keys), dtype=np.uint64).reshape(-1, 2)
    return np.concatenate((_unpack(pairs[:, 0]), _unpack(pairs[:, 1])), axis=1).astype(float)
