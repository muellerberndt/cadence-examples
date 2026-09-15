"""A deliberately simple Connect Four reference for the tests: win detection by trying every
window of ``connect`` cells in every direction from every cell, legality by looking at the
top cell of a column, a draw by a full board without a line. Written independently of
``env.py``; it shares no code with the engine."""

from __future__ import annotations

STEPS = ((0, 1), (1, 0), (1, 1), (1, -1))


def reference_winner(cells: list[int], rows: int, cols: int, connect: int) -> int | None:
    """0 or 1 for the player with a line, None otherwise. ``cells`` is row-major, row 0 at
    the bottom, 1 the first player's stone and 2 the second's."""
    grid = [[cells[r * cols + c] for c in range(cols)] for r in range(rows)]
    for stone in (1, 2):
        for r in range(rows):
            for c in range(cols):
                for dr, dc in STEPS:
                    window = []
                    for k in range(connect):
                        rr, cc = r + k * dr, c + k * dc
                        if 0 <= rr < rows and 0 <= cc < cols:
                            window.append(grid[rr][cc])
                    if len(window) == connect and all(v == stone for v in window):
                        return stone - 1
    return None


def reference_legal(cells: list[int], rows: int, cols: int) -> list[int]:
    return [c for c in range(cols) if cells[(rows - 1) * cols + c] == 0]


def reference_terminal(cells: list[int], rows: int, cols: int, connect: int) -> tuple[bool, int | None]:
    """(over, winner): over when a line exists or no cell is empty."""
    winner = reference_winner(cells, rows, cols, connect)
    return winner is not None or all(v != 0 for v in cells), winner


def reference_drop(cells: list[int], rows: int, cols: int, col: int, stone: int) -> list[int]:
    """The cells after ``stone`` falls into ``col`` (the lowest empty cell of the column)."""
    out = list(cells)
    for r in range(rows):
        if out[r * cols + col] == 0:
            out[r * cols + col] = stone
            return out
    raise ValueError("column is full")
