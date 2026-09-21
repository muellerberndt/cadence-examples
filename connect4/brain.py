"""The Connect Four brain: a value patch that reads positions, and a search that imagines moves.

The value patch (``patch.py``) is one ``cadence.RecordPatchNet``. It says how the game ends
for the side that just placed a stone. The search is supplied: it knows the rules, tries
columns from the centre outward, imagines the boards they lead to and reads the value patch
where it stops looking. A completed line and a full board are read off the rules, so a win
the search reaches is proven and outranks any value the patch reads. The search deepens one
ply at a time until its budget of distinct read positions is spent; once ``late_stones`` are on the
board it looks as far as the end of the game, within ``late_budget``.

Values are those of the side to move at a node. A proven win ``d`` plies away is
``PROVEN - d / 100`` and a proven loss its negative, so the search prefers the nearest win
and the farthest loss; every value the patch reads lies inside ``(-PROVEN + 1, PROVEN - 1)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .game import Position
from .patch import GAMMA, ValuePatch

PROVEN = 2.0
EXACT, LOWER, UPPER = 0, 1, 2


class Spent(Exception):
    """The budget of read positions ran out inside a deepening."""


@dataclass
class Thought:
    """What one search found: the column, its value, and what the page shows of the search."""

    column: int
    value: float
    depth: int
    reads: int
    proven: bool
    columns: dict[int, float] = field(default_factory=dict)
    line: list[int] = field(default_factory=list)


def proven(value: float) -> bool:
    return abs(value) > PROVEN - 1.0


class Brain:
    def __init__(self, patch: ValuePatch, *, reads: int = 2000, late_stones: int = 18, late_reads: int = 200000,
                 visits: int = 2000000, seed: int = 0) -> None:
        self.patch = patch
        self.reads_limit, self.late_stones, self.late_reads, self.visits_limit = int(reads), int(late_stones), int(late_reads), int(visits)
        self._visits = 0
        self.rng = np.random.default_rng(seed)
        self._table: dict[tuple[int, int], tuple[int, int, float, int]] = {}
        self._values: dict[tuple[int, int], float] = {}
        self._reads = 0
        self._limit = 0

    # ------------------------------------------------------------------ search

    def _frontier(self, position: Position, ply: int) -> tuple[float, int]:
        """The value of a node one ply above the horizon: the best of its imagined boards."""
        legal = position.legal()
        children = [position.play(c) for c in legal]
        values = np.empty(len(legal))
        unread = []
        for k, child in enumerate(children):
            if child.full:
                values[k] = 0.0
            elif any(child.wins(c) for c in child.legal()):
                values[k] = -(PROVEN - (ply + 2) / 100)  # the other side completes a line next
            else:
                unread.append(k)
        if unread:
            # A position is read once in a search and its value kept: the library's batched
            # read of a row differs in the last bits with the size of the batch, and a search
            # that met one position twice would otherwise compare it with itself and differ.
            keys = [children[k].key for k in unread]
            fresh = list(dict.fromkeys(key for key in keys if key not in self._values))
            if self._reads + len(fresh) > self._limit:
                raise Spent
            if fresh:
                self._reads += len(fresh)
                self._values.update(zip(fresh, np.clip(self.patch.values(fresh), -0.999, 0.999).tolist()))
            values[unread] = [self._values[key] for key in keys]
        best = int(np.argmax(values))
        return float(values[best]), legal[best]

    def _negamax(self, position: Position, depth: int, alpha: float, beta: float, ply: int) -> float:
        self._visits += 1
        if self._visits > self.visits_limit:
            raise Spent
        legal = position.legal()
        for column in legal:
            if position.wins(column):
                return PROVEN - (ply + 1) / 100
        if position.plies == 41:  # one empty cell that completes nothing: the board fills
            return 0.0
        # What the other side threatens to complete at once decides the move: two such columns
        # cannot both be blocked, and one must be.
        other = Position(position.mover ^ position.mask, position.mask, position.plies)
        threats = [c for c in legal if other.wins(c)]
        if len(threats) > 1:
            return -(PROVEN - (ply + 2) / 100)
        forced = threats[0] if threats else None
        key = (position.mover, position.mask)
        entry = self._table.get(key)
        first = None
        if entry is not None:
            held_depth, flag, value, first = entry
            if held_depth >= depth or proven(value):
                if flag == EXACT:
                    return value
                if flag == LOWER:
                    alpha = max(alpha, value)
                else:
                    beta = min(beta, value)
                if alpha >= beta:
                    return value
        if forced is not None:
            value = -self._negamax(position.play(forced), max(depth - 1, 1), -beta, -alpha, ply + 1)
            flag = UPPER if value <= alpha else LOWER if value >= beta else EXACT
            self._table[key] = (depth, flag, value, forced)
            return value
        if depth <= 1:
            value, column = self._frontier(position, ply)
            self._table[key] = (1, EXACT, value, column)
            return value
        order = list(legal)
        if first is not None and first in order:
            order.remove(first)
            order.insert(0, first)
        best, best_column, floor = -np.inf, order[0], alpha
        for column in order:
            value = -self._negamax(position.play(column), depth - 1, -beta, -alpha, ply + 1)
            if value > best:
                best, best_column = value, column
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        flag = UPPER if best <= floor else LOWER if best >= beta else EXACT
        self._table[key] = (depth, flag, best, best_column)
        return best

    def think(self, position: Position) -> Thought:
        """Search ``position`` for the side to move. Reads privately; the patch does not change."""
        legal = position.legal()
        if not legal:
            raise ValueError("the board is full")
        self._table, self._values, self._reads, self._visits = {}, {}, 0, 0
        self._limit = self.late_reads if position.plies >= self.late_stones else self.reads_limit
        deepest = 42 - position.plies
        thought = Thought(legal[0], 0.0, 0, 0, False)
        for depth in range(1, deepest + 1):
            try:
                columns = {}
                alpha = -np.inf
                for column in self._ordered(position, legal, thought.column):
                    if position.wins(column):
                        value = PROVEN - 1 / 100
                    elif position.plies == 41:
                        value = 0.0
                    elif depth == 1:
                        value = -self._negamax(position.play(column), 1, -np.inf, np.inf, 1)
                    else:
                        value = -self._negamax(position.play(column), depth - 1, -np.inf, -alpha, 1)
                    columns[column] = value
                    alpha = max(alpha, value - 1e-9)
            except Spent:
                break
            top = max(columns.values())
            ties = [c for c, v in columns.items() if v >= top - 1e-9]
            choice = thought.column if thought.column in ties else min(ties, key=lambda c: abs(c - 3))
            thought = Thought(choice, top, depth, self._reads, proven(top), columns, self._line(position, choice))
            if proven(top) or (depth >= deepest):
                break
        thought.reads = self._reads
        return thought

    @staticmethod
    def _ordered(position: Position, legal: list[int], first: int) -> list[int]:
        return [first] + [c for c in legal if c != first] if first in legal else list(legal)

    def _line(self, position: Position, column: int, length: int = 8) -> list[int]:
        """The line the search expects, read back from its table."""
        line, position = [column], position.play(column)
        while len(line) < length and not (position.lost or position.full):
            entry = self._table.get((position.mover, position.mask))
            if entry is None or entry[3] not in position.legal():
                break
            line.append(entry[3])
            position = position.play(entry[3])
        return line

    # ---------------------------------------------------------------- learning

    def review(self, columns: list[int], brain_first: bool, *, rate: float = 0.0) -> list[dict]:
        """Learn from a finished game by replaying it backwards.

        The end of a game is a fact: the positions of its last plies are written with how it
        ended. Going back from there, the brain searches each of its own positions again,
        now with what it has just written. Where the search still chooses the column it
        played, the position's value is what that search finds and is written too; where
        the search now chooses another column, the brain has found the move it would change,
        and the review stops: earlier positions led there through a choice it no longer
        makes. A game that ended no worse than the brain expected teaches only its ending.

        ``rate`` is the slow step of each write (0 writes records alone). Returns one entry
        per written position, for the page to show."""
        positions = [Position()]
        for column in columns:
            positions.append(positions[-1].play(column))
        last = positions[-1]
        if not (last.lost or last.full):
            raise ValueError("a review needs a finished game")
        written: list[dict] = []

        def write(position: Position, value: float, why: str) -> None:
            self.patch.learn([position.key], np.array([float(np.clip(value, -1.0, 1.0))]), rate=rate)
            written.append({"stones": position.plies, "value": float(value), "why": why})

        # the ending: the last two positions, as the side that placed each stone sees them
        outcome = 1.0 if last.lost else 0.0
        write(last, outcome, "the game ended here")
        if len(positions) > 2:
            write(positions[-2], -outcome * GAMMA, "one ply before the end")
        # backwards through the brain's own moves
        mine = [k for k in range(len(columns)) if (k % 2 == 0) == brain_first]
        for k in reversed(mine):
            before, after = positions[k], positions[k + 1]
            if after is last or after is positions[-2]:
                continue
            thought = self.think(before)
            if thought.column != columns[k]:
                written.append({"stones": before.plies, "value": thought.value, "why": "it would now play another column here", "column": thought.column})
                break
            # a proven result lies ``d`` plies from the position before the move, so ``d - 1`` from the one written
            value = thought.value if not proven(thought.value) else float(np.sign(thought.value)) * GAMMA ** (round((PROVEN - abs(thought.value)) * 100) - 1)
            write(after, value, "it would play the same column; the value it now finds")
        return written

    def __call__(self, cells: np.ndarray, legal: np.ndarray) -> int:
        """The brain as an opponent ``(cells, legal) -> column`` (see ``opponents.py``)."""
        from .opponents import position_of

        return self.think(position_of(cells)).column
