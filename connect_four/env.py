"""Connect Four: a standalone engine, and a world that emits S00 moments to the candidate.

``Board`` is the rule engine: gravity, legality, alternating turns, wins in the four
directions, draws. It is used by the world, the opponents, the controls and the evaluator.
The candidate never receives a board: it receives moments, player-relative to itself.

Observation fields (``fields(config)``), every one categorical and fully observed:

    c00..c41   one cell each, row-major from the bottom-left (row 0 is the bottom row,
               index = row * cols + col), width 3: empty, self, opponent
    mover      whose move produced this board, width 2: candidate, opponent; the empty
               board at the start of a game the candidate opens counts as the opponent's
               (the other side moves next, as on every decision moment)
    phase      width 2: the candidate has no stone on the board yet, or it has moved

The goal port has width 3: ``[candidate to move, opponent to move, candidate opened]``.
The first two are one-hot for the side that moves next by alternation, terminal moments
included, so the goal never reveals whether the game is over; the third is 1 through a
game the candidate plays first.

One candidate decision spans its own move and the opponent's reply:

    decision moment      the board after the opponent's move (or the opening board);
                         no feedback; legal columns in the mask
    intermediate moment  the board after the candidate's move; feedback for the decision
                         with the executed column and a known reward (0, or the terminal
                         outcome when the move ended the game); mask all False
    reply moment         the next decision moment, or the terminal moment after the
                         opponent's reply (reward -1 for a loss, 0 for a draw)

Terminal rewards are +1, 0, -1 for a candidate win, draw and loss. The S00 contract ties a
known reward to an executed decision, so the terminal reply moment carries its reward with
``reward_known`` False; its ``terminated`` flag and ``reward`` state the outcome. Episode
ids count games of a life, event ids count moments of a life, ticks count moments of an
episode. ``state`` and ``resume`` serve tests, controls and evaluation only.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np

from agent.life import Moment

STAGE = "S03"
CELL = ("empty", "self", "opponent")
MOVER = ("candidate", "opponent")
TURN = ("candidate", "opponent")
GOAL_WIDTH = 3
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))

Opponent = Callable[[np.ndarray, np.ndarray], int]  # (cells relative to the mover, legal mask) -> column


@dataclass(frozen=True)
class GameConfig:
    rows: int = 6
    cols: int = 7
    connect: int = 4

    @property
    def cells(self) -> int:
        return self.rows * self.cols

    @classmethod
    def debug(cls) -> GameConfig:
        """The 4 by 4 connect-3 board for debugging."""
        return cls(rows=4, cols=4, connect=3)


def field_names(config: GameConfig) -> list[str]:
    return [f"c{i:02d}" for i in range(config.cells)]


def fields(config: GameConfig) -> dict[str, int]:
    out = dict.fromkeys(field_names(config), len(CELL))
    out["mover"] = len(MOVER)
    out["phase"] = 2
    return out


def one_hot(index: int, width: int) -> np.ndarray:
    out = np.zeros(width)
    out[index] = 1.0
    return out


def observation(cells: np.ndarray, mover: int) -> dict[str, np.ndarray]:
    """The moment's observation of ``cells`` (relative to the candidate, 0/1/2); the phase
    reads off the cells: 0 until the candidate's first stone is on the board."""
    eye = np.eye(len(CELL))
    out = {f"c{i:02d}": eye[int(v)] for i, v in enumerate(cells)}
    out["mover"] = one_hot(mover, len(MOVER))
    out["phase"] = one_hot(1 if any(int(v) == 1 for v in cells) else 0, 2)
    return out


def cells_of(obs: dict[str, np.ndarray], config: GameConfig) -> np.ndarray:
    """Decode the candidate-relative cells (0 empty, 1 self, 2 opponent) of an observation."""
    return np.array([int(np.asarray(obs[f"c{i:02d}"]).argmax()) for i in range(config.cells)])


def goal_vector(to_move: int, candidate_first: bool) -> np.ndarray:
    return np.array([float(to_move == 0), float(to_move == 1), float(candidate_first)])


@dataclass(frozen=True)
class Position:
    """An absolute board (0 empty, 1 first player, 2 second player) and the player to move."""

    cells: tuple[int, ...]
    to_move: int

    def key(self) -> bytes:
        return bytes(self.cells) + bytes([self.to_move])

    def board(self, config: GameConfig) -> Board:
        return Board.from_position(self, config)

    def stones(self) -> int:
        return sum(1 for v in self.cells if v)


class Board:
    """The rules. Cells are a flat list, index = row * cols + col, row 0 at the bottom;
    0 empty, 1 the first player's stone, 2 the second's. Players are 0 and 1."""

    __slots__ = ("cells", "config", "heights", "moves", "to_move", "winner")

    def __init__(self, config: GameConfig | None = None) -> None:
        self.config = config or GameConfig()
        self.cells = [0] * self.config.cells
        self.heights = [0] * self.config.cols
        self.to_move = 0
        self.winner: int | None = None
        self.moves: list[int] = []

    @classmethod
    def from_moves(cls, moves: list[int] | str, config: GameConfig | None = None) -> Board:
        board = cls(config)
        for move in moves:
            board.play(int(move))
        return board

    @classmethod
    def from_position(cls, position: Position, config: GameConfig | None = None) -> Board:
        board = cls(config)
        if len(position.cells) != board.config.cells:
            raise ValueError("position does not fit the board")
        board.cells = list(position.cells)
        board.to_move = int(position.to_move)
        cols = board.config.cols
        for col in range(cols):
            height = 0
            for row in range(board.config.rows):
                if board.cells[row * cols + col]:
                    if height != row:
                        raise ValueError("stones must rest on the bottom or on other stones")
                    height = row + 1
            board.heights[col] = height
        board.winner = board.scan_winner()
        return board

    @classmethod
    def from_relative(cls, cells: np.ndarray, config: GameConfig | None = None) -> Board:
        """A board seen by the mover: 1 is the mover's stone (player 0 here), 2 the other's."""
        board = cls(config)
        return cls.from_position(Position(tuple(int(v) for v in np.asarray(cells).ravel()), 0), board.config)

    def copy(self) -> Board:
        out = Board(self.config)
        out.cells = list(self.cells)
        out.heights = list(self.heights)
        out.to_move = self.to_move
        out.winner = self.winner
        out.moves = list(self.moves)
        return out

    @property
    def full(self) -> bool:
        return all(h >= self.config.rows for h in self.heights)

    @property
    def terminal(self) -> bool:
        return self.winner is not None or self.full

    @property
    def grid(self) -> np.ndarray:
        return np.array(self.cells, dtype=np.int8).reshape(self.config.rows, self.config.cols)

    def position(self) -> Position:
        return Position(tuple(self.cells), self.to_move)

    def key(self) -> bytes:
        return bytes(self.cells) + bytes([self.to_move])

    def is_legal(self, col: int) -> bool:
        return not self.terminal and 0 <= col < self.config.cols and self.heights[col] < self.config.rows

    def legal_columns(self) -> list[int]:
        if self.terminal:
            return []
        return [c for c in range(self.config.cols) if self.heights[c] < self.config.rows]

    def legal_mask(self) -> np.ndarray:
        mask = np.zeros(self.config.cols, bool)
        mask[self.legal_columns()] = True
        return mask

    def play(self, col: int) -> None:
        if not self.is_legal(col):
            raise ValueError(f"column {col} is not legal")
        row = self.heights[col]
        self.cells[row * self.config.cols + col] = self.to_move + 1
        self.heights[col] = row + 1
        if self.connects(row, col):
            self.winner = self.to_move
        self.moves.append(col)
        self.to_move = 1 - self.to_move

    def undo(self) -> None:
        col = self.moves.pop()
        self.heights[col] -= 1
        self.cells[self.heights[col] * self.config.cols + col] = 0
        self.winner = None
        self.to_move = 1 - self.to_move

    def connects(self, row: int, col: int) -> bool:
        """Whether the stone at (row, col) lies on a line of ``connect`` equal stones."""
        rows, cols, connect = self.config.rows, self.config.cols, self.config.connect
        stone = self.cells[row * cols + col]
        if not stone:
            return False
        for dr, dc in DIRECTIONS:
            count = 1
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < rows and 0 <= c < cols and self.cells[r * cols + c] == stone:
                    count += 1
                    r += sign * dr
                    c += sign * dc
            if count >= connect:
                return True
        return False

    def would_connect(self, col: int, player: int) -> bool:
        """Whether a stone of ``player`` dropped in ``col`` now would complete a line."""
        if not (0 <= col < self.config.cols) or self.heights[col] >= self.config.rows:
            return False
        row = self.heights[col]
        index = row * self.config.cols + col
        self.cells[index] = player + 1
        try:
            return self.connects(row, col)
        finally:
            self.cells[index] = 0

    def scan_winner(self) -> int | None:
        """The winner by scanning every occupied cell (used when a position is loaded)."""
        cols = self.config.cols
        for index, stone in enumerate(self.cells):
            if stone and self.connects(index // cols, index % cols):
                return stone - 1
        return None

    def result(self, player: int) -> int:
        """+1, 0, -1 for ``player`` on a terminal board."""
        if not self.terminal:
            raise ValueError("the game is not over")
        if self.winner is None:
            return 0
        return 1 if self.winner == player else -1

    def relative(self, player: int) -> np.ndarray:
        """The cells as ``player`` sees them: 0 empty, 1 own stone, 2 the other's."""
        own = player + 1
        return np.array([0 if v == 0 else (1 if v == own else 2) for v in self.cells])

    def text(self, player: int | None = None) -> str:
        """Rows from the top; absolute (x first player, o second) or relative to ``player``."""
        cells = self.cells if player is None else self.relative(player).tolist()
        cols = self.config.cols
        lines = []
        for row in range(self.config.rows - 1, -1, -1):
            lines.append("".join(".xo"[cells[row * cols + c]] for c in range(cols)))
        return "\n".join(lines)


class World:
    """One life's games, one at a time; the candidate acts through ``act``, the opponent
    through ``reply``. Private state stays behind underscores."""

    def __init__(self, config: GameConfig | None = None, *, life_id: str = "world") -> None:
        self.config = config or GameConfig()
        self.life_id = life_id
        self._board = Board(self.config)
        self._candidate = 0
        self._opponent: Opponent | None = None
        self._episode = -1
        self._event = 0
        self._tick = 0
        self._awaiting_reply = False
        self._closed = True
        self._pending: int | None = None
        self._executed: int | None = None

    # -- privileged access for tests, controls and evaluation only

    def state(self) -> dict:
        """Everything a resumed world needs; the opponent callable is not part of it."""
        return {
            "config": asdict(self.config),
            "cells": list(self._board.cells),
            "to_move": self._board.to_move,
            "moves": list(self._board.moves),
            "winner": self._board.winner,
            "candidate": self._candidate,
            "episode": self._episode,
            "event": self._event,
            "tick": self._tick,
            "awaiting_reply": self._awaiting_reply,
            "closed": self._closed,
        }

    @classmethod
    def resume(cls, state: dict, *, life_id: str | None = None, opponent: Opponent | None = None) -> World:
        """A world in exactly the saved state; the next moment continues the paused game."""
        world = cls(GameConfig(**state["config"]), life_id=life_id or "world")
        world._board = Board.from_position(Position(tuple(state["cells"]), state["to_move"]), world.config)
        world._board.moves = list(state["moves"])
        world._board.winner = state["winner"]
        world._candidate = int(state["candidate"])
        world._episode, world._event, world._tick = int(state["episode"]), int(state["event"]), int(state["tick"])
        world._awaiting_reply, world._closed = bool(state["awaiting_reply"]), bool(state["closed"])
        world._opponent = opponent
        return world

    @property
    def board(self) -> np.ndarray:
        return self._board.grid

    @property
    def candidate(self) -> int:
        """The candidate's player index: 0 when it opened the game, 1 otherwise."""
        return self._candidate

    def position(self) -> Position:
        return self._board.position()

    def legal_columns(self) -> list[int]:
        return self._board.legal_columns()

    def winner(self) -> str | None:
        """'candidate', 'opponent', 'draw', or None while the game continues."""
        if not self._board.terminal:
            return None
        if self._board.winner is None:
            return "draw"
        return "candidate" if self._board.winner == self._candidate else "opponent"

    def outcome(self) -> int | None:
        """+1, 0, -1 for the candidate once the game is over."""
        return None if not self._board.terminal else self._board.result(self._candidate)

    # -- episodes

    def reset(self, *, first: str = "candidate", opponent: Opponent | None = None, opening: int | None = None) -> Moment:
        """A new game. ``first`` names who opens; an opponent opening comes from the bound
        opponent callable or from ``opening``."""
        if first not in ("candidate", "opponent"):
            raise ValueError("first must be 'candidate' or 'opponent'")
        self._board = Board(self.config)
        self._candidate = 0 if first == "candidate" else 1
        self._opponent = opponent
        self._begin_episode()
        if first == "opponent":
            self._board.play(self._opponent_column(opening))
        return self._decision_moment()

    def start_from(self, position: Position, *, opponent: Opponent | None = None) -> Moment:
        """A new episode at ``position`` with the candidate to move (evaluation probes)."""
        board = Board.from_position(position, self.config)
        if board.terminal:
            raise ValueError("the position is over")
        self._board = board
        self._candidate = board.to_move
        self._opponent = opponent
        self._begin_episode()
        return self._decision_moment()

    def _begin_episode(self) -> None:
        self._episode += 1
        self._tick = 0
        self._awaiting_reply = False
        self._closed = False
        self._pending = self._executed = None

    def _opponent_column(self, column: int | None) -> int:
        if column is None:
            if self._opponent is None:
                raise RuntimeError("no opponent is bound and no column was given")
            column = int(self._opponent(self._board.relative(1 - self._candidate), self._board.legal_mask()))
        if not self._board.is_legal(int(column)):
            raise ValueError(f"the opponent's column {column} is not legal")
        return int(column)

    def act(self, decision_id: int, column: int) -> Moment:
        """Execute the candidate's committed decision; the intermediate moment carries its
        feedback and offers no action."""
        if self._closed:
            raise RuntimeError("the game is over")
        if self._awaiting_reply:
            raise RuntimeError("the opponent has not replied")
        if self._board.to_move != self._candidate:
            raise RuntimeError("it is not the candidate's move")
        if not self._board.is_legal(int(column)):
            raise ValueError(f"column {column} is not legal")
        self._pending, self._executed = int(decision_id), int(column)
        self._board.play(int(column))
        terminal = self._board.terminal
        reward = float(self._board.result(self._candidate)) if terminal else 0.0
        m = self._moment(feedback=True, reward=reward, terminated=terminal, truncated=False, mask=np.zeros(self.config.cols, bool))
        self._pending = self._executed = None
        self._awaiting_reply = not terminal
        self._closed = terminal
        return m

    def reply(self, column: int | None = None) -> Moment:
        """The opponent's move (from the bound callable or ``column``); the next decision
        moment, or the terminal moment of the game."""
        if self._closed:
            raise RuntimeError("the game is over")
        if not self._awaiting_reply:
            raise RuntimeError("it is the candidate's move")
        self._board.play(self._opponent_column(column))
        self._awaiting_reply = False
        return self._decision_moment()

    def truncate(self) -> Moment:
        """Close an unfinished episode without an outcome (evaluation probes): a truncated
        moment with the final observation and no action."""
        if self._closed:
            raise RuntimeError("the episode is already closed")
        self._closed = True
        self._awaiting_reply = False
        return self._moment(feedback=False, reward=0.0, terminated=False, truncated=True, mask=np.zeros(self.config.cols, bool))

    def _decision_moment(self) -> Moment:
        terminal = self._board.terminal
        reward = float(self._board.result(self._candidate)) if terminal else 0.0
        self._closed = terminal
        mask = np.zeros(self.config.cols, bool) if terminal else self._board.legal_mask()
        return self._moment(feedback=False, reward=reward, terminated=terminal, truncated=False, mask=mask)

    def _moment(self, *, feedback: bool, reward: float, terminated: bool, truncated: bool, mask: np.ndarray) -> Moment:
        mover = MOVER.index("opponent") if self._board.to_move == self._candidate else MOVER.index("candidate")
        obs = observation(self._board.relative(self._candidate), mover)
        to_move = TURN.index("candidate") if self._board.to_move == self._candidate else TURN.index("opponent")
        m = Moment(
            life_id=self.life_id, episode_id=self._episode, event_id=self._event, tick=self._tick,
            observation=obs, observed={}, action_mask=mask,
            feedback_for=self._pending if feedback else None, executed=self._executed if feedback else None,
            reward=reward, reward_known=feedback, terminated=terminated, truncated=truncated,
            final_observation=obs if truncated else None, goal=goal_vector(to_move, self._candidate == 0),
        )
        self._event += 1
        self._tick += 1
        return m
