"""The independent evaluator: suites generated from the held-out seed namespace, judged by
exhaustive rule checking through the engine, never by a learned model.

- ``tactical_suite``: unseen legal positions with the candidate to move, balanced between
  an immediate win available and a mandatory one-move block, stratified by the candidate's
  side; ``judge`` names the correct columns by trying every column against the rules.
- ``heldout_moves``: legal single moves stratified by column height and mover, with the
  exact next board, for the next-board validity gate.
- ``terminal_boards``: balanced candidate wins, losses, draws and nonterminal boards for
  the terminal-prediction gate.
- ``paired_games``: the paired harness, games alternating the first player with the
  opponent of each pair seeded identically, scored win 1, draw 0.5, loss 0.

Positions come from random legal play under ``seed_for``; the runner may pass the keys of
positions the candidate decided on so the tactical suite excludes them.
"""

from __future__ import annotations

from collections.abc import Callable, Container, Iterable
from dataclasses import dataclass

import numpy as np

from agent.life import seed_for

from .env import (
    MOVER,
    STAGE,
    Board,
    GameConfig,
    Opponent,
    Position,
    goal_vector,
    observation,
)

TERMINAL_CLASSES = ("nonterminal", "win", "draw", "loss")
KINDS = ("win", "block")
SIDES = ("first", "second")
MAX_GAMES = 2_000_000


def immediate_wins(board: Board, player: int) -> list[int]:
    """Columns where a stone of ``player`` would complete a line now."""
    return [c for c in board.legal_columns() if board.would_connect(c, player)]


def classify(board: Board) -> tuple[str, frozenset[int]] | None:
    """('win', winning columns) when the mover can win at once; ('block', {column}) when it
    cannot, exactly one column stops the other side's immediate win, and the block leaves
    the other side no immediate win (a block under a threat loses anyway); else None."""
    if board.terminal:
        return None
    wins = immediate_wins(board, board.to_move)
    if wins:
        return "win", frozenset(wins)
    threats = immediate_wins(board, 1 - board.to_move)
    if len(threats) != 1:
        return None
    blocked = board.copy()
    blocked.play(threats[0])
    if immediate_wins(blocked, blocked.to_move):
        return None
    return "block", frozenset(threats)


def relative_string(cells: Iterable[int]) -> str:
    """The candidate-relative board as a string, row-major from the bottom-left."""
    return "".join(".xo"[int(v)] for v in cells)


def moment_inputs(board: Board, candidate: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """The observation and goal the world would emit for ``board`` seen by ``candidate``."""
    mover = MOVER.index("opponent") if board.to_move == candidate else MOVER.index("candidate")
    obs = observation(board.relative(candidate), mover)
    return obs, goal_vector(0 if board.to_move == candidate else 1, candidate == 0)


def random_game(rng: np.random.Generator, config: GameConfig) -> list[Board]:
    """Every position of one random legal game, the terminal one last."""
    board = Board(config)
    out = [board.copy()]
    while not board.terminal:
        cols = board.legal_columns()
        board.play(int(cols[rng.integers(len(cols))]))
        out.append(board.copy())
    return out


# -- the tactical suite


@dataclass(frozen=True)
class TacticalCase:
    position: Position  # the candidate is the player to move
    kind: str  # win | block
    correct: frozenset[int]

    @property
    def candidate(self) -> int:
        return self.position.to_move

    @property
    def side(self) -> str:
        return SIDES[self.candidate]


def judge(case: TacticalCase, column: int) -> bool:
    return int(column) in case.correct


def tactical_suite(seed: int, config: GameConfig | None = None, n: int = 1000, exclude: Container[bytes] = frozenset()) -> list[TacticalCase]:
    """``n`` cases, ``n // 4`` per (kind, side) stratum, from the held-out stream, one case
    per game and stratum, none whose position key is in ``exclude`` or repeated."""
    config = config or GameConfig()
    rng = np.random.default_rng(seed_for(STAGE, "heldout", seed, 2000, 0))
    quota = {(k, s): n // 4 for k in KINDS for s in SIDES}
    for extra in range(n - 4 * (n // 4)):
        quota[(KINDS[extra % 2], SIDES[extra // 2 % 2])] += 1
    excluded = exclude
    seen: set[bytes] = set()
    out: list[TacticalCase] = []
    games = 0
    while any(quota.values()):
        games += 1
        if games > MAX_GAMES:
            raise RuntimeError("the tactical suite could not be filled")
        taken: set[tuple[str, str]] = set()
        for board in random_game(rng, config):
            found = classify(board)
            if found is None:
                continue
            kind, correct = found
            stratum = (kind, SIDES[board.to_move])
            key = board.key()
            if quota[stratum] <= 0 or stratum in taken or key in excluded or key in seen:
                continue
            out.append(TacticalCase(board.position(), kind, correct))
            seen.add(key)
            taken.add(stratum)
            quota[stratum] -= 1
    return out


# -- held-out single moves


@dataclass(frozen=True)
class HeldoutMove:
    position: Position
    column: int
    candidate: int  # the candidate's player index
    expected: tuple[int, ...]  # the next board relative to the candidate

    @property
    def mover_is_candidate(self) -> bool:
        return self.position.to_move == self.candidate

    def height(self, config: GameConfig) -> int:
        """The height of the column before the move (the stratum)."""
        return column_height(self.position, self.column, config)

    def inputs(self, config: GameConfig) -> tuple[dict[str, np.ndarray], np.ndarray]:
        return moment_inputs(self.position.board(config), self.candidate)

    def current(self, config: GameConfig) -> np.ndarray:
        return self.position.board(config).relative(self.candidate)

    def changed(self, config: GameConfig) -> np.ndarray:
        return np.flatnonzero(np.asarray(self.expected) != self.current(config))


def column_height(position: Position, column: int, config: GameConfig) -> int:
    return sum(1 for r in range(config.rows) if position.cells[r * config.cols + column])


def heldout_moves(seed: int, config: GameConfig | None = None, n: int = 10000) -> list[HeldoutMove]:
    """``n`` legal single moves stratified by the height of the column played and by the
    mover (candidate or opponent), from the held-out stream; at most two per game and
    stratum, no (position, column) repeated."""
    config = config or GameConfig()
    rng = np.random.default_rng(seed_for(STAGE, "heldout", seed, 2000, 1))
    strata = [(h, m) for h in range(config.rows) for m in (True, False)]
    quota = dict.fromkeys(strata, n // len(strata))
    for extra in range(n - len(strata) * (n // len(strata))):
        quota[strata[extra % len(strata)]] += 1
    seen: set[bytes] = set()
    out: list[HeldoutMove] = []
    games = 0
    while any(quota.values()):
        games += 1
        if games > MAX_GAMES:
            raise RuntimeError("the held-out moves could not be filled")
        taken: dict[tuple[int, bool], int] = dict.fromkeys(strata, 0)
        positions = random_game(rng, config)[:-1]
        rng.shuffle(positions)
        for board in positions:
            for col in board.legal_columns():
                mover_is_candidate = bool(rng.integers(2))
                stratum = (board.heights[col], mover_is_candidate)
                key = board.key() + bytes([col])
                if quota[stratum] <= 0 or taken[stratum] >= 2 or key in seen:
                    continue
                candidate = board.to_move if mover_is_candidate else 1 - board.to_move
                nxt = board.copy()
                nxt.play(col)
                out.append(HeldoutMove(board.position(), col, candidate, tuple(int(v) for v in nxt.relative(candidate))))
                seen.add(key)
                taken[stratum] += 1
                quota[stratum] -= 1
    return out


# -- terminal boards


@dataclass(frozen=True)
class TerminalCase:
    position: Position  # to_move: the side that would move next by alternation
    candidate: int
    kind: str  # nonterminal | win | draw | loss

    @property
    def label(self) -> int:
        return TERMINAL_CLASSES.index(self.kind)

    def inputs(self, config: GameConfig) -> tuple[dict[str, np.ndarray], np.ndarray]:
        return moment_inputs(self.position.board(config), self.candidate)


def random_draw(rng: np.random.Generator, config: GameConfig, attempts: int = 200) -> Board | None:
    """A full board without a line: random play that refuses line-completing columns,
    restarted when only such columns remain."""
    for _ in range(attempts):
        board = Board(config)
        while not board.full:
            cols = [c for c in board.legal_columns() if not board.would_connect(c, board.to_move)]
            if not cols:
                break
            board.play(int(cols[rng.integers(len(cols))]))
        if board.full and board.winner is None:
            return board
    return None


def terminal_boards(seed: int, config: GameConfig | None = None, n: int = 2000) -> list[TerminalCase]:
    """``n // 4`` boards per class. Wins and losses are decided random games seen by the
    winner or the loser; draws are line-free full boards; nonterminal boards are random
    positions with at least one stone. Candidate sides alternate within every class."""
    config = config or GameConfig()
    rng = np.random.default_rng(seed_for(STAGE, "heldout", seed, 2000, 2))
    per = n // 4
    seen: set[bytes] = set()
    out: list[TerminalCase] = []

    def add(board: Board, candidate: int, kind: str) -> bool:
        key = board.key() + bytes([candidate])
        if key in seen:
            return False
        seen.add(key)
        out.append(TerminalCase(board.position(), candidate, kind))
        return True

    counts = dict.fromkeys(TERMINAL_CLASSES, 0)
    games = 0
    while counts["win"] < per or counts["loss"] < per or counts["nonterminal"] < per:
        games += 1
        if games > MAX_GAMES:
            raise RuntimeError("the terminal boards could not be filled")
        positions = random_game(rng, config)
        last = positions[-1]
        if last.winner is not None:
            kind = "win" if games % 2 == 0 else "loss"
            if counts[kind] < per:
                candidate = last.winner if kind == "win" else 1 - last.winner
                counts[kind] += add(last, candidate, kind)
        if counts["nonterminal"] < per and len(positions) > 2:
            board = positions[int(rng.integers(1, len(positions) - 1))]
            counts["nonterminal"] += add(board, int(rng.integers(2)), "nonterminal")
    while counts["draw"] < per:
        board = random_draw(rng, config)
        if board is None:
            raise RuntimeError("no line-free full board found")
        counts["draw"] += add(board, counts["draw"] % 2, "draw")
    return out


# -- the paired harness


def paired_games(play: Callable[[int, str, Opponent], int], games: int, opponent_factory: Callable[[int], Opponent], seed: int, environment: int) -> dict:
    """``games`` games; game ``k`` is opened by the candidate when ``k`` is even and by the
    opponent when odd, and the two games of a pair use the same opponent seed. ``play``
    returns the candidate's outcome (+1, 0, -1). Score: win 1, draw 0.5, loss 0."""
    outcomes = []
    for k in range(int(games)):
        first = "candidate" if k % 2 == 0 else "opponent"
        opponent = opponent_factory(seed_for(STAGE, "heldout", seed, environment, k // 2))
        outcomes.append(int(play(k, first, opponent)))
    return summarize(outcomes)


def summarize(outcomes: list[int]) -> dict:
    o = np.asarray(outcomes, float)
    return {
        "games": int(o.size),
        "score": float(np.mean((o + 1) / 2)) if o.size else float("nan"),
        "wins": int(np.sum(o > 0)),
        "draws": int(np.sum(o == 0)),
        "losses": int(np.sum(o < 0)),
        "first_score": float(np.mean((o[0::2] + 1) / 2)) if o.size else float("nan"),
        "second_score": float(np.mean((o[1::2] + 1) / 2)) if o.size > 1 else float("nan"),
    }
