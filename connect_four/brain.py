"""The S03 candidate: record cortices learn the consequences of moves, a supplied search reads them.

Two record cortices (``cadence.Records``) have local receptive fields whose records are shared
across positions, the way a cortical column repeats over space. A reading touches few records,
and the witnessed outcome is written into exactly those.

- Drop records read one column: its cells bottom to top as one-hot (empty, the side to move,
  the other side) and whether the column is the chosen one. The field ``landing`` has
  ``rows + 1`` classes: the row the stone lands in, or none. One instance codes the seven
  column readings of a board as one batch, so every observed move teaches every column. The
  imagined next board is the board plus the mover's stone at each predicted landing.
- Line records read one window of ``connect`` cells along a row, a column or a diagonal,
  relative to the side that just moved. The field ``complete`` says whether the window is a
  completed line of that side, learned from witnessed outcomes: a board after which the game
  goes on has no completed window, and a won board has one among the windows through the new
  stone. The valued field ``value`` holds the outcome of finished games for the side that just
  moved (win 1, draw 0.5, loss 0, stored centred on 0.5); a board's value is the average over
  its windows.

A win is a completed window and a draw a full board without one. The planner is negamax with
alpha-beta pruning over imagined boards: depth 2 within 256 imagined transitions, depth 4
within 1,024 once the drop records' exact validity on the brain's own recent real transitions
reaches the gate. Leaves are scored by the line records. A validator rejects an imagined board
that is not the old board plus one stone of the mover in an empty cell of the chosen column; the
branch ends with the value of the position it came from, and ``planner.invalid`` counts it.
Nothing is repaired, no engine is called and no move label is supplied.

The S00 ``Agent`` owns the event transaction (decision ids, pending feedback, credit, ledger)
with the planner as its controller and small settled regions. The cortices live in ``Brain``
and learn from every observed transition with its mover, before the agent decides.
"""

from __future__ import annotations

import copy
import json
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, ClassVar

import cadence as cd
import numpy as np

from agent.brain import ActorConfig, Agent, AgentConfig, Field, GraphSpec, WorldConfig
from agent.life import Decision, Moment, seed_for

from .env import MOVER, GameConfig, cells_of, fields

STAGE = "S03"
FORMAT = "cadence-s03-brain/1"
M31 = 0x7FFFFFFF
VIEW = np.array([[0, 1, 2], [0, 1, 2], [0, 2, 1]], dtype=np.int8)  # VIEW[side][cells]: side 1 keeps, side 2 swaps
CANDIDATE, OPPONENT = 1, 2  # stone values of a board relative to the candidate
WIN_BONUS = 1e-3  # a proven result one ply sooner scores this much better
LEAF_MARGIN = 0.01  # leaf values stay inside [margin, 1 - margin], below every proven result
TIE = 1e-9

DEFAULTS: dict[str, Any] = {
    "graph": {"workspace": 16, "dynamics": 8, "source_scale": 2.0, "bias": 0.25, "input_gain": 2.0, "dt": 0.5},
    "world": {},
    "actor": {"epsilon": 0.0},
    "drop_records": {"cells": 2000, "active": 20, "rate": 0.5, "habituation": 0.002, "bias": 0.3, "chosen_gain": 1.0},
    "line_records": {"cells": 2000, "active": 20, "rate": 0.5, "value_rate": 0.1, "habituation": 0.002, "bias": 0.3},
    "planner": {"validity_window": 500, "validity_min": 200, "bootstrap": False},
}


def brain_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """The brain's own sections of ``config["brain"]`` over the defaults."""
    own = config.get("brain", {})
    return {k: {**v, **own.get(k, {})} for k, v in DEFAULTS.items()}


def brain_seed(seed: int, stream: int, index: int = 0) -> int:
    """One generator seed per brain stream, from the stage's seed namespace."""
    return seed_for(STAGE, "brain", seed, stream, index) & M31


def checkpoint_files(stem: str | Path) -> tuple[Path, Path]:
    """The two files of a checkpoint: the agent's snapshot and the record cortices."""
    stem = Path(stem)
    return stem.with_name(stem.name + ".agent.npz"), stem.with_name(stem.name + ".records.npz")


def side_to_move(observation: Mapping[str, np.ndarray], goal: np.ndarray | None) -> int:
    """The side that moves next: from the goal when given, else the other side of ``mover``."""
    if goal is not None:
        g = np.asarray(goal, float)
        return CANDIDATE if g[0] >= g[1] else OPPONENT
    mover = int(np.asarray(observation["mover"]).argmax())
    return OPPONENT if mover == MOVER.index("candidate") else CANDIDATE


class Layout:
    """The supplied layout of the receptive fields: the cells of each column bottom to top and
    the windows of ``connect`` cells along rows, columns and both diagonals."""

    def __init__(self, game: GameConfig) -> None:
        self.game = game
        self.rows, self.cols, self.connect, self.cells = game.rows, game.cols, game.connect, game.cells
        self.columns = np.array([[r * game.cols + c for r in range(game.rows)] for c in range(game.cols)])
        windows = []
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            for r in range(game.rows):
                for c in range(game.cols):
                    spots = [(r + k * dr, c + k * dc) for k in range(game.connect)]
                    if all(0 <= rr < game.rows and 0 <= cc < game.cols for rr, cc in spots):
                        windows.append([rr * game.cols + cc for rr, cc in spots])
        self.windows = np.array(windows)
        self.top = self.columns[:, -1]
        self.column_powers = 3 ** np.arange(game.rows)
        self.window_powers = 3 ** np.arange(game.connect)
        self.column_patterns = 3**game.rows
        self.window_patterns = 3**game.connect
        self.column_of = np.arange(game.cells) % game.cols

    def column_ids(self, rel: np.ndarray) -> np.ndarray:
        """Column pattern ids of boards seen by one side: ``(..., cols)``."""
        return rel[..., self.columns] @ self.column_powers

    def window_ids(self, rel: np.ndarray) -> np.ndarray:
        """Window pattern ids of boards seen by one side: ``(..., windows)``."""
        return rel[..., self.windows] @ self.window_powers


def one_hot_digits(ids: np.ndarray, digits: int) -> np.ndarray:
    """The categorical reading of pattern ids: one one-hot of width 3 per base-3 digit."""
    d = (np.asarray(ids)[:, None] // 3 ** np.arange(digits)[None, :]) % 3
    return np.eye(3)[d].reshape(len(ids), 3 * digits)


class DropRecords:
    """Records over column readings; ``landing`` is the row a dropped stone occupies, or none."""

    def __init__(self, layout: Layout, cfg: Mapping[str, Any], seed: int) -> None:
        self.layout = layout
        self.gain = float(cfg["chosen_gain"])
        self.records = cd.Records(
            3 * layout.rows + 1, {"landing": layout.rows + 1}, cells=int(cfg["cells"]), active=int(cfg["active"]),
            rate=float(cfg["rate"]), habituation=float(cfg["habituation"]), bias=float(cfg["bias"]), seed=seed,
        )
        self._reads = np.zeros((2 * layout.column_patterns, layout.rows + 1))
        self._known = np.zeros(2 * layout.column_patterns, bool)

    def readings(self, ids: np.ndarray) -> np.ndarray:
        """Readings of column ids: the content digits as one-hot and the chosen flag (id offset)."""
        n = self.layout.column_patterns
        return np.concatenate([one_hot_digits(ids % n, self.layout.rows), (ids // n)[:, None] * self.gain], axis=1)

    def reads(self, ids: np.ndarray) -> np.ndarray:
        """The landing reads of column ids, cached until the records change."""
        ids = np.asarray(ids)
        need = np.unique(ids[~self._known[ids]])
        if need.size:
            self._reads[need] = self.records.read(self.records.code(self.readings(need), adapt=False))["landing"]
            self._known[need] = True
        return self._reads[ids]

    def landings(self, rel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predicted landing row per column of one board seen by the side to move, for an
        unchosen and for a chosen column (``rows`` means none)."""
        ids = self.layout.column_ids(rel)
        reads = self.reads(np.concatenate([ids, ids + self.layout.column_patterns]))
        land = reads.argmax(axis=1)
        return land[: self.layout.cols], land[self.layout.cols :]

    def learn(self, rel: np.ndarray, column: int, rows: np.ndarray, *, adapt: bool) -> int:
        """Write one witnessed move: the board before it seen by the mover, the chosen column,
        and the observed row of the new stone per column (``rows`` for none)."""
        lay = self.layout
        ids = lay.column_ids(rel) + lay.column_patterns * (np.arange(lay.cols) == column)
        code = self.records.code(self.readings(ids), adapt=adapt)
        eye = np.eye(lay.rows + 1)
        for j in range(lay.cols):
            self.records.write(code[:, j], {"landing": eye[int(rows[j])]})
        self._known[:] = False
        return lay.cols

    def corrupt(self, rng: np.random.Generator) -> None:
        """Permute the landing records across code cells: every reading reads other cells' records."""
        table = self.records.tables["landing"]
        perm = rng.permutation(len(table))
        while np.array_equal(perm, np.arange(len(table))):
            perm = rng.permutation(len(table))
        self.records.tables["landing"] = table[perm].copy()
        self._known[:] = False


class LineRecords:
    """Records over window readings: ``complete`` (a completed line of the side that just
    moved) at the consequence rate, and the valued ``value`` at the value rate."""

    def __init__(self, layout: Layout, cfg: Mapping[str, Any], seed: int) -> None:
        self.layout = layout
        self.records = cd.Records(
            3 * layout.connect, {"complete": 2, "value": 1}, cells=int(cfg["cells"]), active=int(cfg["active"]),
            rate=float(cfg["rate"]), valued=["value"], valued_rate=float(cfg["value_rate"]),
            habituation=float(cfg["habituation"]), bias=float(cfg["bias"]), seed=seed,
        )
        self.patterns = one_hot_digits(np.arange(layout.window_patterns), layout.connect)
        self.complete = np.zeros(layout.window_patterns)
        self.value = np.full(layout.window_patterns, 0.5)
        self._fresh = False

    def refresh(self) -> None:
        """The reads of every window pattern under the current records."""
        if self._fresh:
            return
        reads = self.records.read(self.records.code(self.patterns, adapt=False))
        positive = np.maximum(reads["complete"], 0.0)
        total = positive.sum(axis=1)
        self.complete = np.where(total > 0, positive[:, 1] / np.maximum(total, 1e-12), 0.0)
        self.value = 0.5 + reads["value"][:, 0]
        self._fresh = True

    def learn_lines(self, rel_mover: np.ndarray, new_cell: int, won: bool, *, adapt: bool) -> int:
        """Write one observed board after a move: seen by the mover and by the other side.

        A board after which the game went on (or a draw) has no completed window for either
        side. A won board has at least one completed window of the mover among the windows
        through the new stone: the candidates not also seen elsewhere on the board take
        ``complete`` where their read is highest; every other window is incomplete."""
        lay = self.layout
        written = 0
        through = (lay.windows == new_cell).any(axis=1)
        for mover_view, rel in ((True, rel_mover), (False, VIEW[OPPONENT][rel_mover])):
            ids = lay.window_ids(rel)
            code = self.records.code(self.patterns[ids], adapt=adapt)
            first = {int(i): int(k) for k, i in reversed(list(enumerate(ids)))}
            incomplete = set(first)
            if mover_view and won:
                elsewhere = {int(i) for i in ids[~through]}
                candidates = sorted({int(i) for i in ids[through]} - elsewhere)
                incomplete = elsewhere
                if candidates:
                    reads = self.records.read(code[:, [first[i] for i in candidates]])["complete"]
                    positive = np.maximum(reads, 0.0)
                    prob = positive[:, 1] / np.maximum(positive.sum(axis=1), 1e-12)
                    top = float(prob.max())
                    for i, p in zip(candidates, prob, strict=True):
                        if p >= top - TIE:
                            self.records.write(code[:, first[i]], {"complete": np.array([0.0, 1.0])})
                            written += 1
            for i in sorted(incomplete):
                self.records.write(code[:, first[i]], {"complete": np.array([1.0, 0.0])})
                written += 1
        self._fresh = False
        return written

    def learn_values(self, boards: list[tuple[np.ndarray, float]]) -> int:
        """Write a finished game: each window pattern of the game once, toward the mean outcome
        of the boards that held it (each board seen by the side that had just moved)."""
        total: dict[int, float] = {}
        count: dict[int, int] = {}
        for ids, outcome in boards:
            for i in np.unique(ids):
                total[int(i)] = total.get(int(i), 0.0) + outcome
                count[int(i)] = count.get(int(i), 0) + 1
        if not total:
            return 0
        order = sorted(total)
        code = self.records.code(self.patterns[order], adapt=False)
        for k, i in enumerate(order):
            self.records.write(code[:, k], {"value": np.array([total[i] / count[i] - 0.5])})
        self._fresh = False
        return len(order)


class Imagination:
    """Imagined consequences read from the records: composed boards, their validity, terminal
    status and value. Reads only."""

    def __init__(self, layout: Layout, drop: DropRecords, lines: LineRecords) -> None:
        self.layout, self.drop, self.lines = layout, drop, lines

    def legal(self, board: np.ndarray) -> np.ndarray:
        """The columns of an imagined board whose top cell is empty (the interface's legality)."""
        return np.flatnonzero(board[self.layout.top] == 0)

    def compose(self, board: np.ndarray, side: int, columns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The imagined boards after ``side`` drops a stone in each of ``columns``, and whether
        the validator accepts each: one new stone of ``side``, in an empty cell of the chosen
        column, nothing else changed. Rejected boards are returned as composed."""
        lay = self.layout
        columns = np.asarray(columns, dtype=np.int64)
        unchosen, chosen = self.drop.landings(VIEW[side][board])
        children = np.repeat(board[None, :], len(columns), axis=0)
        for j in range(lay.cols):
            rows = np.where(columns == j, chosen[j], unchosen[j])
            put = np.flatnonzero(rows < lay.rows)
            if put.size:
                children[put, rows[put] * lay.cols + j] = side
        changed = children != board[None, :]
        cell = changed.argmax(axis=1)
        valid = (changed.sum(axis=1) == 1) & (lay.column_of[cell] == columns) & (board[cell] == 0)
        return children, valid

    def outcome(self, boards: np.ndarray, mover: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """For boards just moved by ``mover``: a completed window of the mover, a full board, and
        the value for the mover (average over windows)."""
        self.lines.refresh()
        ids = self.layout.window_ids(VIEW[mover][boards])
        won = self.lines.complete[ids].max(axis=-1) > 0.5
        full = (boards != 0).all(axis=-1)
        value = self.lines.value[ids].mean(axis=-1)
        return won, full, value

    def static(self, board: np.ndarray, side: int) -> float:
        """The value of a board for ``side``, the side to move (the other side just moved)."""
        _, _, value = self.outcome(board[None, :], OPPONENT if side == CANDIDATE else CANDIDATE)
        return float(np.clip(1.0 - value[0], LEAF_MARGIN, 1.0 - LEAF_MARGIN))

    def one_step(self, board: np.ndarray, side: int, columns: np.ndarray) -> np.ndarray:
        """The learned value of each column's imagined board for ``side``: a completed line 1, a
        full board 0.5, else the value records; a rejected board keeps the position's value."""
        children, valid = self.compose(board, side, columns)
        won, full, value = self.outcome(children, side)
        out = np.clip(value, LEAF_MARGIN, 1.0 - LEAF_MARGIN)
        out = np.where(full, 0.5, out)
        out = np.where(won, 1.0, out)
        if not valid.all():
            out = np.where(valid, out, self.static(board, side))
        return out


class Spent(Exception):
    """The decision's budget of imagined transitions is spent."""


class Planner:
    """Negamax with alpha-beta pruning and iterative deepening over imagined boards.

    Statistics over the life: ``nodes`` (imagined transitions), ``searches``, ``invalid``
    (boards the validator rejected), ``depths`` (completed depth per search), ``budget``."""

    def __init__(self, imagination: Imagination, rng: np.random.Generator) -> None:
        self.imagination = imagination
        self.rng = rng
        self.nodes = 0
        self.searches = 0
        self.invalid = 0
        self.depths: dict[int, int] = {}
        self.budget = 0
        self._cache: dict[tuple[bytes, int], tuple] = {}
        self._spent = 0
        self._limit = 0

    @property
    def depth(self) -> float:
        n = sum(self.depths.values())
        return float(sum(d * k for d, k in self.depths.items()) / n) if n else 0.0

    def stats(self) -> dict[str, Any]:
        return {"nodes": self.nodes, "searches": self.searches, "invalid": self.invalid, "depth": self.depth,
                "depths": {str(k): v for k, v in sorted(self.depths.items())}, "budget": self.budget}

    def _expand(self, board: np.ndarray, side: int) -> tuple:
        key = (board.tobytes(), side)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        im = self.imagination
        columns = im.legal(board)
        if self._spent + len(columns) > self._limit:
            raise Spent
        self._spent += len(columns)
        children, valid = im.compose(board, side, columns)
        self.invalid += int((~valid).sum())
        won, full, value = im.outcome(children, side)
        order = np.lexsort((np.abs(columns - (im.layout.cols - 1) / 2), -np.where(won, 2.0, value)))
        entry = (columns, children, valid, won, full, np.clip(value, LEAF_MARGIN, 1.0 - LEAF_MARGIN), order)
        self._cache[key] = entry
        return entry

    def _negamax(self, board: np.ndarray, side: int, depth: int, alpha: float, beta: float, ply: int) -> float:
        """The value of ``board`` for ``side`` to move, searched ``depth`` plies."""
        columns, children, valid, won, full, value, order = self._expand(board, side)
        if not len(columns):
            return 0.5
        other = OPPONENT if side == CANDIDATE else CANDIDATE
        best = -1.0
        uncertain = None
        for k in order:
            if not valid[k]:
                if uncertain is None:
                    uncertain = self.imagination.static(board, side)
                v = uncertain
            elif won[k]:
                v = 1.0 - WIN_BONUS * ply
            elif full[k]:
                v = 0.5
            elif depth <= 1:
                v = float(value[k])
            else:
                v = 1.0 - self._negamax(children[k], other, depth - 1, 1.0 - beta, 1.0 - alpha, ply + 1)
            best = max(best, v)
            alpha = max(alpha, v)
            if alpha >= beta:
                break
        return best

    def search(self, board: np.ndarray, legal: np.ndarray, depth: int, budget: int) -> tuple[int, dict[str, Any]]:
        """The candidate's column for ``board`` (relative to the candidate, the candidate to
        move) and what the search saw: value, completed depth, imagined transitions."""
        self._cache = {}
        self._spent = 0
        self._limit = int(budget)
        self.budget = int(budget)
        invalid_before = self.invalid
        columns, children, valid, won, full, value, order = self._expand(board, CANDIDATE)
        rank = {int(columns[k]): i for i, k in enumerate(order)}
        root = [int(c) for c in np.flatnonzero(legal)]
        sequence = sorted(root, key=lambda c: rank.get(c, 0))
        choice, completed, root_value = list(root), 0, 0.5
        values: dict[int, float] = {}
        for d in range(1, depth + 1):
            sequence = sorted(root, key=lambda c: (-values.get(c, 0.0), rank.get(c, 0)))
            found: dict[int, float] = {}
            best = -1.0
            try:
                for c in sequence:
                    k = int(np.flatnonzero(columns == c)[0])
                    if not valid[k]:
                        v = self.imagination.static(board, CANDIDATE)
                    elif won[k]:
                        v = 1.0 - WIN_BONUS
                    elif full[k]:
                        v = 0.5
                    elif d == 1:
                        v = float(value[k])
                    else:
                        # a move cut off by this window is worse than the best by more than a tie
                        v = 1.0 - self._negamax(children[k], OPPONENT, d - 1, 0.0, 1.0 - (best - 2 * TIE), 2)
                    found[c] = v
                    best = max(best, v)
            except Spent:
                break
            values = found
            top = max(values.values())
            choice = [c for c in sequence if values[c] >= top - TIE]
            completed, root_value = d, top
            if top >= 1.0 - WIN_BONUS * depth - TIE or top <= WIN_BONUS * depth + TIE:
                break  # a proven result: a deeper search cannot change the choice
        column = int(choice[int(self.rng.integers(len(choice)))])
        k = int(np.flatnonzero(columns == column)[0])
        self.searches += 1
        self.nodes += self._spent
        self.depths[completed] = self.depths.get(completed, 0) + 1
        info = {"value": root_value, "depth": completed, "expansions": self._spent, "invalid": self.invalid - invalid_before,
                "next_board": children[k].astype(float), "valid": float(valid[k]),
                "root_values": {c: (v, children[int(np.flatnonzero(columns == c)[0])], bool(valid[int(np.flatnonzero(columns == c)[0])])) for c, v in values.items()}}
        self._cache = {}
        return column, info


class Brain:
    """The candidate behind the one ``step`` contract; see ``brain_interface.md``."""

    SUPPLIED: ClassVar[list[str]] = [
        "the Connect Four world, its opponents and its rules outside the brain; the legality mask of real moments (interface knowledge)",
        "the categorical cell encoding relative to the candidate, the mover and phase fields and the goal port",
        "the layout of the receptive fields: the cells of each column and the windows of connect cells along rows, columns and diagonals; a reading seen by one side",
        "the fixed random expansions and winner-take-all inhibition of the record cortices, their write rates and the multiple-instance write of a won board",
        "the composition of an imagined board (the mover's stone at each predicted landing) and the validator that rejects a board which is not the old board plus one stone in the chosen column, without repairing it",
        "the terminal classes composed from the line records (a win is a completed window, a draw a full board without one) and the legal columns of an imagined board (its top cell is empty)",
        "negamax with alpha-beta pruning and iterative deepening: depth 2 within 256 imagined transitions, depth 4 within 1,024 once the drop records' exact validity on recent real transitions reaches the gate",
        "the one-step snapshot policy without search (the no-planning control and the snapshot opponent)",
        "the S00 event transaction and its small settled regions (workspace, dynamics, critic)",
    ]
    LEARNED: ClassVar[list[str]] = [
        "drop records: the landing row of a dropped stone per column reading, from every observed move of either side",
        "line records, complete: which windows are completed lines, from witnessed game continuations and wins",
        "line records, value: the outcome of finished games for the side that just moved, per window pattern",
        "the S00 critic over the workspace (reported; the planner does not read it)",
    ]

    agent: Agent
    planner: Planner

    def __init__(self, config: dict[str, Any], seed: int, *, learning: bool = True) -> None:
        self.config = config
        self.seed = int(seed)
        self.learning = bool(learning)
        self.game = GameConfig(**config["game"])
        self.layout = Layout(self.game)
        cfg = brain_config(config)
        self.cfg = cfg
        self.drop = DropRecords(self.layout, cfg["drop_records"], brain_seed(seed, 2))
        self.lines = LineRecords(self.layout, cfg["line_records"], brain_seed(seed, 3))
        self.imagination = Imagination(self.layout, self.drop, self.lines)
        self.planner = Planner(self.imagination, np.random.default_rng(brain_seed(seed, 4)))
        planning = config["planning"]
        self.depth, self.budget = int(planning["depth"]), int(planning["budget"])
        self.extended_depth, self.extended_budget = int(planning["extended_depth"]), int(planning["extended_budget"])
        self.bootstrap = bool(cfg["planner"].get("bootstrap", False))
        self.validity_gate = float(config["gates"]["next_board_validity"])
        self.validity: deque[bool] = deque(maxlen=int(cfg["planner"]["validity_window"]))
        self.validity_min = int(cfg["planner"]["validity_min"])
        self.agent = Agent(self._agent_config(), seed=brain_seed(seed, 0), stream_seeds=[brain_seed(seed, 1)], planner=self._plan)
        self.is_copy = False
        self.snapshots = 0
        self.counts = {"transitions": 0, "drop_writes": 0, "line_writes": 0, "value_writes": 0, "bootstrap_writes": 0, "games_written": 0, "validity_scored": 0}
        self._episode = -1
        self._board: np.ndarray | None = None
        self._held: tuple[int, int] | None = None
        self._boards: list[tuple[np.ndarray, int]] = []

    def _agent_config(self) -> AgentConfig:
        g = self.cfg["graph"]
        spec = GraphSpec(
            observation=tuple(Field(name, "categorical", width) for name, width in fields(self.game).items()),
            prediction=(), action_fields=(self.game.cols,), goal=3, workspace=int(g["workspace"]), dynamics=int(g["dynamics"]),
            missing_flags=False, source_scale=float(g["source_scale"]), bias=float(g["bias"]), input_gain=float(g["input_gain"]), dt=float(g["dt"]),
        )
        return AgentConfig(graph=spec, world=WorldConfig(**self.cfg["world"]), actor=ActorConfig(**self.cfg["actor"]),
                           records=None, controller="planner", learning=self.learning)

    # -- the event stream

    def step(self, moment: Moment) -> Decision | None:
        new_episode = moment.episode_id != self._episode
        intermediate = moment.has_feedback and not moment.terminated and not moment.truncated
        if new_episode and self._held is not None:
            raise ValueError("a decision awaits its reply moment; a new episode cannot start")
        if intermediate:
            pending = self.agent.pending[0]
            if pending is None or moment.feedback_for != pending.decision.decision_id or moment.executed != pending.decision.action:
                raise ValueError("the intermediate moment names no pending decision of this brain")
            if moment.any_legal or self._held is not None:
                raise ValueError("an intermediate moment offers no action and follows its decision once")
        cells = cells_of(moment.observation, self.game).astype(np.int8)
        if new_episode:
            self._episode, self._board, self._boards = moment.episode_id, None, []
        elif self._board is not None:
            self._observe(self._board, cells, moment)
        self._board = cells
        if intermediate:
            self._held = (int(moment.feedback_for), int(moment.executed))
            return None
        if self._held is not None and not moment.has_feedback:
            decision_id, column = self._held
            moment = Moment(**{**moment.__dict__, "feedback_for": decision_id, "executed": column, "reward": moment.reward, "reward_known": True})
        self._held = None
        if moment.terminated:
            self._finish(moment)
        elif moment.truncated:
            self._boards = []
        return self.agent.step(moment)

    def _observe(self, before: np.ndarray, after: np.ndarray, moment: Moment) -> None:
        """One observed transition: the stone that appeared, its column and its mover."""
        changed = np.flatnonzero(after != before)
        if changed.size != 1 or before[changed[0]] != 0:
            return
        cell = int(changed[0])
        mover = CANDIDATE if int(np.asarray(moment.observation["mover"]).argmax()) == MOVER.index("candidate") else OPPONENT
        if int(after[cell]) != mover:
            return
        lay = self.layout
        column = int(moment.executed) if (moment.has_feedback and mover == CANDIDATE) else int(lay.column_of[cell])
        won = bool(moment.terminated and moment.reward != 0.0)
        if not self.is_copy:
            predicted, valid = self.imagination.compose(before, mover, np.array([column]))
            self.validity.append(bool(valid[0] and np.array_equal(predicted[0], after)))
            self.counts["validity_scored"] += 1
        self.counts["transitions"] += 1
        if not self.learning:
            return
        rel_before = VIEW[mover][before]
        rows = np.full(lay.cols, lay.rows)
        rows[lay.column_of[cell]] = cell // lay.cols
        self.counts["drop_writes"] += self.drop.learn(rel_before, column, rows, adapt=True)
        rel_after = VIEW[mover][after]
        self.counts["line_writes"] += self.lines.learn_lines(rel_after, cell, won, adapt=True)
        self._boards.append((lay.window_ids(rel_after), mover))

    def _finish(self, moment: Moment) -> None:
        """A finished game: its boards' window patterns take the outcome of their movers."""
        if self.learning and self._boards:
            candidate = {1.0: 1.0, 0.0: 0.5, -1.0: 0.0}[float(np.sign(moment.reward))]
            boards = [(ids, candidate if mover == CANDIDATE else 1.0 - candidate) for ids, mover in self._boards]
            self.counts["value_writes"] += self.lines.learn_values(boards)
            self.counts["games_written"] += 1
        self._boards = []

    # -- deciding

    @property
    def extended(self) -> bool:
        return len(self.validity) >= self.validity_min and float(np.mean(self.validity)) >= self.validity_gate

    def _plan(self, agent: Agent, row: int, moment: Moment, drive: np.ndarray, free: Any) -> tuple[int, dict[str, np.ndarray], dict[str, int]]:
        board = cells_of(moment.observation, self.game).astype(np.int8)
        depth, budget = (self.extended_depth, self.extended_budget) if self.extended else (self.depth, self.budget)
        column, info = self.planner.search(board, np.asarray(moment.action_mask, bool), depth, budget)
        if self.bootstrap and self.learning and info["depth"] >= 2:
            self._bootstrap(board, info)
        prediction = {"next_board": info["next_board"], "valid": np.array([info["valid"]]), "value": np.array([info["value"]])}
        return column, prediction, {"expansions": int(info["expansions"]), "depth": int(info["depth"]), "invalid": int(info["invalid"])}

    def _bootstrap(self, board: np.ndarray, info: dict[str, Any]) -> None:
        """What a search of at least two plies found, written into the value records as if it
        were an outcome: the root board (the opponent just moved) toward one minus the root
        value, and every searched root move's imagined board (the candidate just moved) toward
        its searched value. A forced win or loss within the horizon is thereby remembered by
        the patterns of the board that led to it, which the one-ply read then sees."""
        lay = self.layout
        boards = [(lay.window_ids(VIEW[OPPONENT][board]), float(np.clip(1.0 - info["value"], 0.0, 1.0)))]
        boards += [(lay.window_ids(VIEW[CANDIDATE][child]), float(np.clip(v, 0.0, 1.0))) for v, child, valid in info["root_values"].values() if valid]
        self.counts["bootstrap_writes"] = self.counts.get("bootstrap_writes", 0) + self.lines.learn_values(boards)

    def new_world(self) -> None:
        """Attach to another world: fresh event cursors and no board from the last one. The
        cortices, the agent's parameters and the validity record stay. Refused while a decision
        awaits its outcome."""
        self.agent.new_stream(0)
        self._episode, self._board, self._held, self._boards = -1, None, None, []

    def set_epsilon(self, epsilon: float) -> None:
        self.agent.config = replace(self.agent.config, actor=replace(self.agent.config.actor, epsilon=float(epsilon)))

    # -- copies and controls

    def frozen(self) -> Brain:
        """An isolated read-only copy: learning off, copies of every cortex, fresh cursors."""
        out = copy.copy(self)
        out.agent = self.agent.frozen()
        out.agent.planner = out._plan
        out.learning = False
        out.is_copy = True
        out.drop = copy.deepcopy(self.drop)
        out.lines = copy.deepcopy(self.lines)
        out.imagination = Imagination(self.layout, out.drop, out.lines)
        out.planner = Planner(out.imagination, copy.deepcopy(self.planner.rng))
        out.validity = deque(self.validity, maxlen=self.validity.maxlen)
        out.counts = dict.fromkeys(self.counts, 0)
        out._episode, out._board, out._held, out._boards = -1, None, None, []
        return out

    def snapshot_policy(self) -> Callable[[np.ndarray, np.ndarray], int]:
        """A saved one-step policy of the current records: each legal column's imagined board
        valued by the line records; no search, no learning, unchanged by later learning."""
        imagination = Imagination(self.layout, copy.deepcopy(self.drop), copy.deepcopy(self.lines))
        rng = np.random.default_rng(brain_seed(self.seed, 5, self.snapshots))
        self.snapshots += 1

        def policy(cells: np.ndarray, legal: np.ndarray) -> int:
            board = np.asarray(cells, dtype=np.int8).ravel()
            columns = np.flatnonzero(np.asarray(legal, bool))
            values = imagination.one_step(board, CANDIDATE, columns)
            best = np.flatnonzero(values >= values.max() - TIE)
            return int(columns[best[int(rng.integers(len(best)))]])

        policy.imagination = imagination  # type: ignore[attr-defined]  # the saved records, for inspection
        return policy

    def predict_board(self, observation: Mapping[str, np.ndarray], column: int, *, goal: np.ndarray | None = None) -> np.ndarray:
        board = cells_of(observation, self.game).astype(np.int8)
        children, _ = self.imagination.compose(board, side_to_move(observation, goal), np.array([int(column)]))
        return children[0].astype(np.int64)

    def predict_terminal(self, observation: Mapping[str, np.ndarray], *, goal: np.ndarray | None = None) -> np.ndarray:
        board = cells_of(observation, self.game).astype(np.int8)
        self.lines.refresh()
        p_win = float(self.lines.complete[self.layout.window_ids(VIEW[CANDIDATE][board])].max())
        p_loss = float(self.lines.complete[self.layout.window_ids(VIEW[OPPONENT][board])].max())
        full = float((board != 0).all())
        line = max(p_win, p_loss)
        return np.array([(1.0 - line) * (1.0 - full), p_win, (1.0 - line) * full, p_loss])

    def corrupt_dynamics(self) -> None:
        self.drop.corrupt(np.random.default_rng(brain_seed(self.seed, 6)))

    # -- inspection and checkpoints

    def state(self) -> dict[str, Any]:
        return {**self.counts, "validity": float(np.mean(self.validity)) if self.validity else None, "validity_samples": len(self.validity),
                "extended": self.extended, "planner": self.planner.stats(), "record_parameters": self.parameters()}

    def parameters(self) -> dict[str, int]:
        return {"drop_records": self.drop.records.parameters(), "line_records": self.lines.records.parameters(), **self.agent.parameters()}

    def save(self, stem: str | Path) -> list[Path]:
        """A checkpoint between games that rebuilds this brain completely: the agent's snapshot
        (``agent.save``) at ``stem.agent.npz`` and, at ``stem.records.npz``, every record cortex's
        configuration, fixed projection and offsets, tables and running statistics, the layout of
        the receptive fields, the planner's settings, statistics and generator, and the validity
        record that decides the search depth."""
        if self._held is not None or self.agent.pending[0] is not None:
            raise ValueError("a checkpoint is taken between games, with no decision pending")
        agent_file, records_file = checkpoint_files(stem)
        self.agent.save(agent_file)
        data: dict[str, np.ndarray] = {"validity": np.array(list(self.validity), dtype=bool)}
        cortices: dict[str, dict[str, Any]] = {}
        for name, cortex in (("drop", self.drop.records), ("lines", self.lines.records)):
            data[f"{name}/projection"] = cortex.projection
            data[f"{name}/offset"] = cortex.offset
            data[f"{name}/mean"] = cortex.mean
            data[f"{name}/pathway_norm"] = cortex.pathway_norm
            for field, table in cortex.tables.items():
                data[f"{name}/table/{field}"] = table
            cortices[name] = {**cortex.to_dict(), "seen": cortex.seen, "writes": cortex.writes}
        cortices["drop"]["chosen_gain"] = self.drop.gain
        cortices["drop"]["reading"] = ("one column: a one-hot (empty, the side to move, the other side) per cell from the bottom, then the chosen flag "
                                       "times chosen_gain; landing classes are the rows from the bottom, then none")
        cortices["lines"]["reading"] = ("one window: a one-hot (empty, the side that just moved, the other side) per cell in window order; complete "
                                        "classes are incomplete and complete; value is the outcome for the side that just moved minus 0.5")
        meta = {
            "format": FORMAT, "stage": STAGE, "seed": self.seed, "learning": self.learning, "is_copy": self.is_copy,
            "game": asdict(self.game), "planning": dict(self.config["planning"]), "validity_gate": self.validity_gate, "brain": self.cfg,
            "layout": {"columns": self.layout.columns.tolist(), "windows": self.layout.windows.tolist()}, "cortices": cortices,
            "constants": {"leaf_margin": LEAF_MARGIN, "win_bonus": WIN_BONUS, "tie": TIE}, "counts": dict(self.counts),
            "snapshots": self.snapshots, "planner": self.planner.stats(), "planner_generator": self.planner.rng.bit_generator.state,
            "agent_file": agent_file.name,
        }
        data["meta"] = np.array(json.dumps(meta, sort_keys=True))
        tmp = records_file.with_name(records_file.name + ".tmp")
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **data)
        tmp.replace(records_file)
        return [agent_file, records_file]

    @classmethod
    def load(cls, stem: str | Path) -> Brain:
        """The brain a checkpoint saved, its agent resumed with this brain's planner."""
        agent_file, records_file = checkpoint_files(stem)
        with np.load(records_file, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            if meta.get("format") != FORMAT:
                raise ValueError("not an S03 brain checkpoint")
            config = {"game": meta["game"], "planning": meta["planning"], "gates": {"next_board_validity": meta["validity_gate"]}, "brain": meta["brain"]}
            brain = cls(config, int(meta["seed"]), learning=bool(meta["learning"]))
            for name, cortex in (("drop", brain.drop.records), ("lines", brain.lines.records)):
                # the expansion rebuilds from its seed up to the rounding of the platform's log and
                # cos (about 2e-16); the checkpoint's own arrays are installed so the life continues exactly
                if not (np.allclose(cortex.projection, data[f"{name}/projection"], rtol=0.0, atol=1e-12) and np.allclose(cortex.offset, data[f"{name}/offset"], rtol=0.0, atol=1e-12)):
                    raise ValueError(f"the {name} projection does not rebuild from its seed")
                cortex.projection = data[f"{name}/projection"].copy()
                cortex.offset = data[f"{name}/offset"].copy()
                cortex.mean = data[f"{name}/mean"].copy()
                cortex.pathway_norm = data[f"{name}/pathway_norm"].copy()
                cortex.seen, cortex.writes = int(meta["cortices"][name]["seen"]), int(meta["cortices"][name]["writes"])
                for field in cortex.tables:
                    cortex.tables[field] = data[f"{name}/table/{field}"].copy()
            brain.validity.extend(bool(v) for v in data["validity"])
        brain.drop._known[:] = False
        brain.lines._fresh = False
        brain.counts = {k: int(v) for k, v in meta["counts"].items()}
        brain.snapshots = int(meta["snapshots"])
        stats = meta["planner"]
        brain.planner.nodes, brain.planner.searches, brain.planner.invalid = int(stats["nodes"]), int(stats["searches"]), int(stats["invalid"])
        brain.planner.depths = {int(k): int(v) for k, v in stats["depths"].items()}
        brain.planner.rng.bit_generator.state = meta["planner_generator"]
        brain.is_copy = bool(meta["is_copy"])
        brain.agent = Agent.load(agent_file, planner=brain._plan)
        return brain
