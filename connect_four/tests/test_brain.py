import copy
import json
from pathlib import Path

import numpy as np
import pytest

from agent.life import Moment
from connect_four.brain import CANDIDATE, OPPONENT, Brain
from connect_four.env import MOVER, GameConfig, World, observation
from connect_four.evaluate import TERMINAL_CLASSES
from connect_four.opponents import RandomOpponent
from connect_four.tests.reference import reference_drop, reference_winner

STAGE_DIR = Path(__file__).resolve().parents[1]
BASE = json.loads((STAGE_DIR / "config.json").read_text())
DEBUG = GameConfig.debug()


def debug_config() -> dict:
    config = copy.deepcopy(BASE)
    config["game"] = dict(config["debug_game"])
    return config


def play(brain: Brain, world: World, first: str, opponent) -> int:
    m = world.reset(first=first, opponent=opponent)
    while True:
        d = brain.step(m)
        if d is None:
            return world.outcome()
        mid = world.act(d.decision_id, d.action)
        assert brain.step(mid) is None
        if mid.terminated:
            return world.outcome()
        m = world.reply()


def lived(games: int, seed: int = 0, epsilon: float = 1.0) -> tuple[Brain, World]:
    brain = Brain(debug_config(), seed)
    world = World(DEBUG, life_id=f"test-{seed}")
    brain.set_epsilon(epsilon)
    for g in range(games):
        play(brain, world, "candidate" if g % 2 == 0 else "opponent", RandomOpponent(100 + g))
    return brain, world


def observed(cells: list[int], to_move: int) -> dict:
    """The observation of a candidate-relative board with ``to_move`` (1 candidate, 2 opponent) next."""
    mover = MOVER.index("candidate") if to_move == OPPONENT else MOVER.index("opponent")
    return observation(np.asarray(cells), mover)


def random_positions(n: int, seed: int) -> list[tuple[list[int], int]]:
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        cells, stone = [0] * DEBUG.cells, 1
        for _ in range(int(rng.integers(0, 12))):
            legal = [c for c in range(DEBUG.cols) if cells[(DEBUG.rows - 1) * DEBUG.cols + c] == 0]
            if not legal or reference_winner(cells, DEBUG.rows, DEBUG.cols, DEBUG.connect) is not None:
                break
            cells = reference_drop(cells, DEBUG.rows, DEBUG.cols, int(legal[rng.integers(len(legal))]), stone)
            stone = 3 - stone
        if reference_winner(cells, DEBUG.rows, DEBUG.cols, DEBUG.connect) is None and any(v == 0 for v in cells[-DEBUG.cols :]):
            out.append((cells, stone))
    return out


def exact_next_boards(brain: Brain, positions) -> float:
    hits = []
    for cells, stone in positions:
        for col in range(DEBUG.cols):
            if cells[(DEBUG.rows - 1) * DEBUG.cols + col] == 0:
                predicted = brain.predict_board(observed(cells, stone), col)
                hits.append(predicted.tolist() == reference_drop(cells, DEBUG.rows, DEBUG.cols, col, stone))
    return float(np.mean(hits))


def test_gravity_is_learned_after_a_few_games():
    positions = random_positions(150, seed=1)
    assert exact_next_boards(Brain(debug_config(), 0), positions) < 0.2  # empty records compose no valid move
    brain, _ = lived(games=12)
    assert brain.counts["transitions"] > 80 and brain.counts["drop_writes"] == DEBUG.cols * brain.counts["transitions"]
    assert exact_next_boards(brain, positions) >= 0.97
    assert brain.extended == (np.mean(brain.validity) >= BASE["gates"]["next_board_validity"] and len(brain.validity) >= brain.validity_min)


def test_a_completed_line_is_detected_and_terminal_classes_compose():
    boards = {
        "win": [1, 1, 1, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # the candidate's bottom row
        "loss": [1, 2, 1, 0, 1, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0],  # the opponent's column 1
        "nonterminal": [1, 1, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "draw": [1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1],
    }
    diagonal = [1, 2, 2, 0, 0, 1, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    for name, cells in boards.items():
        winner = reference_winner(cells, DEBUG.rows, DEBUG.cols, DEBUG.connect)
        assert winner == {"win": 0, "loss": 1}.get(name), name
    assert reference_winner(diagonal, DEBUG.rows, DEBUG.cols, DEBUG.connect) == 0
    fresh = Brain(debug_config(), 0)
    assert TERMINAL_CLASSES[int(np.argmax(fresh.predict_terminal(observed(boards["win"], OPPONENT))))] == "nonterminal"
    brain, _ = lived(games=20)
    for name, cells in boards.items():
        mover = OPPONENT if name == "win" else CANDIDATE
        scores = brain.predict_terminal(observed(cells, mover))
        assert scores.shape == (4,) and TERMINAL_CLASSES[int(np.argmax(scores))] == name, (name, scores)
    assert TERMINAL_CLASSES[int(np.argmax(brain.predict_terminal(observed(diagonal, OPPONENT))))] == "win"


def test_the_validator_counts_rejected_boards_and_never_repairs_them():
    brain = Brain(debug_config(), 0)  # empty records: every column reads the first landing class
    empty = np.zeros(DEBUG.cells, np.int8)
    children, valid = brain.imagination.compose(empty, CANDIDATE, np.arange(DEBUG.cols))
    assert not valid.any() and (children != 0).sum(axis=1).tolist() == [DEBUG.cols] * DEBUG.cols
    assert int((brain.predict_board(observed([0] * DEBUG.cells, CANDIDATE), 2) != 0).sum()) == DEBUG.cols  # returned as composed
    world = World(DEBUG, life_id="validator")
    d = brain.step(world.reset(first="candidate"))
    assert brain.planner.invalid >= DEBUG.cols and d.budget["invalid"] >= DEBUG.cols and d.prediction["valid"][0] == 0.0
    assert d.action_mask[d.action] and d.controller == "planner"
    none = DEBUG.rows

    def with_landings(unchosen, chosen):
        brain.drop.landings = lambda rel: (np.asarray(unchosen), np.asarray(chosen))

    board = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], np.int8)
    with_landings([none] * 4, [1, 0, 0, 0])
    children, valid = brain.imagination.compose(board, OPPONENT, np.array([0, 1]))
    assert valid.tolist() == [True, True] and children[0, 4] == OPPONENT and children[1, 1] == OPPONENT
    with_landings([none] * 4, [0, 0, 0, 0])  # onto the occupied cell of column 0
    children, valid = brain.imagination.compose(board, OPPONENT, np.array([0]))
    assert not valid[0] and children[0, 0] == OPPONENT  # the overwritten stone stays in the returned board
    with_landings([none, 0, none, none], [1, 0, 0, 0])  # an extra stone in an unchosen column
    children, valid = brain.imagination.compose(board, OPPONENT, np.array([0]))
    assert not valid[0] and children[0, 1] == OPPONENT and children[0, 4] == OPPONENT
    with_landings([none] * 4, [none] * 4)  # no stone at all
    assert not brain.imagination.compose(board, OPPONENT, np.array([2]))[1][0]
    with_landings([none] * 4, [3, 3, 3, 3])  # a floating stone: gravity is learned, the validator does not know it
    children, valid = brain.imagination.compose(board, OPPONENT, np.array([2]))
    assert valid[0] and children[0, 14] == OPPONENT and children[0, 2] == 0


def test_frozen_copies_share_nothing_and_never_learn():
    brain, world = lived(games=10)
    frozen = brain.frozen()
    live_tables = {k: v.copy() for k, v in {**brain.drop.records.tables, **brain.lines.records.tables}.items()}
    frozen_tables = {k: v.copy() for k, v in {**frozen.drop.records.tables, **frozen.lines.records.tables}.items()}
    live_hash = brain.agent.state_hash()
    frozen.set_epsilon(0.0)
    other = World(DEBUG, life_id="frozen")
    for g in range(4):
        play(frozen, other, "candidate" if g % 2 == 0 else "opponent", RandomOpponent(g))
    assert frozen.planner.searches > 0 and frozen.agent.config.learning is False
    assert frozen.counts["drop_writes"] == frozen.counts["line_writes"] == frozen.counts["value_writes"] == 0
    for name, table in {**frozen.drop.records.tables, **frozen.lines.records.tables}.items():
        np.testing.assert_array_equal(table, frozen_tables[name])
    for name, table in {**brain.drop.records.tables, **brain.lines.records.tables}.items():
        np.testing.assert_array_equal(table, live_tables[name])
    assert brain.agent.state_hash() == live_hash and brain.planner.searches == 0
    for g in range(3):
        play(brain, world, "candidate", RandomOpponent(50 + g))  # the live brain keeps learning
    assert brain.counts["drop_writes"] > 0 and not np.array_equal(brain.lines.records.tables["value"], live_tables["value"])
    for name, table in {**frozen.drop.records.tables, **frozen.lines.records.tables}.items():
        np.testing.assert_array_equal(table, frozen_tables[name])
    corrupted = brain.frozen()
    corrupted.corrupt_dynamics()
    assert not np.array_equal(corrupted.drop.records.tables["landing"], brain.drop.records.tables["landing"])
    assert exact_next_boards(corrupted, random_positions(40, 3)) < exact_next_boards(brain, random_positions(40, 3))


def test_the_snapshot_policy_never_learns():
    brain, world = lived(games=10)
    policy = brain.snapshot_policy()
    saved = policy.imagination
    tables = {k: v.copy() for k, v in {**saved.drop.records.tables, **saved.lines.records.tables}.items()}
    positions = random_positions(20, seed=5)
    values = [saved.one_step(np.asarray(cells, np.int8), CANDIDATE, np.flatnonzero(np.asarray(cells[-DEBUG.cols :]) == 0)) for cells, _ in positions]
    counts = dict(brain.counts)
    for cells, _ in positions:
        legal = np.asarray(cells[-DEBUG.cols :]) == 0
        assert legal[policy(np.asarray(cells), legal)]
    assert brain.counts == counts and brain.planner.searches == 0
    for g in range(4):
        play(brain, world, "opponent", RandomOpponent(70 + g))
    assert not np.array_equal(brain.lines.records.tables["value"], tables["value"])
    for name, table in {**saved.drop.records.tables, **saved.lines.records.tables}.items():
        np.testing.assert_array_equal(table, tables[name])
    for (cells, _), before in zip(positions, values, strict=True):
        np.testing.assert_array_equal(saved.one_step(np.asarray(cells, np.int8), CANDIDATE, np.flatnonzero(np.asarray(cells[-DEBUG.cols :]) == 0)), before)


def test_a_checkpoint_rebuilds_the_brain_completely(tmp_path):
    brain, world = lived(games=8)
    files = brain.save(tmp_path / "brain_games000008")
    assert [f.name for f in files] == ["brain_games000008.agent.npz", "brain_games000008.records.npz"]
    loaded = Brain.load(tmp_path / "brain_games000008")
    assert loaded.agent.state_hash() == brain.agent.state_hash() and loaded.counts == brain.counts
    assert list(loaded.validity) == list(brain.validity) and loaded.planner.stats() == brain.planner.stats()
    for cortex in ("drop", "lines"):
        a, b = getattr(brain, cortex).records, getattr(loaded, cortex).records
        assert a.to_dict() == b.to_dict() and a.seen == b.seen and a.writes == b.writes
        np.testing.assert_array_equal(a.mean, b.mean)
        for name in a.tables:
            np.testing.assert_array_equal(a.tables[name], b.tables[name])
    for cells, stone in random_positions(15, seed=9):
        obs = observed(cells, stone)
        np.testing.assert_array_equal(brain.predict_terminal(obs), loaded.predict_terminal(obs))
        for col in range(DEBUG.cols):
            if cells[-DEBUG.cols + col] == 0:
                np.testing.assert_array_equal(brain.predict_board(obs, col), loaded.predict_board(obs, col))
    resumed = World.resume(world.state(), life_id=world.life_id)
    for g in range(3):  # the saved life continues exactly: the same exploration, the same writes
        assert play(brain, world, "candidate", RandomOpponent(900 + g)) == play(loaded, resumed, "candidate", RandomOpponent(900 + g))
    np.testing.assert_array_equal(brain.lines.records.tables["value"], loaded.lines.records.tables["value"])
    assert brain.agent.state_hash() == loaded.agent.state_hash()
    brain.step(world.reset(first="candidate", opponent=RandomOpponent(1)))
    with pytest.raises(ValueError):
        brain.save(tmp_path / "pending")  # never while a decision awaits its outcome


def test_the_intermediate_moment_is_merged_into_the_reply_feedback():
    brain = Brain(debug_config(), 0)
    brain.set_epsilon(1.0)
    world = World(DEBUG, life_id="merge")
    m = world.reset(first="candidate", opponent=RandomOpponent(3))
    d = brain.step(m)
    assert d.decision_id == 0 and brain.agent.pending[0].decision.decision_id == 0
    mid = world.act(d.decision_id, d.action)
    with pytest.raises(ValueError):
        brain.step(Moment(**{**mid.__dict__, "feedback_for": 7}))  # names another decision
    assert brain.counts["transitions"] == 0  # nothing was learned from the refused moment
    assert brain.step(mid) is None
    assert brain.agent.pending[0] is not None and brain.agent.ledger.real_transitions == 0  # the agent waits for the reply
    assert brain.counts["transitions"] == 1  # the candidate's own move was learned, with the candidate as mover
    reply = world.reply()
    d2 = brain.step(reply)
    assert d2.decision_id == 1 and brain.agent.ledger.real_transitions == 1 and brain.counts["transitions"] == 2
    while True:
        mid = world.act(d2.decision_id, d2.action)
        assert brain.step(mid) is None
        if mid.terminated:
            assert brain.agent.pending[0] is None  # a terminal intermediate moment goes to the agent itself
            break
        reply = world.reply()
        d2 = brain.step(reply)
        if d2 is None:
            assert reply.terminated and brain.agent.pending[0] is None
            break
    assert brain.counts["games_written"] == 1 and brain.counts["value_writes"] > 0
    probe = World(DEBUG, life_id="probe")
    frozen = brain.frozen()
    start = probe.start_from(type(probe.position())((1, 2, 0, 0) + (0,) * 12, 0))
    decision = frozen.step(start)
    mid = probe.act(decision.decision_id, decision.action)
    assert frozen.step(mid) is None
    if not mid.terminated:
        assert frozen.step(probe.truncate()) is None and frozen.agent.pending[0] is None
