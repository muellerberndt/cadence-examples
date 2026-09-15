import ast
from pathlib import Path

import numpy as np
import pytest

from connect_four.env import (
    CELL,
    MOVER,
    Board,
    GameConfig,
    World,
    cells_of,
    fields,
)
from connect_four.tests.reference import (
    reference_drop,
    reference_legal,
    reference_terminal,
    reference_winner,
)

STAGE_DIR = Path(__file__).resolve().parents[1]


def same_moment(a, b) -> None:
    for name in ("episode_id", "event_id", "tick", "feedback_for", "executed", "reward", "reward_known", "terminated", "truncated"):
        assert getattr(a, name) == getattr(b, name), name
    np.testing.assert_array_equal(a.action_mask, b.action_mask)
    np.testing.assert_array_equal(a.goal, b.goal)
    assert set(a.observation) == set(b.observation)
    for name in a.observation:
        np.testing.assert_array_equal(a.observation[name], b.observation[name])


def test_full_columns_are_illegal_and_leave_the_mask():
    w = World()
    m = w.reset()
    for k in range(3):
        mid = w.act(k, 0)
        assert not mid.terminated
        m = w.reply(0)
    assert not m.terminated and not m.action_mask[0] and m.action_mask[1:].all()
    assert w.legal_columns() == [1, 2, 3, 4, 5, 6]
    with pytest.raises(ValueError):
        w.act(3, 0)
    b = Board()
    for _ in range(6):
        b.play(0)
    assert not b.is_legal(0) and b.winner is None
    with pytest.raises(ValueError):
        b.play(0)


def test_wins_in_all_four_directions():
    sequences = {
        "horizontal": [0, 0, 1, 1, 2, 2, 3],
        "vertical": [0, 1, 0, 1, 0, 1, 0],
        "diagonal_up_right": [0, 1, 1, 2, 2, 3, 2, 3, 3, 6, 3],
        "diagonal_up_left": [6, 5, 5, 4, 4, 3, 4, 3, 3, 0, 3],
    }
    for moves in sequences.values():
        b = Board.from_moves(moves)
        assert b.winner == 0 and b.terminal and b.result(0) == 1 and b.result(1) == -1
        assert reference_winner(b.cells, 6, 7, 4) == 0
        assert b.legal_columns() == []
        before = Board.from_moves(moves[:-1])
        assert before.winner is None and not before.terminal


def test_the_last_cell_can_win_rather_than_draw():
    rng = np.random.default_rng(3)
    found = None
    for _ in range(2000):
        b = Board()
        while not b.full:
            cols = [c for c in b.legal_columns() if not b.would_connect(c, b.to_move)]
            if not cols:
                break
            b.play(int(cols[rng.integers(len(cols))]))
        if not b.full and len(b.legal_columns()) == 1 and b.heights[b.legal_columns()[0]] == 5:
            found = b
            break
    assert found is not None, "no forced last-cell win found"
    col = found.legal_columns()[0]
    mover = found.to_move
    w = World()
    m = w.start_from(found.position())
    assert w.candidate == mover and m.action_mask.sum() == 1 and m.action_mask[col]
    mid = w.act(0, col)
    assert mid.terminated and mid.reward == 1.0 and w.winner() == "candidate"
    assert reference_terminal(w.position().cells, 6, 7, 4) == (True, mover)
    # and a full board without a line is a draw, reward 0
    rng = np.random.default_rng(4)
    for _ in range(2000):
        b = Board()
        while not b.full:
            cols = [c for c in b.legal_columns() if not b.would_connect(c, b.to_move)]
            if not cols:
                break
            b.play(int(cols[rng.integers(len(cols))]))
        if b.full:
            break
    assert b.full and b.winner is None and b.terminal and b.result(0) == 0
    last = b.moves[-1]
    w = World()
    prefix = Board.from_moves(b.moves[:-1])
    w.start_from(prefix.position())
    mid = w.act(0, last)
    assert mid.terminated and mid.reward == 0.0 and w.winner() == "draw" and w.outcome() == 0


def test_terminal_before_the_next_actor_call():
    w = World()
    w.reset()
    for k, c in enumerate((0, 1, 2)):
        w.act(k, c)
        w.reply(c)
    m = w.act(3, 3)
    assert m.terminated and m.reward == 1.0 and m.reward_known
    assert m.feedback_for == 3 and m.executed == 3 and not m.action_mask.any()
    assert w.winner() == "candidate" and w.outcome() == 1
    with pytest.raises(RuntimeError):
        w.reply(0)
    with pytest.raises(RuntimeError):
        w.act(4, 0)
    # the opponent's reply ends the game: the terminal moment is the reply moment
    w = World()
    w.reset()
    for k, c in enumerate((0, 1, 2)):
        w.act(k, 6)
        w.reply(c)
    w.act(3, 5)
    m = w.reply(3)
    assert m.terminated and m.reward == -1.0 and not m.reward_known and m.feedback_for is None
    assert not m.action_mask.any() and w.winner() == "opponent" and w.outcome() == -1
    with pytest.raises(RuntimeError):
        w.act(4, 0)
    with pytest.raises(RuntimeError):
        w.reply(0)


def test_player_perspective_is_relative_to_the_candidate():
    a = World()
    ma = a.reset(first="candidate")
    assert ma.observation["mover"].argmax() == MOVER.index("opponent") and ma.goal.tolist() == [1.0, 0.0, 1.0]
    assert cells_of(ma.observation, a.config).sum() == 0 and ma.observation["phase"].argmax() == 0
    mid = a.act(0, 3)
    assert mid.observation["c03"].argmax() == CELL.index("self")
    m = a.reply(2)
    assert m.observation["c02"].argmax() == CELL.index("opponent") and m.observation["c03"].argmax() == CELL.index("self")
    assert a.board[0, 3] == 1 and a.board[0, 2] == 2 and a.candidate == 0
    b = World()
    mb = b.reset(first="opponent", opening=3)
    assert mb.observation["c03"].argmax() == CELL.index("opponent") and mb.tick == 0
    assert mb.observation["mover"].argmax() == MOVER.index("opponent") and mb.goal.tolist() == [1.0, 0.0, 0.0]
    assert mb.observation["phase"].argmax() == 0 and mb.action_mask.all()
    mid = b.act(0, 2)
    assert mid.observation["c02"].argmax() == CELL.index("self") and mid.goal.tolist() == [0.0, 1.0, 0.0]
    assert b.board[0, 3] == 1 and b.board[0, 2] == 2 and b.candidate == 1
    np.testing.assert_array_equal(a.board, b.board)  # the same absolute board, two perspectives
    assert cells_of(m.observation, a.config).tolist() != cells_of(mid.observation, b.config).tolist()


def test_the_intermediate_moment_offers_no_action_and_carries_the_feedback():
    w = World()
    m = w.reset()
    assert m.feedback_for is None and m.executed is None and not m.reward_known and m.action_mask.all()
    mid = w.act(7, 3)
    assert mid.feedback_for == 7 and mid.executed == 3 and mid.reward_known and mid.reward == 0.0
    assert not mid.terminated and not mid.truncated and not mid.action_mask.any() and not mid.any_legal
    assert mid.observation["mover"].argmax() == MOVER.index("candidate") and mid.goal.tolist() == [0.0, 1.0, 1.0]
    assert mid.event_id == m.event_id + 1 and mid.tick == 1 and mid.episode_id == m.episode_id
    with pytest.raises(RuntimeError):
        w.act(8, 0)  # the opponent has not replied
    nxt = w.reply(3)
    assert nxt.feedback_for is None and nxt.action_mask.all() and not nxt.reward_known
    assert nxt.observation["mover"].argmax() == MOVER.index("opponent") and nxt.goal.tolist() == [1.0, 0.0, 1.0]
    assert nxt.event_id == mid.event_id + 1 and nxt.tick == 2 and nxt.observation["phase"].argmax() == 1
    with pytest.raises(RuntimeError):
        w.reply(0)  # it is the candidate's move
    assert set(nxt.observation) == set(fields(w.config)) and all(v.shape == (n,) for v, n in zip(nxt.observation.values(), fields(w.config).values(), strict=False))


def test_the_brain_module_imports_no_evaluator_controls_opponents_or_engine():
    tree = ast.parse((STAGE_DIR / "brain.py").read_text())
    modules: set[str] = set()
    names: set[str] = set()
    attributes: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.add(node.module)
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Attribute):
            attributes.add(node.attr)
    parts = {part for module in modules for part in module.split(".")} | names
    assert not parts & {"evaluate", "controls", "opponents"}
    assert "Board" not in names and "Board" not in attributes  # the engine stays outside the brain


def test_the_brain_stub_declares_the_interface():
    from connect_four.brain import Brain

    for name in ("step", "frozen", "set_epsilon", "snapshot_policy", "predict_board", "predict_terminal", "corrupt_dynamics"):
        assert callable(getattr(Brain, name, None)), name
    for name in ("agent", "planner"):
        assert hasattr(Brain, name) or name in getattr(Brain, "__annotations__", {}), name


def test_pause_and_resume_a_game_through_state():
    w = World(life_id="a")
    m = w.reset(first="opponent", opening=3)
    for k, (c, r) in enumerate(((2, 4), (2, 4))):
        w.act(k, c)
        m = w.reply(r)
        assert not m.terminated
    saved = w.state()
    w2 = World.resume(saved, life_id="a")
    assert w2.state() == saved and w2.legal_columns() == w.legal_columns()
    for k, (c, r) in enumerate(((1, 5), (0, 3)), start=2):
        same_moment(w.act(k, c), w2.act(k, c))
        ma, mb = w.reply(r), w2.reply(r)
        same_moment(ma, mb)
        assert not ma.terminated
    assert w.state() == w2.state()
    mid = w.act(4, 6)
    paused = w.state()
    assert paused["awaiting_reply"] and not mid.terminated
    w3 = World.resume(paused, life_id="a")
    with pytest.raises(RuntimeError):
        w3.act(5, 0)
    same_moment(w.reply(4), w3.reply(4))
    assert w.state() == w3.state() and w3.board.tolist() == w.board.tolist()


def test_the_engine_agrees_with_the_reference_checker_on_random_games():
    # every ply of hundreds of games on three boards, then the verdict and its timing on
    # thousands more standard games
    for config, games, seed in ((GameConfig(), 400, 1), (GameConfig.debug(), 2500, 2), (GameConfig(rows=5, cols=6, connect=4), 300, 3)):
        rows, cols, connect = config.rows, config.cols, config.connect
        rng = np.random.default_rng(seed)
        ends = {"win": 0, "draw": 0}
        for _ in range(games):
            board = Board(config)
            while True:
                over, winner = reference_terminal(board.cells, rows, cols, connect)
                assert board.terminal == over and board.winner == winner
                assert board.legal_columns() == (reference_legal(board.cells, rows, cols) if not over else [])
                rebuilt = Board.from_position(board.position(), config)
                assert rebuilt.heights == board.heights and rebuilt.winner == board.winner and rebuilt.cells == board.cells
                if over:
                    ends["win" if winner is not None else "draw"] += 1
                    break
                legal = board.legal_columns()
                col = int(legal[rng.integers(len(legal))])
                expected = reference_drop(board.cells, rows, cols, col, board.to_move + 1)
                board.play(col)
                assert board.cells == expected
        assert ends["win"] > 0
    rng = np.random.default_rng(4)
    for _ in range(3000):
        board = Board()
        while not board.terminal:
            legal = board.legal_columns()
            board.play(int(legal[rng.integers(len(legal))]))
        before = Board.from_moves(board.moves[:-1])
        assert reference_terminal(before.cells, 6, 7, 4) == (False, None)
        assert reference_terminal(board.cells, 6, 7, 4) == (True, board.winner)


def test_world_moments_agree_with_the_reference_on_random_games():
    rng = np.random.default_rng(5)
    w = World(life_id="ref")
    outcomes = {1: 0, 0: 0, -1: 0}
    for game in range(150):
        first = "candidate" if game % 2 == 0 else "opponent"
        opening = int(rng.integers(7)) if first == "opponent" else None
        m = w.reset(first=first, opening=opening)
        assert m.episode_id == game and m.tick == 0
        while not m.terminated:
            cells = w.position().cells
            legal = reference_legal(cells, 6, 7)
            assert np.flatnonzero(m.action_mask).tolist() == legal
            np.testing.assert_array_equal(cells_of(m.observation, w.config), Board.from_position(w.position()).relative(w.candidate))
            col = int(legal[rng.integers(len(legal))])
            mid = w.act(game * 100 + m.tick, col)
            over, winner = reference_terminal(w.position().cells, 6, 7, 4)
            assert mid.terminated == over and not mid.action_mask.any()
            if over:
                assert mid.reward == (0.0 if winner is None else 1.0)
                m = mid
                break
            legal = reference_legal(w.position().cells, 6, 7)
            m = w.reply(int(legal[rng.integers(len(legal))]))
            over, winner = reference_terminal(w.position().cells, 6, 7, 4)
            assert m.terminated == over
            if over:
                assert m.reward == (0.0 if winner is None else -1.0) and not m.reward_known
        outcomes[int(m.reward)] += 1
        assert w.outcome() == int(m.reward)
    assert outcomes[1] > 0 and outcomes[-1] > 0


def test_the_debug_configuration_is_four_by_four_connect_three():
    config = GameConfig.debug()
    w = World(config)
    m = w.reset()
    assert len(fields(config)) == 18 and set(m.observation) == set(fields(config)) and m.action_mask.shape == (4,)
    b = Board.from_moves([0, 1, 0, 1, 0], config)
    assert b.winner == 0 and reference_winner(b.cells, 4, 4, 3) == 0
    w.act(0, 0)
    w.reply(1)
    w.act(1, 0)
    w.reply(1)
    mid = w.act(2, 0)
    assert mid.terminated and mid.reward == 1.0 and w.winner() == "candidate"


def test_truncate_closes_an_unfinished_episode():
    w = World()
    w.reset()
    w.act(0, 3)
    w.reply(3)
    m = w.truncate()
    assert m.truncated and not m.terminated and m.final_observation is not None
    assert not m.action_mask.any() and m.feedback_for is None and not m.reward_known
    with pytest.raises(RuntimeError):
        w.act(1, 0)
    with pytest.raises(RuntimeError):
        w.truncate()
    m2 = w.reset()
    assert m2.episode_id == m.episode_id + 1 and m2.event_id == m.event_id + 1 and m2.tick == 0


def test_start_from_a_position_makes_the_mover_the_candidate():
    b = Board.from_moves([3, 3, 4, 4, 5])
    w = World()
    m = w.start_from(b.position())
    assert w.candidate == 1 and m.goal.tolist() == [1.0, 0.0, 0.0] and m.tick == 0
    assert cells_of(m.observation, w.config).tolist() == b.relative(1).tolist()
    assert m.observation["c05"].argmax() == CELL.index("opponent") and m.observation["c03"].argmax() == CELL.index("opponent")
    assert m.observation["c10"].argmax() == CELL.index("self")  # row 1, column 3: the second player's stone
    assert m.observation["phase"].argmax() == 1 and m.observation["mover"].argmax() == MOVER.index("opponent")
    with pytest.raises(ValueError):
        w.start_from(Board.from_moves([0, 0, 1, 1, 2, 2, 3]).position())
    with pytest.raises(ValueError):
        Board.from_position(type(b.position())((1,) + (0,) * 40 + (2,), 0))  # a floating stone
