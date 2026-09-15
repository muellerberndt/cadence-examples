from collections import Counter

import numpy as np
import pytest

from connect_four.controls import MinimaxPolicy, RandomPolicy
from connect_four.env import Board, GameConfig, World, cells_of
from connect_four.evaluate import (
    TERMINAL_CLASSES,
    classify,
    heldout_moves,
    judge,
    paired_games,
    random_draw,
    tactical_suite,
    terminal_boards,
)
from connect_four.opponents import (
    DEFAULT_WEIGHTS,
    OnePlyOpponent,
    OpponentMixture,
    RandomOpponent,
    SnapshotOpponent,
)
from connect_four.tests.reference import reference_drop, reference_winner

CONFIG = GameConfig()


def relative_and_mask(case):
    board = case.position.board(CONFIG)
    return board.relative(case.candidate), board.legal_mask()


def test_the_tactical_suite_is_balanced_legal_unseen_and_judged_by_the_rules():
    suite = tactical_suite(0, n=200)
    assert len(suite) == 200
    assert all(v == 50 for v in Counter((c.kind, c.side) for c in suite).values())
    keys = [c.position.key() for c in suite]
    assert len(set(keys)) == len(keys)
    assert [c.position for c in tactical_suite(0, n=200)] == [c.position for c in suite]
    again = tactical_suite(0, n=200, exclude={keys[0], keys[1]})
    assert keys[0] not in {c.position.key() for c in again} and keys[1] not in {c.position.key() for c in again}
    assert [c.position for c in tactical_suite(1, n=200)] != [c.position for c in suite]
    for case in suite:
        board = case.position.board(CONFIG)
        mover = board.to_move
        assert not board.terminal and case.candidate == mover
        legal = board.legal_columns()
        assert case.correct and case.correct <= set(legal)
        winning = {c for c in legal if reference_winner(reference_drop(board.cells, 6, 7, c, mover + 1), 6, 7, 4) == mover}
        threats = {c for c in legal if reference_winner(reference_drop(board.cells, 6, 7, c, 2 - mover), 6, 7, 4) == 1 - mover}
        if case.kind == "win":
            assert case.correct == winning
        else:
            assert not winning and len(threats) == 1 and case.correct == threats
            blocked = reference_drop(board.cells, 6, 7, next(iter(threats)), mover + 1)
            open_cols = [c for c in range(7) if blocked[5 * 7 + c] == 0]
            above = [c for c in open_cols if reference_winner(reference_drop(blocked, 6, 7, c, 2 - mover), 6, 7, 4) == 1 - mover]
            assert not above  # the block parries for at least one ply
        assert judge(case, next(iter(case.correct))) and not judge(case, next(c for c in legal + [-1] if c not in case.correct))
    assert classify(Board()) is None


def test_the_controls_and_the_one_ply_opponent_solve_the_suite_and_random_does_not():
    suite = tactical_suite(1, n=200)
    for policy in (MinimaxPolicy(256, seed=0), MinimaxPolicy(1024, seed=0), OnePlyOpponent(0)):
        hits = [judge(case, policy(*relative_and_mask(case))) for case in suite]
        assert np.mean(hits) == 1.0, policy.name
    random = RandomPolicy(0)
    assert np.mean([judge(case, random(*relative_and_mask(case))) for case in suite]) < 0.6
    # The matched quantity is the node budget. A win in one is a forced result, which ends the
    # deepening at depth 1; every other case completes at least the candidate's planning depth
    # of 2 within 256 nodes, and 1,024 nodes reach at least as deep on the same position.
    small, big = MinimaxPolicy(256, seed=0), MinimaxPolicy(1024, seed=0)
    kinds = Counter(case.kind for case in suite[:20])
    assert kinds["win"] and kinds["block"]
    for case in suite[:20]:
        for policy in (small, big):
            before = policy.nodes
            policy(*relative_and_mask(case))
            assert policy.nodes - before <= policy.budget, policy.name  # one node per board evaluated
        if case.kind == "win":
            assert small.depths[-1] == 1 and big.depths[-1] == 1
        else:
            assert small.depths[-1] >= 2
        assert big.depths[-1] >= small.depths[-1]
    assert small.stats()["calls"] == big.stats()["calls"] == 20


def test_minimax_finds_a_two_move_win():
    # x on (0,2) and (0,3), o stacked in column 6, x to move: 1 or 4 makes a double threat
    board = Board.from_moves([2, 6, 3, 6])
    policy = MinimaxPolicy(1024, seed=0)
    choices = {policy(board.relative(0), board.legal_mask()) for _ in range(5)}
    assert choices <= {1, 4}
    # and the block: o threatens (0,3) with 0,1,2 on the bottom row; x must play 3
    board = Board.from_moves([6, 0, 6, 1, 5, 2])
    assert MinimaxPolicy(256, seed=0)(board.relative(0), board.legal_mask()) == 3


def test_heldout_moves_are_stratified_by_height_and_mover_with_exact_next_boards():
    moves = heldout_moves(0, n=240)
    assert len(moves) == 240
    counts = Counter((m.height(CONFIG), m.mover_is_candidate) for m in moves)
    assert len(counts) == 12 and all(v == 20 for v in counts.values())
    assert len({m.position.key() + bytes([m.column]) for m in moves}) == 240
    for m in moves:
        board = m.position.board(CONFIG)
        assert not board.terminal and board.is_legal(m.column)
        nxt = reference_drop(board.cells, 6, 7, m.column, board.to_move + 1)
        relative = Board.from_position(type(m.position)(tuple(nxt), 1 - board.to_move), CONFIG).relative(m.candidate)
        assert tuple(relative.tolist()) == m.expected
        assert m.changed(CONFIG).size == 1 and m.expected[m.changed(CONFIG)[0]] == (1 if m.mover_is_candidate else 2)
        obs, goal = m.inputs(CONFIG)
        np.testing.assert_array_equal(cells_of(obs, CONFIG), m.current(CONFIG))
        assert goal.tolist()[:2] == ([1.0, 0.0] if m.mover_is_candidate else [0.0, 1.0])
        assert obs["mover"].argmax() == (1 if m.mover_is_candidate else 0)
    assert [m.position for m in heldout_moves(0, n=240)] == [m.position for m in moves]


def test_terminal_boards_are_balanced_and_labelled_by_the_rules():
    cases = terminal_boards(0, n=80)
    assert len(cases) == 80 and all(v == 20 for v in Counter(c.kind for c in cases).values())
    assert {c.kind for c in cases} == set(TERMINAL_CLASSES)
    for case in cases:
        board = case.position.board(CONFIG)
        winner = reference_winner(board.cells, 6, 7, 4)
        if case.kind == "win":
            assert winner == case.candidate
        elif case.kind == "loss":
            assert winner == 1 - case.candidate
        elif case.kind == "draw":
            assert winner is None and board.full
        else:
            assert winner is None and not board.full and board.position().stones() >= 1
        obs, goal = case.inputs(CONFIG)
        np.testing.assert_array_equal(cells_of(obs, CONFIG), board.relative(case.candidate))
        assert goal[:2].sum() == 1.0
    assert Counter(c.candidate for c in cases if c.kind == "draw") == {0: 10, 1: 10}
    assert random_draw(np.random.default_rng(0), CONFIG) is not None


def test_the_paired_harness_alternates_the_opener_and_pairs_the_opponent_seeds():
    record = []

    def play(k, first, opponent):
        record.append((k, first, opponent))
        return 1 if first == "candidate" else -1

    out = paired_games(play, 10, lambda s: s, seed=0, environment=2001)
    assert out["games"] == 10 and out["score"] == 0.5 and out["wins"] == 5 and out["losses"] == 5
    assert out["first_score"] == 1.0 and out["second_score"] == 0.0
    assert [r[1] for r in record] == ["candidate", "opponent"] * 5
    seeds = [r[2] for r in record]
    assert all(seeds[2 * j] == seeds[2 * j + 1] for j in range(5)) and len(set(seeds)) == 5
    other = []
    paired_games(lambda k, f, o: other.append(o) or 0, 10, lambda s: s, seed=0, environment=2002)
    assert not set(other) & set(seeds)  # another environment id, other opponent seeds


def play_with(policy):
    """A game runner for the harness: ``policy`` moves for the candidate on a fresh world."""
    world = World(life_id="harness")

    def play(k, first, opponent):
        m = world.reset(first=first, opponent=opponent)
        while not m.terminated:
            col = policy(cells_of(m.observation, CONFIG), m.action_mask)
            mid = world.act(k * 100 + m.tick, col)
            if mid.terminated:
                return world.outcome()
            m = world.reply()
        return world.outcome()

    return play


def test_one_ply_beats_random_and_random_draws_even_with_random():
    strong = paired_games(play_with(OnePlyOpponent(7)), 100, RandomOpponent, seed=0, environment=2001)
    assert strong["score"] > 0.8
    even = paired_games(play_with(RandomPolicy(7)), 300, RandomOpponent, seed=0, environment=2001)
    assert 0.35 < even["score"] < 0.65
    minimax = paired_games(play_with(MinimaxPolicy(256, seed=1)), 40, OnePlyOpponent, seed=0, environment=2002)
    assert minimax["score"] > 0.6


def test_the_opponent_mixture_draws_by_weight_and_redistributes_without_snapshots():
    mix = OpponentMixture(DEFAULT_WEIGHTS, {"random": RandomOpponent(0), "one_ply": OnePlyOpponent(0)}, seed=0)
    names = Counter(mix.new_game()[0] for _ in range(1400))
    assert "snapshots" not in names and abs(names["random"] / 1400 - 4 / 7) < 0.05
    for k in range(10):
        mix.add_snapshot(RandomPolicy(k), f"s{k}")
    assert len(mix.snapshots) == 8 and [s.name for s in mix.snapshots] == [f"s{k}" for k in range(2, 10)]
    names = Counter(mix.new_game()[0] for _ in range(3000))
    for name, weight in DEFAULT_WEIGHTS.items():
        assert abs(names[name] / 3000 - weight) < 0.04, name
    assert mix.draws["snapshots"] == names["snapshots"]
    with pytest.raises(ValueError):
        OpponentMixture({"random": 1.0, "mystery": 1.0}, {"random": RandomOpponent(0)}, seed=0)
    bad = SnapshotOpponent(lambda cells, legal: 0, "bad")
    legal = np.ones(7, bool)
    legal[0] = False
    with pytest.raises(ValueError):
        bad(np.zeros(42, int), legal)
