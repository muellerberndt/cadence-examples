"""The two online world models: shapes, one step of learning, the ring, batched imagination."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from agent.brain import Field
from arm.brain import PREDICTED, graph_spec
from arm.brain import OnlineMLP as ArmMLP
from arm.env import Arm, ArmConfig
from comparisons.networks import OnlineMLP, OnlineTransformer, Reading

ROOT = Path(__file__).resolve().parents[2]
ARM_CONFIG = json.loads((ROOT / "arm" / "config.json").read_text())

MIXED = Reading(
    observation=(
        Field("room", "categorical", 4),
        Field("colour", "categorical", 3),
        Field("speed", "continuous", 2, -1.0, 1.0),
    ),
    prediction=(
        Field("next_room", "categorical", 4, source="room"),
        Field("reward", "continuous", 1, -1.0, 1.0, source="reward"),
    ),
    actions=3,
    goal=2,
    reads=5,
    flags=True,
)


def mixed_moment(rng: np.random.Generator) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    observation = {
        "room": np.eye(4)[int(rng.integers(4))],
        "colour": np.eye(3)[int(rng.integers(3))],
        "speed": rng.uniform(-1.0, 1.0, 2),
    }
    observed = {"room": np.ones(4, bool), "colour": np.zeros(3, bool), "speed": np.ones(2, bool)}
    return observation, observed, rng.uniform(0, 1, 2), rng.uniform(0, 1, 5)


def targets(rng: np.random.Generator) -> dict:
    return {"next_room": np.eye(4)[int(rng.integers(4))], "reward": np.array([float(rng.uniform(-1, 1))])}


def networks(**over):
    return [OnlineMLP(MIXED, hidden=(16,), seed=0, **over), OnlineTransformer(MIXED, width=16, heads=2, layers=1, feedforward=32, seed=0, **over)]


@pytest.mark.parametrize("model", networks())
def test_predictions_have_the_declared_shapes_and_ranges(model):
    rng = np.random.default_rng(1)
    observation, observed, goal, reads = mixed_moment(rng)
    out = model.predict_batch([observation] * 3, [0, 1, 2], observed=[observed] * 3, goal=goal, reads=reads)
    assert len(out) == 3
    for prediction in out:
        assert sorted(prediction) == ["next_room", "reward"]
        assert prediction["next_room"].shape == (4,)
        assert np.isclose(prediction["next_room"].sum(), 1.0) and (prediction["next_room"] >= 0).all()
        assert prediction["reward"].shape == (1,)
        assert -1.0 <= float(prediction["reward"][0]) <= 1.0


@pytest.mark.parametrize("model", networks())
def test_the_reading_covers_every_field_the_flags_the_goal_the_action_and_the_reads(model):
    widths = [f.width + 1 for f in MIXED.observation] + [MIXED.goal, MIXED.actions, MIXED.reads]
    assert model.encoder.token_widths == widths
    assert model.encoder.width == sum(widths)
    rng = np.random.default_rng(2)
    observation, observed, goal, reads = mixed_moment(rng)
    row = model.encoder.encode([observation], [1], [observed], goal, reads)[0]
    assert row[model.encoder.action_at + 1] == 1.0
    # colour is unobserved: its values read zero and its missing flag is on
    assert list(model.encoder.flags) == [4, 8, 11]
    assert row[8] == 1.0 and row[5:8].tolist() == [0.0, 0.0, 0.0]
    assert row[0:4].tolist() == observation["room"].tolist() and row[4] == 0.0


@pytest.mark.parametrize("model", networks(lr=0.01))
def test_one_step_of_learning_reduces_the_loss_of_that_transition(model):
    rng = np.random.default_rng(3)
    observation, observed, goal, reads = mixed_moment(rng)
    outcome = targets(rng)
    before = model.loss(observation, 2, outcome, observed=observed, goal=goal, reads=reads)
    reported = model.learn(observation, 2, outcome, observed=observed, goal=goal, reads=reads)
    after = model.loss(observation, 2, outcome, observed=observed, goal=goal, reads=reads)
    assert np.isclose(reported, before, rtol=1e-4)
    assert after < before
    assert model.updates == 1 and model.replay_updates == 0 and not model.ring


@pytest.mark.parametrize("model", networks(replay=1, capacity=3))
def test_the_ring_holds_the_last_transitions_and_replays_stored_ones(model):
    rng = np.random.default_rng(4)
    seen = []
    replayed = []
    original = model._update

    def watched(row, encoded):
        replayed.append(row.copy())
        return original(row, encoded)

    model._update = watched
    for step in range(5):
        observation, observed, goal, reads = mixed_moment(rng)
        row = model.encoder.encode([observation], [step % 3], [observed], goal, reads)[0]
        seen.append(row)
        model.learn(observation, step % 3, targets(rng), observed=observed, goal=goal, reads=reads)
    assert model.updates == 5
    assert model.replay_updates == 4  # every transition but the first replays one stored transition
    assert len(model.ring) == 3 and model.capacity == 3
    held = [row for row, _ in model.ring]
    assert all(any(np.array_equal(row, kept) for kept in seen[2:]) for row in held)  # the last three
    for k in range(5):
        real = replayed[2 * k - 1] if k else replayed[0]
        assert np.array_equal(real, seen[k])  # the real transition of each step is learned first
    for k in range(1, 5):
        drawn = replayed[2 * k]
        assert any(np.array_equal(drawn, earlier) for earlier in seen[:k])  # stored before this one
        assert not np.array_equal(drawn, seen[k])


def test_the_transformers_imagined_batch_matches_its_single_predictions():
    model = OnlineTransformer(MIXED, width=32, heads=4, layers=2, feedforward=64, seed=7)
    rng = np.random.default_rng(5)
    observation, observed, goal, reads = mixed_moment(rng)
    for step in range(3):  # after a few writes the readout is no longer flat
        model.learn(observation, step % 3, targets(rng), observed=observed, goal=goal, reads=reads)
    batch = [mixed_moment(rng) for _ in range(6)]
    actions = [k % 3 for k in range(6)]
    together = model.predict_batch([b[0] for b in batch], actions, observed=[b[1] for b in batch], goal=goal, reads=reads)
    for (observation, observed, _, _), action, batched in zip(batch, actions, together, strict=True):
        single = model.predict(observation, action, observed=observed, goal=goal, reads=reads)
        for name, value in single.items():
            np.testing.assert_allclose(value, batched[name], atol=1e-5)


def test_the_perceptron_with_one_hidden_layer_is_the_arms_forward_model():
    """The arm's own baseline, generalised: the same reading, the same steps, the same numbers."""
    reading = Reading.of(graph_spec(ARM_CONFIG), goal=False, flags=False, reads=False)
    mine = OnlineMLP(reading, hidden=(64,), lr=ARM_CONFIG["baselines"]["mlp_lr"], seed=5)
    theirs = ArmMLP(hidden=64, lr=ARM_CONFIG["baselines"]["mlp_lr"], seed=5)
    assert mine.parameters() == theirs.parameters()
    env = Arm(ArmConfig(), seed=3)
    moment = env.reset()
    rng = np.random.default_rng(0)
    for step in range(120):
        observation = {k: np.asarray(v) for k, v in moment.observation.items()}
        action = int(rng.integers(9))
        for name, value in mine.predict(observation, action).items():
            np.testing.assert_allclose(value, theirs.predict(observation, action)[name], atol=1e-9)
        moment = env.act(step, action)
        outcome = {n: np.asarray(moment.observation[n]) for n in PREDICTED}
        np.testing.assert_allclose(mine.learn(observation, action, outcome), theirs.learn(observation, action, outcome), rtol=1e-9)
        if moment.terminated or moment.truncated:
            moment = env.reset()


@pytest.mark.parametrize("model", networks(replay=1))
def test_a_copy_learns_on_its_own(model):
    rng = np.random.default_rng(6)
    observation, observed, goal, reads = mixed_moment(rng)
    outcome = targets(rng)
    model.learn(observation, 0, outcome, observed=observed, goal=goal, reads=reads)
    copied = model.copy(ring=False)
    assert not copied.ring and len(model.ring) == 1
    before = model.predict(observation, 0, observed=observed, goal=goal, reads=reads)
    for _ in range(3):
        copied.learn(observation, 0, outcome, observed=observed, goal=goal, reads=reads)
    after = model.predict(observation, 0, observed=observed, goal=goal, reads=reads)
    for name, value in before.items():
        np.testing.assert_array_equal(value, after[name])
    assert copied.updates == 4 and model.updates == 1
