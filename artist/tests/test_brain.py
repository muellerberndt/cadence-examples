"""The brain's contract: the declared graph, imagination that writes nothing, and the pen test."""

import json
from pathlib import Path

import numpy as np

from artist import env as E
from artist.brain import (
    ArtistLife,
    agent_config,
    mark_weights,
)

CONFIG = json.loads((Path(__file__).resolve().parents[1] / "config.json").read_text())


def small(**over) -> dict:
    """The stage's configuration with a small code: the contract, not the learning."""
    config = json.loads(json.dumps(CONFIG))
    config["records"].update(granules=600, active=12)
    config["graph"].update(workspace=16, dynamics=16)
    config.update(over)
    return config


def life_and_world(seed: int = 0, **over):
    config = small(**over)
    life = ArtistLife(config, seed)
    world = E.CanvasWorld(life.geometry, E.Arm(E.body_config(), seed=seed), seed=seed)
    return life, world, config


def test_the_graph_declares_the_environment_s_fields_and_the_two_action_slots():
    spec = agent_config(small()).graph
    assert [f.name for f in spec.observation] == list(E.FIELDS)
    assert [(f.name, f.width) for f in spec.observation] == [(n, w) for n, (w, _, _) in E.FIELDS.items()]
    assert [f.name for f in spec.prediction] == list(E.PREDICTED)
    assert all(f.source == f.name for f in spec.prediction)
    assert spec.action_fields == E.ACTION_FIELDS and spec.actions == 11
    assert spec.goal == 4


def test_the_committed_intention_enters_the_goal_port_and_readback_removal_flags_the_canvas():
    life, world, _ = life_and_world()
    m = world.reset(E.drawing("development", 0, 0, world.config))
    life.begin(world.view())
    dressed = life.dress(m)
    np.testing.assert_allclose(np.asarray(dressed.goal), life.intention.encode(max(life.search.distances), life.intention_gain))
    assert all(flag.all() for flag in dressed.observed.values())
    life.readback = False
    removed = life.dress(m)
    assert not removed.observed["canvas_local"].any() and not removed.observed["mark"].any()
    assert removed.observed["target_local"].all() and removed.observed["hand"].all()


def test_a_decision_leaves_the_canvas_and_the_agent_s_memory_untouched():
    life, world, _ = life_and_world(seed=2)
    m = world.reset(E.drawing("development", 3, 0, world.config))
    life.begin(world.view())
    for k in range(3):
        canvas = world.canvas
        before = life.agent.state_hash()
        decision = life.step(m, world.view())
        assert np.array_equal(canvas, world.canvas), "the brain wrote to the canvas"
        assert life.agent.state_hash() != before or k == 0  # the decision is committed, nothing else
        imagined = life.agent.state_hash()
        life.agent.predict_batch([{k: np.asarray(v) for k, v in m.observation.items()}], [0], goal=np.asarray(life.intention.encode(12.0, 0.3)), valued=False)
        assert life.agent.state_hash() == imagined, "imagination changed the agent"
        assert np.array_equal(canvas, world.canvas)
        m = life.feedback(world, decision)


def test_every_choice_is_legal_and_the_action_split_matches_the_environment():
    life, world, _ = life_and_world(seed=1)
    m = world.reset(E.drawing("development", 5, 0, world.config))
    life.begin(world.view())
    assert m.action_mask.shape == (E.ACTIONS,) and m.action_mask.all()
    for k in range(4):
        decision = life.step(m, world.view())
        torque, pen = E.split_action(decision.action)
        assert life.agent._split_action(decision.action) == [torque, pen]
        m = life.feedback(world, decision)
        assert m.executed == decision.action


class InkModel:
    """A supplied model: the body barely moves and the pen inks the cell under it when down."""

    def predict_batch(self, observations, actions, *, goal=None, valued=True):
        out = []
        for action in actions:
            window = np.zeros((E.WINDOW, E.WINDOW))
            if action % 2:
                window[E.WINDOW // 2, E.WINDOW // 2] = 1.0
            out.append({"dd_hand": np.zeros(2), "d_velocity": np.zeros(2), "d_angles": np.zeros(2), "mark": window.ravel()})
        return out


def test_the_pen_goes_down_on_the_target_and_stays_up_away_from_it():
    life, world, _ = life_and_world(seed=4)
    target = E.drawing("development", 9, 0, world.config)
    world.reset(target)
    life.begin(world.view())
    points = np.argwhere(target.target)
    on_target = points[0].astype(float)
    sheet = E.Sheet(target.target, cap=world.config.cap)
    observation = {k: np.asarray(v, float) for k, v in world.observation().items()}
    intention = life.intention
    action, _, _ = life.planner.plan(InkModel(), observation, sheet, on_target, intention, world.config, world.canvas, world.target, np.zeros(4))
    assert action % 2 == 1, "ink on an uncovered target pixel is worth taking"
    far = np.array([float(world.config.size - 1), float(world.config.size - 1)])
    while target.target[int(far[0]) - 3 : int(far[0]) + 1, int(far[1]) - 3 : int(far[1]) + 1].any():
        far -= 4
    action, _, _ = life.planner.plan(InkModel(), observation, sheet, far, intention, world.config, world.canvas, world.target, np.zeros(4))
    assert action % 2 == 0, "ink far from the target only costs"


def test_a_predicted_window_is_read_as_expected_ink_at_the_pen_s_place():
    window = np.zeros((E.WINDOW, E.WINDOW))
    window[E.WINDOW // 2, E.WINDOW // 2 + 1] = 0.4
    window[E.WINDOW // 2, E.WINDOW // 2] = 0.2
    points, weights = mark_weights(window.ravel(), E.window_origin(np.array([10.0, 12.0]), E.WINDOW), 0.1, 32)
    assert sorted(points.tolist()) == [[10, 12], [10, 13]]
    assert sorted(weights.tolist()) == [0.2, 0.4]
    empty, none = mark_weights(np.zeros(E.WINDOW**2), (0, 0), 0.1, 32)
    assert len(empty) == 0 and len(none) == 0


def test_an_evaluation_copy_is_isolated_from_the_life():
    life, world, _ = life_and_world(seed=6)
    m = world.reset(E.drawing("development", 11, 0, world.config))
    life.begin(world.view())
    for _ in range(2):
        m = life.feedback(world, life.step(m, world.view()))
    parent = life.agent.state_hash()
    copy = life.replica(frozen=False)
    other = E.CanvasWorld(life.geometry, E.Arm(E.body_config(), seed=7), seed=7)
    m2 = other.reset(E.drawing("heldout", 1, 0, other.config))
    copy.begin(other.view())
    for _ in range(3):
        m2 = copy.feedback(other, copy.step(m2, other.view()))
    assert life.agent.state_hash() == parent
    assert copy.agent.state_hash() != parent
    frozen = life.replica(frozen=True)
    assert frozen.agent.config.learning is False and life.agent.config.learning is True
