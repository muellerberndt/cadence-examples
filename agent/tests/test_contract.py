"""S00 contract tests: causal feedback, credit, masks, heads, memory, imagination, checkpoints."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest
from cadence.brain import Nudge

from agent.brain import (
    ActorConfig,
    Agent,
    AgentConfig,
    Field,
    GraphSpec,
    Mulberry32,
    WorldConfig,
)
from agent.tests.worlds import DelayedCue, RingWorld, one_hot
from agent.life import Decision, Moment, seed_for

HERE = Path(__file__).resolve().parents[1]
CONFIG = json.loads((Path(__file__).resolve().parent / "config.json").read_text())


def config(**over) -> AgentConfig:
    base = copy.deepcopy(CONFIG["agent"])
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key].update(value)
        else:
            base[key] = value
    return AgentConfig.from_dict(base)


def ring_config(**over) -> AgentConfig:
    graph = {
        "observation": [{"name": "room", "kind": "categorical", "width": 4}],
        "prediction": [{"name": "next_room", "kind": "categorical", "width": 4, "source": "room"}],
        "action_fields": [2],
    }
    over.setdefault("graph", {})
    over["graph"] = {**graph, **over["graph"]}
    return config(**over)


def episode(agent: Agent, env) -> list[Decision]:
    """Run one episode through the agent; the environment feeds every outcome back once."""
    m = env.reset()
    out = []
    while True:
        d = agent.step(m)
        if d is None:
            return out
        out.append(d)
        m = env.act(d.decision_id, d.action)


def train_cue(seed: int, episodes: int, *, delay: int = 8, shuffle: bool = False, **over) -> tuple[Agent, list[float]]:
    agent = Agent(config(**over), seed=seed)
    env = DelayedCue(delay=delay, seed=seed_for("S00", "development", seed, 0, 0), shuffle=shuffle)
    hits = []
    for _ in range(episodes):
        decisions = episode(agent, env)
        hits.append(float(decisions[0].action == env.correct_first))
    return agent, hits


def evaluate_cue(agent: Agent, seed: int, episodes: int, *, delay: int = 8, shuffle: bool = False) -> float:
    """Competence on a held-out set through an isolated frozen copy; the live agent is untouched."""
    frozen = agent.frozen()
    env = DelayedCue(delay=delay, seed=seed_for("S00", "heldout", seed, 2000, 0), shuffle=shuffle)
    hits = [float(episode(frozen, env)[0].action == env.correct_first) for _ in range(episodes)]
    return float(np.mean(hits))


# -- 1. causality


def test_causality_future_outcomes_cannot_change_earlier_decisions_or_drives():
    a, b = Agent(config(), seed=3), Agent(config(), seed=3)
    env_a, env_b = DelayedCue(delay=3, seed=7), DelayedCue(delay=3, seed=7)
    trail_a, trail_b = [], []
    for k in range(6):
        ma, mb = env_a.reset(), env_b.reset()
        while True:
            da, db = a.step(ma), b.step(mb)
            if da is None:
                break
            trail_a.append((da.action, a.pending[0].drive.copy(), da.prediction["next_cue"].copy()))
            trail_b.append((db.action, b.pending[0].drive.copy(), db.prediction["next_cue"].copy()))
            ma, mb = env_a.act(da.decision_id, da.action), env_b.act(db.decision_id, db.action)
            if k == 4 and mb.terminated:
                # a different future outcome for b only
                mb = Moment(**{**mb.__dict__, "reward": 1.0 - mb.reward})
    split = sum(1 for _ in range(4) for _ in range(4))  # decisions of the first four episodes
    for (act_a, drive_a, pred_a), (act_b, drive_b, pred_b) in zip(trail_a[:split], trail_b[:split], strict=True):
        assert act_a == act_b
        np.testing.assert_array_equal(drive_a, drive_b)
        np.testing.assert_array_equal(pred_a, pred_b)
    assert a.state_hash() != b.state_hash()


# -- 2. exactly-once feedback


def test_feedback_is_processed_exactly_once_and_unknown_reward_teaches_no_policy():
    agent = Agent(config(), seed=0)
    env = DelayedCue(delay=2, seed=1)
    m = env.reset()
    d = agent.step(m)
    outcome = env.act(d.decision_id, d.action)
    motor_before = agent.brain.efficacy[agent.motor_edges].copy()
    deltas = agent.ledger.deltas
    unknown = Moment(**{**outcome.__dict__, "reward_known": False})
    agent.step(unknown)
    np.testing.assert_array_equal(agent.brain.efficacy[agent.motor_edges], motor_before)
    assert agent.ledger.deltas == deltas  # no reward event, no credit
    with pytest.raises(ValueError, match="out-of-order|duplicate|pending|awaits"):
        agent.step(unknown)  # the same event again
    # a genuine zero reward is a processed transition
    d2 = agent.pending[0].decision
    zero = env.act(d2.decision_id, d2.action)
    assert zero.reward_known and zero.reward == 0.0
    agent.step(zero)
    assert agent.ledger.deltas == deltas + 1


# -- 3. executed action credit


def test_feedback_for_a_different_executed_action_is_rejected():
    agent = Agent(config(), seed=0)
    env = DelayedCue(delay=2, seed=1)
    d = agent.step(env.reset())
    outcome = env.act(d.decision_id, d.action)
    swapped = Moment(**{**outcome.__dict__, "executed": 1 - outcome.executed})
    before = agent.state_hash()
    with pytest.raises(ValueError, match="executed action"):
        agent.step(swapped)
    assert agent.state_hash() == before
    wrong_id = Moment(**{**outcome.__dict__, "feedback_for": outcome.feedback_for + 5})
    with pytest.raises(ValueError, match="names decision"):
        agent.step(wrong_id)


# -- 4. delayed credit (the learning gate)


@pytest.mark.slow
def test_delayed_credit_learns_the_cue_and_shuffled_association_stays_near_chance():
    spec = CONFIG["delayed_cue"]
    seed = CONFIG["seeds"]["development"][0]
    agent, curve = train_cue(seed, spec["training_episodes"], delay=spec["delay"])
    assert agent.ledger.real_transitions <= CONFIG["pilot_cap_transitions"]
    learned = evaluate_cue(agent, seed, spec["heldout_episodes"], delay=spec["delay"])
    assert learned >= spec["threshold"], (learned, np.mean(curve[-100:]))
    control, _ = train_cue(seed, spec["training_episodes"], delay=spec["delay"], shuffle=True)
    chance = evaluate_cue(control, seed, spec["heldout_episodes"], delay=spec["delay"], shuffle=True)
    assert chance <= spec["shuffled_ceiling"], chance
    no_trace, _ = train_cue(seed, 300, delay=spec["delay"], actor={"lam": 0.0})
    assert evaluate_cue(no_trace, seed, 100, delay=spec["delay"]) < learned


# -- 5. observed-value mask


def test_unknown_values_get_no_teaching_drive_and_all_missing_leaves_weights_unchanged():
    cfg = ring_config(graph={"prediction": [
        {"name": "next_room", "kind": "categorical", "width": 4, "source": "room"},
        {"name": "sensor", "kind": "continuous", "width": 2, "lo": 0.0, "hi": 1.0, "source": "sensor"},
    ], "observation": [
        {"name": "room", "kind": "categorical", "width": 4},
        {"name": "sensor", "kind": "continuous", "width": 2},
    ]})
    agent = Agent(cfg, seed=0)
    def moment(event, feedback=None, observed=None, room=0):
        return Moment(life_id="t", episode_id=0, event_id=event, tick=event,
                      observation={"room": one_hot(room, 4), "sensor": np.array([0.7, 0.2])},
                      observed={} if observed is None else observed, action_mask=np.ones(2, bool),
                      feedback_for=None if feedback is None else feedback.decision_id,
                      executed=None if feedback is None else feedback.action, reward_known=feedback is not None)
    d = agent.step(moment(0))
    # one known continuous value: the direct target drive on the unknown neuron is zero
    partly = moment(1, d, observed={"sensor": np.array([True, False]), "room": np.ones(4, bool)}, room=1)
    targets = agent._outcome_targets(partly)
    assert targets["sensor"][1].tolist() == [True, False]
    n = agent.connectome.n
    mask = np.zeros(n); neurons = agent.ports.prediction_fields["sensor"]
    mask[neurons[targets["sensor"][1]]] = 1.0
    target = np.zeros((1, n)); target[0, neurons] = targets["sensor"][0]
    force = Nudge(target, mask, 0.05).drive(np.zeros((1, n)))
    assert force[0, neurons[1]] == 0.0 and force[0, neurons[0]] != 0.0
    before = agent.brain.efficacy.copy()
    agent.step(partly)
    assert not np.array_equal(agent.brain.efficacy, before)  # the known values taught
    d2 = agent.pending[0].decision
    nothing = moment(2, d2, observed={"sensor": np.zeros(2, bool), "room": np.zeros(4, bool)}, room=2)
    world_before = agent.brain.efficacy[agent.world_edges].copy()
    agent.step(nothing)
    np.testing.assert_array_equal(agent.brain.efficacy[agent.world_edges], world_before)
    assert "next_room" not in agent.score(d2.prediction, nothing)


# -- 6. multiple heads


def test_world_update_reaches_every_head_and_preserves_frozen_motor_pairs():
    agent = Agent(ring_config(), seed=0)
    env = RingWorld(seed=2)
    d = agent.step(env.reset())
    before = agent.brain.efficacy.copy()
    version = agent.parameter_version
    outcome = env.act(d.decision_id, d.action)
    d2 = agent.step(outcome)
    assert agent.motor.brain is agent.world.brain
    assert agent.parameter_version > version
    assert d2.parameter_version == agent.parameter_version
    np.testing.assert_array_equal(agent.brain.efficacy[agent.motor_edges], before[agent.motor_edges])
    assert np.any(agent.brain.efficacy[agent.world_edges] != before[agent.world_edges])
    paired = np.flatnonzero(agent.world.reverse >= 0)
    np.testing.assert_allclose(agent.brain.efficacy[paired], agent.brain.efficacy[agent.world.reverse[paired]], atol=1e-14)
    np.testing.assert_array_equal(agent.brain.bias[agent.ports.sources], 0.0)


# -- 7. masked actor


def test_masked_actor_never_samples_illegal_actions_and_normalizes():
    spec = GraphSpec(observation=(Field("x", "categorical", 4),), prediction=(), action_fields=(6,), workspace=16, dynamics=8)
    agent = Agent(AgentConfig(graph=spec), seed=0)
    rng = np.random.default_rng(0)
    event = 0
    for k in range(10_000):
        mask = rng.random(6) < 0.5
        if not mask.any():
            mask[rng.integers(6)] = True
        m = Moment(life_id="t", episode_id=k, event_id=event, tick=0, observation={"x": one_hot(int(rng.integers(4)), 4)}, observed={}, action_mask=mask)
        event += 1
        d = agent.step(m)
        assert mask[d.action]
        assert abs(d.probabilities.sum() - 1.0) < 1e-12 and not d.probabilities[~mask].any()
        if mask.sum() == 1:
            assert not agent.pending[0].score_edges.any()
        # close the episode with a zero-reward terminal outcome so the next event is a fresh start
        agent.step(Moment(life_id="t", episode_id=k, event_id=event, tick=1, observation={"x": one_hot(0, 4)}, observed={}, action_mask=mask,
                          feedback_for=d.decision_id, executed=d.action, reward=0.0, reward_known=True, terminated=True))
        event += 1


def test_actor_score_follows_the_finite_difference_policy_gradient():
    spec = GraphSpec(observation=(Field("x", "categorical", 2),), prediction=(), action_fields=(3,), workspace=6, dynamics=4, readout_init=1.0)
    cfg = AgentConfig(
        graph=spec,
        actor=ActorConfig(beta=0.005, temperature=0.5),
        world=WorldConfig(residual_tolerance=1e-10, chunk=8, free_steps=2000, nudged_steps=2000),
    )
    agent = Agent(cfg, seed=1)
    m = Moment(life_id="t", episode_id=0, event_id=0, tick=0, observation={"x": one_hot(1, 2)}, observed={}, action_mask=np.array([True, True, False]))
    d = agent.step(m)
    score = agent.pending[0].score_edges
    action = d.action
    T = cfg.actor.temperature
    drive = agent._drive([m])

    def log_policy(brain):
        state = brain.settle_batch(drive, steps=4000, tolerance=1e-12)
        return T * float(np.log(agent._policy(state, 0, m.action_mask)[0][action]))

    pairs = np.flatnonzero(agent.motor_edges & (agent.motor.reverse > np.arange(agent.connectome.synapses)))
    fd, ep = [], []
    for e in pairs[:12]:
        partner = agent.motor.reverse[e]
        eps = 1e-4
        up, down = agent.brain.efficacy.copy(), agent.brain.efficacy.copy()
        up[[e, partner]] += eps
        down[[e, partner]] -= eps
        fd.append((log_policy(agent.brain.with_parameters(efficacy=up)) - log_policy(agent.brain.with_parameters(efficacy=down))) / (2 * eps))
        ep.append(score[e] + score[partner])  # the pair's contrast: both directions
    fd, ep = np.array(fd), np.array(ep)
    assert np.corrcoef(fd, ep)[0, 1] > 0.9, (fd, ep)
    assert (np.sign(fd) == np.sign(ep)).mean() >= 0.9


# -- 8. heterogeneous fields


def test_heterogeneous_action_fields_never_cross_slots_and_credit_the_joint_choice():
    spec = GraphSpec(observation=(Field("x", "categorical", 2),), prediction=(), action_fields=(3, 2, 5), workspace=16, dynamics=8)
    agent = Agent(AgentConfig(graph=spec), seed=0)
    assert int(np.prod(spec.action_fields)) == 30
    for joint in range(30):
        parts = agent._split_action(joint)
        assert all(0 <= p < k for p, k in zip(parts, spec.action_fields, strict=True))
        assert agent._joint_action(parts) == joint
    mask = np.zeros(30, bool)
    mask[[0, 7, 29, 13]] = True
    m = Moment(life_id="t", episode_id=0, event_id=0, tick=0, observation={"x": one_hot(0, 2)}, observed={}, action_mask=mask)
    d = agent.step(m)
    assert mask[d.action]
    joint, marginals = agent._policy(agent.warm, 0, mask)
    assert all(abs(p.sum() - 1) < 1e-12 for p in marginals)
    p_before = joint[d.action]
    # the target of the score lies inside each field's own slot
    target_slots = [agent.ports.motor_slots[k][agent._split_action(d.action)[k]] for k in range(3)]
    assert all(t in agent.ports.motor for t in target_slots)
    outcome = Moment(life_id="t", episode_id=0, event_id=1, tick=1, observation={"x": one_hot(0, 2)}, observed={}, action_mask=mask,
                     feedback_for=d.decision_id, executed=d.action, reward=1.0, reward_known=True)
    agent.step(outcome)
    frozen = agent.clone()
    state = frozen.brain.settle_batch(frozen._drive([m]), steps=400, tolerance=1e-8)
    joint_after, _ = frozen._policy(state, 0, mask)
    assert joint_after[d.action] > p_before


# -- 9. episode and lifetime


def test_episode_reset_clears_transient_state_but_keeps_weights_and_a_new_life_erases_all():
    cfg = ring_config(graph={"episodic": 8}, episodic={"key": "observation", "decay": 1.0, "rate": 1.0, "amplitude": 1.0, "reset_on_episode": False})
    agent = Agent(cfg, seed=0)
    agent.recall_value = lambda m: np.pad(m.observation["room"], (0, 4))
    env = RingWorld(seed=3, episode_length=6)
    episode(agent, env)
    trained = agent.brain.efficacy.copy()
    assert agent.episodic.writes > 0 and np.abs(agent.context.trace).sum() == 0.0  # reset at the terminal
    assert agent.pending[0] is None and not agent.elig.any()
    episode(agent, env)
    assert agent.ledger.real_transitions == 12
    fresh = Agent(cfg, seed=0)
    assert not np.array_equal(fresh.brain.efficacy, agent.brain.efficacy)
    assert fresh.episodic.writes == 0 and not fresh.episodic.strength.any()
    # a reset moment with a pending decision is refused: an outcome must close the old episode first
    agent.step(env.reset())
    with pytest.raises(ValueError, match="awaits feedback"):
        agent.step(Moment(life_id="ring", episode_id=env._episode + 1, event_id=env._event + 1, tick=0, observation={"room": one_hot(0, 4)}, observed={}, action_mask=np.ones(2, bool)))
    del trained


# -- 10. row identity (separate lives)


def test_separate_lives_share_no_fast_memory():
    cfg = ring_config(graph={"episodic": 4}, episodic={"key": "observation"})
    a, b = Agent(cfg, seed=0), Agent(cfg, seed=0)
    for agent in (a, b):
        agent.recall_value = lambda m: m.observation["room"]
    env = RingWorld(seed=4, episode_length=5)
    episode(a, env)
    assert a.episodic.writes > 0
    assert b.episodic.writes == 0 and not b.episodic.strength.any()
    key = np.zeros((1, len(a.episodic.pre)))
    key[0, 0] = 1.0
    assert not b.episodic.recall(key).any()


# -- 11. read-only imagination


def test_imagination_writes_nothing():
    agent = Agent(ring_config(), seed=0)
    env = RingWorld(seed=5)
    episode(agent, env)
    d = agent.step(env.reset())  # a pending decision exists
    before = agent.state_hash()
    rng = np.random.default_rng(0)
    for _ in range(1000):
        agent.predict({"room": one_hot(int(rng.integers(4)), 4)}, int(rng.integers(2)))
    assert agent.state_hash() == before
    assert agent.pending[0].decision.decision_id == d.decision_id
    assert agent.ledger.imagined == 1000


# -- 12. checkpoint


def test_resume_between_decision_and_feedback_matches_the_uninterrupted_life(tmp_path):
    a = Agent(config(), seed=0)
    env = DelayedCue(delay=4, seed=9)
    for _ in range(3):
        episode(a, env)
    m = env.reset()
    d = a.step(m)  # decided, not yet fed back
    path = a.save(tmp_path / "life.npz")
    b = Agent.load(path)
    assert b.state_hash() == a.state_hash()
    env_b = copy.deepcopy(env)
    ma, mb = env.act(d.decision_id, d.action), env_b.act(d.decision_id, d.action)
    for _ in range(100):
        da, db = a.step(ma), b.step(mb)
        assert (da is None) == (db is None)
        if da is None:
            ma, mb = env.reset(), env_b.reset()
            continue
        assert da.action == db.action and da.decision_id == db.decision_id
        ma, mb = env.act(da.decision_id, da.action), env_b.act(db.decision_id, db.action)
    assert a.state_hash() == b.state_hash()
    assert a.ledger.real_transitions == b.ledger.real_transitions


# -- 13. leakage


def test_agent_code_cannot_reach_environment_internals():
    source = (HERE / "brain.py").read_text()
    assert not re.search(r"^\s*(from|import)\s+.*\benv\b", source, re.MULTILINE)
    for name in ("next_room", "correct_first", "_cue", "_rule", "_room"):
        assert name not in source
    env = RingWorld(seed=0)
    m = env.reset()
    assert set(m.__dataclass_fields__) == {
        "life_id", "episode_id", "event_id", "tick", "observation", "observed", "action_mask", "dt", "feedback_for",
        "executed", "reward", "reward_known", "terminated", "truncated", "final_observation", "goal", "demonstration", "replay_of",
    }
    with pytest.raises(ValueError, match="replayed"):
        Agent(ring_config(), seed=0).step(Moment(**{**m.__dict__, "replay_of": 0}))


def test_portable_generator_matches_the_page_renderer_recipe():
    """mulberry32 as in brain_scan.js: the first draws of seed 1 and seed 12345."""
    r = Mulberry32(1)
    draws = [r.random() for _ in range(3)]
    assert all(0 <= x < 1 for x in draws) and len(set(draws)) == 3
    r2 = Mulberry32(1)
    assert [r2.random() for _ in range(3)] == draws
    digest = hashlib.sha256(json.dumps(draws).encode()).hexdigest()
    assert len(digest) == 64
