"""The arm's brain: the S00 agent with a learned forward model and a supplied planner.

The graph is the S00 experience agent with the arm's sensors, the target marker as the
goal and nine motor neurons. The world head predicts the next decision's measured hand
displacement, velocity change and angle change. Decisions come from a supplied bounded
search over the learned model (``ModelPlanner``): every candidate torque pair is imagined
through ``Agent.predict`` (read-only), the imagined next observation is composed from the
predicted deltas, and the sequence ending nearest the target wins. The learned component
is the world model; the search is supplied computation and is reported as such.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Protocol

import numpy as np

from agent.brain import ActorConfig, Agent, AgentConfig, Field, GraphSpec, RecordsConfig, WorldConfig
from agent.life import Moment
from .env import FIELDS, TORQUES

PREDICTED = ("dd_hand", "d_velocity", "d_angles")


def graph_spec(config: dict[str, Any]) -> GraphSpec:
    observation = tuple(Field(name, "continuous", width, lo, hi) for name, (width, lo, hi) in FIELDS.items())
    prediction = tuple(Field(name, "continuous", FIELDS[name][0], FIELDS[name][1], FIELDS[name][2], source=name) for name in PREDICTED)
    g = config["graph"]
    return GraphSpec(
        observation=observation, prediction=prediction, action_fields=(len(TORQUES),), goal=2,
        perceptual=g.get("perceptual", 0), workspace=g["workspace"], dynamics=g["dynamics"], episodic=g.get("episodic", 0),
        missing_flags=g.get("missing_flags", True), density=g.get("density", 1.0), scale=g.get("scale", 1.0),
        source_scale=g.get("source_scale", 2.0), sensory_to_dynamics=g.get("sensory_to_dynamics", True), bias=g.get("bias", 0.25),
        input_gain=g.get("input_gain", 2.0), field_gains=g.get("field_gains", {}), action_gain=g.get("action_gain"), readout_init=g.get("readout_init", 0.0), dt=g.get("dt", 0.5), lateral=g.get("lateral", 0.0),
    )


def agent_config(config: dict[str, Any], **over: Any) -> AgentConfig:
    cfg = AgentConfig(
        graph=graph_spec(config), world=WorldConfig(**config["world"]), actor=ActorConfig(**config["actor"]),
        records=None if config.get("records") is None else RecordsConfig(**config["records"]),
        context_decay=config.get("context_decay", 0.8), context_amplitude=config.get("context_amplitude", 1.0), controller="planner",
    )
    return replace(cfg, **over) if over else cfg


class Model(Protocol):
    def predict_batch(self, observations: list[dict[str, np.ndarray]], actions: list[int], *, goal: np.ndarray | None = None, valued: bool = True) -> list[dict[str, np.ndarray]]: ...


def compose(observation: dict[str, np.ndarray], deltas: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The imagined next observation from predicted deltas: kinematic bookkeeping only."""
    angles = observation["angles"]
    theta = np.array([np.arctan2(angles[0], angles[1]), np.arctan2(angles[2], angles[3])]) + deltas["d_angles"]
    velocity = np.clip(observation["velocity"] + deltas["d_velocity"] / 2.0, -1.0, 1.0)
    d_hand = observation["d_hand"] + deltas["dd_hand"]
    return {
        "angles": np.array([np.sin(theta[0]), np.cos(theta[0]), np.sin(theta[1]), np.cos(theta[1])]),
        "velocity": velocity,
        "hand": observation["hand"] + d_hand,
        "d_hand": d_hand,
        "dd_hand": deltas["dd_hand"],
        "d_velocity": deltas["d_velocity"],
        "d_angles": deltas["d_angles"],
        "flags": np.zeros(2),
    }


@dataclass
class ModelPlanner:
    """Bounded beam search through a model's imagined consequences; returns the first action.

    Every level imagines all nine torques for each beam entry in one batched prediction.
    A leaf is scored by the distance to the target after the hand coasts ``coast`` more
    decisions at its imagined displacement, so a torque's effect on the velocity counts
    beyond the one decision it acts in. The model supplies every consequence; the search
    is supplied computation."""

    depth: int = 1
    beam: int = 4
    coast: int = 3
    rollout: int = 1  # decisions each candidate torque is held for while imagining (constant-action rollouts)
    objective: str = "velocity"  # "velocity": track a desired hand velocity toward the target; "coast": distance after coasting
    gain: float = 3.0  # desired hand speed per unit distance, in arm lengths per second
    max_speed: float = 0.6  # arm lengths per second
    decision_seconds: float = 0.1

    def leaf_score(self, nxt: dict[str, np.ndarray], target: np.ndarray) -> float:
        if self.objective == "coast":
            coasted = nxt["hand"] + self.coast * nxt["d_hand"]
            return -float(np.linalg.norm(coasted - target)) - 0.5 * float(np.linalg.norm(nxt["hand"] - target))
        error = target - nxt["hand"]
        distance = float(np.linalg.norm(error))
        desired = error * min(self.gain, self.max_speed / max(distance, 1e-9))  # a clipped velocity field
        return -float(np.linalg.norm(nxt["d_hand"] / self.decision_seconds - desired))

    def plan(self, model: Model, observation: dict[str, np.ndarray], goal: np.ndarray) -> tuple[int, dict[str, np.ndarray], dict[str, int]]:
        target = goal * 2.0 - 1.0
        expansions = 0
        beam: list[tuple[int | None, dict[str, np.ndarray], dict[str, np.ndarray] | None]] = [(None, observation, None)]
        for level in range(self.depth):
            observations = [entry[1] for entry in beam for _ in range(len(TORQUES))]
            actions = [a for _ in beam for a in range(len(TORQUES))]
            predictions = model.predict_batch(observations, actions, goal=goal, valued=False)  # the consequences alone
            expansions += len(actions)
            composed = [compose(obs, pred) for obs, pred in zip(observations, predictions, strict=True)]
            first_predictions = list(predictions)
            for _ in range(self.rollout - 1):  # hold each candidate torque: its effect accumulates
                more = model.predict_batch(composed, actions, goal=goal, valued=False)
                expansions += len(actions)
                composed = [compose(obs, pred) for obs, pred in zip(composed, more, strict=True)]
            candidates = []
            for (first, obs, first_pred), action, nxt, pred in zip([entry for entry in beam for _ in range(len(TORQUES))], actions, composed, first_predictions, strict=True):
                score = self.leaf_score(nxt, target)
                candidates.append((score, action if first is None else first, nxt, pred if first_pred is None else first_pred))
            candidates.sort(key=lambda c: -c[0])
            beam = [(c[1], c[2], c[3]) for c in candidates[: self.beam]]
        first, _, first_pred = beam[0]
        assert first is not None and first_pred is not None
        return int(first), first_pred, {"expansions": expansions, "depth": self.depth, "beam": self.beam}

    def for_agent(self):
        """The callable an S00 ``Agent(controller="planner")`` expects."""
        def planner(agent: Agent, row: int, moment: Moment, drive: np.ndarray, free: Any):
            observation = {k: np.asarray(v) for k, v in moment.observation.items()}
            assert moment.goal is not None
            return self.plan(agent, observation, np.asarray(moment.goal))
        return planner


# -- baselines: outside the candidate's imports, sharing the same planner and moments


class RandomController:
    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.decisions = 0

    def step(self, moment: Moment) -> int | None:
        if moment.terminated or moment.truncated:
            return None
        self.decisions += 1
        return int(self.rng.integers(len(TORQUES)))


class JacobianPD:
    """The supplied analytic controller: Jacobian-transpose PD, discretized to the torque grid."""

    def __init__(self, lengths: tuple[float, float] = (0.5, 0.5), kp: float = 6.0, kd: float = 1.5, deadband: float = 0.15) -> None:
        self.lengths, self.kp, self.kd, self.deadband = lengths, kp, kd, deadband

    def step(self, moment: Moment) -> int | None:
        if moment.terminated or moment.truncated:
            return None
        from .env import jacobian

        o = moment.observation
        theta = np.array([np.arctan2(o["angles"][0], o["angles"][1]), np.arctan2(o["angles"][2], o["angles"][3])])
        target = np.asarray(moment.goal) * 2.0 - 1.0
        force = self.kp * (target - o["hand"])
        torque = jacobian(theta, self.lengths).T @ force - self.kd * o["velocity"] * 2.0
        choice = np.where(torque > self.deadband, 1.0, np.where(torque < -self.deadband, -1.0, 0.0))
        return int(np.flatnonzero((TORQUES == choice).all(axis=1))[0])


class OnlineMLP:
    """A conventional forward model: one hidden layer, trained online by backpropagation with
    Adam on the same (observation, action) -> deltas stream, with the same planner."""

    def __init__(self, hidden: int = 64, lr: float = 1e-3, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self.inputs = sum(w for w, _, _ in FIELDS.values()) + len(TORQUES)
        self.outputs = sum(FIELDS[n][0] for n in PREDICTED)
        self.hidden = hidden
        self.lr = lr
        self.w1 = self.rng.normal(0, np.sqrt(2 / self.inputs), (self.inputs, hidden))
        self.b1 = np.zeros(hidden)
        self.w2 = np.zeros((hidden, self.outputs))
        self.b2 = np.zeros(self.outputs)
        self.params = [self.w1, self.b1, self.w2, self.b2]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0
        self.updates = 0

    def parameters(self) -> int:
        return int(sum(p.size for p in self.params))

    def _x(self, observation: dict[str, np.ndarray], action: int) -> np.ndarray:
        parts = [(np.asarray(observation[n]) - lo) / (hi - lo) for n, (_, lo, hi) in FIELDS.items()]
        onehot = np.zeros(len(TORQUES)); onehot[action] = 1.0
        return np.concatenate([*parts, onehot])

    def _y(self, deltas: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([(np.asarray(deltas[n]) - FIELDS[n][1]) / (FIELDS[n][2] - FIELDS[n][1]) for n in PREDICTED])

    def _decode(self, y: np.ndarray) -> dict[str, np.ndarray]:
        out, at = {}, 0
        for n in PREDICTED:
            w, lo, hi = FIELDS[n]
            out[n] = lo + np.clip(y[at:at + w], 0.0, 1.0) * (hi - lo)
            at += w
        return out

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = np.tanh(x @ self.w1 + self.b1)
        return h, h @ self.w2 + self.b2

    def predict(self, observation: dict[str, np.ndarray], action: int, *, goal: np.ndarray | None = None) -> dict[str, np.ndarray]:
        _, y = self.forward(self._x(observation, action))
        return self._decode(y)

    def predict_batch(self, observations: list[dict[str, np.ndarray]], actions: list[int], *, goal: np.ndarray | None = None, valued: bool = True) -> list[dict[str, np.ndarray]]:
        return [self.predict(o, a) for o, a in zip(observations, actions, strict=True)]

    def learn(self, observation: dict[str, np.ndarray], action: int, deltas: dict[str, np.ndarray]) -> float:
        x, y = self._x(observation, action), self._y(deltas)
        h, out = self.forward(x)
        err = out - y
        loss = float(np.mean(err**2))
        g2 = np.outer(h, err) * 2 / len(y)
        gb2 = err * 2 / len(y)
        dh = (err @ self.w2.T) * (1 - h**2) * 2 / len(y)
        g1 = np.outer(x, dh)
        gb1 = dh
        self.t += 1
        for p, g, m, v in zip(self.params, (g1, gb1, g2, gb2), self.m, self.v, strict=True):
            m *= 0.9; m += 0.1 * g
            v *= 0.999; v += 0.001 * g * g
            mhat = m / (1 - 0.9**self.t)
            vhat = v / (1 - 0.999**self.t)
            p -= self.lr * mhat / (np.sqrt(vhat) + 1e-8)
        self.updates += 1
        return loss


class ModelController:
    """A baseline life: an online model plus the shared planner, with the same step contract
    and the same bounded replay budget (a ring of raw transitions, ``replay`` per real one)."""

    def __init__(self, model: OnlineMLP, planner: ModelPlanner, seed: int, epsilon: float = 0.1, learning: bool = True, replay: int = 1, capacity: int = 4096) -> None:
        self.model, self.planner, self.epsilon, self.learning = model, planner, epsilon, learning
        self.rng = np.random.default_rng(seed)
        self.pending: tuple[dict[str, np.ndarray], int] | None = None
        self.last_prediction: dict[str, np.ndarray] | None = None
        self.decisions = 0
        self.losses: list[float] = []
        self.replay, self.capacity = replay, capacity
        self.ring: list[tuple[dict[str, np.ndarray], int, dict[str, np.ndarray]]] = []
        self.ring_at = 0
        self.replay_writes = 0

    def score(self, prediction: dict[str, np.ndarray], moment: Moment) -> dict[str, float]:
        out = {}
        for n in PREDICTED:
            _, lo, hi = FIELDS[n]
            err = (np.asarray(prediction[n]) - np.asarray(moment.observation[n])) / (hi - lo) * 0.75
            out[f"{n}/mse"] = float(np.mean(err**2))
        return out

    def step(self, moment: Moment) -> int | None:
        if moment.has_feedback and self.pending is not None and self.learning:
            obs, action = self.pending
            outcome = {n: np.asarray(moment.observation[n]) for n in PREDICTED}
            self.losses.append(self.model.learn(obs, action, outcome))
            if len(self.ring) < self.capacity:
                self.ring.append((obs, action, outcome))
            else:
                self.ring[self.ring_at] = (obs, action, outcome)
                self.ring_at = (self.ring_at + 1) % self.capacity
            stored = len(self.ring) - 1
            for _ in range(self.replay if stored > 0 else 0):
                o, a, t = self.ring[int(self.rng.integers(stored))]
                self.model.learn(o, a, t)
                self.replay_writes += 1
        self.pending = None
        if moment.terminated or moment.truncated:
            return None
        observation = {k: np.asarray(v) for k, v in moment.observation.items()}
        if self.rng.random() < self.epsilon:
            action = int(self.rng.integers(len(TORQUES)))
            self.last_prediction = self.model.predict(observation, action)
        else:
            action, self.last_prediction, _ = self.planner.plan(self.model, observation, np.asarray(moment.goal))
        self.pending = (observation, action)
        self.decisions += 1
        return action
