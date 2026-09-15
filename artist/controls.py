"""Baselines and references, outside the candidate's imports: random, oracle, online MLP.

The random scribbler calibrates the metric's floor. The oracle is the upper reference the packet
asks for: it is handed the target's polylines and drives the pen along them with the supplied
analytic controller, so it measures what this body can draw in this budget with no learning
problem left. The online MLP is the conventional world model: the same reading, the same
predicted fields, the same two-level search, trained online by backpropagation with Adam, under
the same replay policy as the candidate. None of these is imported by ``brain.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .brain import CanvasPlanner, IntentionSearch, empty_intention
from .env import (
    ACTIONS,
    FIELDS,
    PREDICTED,
    CanvasConfig,
    Sheet,
    hand_of,
    join_action,
    pixel_of,
)
from .paths import ensure

ensure()

from agent.life import Moment
from arm.env import TORQUES, jacobian


@dataclass
class Choice:
    """What a baseline commits: the same fields of a decision the runner and the log read."""

    decision_id: int
    action: int
    controller: str = "supplied"
    prediction: dict[str, np.ndarray] = field(default_factory=dict)
    budget: dict[str, int] = field(default_factory=dict)
    intention: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"decision_id": self.decision_id, "action": int(self.action), "controller": self.controller,
                "prediction": {k: np.asarray(v).tolist() for k, v in self.prediction.items()}, "budget": dict(self.budget)}


class Baseline:
    """The controller interface the runner uses for every life."""

    planning = True

    def begin(self, view: dict[str, Any]) -> None:
        return None

    def detach(self) -> None:
        return None

    def set_epsilon(self, epsilon: float) -> None:
        return None

    def feedback(self, world, decision: Choice) -> Moment:
        return world.act(decision.decision_id, decision.action)


class RandomScribbler(Baseline):
    """Uniform over the eighteen choices: the metric's floor."""

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.next_id = 10**7

    def step(self, moment: Moment, view: dict[str, Any] | None = None) -> Choice | None:
        if moment.terminated or moment.truncated:
            return None
        self.next_id += 1
        return Choice(self.next_id, int(self.rng.integers(ACTIONS)), controller="random")


def pd_torque(observation: dict[str, np.ndarray], target: np.ndarray, lengths: tuple[float, float], kp: float, kd: float, deadband: float) -> int:
    """The supplied analytic controller: Jacobian-transpose PD, discretised to the torque grid."""
    angles = np.asarray(observation["angles"], float)
    theta = np.array([np.arctan2(angles[0], angles[1]), np.arctan2(angles[2], angles[3])])
    force = kp * (np.asarray(target, float) - np.asarray(observation["hand"], float))
    torque = jacobian(theta, lengths).T @ force - kd * np.asarray(observation["velocity"], float) * 2.0
    choice = np.where(torque > deadband, 1.0, np.where(torque < -deadband, -1.0, 0.0))
    return int(np.flatnonzero((TORQUES == choice).all(axis=1))[0])


class OracleArtist(Baseline):
    """The supplied upper reference: the target's own path, walked with the analytic controller."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.geometry = CanvasConfig(**config["canvas"])
        b = config["baselines"]
        self.kp, self.kd, self.deadband, self.radius = b["oracle_kp"], b["oracle_kd"], b["oracle_deadband"], b["oracle_radius"]
        self.dwell = int(b.get("oracle_dwell", 8))  # decisions before the next waypoint is taken anyway
        self.lengths = tuple(config["body"]["lengths"])
        self.waypoints: list[tuple[np.ndarray, int]] = []
        self.at = 0
        self.held = 0
        self.next_id = 2 * 10**7

    def plan_path(self, drawing) -> None:
        """Sample the target's polylines about one pixel apart; the pen lifts between paths."""
        self.waypoints, self.at, self.held = [], 0, 0
        for path in drawing.paths:
            for k, (a, b) in enumerate(zip(path[:-1], path[1:], strict=True)):
                steps = max(1, int(round(float(np.linalg.norm(b - a)))))
                for s in range(steps + 1):
                    point = a + (b - a) * (s / steps)
                    self.waypoints.append((point, 0 if (k == 0 and s == 0) else 1))

    def step(self, moment: Moment, view: dict[str, Any] | None = None) -> Choice | None:
        if moment.terminated or moment.truncated or self.at >= len(self.waypoints):
            return None
        point, pen = self.waypoints[self.at]
        hand = np.asarray(moment.observation["hand"], float)
        self.held += 1
        if float(np.linalg.norm(pixel_of(hand, self.geometry) - point)) <= self.radius or self.held > self.dwell:
            self.at += 1
            self.held = 0
            if self.at >= len(self.waypoints):
                return None
            point, pen = self.waypoints[self.at]
        target = hand_of(point, self.geometry)
        self.next_id += 1
        return Choice(self.next_id, join_action(pd_torque(moment.observation, target, self.lengths, self.kp, self.kd, self.deadband), pen), controller="oracle")


class OnlineCanvasMLP:
    """One hidden layer over the same reading, trained online by backpropagation with Adam."""

    def __init__(self, hidden: int = 128, lr: float = 1e-3, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self.inputs = sum(width for width, _, _ in FIELDS.values()) + sum((9, 2))
        self.outputs = sum(FIELDS[name][0] for name in PREDICTED)
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
        parts = [(np.asarray(observation[name], float) - lo) / (hi - lo) for name, (_, lo, hi) in FIELDS.items()]
        torque, pen = divmod(int(action), 2)
        slots = np.zeros(11)
        slots[torque] = 1.0
        slots[9 + pen] = 1.0
        return np.concatenate([*parts, slots])

    def _y(self, deltas: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([(np.asarray(deltas[n], float) - FIELDS[n][1]) / (FIELDS[n][2] - FIELDS[n][1]) for n in PREDICTED])

    def _decode(self, y: np.ndarray) -> dict[str, np.ndarray]:
        out, at = {}, 0
        for name in PREDICTED:
            width, lo, hi = FIELDS[name]
            out[name] = lo + np.clip(y[at : at + width], 0.0, 1.0) * (hi - lo)
            at += width
        return out

    def predict_batch(self, observations, actions, *, goal=None, valued: bool = True) -> list[dict[str, np.ndarray]]:
        x = np.stack([self._x(o, a) for o, a in zip(observations, actions, strict=True)])
        y = np.tanh(x @ self.w1 + self.b1) @ self.w2 + self.b2
        return [self._decode(row) for row in y]

    def learn(self, observation: dict[str, np.ndarray], action: int, deltas: dict[str, np.ndarray]) -> float:
        x, y = self._x(observation, action), self._y(deltas)
        h = np.tanh(x @ self.w1 + self.b1)
        error = h @ self.w2 + self.b2 - y
        loss = float(np.mean(error**2))
        scale = 2 / len(y)
        gradients = (np.outer(x, (error @ self.w2.T) * (1 - h**2) * scale), (error @ self.w2.T) * (1 - h**2) * scale, np.outer(h, error) * scale, error * scale)
        self.t += 1
        for p, g, m, v in zip(self.params, gradients, self.m, self.v, strict=True):
            m *= 0.9
            m += 0.1 * g
            v *= 0.999
            v += 0.001 * g * g
            p -= self.lr * (m / (1 - 0.9**self.t)) / (np.sqrt(v / (1 - 0.999**self.t)) + 1e-8)
        self.updates += 1
        return loss


class MLPArtist(Baseline):
    """The conventional world-model controller: the MLP under the candidate's two-level search."""

    def __init__(self, config: dict[str, Any], seed: int, *, replay: int = 0, capacity: int = 4096) -> None:
        self.geometry = CanvasConfig(**config["canvas"])
        self.search = IntentionSearch(**config["intention"])
        self.planner = CanvasPlanner(**config["planner"])
        self.replan_every = int(config["intention_interval"])
        self.intention_gain = float(config["intention_gain"])
        self.model = OnlineCanvasMLP(hidden=config["baselines"]["mlp_hidden"], lr=config["baselines"]["mlp_lr"], seed=seed)
        self.rng = np.random.default_rng(seed * 31 + 7)
        self.epsilon = 0.0
        self.planning = True
        self.replay, self.capacity = replay, capacity
        self.ring: list[tuple[dict[str, np.ndarray], int, dict[str, np.ndarray]]] = []
        self.ring_at = 0
        self.replay_writes = 0
        self.pending: tuple[dict[str, np.ndarray], int] | None = None
        self.sheet: Sheet | None = None
        self.belief = np.zeros((self.geometry.size, self.geometry.size), bool)
        self.target = np.zeros_like(self.belief)
        self.pen_pixel = np.zeros(2)
        self.intention = empty_intention()
        self.countdown = 0
        self.next_id = 3 * 10**7
        self.losses: list[float] = []

    def set_epsilon(self, epsilon: float) -> None:
        self.epsilon = float(epsilon)

    def begin(self, view: dict[str, Any]) -> None:
        self.target = np.asarray(view["target"], bool).copy()
        self.belief = np.asarray(view["canvas"], bool).copy()
        self.sheet = Sheet(self.target, cap=self.geometry.cap)
        self.sheet.observe(self.belief)
        self.pen_pixel = np.asarray(view["pen"], float)
        self.intention = empty_intention()
        self.countdown = 0

    def detach(self) -> None:
        self.pending = None

    def step(self, moment: Moment, view: dict[str, Any] | None = None) -> Choice | None:
        if moment.has_feedback and self.pending is not None:
            observation, action = self.pending
            outcome = {name: np.asarray(moment.observation[name], float) for name in PREDICTED}
            self.losses.append(self.model.learn(observation, action, outcome))
            if len(self.ring) < self.capacity:
                self.ring.append((observation, action, outcome))
            else:
                self.ring[self.ring_at] = (observation, action, outcome)
                self.ring_at = (self.ring_at + 1) % self.capacity
            for _ in range(self.replay if len(self.ring) > 1 else 0):
                o, a, t = self.ring[int(self.rng.integers(len(self.ring)))]
                self.model.learn(o, a, t)
                self.replay_writes += 1
        self.pending = None
        if view is not None:
            self.pen_pixel = np.asarray(view["pen"], float)
            self.belief = np.asarray(view["canvas"], bool).copy()
            if self.sheet is not None:
                self.sheet.observe(self.belief)
        if moment.terminated or moment.truncated:
            return None
        observation = {k: np.asarray(v, float) for k, v in moment.observation.items()}
        goal = self.intention.encode(max(self.search.distances), self.intention_gain)
        if self.planning and self.sheet is not None and self.countdown <= 0:
            self.intention, _ = self.search.choose(self.model, observation, self.sheet, self.pen_pixel, self.geometry, goal)
            self.countdown = self.replan_every
            goal = self.intention.encode(max(self.search.distances), self.intention_gain)
        self.countdown -= 1
        if self.epsilon > 0 and self.rng.random() < self.epsilon:
            action = int(self.rng.integers(ACTIONS))
            prediction = self.model.predict_batch([observation], [action])[0]
            budget: dict[str, int] = {}
        else:
            action, prediction, budget = self.planner.plan(self.model, observation, self.sheet if self.sheet is not None else Sheet(self.target, cap=self.geometry.cap), self.pen_pixel, self.intention, self.geometry, self.belief, self.target, goal)
        self.pending = (observation, action)
        self.next_id += 1
        return Choice(self.next_id, action, controller="mlp", prediction=prediction, budget=budget)
