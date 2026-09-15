"""The artist's brain: the S00 agent with a records world head and a supplied two-level search.

The records head reads the body sensors, the pen state, the ink of the last decision, the canvas
and target windows around the pen, the committed intention and the candidate action, and holds
the S01 body consequences (hand acceleration, velocity change, angle change) and the ink the
next decision leaves in the window around the pen. Both are witnessed sensors; no stroke, path
or next-error location is ever supplied as a label.

Decisions come from supplied search over those learned consequences. The slow level proposes an
intention, a local endpoint and a pen state from a fixed polar grid, and scores the canvas it
imagines by the supplied objective: the ink comes from the learned model, the score from
``Sheet``. The fast level scores the eighteen choices by the ink they are expected to leave and
by how well the imagined hand velocity tracks the intention, through the S01 velocity-field
objective and the S01 composition of an imagined body state. What the model predicts is a
conditional mean, so a predicted window is read as expected ink, weighted cell by cell and
measured against the strongest cell of the decision, never as a fixed number of pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Protocol

import numpy as np

from .env import (
    ACTION_FIELDS,
    ACTIONS,
    FIELDS,
    PREDICTED,
    WINDOW,
    CanvasConfig,
    Sheet,
    crop,
    hand_of,
    join_action,
    pixel_of,
    window_origin,
)
from .paths import ensure

ensure()

from agent.brain import (
    ActorConfig,
    Agent,
    AgentConfig,
    Field,
    GraphSpec,
    RecordsConfig,
    WorldConfig,
)
from agent.life import Moment
from arm.brain import ModelPlanner, compose

GOAL = 4  # the intention port: direction, distance, pen state


def graph_spec(config: dict[str, Any]) -> GraphSpec:
    observation = tuple(Field(name, "continuous", width, lo, hi) for name, (width, lo, hi) in FIELDS.items())
    prediction = tuple(Field(name, "continuous", FIELDS[name][0], FIELDS[name][1], FIELDS[name][2], source=name) for name in PREDICTED)
    g = config["graph"]
    return GraphSpec(
        observation=observation, prediction=prediction, action_fields=ACTION_FIELDS, goal=GOAL,
        perceptual=g.get("perceptual", 0), workspace=g["workspace"], dynamics=g["dynamics"], episodic=g.get("episodic", 0),
        missing_flags=g.get("missing_flags", True), flags_per_field=g.get("flags_per_field", True),
        density=g.get("density", 1.0), scale=g.get("scale", 1.0), source_scale=g.get("source_scale", 2.0),
        sensory_to_dynamics=g.get("sensory_to_dynamics", True), bias=g.get("bias", 0.25), input_gain=g.get("input_gain", 2.0),
        field_gains=g.get("field_gains", {}), action_gain=g.get("action_gain"), readout_init=g.get("readout_init", 0.0),
        dt=g.get("dt", 0.5), lateral=g.get("lateral", 0.0),
    )


def agent_config(config: dict[str, Any], **over: Any) -> AgentConfig:
    cfg = AgentConfig(
        graph=graph_spec(config), world=WorldConfig(**config["world"]), actor=ActorConfig(**config["actor"]),
        records=None if config.get("records") is None else RecordsConfig(**config["records"]),
        context_decay=config.get("context_decay", 0.8), context_amplitude=config.get("context_amplitude", 1.0),
        controller="planner",
    )
    return replace(cfg, **over) if over else cfg


class Model(Protocol):
    def predict_batch(self, observations: list[dict[str, np.ndarray]], actions: list[int], *, goal: np.ndarray | None = None, valued: bool = True) -> list[dict[str, np.ndarray]]: ...


# -- imagined canvases


def mark_weights(predicted: np.ndarray, origin: tuple[int, int], floor: float, size: int) -> tuple[np.ndarray, np.ndarray]:
    """The canvas pixels a predicted window expects ink in, with the weight it gives each."""
    grid = np.asarray(predicted, float).reshape(WINDOW, WINDOW)
    index = np.argwhere(grid >= floor)
    if not len(index):
        return np.zeros((0, 2), np.int64), np.zeros(0)
    points = index + np.array(origin, np.int64)
    inside = (points >= 0).all(axis=1) & (points[:, 0] < size) & (points[:, 1] < size)
    return points[inside], grid[index[inside, 0], index[inside, 1]]


def compose_canvas(observation: dict[str, np.ndarray], deltas: dict[str, np.ndarray], pen: int, canvas: np.ndarray, target: np.ndarray, config: CanvasConfig) -> dict[str, np.ndarray]:
    """The imagined next observation: the S01 body composition plus the pen, the ink and the
    windows the moved pen would look at. Bookkeeping around learned consequences, never ink."""
    out = compose(observation, deltas)
    origin = window_origin(pixel_of(out["hand"], config), config.window)
    out["pen"] = np.array([float(pen)])
    out["mark"] = np.clip(np.asarray(deltas["mark"], float), 0.0, 1.0)
    out["canvas_local"] = crop(canvas, origin, config.window).ravel()
    out["target_local"] = crop(target, origin, config.window).ravel()
    return out


# -- the slow level: an intention


@dataclass(frozen=True)
class Intention:
    """A proposed stroke: where the pen should go next and whether it draws on the way."""

    endpoint: np.ndarray
    pen: int
    direction: np.ndarray
    distance: float
    score: float = 0.0
    valid: bool = True

    def encode(self, max_distance: float, gain: float) -> np.ndarray:
        return gain * np.array([
            0.5 + 0.5 * float(self.direction[0]),
            0.5 + 0.5 * float(self.direction[1]),
            min(float(self.distance) / max_distance, 1.0),
            float(self.pen),
        ])


def empty_intention() -> Intention:
    """No intention yet: the fast level holds the pen where it is."""
    return Intention(np.zeros(2), 0, np.zeros(2), 0.0, valid=False)


@dataclass
class IntentionSearch:
    """Supplied search over local endpoints, scored by the canvases the learned ink model imagines."""

    directions: int = 8
    distances: tuple[float, ...] = (3.0, 7.0, 12.0)
    beam: int = 4
    horizon: int = 2
    approach: float = 0.1  # weight of the pen's distance to the nearest target pixel no ink has reached
    travel: float = 0.01  # weight of the path length: ties go to the shorter move
    step: float = 1.2  # the per-decision displacement a stroke is imagined in, in pixels
    relative: float = 0.35  # a window cell counts as ink at this fraction of the strongest cell of the query

    def units(self) -> np.ndarray:
        angles = np.arange(self.directions) * 2 * np.pi / self.directions
        return np.stack([np.sin(angles), np.cos(angles)], axis=1)  # rows, columns

    def stamps(self, model: Model, observation: dict[str, np.ndarray], config: CanvasConfig, goal: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
        """What the learned model says one decision of motion leaves, per direction and pen state.

        The query holds no torque; the imagined reading moves the hand one nominal step along the
        direction, so the ink the model predicts is the ink it has learned that motion leaves. The
        cells kept are those at ``relative`` of the strongest cell over every query, so a pen state
        the model expects no ink from stamps nothing."""
        observations, actions, keys = [], [], []
        for k, unit in enumerate(self.units()):
            delta = np.array([unit[1], -unit[0]]) * self.step * config.pixel  # pixels (row, column) to body coordinates
            for pen in (0, 1):
                imagined = {name: np.asarray(value, float).copy() for name, value in observation.items()}
                imagined["d_hand"] = delta
                imagined["dd_hand"] = np.zeros(2)
                imagined["pen"] = np.array([float(pen)])
                observations.append(imagined)
                actions.append(join_action(4, pen))  # the torque pair (0, 0)
                keys.append((k, pen))
        predictions = model.predict_batch(observations, actions, goal=goal, valued=False)
        windows = [np.asarray(p["mark"], float).reshape(WINDOW, WINDOW) for p in predictions]
        peak = max(float(w.max()) for w in windows)
        floor = max(self.relative * peak, 1e-6)
        return {key: np.argwhere(window >= floor) - WINDOW // 2 for key, window in zip(keys, windows, strict=True)}

    def marks(self, offsets: np.ndarray, start: np.ndarray, unit: np.ndarray, distance: float, size: int) -> np.ndarray:
        """The pixels a stroke of that length would ink, the stamp repeated along the path."""
        if not len(offsets):
            return np.zeros((0, 2), np.int64)
        steps = np.arange(0.0, distance + 1e-9, self.step) if distance > 0 else np.zeros(1)
        centres = np.rint(start[None, :] + steps[:, None] * unit[None, :]).astype(np.int64)
        points = (centres[:, None, :] + offsets[None, :, :]).reshape(-1, 2)
        inside = (points >= 0).all(axis=1) & (points[:, 0] < size) & (points[:, 1] < size)
        return np.unique(points[inside], axis=0)

    def choose(self, model: Model, observation: dict[str, np.ndarray], sheet: Sheet, pen_pixel: np.ndarray, config: CanvasConfig, goal: np.ndarray) -> tuple[Intention, dict[str, int]]:
        stamps = self.stamps(model, observation, config, goal)
        units = self.units()
        size = config.size
        expansions = 2 * self.directions
        beam = [(sheet.copy(), np.asarray(pen_pixel, float), 0.0, None)]
        best: tuple[float, Intention] | None = None
        for _ in range(self.horizon):
            children = []
            for state, position, score, first in beam:
                for (k, pen), offsets in stamps.items():
                    for distance in (0.0, *self.distances):
                        unit = units[k]
                        end = position + unit * distance
                        if (end < 0).any() or (end > size - 1).any():
                            continue
                        child = state.copy()
                        gain = -child.apply(self.marks(offsets, position, unit, distance, size))
                        total = score + gain - self.travel * distance / size
                        leaf = total - self.approach * self.reach(child, end) / size
                        intention = Intention(end, int(pen), unit, float(distance), leaf)
                        children.append((child, end, total, first if first is not None else intention, leaf))
                        expansions += 1
            if not children:
                break
            children.sort(key=lambda c: -c[4])
            for _state, _end, _total, first, leaf in children[: self.beam]:
                if best is None or leaf > best[0]:
                    best = (leaf, first)
            beam = [(state, end, total, first) for state, end, total, first, _ in children[: self.beam]]
        if best is None:
            return empty_intention(), {"intentions": expansions}
        return best[1], {"intentions": expansions}

    @staticmethod
    def reach(state: Sheet, position: np.ndarray) -> float:
        """Pixels from a position to the nearest target pixel no ink has reached."""
        uncovered = state.uncovered()
        if not len(uncovered):
            return 0.0
        return float(np.linalg.norm(uncovered - position[None, :], axis=1).min())


# -- the fast level: torques and the pen


@dataclass
class CanvasPlanner:
    """Bounded search over the eighteen choices through the learned body and ink models.

    Every candidate is imagined ``rollout`` decisions deep by holding it. The ink it is expected
    to leave is scored by the supplied objective, weighted by the model's own prediction, and its
    body consequence by the S01 velocity-field objective toward the intention's endpoint."""

    rollout: int = 2
    canvas_weight: float = 3.0
    relative: float = 0.25  # a window cell counts at this fraction of the strongest cell of the decision
    objective: str = "velocity"  # the S01 leaf objectives: track a desired hand velocity, or coast
    coast: int = 3
    gain: float = 4.0
    max_speed: float = 0.18
    decision_seconds: float = 0.1

    def tracker(self) -> ModelPlanner:
        return ModelPlanner(depth=1, beam=1, rollout=1, objective=self.objective, coast=self.coast, gain=self.gain, max_speed=self.max_speed, decision_seconds=self.decision_seconds)

    def plan(self, model: Model, observation: dict[str, np.ndarray], sheet: Sheet, pen_pixel: np.ndarray, intention: Intention, config: CanvasConfig, canvas: np.ndarray, target: np.ndarray, goal: np.ndarray) -> tuple[int, dict[str, np.ndarray], dict[str, int]]:
        tracker = self.tracker()
        endpoint = hand_of(intention.endpoint if intention.valid else pen_pixel, config)
        actions = list(range(ACTIONS))
        observations = [observation] * ACTIONS
        sheets = [sheet.copy() for _ in actions]
        pixels = [np.asarray(pen_pixel, float) for _ in actions]
        scores = np.zeros(ACTIONS)
        first: list[dict[str, np.ndarray]] = []
        expansions = 0
        rollout = max(1, self.rollout)
        for step in range(rollout):
            predictions = model.predict_batch(observations, actions, goal=goal, valued=False)
            expansions += len(actions)
            if step == 0:
                first = list(predictions)
            floor = max(self.relative * max(float(np.max(p["mark"])) for p in predictions), 1e-6)
            composed = []
            for k, (obs, action, prediction) in enumerate(zip(observations, actions, predictions, strict=True)):
                points, weights = mark_weights(prediction["mark"], window_origin(pixels[k], config.window), floor, config.size)
                scores[k] -= self.canvas_weight * sheets[k].apply(points, weights)
                nxt = compose_canvas(obs, prediction, action % 2, sheets[k].drawn, target, config)
                if step == rollout - 1:
                    scores[k] += tracker.leaf_score(nxt, endpoint)
                composed.append(nxt)
                pixels[k] = pixel_of(nxt["hand"], config)
            observations = composed
        action = int(np.argmax(scores))
        return action, first[action], {"expansions": expansions, "rollout": rollout}


# -- the candidate life


class ArtistLife:
    """One artist: the agent, its belief of the canvas, its intention and the supplied search."""

    def __init__(self, config: dict[str, Any], seed: int, *, learning: bool = True, readback: bool = True, shuffle_actions: bool = False) -> None:
        self.config = config
        self.geometry = CanvasConfig(**config["canvas"])
        self.search = IntentionSearch(**config["intention"])
        self.planner = CanvasPlanner(**config["planner"])
        self.readback = readback
        self.replan_every = int(config["intention_interval"])
        self.intention_gain = float(config["intention_gain"])
        self.agent = Agent(agent_config(config, learning=learning), seed=seed, planner=self._planner())
        self.shuffle = shuffle_actions
        self.shuffle_rng = np.random.default_rng(seed * 977 + 5)
        self.planning = True  # the scribble phase acts at random and skips the slow level
        self.sheet: Sheet | None = None
        self.belief = np.zeros((self.geometry.size, self.geometry.size), bool)
        self.target = np.zeros_like(self.belief)
        self.pen_pixel = np.zeros(2)
        self.intention = empty_intention()
        self.countdown = 0
        self.budget: dict[str, int] = {}
        self.decision = None
        self.last_prediction: dict[str, np.ndarray] | None = None

    # -- the agent's controller

    def _planner(self):
        def planner(agent: Agent, row: int, moment: Moment, drive: np.ndarray, free: Any):
            assert self.sheet is not None
            observation = {k: np.asarray(v) for k, v in moment.observation.items()}
            action, prediction, budget = self.planner.plan(agent, observation, self.sheet, self.pen_pixel, self.intention, self.geometry, self.belief, self.target, np.asarray(moment.goal))
            return action, prediction, {**budget, **self.budget}
        return planner

    def set_epsilon(self, epsilon: float) -> None:
        self.agent.config = replace(self.agent.config, actor=replace(self.agent.config.actor, epsilon=epsilon))

    # -- perception

    def begin(self, view: dict[str, Any]) -> None:
        """A new drawing: the target is seen, the belief starts from the canvas in front of it."""
        self.target = np.asarray(view["target"], bool).copy()
        self.belief = np.asarray(view["canvas"], bool).copy()
        self.sheet = Sheet(self.target, cap=self.geometry.cap)
        self.sheet.observe(self.belief)
        self.pen_pixel = np.asarray(view["pen"], float)
        self.intention = empty_intention()
        self.countdown = 0

    def replica(self, *, frozen: bool = False, readback: bool | None = None, agent: Agent | None = None) -> ArtistLife:
        """An isolated life for evaluation: a copy of the agent, fresh belief, exploration off.

        ``frozen`` reports initial competence and never learns; otherwise the copy keeps learning.
        ``agent`` installs a given agent instead (a loaded or corrupted checkpoint). The planner
        closure is rebound to the copy, so the parent life is untouched."""
        out = ArtistLife.__new__(ArtistLife)
        out.__dict__.update(self.__dict__)
        out.agent = agent if agent is not None else (self.agent.frozen() if frozen else self.agent.clone())
        out.agent.planner = out._planner()
        out.agent.config = replace(out.agent.config, actor=replace(out.agent.config.actor, epsilon=0.0))
        out.readback = self.readback if readback is None else readback
        out.shuffle = False
        out.planning = True
        out.sheet = None
        out.belief = np.zeros((self.geometry.size, self.geometry.size), bool)
        out.target = np.zeros_like(out.belief)
        out.intention = empty_intention()
        out.countdown = 0
        out.budget = {}
        out.decision = None
        out.last_prediction = None
        out.detach()  # the copy attaches to its own environment: fresh cursors, no pending decision
        return out

    def detach(self) -> None:
        """Leave one environment for another: a decision without its outcome is dropped, the
        stream's cursors and transient state start afresh, learned records and weights stay."""
        self.agent.abandon(0)
        self.agent.new_stream(0)

    def observe(self, view: dict[str, Any]) -> None:
        self.pen_pixel = np.asarray(view["pen"], float)
        if not self.readback:
            return  # the canvas channel is removed: the belief and the objective stop at the last sight
        self.belief = np.asarray(view["canvas"], bool).copy()
        assert self.sheet is not None
        self.sheet.observe(self.belief)

    def dress(self, moment: Moment) -> Moment:
        """The moment the agent reads: the committed intention in the goal port, the canvas
        channels flagged unobserved when readback is removed."""
        changes: dict[str, Any] = {"goal": self.intention.encode(max(self.search.distances), self.intention_gain)}
        if not self.readback:
            changes["observed"] = {name: np.zeros_like(value, bool) if name in ("canvas_local", "mark") else np.ones_like(value, bool) for name, value in moment.observation.items()}
        return Moment(**{**moment.__dict__, **changes})

    # -- one decision

    def step(self, moment: Moment, view: dict[str, Any] | None = None):
        if view is not None:
            self.observe(view)
        if self.planning and self.sheet is not None and self.countdown <= 0 and not (moment.terminated or moment.truncated):
            goal = self.intention.encode(max(self.search.distances), self.intention_gain)
            observation = {k: np.asarray(v) for k, v in moment.observation.items()}
            self.intention, self.budget = self.search.choose(self.agent, observation, self.sheet, self.pen_pixel, self.geometry, goal)
            self.countdown = self.replan_every
        self.countdown -= 1
        decision = self.agent.step(self.dress(moment))
        self.decision = decision
        self.last_prediction = None if decision is None else decision.prediction
        return decision

    def feedback(self, world, decision) -> Moment:
        """Execute the committed choice; the shuffled control feeds back another choice's outcome."""
        if not self.shuffle:
            return world.act(decision.decision_id, decision.action)
        executed = int(self.shuffle_rng.integers(ACTIONS))
        m = world.act(decision.decision_id, executed)
        return Moment(**{**m.__dict__, "executed": decision.action})
