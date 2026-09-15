"""The canvas world: the accepted S01 arm body holds a pen over a 32x32 monochrome canvas.

The body is ``arm.env.Arm`` from the accepted S01 stage, imported and held, never copied: its
integrator, its sensors and its bounds are the ones S01 was accepted with, and its own reward
and termination are switched off here (success radius 0, no horizon). A decision is one of
eighteen choices, nine torque pairs by two pen states, joint index ``torque * 2 + pen``. The
environment rasterises the stroke the pen sweeps during the decision, one pixel wide, so no
path ever leaves this module. The moment exposes the body sensors, the pen state, the ink the
last decision left in the 8x8 window centred on the pen it started from, and the current canvas
and target in the aligned 8x8 window centred on the pen now. The full canvas and target are the
visual channels read by the supplied planner through ``view``.

The measurements are supplied to the evaluator: the symmetric Chamfer distance between the
drawn and target foreground, normalised by the canvas diagonal, one when exactly one of them
is empty, and the foreground F1 within one pixel. The reward of a decision is the reduction of
the measured discrepancy minus the torque cost. ``Sheet`` is the planner's objective, the same
discrepancy with the target term capped, so ink that lands far from the target earns nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .paths import ensure

ensure()

from agent.life import Moment, seed_for
from arm.env import FIELDS as BODY_FIELDS
from arm.env import TORQUES, Arm, ArmConfig

STAGE = "S04"
SIZE = 32
WINDOW = 8
ACTIONS = 2 * len(TORQUES)  # nine torque pairs by pen up/down
ACTION_FIELDS = (len(TORQUES), 2)
FIELDS: dict[str, tuple[int, float, float]] = {
    **BODY_FIELDS,
    "pen": (1, 0.0, 1.0),
    "mark": (WINDOW * WINDOW, 0.0, 1.0),  # the ink the last decision left in the window it started in
    "canvas_local": (WINDOW * WINDOW, 0.0, 1.0),
    "target_local": (WINDOW * WINDOW, 0.0, 1.0),
}
PREDICTED = ("dd_hand", "d_velocity", "d_angles", "mark")
FAMILIES = ("segment", "two_segments", "polygon", "curve", "composition")
SIMPLE = ("segment", "two_segments")


def split_action(action: int) -> tuple[int, int]:
    """The joint action index into the torque pair and the pen state (row-major, as the brain)."""
    if not 0 <= int(action) < ACTIONS:
        raise ValueError("action outside the eighteen choices")
    return divmod(int(action), 2)


def join_action(torque: int, pen: int) -> int:
    return int(torque) * 2 + int(pen)


# -- geometry and rasterisation


@dataclass(frozen=True)
class CanvasConfig:
    size: int = SIZE
    window: int = WINDOW
    extent: float = 0.7  # the canvas side in arm lengths
    centre: tuple[float, float] = (0.0, 0.5)  # its middle in body coordinates; the fold at the origin stays outside
    margin: int = 3  # target shapes keep this many pixels from the border
    max_decisions: int = 200  # 1,000 body steps at five steps per decision
    patience: int = 60  # decisions without a better discrepancy, counted from the first mark, before the drawing is abandoned
    finish: float = 0.02  # a drawing is complete below this discrepancy
    torque_cost: float = 0.001
    pen_offset: tuple[float, float] = (0.0, 0.0)  # pixels: the perturbation that displaces the ink
    cap: float = 4.0  # the planner objective's capped distance, in pixels

    @property
    def pixel(self) -> float:
        return self.extent / self.size

    @property
    def diagonal(self) -> float:
        return float(np.hypot(self.size, self.size))


def body_config(lengths: tuple[float, float] = (0.5, 0.5), gain: float = 1.0, substeps: int = 5) -> ArmConfig:
    """The S01 body with its own goal switched off: this world supplies the reward."""
    return ArmConfig(lengths=lengths, substeps=substeps, gain=gain, success_radius=0.0, success_hold=10**9, horizon=10**9)


def pixel_of(hand: np.ndarray, config: CanvasConfig) -> np.ndarray:
    """Continuous pixel coordinates (row, column) of a hand position; integers are pixel centres."""
    middle = (config.size - 1) / 2.0
    column = (float(hand[0]) - config.centre[0]) / config.pixel + middle
    row = middle - (float(hand[1]) - config.centre[1]) / config.pixel
    return np.array([row, column])


def hand_of(pixel: np.ndarray, config: CanvasConfig) -> np.ndarray:
    middle = (config.size - 1) / 2.0
    x = config.centre[0] + (float(pixel[1]) - middle) * config.pixel
    y = config.centre[1] - (float(pixel[0]) - middle) * config.pixel
    return np.array([x, y])


def bresenham(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """The one-pixel-wide line between two pixel centres, endpoints included."""
    r0, c0 = int(round(float(start[0]))), int(round(float(start[1])))
    r1, c1 = int(round(float(end[0]))), int(round(float(end[1])))
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr, sc = (1 if r1 > r0 else -1), (1 if c1 > c0 else -1)
    error = dr - dc
    points = [(r0, c0)]
    while (r0, c0) != (r1, c1):
        double = 2 * error
        if double > -dc:
            error -= dc
            r0 += sr
        if double < dr:
            error += dr
            c0 += sc
        points.append((r0, c0))
    return np.array(points, dtype=np.int64)


def draw_points(canvas: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Ink the points that lie on the canvas; returns the indices actually inked."""
    if not len(points):
        return points
    inside = (points[:, 0] >= 0) & (points[:, 0] < canvas.shape[0]) & (points[:, 1] >= 0) & (points[:, 1] < canvas.shape[1])
    points = points[inside]
    canvas[points[:, 0], points[:, 1]] = True
    return points


def render(paths: list[np.ndarray], size: int = SIZE) -> np.ndarray:
    canvas = np.zeros((size, size), bool)
    for path in paths:
        for a, b in zip(path[:-1], path[1:], strict=True):
            draw_points(canvas, bresenham(a, b))
    return canvas


def window_origin(pixel: np.ndarray, window: int) -> tuple[int, int]:
    return int(round(float(pixel[0]))) - window // 2, int(round(float(pixel[1]))) - window // 2


def crop(canvas: np.ndarray, origin: tuple[int, int], window: int) -> np.ndarray:
    """The ``window`` by ``window`` patch at ``origin``; outside the canvas reads empty."""
    out = np.zeros((window, window), float)
    r0, c0 = origin
    r1, c1 = max(r0, 0), max(c0, 0)
    r2, c2 = min(r0 + window, canvas.shape[0]), min(c0 + window, canvas.shape[1])
    if r1 < r2 and c1 < c2:
        out[r1 - r0 : r2 - r0, c1 - c0 : c2 - c0] = canvas[r1:r2, c1:c2]
    return out


# -- the supplied measurements


def _coordinates(canvas: np.ndarray) -> np.ndarray:
    rows, columns = np.nonzero(canvas)
    return np.stack([rows, columns], axis=1).astype(float)


def _pairwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)


def chamfer(drawn: np.ndarray, target: np.ndarray) -> float:
    """Symmetric Chamfer distance of the foregrounds, normalised by the canvas diagonal."""
    d, t = _coordinates(drawn), _coordinates(target)
    if not len(d) and not len(t):
        return 0.0
    if not len(d) or not len(t):
        return 1.0
    distances = _pairwise(d, t)
    diagonal = float(np.hypot(*drawn.shape))
    return float(0.5 * (distances.min(axis=1).mean() + distances.min(axis=0).mean()) / diagonal)


def foreground_f1(drawn: np.ndarray, target: np.ndarray, tolerance: float = 1.0) -> dict[str, float]:
    """Foreground precision, recall and F1 within ``tolerance`` pixels."""
    d, t = _coordinates(drawn), _coordinates(target)
    if not len(d) and not len(t):
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not len(d) or not len(t):
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    distances = _pairwise(d, t)
    precision = float((distances.min(axis=1) <= tolerance + 1e-9).mean())
    recall = float((distances.min(axis=0) <= tolerance + 1e-9).mean())
    total = precision + recall
    return {"precision": precision, "recall": recall, "f1": float(2 * precision * recall / total) if total > 0 else 0.0}


def distance_map(target: np.ndarray) -> np.ndarray:
    """Distance from every pixel to the nearest target pixel; the diagonal when the target is empty."""
    t = _coordinates(target)
    rows, columns = np.indices(target.shape)
    grid = np.stack([rows.ravel(), columns.ravel()], axis=1).astype(float)
    if not len(t):
        return np.full(target.shape, float(np.hypot(*target.shape)))
    return _pairwise(grid, t).min(axis=1).reshape(target.shape)


@dataclass
class Sheet:
    """The planner's objective on one drawing: the discrepancy with the target term capped.

    Ink that lands further than ``cap`` pixels from the target earns nothing and costs the full
    capped distance, so the search never marks the canvas to escape the empty-canvas score of
    one. Witnessed ink is certain; imagined ink carries the weight the learned model gives it,
    and the objective is then the expectation under independent cells. The evaluator uses
    ``chamfer`` and ``foreground_f1``; this is what the search reads."""

    target: np.ndarray
    cap: float = 4.0
    floor: float = 1e-3  # an imagined cell below this weight is not carried
    distances: np.ndarray = field(init=False)
    target_points: np.ndarray = field(init=False)
    mass: np.ndarray = field(init=False)  # ink already on the canvas, or expected under imagined marks
    ink: float = field(init=False, default=0.0)
    nearest: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.distances = np.minimum(distance_map(self.target), self.cap) / self.cap
        self.target_points = _coordinates(self.target)
        self.mass = np.zeros(self.target.shape)
        self.nearest = np.ones(len(self.target_points))

    def observe(self, canvas: np.ndarray) -> None:
        """Set the objective to a witnessed canvas (the brain's belief of what is drawn)."""
        self.mass = np.asarray(canvas, bool).astype(float)
        points = _coordinates(self.mass >= 0.5)
        self.ink = float(self.distances[self.mass >= 0.5].sum())
        if len(self.target_points) and len(points):
            self.nearest = np.minimum(_pairwise(self.target_points, points).min(axis=1) / self.cap, 1.0)
        else:
            self.nearest = np.ones(len(self.target_points))

    @property
    def drawn(self) -> np.ndarray:
        return self.mass >= 0.5

    @property
    def value(self) -> float:
        n = max(len(self.target_points), 1)
        return float((self.ink + self.nearest.sum()) / n)

    def peek(self, points: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, float, np.ndarray, np.ndarray]:
        """The objective's change if ``points`` were inked, with the new ink, reach and weights.

        Without weights the ink is certain and the update is one vector operation; with weights
        each cell is inked with that probability, in decreasing order, and the reach of every
        target pixel moves by its expectation."""
        if not len(points):
            return 0.0, self.ink, self.nearest, np.zeros(0)
        rows, columns = points[:, 0], points[:, 1]
        room = 1.0 - self.mass[rows, columns]
        if weights is None:
            applied = room
            ink = self.ink + float((room * self.distances[rows, columns]).sum())
            nearest = self.nearest
            if len(self.target_points):
                reach = np.minimum(_pairwise(self.target_points, points.astype(float)).min(axis=1) / self.cap, 1.0)
                nearest = np.minimum(self.nearest, reach)
        else:
            applied = np.clip(np.asarray(weights, float), 0.0, 1.0) * room
            ink, nearest = self.ink, self.nearest
            for index in np.argsort(-applied):
                weight = float(applied[index])
                if weight <= self.floor:
                    break
                ink += weight * float(self.distances[rows[index], columns[index]])
                if len(self.target_points):
                    reach = np.minimum(np.linalg.norm(self.target_points - points[index].astype(float), axis=1) / self.cap, 1.0)
                    nearest = nearest - weight * (nearest - np.minimum(nearest, reach))
        n = max(len(self.target_points), 1)
        return float((ink + nearest.sum()) / n - self.value), ink, nearest, applied

    def apply(self, points: np.ndarray, weights: np.ndarray | None = None) -> float:
        change, ink, nearest, applied = self.peek(points, weights)
        if len(points):
            self.mass[points[:, 0], points[:, 1]] = np.minimum(self.mass[points[:, 0], points[:, 1]] + applied, 1.0)
        self.ink, self.nearest = ink, nearest
        return change

    def copy(self) -> Sheet:
        out = object.__new__(Sheet)
        out.target, out.cap, out.floor = self.target, self.cap, self.floor
        out.distances, out.target_points = self.distances, self.target_points
        out.mass, out.ink, out.nearest = self.mass.copy(), self.ink, self.nearest.copy()
        return out

    def uncovered(self) -> np.ndarray:
        """The target pixels no ink has reached within the cap; the search approaches these."""
        if not len(self.target_points):
            return self.target_points
        return self.target_points[self.nearest >= 1.0 - 1e-9]


# -- target families and splits


def _rotate(points: np.ndarray, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return points @ np.array([[c, -s], [s, c]]).T


def _place(paths: list[np.ndarray], rng: np.random.Generator, config: CanvasConfig) -> list[np.ndarray]:
    """Fit a shape inside the margin and translate it to a drawn position; every pixel stays on
    the canvas, so a target is never clipped by the border."""
    stacked = np.concatenate(paths)
    low, high = stacked.min(axis=0), stacked.max(axis=0)
    box = config.size - 1 - 2 * config.margin
    shrink = min(1.0, box / max(float((high - low).max()), 1e-9))
    if shrink < 1.0:
        middle = (low + high) / 2
        paths = [(path - middle) * shrink + middle for path in paths]
        stacked = np.concatenate(paths)
        low, high = stacked.min(axis=0), stacked.max(axis=0)
    lo = config.margin - low
    hi = config.size - 1 - config.margin - high
    return [path + np.array([rng.uniform(lo[k], hi[k]) for k in range(2)]) for path in paths]


def _arc(radius: float, span: float, start: float, points: int = 24) -> np.ndarray:
    angles = start + np.linspace(0.0, span, points)
    return np.stack([radius * np.sin(angles), radius * np.cos(angles)], axis=1)


def shape(family: str, rng: np.random.Generator, rotation: float, scale: float, config: CanvasConfig) -> list[np.ndarray]:
    """The polylines of one target shape in pixel coordinates (row, column)."""
    span = config.size - 2 * config.margin - 4
    length = scale * span
    if family == "segment":
        paths = [np.array([[-length / 2, 0.0], [length / 2, 0.0]])]
    elif family == "two_segments":
        angle = rng.uniform(np.pi / 3, 2 * np.pi / 3)
        second = np.array([np.cos(angle), np.sin(angle)]) * length * rng.uniform(0.6, 1.0)
        paths = [np.array([[-length / 2, 0.0], [0.0, 0.0], list(second)])]
    elif family == "polygon":
        sides = int(rng.integers(3, 5))
        angles = np.sort(rng.uniform(0, 2 * np.pi, sides))
        radius = length / 2
        corners = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
        paths = [np.concatenate([corners, corners[:1]])]
    elif family == "curve":
        paths = [_arc(length / 2, rng.uniform(np.pi * 0.7, np.pi * 1.5), rng.uniform(0, 2 * np.pi))]
    elif family == "composition":  # held out: two components from different families
        first = np.array([[-length * 0.35, 0.0], [length * 0.35, 0.0]])
        second = _arc(length / 3, rng.uniform(np.pi * 0.6, np.pi), rng.uniform(0, 2 * np.pi))
        offset = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1)])
        offset = offset / max(float(np.linalg.norm(offset)), 1e-9) * length * 0.55
        paths = [first, second + offset]
    else:
        raise ValueError(f"unknown family {family!r}")
    return _place([_rotate(path, rotation) for path in paths], rng, config)


@dataclass(frozen=True)
class Drawing:
    index: int
    split: str
    family: str
    rotation: float
    scale: float
    paths: tuple[np.ndarray, ...]
    target: np.ndarray
    theta: np.ndarray  # the body configuration the drawing starts from

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "split": self.split, "family": self.family, "rotation": float(self.rotation), "scale": float(self.scale), "pixels": int(self.target.sum())}


DEVELOPMENT_FAMILIES = FAMILIES[:4]  # compositions are unseen until the held-out suite
ROTATIONS = 8


def drawing(split: str, index: int, brain_seed: int, config: CanvasConfig, *, families: tuple[str, ...] | None = None) -> Drawing:
    """One target, balanced by family; rotation, scale and composition split the suites.

    Development rotations sit on the eight-way grid, held-out rotations halfway between them;
    development scales lie in [0.55, 0.80], held-out scales outside that band; the composition
    family appears only in the held-out suite."""
    held_out = split not in ("development", "validation")
    names = families if families is not None else (FAMILIES if held_out else DEVELOPMENT_FAMILIES)
    rng = np.random.default_rng(seed_for(STAGE, split, brain_seed, 2000 + index if held_out else 1000 + index, 0))
    family = names[index % len(names)]
    step = 2 * np.pi / ROTATIONS
    rotation = step * ((index // len(names)) % ROTATIONS) + (step / 2 if held_out else 0.0) + rng.uniform(-step / 8, step / 8)
    scale = rng.uniform(0.40, 0.55) if held_out and index % 2 == 0 else (rng.uniform(0.80, 0.95) if held_out else rng.uniform(0.55, 0.80))
    paths = shape(family, rng, rotation, scale, config)
    target = render(paths, config.size)
    return Drawing(index, split, family, rotation, scale, tuple(paths), target, start_angles(rng, config))


def start_angles(rng: np.random.Generator, config: CanvasConfig, lengths: tuple[float, float] = (0.5, 0.5)) -> np.ndarray:
    """A body configuration whose hand starts somewhere on the canvas."""
    for _ in range(500):
        theta = rng.uniform(-np.pi, np.pi, 2)
        hand = np.array([
            lengths[0] * np.cos(theta[0]) + lengths[1] * np.cos(theta[0] + theta[1]),
            lengths[0] * np.sin(theta[0]) + lengths[1] * np.sin(theta[0] + theta[1]),
        ])
        pixel = pixel_of(hand, config)
        if (pixel >= config.margin).all() and (pixel <= config.size - 1 - config.margin).all():
            return theta
    return rng.uniform(-np.pi, np.pi, 2)


# -- the world


@dataclass
class CanvasWorld:
    """One canvas, one target, one drawing at a time; ``act`` executes a committed decision."""

    config: CanvasConfig = field(default_factory=CanvasConfig)
    body: Arm = field(default_factory=lambda: Arm(body_config()))
    seed: int = 0
    life_id: str = "artist"
    _rng: np.random.Generator = field(init=False, repr=False)
    _episode: int = field(default=-1, init=False)
    _event: int = field(default=0, init=False)
    _tick: int = field(default=0, init=False)
    _canvas: np.ndarray = field(init=False)
    _target: np.ndarray = field(init=False)
    _drawing: Drawing | None = field(default=None, init=False)
    _pen: int = field(default=0, init=False)
    _mark: np.ndarray = field(init=False)
    _discrepancy: float = field(default=1.0, init=False)
    _best: float = field(default=1.0, init=False)
    _since_best: int = field(default=0, init=False)
    _pending: int | None = field(default=None, init=False)
    _decision_action: int | None = field(default=None, init=False)
    strokes: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._canvas = np.zeros((self.config.size, self.config.size), bool)
        self._target = np.zeros((self.config.size, self.config.size), bool)
        self._mark = np.zeros(self.config.window**2)

    # -- state for the evaluator and the page; never handed to the brain

    @property
    def canvas(self) -> np.ndarray:
        return self._canvas.copy()

    @property
    def target(self) -> np.ndarray:
        return self._target.copy()

    @property
    def discrepancy(self) -> float:
        """The measured discrepancy after the last decision (the reward is its reduction)."""
        return float(self._discrepancy)

    @property
    def pen_pixel(self) -> np.ndarray:
        return pixel_of(self.body.hand, self.config)

    def view(self) -> dict[str, Any]:
        """The visual channels the supplied planner reads: both aligned images and the gaze."""
        return {"canvas": self._canvas.copy(), "target": self._target.copy(), "pen": self.pen_pixel, "pen_state": self._pen}

    def measure(self) -> dict[str, float]:
        out = {"chamfer": chamfer(self._canvas, self._target), "ink": float(self._canvas.sum()), "strokes": float(self.strokes)}
        out.update(foreground_f1(self._canvas, self._target))
        return out

    # -- drawings

    def reset(self, drawing_: Drawing | None = None, *, theta: np.ndarray | None = None) -> Moment:
        if drawing_ is None:
            drawing_ = drawing("development", int(self._rng.integers(10**6)), self.seed, self.config)
        self._drawing = drawing_
        self._target = np.asarray(drawing_.target, bool).copy()
        self._canvas = np.zeros_like(self._target)
        self._episode += 1
        self._tick = 0
        self._pen = 0
        self.strokes = 0
        self._mark = np.zeros(self.config.window**2)
        self._pending = None
        self.body.reset(theta=drawing_.theta if theta is None else theta)
        self._discrepancy = chamfer(self._canvas, self._target)
        self._best, self._since_best = self._discrepancy, 0
        return self._moment(feedback=False, reward=0.0, terminated=False, truncated=False)

    def observation(self) -> dict[str, np.ndarray]:
        origin = window_origin(self.pen_pixel, self.config.window)
        body = self.body.observation()
        return {
            **body,
            "pen": np.array([float(self._pen)]),
            "mark": self._mark.copy(),
            "canvas_local": crop(self._canvas, origin, self.config.window).ravel(),
            "target_local": crop(self._target, origin, self.config.window).ravel(),
        }

    def _moment(self, *, feedback: bool, reward: float, terminated: bool, truncated: bool) -> Moment:
        m = Moment(
            life_id=self.life_id,
            episode_id=self._episode,
            event_id=self._event,
            tick=self._tick,
            dt=self.body.config.dt * self.body.config.substeps,
            observation=self.observation(),
            observed={},
            action_mask=np.ones(ACTIONS, bool),
            feedback_for=self._pending if feedback else None,
            executed=self._decision_action if feedback else None,
            reward=reward,
            reward_known=feedback,
            terminated=terminated,
            truncated=truncated,
            final_observation=self.observation() if truncated else None,
            goal=np.zeros(4),  # the intention the brain commits is written in by its own loop
        )
        self._event += 1
        return m

    def act(self, decision_id: int, action: int) -> Moment:
        if self._pending is not None:
            raise RuntimeError("the previous decision has not been fed back")
        torque, pen = split_action(action)
        self._pending, self._decision_action = decision_id, int(action)
        window = self.config.window
        origin = window_origin(self.pen_pixel, window)
        before = crop(self._canvas, origin, window)
        start = self.pen_pixel + np.asarray(self.config.pen_offset)
        self.body.act(decision_id, torque)
        end = self.pen_pixel + np.asarray(self.config.pen_offset)
        if pen:
            drawn = draw_points(self._canvas, bresenham(start, end))
            self.strokes += int(len(drawn) > 0 and not self._pen)
        self._pen = int(pen)
        self._mark = (crop(self._canvas, origin, window) - before).ravel()
        self._tick += 1
        after = chamfer(self._canvas, self._target)
        cost = self.config.torque_cost * float(TORQUES[torque] @ TORQUES[torque])
        reward = (self._discrepancy - after) - cost
        self._discrepancy = after
        if after < self._best - 1e-9:
            self._best, self._since_best = after, 0
        elif self._canvas.any():
            # the clock starts at the first mark: travelling with the pen up never improves the
            # discrepancy, so counting it abandons a drawing in the middle of its approach
            self._since_best += 1
        terminated = after <= self.config.finish
        truncated = not terminated and (self._tick >= self.config.max_decisions or self._since_best >= self.config.patience)
        m = self._moment(feedback=True, reward=reward, terminated=terminated, truncated=truncated)
        self._pending = None
        return m
