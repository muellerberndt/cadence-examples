"""A deterministic two-link planar arm exposing only moments.

Links of length 0.5 each (a reach of 1.0), angles in [-pi, pi], angular velocities clipped
to +-2 rad/s, a 50 Hz body clock with semi-implicit Euler: joint acceleration is the
commanded torque times the motor gain minus 0.1 times the velocity. A decision holds one
of nine torque pairs {-1, 0, +1}^2 for ``substeps`` body steps (5: decisions at 10 Hz), so
one decision changes a joint velocity by up to 0.1 rad/s; the assumption is recorded in
the stage config. The sensors are proprioception: sine and cosine of both angles, the
normalized velocities, the hand position, the measured displacement and velocity and
angle changes over the last decision, and the two velocity-limit flags. The goal is the
visible target marker. Forward kinematics serve rendering and the sensors; no inverse
map, Jacobian or target-action pair leaves this module through a moment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from agent.life import Moment

TORQUES = np.array([(a, b) for a in (-1.0, 0.0, 1.0) for b in (-1.0, 0.0, 1.0)])
FIELDS = {
    "angles": (4, -1.0, 1.0),
    "velocity": (2, -1.0, 1.0),
    "hand": (2, -1.0, 1.0),
    "d_hand": (2, -0.12, 0.12),  # typical displacements are 0.03 per decision; wider bounds compress the code
    "dd_hand": (2, -0.03, 0.03),  # the change of the displacement: the hand's measured acceleration
    "d_velocity": (2, -0.15, 0.15),
    "d_angles": (2, -0.25, 0.25),
    "flags": (2, 0.0, 1.0),
}


def wrap(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


@dataclass
class ArmConfig:
    lengths: tuple[float, float] = (0.5, 0.5)
    dt: float = 0.02
    substeps: int = 5
    max_velocity: float = 2.0
    friction: float = 0.1
    gain: float = 1.0
    success_radius: float = 0.05
    success_hold: int = 10
    horizon: int = 200
    target_radius: tuple[float, float] = (0.2, 0.95)
    torque_cost: float = 0.001


@dataclass
class Arm:
    """The body, one target, one episode at a time; ``act`` executes a committed decision."""

    config: ArmConfig = field(default_factory=ArmConfig)
    seed: int = 0
    life_id: str = "arm"
    moving_target: object = None  # a callable tick -> (x, y), or None for a static target
    _rng: np.random.Generator = field(init=False, repr=False)
    _episode: int = field(default=-1, init=False)
    _event: int = field(default=0, init=False)
    _tick: int = field(default=0, init=False)
    _theta: np.ndarray = field(default_factory=lambda: np.zeros(2), init=False)
    _omega: np.ndarray = field(default_factory=lambda: np.zeros(2), init=False)
    _target: np.ndarray = field(default_factory=lambda: np.zeros(2), init=False)
    _previous: dict = field(default_factory=dict, init=False)
    _flags: np.ndarray = field(default_factory=lambda: np.zeros(2), init=False)
    _hold: int = field(default=0, init=False)
    _pending: int | None = field(default=None, init=False)
    _decision_action: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    # -- kinematics (supplied for sensors and rendering)

    def hand_of(self, theta: np.ndarray) -> np.ndarray:
        l1, l2 = self.config.lengths
        return np.array([
            l1 * np.cos(theta[0]) + l2 * np.cos(theta[0] + theta[1]),
            l1 * np.sin(theta[0]) + l2 * np.sin(theta[0] + theta[1]),
        ])

    @property
    def hand(self) -> np.ndarray:
        return self.hand_of(self._theta)

    @property
    def target(self) -> np.ndarray:
        return self._target.copy()

    @property
    def state(self) -> dict[str, np.ndarray]:
        """Evaluation and rendering only: never handed to the agent."""
        return {"theta": self._theta.copy(), "omega": self._omega.copy(), "target": self._target.copy(), "hand": self.hand}

    def distance(self) -> float:
        return float(np.linalg.norm(self.hand - self._target))

    # -- episodes

    def sample_target(self) -> np.ndarray:
        r = self._rng.uniform(*self.config.target_radius)
        a = self._rng.uniform(-np.pi, np.pi)
        return np.array([r * np.cos(a), r * np.sin(a)])

    def reset(self, *, theta: np.ndarray | None = None, target: np.ndarray | None = None) -> Moment:
        self._episode += 1
        self._tick = 0
        self._hold = 0
        self._pending = None
        self._theta = self._rng.uniform(-np.pi, np.pi, 2) if theta is None else np.asarray(theta, float).copy()
        self._omega = np.zeros(2)
        self._flags = np.zeros(2)
        self._target = self.sample_target() if target is None else np.asarray(target, float).copy()
        if self.moving_target is not None:
            self._target = np.asarray(self.moving_target(0), float)
        self._previous = {"hand": self.hand, "omega": self._omega.copy(), "theta": self._theta.copy(), "d_hand": np.zeros(2)}
        return self._moment(feedback=False, reward=0.0, terminated=False, truncated=False)

    def observation(self) -> dict[str, np.ndarray]:
        hand = self.hand
        d_hand = hand - self._previous["hand"]
        return {
            "angles": np.array([np.sin(self._theta[0]), np.cos(self._theta[0]), np.sin(self._theta[1]), np.cos(self._theta[1])]),
            "velocity": self._omega / self.config.max_velocity,
            "hand": hand,
            "d_hand": d_hand,
            "dd_hand": d_hand - self._previous["d_hand"],
            "d_velocity": self._omega - self._previous["omega"],
            "d_angles": wrap(self._theta - self._previous["theta"]),
            "flags": self._flags.copy(),
        }

    def goal(self) -> np.ndarray:
        return (self._target + 1.0) / 2.0  # the marker, in [0, 1] for the goal port

    def _moment(self, *, feedback: bool, reward: float, terminated: bool, truncated: bool) -> Moment:
        m = Moment(
            life_id=self.life_id,
            episode_id=self._episode,
            event_id=self._event,
            tick=self._tick,
            dt=self.config.dt * self.config.substeps,
            observation=self.observation(),
            observed={},
            action_mask=np.ones(len(TORQUES), bool),
            feedback_for=self._pending if feedback else None,
            executed=self._decision_action if feedback else None,
            reward=reward,
            reward_known=feedback,
            terminated=terminated,
            truncated=truncated,
            final_observation=self.observation() if truncated else None,
            goal=self.goal(),
        )
        self._event += 1
        return m

    def integrate(self, action: int) -> None:
        """Hold one torque pair for ``substeps`` body steps; record velocity-limit flags."""
        c = self.config
        torque = TORQUES[int(action)] * c.gain
        flags = np.zeros(2)
        for _ in range(c.substeps):
            self._omega = self._omega + c.dt * (torque - c.friction * self._omega)
            hit = np.abs(self._omega) >= c.max_velocity
            flags = np.maximum(flags, hit.astype(float))
            self._omega = np.clip(self._omega, -c.max_velocity, c.max_velocity)
            self._theta = wrap(self._theta + c.dt * self._omega)
        self._flags = flags

    def act(self, decision_id: int, action: int) -> Moment:
        if self._pending is not None:
            raise RuntimeError("the previous decision has not been fed back")
        if not 0 <= int(action) < len(TORQUES):
            raise ValueError("action outside the nine torque pairs")
        self._pending, self._decision_action = decision_id, int(action)
        before = self.distance()
        self._previous = {"hand": self.hand, "omega": self._omega.copy(), "theta": self._theta.copy(), "d_hand": self.hand - self._previous["hand"]}
        self.integrate(int(action))
        self._tick += 1
        if self.moving_target is not None:
            self._target = np.asarray(self.moving_target(self._tick), float)
        after = self.distance()
        torque = TORQUES[int(action)]
        reward = (before - after) - self.config.torque_cost * float(torque @ torque)
        self._hold = self._hold + 1 if after <= self.config.success_radius else 0
        terminated = self.moving_target is None and self._hold >= self.config.success_hold
        if terminated:
            reward += 1.0
        truncated = not terminated and self._tick >= self.config.horizon
        m = self._moment(feedback=True, reward=reward, terminated=terminated, truncated=truncated)
        self._pending = None
        return m


def jacobian(theta: np.ndarray, lengths: tuple[float, float]) -> np.ndarray:
    """Analytic Jacobian: baselines and evaluation only."""
    l1, l2 = lengths
    s1, c1 = np.sin(theta[0]), np.cos(theta[0])
    s12, c12 = np.sin(theta[0] + theta[1]), np.cos(theta[0] + theta[1])
    return np.array([[-l1 * s1 - l2 * s12, -l2 * s12], [l1 * c1 + l2 * c12, l2 * c12]])


def moving_path(kind: str, rng: np.random.Generator, decisions_per_second: float = 10.0):
    """A fresh line, circle or smooth combination; the path is hidden, the agent sees positions."""
    def inside(radius: float) -> np.ndarray:
        r = np.sqrt(rng.uniform(0.04, radius**2))
        a = rng.uniform(-np.pi, np.pi)
        return np.array([r * np.cos(a), r * np.sin(a)])

    if kind == "line":
        a, b = inside(0.85), inside(0.85)
        speed = rng.uniform(0.02, 0.05)  # arm lengths per second
        length = max(float(np.linalg.norm(b - a)), 1e-6)

        def path(tick: int) -> np.ndarray:
            s = (speed * tick / decisions_per_second) % (2 * length)
            s = s if s <= length else 2 * length - s
            return a + (b - a) * (s / length)
    elif kind == "circle":
        centre = inside(0.3)
        radius = rng.uniform(0.15, 0.3)
        omega = rng.uniform(0.05, 0.15) * rng.choice([-1, 1])  # radians per second
        phase = rng.uniform(0, 2 * np.pi)

        def path(tick: int) -> np.ndarray:
            t = tick / decisions_per_second
            return centre + radius * np.array([np.cos(omega * t + phase), np.sin(omega * t + phase)])
    else:
        c1 = moving_path("circle", rng, decisions_per_second)
        c2 = moving_path("line", rng, decisions_per_second)

        def path(tick: int) -> np.ndarray:
            return 0.5 * (c1(tick) + c2(tick))
    return path
