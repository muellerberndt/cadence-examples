"""The odour arena: the room's plan, two odour sources, sugar under one of them.

A batch of flies at cruising height, each a position and a heading. Two odours spread from two
sources as Gaussian fields; the fly's two antennae sample each field on a bilateral baseline
(declared: the baseline stands in for the casting sweeps of a real fly's search), and the
olfactory receptor neurons of each odour class receive a rectified left-right comparison plus
the concentration. A search ends at a source (sugar, or an empty source) or at the time limit.

Three tasks share the arena. In the T-maze (the experiment of the paper, the assay of Tully and
Quinn 1985 in which every olfactory conditioning of the animal is scored) the fly stands at the
junction of two arms, one smelling of each odour, sugar at the end of one; every decision it
approaches the odour of the arm it faces or turns to face the other arm; a search ends at an
arm's end. Avoiding one odour therefore means taking the other, as in the animal's choice, and a
fly that approaches everything scores one half. In the open-field valence task the brain chooses
the same way but the arena supplies a turn toward or away from the smell in the room's plan, so
avoidance can lead nowhere (this task times out under a naive fly and is kept for the record).
In the steering task the brain turns the fly left or right itself. Metres throughout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

__all__ = ["ArenaConfig", "Arena", "ODOURS", "ACTIONS", "STEER_ACTIONS"]

ODOURS = ("decaying_fruit", "yeasty")  # the two odour classes of the release the sources emit
ACTIONS = ("approach", "avoid")  # the valence task: what to do about the odour smelled most
STEER_ACTIONS = ("left", "right", "straight")  # the steering task


@dataclass(frozen=True)
class ArenaConfig:
    room: tuple[float, float] = (4.0, 3.0)  # m
    speed: float = 0.3  # m/s cruise
    decision_s: float = 0.1  # s between decisions
    turn: float = np.pi / 4  # rad per turn decision, and the largest supplied turn
    source_distance: float = 1.0  # m from the start to each source
    source_angle: float = np.pi / 5  # rad, each source this far off the start heading
    sigma: float = 0.5  # m, the odour field's width
    baseline: float = 0.05  # m, half the distance between the antennae as sampled (declared)
    reach: float = 0.1  # m, a source counts as reached inside this radius
    limit: int = 120  # decisions before a search times out
    comparison: float = 3.0  # drive per unit of log concentration ratio
    magnitude: float = 1.0  # drive per unit of concentration at the head
    task: str = "tmaze"  # "tmaze": the two-arm choice; "valence": approach or avoid in the open field; "steer": turn left, right or fly straight
    arm: float = 0.3  # m, the length of a T-maze arm (ten decisions at cruise; the walking assay's arms are about as long)
    arm_floor: float = 0.3  # the odour concentration of the arm the fly faces at the junction (rising to one at its end)
    arm_other: float = 0.05  # the concentration of the other arm's odour, at the junction
    smell_floor: float = 0.02  # concentration below which nothing is smelled and the fly wanders
    empty_reward: float = 0.0  # the outcome at the other source: 0 for nothing, -1 for quinine on it (the differential assay)
    step_cost: float = 0.0  # the cost of every decision: hunger and effort, so a long search pays
    wander: float = 0.15  # rad, the standard deviation of the heading noise every decision (casting, declared)

    def to_dict(self) -> dict:
        return asdict(self)


class Arena:
    def __init__(self, batch: int, seed: int = 0, meaning: int = 0, config: ArenaConfig | None = None) -> None:
        self.cfg = config or ArenaConfig()
        self.batch = batch
        self.rng = np.random.default_rng(seed)
        self.meaning = meaning  # which source carries sugar: 0 = decaying_fruit, 1 = yeasty
        self.p = np.zeros((batch, 2)); self.h = np.zeros(batch); self.sources = np.zeros((batch, 2, 2)); self.steps = np.zeros(batch, dtype=int)
        self.dist = np.zeros(batch)
        self.arms = np.zeros(batch, dtype=int); self.face = np.zeros(batch, dtype=int); self.along = np.zeros(batch)  # the T-maze
        for i in range(batch):
            self._new(i)

    def _new(self, i: int) -> None:
        c = self.cfg
        if c.task == "tmaze":
            self.arms[i] = self.rng.integers(0, 2)  # which odour is in arm 0 (arm 1 has the other)
            self.face[i] = self.rng.integers(0, 2)  # the arm the fly faces at the junction
            self.along[i] = 0.0
            self.steps[i] = 0
            self.dist[i] = c.arm  # to the sugar arm's end, through the junction if need be
            return
        margin = c.source_distance + 0.3
        self.p[i] = [self.rng.uniform(margin, c.room[0] - margin), self.rng.uniform(margin, c.room[1] - margin)]
        self.h[i] = self.rng.uniform(-np.pi, np.pi)
        side = self.rng.choice([-1.0, 1.0])
        for k, sign in enumerate((side, -side)):
            a = self.h[i] + sign * c.source_angle
            self.sources[i, k] = self.p[i] + c.source_distance * np.array([np.cos(a), np.sin(a)])
        self.steps[i] = 0
        self.dist[i] = np.linalg.norm(self.sources[i, self.meaning] - self.p[i])

    def concentration(self, points: np.ndarray, k: int) -> np.ndarray:
        d2 = ((points - self.sources[:, k]) ** 2).sum(axis=1)
        return np.exp(-d2 / (2 * self.cfg.sigma ** 2))

    def faced_odour(self) -> np.ndarray:
        """T-maze: the odour of the arm the fly faces (0 fruit, 1 yeast)."""
        return np.where(self.face == 0, self.arms, 1 - self.arms)

    def observe(self) -> np.ndarray:
        """(batch, 6): for each odour the left and right ORN drives, then its concentration at the head."""
        c = self.cfg
        if c.task == "tmaze":
            out = np.zeros((self.batch, 6))
            faced = self.faced_odour()
            level = c.arm_floor + (1.0 - c.arm_floor) * np.clip(self.along / c.arm, 0.0, 1.0)
            other = c.arm_other * (1.0 - np.clip(self.along / c.arm, 0.0, 1.0))
            for k in range(2):
                ck = np.where(faced == k, level, other)
                out[:, 3 * k] = np.clip(c.magnitude * ck, 0.0, 1.0); out[:, 3 * k + 1] = out[:, 3 * k]; out[:, 3 * k + 2] = ck
            return out
        n = np.stack([-np.sin(self.h), np.cos(self.h)], axis=1)  # the left normal
        left, right = self.p + c.baseline * n, self.p - c.baseline * n
        out = np.zeros((self.batch, 6))
        for k in range(2):
            cl, cr, ch = self.concentration(left, k), self.concentration(right, k), self.concentration(self.p, k)
            ratio = np.log(cl + 1e-9) - np.log(cr + 1e-9)
            out[:, 3 * k] = np.clip(c.comparison * np.maximum(ratio, 0.0) + c.magnitude * ch, 0.0, 1.0)
            out[:, 3 * k + 1] = np.clip(c.comparison * np.maximum(-ratio, 0.0) + c.magnitude * ch, 0.0, 1.0)
            out[:, 3 * k + 2] = ch
        return out

    def smelled(self) -> np.ndarray:
        """Which odour is smelled most at the head: 0 or 1, or -1 when both are below the floor."""
        if self.cfg.task == "tmaze":
            return self.faced_odour()
        c0, c1 = self.concentration(self.p, 0), self.concentration(self.p, 1)
        k = np.where(c1 > c0, 1, 0)
        return np.where(np.maximum(c0, c1) < self.cfg.smell_floor, -1, k)

    def bearing(self, k: np.ndarray) -> np.ndarray:
        """The signed turn from the heading to source k's direction, in (-pi, pi]."""
        idx = np.clip(k, 0, 1)
        src = self.sources[np.arange(self.batch), idx]
        want = np.arctan2(src[:, 1] - self.p[:, 1], src[:, 0] - self.p[:, 0])
        return (want - self.h + np.pi) % (2 * np.pi) - np.pi

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        c = self.cfg
        action = np.asarray(action)
        if c.task == "tmaze":
            return self._step_tmaze(action)
        if c.task == "valence":
            k = self.smelled()
            toward = self.bearing(k)
            away = (toward + np.pi + np.pi) % (2 * np.pi) - np.pi
            want = np.where(action == 0, toward, away)
            turn = np.clip(want, -c.turn, c.turn)
            turn = np.where(k < 0, self.rng.uniform(-c.turn, c.turn, size=self.batch), turn)  # nothing smelled: a wander
            self.h = self.h + turn + self.rng.normal(0.0, c.wander, size=self.batch)
        else:
            self.h = self.h + np.where(action == 0, c.turn, np.where(action == 1, -c.turn, 0.0))
        self.h = (self.h + np.pi) % (2 * np.pi) - np.pi
        step = c.speed * c.decision_s
        self.p = self.p + step * np.stack([np.cos(self.h), np.sin(self.h)], axis=1)
        self.p = np.clip(self.p, 0.05, np.array(c.room) - 0.05)
        self.steps += 1
        new_dist = np.linalg.norm(self.sources[:, self.meaning] - self.p, axis=1)
        approach = (self.dist - new_dist) / step
        self.dist = new_dist
        food = new_dist < c.reach
        empty = (np.linalg.norm(self.sources[:, 1 - self.meaning] - self.p, axis=1) < c.reach) & ~food
        timeout = (self.steps >= c.limit) & ~food & ~empty
        done = food | empty | timeout
        reward = food.astype(float) + c.empty_reward * empty.astype(float) - c.step_cost
        info = {"food": food, "empty": empty, "timeout": timeout, "steps": self.steps.copy(), "approach": approach, "smelled": self.smelled() if c.task != "valence" else k}
        for i in np.flatnonzero(done):
            self._new(i)
        return self.observe(), reward, done, info

    def _step_tmaze(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Approach (0): a step down the faced arm; avoid (1): back to the junction facing the other arm."""
        c = self.cfg
        smelled = self.faced_odour()
        step = c.speed * c.decision_s
        approach = action == 0
        self.along = np.where(approach, self.along + step, 0.0)
        self.face = np.where(approach, self.face, 1 - self.face)
        self.steps += 1
        sugar_arm = np.where(self.arms == self.meaning, 0, 1)
        new_dist = np.where(self.face == sugar_arm, c.arm - self.along, c.arm + self.along)
        gain = (self.dist - new_dist) / step
        self.dist = new_dist
        at_end = self.along >= c.arm
        food = at_end & (self.face == sugar_arm)
        empty = at_end & ~food
        timeout = (self.steps >= c.limit) & ~at_end
        done = food | empty | timeout
        reward = food.astype(float) + c.empty_reward * empty.astype(float) - c.step_cost
        info = {"food": food, "empty": empty, "timeout": timeout, "steps": self.steps.copy(), "approach": gain, "smelled": smelled}
        for i in np.flatnonzero(done):
            self._new(i)
        return self.observe(), reward, done, info
