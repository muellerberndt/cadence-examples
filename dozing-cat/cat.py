"""The dozing cat: one belief patch watches a sill through a coarse retina, a resting habit
holds its paw, and a governor decides from the brain's own surprise when the cat wakes
(imagines and chases), when it dozes (the habit) and when it learns.

The sill is the unit box. The cat's paw is a point moved by a two-dimensional action per
decision at a limited speed. A laser dot appears at random times, follows a path (a slow
random walk, a smooth curve, or a jittering walk that vanishes early), and vanishes when the
paw has been within the catch radius for a few decisions (a catch) or after a timeout (a
miss). Late in the life the dot's law changes for good (it moves faster, or wraps through the
walls instead of bouncing). The brain reads a six-by-six retina of Gaussian bumps over the
dot and its own paw position; it never reads the dot's coordinates.

The cortex is one plain ``BeliefPatch`` that predicts the change of every reading under the
executed action. Its three signals are read from that patch: the repair residual
(``BeliefPath.residual``), the surprise (its own one-step read against what it then read, in
units of each reading's standard deviation, every channel at weight one) and the learning
signal's validity (whether the observed window's loss fell). The habit is the cheap routine:
the paw drifts home, one moment of the patch per decision. Imagination is ten candidate
pushes held for a horizon inside the belief's private imagination, the best executed: the
chase. Learning observes the executed window and refits the habit in imagination.

Two governors read the same readback port (the surprise over its baseline, its slow average,
the residual, the current mode): a settling patch (a small ``cadence.Brain`` whose settled
motor state is the mode; its synapses, gains and time constant are genes) and the room's
threshold rule (its thresholds are genes: the hand-designed control). Two controls bracket
them: always awake (imagination every decision) and never wakes (the habit alone). Every
genome is a dict for ``cadence.evolve(..., mutate=cadence.genes(space))``.

    python cat.py pretrain --out pretrained/cat_belief.npz
    python cat.py lives --arms patch,threshold,always_awake,never_wakes --out receipts/cat_lives_hand_set.json
    python cat.py evolve --which patch --out receipts/cat_evolution_patch.json
    python cat.py evolve --which threshold --out receipts/cat_evolution_threshold.json
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import multiprocessing
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

import cadence
from cadence import BeliefPatch, Brain, Connectome, DenseBlock, NeuronModel, StructuredPort, evolve, genes

HERE = Path(__file__).resolve().parent
LIBRARY = Path(cadence.__file__).resolve().parents[2]

# ----------------------------------------------------------------------------- the sill
GRID = 6
RETINA = GRID * GRID
# The fovea readout of the retina: the dot's centroid (two channels, centred on the sill), its
# optic flow (the centroid's change since the last decision, in units of the routine speed, as
# the room reads a rate beside a temperature) and its mass (one while a dot is present).
FOVEA = RETINA
FLOW = RETINA + 2
MASS = RETINA + 4
PAW = RETINA + 5  # the paw's own position, two channels
READINGS = RETINA + 7
PREDICTED = np.array([FOVEA, FOVEA + 1, MASS, PAW, PAW + 1])  # the change of these is what the belief predicts
OUTPUTS = len(PREDICTED)
ACTIONS = 2
PAW_SPEED = 0.05  # box units per decision at a full push
CATCH_RADIUS = 0.08
CATCH_HOLD = 3  # decisions within the radius before the dot counts as caught
EFFORT = 0.02  # the price of a push in the task cost
_GRID = (np.arange(GRID) + 0.5) / GRID
_SIGMA = 0.7 / GRID
FASTER = 2.67  # the changed law ``faster``: the dot's speed multiplied by this
GRAVITY = 0.008  # the changed law ``gravity``: the dot's velocity gains this much downward per decision, and it bounces
VMAX = 0.045  # the dot's speed under gravity is capped here (the paw moves at 0.05)
CLIP = 4.0  # standardized targets are clipped here: a dot's appearance or vanishing is a jump of tens of motion units, and unclipped it swamps the motion the belief can learn


def retina(position: np.ndarray) -> np.ndarray:
    """Gaussian bumps on the grid, peak one, for positions (..., 2)."""
    position = np.asarray(position, dtype=float)
    dx = position[..., 0, None] - _GRID
    dy = position[..., 1, None] - _GRID
    bx = np.exp(-0.5 * (dx / _SIGMA) ** 2)
    by = np.exp(-0.5 * (dy / _SIGMA) ** 2)
    return (by[..., :, None] * bx[..., None, :]).reshape(*position.shape[:-1], RETINA)


def reading_of(dot: np.ndarray | None, flow: np.ndarray | None, paw: np.ndarray) -> np.ndarray:
    """The readings: the retina over the dot, the fovea readout (the dot's centroid centred on the
    sill, its flow, its mass), and the paw's position, centred."""
    out = np.zeros(READINGS)
    if dot is not None:
        out[:RETINA] = retina(dot)
        out[FOVEA : FOVEA + 2] = 2.0 * np.asarray(dot) - 1.0
        if flow is not None:
            out[FLOW : FLOW + 2] = np.clip(np.asarray(flow) / SCHEDULE["dot_speed"], -4.0, 4.0)
        out[MASS] = 1.0
    out[PAW : PAW + 2] = 2.0 * np.asarray(paw) - 1.0
    return out


def paw_of(reading: np.ndarray) -> np.ndarray:
    return np.clip((np.asarray(reading)[..., PAW : PAW + 2] + 1.0) / 2.0, 0.0, 1.0)


def dot_of(reading: np.ndarray) -> np.ndarray:
    return np.clip((np.asarray(reading)[..., FOVEA : FOVEA + 2] + 1.0) / 2.0, 0.0, 1.0)


def task_cost(reading: np.ndarray, action: np.ndarray) -> np.ndarray:
    """What the sill is for: the paw on the dot when there is one (the mass says how much of
    one), with a small price on the push."""
    reading, action = np.asarray(reading), np.asarray(action)
    mass = np.clip(reading[..., MASS], 0.0, 1.0)
    return mass * ((paw_of(reading) - dot_of(reading)) ** 2).sum(axis=-1) + EFFORT * (action**2).sum(axis=-1)


SCHEDULE: dict[str, Any] = {
    "decisions": 4000,
    "dot_gap": [40, 140],  # decisions between a dot's vanishing and the next appearance, uniform
    "dot_timeout": 150,  # an uncaught dot vanishes after this many decisions: a miss
    "dot_speed": 0.015,  # box units per decision
    "turn": 0.35,  # the walk's heading noise per decision (radians)
    "kinds": {"walk": 0.5, "curve": 0.3, "jitter": 0.2},
    "jitter_length": [8, 20],
    "change_at": 2400,  # the moment the dot's law changes for good
    "change": "gravity",  # gravity | faster | wrap | none
}
LAWS = ("gravity", "faster", "wrap")

KINDS = ("walk", "curve", "jitter")


def draw_events(rng: np.random.Generator, schedule: dict[str, Any]) -> list[dict[str, Any]]:
    """The dots of a life, drawn once per seed: for each, the gap after the previous dot's
    vanishing, its kind, its start, its heading, its curvature and (for a jitter) its length."""
    names = list(schedule["kinds"])
    probs = np.array([schedule["kinds"][k] for k in names], dtype=float)
    probs /= probs.sum()
    dots = []
    for _ in range(schedule["decisions"] // 20 + 4):  # more than a life can consume
        kind = names[int(rng.choice(len(names), p=probs))]
        dots.append(
            {
                "gap": int(rng.integers(schedule["dot_gap"][0], schedule["dot_gap"][1] + 1)),
                "kind": kind,
                "start": [float(v) for v in rng.uniform(0.1, 0.9, 2)],
                "heading": float(rng.uniform(0.0, 2 * np.pi)),
                "curve": float(rng.choice([-1.0, 1.0]) * rng.uniform(0.05, 0.15)),
                "length": int(rng.integers(schedule["jitter_length"][0], schedule["jitter_length"][1] + 1)),
            }
        )
    return dots


class World:
    """The sill. ``step(a)`` executes one push of the paw and returns the next reading and what
    happened at that decision. ``source`` is ``schedule`` (the drawn dots) or ``mouse`` (the
    page's pointer is the dot)."""

    def __init__(self, schedule: dict[str, Any], events: list[dict[str, Any]], rng: np.random.Generator, source: str = "schedule") -> None:
        self.schedule, self.events, self.rng, self.source = schedule, list(events), rng, source
        self.paw = np.array([0.5, 0.5])
        self.dot: np.ndarray | None = None
        self.previous: np.ndarray | None = None  # the dot a decision ago, for the flow
        self.vel = np.zeros(2)
        self.kind = "walk"
        self.curve = 0.0
        self.length = 0
        self.t = 0
        self.hold = 0
        self.age = 0
        self.wait = self.events[0]["gap"] if self.events else 10**9
        self.index = 0
        self.law = "routine"
        self.changed = False
        self.mouse: np.ndarray | None = None
        self.mouse_caught: np.ndarray | None = None
        self.catches = 0
        self.misses = 0
        self.dots = 0
        self.pending: list[str] = []
        self.script_left = 0  # the demo: scheduled dots run for this many decisions in mouse mode

    @property
    def present(self) -> bool:
        return self.dot is not None

    @property
    def speed(self) -> float:
        return float(self.schedule["dot_speed"]) * (FASTER if self.law == "faster" else 1.0)

    @property
    def flow(self) -> np.ndarray | None:
        if self.dot is None or self.previous is None:
            return None
        return self.dot - self.previous

    def reading(self) -> np.ndarray:
        return reading_of(self.dot, self.flow, self.paw)

    def fire(self, event: str) -> None:
        self.pending.append(event)

    def _spawn(self, event: dict[str, Any]) -> None:
        self.dot = np.array(event["start"], dtype=float)
        self.previous = None
        h = float(event["heading"])
        self.vel = self.speed * np.array([np.cos(h), np.sin(h)])
        self.kind, self.curve, self.length = event["kind"], float(event["curve"]), int(event["length"])
        self.age, self.hold = 0, 0
        self.dots += 1

    def _vanish(self) -> None:
        self.dot = None
        self.previous = None
        self.hold = 0
        if self.source == "schedule" or self.script_left > 0:
            self.index += 1
            self.wait = self.events[self.index]["gap"] if self.index < len(self.events) else 10**9

    def _advance(self) -> None:
        """The dot's law: a walk (heading noise), a curve (constant turning), a jitter (a walk
        with jumps); reflection at the walls. The changed laws: ``gravity`` pulls the velocity
        down every decision (the dot falls and bounces), ``faster`` multiplies the speed, ``wrap``
        carries the dot through a wall to the opposite side."""
        assert self.dot is not None
        s = self.schedule
        rot = self.curve if self.kind == "curve" else float(self.rng.normal(0.0, s["turn"]))
        c, sn = np.cos(rot), np.sin(rot)
        v = np.array([c * self.vel[0] - sn * self.vel[1], sn * self.vel[0] + c * self.vel[1]])
        if self.law == "gravity":
            v = v + np.array([0.0, -GRAVITY])
            speed = float(np.linalg.norm(v))
            if speed > VMAX:
                v = v / speed * VMAX
        else:
            v = v / max(float(np.linalg.norm(v)), 1e-9) * self.speed
        step = v.copy()
        if self.kind == "jitter":
            step = step + self.rng.normal(0.0, 0.03, 2)
        p = self.dot + step
        for axis in range(2):
            if p[axis] < 0.0 or p[axis] > 1.0:
                if self.law == "wrap":
                    p[axis] = p[axis] % 1.0
                else:
                    p[axis] = -p[axis] if p[axis] < 0.0 else 2.0 - p[axis]
                    v[axis] = -v[axis]
        self.vel = v
        self.previous = self.dot
        self.dot = np.clip(p, 0.0, 1.0)

    def step(self, a: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        s = self.schedule
        flags = {"appear": False, "catch": False, "miss": False, "change": False, "vanish": False, "present": False}
        if s["change"] != "none" and self.t == s["change_at"] and self.source == "schedule":
            self.pending.append(s["change"])
        pending, self.pending = self.pending, []
        for event in pending:
            if event in LAWS:
                self.law, self.changed, flags["change"] = event, True, True
            elif event == "restore":
                self.law, self.changed = "routine", False
            elif event == "script":
                self.script_left = 600
                if self.dot is None:
                    self.wait = min(self.wait, 5)
        # the paw
        a = np.clip(np.asarray(a, dtype=float), -1.0, 1.0)
        self.paw = np.clip(self.paw + PAW_SPEED * a, 0.0, 1.0)
        # the dot
        if self.source == "mouse" and self.script_left <= 0:
            if self.mouse is None:
                self.dot = None
                self.previous = None
            else:
                if self.mouse_caught is not None and np.linalg.norm(self.mouse - self.mouse_caught) > 0.03:
                    self.mouse_caught = None
                if self.mouse_caught is None:
                    if self.dot is None:
                        flags["appear"] = True
                        self.dots += 1
                        self.age, self.hold = 0, 0
                        self.previous = None
                    else:
                        self.previous = self.dot
                    self.dot = self.mouse.copy()
                else:
                    self.dot = None
                    self.previous = None
        else:
            if self.script_left > 0:
                self.script_left -= 1
            if self.dot is not None:
                self._advance()
                self.age += 1
                if self.kind == "jitter" and self.age >= self.length:
                    self._vanish()
                    flags["vanish"] = True
                elif self.age >= s["dot_timeout"]:
                    self._vanish()
                    self.misses += 1
                    flags["miss"] = flags["vanish"] = True
            elif self.source == "schedule" or self.script_left > 0:
                self.wait -= 1
                if self.wait <= 0 and self.index < len(self.events):
                    self._spawn(self.events[self.index])
                    flags["appear"] = True
        # the catch
        if self.dot is not None:
            if np.linalg.norm(self.paw - self.dot) <= CATCH_RADIUS:
                self.hold += 1
            else:
                self.hold = 0
            if self.hold >= CATCH_HOLD:
                self.catches += 1
                flags["catch"] = flags["vanish"] = True
                if self.source == "mouse" and self.script_left <= 0:
                    self.mouse_caught = None if self.mouse is None else self.mouse.copy()
                    self.dot = None
                    self.hold = 0
                else:
                    self._vanish()
        flags["present"] = self.dot is not None
        self.t += 1
        return self.reading(), flags


# ----------------------------------------------------------------------------- the cortex
def make_patch(seed: int, belief: int = 32, encoded: int = 24) -> BeliefPatch:
    port = StructuredPort(READINGS, [DenseBlock(0, READINGS, encoded)])
    return BeliefPatch(port, actions=ACTIONS, belief=belief, outputs=OUTPUTS, iterations=2, damping=0.5, cells=512, active=8, record_width=16, seed=seed)


def moment_macs(patch: BeliefPatch) -> int:
    """Multiply-accumulates of one moment of the patch: the port, the transition and gate,
    the store's code and read per iteration, the repair map, the readout."""
    za = patch.belief + patch.actions
    fi = patch.belief + patch.encoded + patch.belief + patch.record_width + 1
    store = (patch.encoded + patch.belief) * patch.records.cells + patch.records.active * patch.record_width
    return int(patch.inputs * patch.encoded + 2 * patch.belief * za + patch.iterations * (store + patch.belief * fi) + store + patch.outputs * patch.belief)


def held_pushes(rng: np.random.Generator, world: World, n: int, pursuit: float = 0.35) -> np.ndarray:
    """Random pushes of the paw held for one to eight decisions each; a share of the spans
    push toward the dot (a noisy pursuit), so the corpus holds chases and catches. Returns
    the pushes as they are executed (the pursuit reads the world at each span's start)."""
    out = np.empty((n, ACTIONS))
    t = 0
    chase = False
    while t < n:
        span = int(rng.integers(1, 9))
        chase = rng.random() < pursuit
        out[t : t + span] = rng.uniform(-1.0, 1.0, 2) if not chase else np.array([np.nan, np.nan])
        t += span
    return out


def rollout(world: World, pushes: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Readings, executed pushes and the change of the reading that followed each. A NaN push
    is a pursuit: the direction to the dot with noise (a random push when there is none)."""
    r = world.reading()
    o, a, y = [], [], []
    current = np.zeros(2)
    for value in pushes:
        if np.isnan(value[0]):
            if world.dot is not None:
                d = world.dot - world.paw
                current = np.clip(d / max(np.linalg.norm(d), 1e-6) * rng.uniform(0.6, 1.0) + rng.normal(0.0, 0.15, 2), -1.0, 1.0)
            else:
                current = rng.uniform(-1.0, 1.0, 2)
        else:
            current = value
        r2, _ = world.step(current)
        o.append(r)
        a.append(np.clip(current, -1.0, 1.0))
        y.append(target_of(r, r2))
        r = r2
    return np.array(o), np.array(a), np.array(y)


def target_of(r: np.ndarray, r2: np.ndarray) -> np.ndarray:
    """The raw change of the predicted readings (centroid, mass, paw) between two moments."""
    return np.asarray(r2)[..., PREDICTED] - np.asarray(r)[..., PREDICTED]


def motion_scale(y: np.ndarray) -> np.ndarray:
    """The standard deviation of each predicted change over the moments without a jump (a dot
    appearing or vanishing), floored: the units of the targets and of the surprise, so the
    motion of a dot and the paw's push have unit curvature and a jump is a bounded outlier."""
    flat = y.reshape(-1, OUTPUTS)
    jump = np.abs(flat[:, 2]) > 0.5
    return np.maximum(flat[~jump].std(axis=0), 0.02)


STREAM = 400


def corpus(seed: int, episodes: int, length: int, schedule: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random pushes on the routine sill with dots at the life's rates: ``episodes`` streams of
    ``STREAM`` decisions, chopped into chunks of ``length``."""
    rng = np.random.default_rng(seed)
    routine = {**schedule, "change": "none", "decisions": STREAM}
    os_, as_, ys = [], [], []
    for _ in range(episodes):
        world = World(routine, draw_events(rng, routine), rng)
        world.paw = rng.uniform(0.05, 0.95, 2)
        world.wait = int(rng.integers(1, 40))
        o, a, y = rollout(world, held_pushes(rng, world, STREAM), rng)
        for start in range(0, STREAM - length + 1, length):
            os_.append(o[start : start + length])
            as_.append(a[start : start + length])
            ys.append(y[start : start + length])
    return np.stack(os_), np.stack(as_), np.stack(ys)


def one_step_grade(patch: BeliefPatch, o: np.ndarray, a: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Held-out one-step read against persistence, for the retina and the paw, and the
    open-loop imagination at 1, 2, 4 and 8 decisions from the live belief."""
    scratch = BeliefPatch.restore(patch.snapshot())
    scratch.reset()
    path = scratch.assimilate(o, a)
    err = ((path.output - y) ** 2).sum(axis=(0, 1))
    base = (y**2).sum(axis=(0, 1))
    explained = 1.0 - err / np.maximum(base, 1e-12)
    groups = {"position": float(1.0 - err[:2].sum() / max(base[:2].sum(), 1e-12)), "mass": float(1.0 - err[2] / max(base[2], 1e-12)), "paw": float(1.0 - err[3:].sum() / max(base[3:].sum(), 1e-12))}
    present = o[..., MASS] > 0.5
    jump = np.abs(y[..., 2]) >= 1.0
    for name, sel in (("position_motion", present & ~jump), ("position_jump", jump), ("quiet", ~present & ~jump)):
        if sel.any():
            e2 = ((path.output[sel, :2] - y[sel, :2]) ** 2).sum()
            b2 = (y[sel, :2] ** 2).sum()
            groups[name] = float(1.0 - e2 / max(b2, 1e-12))
            groups[name + "_mse"] = float(e2 / (sel.sum() * 2))
            groups[name + "_share"] = float(sel.mean())
    horizons = (1, 2, 4, 8)
    rows: dict[int, list[tuple[float, float]]] = {h: [] for h in horizons}
    for t in range(1, o.shape[1] - max(horizons), 4):
        boundary = path.belief[:, t - 1]
        imagined = scratch.imagine(a[:, t : t + max(horizons)], state=boundary)
        drift = np.cumsum(imagined.output[:, :, :2], axis=1)
        truth = np.cumsum(y[:, t : t + max(horizons), :2], axis=1)
        for h in horizons:
            rows[h].append((float(((drift[:, h - 1] - truth[:, h - 1]) ** 2).sum()), float((truth[:, h - 1] ** 2).sum())))
    return {"explained_one_step": groups, "explained_per_channel_mean": float(explained.mean()), "imagination": {str(h): 1.0 - sum(m for m, _ in r) / max(sum(p for _, p in r), 1e-12) for h, r in rows.items()}, "loss": float(path.loss) if path.loss is not None else None}


HIDDEN = 0.3


def hidden_moments(rng: np.random.Generator, t: int, share: float) -> np.ndarray:
    """An ``observed`` mask for ``observe``: spans without evidence, so the transition must carry
    the belief; the imagination loss within the library's API."""
    m = np.ones(t, dtype=bool)
    k = 1
    while k < t:
        if rng.random() < share:
            span = int(rng.integers(2, 9))
            m[k : k + span] = False
            k += span
        k += 1
    return m


def routine_surprise(patch: BeliefPatch, ho: np.ndarray, ha: np.ndarray, hy: np.ndarray) -> dict[str, float]:
    """The pretrained belief's surprise and residual on held-out routine streams: the floors
    the governors compare with. The surprise is the mean over channels of the squared
    standardized error; its median over moments (mostly quiet) and mean (dots included)."""
    scratch = BeliefPatch.restore(patch.snapshot())
    scratch.reset()
    path = scratch.assimilate(ho, ha)
    s = ((path.output - hy) ** 2).mean(axis=-1)
    present = ho[..., MASS] > 0.5
    return {"median": float(np.median(s)), "mean": float(s.mean()), "quiet_median": float(np.median(s[~present])) if (~present).any() else float(np.median(s)), "dot_median": float(np.median(s[present])) if present.any() else float("nan"), "residual_median": float(np.median(path.residual)), "present_share": float(present.mean())}


def pretrain(seed: int, schedule: dict[str, Any], *, episodes: int = 100, length: int = 32, batch: int = 32, rate: float = 0.3, epochs: int = 40, belief: int = 32, encoded: int = 24, fill_store: bool = True, log: bool = True, progress: Any = None) -> tuple[BeliefPatch, dict[str, Any]]:
    """The belief learns the routine sill from random and pursuing pushes: ``observe`` on
    batches of chunks; standardized targets; every other epoch hides spans of moments."""
    o, a, y = corpus(seed, episodes, length, schedule)
    ho, ha, hy = corpus(seed + 500, 30, length, schedule)
    scale = motion_scale(y)
    y, hy = np.clip(y / scale, -CLIP, CLIP), np.clip(hy / scale, -CLIP, CLIP)
    patch = make_patch(seed, belief=belief, encoded=encoded)
    rng = np.random.default_rng(seed + 7)
    curve = []
    t0 = time.perf_counter()
    for epoch in range(epochs):
        order = rng.permutation(len(o))
        losses = []
        for i in range(0, len(order), batch):
            idx = order[i : i + batch]
            patch.reset()
            mask = hidden_moments(rng, length, HIDDEN) if epoch % 2 == 1 else None
            result = patch.observe(o[idx], a[idx], y[idx], rate=rate, write=False, observed=mask)
            if result.initial_loss is not None:
                losses.append(result.initial_loss)
        if epoch % 5 == 4 or epoch == epochs - 1:
            grade = one_step_grade(patch, ho, ha, hy)
            curve.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), **grade, "seconds": time.perf_counter() - t0})
            if log:
                print(json.dumps(curve[-1]), flush=True)
        if progress is not None:
            progress(epoch + 1, float(np.mean(losses)), curve[-1] if curve else None)
    without_store = one_step_grade(patch, ho, ha, hy)
    if fill_store:
        for i in range(0, len(o), batch):
            patch.reset()
            patch.observe(o[i : i + batch], a[i : i + batch], y[i : i + batch], rate=0.0, write=True)
    patch.reset()
    grade = one_step_grade(patch, ho, ha, hy)
    floors = routine_surprise(patch, ho, ha, hy)
    return patch, {"seed": seed, "episodes": episodes, "length": length, "batch": batch, "rate": rate, "epochs": epochs, "belief": belief, "encoded": encoded, "clip": CLIP, "fill_store": fill_store, "scale": [float(v) for v in scale], "curve": curve, "without_store": without_store, "with_store": grade, "routine": floors, "parameters": int(sum(v.size for v in patch.parameters().values())), "moment_macs": moment_macs(patch)}


# ----------------------------------------------------------------------------- the habit
def habit_act(h: dict[str, Any], r: np.ndarray) -> np.ndarray:
    """The habit: the paw drifts toward home; its rest position and drift gain are genes."""
    paw = paw_of(np.asarray(r))
    home = np.array([h["home_x"], h["home_y"]])
    return np.clip(float(h["drift"]) * (home - paw) / PAW_SPEED * 0.1, -1.0, 1.0)


HABIT_KEYS = ("home_x", "home_y", "drift")


def random_starts(rng: np.random.Generator, n: int) -> np.ndarray:
    """Readings from random paws, a dot present in half of them."""
    return np.stack([reading_of(rng.uniform(0.05, 0.95, 2) if rng.random() < 0.5 else None, rng.normal(0.0, SCHEDULE["dot_speed"], 2), rng.uniform(0.05, 0.95, 2)) for _ in range(n)])


def imagined_habit_cost(patch: BeliefPatch, h: dict[str, Any], starts_o: np.ndarray, starts_z: np.ndarray, steps: int, scale: np.ndarray) -> tuple[float, int]:
    r, z = starts_o.copy(), starts_z
    total = 0.0
    for _ in range(steps):
        a = habit_act(h, r)
        path = patch.imagine(a[:, None, :], state=z)
        r[:, PREDICTED] += path.output[:, 0] * scale
        z = path.final_state
        total += float(np.mean(task_cost(r, a)))
    return total / steps, len(starts_o) * steps


def fit_habit(patch: BeliefPatch, rng: np.random.Generator, scale: np.ndarray, start: dict[str, Any], *, starts: int = 8, steps: int = 12) -> tuple[dict[str, Any], float, int]:
    """A pattern search over the habit's three genes, every trial evaluated in imagination
    only, from a scratch copy of the patch."""
    scratch = BeliefPatch.restore(patch.snapshot())
    scratch.reset()
    o0 = random_starts(rng, starts)
    z0 = scratch.assimilate(o0[:, None], np.zeros((starts, 1, ACTIONS))).final_state
    h = {k: float(start[k]) for k in HABIT_KEYS}
    best, moments = imagined_habit_cost(scratch, h, o0, z0, steps, scale)
    step = 0.25
    while step > 4e-3:
        improved = False
        for k in HABIT_KEYS:
            for sign in (1.0, -1.0):
                trial = dict(h)
                trial[k] = float(np.clip(h[k] + sign * step, 0.0, 1.0 if k != "drift" else 2.0))
                cost, m = imagined_habit_cost(scratch, trial, o0, z0, steps, scale)
                moments += m
                if cost < best:
                    h, best, improved = trial, cost, True
        step = step if improved else step / 2
    return h, best, moments


# ----------------------------------------------------------------------------- the genomes
# The learning machinery every arm shares (hand-set constants; candidate genes, see REPORT.md).
MACHINERY: dict[str, Any] = {
    "window": 96,  # decisions of the executed window that learning observes
    "learn_rate": 0.1,
    "passes": 6,
    "horizon": 6,  # decisions imagined per candidate push
    "spread": 1.0,  # the candidates' magnitude: the habit's push offset by this times the eight directions
    "hold": True,  # the candidate push is held for the horizon (False: the habit closes the loop)
    "rollback": True,
    "habituate": 2.0,
    "refit": True,
    "min_cooldown": 32,  # decisions after a learn call before another may run, for every governor
    "write": False,
}

HABIT_HAND_SET: dict[str, Any] = {"home_x": 0.5, "home_y": 0.5, "drift": 0.3}
HABIT_SPACE: dict[str, tuple[Any, ...]] = {"home_x": ("linear", 0.1, 0.0, 1.0), "home_y": ("linear", 0.1, 0.0, 1.0), "drift": ("log", 0.3, 0.02, 2.0)}
BASELINE_HAND_SET: dict[str, Any] = {"baseline_rate": 0.02, "floor": 1.0}
BASELINE_SPACE: dict[str, tuple[Any, ...]] = {"baseline_rate": ("log", 0.3, 0.001, 0.5), "floor": ("log", 0.3, 0.1, 3.0)}

THRESHOLD_HAND_SET: dict[str, Any] = {
    **HABIT_HAND_SET,
    **BASELINE_HAND_SET,
    "k_imagine": 8.0,  # a spike: the surprise above this many baselines recruits imagination
    "imagine_budget": 12.0,
    "k_learn": 8.0,  # persistence: the surprise above this many baselines (a routine chase runs at three to four)
    "persist": 96.0,
    "persist_share": 0.25,  # for at least this share of the last ``persist`` decisions (a dot is present about a third of the time)
    "renew": True,
    "cooldown": 64.0,
}
THRESHOLD_SPACE: dict[str, tuple[Any, ...]] = {
    **HABIT_SPACE,
    **BASELINE_SPACE,
    "k_imagine": ("log", 0.3, 1.5, 100.0),
    "imagine_budget": ("log", 0.3, 1.0, 40.0),
    "k_learn": ("log", 0.3, 1.2, 30.0),
    "persist": ("log", 0.3, 4.0, 200.0),
    "persist_share": ("linear", 0.1, 0.5, 1.0),
    "renew": ("choice", True, False),
    "cooldown": ("log", 0.3, 32.0, 512.0),
}

READBACK = ("fast", "slow", "residual", "mode_habit", "mode_imagine", "mode_learn", "one")
CORTEX = 4
MOTOR = ("habit", "imagine", "learn")


def _patch_hand_set() -> dict[str, Any]:
    """The hand-designed wiring: the threshold rule written as synapses. A spike drives the
    imagine unit; the current mode feeds itself back (hysteresis); the slow average drives the
    learn unit; the habit unit rests on a bias; the motor units compete."""
    g: dict[str, Any] = {**HABIT_HAND_SET, **BASELINE_HAND_SET, "slow_rate": 0.02, "warm": False, "lateral_cortex": -0.5, "lateral_motor": -1.0}
    for i in range(len(READBACK)):
        for j in range(CORTEX):
            g[f"rc_{i}_{j}"] = 0.0
        for k in range(len(MOTOR)):
            g[f"rm_{i}_{k}"] = 0.0
    for j in range(CORTEX):
        g[f"cb_{j}"] = 0.0
        for k in range(len(MOTOR)):
            g[f"cm_{j}_{k}"] = 0.0
    for k in range(len(MOTOR)):
        g[f"mb_{k}"] = 0.0
    g["rm_0_1"] = 1.0  # fast surprise -> imagine: a spike of eight baselines drives 2.2
    g["rm_4_1"] = 0.4  # imagine -> imagine: the chase holds while the surprise stays a few baselines up (at 0.6 a quiet field after a catch, half a baseline, kept the imagine unit up on its own)
    g["rm_6_0"] = 0.9  # one -> habit: the rest state
    g["rm_1_2"] = 1.5  # slow surprise -> learn: the routine's average log-ratio is about 0.5, a changed law's about 1.4
    g["rm_6_2"] = -0.9  # one -> learn: a bias against, so the routine does not learn
    return g


PATCH_HAND_SET: dict[str, Any] = _patch_hand_set()
PATCH_SPACE: dict[str, tuple[Any, ...]] = {
    **HABIT_SPACE,
    **BASELINE_SPACE,
    "slow_rate": ("log", 0.4, 0.002, 0.3),
    "warm": ("choice", True, False),
    "lateral_cortex": ("linear", 0.3, -2.0, 0.0),
    "lateral_motor": ("linear", 0.3, -3.0, 0.0),
    **{k: ("linear", 0.3, -2.5, 2.5) for k in PATCH_HAND_SET if k.startswith(("rc_", "rm_", "cm_", "cb_", "mb_"))},
}

SPACES = {"patch": PATCH_SPACE, "threshold": THRESHOLD_SPACE}
HAND_SETS = {"patch": PATCH_HAND_SET, "threshold": THRESHOLD_HAND_SET}


def random_genome(rng: np.random.Generator, space: dict[str, tuple[Any, ...]], base: dict[str, Any]) -> dict[str, Any]:
    """A genome drawn uniformly over the space (log-uniform for log genes): the random-search control."""
    out = dict(base)
    for name, spec in space.items():
        if spec[0] == "log":
            out[name] = float(np.exp(rng.uniform(np.log(spec[2]), np.log(spec[3]))))
        elif spec[0] == "linear":
            out[name] = float(rng.uniform(spec[2], spec[3]))
        elif spec[0] == "int":
            out[name] = int(rng.integers(spec[1], spec[2] + 1))
        else:
            out[name] = spec[1 + int(rng.integers(len(spec) - 1))]
    return out


# ----------------------------------------------------------------------------- the governor patch
GOVERNOR_MODEL = NeuronModel(dt=0.5, slope=2.0, threshold=0.5, gain=1.0, stimulus_amplitude=1.0)


class GovernorPatch:
    """A settling patch of the same kind as every other cadence brain: a readback port of seven
    units, a cortex of four, a motor of three. Every synapse is a gene; the settled motor
    state is the mode. Nothing in it learns within a life."""

    def __init__(self, genome: dict[str, Any]) -> None:
        self.g = dict(genome)
        nr, nc, nm = len(READBACK), CORTEX, len(MOTOR)
        self.readback = np.arange(0, nr)
        self.cortex = np.arange(nr, nr + nc)
        self.motor = np.arange(nr + nc, nr + nc + nm)
        self.n = nr + nc + nm
        pre, post, w = [], [], []
        g = self.g
        for i in range(nr):
            for j in range(nc):
                if g[f"rc_{i}_{j}"] != 0.0:
                    pre.append(self.readback[i]), post.append(self.cortex[j]), w.append(float(g[f"rc_{i}_{j}"]))
            for k in range(nm):
                if g[f"rm_{i}_{k}"] != 0.0:
                    pre.append(self.readback[i]), post.append(self.motor[k]), w.append(float(g[f"rm_{i}_{k}"]))
        for j in range(nc):
            for k in range(nm):
                if g[f"cm_{j}_{k}"] != 0.0:
                    pre.append(self.cortex[j]), post.append(self.motor[k]), w.append(float(g[f"cm_{j}_{k}"]))
            for j2 in range(nc):
                if j != j2 and g["lateral_cortex"] != 0.0:
                    pre.append(self.cortex[j]), post.append(self.cortex[j2]), w.append(float(g["lateral_cortex"]))
        for k in range(nm):
            for k2 in range(nm):
                if k != k2 and g["lateral_motor"] != 0.0:
                    pre.append(self.motor[k]), post.append(self.motor[k2]), w.append(float(g["lateral_motor"]))
        self.table = {(int(p), int(q)): float(v) for p, q, v in zip(pre, post, w, strict=True)}
        self.connectome = Connectome.from_synapses(self.n, pre=pre, post=post, sign=w, populations={"readback": self.readback, "cortex": self.cortex, "motor": self.motor}, label="governor")
        bias = np.zeros(self.n)
        bias[self.cortex] = [float(g[f"cb_{j}"]) for j in range(nc)]
        bias[self.motor] = [float(g[f"mb_{k}"]) for k in range(nm)]
        self.brain = Brain(self.connectome, GOVERNOR_MODEL, bias=bias)
        self.synapses = int(self.connectome.synapses)
        self.state = None
        self.activation = np.zeros(self.n)

    def settle(self, readback: np.ndarray) -> tuple[str, int]:
        """The mode from the settled state under the readback as the drive on the port; the
        steps taken are the patch's cost."""
        drive = np.zeros((1, self.n))
        drive[0, self.readback] = readback
        result = self.brain.equilibrate(drive, budget=200, chunk=10, tolerance=1e-3, state=self.state if self.g.get("warm") else None)
        self.state = result.state
        self.activation = result.state.activation[0].copy()
        motor = self.activation[self.motor]
        mode = MOTOR[int(np.argmax(motor))] if motor.max() > 1e-6 else "habit"
        return mode, int(result.steps)


# ----------------------------------------------------------------------------- the life
MODES = {"habit": 0, "imagine": 1, "learn": 2}
ARMS = ("patch", "threshold", "always_awake", "never_wakes")
DIRECTIONS = np.array([[0.0, 0.0]] + [[np.cos(t), np.sin(t)] for t in np.arange(8) * np.pi / 4])


class Life:
    """One continuing life: the governor names the mode from the brain's own signals, the paw
    acts by the habit or by imagination, the belief assimilates the moment, the surprise is
    measured; learning observes the executed window when the governor says so."""

    def __init__(self, patch: BeliefPatch, world: World, genome: dict[str, Any], scale: np.ndarray, floors: dict[str, float], *, arm: str = "patch", seed: int = 0, keep_log: bool = True, machinery: dict[str, Any] | None = None) -> None:
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}")
        self.patch, self.world, self.g, self.arm = patch, world, dict(genome), arm
        self.m = {**MACHINERY, **(machinery or {})}
        self.scale = np.asarray(scale, dtype=float)
        self.rng = np.random.default_rng(seed + 12345)
        self.habit = {k: float(self.g[k]) for k in HABIT_KEYS}
        self.baseline0 = float(floors["median"])
        self.residual0 = max(float(floors.get("residual_median", 1.0)), 1e-6)
        self.floor = float(self.g["floor"]) * self.baseline0
        self.baseline = max(self.baseline0, self.floor)
        self.governor = GovernorPatch(self.g) if arm == "patch" else None
        self.macs = moment_macs(patch)
        self.t = 0
        self.mode = "habit"
        self.imagine_left = 0
        self.above = 0
        self.recent: list[float] = []
        self.cooldown_left = 0
        self.since_learn = 10**9
        self.slow = 0.0
        self.surprise_last = 0.0
        self.residual_last = 0.0
        self.readback_last = np.zeros(len(READBACK))
        self.o: list[np.ndarray] = []
        self.a: list[np.ndarray] = []
        self.y: list[np.ndarray] = []
        self.states: list[np.ndarray | None] = []
        self.learns: list[dict[str, Any]] = []
        self.dots: list[dict[str, Any]] = []  # per dot: appearance, catch or miss, wake latency
        self.keep_log = keep_log
        self.log: dict[str, list[Any]] = {k: [] for k in ("mode", "residual", "surprise", "baseline", "moments", "governor_steps", "ms", "present", "catch", "miss", "appear", "change", "cost", "distance")}
        self.totals = {"assimilate": 0, "imagine": 0, "observe": 0, "refit": 0, "governor": 0.0, "imagine_calls": 0, "learn_calls": 0, "kept": 0, "undone": 0, "ms": {"habit": 0.0, "imagine": 0.0, "learn": 0.0}, "decisions": {"habit": 0, "imagine": 0, "learn": 0}}
        self.patch.reset()

    # -- the readback port: what both governors read. ``fast`` is the last surprise over the
    #    baseline, log-compressed; ``slow`` a running average of the same at the genome's rate
    #    (a single spike moves it a little, a changed law moves it for good); the residual over
    #    its routine median; the last mode; a constant.
    def readback(self) -> np.ndarray:
        one_hot = np.eye(3)[MODES[self.mode]]
        return np.array([np.log1p(self.surprise_last / self.baseline), self.slow, self.residual_last / self.residual0, *one_hot, 1.0])

    # -- the three modes
    def _imagine(self, r: np.ndarray, z: np.ndarray | None) -> tuple[np.ndarray, int]:
        """Ten candidate pushes (rest, the eight directions at ``spread``, and the habit's own),
        each held for the horizon in the belief's private imagination; the imagined cost is
        the task's; the best first push is returned. (A first version centred the eight
        directions on the habit's push, as the room does: once a refit had made the habit a
        strong drift home, every candidate pointed toward home and the chase could not leave.)"""
        h = int(self.m["horizon"])
        candidates = np.clip(np.concatenate([float(self.m["spread"]) * DIRECTIONS, habit_act(self.habit, r)[None]]), -1.0, 1.0)
        n = len(candidates)
        zz = np.zeros((n, self.patch.belief)) if z is None else np.repeat(z, n, axis=0)
        rr = np.repeat(r[None], n, axis=0)
        a = candidates.copy()
        cost = np.zeros(n)
        for _ in range(h):
            path = self.patch.imagine(a[:, None, :], state=zz)
            rr[:, PREDICTED] += path.output[:, 0] * self.scale
            zz = path.final_state
            cost += task_cost(rr, a)
            if not self.m["hold"]:
                a = habit_act(self.habit, rr)
        return candidates[int(np.argmin(cost))], n * h

    def _learn(self) -> dict[str, Any]:
        """``observe`` on the executed window from its boundary, for ``passes``; the validity is
        whether the window's loss fell; an invalid update is undone."""
        m = self.m
        w = int(min(round(m["window"]), len(self.a)))
        i0, i1 = len(self.a) - w, len(self.a)
        o = np.array(self.o[i0:i1])[None]
        a = np.array(self.a[i0:i1])[None]
        y = np.array(self.y[i0:i1])[None]
        snapshot = self.patch.snapshot()
        before = self.patch.parameters()
        boundary = self.states[i0]
        loss0, last_loss, moments, passes_kept = None, None, 0, 0
        for _ in range(int(m["passes"])):
            kept_snapshot = self.patch.snapshot()
            result = self.patch.observe(o, a, y, rate=float(m["learn_rate"]), write=bool(m["write"]), state=boundary)
            moments += 3 * w
            if loss0 is None:
                loss0 = result.initial_loss
            if not result.updated:
                break
            if last_loss is not None and result.initial_loss is not None and result.initial_loss > last_loss:
                self.patch = BeliefPatch.restore(kept_snapshot)
                break
            last_loss = result.initial_loss
            passes_kept += 1
        check = self.patch.observe(o, a, y, rate=0.0, write=False, state=boundary)
        moments += w
        loss1 = check.path.loss
        after = self.patch.parameters()
        size = float(np.sqrt(sum(float(np.sum((after[k] - before[k]) ** 2)) for k in after)))
        valid = loss0 is not None and loss1 is not None and loss1 < 0.9 * loss0
        entry = {"t": self.t, "window": w, "loss_before": loss0, "loss_after": loss1, "update_size": size, "valid": bool(valid), "kept": bool(valid or not m["rollback"]), "passes_kept": passes_kept, "refit": False, "habit_before": dict(self.habit), "moments": moments}
        if not valid and m["rollback"]:
            self.patch = BeliefPatch.restore(snapshot)
            self.totals["undone"] += 1
            self.baseline = min(self.baseline * float(m["habituate"]), float(np.median(self.recent)) if self.recent else self.baseline)
        else:
            self.totals["kept"] += 1
            errs = ((check.path.output[0] - y[0]) ** 2).mean(axis=-1)
            self.baseline = max(self.floor, float(np.median(errs)))
            if m["refit"]:
                self.habit, cost, refit_moments = fit_habit(self.patch, self.rng, self.scale, self.habit)
                entry.update({"refit": True, "habit_after": dict(self.habit), "imagined_cost": cost, "refit_moments": refit_moments})
                self.totals["refit"] += refit_moments
        # the live state continues from the window's end under the new parameters (observe set it)
        self.totals["observe"] += moments
        self.totals["learn_calls"] += 1
        self.above = 0
        self.recent = []
        self.slow = 0.0
        self.cooldown_left = int(round(self.g.get("cooldown", 0)))
        self.since_learn = 0
        self.learns.append(entry)
        return entry

    # -- one decision
    def decide(self) -> dict[str, Any]:
        g, m, r = self.g, self.m, self.world.reading()
        z = self.patch.state
        started = time.perf_counter()
        rb = self.readback()
        self.readback_last = rb
        governor_steps = 0
        can_learn = self.since_learn >= int(m["min_cooldown"]) and len(self.a) >= 32 and self.cooldown_left == 0
        learn_now, imagine_now = False, False
        if self.arm == "patch":
            assert self.governor is not None
            want, governor_steps = self.governor.settle(rb)
            learn_now = want == "learn" and can_learn
            imagine_now = want in ("imagine", "learn")
        elif self.arm in ("threshold", "always_awake"):
            persist = int(round(g["persist"]))
            share = float(np.mean(np.array(self.recent) > g["k_learn"] * self.baseline)) if self.recent else 0.0
            learn_now = len(self.recent) >= persist and share >= g["persist_share"] and can_learn
            imagine_now = self.imagine_left > 0 or self.arm == "always_awake"
        entry = self._learn() if learn_now else None
        if entry is not None:
            imagine_now = True
        moments = 0
        if imagine_now:
            action, moments = self._imagine(r, z)
            self.imagine_left = max(0, self.imagine_left - 1)
            mode = "imagine"
        else:
            action, mode = habit_act(self.habit, r), "habit"
        if entry is not None:
            mode = "learn"
        path = self.patch.assimilate(r[None, None], action[None, None])
        expected, residual = path.output[0, 0], float(path.residual[0, 0])
        r2, flags = self.world.step(action)
        y = np.clip(target_of(r, r2) / self.scale, -CLIP, CLIP)
        surprise = float(np.mean((expected - y) ** 2))
        ms = 1000.0 * (time.perf_counter() - started)
        # the executed window
        self.o.append(r)
        self.a.append(action)
        self.y.append(y)
        self.states.append(z)
        if len(self.a) > 512:
            del self.o[0], self.a[0], self.y[0], self.states[0]
        # the threshold rule's bookkeeping (the always-awake control learns by it too)
        if self.arm in ("threshold", "always_awake"):
            spike = surprise > g["k_imagine"] * self.baseline and (self.imagine_left == 0 or bool(g["renew"]))
            if spike:
                self.imagine_left = max(self.imagine_left, int(round(g["imagine_budget"])))
            self.recent.append(surprise)
            del self.recent[: -int(round(g["persist"]))]
            threshold = g["k_learn"] * self.baseline
        else:
            self.recent.append(surprise)
            del self.recent[:-96]
            threshold = 3.0 * self.baseline
        if surprise > threshold:
            self.above += 1
        else:
            self.above = 0
            self.baseline = max(self.floor, self.baseline + g["baseline_rate"] * (surprise - self.baseline))
        rate = float(self.g.get("slow_rate", 0.02))
        self.slow = self.slow + rate * (float(np.log1p(surprise / self.baseline)) - self.slow)
        if self.cooldown_left > 0:
            self.cooldown_left -= 1
        self.since_learn += 1
        self.surprise_last, self.residual_last, self.mode = surprise, residual, mode
        gov_moments = governor_steps * (self.governor.synapses if self.governor is not None else 0) / self.macs
        self.totals["assimilate"] += 1
        self.totals["imagine"] += moments
        self.totals["imagine_calls"] += int(moments > 0)
        self.totals["governor"] += gov_moments
        self.totals["ms"][mode] += ms
        self.totals["decisions"][mode] += 1
        # the dots' ledger
        if flags["appear"]:
            self.dots.append({"appear": self.t, "wake": None, "end": None, "outcome": None})
        if self.dots and self.dots[-1]["end"] is None:
            if mode != "habit" and self.dots[-1]["wake"] is None:
                self.dots[-1]["wake"] = self.t
            if flags["catch"]:
                self.dots[-1].update({"end": self.t, "outcome": "catch"})
            elif flags["miss"]:
                self.dots[-1].update({"end": self.t, "outcome": "miss"})
            elif flags["vanish"]:
                self.dots[-1].update({"end": self.t, "outcome": "vanished"})
        if self.keep_log:
            L = self.log
            L["mode"].append(MODES[mode])
            L["residual"].append(residual)
            L["surprise"].append(surprise)
            L["baseline"].append(self.baseline)
            L["moments"].append(1 + moments + gov_moments + (entry["moments"] + entry.get("refit_moments", 0) if entry else 0))
            L["governor_steps"].append(governor_steps)
            L["ms"].append(ms)
            L["present"].append(bool(self.world.present or flags["present"]))
            L["catch"].append(bool(flags["catch"]))
            L["miss"].append(bool(flags["miss"]))
            L["appear"].append(bool(flags["appear"]))
            L["change"].append(bool(flags["change"]))
            L["cost"].append(float(task_cost(r, action)))
            L["distance"].append(float(np.linalg.norm(self.world.paw - self.world.dot)) if self.world.dot is not None else float("nan"))
        self.t += 1
        return {"mode": mode, "action": action, "residual": residual, "surprise": surprise, "baseline": self.baseline, "expected": expected, "reading": r2, "flags": flags, "learn": entry, "ms": ms, "governor_steps": governor_steps, "readback": rb, "moments": 1 + moments + gov_moments}

    def run(self, decisions: int) -> None:
        for _ in range(decisions):
            self.decide()

    def compute(self) -> dict[str, Any]:
        t = self.totals
        total = t["assimilate"] + t["imagine"] + t["observe"] + t["refit"] + t["governor"]
        n = max(1, sum(t["decisions"].values()))
        return {**{k: v for k, v in t.items() if k not in ("ms", "decisions")}, "total_moments": float(total), "moments_per_decision": total / n, "ms_per_decision": {k: (t["ms"][k] / t["decisions"][k] if t["decisions"][k] else None) for k in t["ms"]}, "decisions_by_mode": dict(t["decisions"]), "wall_ms": sum(t["ms"].values()), "moment_macs": self.macs}


def metrics(life: Life, schedule: dict[str, Any]) -> dict[str, Any]:
    """The measurements: the awake share, the catches, the latencies, the false wakes, the
    learning before and after the change, the surprise term, the compute."""
    L = {k: np.array(v) for k, v in life.log.items()}
    n = len(L["mode"])
    change_at = schedule["change_at"] if schedule["change"] != "none" else n
    awake = L["mode"] != MODES["habit"]
    present = L["present"].astype(bool)
    pre = np.arange(n) < change_at
    dots = [d for d in life.dots if d["end"] is not None]
    caught = [d for d in dots if d["outcome"] == "catch"]
    missed = [d for d in dots if d["outcome"] == "miss"]
    wake_lat = [d["wake"] - d["appear"] for d in dots if d["wake"] is not None]
    catch_lat = [d["end"] - d["appear"] for d in caught]
    transitions = np.flatnonzero(awake[1:] & ~awake[:-1]) + 1
    false_wakes = int(sum(1 for t in transitions if not present[t] and not present[max(0, t - 1)]))
    learn_t = np.array([e["t"] for e in life.learns], dtype=int)
    kept_t = np.array([e["t"] for e in life.learns if e["kept"]], dtype=int)

    def catch_rate(sel: list[dict[str, Any]]) -> float | None:
        counted = [d for d in sel if d["outcome"] in ("catch", "miss")]
        return float(sum(d["outcome"] == "catch" for d in counted) / len(counted)) if counted else None

    out: dict[str, Any] = {
        "decisions": int(n),
        "dots": len(dots),
        "dots_counted": len(caught) + len(missed),
        "catches": len(caught),
        "misses": len(missed),
        "catch_rate": catch_rate(dots),
        "catch_rate_before_change": catch_rate([d for d in dots if d["appear"] < change_at]),
        "catch_rate_after_change": catch_rate([d for d in dots if d["appear"] >= change_at]),
        "catch_latency_mean": float(np.mean(catch_lat)) if catch_lat else None,
        "wake_latency_mean": float(np.mean(wake_lat)) if wake_lat else None,
        "wake_latency_median": float(np.median(wake_lat)) if wake_lat else None,
        "awake_share": float(awake.mean()),
        "awake_share_with_dot": float(awake[present].mean()) if present.any() else None,
        "awake_share_without_dot": float(awake[~present].mean()) if (~present).any() else None,
        "present_share": float(present.mean()),
        "false_wakes": false_wakes,
        "wakes": int(len(transitions)),
        "learn_calls": int(len(learn_t)),
        "kept_updates": int(len(kept_t)),
        "learn_calls_before_change": int((learn_t < change_at).sum()),
        "kept_updates_before_change": int((kept_t < change_at).sum()),
        "learn_calls_after_change": int((learn_t >= change_at).sum()),
        "kept_updates_after_change": int((kept_t >= change_at).sum()),
        "task_cost_mean": float(L["cost"].mean()),
        "surprise_term": float(np.mean(np.minimum(L["surprise"], 1.0))),
        "surprise_mean": float(L["surprise"].mean()),
        "surprise_median_quiet": float(np.median(L["surprise"][~present])) if (~present).any() else None,
        "surprise_median_with_dot": float(np.median(L["surprise"][present])) if present.any() else None,
        "moments_per_decision": float(L["moments"].mean()),
        "governor_steps_mean": float(L["governor_steps"].mean()),
    }
    if schedule["change"] != "none" and change_at < n:
        after = learn_t[learn_t >= change_at]
        out["detection_latency"] = int(after[0] - change_at) if len(after) else None
        seg_before = present & pre
        seg_after = present & ~pre
        out["surprise_with_dot_before_change"] = float(np.median(L["surprise"][seg_before])) if seg_before.any() else None
        out["surprise_with_dot_after_change"] = float(np.median(L["surprise"][seg_after])) if seg_after.any() else None
        last = present & (np.arange(n) >= n - 600)
        out["surprise_with_dot_last_600"] = float(np.median(L["surprise"][last])) if last.any() else None
        out["surprise_term_after_change"] = float(np.mean(np.minimum(L["surprise"][~pre], 1.0)))
    return out


def build(belief_path: Path, seed: int, schedule: dict[str, Any], genome: dict[str, Any], arm: str, scale: np.ndarray, floors: dict[str, float], keep_log: bool = True) -> tuple[Life, list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    events = draw_events(rng, schedule)
    world = World(schedule, events, rng)
    patch = BeliefPatch.load(belief_path)
    return Life(patch, world, genome, scale, floors, arm=arm, seed=seed, keep_log=keep_log), events


def run_life(belief_path: Path, seed: int, schedule: dict[str, Any], genome: dict[str, Any], arm: str, scale: np.ndarray, floors: dict[str, float], keep_log: bool = True) -> tuple[Life, list[dict[str, Any]]]:
    life, events = build(belief_path, seed, schedule, genome, arm, scale, floors, keep_log)
    life.run(schedule["decisions"])
    return life, events


# ----------------------------------------------------------------------------- fitness
PRICE_MOMENTS = 0.004  # per moment-evaluation per decision, in units of the catch rate
PRICE_SURPRISE = 0.5  # per unit of the mean surprise (every channel at weight one, capped at one): a repaired belief is worth about what a learn call costs


def fitness_parts(life: Life, schedule: dict[str, Any]) -> dict[str, float]:
    m = metrics(life, schedule)
    c = life.compute()
    rate = m["catch_rate"] if m["catch_rate"] is not None else 0.0
    return {"catch_rate": rate, "moments_per_decision": c["moments_per_decision"], "surprise_term": m["surprise_term"], "fitness": rate - PRICE_MOMENTS * c["moments_per_decision"] - PRICE_SURPRISE * m["surprise_term"]}


def fitness_of(belief_path: str, schedule: dict[str, Any], scale: list[float], floors: dict[str, float], seeds: list[int], arm: str, genome: dict[str, Any], seed: int) -> float:
    """The catch rate minus the priced compute minus the priced raw surprise, averaged over
    the training lives. The ``seed`` of ``evolve`` is unused: every genome meets the same lives."""
    scores = []
    for s in seeds:
        life, _ = run_life(Path(belief_path), s, schedule, genome, arm, np.array(scale), floors, keep_log=True)
        scores.append(fitness_parts(life, schedule)["fitness"])
    return float(np.mean(scores))


def score_row(belief_path: Path, schedule: dict[str, Any], scale: np.ndarray, floors: dict[str, float], arm: str, genome: dict[str, Any], seed: int) -> dict[str, Any]:
    life, _ = run_life(belief_path, seed, schedule, genome, arm, scale, floors)
    m = metrics(life, schedule)
    return {"seed": seed, **fitness_parts(life, schedule), "awake_share": m["awake_share"], "catches": m["catches"], "misses": m["misses"], "false_wakes": m["false_wakes"], "learn_calls": m["learn_calls"], "kept_updates": m["kept_updates"], "learn_calls_before_change": m["learn_calls_before_change"], "detection_latency": m.get("detection_latency"), "catch_rate_after_change": m["catch_rate_after_change"], "wake_latency_mean": m["wake_latency_mean"]}


def _score_job(args: tuple[Any, ...]) -> dict[str, Any]:
    return score_row(*args)


def score_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"rows": rows, **{k: float(np.mean([r[k] for r in rows])) for k in ("fitness", "catch_rate", "moments_per_decision", "surprise_term", "awake_share")}}


def score_parts(belief_path: Path, schedule: dict[str, Any], scale: np.ndarray, floors: dict[str, float], seeds: list[int], arm: str, genome: dict[str, Any], mapper: Any = map) -> dict[str, Any]:
    rows = list(mapper(_score_job, [(belief_path, schedule, scale, floors, arm, genome, s) for s in seeds]))
    return score_summary(rows)


# ----------------------------------------------------------------------------- receipts
def revision() -> dict[str, Any]:
    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def git(repo: Path) -> str:
        try:
            return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        except Exception:  # noqa: BLE001
            return "unknown"

    return {"library": {"path": str(LIBRARY), "commit": git(LIBRARY), "version": cadence.__version__, "belief_py_sha256": sha(Path(cadence.__file__).with_name("belief.py"))}, "cat_py_sha256": sha(Path(__file__).resolve()), "python": subprocess.run(["python3", "--version"], capture_output=True, text=True).stdout.strip()}


def machine() -> dict[str, Any]:
    load = os.getloadavg()
    cpus = os.cpu_count() or 1
    return {"cpus": cpus, "load_average": [round(v, 1) for v in load], "timing_valid": bool(max(load) < cpus), "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}, "note": "wall-clock numbers are valid only when the load average stayed under the core count; the moment counts are exact"}


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_receipt(path: Path, kind: str, body: dict[str, Any]) -> None:
    from cadence.receipts import canonical_json

    body = _clean({"kind": kind, "written": time.strftime("%Y-%m-%dT%H:%M:%S"), "revision": revision(), "machine": machine(), **body})
    text = canonical_json(body)
    body["receipt_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json.loads(canonical_json(body)), indent=1))


def load_context(belief_path: Path) -> dict[str, Any]:
    return json.loads(belief_path.with_suffix(".json").read_text())


def genome_from(path: str | None, which: str) -> tuple[dict[str, Any], str]:
    if path is None:
        return dict(HAND_SETS[which]), "hand_set"
    body = json.loads(Path(path).read_text())
    if "evolved" in body:
        return dict(body["evolved"]["genome"]), f"{path}:evolved"
    return dict(body["genome"]), path


# ----------------------------------------------------------------------------- commands
def cmd_pretrain(a: argparse.Namespace) -> None:
    schedule = {**SCHEDULE, "change": "none"}
    patch, info = pretrain(a.seed, schedule, episodes=a.episodes, rate=a.rate, epochs=a.epochs, belief=a.belief_units, encoded=a.encoded, fill_store=not a.empty_store)
    rng = np.random.default_rng(a.seed + 99)
    habit, cost, moments = fit_habit(patch, rng, np.array(info["scale"]), HABIT_HAND_SET)
    info.update({"habit_fitted": habit, "habit_imagined_cost": cost, "habit_fit_moments": moments})
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    patch.save(out)
    out.with_suffix(".json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k != "curve"}, indent=1))


def _life_row(belief: Path, seed: int, schedule: dict[str, Any], genome: dict[str, Any], arm: str, scale: np.ndarray, floors: dict[str, float], logs_dir: Path | None) -> dict[str, Any]:
    t0 = time.perf_counter()
    life, events = run_life(belief, seed, schedule, genome, arm, scale, floors)
    m, c = metrics(life, schedule), life.compute()
    row = {"seed": seed, "arm": arm, "metrics": m, "compute": c, "fitness": fitness_parts(life, schedule), "learns": life.learns, "dots": life.dots, "final_habit": life.habit, "seconds": time.perf_counter() - t0}
    if logs_dir is not None:
        logs_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(logs_dir / f"{schedule['change']}_{arm}_seed{seed}.npz", **{k: np.array(v) for k, v in life.log.items()})
    print(json.dumps({"arm": arm, "seed": seed, "catches": m["catches"], "misses": m["misses"], "catch_rate": m["catch_rate"], "awake_share": round(m["awake_share"], 3), "moments_per_decision": round(c["moments_per_decision"], 2), "false_wakes": m["false_wakes"], "learn_calls": m["learn_calls"], "kept": m["kept_updates"], "before_change": m["learn_calls_before_change"], "detection": m.get("detection_latency"), "after_change_rate": m["catch_rate_after_change"], "fitness": round(row["fitness"]["fitness"], 4), "seconds": round(row["seconds"], 1)}), flush=True)
    return row


def _life_job(args: tuple[Any, ...]) -> dict[str, Any]:
    return _life_row(*args)


SUMMARY_KEYS = ("catch_rate", "catch_rate_before_change", "catch_rate_after_change", "catches", "misses", "dots_counted", "catch_latency_mean", "wake_latency_mean", "awake_share", "awake_share_with_dot", "awake_share_without_dot", "false_wakes", "wakes", "learn_calls", "kept_updates", "learn_calls_before_change", "kept_updates_before_change", "learn_calls_after_change", "kept_updates_after_change", "detection_latency", "surprise_term", "surprise_term_after_change", "surprise_median_quiet", "surprise_median_with_dot", "surprise_with_dot_before_change", "surprise_with_dot_after_change", "surprise_with_dot_last_600", "task_cost_mean", "moments_per_decision", "governor_steps_mean")


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    agg: dict[str, Any] = {}
    for k in SUMMARY_KEYS:
        vals = [r["metrics"].get(k) for r in rows]
        got = [v for v in vals if v is not None]
        agg[k] = {"mean": float(np.mean(got)) if got else None, "values": vals}
    agg["fitness"] = {"mean": float(np.mean([r["fitness"]["fitness"] for r in rows])), "values": [r["fitness"]["fitness"] for r in rows]}
    agg["moments_by_kind"] = {k: float(np.mean([r["compute"][k] for r in rows])) for k in ("assimilate", "imagine", "observe", "refit", "governor")}
    agg["wall_ms_per_decision"] = float(np.mean([r["compute"]["wall_ms"] / max(1, r["metrics"]["decisions"]) for r in rows]))
    return agg


def cmd_lives(a: argparse.Namespace) -> None:
    belief = Path(a.belief)
    ctx = load_context(belief)
    schedule = {**SCHEDULE, "change": a.change, "decisions": a.decisions}
    scale, floors = np.array(ctx["scale"]), ctx["routine"]
    arms = a.arms.split(",")
    genomes = {}
    for arm in arms:
        which = "patch" if arm == "patch" else "threshold"
        path = a.patch_genome if which == "patch" else a.threshold_genome
        genomes[arm] = genome_from(path, which)
    logs_dir = Path(a.logs) if a.logs else None
    jobs = [(belief, seed, schedule, genomes[arm][0], arm, scale, floors, logs_dir) for arm in arms for seed in a.seeds]
    if a.workers > 1:
        with multiprocessing.get_context("fork").Pool(a.workers) as pool:
            rows = pool.map(_life_job, jobs)
    else:
        rows = [_life_job(j) for j in jobs]
    results: dict[str, list[dict[str, Any]]] = {arm: [r for r in rows if r["arm"] == arm] for arm in arms}
    summary = {arm: summarize_rows(rs) for arm, rs in results.items()}
    body = {"schedule": schedule, "machinery": MACHINERY, "genomes": {arm: g for arm, (g, _) in genomes.items()}, "genome_sources": {arm: s for arm, (_, s) in genomes.items()}, "seeds": a.seeds, "arms": arms, "belief": {"path": str(belief), "sha256": hashlib.sha256(belief.read_bytes()).hexdigest(), "context": {k: v for k, v in ctx.items() if k != "curve"}}, "prices": {"moments": PRICE_MOMENTS, "surprise": PRICE_SURPRISE}, "summary": summary, "lives": results}
    write_receipt(Path(a.out), "cadence-life/dozing-cat-lives/v1", body)
    print(json.dumps({arm: {k: v["mean"] for k, v in s.items() if isinstance(v, dict) and "mean" in v} for arm, s in summary.items()}, indent=1))


def cmd_evolve(a: argparse.Namespace) -> None:
    belief = Path(a.belief)
    ctx = load_context(belief)
    schedule = {**SCHEDULE, "change": a.change, "decisions": a.decisions}
    scale, floors = ctx["scale"], ctx["routine"]
    which = a.which
    space, start = SPACES[which], HAND_SETS[which]
    train_seeds, test_seeds = list(a.train_seeds), list(a.test_seeds)
    fitness = functools.partial(fitness_of, str(belief), schedule, scale, floors, train_seeds, which)
    t0 = time.perf_counter()
    history: list[dict[str, Any]] = []
    out = Path(a.out)

    def report(lineage: Any) -> None:
        history.append({"generation": len(lineage.generations), "best_fitness": float(lineage.best_fitness), "seconds": time.perf_counter() - t0})
        print(json.dumps({"generation": len(lineage.generations), "best": float(lineage.best_fitness), "mean": float(lineage.generations[-1]["mean_fitness"]), "seconds": round(time.perf_counter() - t0)}), flush=True)
        out.with_suffix(".progress.json").write_text(json.dumps({"which": which, "history": history, "best": lineage.best}, indent=1, default=str))

    with multiprocessing.get_context("fork").Pool(a.workers) as pool:
        if a.stage in ("evolve", "all"):
            lineage = evolve(fitness, dict(start), mutate=genes(space, rate=a.mutation_rate), generations=a.generations, population=a.population, keep=a.keep, seed=a.seed, mapper=pool.map, report=report)
            evolved = {"genome": lineage.best, "train_fitness": float(lineage.best_fitness), "seconds": time.perf_counter() - t0, "history": history, "lineage": [{k: (v if not isinstance(v, np.ndarray) else v.tolist()) for k, v in g.items()} for g in lineage.generations]}
            out.with_suffix(".evolved.json").write_text(json.dumps(evolved, indent=1, default=str))
            if a.stage == "evolve":
                return
        else:
            evolved = json.loads(out.with_suffix(".evolved.json").read_text())
        if a.stage in ("random", "all"):
            rng = np.random.default_rng(a.seed + 777)
            draws = [random_genome(rng, space, start) for _ in range(a.generations * a.population)]
            t1 = time.perf_counter()
            random_scores = list(pool.map(functools.partial(fitness, seed=0), draws))
            random = {"genome": draws[int(np.argmax(random_scores))], "train_fitness": float(max(random_scores)), "scores": [float(s) for s in random_scores], "genomes": draws, "seconds": time.perf_counter() - t1}
            out.with_suffix(".random.json").write_text(json.dumps(random, indent=1, default=str))
            if a.stage == "random":
                return
        else:
            random = json.loads(out.with_suffix(".random.json").read_text())
        if a.stage in ("held_out", "all"):
            named = (("hand_set", start), ("evolved", evolved["genome"]), ("random_search", random["genome"]))
            jobs = [(belief, schedule, np.array(scale), floors, which, g, s) for _, g in named for s in test_seeds]
            rows = pool.map(_score_job, jobs)
            held = {name: score_summary(rows[i * len(test_seeds) : (i + 1) * len(test_seeds)]) for i, (name, _) in enumerate(named)}
        else:
            return
    body = {"which": which, "schedule": schedule, "machinery": MACHINERY, "space": {k: list(v) for k, v in space.items()}, "hand_set": start, "prices": {"moments": PRICE_MOMENTS, "surprise": PRICE_SURPRISE}, "fitness": "catch rate - price_moments * moments per decision - price_surprise * mean(min(surprise, 1))", "train_seeds": train_seeds, "test_seeds": test_seeds, "generations": a.generations, "population": a.population, "keep": a.keep, "mutation_rate": a.mutation_rate, "seed": a.seed, "evolved": evolved, "random_search": random, "held_out": held, "winner": max(held, key=lambda k: held[k]["fitness"]), "belief": {"path": str(belief), "sha256": hashlib.sha256(belief.read_bytes()).hexdigest()}}
    write_receipt(out, "cadence-life/dozing-cat-evolution/v1", body)
    print(json.dumps({"which": which, "winner": body["winner"], "held_out": {k: {kk: v[kk] for kk in ("fitness", "catch_rate", "moments_per_decision", "surprise_term", "awake_share")} for k, v in held.items()}}, indent=1))


def cmd_timing(a: argparse.Namespace) -> None:
    """The per-decision costs in one process: one moment, one imagination call, one governor
    settle, one learn call, one habit refit, each the median of repeats."""
    belief = Path(a.belief)
    ctx = load_context(belief)
    schedule = {**SCHEDULE, "change": a.change}
    scale, floors = np.array(ctx["scale"]), ctx["routine"]
    patch = BeliefPatch.load(belief)
    patch.reset()
    r = np.zeros(READINGS)
    reps = []
    for _ in range(300):
        t0 = time.perf_counter()
        patch.assimilate(r[None, None], np.zeros((1, 1, ACTIONS)))
        reps.append(1000.0 * (time.perf_counter() - t0))
    moment_ms = float(np.median(reps))
    probe = Life(BeliefPatch.load(belief), World(schedule, draw_events(np.random.default_rng(0), schedule), np.random.default_rng(0)), PATCH_HAND_SET, scale, floors, arm="patch", seed=0, keep_log=False)
    probe.run(40)
    reps = []
    for _ in range(100):
        t0 = time.perf_counter()
        probe._imagine(probe.world.reading(), probe.patch.state)
        reps.append(1000.0 * (time.perf_counter() - t0))
    imagine_ms = float(np.median(reps))
    reps, steps = [], []
    for _ in range(100):
        t0 = time.perf_counter()
        _, s = probe.governor.settle(probe.readback())  # type: ignore[union-attr]
        reps.append(1000.0 * (time.perf_counter() - t0))
        steps.append(s)
    governor_ms = float(np.median(reps))
    probe.run(int(MACHINERY["window"]) + 8)
    reps, entries = [], []
    for _ in range(3):
        t0 = time.perf_counter()
        entry = probe._learn()
        reps.append(1000.0 * (time.perf_counter() - t0))
        entries.append({k: entry[k] for k in ("window", "passes_kept", "kept", "moments") if k in entry})
    learn_ms = float(np.median(reps))
    reps = []
    for _ in range(3):
        t0 = time.perf_counter()
        fit_habit(probe.patch, np.random.default_rng(1), scale, HABIT_HAND_SET)
        reps.append(1000.0 * (time.perf_counter() - t0))
    refit_ms = float(np.median(reps))
    body = {"schedule": schedule, "single_process": True, "pieces": {"one_moment_of_the_patch_ms": moment_ms, "moment_macs": moment_macs(patch), "one_imagination_call_ms": imagine_ms, "imagination_moments": (len(DIRECTIONS) + 1) * int(MACHINERY["horizon"]), "one_governor_settle_ms": governor_ms, "governor_steps_median": float(np.median(steps)), "governor_synapses": probe.governor.synapses, "one_learn_call_ms": learn_ms, "learn_calls": entries, "one_habit_refit_ms": refit_ms}}  # type: ignore[union-attr]
    write_receipt(Path(a.out), "cadence-life/dozing-cat-timing/v1", body)
    print(json.dumps({"machine": machine(), **body["pieces"]}, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("pretrain")
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--episodes", type=int, default=100)
    q.add_argument("--rate", type=float, default=0.3)
    q.add_argument("--epochs", type=int, default=40)
    q.add_argument("--belief-units", type=int, default=32)
    q.add_argument("--encoded", type=int, default=24)
    q.add_argument("--empty-store", action="store_true", help="leave the record store empty (it reads zero) instead of filling it with the routine corpus")
    q.add_argument("--out", default=str(HERE / "pretrained" / "cat_belief.npz"))
    q.set_defaults(func=cmd_pretrain)
    q = sub.add_parser("lives")
    q.add_argument("--belief", default=str(HERE / "pretrained" / "cat_belief.npz"))
    q.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    q.add_argument("--arms", default="patch,threshold,always_awake,never_wakes")
    q.add_argument("--change", default=SCHEDULE["change"])
    q.add_argument("--decisions", type=int, default=SCHEDULE["decisions"])
    q.add_argument("--patch-genome", default=None, help="an evolution receipt whose evolved patch genome is used")
    q.add_argument("--threshold-genome", default=None, help="an evolution receipt whose evolved threshold genome is used")
    q.add_argument("--logs", default=None)
    q.add_argument("--workers", type=int, default=3)
    q.add_argument("--out", required=True)
    q.set_defaults(func=cmd_lives)
    q = sub.add_parser("evolve")
    q.add_argument("--which", choices=("patch", "threshold"), required=True)
    q.add_argument("--belief", default=str(HERE / "pretrained" / "cat_belief.npz"))
    q.add_argument("--change", default=SCHEDULE["change"])
    q.add_argument("--decisions", type=int, default=SCHEDULE["decisions"])
    q.add_argument("--train-seeds", type=int, nargs="+", default=[100, 101])
    q.add_argument("--test-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    q.add_argument("--generations", type=int, default=6)
    q.add_argument("--population", type=int, default=12)
    q.add_argument("--keep", type=int, default=4)
    q.add_argument("--mutation-rate", type=float, default=0.3)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--workers", type=int, default=3)
    q.add_argument("--stage", choices=("evolve", "random", "held_out", "all"), default="all", help="one stage at a time keeps every run short; the receipt is written by held_out or all")
    q.add_argument("--out", required=True)
    q.set_defaults(func=cmd_evolve)
    q = sub.add_parser("timing")
    q.add_argument("--belief", default=str(HERE / "pretrained" / "cat_belief.npz"))
    q.add_argument("--change", default=SCHEDULE["change"])
    q.add_argument("--out", default=str(HERE / "receipts" / "cat_timing.json"))
    q.set_defaults(func=cmd_timing)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
