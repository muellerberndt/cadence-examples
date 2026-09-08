"""One equilibrium net that is the parrot's auditory memory and its vocal mirror.

Owners: an auditory context (recent cochlear frames and averaged bins behind them), two
hidden populations that both read it, and two output groups: the next cochlear frame the
parrot expects (the memory, on the auditory population, as the caudomedial nidopallium
holds familiar sounds) and the motor command that would produce the sound it is hearing
(the mirror, on the vocal population, as the song system's mirror neurons do). Keeping the
populations apart keeps one learner's nudges from moving the other's seams: a nudge on
the memory group changes the auditory owners, whose seams then move, while the vocal
owners, tied only to the clamped context, barely stir. Both learn by the free/nudged rule
with the quadratic nudge on their own group; the seams decay a little every update, so
what is not heard again fades and what repeats stays. Motor commands are population codes
(a bump over a few owners per muscle), read out as the bump-weighted mean, because the
rule learns a pattern far better than a level.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

import cadence as cd
from cadence.settle import Nudge

from syrinx import CHANNELS

WINDOW = 8  # recent frames of context: 80 ms at full resolution
SCALES = ((6, 20), (6, 5))  # older context as averaged bins, slowest first: six of 200 ms (1.2 s), then six of 50 ms (300 ms); their fill is the parrot's clock
ROWS = sum(count for count, _ in SCALES)
NEED = WINDOW + sum(count * span for count, span in SCALES)  # frames of history a context reads: 158
CONTEXT = ROWS + WINDOW
AUDITORY, VOCAL = 128, 96  # hidden owners under the memory group and under the motor group
HIDDEN = AUDITORY + VOCAL
MOTOR = {"tension": 8, "pressure": 4, "tract": 4}  # owners per muscle group


def bumps(level: float | np.ndarray, k: int) -> np.ndarray:
    """A level in [0, 1] as a bump over k evenly spaced centres."""
    centres = np.linspace(0.0, 1.0, k)
    level = np.asarray(level, dtype=float)[..., None]
    return np.exp(-((level - centres) ** 2) / (2 * (1.0 / (k - 1)) ** 2))


def level_of(pattern: np.ndarray) -> np.ndarray:
    """The bump-weighted mean of a pattern over evenly spaced centres."""
    k = pattern.shape[-1]
    centres = np.linspace(0.0, 1.0, k)
    w = np.clip(pattern, 0.0, None) + 1e-6
    return (w * centres).sum(axis=-1) / w.sum(axis=-1)


def context(frames: list[np.ndarray] | np.ndarray, end: int | None = None) -> np.ndarray:
    """The (CONTEXT, CHANNELS) context ending before ``end``: averaged bins slowest and oldest first, then the recent frames."""
    h = np.asarray(frames if end is None else frames[:end], dtype=float).reshape(-1, CHANNELS)
    if len(h) < NEED:
        h = np.concatenate([np.zeros((NEED - len(h), CHANNELS)), h])
    h = h[-NEED:]
    rows, k = [], 0
    for count, span in SCALES:
        rows.append(h[k : k + count * span].reshape(count, span, CHANNELS).mean(axis=1))
        k += count * span
    rows.append(h[k:])
    return np.concatenate(rows)


def advance(ctx: np.ndarray, frame: np.ndarray, tail: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Slide a context by one frame; ``tail`` holds the frames behind it so the averaged bins stay exact."""
    tail = [*tail, frame][-NEED:]
    return context(tail), tail


def wiring_of(inputs: int, outputs: int, seed: int) -> cd.Wiring:
    """Context to two hidden populations by tied seams; the auditory one to the memory group, the vocal one to the motor group."""
    rng = np.random.default_rng(seed)
    n_in, n_a, n_v = inputs, AUDITORY, VOCAL
    ctx = np.arange(n_in)
    aud = np.arange(n_in, n_in + n_a)
    voc = np.arange(n_in + n_a, n_in + n_a + n_v)
    pred = np.arange(n_in + n_a + n_v, n_in + n_a + n_v + CHANNELS)
    mot = np.arange(pred[-1] + 1, pred[-1] + 1 + outputs - CHANNELS)
    pre, post, sign = [], [], []

    def block(a: np.ndarray, b: np.ndarray) -> None:
        i, j = np.meshgrid(a, b, indexing="ij")
        magnitude = rng.uniform(0.0, 1.0, size=i.size) * np.sqrt(6.0 / (len(a) + len(b)))
        s = rng.choice([-1.0, 1.0], size=i.size) * magnitude
        pre.extend([i.ravel(), j.ravel()])
        post.extend([j.ravel(), i.ravel()])
        sign.extend([s, s])

    block(ctx, aud)
    block(aud, pred)
    block(ctx, voc)
    block(voc, mot)
    n = int(mot[-1] + 1)
    sets = {"input": range(n_in), "hidden": range(n_in, n_in + n_a + n_v), "auditory": aud.tolist(), "vocal": voc.tolist(), "output": range(pred[0], n)}
    return cd.Wiring.from_edges(n, pre=np.concatenate(pre), post=np.concatenate(post), sign=np.concatenate(sign), sets=sets, label=f"parrot:{n_in}x({n_a}+{n_v})x{outputs}")


class Brain:
    def __init__(self, seed: int, *, eta: float = 0.3, decay: float = 3e-4, beta: float = 0.1, momentum: float = 0.0, backend: str = "cpu") -> None:
        self.inputs = CONTEXT * CHANNELS
        self.motor_width = sum(MOTOR.values())
        self.outputs = CHANNELS + self.motor_width
        self.wiring = wiring_of(self.inputs, self.outputs, seed)
        out = np.array(self.wiring.sets["output"])
        self.predict_index = out[:CHANNELS]
        self.motor_index = out[CHANNELS:]
        self.groups: dict[str, np.ndarray] = {}
        k = 0
        for name, width in MOTOR.items():
            self.groups[name] = self.motor_index[k : k + width]
            k += width
        # a stream has no epochs: a constant, gentle rate; at 1.0 with momentum the memory's seams saturated within a minute
        config = cd.LearnerConfig(beta=beta, eta=eta, eta_bias=0.02, nudge="quadratic", tolerance=3e-3, nudged_steps=12, free_steps=60, momentum=momentum, decay=decay)
        self.learner = cd.Learner(cd.Settlement(self.wiring, cd.learning_rule(dt=1.0), backend=backend), self.wiring.sets["output"], config)
        self.predict_mask = np.zeros(self.wiring.n)
        self.predict_mask[self.predict_index] = 1.0
        self.motor_mask = np.zeros(self.wiring.n)
        self.motor_mask[self.motor_index] = 1.0
        # each population learns, and decays, only on its own updates: the memory's seams and owners on a memory
        # update, the mirror's on a mirror update, so one learner's decay never erodes the other's seams
        w = self.wiring
        aud = np.zeros(w.n, dtype=bool)
        aud[list(w.sets["auditory"])] = True
        aud[self.predict_index] = True
        voc = np.zeros(w.n, dtype=bool)
        voc[list(w.sets["vocal"])] = True
        voc[self.motor_index] = True
        self.memory_edges, self.memory_owners = aud[w.pre] | aud[w.post], aud
        self.mirror_edges, self.mirror_owners = voc[w.pre] | voc[w.post], voc

    def drive(self, windows: np.ndarray) -> np.ndarray:
        """Clamp levels for a batch of contexts, (batch, CONTEXT, CHANNELS)."""
        flat = np.asarray(windows, dtype=float).reshape(len(windows), -1)
        return self.learner.engine.clamp_levels(np.pad(flat, ((0, 0), (0, self.wiring.n - self.inputs))))

    def free(self, windows: np.ndarray) -> np.ndarray:
        drive = self.drive(windows)
        return self.learner.free(drive).activation

    def predicted(self, activation: np.ndarray) -> np.ndarray:
        """The expected next cochlear frame, as levels in [0, 1]."""
        return np.clip(activation[:, self.predict_index], 0.0, 1.0)

    def motor(self, activation: np.ndarray) -> dict[str, np.ndarray]:
        return {name: level_of(activation[:, idx]) for name, idx in self.groups.items()}

    def imagine(self, windows: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
        """What comes next after these contexts, the command that would produce it, and the context the mirror saw.

        Two settlements: the memory expects the next frame from what is heard; then the mirror,
        shown a context whose latest frame is that expectation, answers with the command that
        makes such a sound (it learned that association from its own babbling)."""
        expected = self.predicted(self.free(windows))
        shifted = np.concatenate([windows[:, 1:], expected[:, None, :]], axis=1)
        return expected, self.motor(self.free(shifted)), shifted

    def reinforce(self, windows: np.ndarray, commands: dict[str, np.ndarray], advantage: np.ndarray) -> dict[str, float]:
        """The command taken, pulled toward (advantage > 0) or pushed from (advantage < 0) in the context that chose it."""
        return self._learn(windows, self.motor_pattern(commands), self.motor_mask, advantage)

    def motor_pattern(self, command: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([bumps(command[name], width) for name, width in MOTOR.items()], axis=-1)

    def _learn(self, windows: np.ndarray, target_levels: np.ndarray, mask: np.ndarray, weight: np.ndarray | None) -> dict[str, float]:
        """One free/nudged update: the masked output group pulled toward its target levels."""
        learner = self.learner
        drive = self.drive(windows)
        free = learner.free(drive)
        target = np.zeros((len(windows), self.wiring.n))
        idx = np.flatnonzero(mask)
        target[:, idx] = target_levels
        cfg = learner.config
        plus = learner.engine.settle_batch(drive, steps=cfg.nudged_steps, state=free, nudge=Nudge(target, mask, cfg.beta, weight=weight), tolerance=cfg.tolerance)
        minus = learner.engine.settle_batch(drive, steps=cfg.nudged_steps, state=free, nudge=Nudge(target, mask, -cfg.beta, weight=weight), tolerance=cfg.tolerance)
        memory = mask is self.predict_mask
        learner.trainable_overlaps = self.memory_edges if memory else self.mirror_edges
        learner.trainable_owners = self.memory_owners if memory else self.mirror_owners
        return learner.update(free, plus, minus)

    def learn_memory(self, windows: np.ndarray, next_frames: np.ndarray, weight: np.ndarray | None = None) -> dict[str, float]:
        """The memory: after this context, expect this frame."""
        return self._learn(windows, next_frames, self.predict_mask, weight)

    def learn_inverse(self, windows: np.ndarray, commands: dict[str, np.ndarray], weight: np.ndarray | None = None) -> dict[str, float]:
        """The mirror: this sound is what that command produced."""
        return self._learn(windows, self.motor_pattern(commands), self.motor_mask, weight)

    def prediction_error(self, windows: np.ndarray, next_frames: np.ndarray) -> np.ndarray:
        """Mean squared error of the expected next frame, per row: low where a sound is known."""
        return ((self.predicted(self.free(windows)) - next_frames) ** 2).mean(axis=1)

    def parameters(self) -> int:
        return self.learner.parameters()

    def export(self) -> dict[str, Any]:
        engine = self.learner.engine
        rule = engine.rule
        return {
            "n": int(self.wiring.n), "window": WINDOW, "scales": [list(s) for s in SCALES], "need": NEED, "channels": CHANNELS, "hidden": HIDDEN, "auditory": AUDITORY, "vocal": VOCAL, "motor": MOTOR,
            "sets": {k: list(map(int, v)) for k, v in self.wiring.sets.items()}, "predict": self.predict_index.tolist(), "groups": {k: v.tolist() for k, v in self.groups.items()},
            "W": [round(float(w), 5) for w in engine.dense().ravel()], "bias": [round(float(v), 5) for v in engine.bias],
            "rule": {"slope": rule.slope, "threshold": rule.threshold, "leak": rule.leak, "dt": rule.dt, "clamp": rule.clamp_amplitude, "rest": rule.rest_emission},
            "learner": {k: v for k, v in self.learner.config.to_dict().items() if isinstance(v, (int, float, str, bool))},
        }
