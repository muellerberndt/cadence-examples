"""One equilibrium net that is the parrot's auditory memory and its vocal mirror.

Two populations of hidden owners share one net and one learning rule but read different
inputs and answer on different output groups.

The memory (the auditory population, as the caudomedial nidopallium holds the sounds a
bird knows) reads a *cue* and a *clock*. The cue is the first 160 ms of a sound after
lateral inhibition, held for as long as the sound lasts: the parrot's working memory of
which sound is playing. The clock is a bump of activity that moves along a chain of owners
as the frames pass since the sound began, as the sparse sequence in HVC does; it is the
body's clock, and the net learns what the cochlea reported at each of its ticks. The
memory's answer is the cochlear frame it expects at this tick of this sound. Replaying a
memory is running the clock from zero with the cue held; the memory answers frame by frame
and nothing it says is fed back, so a replay cannot drift into a blur.

The mirror (the vocal population, as the song system's mirror neurons) reads the last four
cochlear frames as their peaks after lateral inhibition, their dominant channel as a bump (a
pitch code) and their loudness as a bump, and answers with the muscle command
that would make such a sound: it learns that association from the parrot's own babbling,
where the command that produced each frame is known. A replayed expectation, sharpened and
shown to the mirror as if heard, comes back as the command that reproduces it.

Both learn by the free/nudged rule with a quadratic nudge on their own group; each update
moves and decays only its own population's seams, so what is not heard again fades and what
repeats stays, and one learner's decay never erodes the other's seams. Motor commands are
population codes (a bump over a few owners per muscle) read out as the bump-weighted mean,
because the rule learns a pattern far better than a level.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

import cadence as cd
from cadence.settle import Nudge

from syrinx import CHANNELS

WINDOW = 16  # frames of a sound's onset the parrot holds as its cue: 160 ms (80 ms confused the phone with the whistle)
MIRROR_WINDOW = 4  # frames the mirror reads: 40 ms, the syrinx and the cochlea both lag the command a little
LEVELS = 6  # owners per frame that carry its loudness as a bump: silence has a pattern of its own, so quiet can mean a closed air sac
PITCH_WIDTH = 1.0  # channels: the width of the bump that carries a frame's dominant channel (a pitch code, as the auditory forebrain has pitch-tuned neurons)
FAINT = 0.1  # a frame below this level carries no pitch
PHASES, MAX_PHASE = 50, 200  # the clock: a bump over 50 owners spanning 200 frames (2 s), one owner per 40 ms
SEQUENCE, ACTIVE = 768, 24  # sequence owners: fixed random conjunctions of the cue and the clock, the ACTIVE strongest awake at any moment
VOCAL = 96  # hidden owners under the motor group
HIDDEN = SEQUENCE + VOCAL
MOTOR = {"tension": 8, "pressure": 4, "tract": 4}  # owners per muscle group
PEAK = 0.7  # lateral inhibition: a channel below this share of the frame's loudest is silenced
PHASE_CENTRES = np.linspace(0.0, MAX_PHASE, PHASES)
PHASE_WIDTH = MAX_PHASE / (PHASES - 1)


def bumps(level: float | np.ndarray, k: int) -> np.ndarray:
    """A level in [0, 1] as a bump over k evenly spaced centres."""
    centres = np.linspace(0.0, 1.0, k)
    level = np.asarray(level, dtype=float)[..., None]
    return np.exp(-((level - centres) ** 2) / (2 * (1.0 / (k - 1)) ** 2))


def level_of(pattern: np.ndarray) -> np.ndarray:
    """Where the bump sits: the centroid of the strongest owner and its two neighbours over evenly spaced centres.

    A centroid over the whole group drifts toward the middle whenever the pattern is not a clean
    bump; reading only around the peak keeps the ends of the range reachable."""
    pattern = np.asarray(pattern, dtype=float)
    k = pattern.shape[-1]
    centres = np.linspace(0.0, 1.0, k)
    peak = pattern.argmax(axis=-1)
    idx = np.clip(peak[..., None] + np.array([-1, 0, 1]), 0, k - 1)
    w = np.clip(np.take_along_axis(pattern, idx, axis=-1), 0.0, None) + 1e-6
    return (w * centres[idx]).sum(axis=-1) / w.sum(axis=-1)


def sharpen(frames: np.ndarray) -> np.ndarray:
    """Lateral inhibition over channels: keep the peaks of each frame, silence the rest.

    The auditory pathway sharpens spectral peaks by lateral inhibition; here a channel survives
    only above ``PEAK`` times the loudest channel of its frame. The cue and the mirror's view are
    both sharpened, so what identifies a sound and what commands a sound is where its peaks are."""
    frames = np.asarray(frames, dtype=float)
    top = frames.max(axis=-1, keepdims=True)
    return np.where(frames >= PEAK * top, frames, 0.0)


def pitch_code(frames: np.ndarray) -> np.ndarray:
    """Each frame's dominant channel as a bump over the channels, nothing for a faint frame: a pitch code the mirror can read
    whatever the peak's width or level (a lone sharp peak in a replayed expectation reads the same as the cochlea's broader one)."""
    frames = np.asarray(frames, dtype=float)
    dominant = frames.argmax(axis=-1)[..., None].astype(float)
    bump = np.exp(-((np.arange(CHANNELS) - dominant) ** 2) / (2 * PITCH_WIDTH**2))
    return np.where(frames.max(axis=-1, keepdims=True) >= FAINT, bump, 0.0)


def phase_code(phase: float | np.ndarray) -> np.ndarray:
    """The clock's bump for a count of frames since the onset."""
    phase = np.asarray(phase, dtype=float)[..., None]
    return np.exp(-((phase - PHASE_CENTRES) ** 2) / (2 * PHASE_WIDTH**2))


def cue_of(frames: np.ndarray) -> np.ndarray:
    """The cue a sound leaves: its first WINDOW cochlear frames after lateral inhibition."""
    frames = np.asarray(frames, dtype=float)
    if len(frames) < WINDOW:
        frames = np.concatenate([frames, np.zeros((WINDOW - len(frames), CHANNELS))])
    return sharpen(frames[:WINDOW])


def wiring_of(seed: int) -> cd.Wiring:
    """Sequence owners to the expected frame (one way); the peaks, the pitch and the loudness of the recent frames to the vocal population to the motor groups (tied seams)."""
    rng = np.random.default_rng(seed)
    sizes = [("sequence", SEQUENCE), ("peaks", MIRROR_WINDOW * CHANNELS), ("pitch", MIRROR_WINDOW * CHANNELS), ("loudness", MIRROR_WINDOW * LEVELS), ("vocal", VOCAL), ("predict", CHANNELS), ("motor", sum(MOTOR.values()))]
    index: dict[str, np.ndarray] = {}
    k = 0
    for name, size in sizes:
        index[name] = np.arange(k, k + size)
        k += size
    pre, post, sign = [], [], []

    def block(a: np.ndarray, b: np.ndarray, back: bool = True) -> None:
        i, j = np.meshgrid(a, b, indexing="ij")
        magnitude = rng.uniform(0.0, 1.0, size=i.size) * np.sqrt(6.0 / (len(a) + len(b)))
        s = rng.choice([-1.0, 1.0], size=i.size) * magnitude
        pre.append(i.ravel())
        post.append(j.ravel())
        sign.append(s)
        if back:  # a tied seam: the same weight both ways
            pre.append(j.ravel())
            post.append(i.ravel())
            sign.append(s)

    # the chain projects one way, as HVC does onto RA: a sequence owner that is not awake at this moment stays exactly
    # silent, so what one moment teaches the memory moves no seam of another moment's owners
    block(index["sequence"], index["predict"], back=False)
    block(index["peaks"], index["vocal"])
    block(index["pitch"], index["vocal"])
    block(index["loudness"], index["vocal"])
    block(index["vocal"], index["motor"])
    sets = {name: idx.tolist() for name, idx in index.items()}
    sets["input"] = [*sets["sequence"], *sets["peaks"], *sets["pitch"], *sets["loudness"]]
    sets["hidden"] = [*sets["vocal"]]
    sets["output"] = [*sets["predict"], *sets["motor"]]
    return cd.Wiring.from_edges(k, pre=np.concatenate(pre), post=np.concatenate(post), sign=np.concatenate(sign), sets=sets, label=f"parrot:{SEQUENCE}x{CHANNELS}|{MIRROR_WINDOW * (2 * CHANNELS + LEVELS)}x{VOCAL}x{sum(MOTOR.values())}")


class Brain:
    def __init__(self, seed: int, *, eta: float = 5.0, decay: float = 1e-4, momentum: float = 0.0, mirror_eta: float = 0.3, mirror_decay: float = 1e-4, mirror_momentum: float = 0.9, beta: float = 0.1, backend: str = "cpu") -> None:
        self.wiring = wiring_of(seed)
        sets = self.wiring.sets
        self.sequence_index = np.array(sets["sequence"])
        self.peaks_index, self.pitch_index, self.loudness_index = np.array(sets["peaks"]), np.array(sets["pitch"]), np.array(sets["loudness"])
        # the chain: each sequence owner reads a fixed random mix of the cue and, separately, of the clock, and fires on their
        # product (a coincidence detector: it needs both); it is wiring the parrot is born with
        rng = np.random.default_rng(seed + 1000)
        self.cue_projection = np.round(rng.normal(size=(SEQUENCE, WINDOW * CHANNELS)) / np.sqrt(WINDOW * CHANNELS), 2)  # two decimals: the page carries the same numbers
        self.clock_projection = np.round(rng.normal(size=(SEQUENCE, PHASES)) / np.sqrt(PHASES), 2)
        self.predict_index, self.motor_index = np.array(sets["predict"]), np.array(sets["motor"])
        self.inputs, self.outputs = len(sets["input"]), len(sets["output"])
        self.groups: dict[str, np.ndarray] = {}
        k = 0
        for name, width in MOTOR.items():
            self.groups[name] = self.motor_index[k : k + width]
            k += width
        base = cd.LearnerConfig(beta=beta, eta=eta, eta_bias=0.02, nudge="quadratic", tolerance=3e-3, nudged_steps=12, free_steps=60, momentum=momentum, decay=decay)
        # the memory: a sparse code wants a plain, large step (each seam is touched only when its owner is awake, so momentum
        # would only scale the step down); the mirror: dense inputs, a gentler rate with momentum
        self.memory_config = dataclasses.replace(base, eta_bias=0.0)  # no bias on the expected frame: a moment the memory never learned must sound like nothing, not like the average sound
        self.mirror_config = dataclasses.replace(base, eta=mirror_eta, decay=mirror_decay, momentum=mirror_momentum)
        self.learner = cd.Learner(cd.Settlement(self.wiring, cd.learning_rule(dt=1.0), backend=backend), sets["output"], base)
        self.predict_mask = np.zeros(self.wiring.n)
        self.predict_mask[self.predict_index] = 1.0
        self.motor_mask = np.zeros(self.wiring.n)
        self.motor_mask[self.motor_index] = 1.0
        w = self.wiring
        aud = np.zeros(w.n, dtype=bool)
        aud[self.predict_index] = True
        voc = np.zeros(w.n, dtype=bool)
        voc[list(sets["vocal"])] = True
        voc[self.motor_index] = True
        self.memory_edges, self.memory_owners = aud[w.pre] | aud[w.post], aud
        self.mirror_edges, self.mirror_owners = voc[w.pre] | voc[w.post], voc
        # each population keeps its own momentum: a memory update must not damp the mirror's running contrast, nor the reverse
        self.velocity = {"memory": (np.zeros(w.edges), np.zeros(w.n)), "mirror": (np.zeros(w.edges), np.zeros(w.n))}

    # --- clamps
    def sequence(self, cues: np.ndarray, phases: np.ndarray) -> np.ndarray:
        """The chain's state for each (cue, tick): the ACTIVE sequence owners that read this conjunction most strongly, graded by how strongly, the rest silent.

        Winner-take-all over a random conjunctive expansion: an owner fires on the product of what
        it reads from the cue and what it reads from the clock, so two sounds, or two moments of
        one sound, wake different owners unless cue and tick both agree, and what one moment
        teaches the memory does not overwrite another (as each of HVC's sequence neurons fires at
        one moment of one vocalisation)."""
        cues = np.asarray(cues, dtype=float).reshape(len(cues), -1)
        cues = cues / (np.linalg.norm(cues, axis=1, keepdims=True) + 1e-9)
        clock = phase_code(phases)
        clock = clock / (np.linalg.norm(clock, axis=1, keepdims=True) + 1e-9)
        x = np.maximum(cues @ self.cue_projection.T, 0.0) * np.maximum(clock @ self.clock_projection.T, 0.0)
        cut = -np.partition(-x, ACTIVE - 1, axis=1)[:, ACTIVE - 1 : ACTIVE]
        awake = np.where(x >= cut, x, 0.0)
        return awake / (awake.max(axis=1, keepdims=True) + 1e-9)

    def levels(self, cues: np.ndarray | None = None, phases: np.ndarray | None = None, frames: np.ndarray | None = None) -> np.ndarray:
        """Clamp levels for a batch: cues (batch, WINDOW, CHANNELS) with phases (batch,), and/or frames (batch, MIRROR_WINDOW, CHANNELS)."""
        batch = len(cues) if cues is not None else len(frames)  # type: ignore[arg-type]
        levels = np.zeros((batch, self.wiring.n))
        if cues is not None:
            assert phases is not None
            levels[:, self.sequence_index] = self.sequence(cues, phases)
        if frames is not None:
            frames = np.asarray(frames, dtype=float)
            levels[:, self.peaks_index] = sharpen(frames).reshape(batch, -1)
            levels[:, self.pitch_index] = pitch_code(frames).reshape(batch, -1)
            levels[:, self.loudness_index] = bumps(frames.max(axis=-1), LEVELS).reshape(batch, -1)
        return levels

    def free(self, levels: np.ndarray) -> np.ndarray:
        return self.learner.free(self.learner.engine.clamp_levels(levels)).activation

    def predicted(self, activation: np.ndarray) -> np.ndarray:
        return np.clip(activation[:, self.predict_index], 0.0, 1.0)

    def motor_of(self, activation: np.ndarray) -> dict[str, np.ndarray]:
        return {name: level_of(activation[:, idx]) for name, idx in self.groups.items()}

    # --- answers
    def expect(self, cues: np.ndarray, phases: np.ndarray) -> np.ndarray:
        """The cochlear frame the memory expects at each (cue, tick)."""
        return self.predicted(self.free(self.levels(cues=cues, phases=phases)))

    def replay(self, cue: np.ndarray, frames: int) -> np.ndarray:
        """Run the clock from zero with a cue held: the memory of that sound, frame by frame."""
        cues = np.repeat(np.asarray(cue, dtype=float)[None], frames, axis=0)
        return self.expect(cues, np.arange(frames))

    def motor(self, frames: np.ndarray) -> dict[str, np.ndarray]:
        """The command that would make each window of frames, (batch, MIRROR_WINDOW, CHANNELS)."""
        return self.motor_of(self.free(self.levels(frames=frames)))

    @staticmethod
    def windows_of(frames: np.ndarray) -> np.ndarray:
        """Each frame with the MIRROR_WINDOW - 1 before it (zeros before the first), (len, MIRROR_WINDOW, CHANNELS)."""
        frames = np.asarray(frames, dtype=float)
        padded = np.concatenate([np.zeros((MIRROR_WINDOW - 1, CHANNELS)), frames])
        return np.stack([padded[i : i + MIRROR_WINDOW] for i in range(len(frames))])

    # --- learning
    def motor_pattern(self, command: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([bumps(command[name], width) for name, width in MOTOR.items()], axis=-1)

    def _learn(self, levels: np.ndarray, target_levels: np.ndarray, mask: np.ndarray, weight: np.ndarray | None) -> dict[str, float]:
        """One free/nudged update: the masked output group pulled toward its target levels; only that group's population moves."""
        learner = self.learner
        memory = mask is self.predict_mask
        learner.config = cfg = self.memory_config if memory else self.mirror_config
        drive = learner.engine.clamp_levels(levels)
        free = learner.free(drive)
        target = np.zeros((len(levels), self.wiring.n))
        target[:, np.flatnonzero(mask)] = target_levels
        plus = learner.engine.settle_batch(drive, steps=cfg.nudged_steps, state=free, nudge=Nudge(target, mask, cfg.beta, weight=weight), tolerance=cfg.tolerance)
        minus = learner.engine.settle_batch(drive, steps=cfg.nudged_steps, state=free, nudge=Nudge(target, mask, -cfg.beta, weight=weight), tolerance=cfg.tolerance)
        learner.trainable_overlaps = self.memory_edges if memory else self.mirror_edges
        learner.trainable_owners = self.memory_owners if memory else self.mirror_owners
        population = "memory" if memory else "mirror"
        learner.velocity, learner.velocity_bias = self.velocity[population]
        report = learner.update(free, plus, minus)
        self.velocity[population] = (learner.velocity, learner.velocity_bias)  # the update makes new arrays
        return report

    def learn_memory(self, cues: np.ndarray, phases: np.ndarray, frames: np.ndarray, weight: np.ndarray | None = None) -> dict[str, float]:
        """The memory: at this tick of this sound, expect this frame."""
        return self._learn(self.levels(cues=cues, phases=phases), np.asarray(frames, dtype=float), self.predict_mask, weight)

    def learn_inverse(self, frames: np.ndarray, commands: dict[str, np.ndarray], weight: np.ndarray | None = None) -> dict[str, float]:
        """The mirror: these frames are what that command produced."""
        return self._learn(self.levels(frames=frames), self.motor_pattern(commands), self.motor_mask, weight)

    def reinforce(self, frames: np.ndarray, commands: dict[str, np.ndarray], advantage: np.ndarray) -> dict[str, float]:
        """The command taken, pulled toward (advantage > 0) or pushed from (advantage < 0) for the frames that chose it."""
        return self._learn(self.levels(frames=frames), self.motor_pattern(commands), self.motor_mask, advantage)

    def prediction_error(self, cues: np.ndarray, phases: np.ndarray, frames: np.ndarray) -> np.ndarray:
        """Mean squared error of the expected frame per row: low where a sound is known."""
        return ((self.expect(cues, phases) - np.asarray(frames, dtype=float)) ** 2).mean(axis=1)

    def parameters(self) -> int:
        return self.learner.parameters()

    def export(self) -> dict[str, Any]:
        engine = self.learner.engine
        rule = engine.rule
        return {
            "n": int(self.wiring.n), "window": WINDOW, "mirror_window": MIRROR_WINDOW, "levels": LEVELS, "pitch_width": PITCH_WIDTH, "faint": FAINT, "phases": PHASES, "max_phase": MAX_PHASE, "channels": CHANNELS, "hidden": HIDDEN, "sequence": SEQUENCE, "active": ACTIVE, "vocal": VOCAL, "motor": MOTOR, "peak": PEAK,
            "cue_projection": np.rint(self.cue_projection.ravel() * 100).astype(int).tolist(), "clock_projection": np.rint(self.clock_projection.ravel() * 100).astype(int).tolist(),  # hundredths
            "sets": {k: list(map(int, v)) for k, v in self.wiring.sets.items()}, "predict": self.predict_index.tolist(), "groups": {k: v.tolist() for k, v in self.groups.items()},
            "edges": {"pre": self.wiring.pre.tolist(), "post": self.wiring.post.tolist(), "scale": [round(float(v), 5) for v in engine.edge_scale]}, "bias": [round(float(v), 5) for v in engine.bias],
            "rule": {"slope": rule.slope, "threshold": rule.threshold, "leak": rule.leak, "dt": rule.dt, "clamp": rule.clamp_amplitude, "rest": rule.rest_emission},
            "learner": {k: v for k, v in self.memory_config.to_dict().items() if isinstance(v, (int, float, str, bool))},
            "mirror_learner": {k: v for k, v in self.mirror_config.to_dict().items() if isinstance(v, (int, float, str, bool))},
        }
