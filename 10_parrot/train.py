"""An African grey in a household: a day of listening, babbling and imitating, then a receipt.

Run:  python train.py                       (two simulated hours, about twenty minutes; writes receipt.json, net.json, bouts.json, imitations/*.wav)
      python train.py --seed 1 --tag _1     (a second parrot for the page: receipt_1.json, net_1.json, ...)
      python train.py --verify receipt.json

The parrot hears the household through its cochlea. Three sounds recur many times a day
(a doorbell, a microwave, a voice saying hello), three are rare (a phone, a whistle, a
siren). One equilibrium net is its whole brain (see brain.py). When a sound begins after
a quiet spell the parrot holds its first 160 ms as a cue and starts a clock; the memory
group learns what the cochlea reports at each tick of that cue, and the seams decay, so
repetition decides what stays. When the house is quiet an arousal integrator fills and
the parrot opens its syrinx: either to babble (smooth random muscle commands, the
exploratory variability of the anterior forebrain pathway) or, from a cue it kept, to
run the clock and sing what the memory replays. Hearing its own voice while babbling, the
same net learns the mirror: the command that produced the sound it hears. That inverse
model is what turns a replayed memory into muscle commands. The receipt scores, per
sound, how well the memory replays it (correlation of cochleagrams) and how well the
imitation matches it (correlation, spectral shape, pitch track), against an untrained
brain, once heard often and once heard rarely.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave as wavefile
from pathlib import Path
from typing import Any

import numpy as np

import cadence as cd

from brain import MAX_PHASE, MIRROR_WINDOW, WINDOW, Brain, cue_of, sharpen
from syrinx import CHANNELS, FRAME, FREQUENT, RARE, SR, Cochlea, Syrinx, pressure_of, waves

HERE = Path(__file__).resolve().parent
SOURCES = [("10_parrot/train.py", Path(__file__).resolve()), ("10_parrot/brain.py", HERE / "brain.py"), ("10_parrot/syrinx.py", HERE / "syrinx.py")]
MINUTES = 60
CHUNK = 8  # frames the brain takes in at once: 80 ms
GAP = (3.0, 9.0)  # seconds between household sounds
RARE_WEIGHT = 0.06  # a rare sound's chance against a frequent one's at each event
LOUD = 0.25  # a cochlear level above this is sound
QUIET = 20  # frames of quiet before a sound counts as beginning
AROUSAL_RATE, AROUSAL_NOISE = 0.12, 0.05  # per second of quiet; a bout starts past 1
BABBLE_SHARE = 0.75  # of bouts, before any cue exists every bout babbles
BABBLE_FRAMES = (80, 200)  # the mirror wants a lot of babble: a day of 34,000 babbled frames puts the imitations' pitch 1.5 channels from the original's, a day of 15,000 leaves them 2.4 off
SYLLABLE = 60  # frames between the new pitch and tract a babbling bout jumps to: subsong is varied, and the mirror needs the whole range
REPLAY_FRAMES = MAX_PHASE  # a replay may run the whole clock
REST = 25  # frames of expected quiet that end an imitation: longer than any gap inside a household sound (the phone's is 20)
QUIET_SHARE = 0.3  # an expected frame below this share of the replay's loudest so far is quiet
CLOSING = 8  # frames of closed air sac the parrot adds at the end of every bout, so the mirror learns that quiet means closed
MUSCLE = 0.5  # share of a new command a muscle takes on in one frame: the syringeal muscles are fast, not instant
REFINE = False  # reinforcement while imitating: noisy commands, and the command taken pulled toward by how much better than usual the result matched the expectation. Measured over a day: it makes the mirror worse (pitch distance 2.3 channels with the frame critic, 2.0 with the pitch critic, 2.4 without any reinforcement but with less babble; the advantage against a global baseline pulls the mapping toward the noise), so it is off; the code stays for the next critic
EXPLORE = 0.04  # standard deviation of the smooth noise on the commands during an imitation bout (the variability LMAN injects)
REFINE_GAIN = 0.5  # weight of the advantage nudge against the mirror's plain association
CRITIC = "pitch"  # what the critic compares when REFINE is on: "frame", the whole heard frame against the expected one; "pitch", where the loudest channel sits and how loud
CUES = 24  # cues the parrot keeps for replay
AFTERGLOW = 30  # frames after a sound ends before its cue is kept for replay
ETA, DECAY = 5.0, 1e-4  # the memory's rate and leak per update (offline, a day: frequent sounds recall 0.99, rare 0.94; at 1.0, 0.97 / 0.87; at 0.3 and 3e-4, 0.94 / 0.70)
MIRROR_ETA, MIRROR_DECAY, MIRROR_MOMENTUM = 0.3, 1e-4, 0.9  # the mirror's (offline on held-out babble: tension r 0.96; the high notes need the low decay, at 3e-4 the mirror tops out at 1900 Hz)


def write_wav(path: Path, sound: np.ndarray) -> None:
    with wavefile.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes((np.clip(sound, -1, 1) * 32767).astype("<i2").tobytes())


def schedule(rng: np.random.Generator, minutes: float, frequent: tuple[str, ...] = FREQUENT, rare: tuple[str, ...] = RARE) -> list[tuple[int, str]]:
    """(start sample, name) of every household sound of the day."""
    names = [*frequent, *rare]
    weights = np.array([1.0] * len(frequent) + [RARE_WEIGHT] * len(rare))
    weights /= weights.sum()
    out, t = [], rng.uniform(*GAP)
    while t < minutes * 60:
        out.append((int(t * SR) // FRAME * FRAME, str(rng.choice(names, p=weights))))
        t += rng.uniform(*GAP) + 2.0
    return out


class Household:
    """Streams the day's sound, frame by frame."""

    def __init__(self, events: list[tuple[int, str]], library: dict[str, np.ndarray]) -> None:
        self.events, self.library = events, library
        self.spans = [(s, s + len(library[n]), n) for s, n in events]

    def frame(self, t: int) -> tuple[np.ndarray, str | None]:
        start = t * FRAME
        out = np.zeros(FRAME)
        name = None
        for s, e, n in self.spans:
            if s < start + FRAME and e > start:
                a, b = max(s, start), min(e, start + FRAME)
                out[a - start : b - start] += self.library[n][a - s : b - s]
                name = n
        return out, name


def stretched_to(original: np.ndarray, produced: np.ndarray) -> np.ndarray:
    idx = np.linspace(0, len(produced) - 1, len(original))
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, len(produced) - 1)
    frac = (idx - lo)[:, None]
    return produced[lo] * (1 - frac) + produced[hi] * frac


def similarity(original: np.ndarray, produced: np.ndarray) -> float:
    """Correlation of two cochleagrams after stretching the produced one to the original's length."""
    if len(produced) < 2:
        return 0.0
    stretched = stretched_to(original, produced)
    a, b = original.ravel() - original.mean(), stretched.ravel() - stretched.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def shape_similarity(original: np.ndarray, produced: np.ndarray) -> float:
    """The same correlation with each frame's mean level removed: the spectral shape over time, not the loudness envelope.

    A broadband voice correlates with anything on the plain measure because both are loud when
    the other is loud; this one asks whether the energy sits in the same channels."""
    if len(produced) < 2:
        return 0.0
    stretched = stretched_to(original, produced)
    a = original - original.mean(axis=1, keepdims=True)
    b = stretched - stretched.mean(axis=1, keepdims=True)
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def pitch_similarity(original: np.ndarray, produced: np.ndarray, loud: float = 0.3) -> tuple[float, float, float]:
    """The pitch track of the imitation against the original's over the frames where both are loud.

    Returns the correlation of the dominant cochlear channel (does the melody go where the
    original's goes; zero when the original holds one pitch, where there is nothing to
    correlate), the mean distance between the dominant channels in channels (about a
    seventh of an octave each), and the share of frames where both are loud."""
    if len(produced) < 2:
        return 0.0, float(CHANNELS), 0.0
    stretched = stretched_to(original, produced)
    both = (original.max(axis=1) > loud) & (stretched.max(axis=1) > loud)
    if both.sum() < 5:
        return 0.0, float(CHANNELS), float(both.mean())
    a, b = original[both].argmax(axis=1).astype(float), stretched[both].argmax(axis=1).astype(float)
    distance = float(np.abs(a - b).mean())
    a, b = a - a.mean(), b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return (float((a * b).sum() / denom) if denom > 0 else 0.0), distance, float(both.mean())


def rhythm_similarity(original: np.ndarray, produced: np.ndarray) -> float:
    """Correlation of the loudness envelopes: are the notes and the gaps where the original's are."""
    if len(produced) < 2:
        return 0.0
    a, b = original.max(axis=1), stretched_to(original, produced).max(axis=1)
    a, b = a - a.mean(), b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


class Parrot:
    def __init__(self, seed: int, backend: str = "cpu") -> None:
        self.rng = np.random.default_rng(seed)
        self.brain = Brain(seed, eta=ETA, decay=DECAY, mirror_eta=MIRROR_ETA, mirror_decay=MIRROR_DECAY, mirror_momentum=MIRROR_MOMENTUM, backend=backend)
        self.syrinx = Syrinx()
        self.cochlea = Cochlea()
        self.history = [np.zeros(CHANNELS) for _ in range(MIRROR_WINDOW)]  # cochlear frames heard, newest last
        self.arousal = 0.0
        self.bout: dict[str, Any] | None = None
        self.episode: dict[str, Any] | None = None  # the household sound playing now: its onset, its cue once 80 ms are in, what it reported at each tick
        self.cues: list[dict[str, Any]] = []
        self.quiet_frames = 0
        self.bouts: list[dict[str, Any]] = []
        self.mismatch = 0.05  # running mean of how far the parrot's own sound falls from what its memory expected
        self.refinements = 0

    # --- vocal behaviour
    def start_bout(self, t: int, familiar: dict[str, float]) -> None:
        babble = not self.cues or self.rng.random() < BABBLE_SHARE
        if babble:
            n = int(self.rng.integers(*BABBLE_FRAMES))
            self.bout = {"kind": "babble", "start": t, "left": n, "state": np.array([self.rng.uniform(0.2, 0.8), 0.7, self.rng.uniform(0.3, 0.7)]), "produced": []}
            return
        weights = np.array([np.exp(-8.0 * familiar.get(c["name"], 1.0)) for c in self.cues])
        cue = self.cues[int(self.rng.choice(len(self.cues), p=weights / weights.sum()))]
        expected, commands = plan(self.brain, cue["cue"])
        self.bout = {"kind": "imitate", "start": t, "left": len(commands), "cue": cue["name"], "expected": expected, "commands": commands, "produced": [], "noise": np.zeros(3), "trace": []}

    def command(self) -> dict[str, float]:
        """The next frame's muscle command from the current bout: smooth noise, or the memory through the mirror."""
        b = self.bout
        assert b is not None
        if b["left"] <= 0:  # closing: the air sac shut for a few frames while the parrot still hears itself fade
            b["closing"] = b.get("closing", CLOSING) - 1
            last = b.get("last", {"tension": 0.5, "pressure": 0.0, "tract": 0.5})
            return {"tension": last["tension"], "pressure": 0.0, "tract": last["tract"]}
        if b["kind"] == "babble":
            drift = self.rng.normal(size=3) * np.array([0.04, 0.03, 0.03])
            target = b.setdefault("target", np.array([0.5, 0.75, 0.5]))
            b["state"] = np.clip(b["state"] + 0.05 * (target - b["state"]) + drift, 0.0, 1.0)
            if len(b["produced"]) % SYLLABLE == 0:  # a new syllable: a fresh pitch, breath and tract setting
                b["state"][0], b["state"][2] = self.rng.uniform(0.05, 0.95), self.rng.uniform(0.1, 0.9)
                target[1] = self.rng.uniform(0.35, 1.0)  # soft, pure notes as well as loud, buzzing ones
            tension, pressure, tract = b["state"]
            if b["left"] < 6:
                pressure *= b["left"] / 6.0  # close the air sac at the end of the bout
            b["last"] = {"tension": float(tension), "pressure": float(pressure), "tract": float(tract)}
            return b["last"]
        k = len(b["produced"])
        motor = dict(b["commands"][k])
        if REFINE:  # smooth exploratory noise on the command, and a trace of what the mirror saw and expected
            b["noise"] = 0.8 * b["noise"] + EXPLORE * self.rng.normal(size=3)
            motor = {key: float(np.clip(v + d, 0.0, 1.0)) for (key, v), d in zip(motor.items(), b["noise"], strict=True)}
            b["trace"].append({"seen": b["expected"][max(0, k - MIRROR_WINDOW + 1) : k + 1], "expected": b["expected"][k], "command": motor})
        b["last"] = motor
        return motor

    # --- one chunk of the day
    def step(self, t0: int, house: Household, familiar: dict[str, float]) -> None:
        chunk_commands: list[dict[str, float] | None] = []
        kinds: list[str | None] = []
        traces: list[tuple[dict[str, Any], np.ndarray]] = []
        for t in range(t0, t0 + CHUNK):
            env, name = house.frame(t)
            if self.bout is None and self.arousal > 1.0 and name is None:
                self.start_bout(t, familiar)
            singing = self.bout is not None
            if singing:
                assert self.bout is not None
                traced = len(self.bout.get("trace", []))
                kinds.append(self.bout["kind"])
                cmd = self.command()
                sound = self.syrinx.frame(cmd["tension"], cmd["pressure"], cmd["tract"])
                self.bout["produced"].append(sound)
                pending = self.bout["trace"][-1] if self.bout.get("trace") and len(self.bout["trace"]) > traced else None
                if self.bout["left"] > 0:
                    self.bout["left"] -= 1
                chunk_commands.append(cmd)
                heard = env + sound
                if self.bout["left"] <= 0 and self.bout.get("closing", CLOSING) <= 0:
                    self.finish_bout(t)
            else:
                chunk_commands.append(None)
                kinds.append(None)
                heard = env
                pending = None
            levels = self.cochlea.frame(heard)
            self.history.append(levels)
            if pending is not None:
                traces.append((pending, levels))
            loud = levels.max() > LOUD
            # the episode: a sound that began after a quiet spell starts the clock; its cue once 160 ms are in; what the
            # cochlea reports at each tick until the clock runs out, the quiet after the sound included (how a sound ends,
            # and that nothing follows, is part of the memory; without it the ticks after a sound answer with other sounds)
            if name is not None and self.episode is None and self.quiet_frames >= QUIET and not singing:
                self.episode = {"onset": t, "name": name, "ticks": [], "after": 0}
            if self.episode is not None:
                e = self.episode
                phase = t - e["onset"]
                e["after"] = e["after"] + 1 if name is None else 0
                e["ticks"].append((phase, levels, singing))  # the parrot's own voice is kept out of the memory
                if phase == WINDOW - 1:
                    e["cue"] = cue_of(np.stack([lv for _, lv, _ in e["ticks"]]))
                if e["after"] == AFTERGLOW and "cue" in e:  # the sound is over: keep its cue for replay
                    self.cues.append({"name": e["name"], "cue": e["cue"], "time": e["onset"]})
                    self.cues = self.cues[-CUES:]
                if phase >= MAX_PHASE - 1:
                    self.episode = None
            self.quiet_frames = 0 if loud else self.quiet_frames + 1
            if name is None and not singing:
                self.arousal += (AROUSAL_RATE + AROUSAL_NOISE * self.rng.normal()) * FRAME / SR
            else:
                self.arousal = max(0.0, self.arousal - 0.5 * FRAME / SR)
        self.learn(chunk_commands, kinds, traces)
        self.history = self.history[-(MIRROR_WINDOW + CHUNK + 2) :]

    def learn(self, chunk_commands: list[dict[str, float] | None], kinds: list[str | None], traces: list[tuple[dict[str, Any], np.ndarray]]) -> None:
        """What this chunk teaches: the memory its ticks, the mirror its babble, the critic its imitation."""
        e = self.episode
        if e is not None and "cue" in e:  # the memory: every tick heard so far that it has not learned, against the cue
            rows = [(phase, levels) for phase, levels, own in e["ticks"][e.get("learned", 0) :] if not own]
            e["learned"] = len(e["ticks"])
            if rows:
                self.brain.learn_memory(np.repeat(e["cue"][None], len(rows), axis=0), np.array([p for p, _ in rows]), np.stack([lv for _, lv in rows]))
        n = len(self.history)
        own = [k for k, cmd in enumerate(chunk_commands) if cmd is not None and k >= 1 and chunk_commands[k - 1] is not None and kinds[k] == "babble"]
        if own:  # the mirror: the frames heard now against the command issued one frame earlier, from babbling only
            # (an imitation bout's commands are the mirror's own answers; learning from them fixes it at whatever it says)
            windows = np.stack([np.stack(self.history[n - CHUNK + k - MIRROR_WINDOW + 1 : n - CHUNK + k + 1]) for k in own])
            cmds = {key: np.array([chunk_commands[k - 1][key] for k in own]) for key in ("tension", "pressure", "tract")}  # type: ignore[index]
            self.brain.learn_inverse(windows, cmds)
        if REFINE and traces:  # reinforcement: each command taken, weighted by how much better than usual its sound matched the expectation
            seen = np.stack([np.concatenate([np.zeros((MIRROR_WINDOW - len(tr["seen"]), CHANNELS)), tr["seen"]]) for tr, _ in traces])
            heard = np.stack([h for _, h in traces])
            mismatch = mismatch_of(heard, np.stack([tr["expected"] for tr, _ in traces]))
            advantage = (self.mismatch - mismatch) / (self.mismatch + 1e-3)
            self.mismatch = 0.98 * self.mismatch + 0.02 * float(mismatch.mean())
            cmds = {key: np.array([tr["command"][key] for tr, _ in traces]) for key in ("tension", "pressure", "tract")}
            self.brain.reinforce(seen, cmds, REFINE_GAIN * np.clip(advantage, -1.0, 1.0))
            self.refinements += len(traces)

    def finish_bout(self, t: int) -> None:
        b = self.bout
        assert b is not None
        produced = np.concatenate(b["produced"]) if b["produced"] else np.zeros(FRAME)
        entry = {"kind": b["kind"], "start_s": b["start"] * FRAME / SR, "frames": len(b["produced"]), "cue": b.get("cue")}
        self.bouts.append(entry)
        self.last_produced = (entry, produced)
        self.bout = None
        self.arousal = 0.0


def mismatch_of(heard: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """How far each heard frame falls from the frame the memory expected, per row.

    "frame": the mean squared difference of the two frames. "pitch": the distance of the loudest channels (in
    channels over the range) plus the difference in loudness; the heard frame is the cochlea's, broad with harmonics,
    the expected one is sharpened, and the squared difference of the two rewards a quiet rendition more than a
    rendition at the right pitch."""
    if CRITIC == "pitch":
        loud_heard, loud_expected = heard.max(axis=1), expected.max(axis=1)
        pitch = np.abs(heard.argmax(axis=1) - expected.argmax(axis=1)) / CHANNELS
        both = (loud_heard >= FAINT_LEVEL) & (loud_expected >= FAINT_LEVEL)
        return np.where(both, pitch, 0.0) + np.abs(loud_heard - loud_expected)
    return ((heard - expected) ** 2).mean(axis=1)


FAINT_LEVEL = 0.1  # below this a frame has no pitch to compare


def familiarity(brain: Brain, cochleagrams: dict[str, np.ndarray]) -> dict[str, float]:
    """Error of the memory's expectation at each tick of each clean sound: low where the sound is known."""
    out = {}
    for name, frames in cochleagrams.items():
        cue = cue_of(frames)
        out[name] = float(brain.prediction_error(np.repeat(cue[None], len(frames), axis=0), np.arange(len(frames)), frames).mean())
    return out


def recall(brain: Brain, frames: np.ndarray) -> tuple[float, np.ndarray]:
    """Replay the memory from the sound's cue and compare with the sound: memory of that sound in particular."""
    replayed = brain.replay(cue_of(frames), len(frames))
    return similarity(frames, replayed), replayed


def plan(brain: Brain, cue: np.ndarray) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Sing from a cue: the clock runs, the memory replays (the template), each expectation is sharpened and goes
    through the mirror to a command. The bout lasts until the memory expects nothing more: a gap inside a sound
    (the beeps' pauses, the phone's) is quiet the mirror answers with a closed air sac, not the end."""
    expected = sharpen(brain.replay(cue, REPLAY_FRAMES))
    expected = expected[: bout_end(expected.max(axis=1))]
    motor = brain.motor(Brain.windows_of(expected))
    commands, last = [], None
    for k in range(len(expected)):  # the muscles are fast but not instant: each command settles over a frame or two
        cmd = {key: float(motor[key][k]) for key in ("tension", "pressure", "tract")}
        if last is not None:
            cmd = {key: (1 - MUSCLE) * last[key] + MUSCLE * cmd[key] for key in cmd}
        commands.append(cmd)
        last = cmd
    return expected, commands


def bout_end(levels: np.ndarray) -> int:
    """Where a replay ends, given the loudest expected level at each tick: a few closing frames into the first REST ticks of
    quiet after the sound began (quiet is below QUIET_SHARE of the loudest level so far); the horizon if the memory never falls quiet."""
    peak, quiet = 0.0, 0
    for k, level in enumerate(levels):
        peak = max(peak, float(level))
        if peak >= LOUD and level < QUIET_SHARE * peak:
            quiet += 1
            if quiet >= REST:
                return k - REST + 1 + CLOSING
        else:
            quiet = 0
    return len(levels)


def imitate(brain: Brain, frames: np.ndarray) -> tuple[np.ndarray, list[dict[str, float]]]:
    """The sound the parrot makes from a sound's cue, and the commands."""
    _, commands = plan(brain, cue_of(frames))
    syrinx = Syrinx()
    sounds = [syrinx.frame(c["tension"], c["pressure"], c["tract"]) for c in commands]
    return (np.concatenate(sounds) if sounds else np.zeros(FRAME)), commands


def evaluate(brain: Brain, library: dict[str, np.ndarray], cochleagrams: dict[str, np.ndarray]) -> dict[str, Any]:
    fam = familiarity(brain, cochleagrams)
    out = {}
    for name, frames in cochleagrams.items():
        rec, replayed = recall(brain, frames)
        sound, commands = imitate(brain, frames)
        produced = Cochlea().frames(sound) if len(sound) >= FRAME else np.zeros((1, CHANNELS))
        pitch, distance, covered = pitch_similarity(frames, produced)
        out[name] = {"familiarity_error": fam[name], "recall": rec, "recall_frames": int(len(replayed)), "imitation_similarity": similarity(frames, produced), "imitation_shape": shape_similarity(frames, produced), "imitation_pitch": pitch, "imitation_pitch_distance": distance, "imitation_rhythm": rhythm_similarity(frames, produced), "imitation_covered": covered, "imitation_frames": int(len(produced)), "commands": commands, "sound": sound}
    return out


def live(seed: int, minutes: float, frequent: tuple[str, ...], rare: tuple[str, ...], library: dict[str, np.ndarray], cochleagrams: dict[str, np.ndarray], label: str) -> tuple[Parrot, dict[str, int], list[dict[str, Any]], float, int]:
    """One parrot through one day; returns it with the counts, the familiarity timeline, the wall-clock, and the event count."""
    rng = np.random.default_rng(seed)
    events = schedule(rng, minutes, frequent, rare)
    house = Household(events, library)
    counts = {name: sum(1 for _, n in events if n == name) for name in library}
    print(f"{label}: a {minutes:.0f}-minute day with {len(events)} household sounds: {counts}", flush=True)
    parrot = Parrot(seed)
    total = int(minutes * 60 * SR / FRAME)
    t0 = time.perf_counter()
    familiar = familiarity(parrot.brain, cochleagrams)
    timeline = []
    for start in range(0, total - CHUNK, CHUNK):
        parrot.step(start, house, familiar)
        if start % (CHUNK * 1500) == 0 and start:  # every two minutes: how familiar is each sound now
            familiar = familiarity(parrot.brain, cochleagrams)
            timeline.append({"minute": start * FRAME / SR / 60, "familiarity_error": familiar, "bouts": len(parrot.bouts)})
            if len(timeline) % 5 == 0:
                kinds = [b["kind"] for b in parrot.bouts]
                print(f"  minute {timeline[-1]['minute']:4.1f}: familiarity {' '.join(f'{k} {v:.3f}' for k, v in familiar.items())}; bouts {len(kinds)} ({kinds.count('imitate')} imitations) ({time.perf_counter() - t0:.0f}s)", flush=True)
    return parrot, counts, timeline, time.perf_counter() - t0, len(events)


def run(seed: int, minutes: float, out: Path, tag: str = "") -> dict[str, Any]:
    library = waves()
    cochleagrams = {name: Cochlea().frames(w) for name, w in library.items()}
    untrained = evaluate(Brain(seed + 1), library, cochleagrams)
    # day A: the household as described; day B: the roles swapped, so every sound is measured once heard often and once rarely
    swapped_frequent, swapped_rare = RARE, FREQUENT
    parrot, counts, timeline, seconds, events = live(seed, minutes, FREQUENT, RARE, library, cochleagrams, "day A")
    trained = evaluate(parrot.brain, library, cochleagrams)
    parrot_b, counts_b, timeline_b, seconds_b, events_b = live(seed + 7, minutes, swapped_frequent, swapped_rare, library, cochleagrams, "day B (roles swapped)")
    trained_b = evaluate(parrot_b.brain, library, cochleagrams)
    (out.parent / "imitations").mkdir(exist_ok=True)
    for name in library:
        write_wav(out.parent / "imitations" / f"{name}_original.wav", library[name])
        write_wav(out.parent / "imitations" / f"{name}_imitation{tag}.wav", trained[name]["sound"])
    imitations = {name: {"commands": r["commands"], "similarity": r["imitation_similarity"]} for name, r in trained.items()}
    (out.parent / f"imitations{tag}.json").write_text(json.dumps(imitations, separators=(",", ":")))
    net = parrot.brain.export()
    net["cues"] = {name: cue_of(frames).round(4).tolist() for name, frames in cochleagrams.items()}
    net["sounds"] = {name: {"frames": int(len(f)), "heard": counts[name], "frequent": name in FREQUENT} for name, f in cochleagrams.items()}
    net["seed"] = seed
    (out.parent / f"net{tag}.json").write_text(json.dumps(net, separators=(",", ":")))
    (out.parent / f"bouts{tag}.json").write_text(json.dumps(parrot.bouts, separators=(",", ":")))
    per_sound = {}
    for name in library:
        often, rarely = (trained[name], trained_b[name]) if name in FREQUENT else (trained_b[name], trained[name])
        heard_often, heard_rarely = (counts[name], counts_b[name]) if name in FREQUENT else (counts_b[name], counts[name])
        per_sound[name] = {
            "heard_often": heard_often, "heard_rarely": heard_rarely, "frequent_on_day_a": name in FREQUENT,
            "recall_heard_often": often["recall"], "recall_heard_rarely": rarely["recall"], "recall_untrained": untrained[name]["recall"],
            "imitation_heard_often": often["imitation_similarity"], "imitation_heard_rarely": rarely["imitation_similarity"], "imitation_untrained": untrained[name]["imitation_similarity"],
            "shape_heard_often": often["imitation_shape"], "shape_heard_rarely": rarely["imitation_shape"], "shape_untrained": untrained[name]["imitation_shape"],
            "pitch_heard_often": often["imitation_pitch"], "pitch_heard_rarely": rarely["imitation_pitch"], "pitch_untrained": untrained[name]["imitation_pitch"],
            "pitch_distance_heard_often": often["imitation_pitch_distance"], "pitch_distance_heard_rarely": rarely["imitation_pitch_distance"], "pitch_distance_untrained": untrained[name]["imitation_pitch_distance"],
            "rhythm_heard_often": often["imitation_rhythm"], "rhythm_heard_rarely": rarely["imitation_rhythm"], "rhythm_untrained": untrained[name]["imitation_rhythm"], "covered_heard_often": often["imitation_covered"],
            "next_frame_error_heard_often": often["familiarity_error"], "next_frame_error_heard_rarely": rarely["familiarity_error"], "imitation_frames_day_a": trained[name]["imitation_frames"],
        }
    mean = lambda key: float(np.mean([r[key] for r in per_sound.values()]))  # noqa: E731
    summary = {
        "recall_heard_often": mean("recall_heard_often"), "recall_heard_rarely": mean("recall_heard_rarely"), "recall_untrained": mean("recall_untrained"),
        "imitation_heard_often": mean("imitation_heard_often"), "imitation_heard_rarely": mean("imitation_heard_rarely"), "imitation_untrained": mean("imitation_untrained"),
        "shape_heard_often": mean("shape_heard_often"), "shape_heard_rarely": mean("shape_heard_rarely"), "shape_untrained": mean("shape_untrained"),
        "pitch_heard_often": mean("pitch_heard_often"), "pitch_heard_rarely": mean("pitch_heard_rarely"), "pitch_untrained": mean("pitch_untrained"), "covered_heard_often": mean("covered_heard_often"),
        "pitch_distance_heard_often": mean("pitch_distance_heard_often"), "pitch_distance_heard_rarely": mean("pitch_distance_heard_rarely"), "pitch_distance_untrained": mean("pitch_distance_untrained"),
        "rhythm_heard_often": mean("rhythm_heard_often"), "rhythm_heard_rarely": mean("rhythm_heard_rarely"), "rhythm_untrained": mean("rhythm_untrained"),
        "repetition_effect_on_recall": mean("recall_heard_often") - mean("recall_heard_rarely"),
        "sounds_recalled_better_when_heard_often": int(sum(r["recall_heard_often"] > r["recall_heard_rarely"] for r in per_sound.values())),
        "bouts": len(parrot.bouts), "imitation_bouts": sum(1 for b in parrot.bouts if b["kind"] == "imitate"), "babble_bouts": sum(1 for b in parrot.bouts if b["kind"] == "babble"), "reinforced_frames": parrot.refinements,
        # the day-A household, for the page and the ladder
        "frequent_recall": float(np.mean([trained[n]["recall"] for n in FREQUENT])), "rare_recall": float(np.mean([trained[n]["recall"] for n in RARE])), "untrained_recall": mean("recall_untrained"),
        "frequent_imitation_similarity": float(np.mean([trained[n]["imitation_similarity"] for n in FREQUENT])), "rare_imitation_similarity": float(np.mean([trained[n]["imitation_similarity"] for n in RARE])), "untrained_imitation_similarity": mean("imitation_untrained"),
    }
    print(f"recall when heard often {summary['recall_heard_often']:.3f}, when heard rarely {summary['recall_heard_rarely']:.3f}, untrained {summary['recall_untrained']:.3f}; imitation {summary['imitation_heard_often']:.3f} / {summary['imitation_heard_rarely']:.3f} / {summary['imitation_untrained']:.3f}; spectral shape {summary['shape_heard_often']:.3f} / {summary['shape_heard_rarely']:.3f} / {summary['shape_untrained']:.3f}; pitch track {summary['pitch_heard_often']:.3f} / {summary['pitch_heard_rarely']:.3f} / {summary['pitch_untrained']:.3f}; pitch distance {summary['pitch_distance_heard_often']:.2f} / {summary['pitch_distance_heard_rarely']:.2f} / {summary['pitch_distance_untrained']:.2f} channels; rhythm {summary['rhythm_heard_often']:.3f} / {summary['rhythm_heard_rarely']:.3f} / {summary['rhythm_untrained']:.3f}; {summary['sounds_recalled_better_when_heard_often']} of {len(per_sound)} sounds recalled better when heard often", flush=True)
    for name, r in per_sound.items():
        print(f"  {name:9s} heard {r['heard_often']:3d} / {r['heard_rarely']:2d}: recall {r['recall_heard_often']:.3f} / {r['recall_heard_rarely']:.3f} (untrained {r['recall_untrained']:.3f}), imitation {r['imitation_heard_often']:.3f} / {r['imitation_heard_rarely']:.3f} (untrained {r['imitation_untrained']:.3f}), shape {r['shape_heard_often']:.3f} / {r['shape_heard_rarely']:.3f}, pitch {r['pitch_heard_often']:.3f} / {r['pitch_heard_rarely']:.3f} (distance {r['pitch_distance_heard_often']:.2f} / {r['pitch_distance_heard_rarely']:.2f}), rhythm {r['rhythm_heard_often']:.3f} / {r['rhythm_heard_rarely']:.3f}, {r['imitation_frames_day_a']} frames", flush=True)
    body = {
        "day": {"minutes": minutes, "events": events, "heard": counts, "gap_seconds": GAP, "rare_weight": RARE_WEIGHT, "frequent": list(FREQUENT), "rare": list(RARE)},
        "day_b": {"events": events_b, "heard": counts_b, "frequent": list(swapped_frequent), "rare": list(swapped_rare), "seconds": seconds_b, "bouts": len(parrot_b.bouts), "timeline": timeline_b},
        "chunk_frames": CHUNK, "window_frames": WINDOW, "mirror_window_frames": MIRROR_WINDOW, "max_phase_frames": MAX_PHASE,
        "brain": {"owners": int(parrot.brain.wiring.n), "hidden": int(parrot.brain.wiring.n - parrot.brain.inputs - parrot.brain.outputs), "parameters": int(parrot.brain.parameters()), "eta": ETA, "decay": DECAY, "mirror_eta": MIRROR_ETA, "mirror_decay": MIRROR_DECAY, "mirror_momentum": MIRROR_MOMENTUM, "learner": parrot.brain.memory_config.to_dict(), "mirror_learner": parrot.brain.mirror_config.to_dict()},
        "arousal": {"rate_per_second": AROUSAL_RATE, "noise": AROUSAL_NOISE, "babble_share": BABBLE_SHARE, "babble_frames": list(BABBLE_FRAMES)}, "reinforcement": {"on": REFINE, "explore": EXPLORE, "gain": REFINE_GAIN, "critic": CRITIC}, "per_sound": per_sound, "summary": summary, "timeline": timeline, "seconds": seconds,
        "boundary": {"learning_rule": "free/nudged contrastive Hebbian, centered, owner-local, quadratic nudges on two output groups of one net, each population's seams decaying on its own updates", "memory_learns_from": "a household sound that began after 200 ms of quiet: its first 160 ms, after lateral inhibition, held as the cue, and the cochlear frame at each tick of a clock that starts at its onset and runs for 2 s, the quiet after the sound included; gated off while the parrot sings (as auditory responses in the song system are)", "mirror_learns_from": "the parrot's own sound during babbling bouts (the last 40 ms, as heard and after lateral inhibition) against the command issued one frame earlier, including the closed frames that end a babble; never from imitation bouts, whose commands are the mirror's own answers", "reinforcement": "off: measured over a day, an advantage-weighted nudge on the command taken (against a running mean of the mismatch between the heard frame and the expected one) made the mirror worse; the babbling is what teaches it", "names_never_reach_the_brain": True, "cues": "the first 160 ms of a sound that began after a quiet spell, up to 24 kept; replay draws among them by familiarity", "two_days": "day B swaps which sounds are frequent, with a fresh parrot and another seed, so each sound is scored once heard often and once heard rarely", "muscles": "each command is taken on at half its difference per frame (a 20 ms time constant), so a single wrong frame from the mirror does not crack the voice", "bout_ends": "a few closing frames into the first 250 ms in which the memory expects less than 30% of the loudest level it has replayed so far (it learned the quiet after every sound); gaps inside a sound are shorter, and the mirror answers them with a closed air sac", "imitation_shape": "the same correlation with each frame's mean level removed, so a loud broadband voice does not score on loudness alone", "imitation_pitch": "correlation of the dominant cochlear channel over the frames where both are loud (does the melody go where the original's goes; zero where the original holds one pitch), and the mean distance between the dominant channels in channels", "imitation_rhythm": "correlation of the loudness envelopes", "evaluation": "recall: the memory replayed from each sound's cue with the clock run from zero, against the sound itself (correlation of cochleagrams); imitation: the same replay, each expectation sharpened and put through the mirror into a fresh syrinx, the produced sound heard through a cochlea and scored against the sound; the untrained brain is the control"},
    }
    r = cd.Receipt.build("cadence-examples/10_parrot/v2", body, sources=SOURCES)
    r.write(out)
    print(f"receipt {out} ({r.digest[:16]}...)")
    return body


def check(body: dict) -> str | None:
    s = body["summary"]
    per = body["per_sound"]
    if abs(float(np.mean([r["recall_heard_often"] for r in per.values()])) - s["recall_heard_often"]) > 1e-9:
        return "mean recall does not follow from the sounds"
    if not (-1 <= s["frequent_imitation_similarity"] <= 1):
        return "similarity out of range"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--minutes", type=float, default=MINUTES)
    parser.add_argument("--tag", default="", help="suffix for a second parrot's net, bouts and imitations (e.g. _1)")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        ok, message = cd.Receipt.verify(args.verify, sources=SOURCES, check=check)
        print(message)
        return 0 if ok else 1
    run(args.seed, args.minutes, args.output or HERE / f"receipt{args.tag}.json", args.tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
