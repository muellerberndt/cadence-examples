"""An African grey in a household: a day of listening, babbling and imitating, then a receipt.

Run:  python train.py                       (two simulated hours, about twenty minutes; writes receipt.json, net.json, bouts.json, imitations/*.wav)
      python train.py --seed 1 --tag _1     (a second parrot for the page: receipt_1.json, net_1.json, ...)
      python train.py --verify receipt.json

The parrot hears the household through its cochlea. Three sounds recur many times a day
(a doorbell, a phone, a microwave), two are rare (a whistle, a siren). One equilibrium
net is its whole brain (see brain.py): the memory group learns to expect the next
cochlear frame of whatever it hears, and the seams decay, so repetition decides what
stays. When the house is quiet an arousal integrator fills and the parrot opens its
syrinx: either to babble (smooth random muscle commands, the exploratory variability
of the anterior forebrain pathway) or, from a cue it noticed at a sound's onset, to
replay a memory and sing it. Hearing its own voice, the same net learns the mirror:
the command that produced the sound it hears. That inverse model is what turns a
memory into muscle commands. The receipt scores, per sound, how familiar the memory is
(next-frame error) and how well the imitation matches the original (correlation of
cochleagrams), against an untrained brain.
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

from brain import NEED, WINDOW, Brain, advance, context
from syrinx import CHANNELS, FRAME, FREQUENT, RARE, SR, Cochlea, Syrinx, pressure_of, waves

HERE = Path(__file__).resolve().parent
SOURCES = [("10_parrot/train.py", Path(__file__).resolve()), ("10_parrot/brain.py", HERE / "brain.py"), ("10_parrot/syrinx.py", HERE / "syrinx.py")]
MINUTES = 60
CHUNK = 8  # frames the brain takes in at once: 80 ms
GAP = (3.0, 9.0)  # seconds between household sounds
RARE_WEIGHT = 0.06  # a rare sound's chance against a frequent one's at each event
LOUD = 0.25  # a cochlear level above this is sound
AROUSAL_RATE, AROUSAL_NOISE = 0.08, 0.05  # per second of quiet; a bout starts past 1
BABBLE_SHARE = 0.5  # of bouts, before any cue exists every bout babbles
BABBLE_FRAMES = (60, 140)
REPLAY_FRAMES = 160
CLOSED = 6  # frames of the mirror asking for no pressure, or of the memory expecting quiet, that end a bout
CLOSING = 8  # frames of closed air sac the parrot adds at the end of every bout, so the mirror learns that quiet means closed
REFINE = True  # reinforcement while imitating: noisy commands, and the command taken pulled toward by how much better than usual the result matched the expectation
EXPLORE = 0.04  # standard deviation of the smooth noise on the commands during an imitation bout (the variability LMAN injects)
REFINE_GAIN = 0.5  # weight of the advantage nudge against the mirror's plain association
CUES = 24  # onset contexts the parrot keeps as replay cues
AFTERGLOW = 30  # frames after a sound ends that the memory still learns from: how a sound stops is part of it
ETA, DECAY = 0.03, 3e-4  # two-day evaluations: at 0.3 recall 0.18; at 0.03 a sound recalls 0.55 after a day that repeated it and 0.32 after one that did not; at 0.01 0.59 and 0.53 (it keeps rare sounds too)


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


class Parrot:
    def __init__(self, seed: int, backend: str = "cpu") -> None:
        self.rng = np.random.default_rng(seed)
        self.brain = Brain(seed, eta=ETA, decay=DECAY, backend=backend)
        self.syrinx = Syrinx()
        self.cochlea = Cochlea()
        self.history = [np.zeros(CHANNELS) for _ in range(NEED)]  # cochlear frames heard, newest last
        self.afterglow = 0
        self.arousal = 0.0
        self.bout: dict[str, Any] | None = None
        self.cues: list[dict[str, Any]] = []
        self.quiet_frames = 0
        self.onset: tuple[int, str] | None = None
        self.commands: list[dict[str, float]] = []  # motor commands issued, one per frame, for the efference copy
        self.bouts: list[dict[str, Any]] = []
        self.mismatch = 0.05  # running mean of how far the parrot's own sound falls from what its memory expected
        self.refinements = 0

    def window(self, end: int | None = None) -> np.ndarray:
        return context(self.history, end)

    # --- vocal behaviour
    def start_bout(self, t: int, familiar: dict[str, float]) -> None:
        babble = not self.cues or self.rng.random() < BABBLE_SHARE
        if babble:
            n = int(self.rng.integers(*BABBLE_FRAMES))
            self.bout = {"kind": "babble", "start": t, "left": n, "state": np.array([self.rng.uniform(0.2, 0.8), 0.7, self.rng.uniform(0.3, 0.7)]), "produced": []}
        else:
            weights = np.array([np.exp(-8.0 * familiar.get(c["name"], 1.0)) for c in self.cues])
            cue = self.cues[int(self.rng.choice(len(self.cues), p=weights / weights.sum()))]
            self.bout = {"kind": "imitate", "start": t, "left": REPLAY_FRAMES, "cue": cue["name"], "window": cue["window"].copy(), "tail": list(cue["tail"]), "closed": 0, "produced": [], "noise": np.zeros(3), "trace": []}

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
            target = np.array([0.5, 0.75, 0.5])
            b["state"] = np.clip(b["state"] + 0.05 * (target - b["state"]) + drift, 0.0, 1.0)
            tension, pressure, tract = b["state"]
            if b["left"] < 6:
                pressure *= b["left"] / 6.0  # close the air sac at the end of the bout
            b["last"] = {"tension": float(tension), "pressure": float(pressure), "tract": float(tract)}
            return b["last"]
        expected, motor, seen = self.brain.imagine(b["window"][None])
        motor = {k: float(v[0]) for k, v in motor.items()}
        if REFINE:  # smooth exploratory noise on the command, and a trace of what the mirror saw and expected
            b["noise"] = 0.8 * b["noise"] + EXPLORE * self.rng.normal(size=3)
            motor = {k: float(np.clip(v + d, 0.0, 1.0)) for (k, v), d in zip(motor.items(), b["noise"], strict=True)}
            b["trace"].append({"seen": seen[0], "expected": expected[0], "command": motor})
        if b["produced"]:  # the context is what the parrot hears of itself, one frame behind; the cue starts it
            b["window"], b["tail"] = advance(b["window"], self.history[-1], b["tail"])
        b["closed"] = b["closed"] + 1 if pressure_of(motor["pressure"]) <= 0.0 or expected[0].max() < LOUD * 0.6 else 0
        if b["closed"] >= CLOSED:
            b["left"] = 0
        b["last"] = motor
        return motor

    # --- one chunk of the day
    def step(self, t0: int, house: Household, familiar: dict[str, float]) -> None:
        frames, names, self_frames = [], [], []
        chunk_commands: list[dict[str, float] | None] = []
        traces: list[tuple[dict[str, Any], np.ndarray]] = []
        for t in range(t0, t0 + CHUNK):
            env, name = house.frame(t)
            if self.bout is None and self.arousal > 1.0 and name is None:
                self.start_bout(t, familiar)
            if self.bout is not None:
                traced = len(self.bout.get("trace", []))
                cmd = self.command()
                sound = self.syrinx.frame(cmd["tension"], cmd["pressure"], cmd["tract"])
                self.bout["produced"].append(sound)
                pending = self.bout["trace"][-1] if self.bout.get("trace") and len(self.bout["trace"]) > traced else None
                if self.bout["left"] > 0:
                    self.bout["left"] -= 1
                chunk_commands.append(cmd)
                heard = env + sound
                self_frames.append(True)
                if self.bout["left"] <= 0 and self.bout.get("closing", CLOSING) <= 0:
                    self.finish_bout(t)
            else:
                chunk_commands.append(None)
                heard = env
                self_frames.append(False)
                pending = None
            levels = self.cochlea.frame(heard)
            self.history.append(levels)
            frames.append(levels)
            names.append(name)
            if self.bout is not None and pending is not None:
                traces.append((pending, levels))
            loud = levels.max() > LOUD
            if name is not None and self.onset is None and self.quiet_frames >= 20:
                self.onset = (t, name)  # a sound began after a quiet spell
            if self.onset is not None and t - self.onset[0] == WINDOW:
                self.cues.append({"name": self.onset[1], "window": self.window(), "tail": [np.array(f) for f in self.history[-NEED:]], "time": t})
                self.cues = self.cues[-CUES:]
                self.onset = None
            self.quiet_frames = 0 if loud else self.quiet_frames + 1
            self.afterglow = AFTERGLOW if name is not None else max(0, self.afterglow - 1)
            names[-1] = name if name is not None else ("afterglow" if self.afterglow > 0 else None)
            if name is None and not self_frames[-1]:
                self.arousal += (AROUSAL_RATE + AROUSAL_NOISE * self.rng.normal()) * FRAME / SR
            else:
                self.arousal = max(0.0, self.arousal - 0.5 * FRAME / SR)
        # learning from this chunk: contexts end at each frame, the frame after is the memory's target
        n = len(self.history)
        idx = list(range(n - CHUNK, n))
        windows = np.stack([context(self.history, i) for i in idx])
        targets = np.stack([self.history[i] for i in idx])
        external = np.array([nm is not None and not sf for nm, sf in zip(names, self_frames, strict=True)])
        if external.any():  # the memory listens to the house, gated off while the parrot sings
            self.brain.learn_memory(windows[external], targets[external])
        own = [k for k, cmd in enumerate(chunk_commands) if cmd is not None and k >= 1 and chunk_commands[k - 1] is not None]
        if own:  # the mirror: the sound heard now against the command issued one frame earlier
            cmds = {key: np.array([chunk_commands[k - 1][key] for k in own]) for key in ("tension", "pressure", "tract")}  # type: ignore[index]
            self.brain.learn_inverse(windows[own], cmds)
        if REFINE and traces:  # reinforcement: each command taken, weighted by how much better than usual its sound matched the expectation
            seen = np.stack([tr["seen"] for tr, _ in traces])
            heard = np.stack([h for _, h in traces])
            mismatch = ((heard - np.stack([tr["expected"] for tr, _ in traces])) ** 2).mean(axis=1)
            advantage = (self.mismatch - mismatch) / (self.mismatch + 1e-3)
            self.mismatch = 0.98 * self.mismatch + 0.02 * float(mismatch.mean())
            cmds = {key: np.array([tr["command"][key] for tr, _ in traces]) for key in ("tension", "pressure", "tract")}
            self.brain.reinforce(seen, cmds, REFINE_GAIN * np.clip(advantage, -1.0, 1.0))
            self.refinements += len(traces)
        self.history = self.history[-(NEED + CHUNK + 2) :]

    def finish_bout(self, t: int) -> None:
        b = self.bout
        assert b is not None
        produced = np.concatenate(b["produced"]) if b["produced"] else np.zeros(FRAME)
        entry = {"kind": b["kind"], "start_s": b["start"] * FRAME / SR, "frames": len(b["produced"]), "cue": b.get("cue")}
        self.bouts.append(entry)
        self.last_produced = (entry, produced)
        self.bout = None
        self.arousal = 0.0


def familiarity(brain: Brain, cochleagrams: dict[str, np.ndarray]) -> dict[str, float]:
    """Next-frame error of the memory on each clean sound heard from quiet: low where the sound is expected."""
    out = {}
    for name, frames in cochleagrams.items():
        padded = np.concatenate([np.zeros((NEED, CHANNELS)), frames])
        windows = np.stack([context(padded, NEED + i) for i in range(len(frames))])
        out[name] = float(brain.prediction_error(windows, frames).mean())
    return out


def onset_cue(frames: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """The context the parrot has 80 ms into a sound that began from quiet, and the frames behind it."""
    tail = [np.zeros(CHANNELS)] * (NEED - WINDOW) + [f for f in frames[:WINDOW]]
    return context(tail), tail


def recall(brain: Brain, frames: np.ndarray) -> tuple[float, np.ndarray]:
    """Replay the memory from the sound's onset cue and compare with the sound: memory of that sound in particular."""
    window, tail = onset_cue(frames)
    expected = []
    for _ in range(len(frames) - WINDOW):
        e = brain.predicted(brain.free(window[None]))[0]
        expected.append(e)
        window, tail = advance(window, e, tail)
    replayed = np.stack(expected) if expected else np.zeros((1, CHANNELS))
    return similarity(frames[WINDOW:], replayed), replayed


def imitate(brain: Brain, cue: np.ndarray, tail: list[np.ndarray]) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Sing from a cue: the memory expects, the mirror commands, the syrinx sounds, the cochlea hears, and the
    heard frame becomes the next context, so the loop runs through the body; the sound and the commands."""
    syrinx, cochlea = Syrinx(), Cochlea()
    window = cue.copy()
    tail = list(tail)
    sounds, commands = [], []
    closed = 0
    for _ in range(REPLAY_FRAMES):
        expected, motor, _ = brain.imagine(window[None])
        cmd = {k: float(v[0]) for k, v in motor.items()}
        commands.append(cmd)
        sound = syrinx.frame(cmd["tension"], cmd["pressure"], cmd["tract"])
        sounds.append(sound)
        window, tail = advance(window, cochlea.frame(sound), tail)
        closed = closed + 1 if pressure_of(cmd["pressure"]) <= 0.0 or expected[0].max() < LOUD * 0.6 else 0
        if closed >= CLOSED:
            break
    return np.concatenate(sounds), commands


def pitch_similarity(original: np.ndarray, produced: np.ndarray, loud: float = 0.3) -> tuple[float, float]:
    """Correlation of the dominant channel over the frames where both sounds are loud, and the share of such frames.

    A voice can be loud in the right places and have the right rough shape without following
    the melody; this asks whether the pitch went up and down when the original's did."""
    if len(produced) < 2:
        return 0.0, 0.0
    stretched = stretched_to(original, produced)
    both = (original.max(axis=1) > loud) & (stretched.max(axis=1) > loud)
    if both.sum() < 5:
        return 0.0, float(both.mean())
    a, b = original[both].argmax(axis=1).astype(float), stretched[both].argmax(axis=1).astype(float)
    a, b = a - a.mean(), b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return (float((a * b).sum() / denom) if denom > 0 else 0.0), float(both.mean())


def evaluate(brain: Brain, library: dict[str, np.ndarray], cochleagrams: dict[str, np.ndarray]) -> dict[str, Any]:
    fam = familiarity(brain, cochleagrams)
    out = {}
    for name, frames in cochleagrams.items():
        cue, tail = onset_cue(frames)  # the first 80 ms, as the parrot would have caught them
        rec, replayed = recall(brain, frames)
        sound, commands = imitate(brain, cue, tail)
        produced = Cochlea().frames(sound) if len(sound) >= FRAME else np.zeros((1, CHANNELS))
        pitch, covered = pitch_similarity(frames, produced)
        out[name] = {"familiarity_error": fam[name], "recall": rec, "recall_frames": int(len(replayed)), "imitation_similarity": similarity(frames, produced), "imitation_shape": shape_similarity(frames, produced), "imitation_pitch": pitch, "imitation_covered": covered, "imitation_frames": int(len(produced)), "commands": commands, "sound": sound}
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
    (HERE / "imitations").mkdir(exist_ok=True)
    for name in library:
        write_wav(HERE / "imitations" / f"{name}_original.wav", library[name])
        write_wav(HERE / "imitations" / f"{name}_imitation{tag}.wav", trained[name]["sound"])
    imitations = {name: {"commands": r["commands"], "similarity": r["imitation_similarity"]} for name, r in trained.items()}
    (HERE / f"imitations{tag}.json").write_text(json.dumps(imitations, separators=(",", ":")))
    net = parrot.brain.export()
    net["cues"] = {name: onset_cue(frames)[0].round(4).tolist() for name, frames in cochleagrams.items()}
    net["sounds"] = {name: {"frames": int(len(f)), "heard": counts[name], "frequent": name in FREQUENT} for name, f in cochleagrams.items()}
    net["seed"] = seed
    (HERE / f"net{tag}.json").write_text(json.dumps(net, separators=(",", ":")))
    (HERE / f"bouts{tag}.json").write_text(json.dumps(parrot.bouts, separators=(",", ":")))
    per_sound = {}
    for name in library:
        often, rarely = (trained[name], trained_b[name]) if name in FREQUENT else (trained_b[name], trained[name])
        heard_often, heard_rarely = (counts[name], counts_b[name]) if name in FREQUENT else (counts_b[name], counts[name])
        per_sound[name] = {
            "heard_often": heard_often, "heard_rarely": heard_rarely, "frequent_on_day_a": name in FREQUENT,
            "recall_heard_often": often["recall"], "recall_heard_rarely": rarely["recall"], "recall_untrained": untrained[name]["recall"],
            "imitation_heard_often": often["imitation_similarity"], "imitation_heard_rarely": rarely["imitation_similarity"], "imitation_untrained": untrained[name]["imitation_similarity"],
            "shape_heard_often": often["imitation_shape"], "shape_heard_rarely": rarely["imitation_shape"], "shape_untrained": untrained[name]["imitation_shape"],
            "pitch_heard_often": often["imitation_pitch"], "pitch_heard_rarely": rarely["imitation_pitch"], "pitch_untrained": untrained[name]["imitation_pitch"], "covered_heard_often": often["imitation_covered"],
            "next_frame_error_heard_often": often["familiarity_error"], "next_frame_error_heard_rarely": rarely["familiarity_error"], "imitation_frames_day_a": trained[name]["imitation_frames"],
        }
    mean = lambda key: float(np.mean([r[key] for r in per_sound.values()]))  # noqa: E731
    summary = {
        "recall_heard_often": mean("recall_heard_often"), "recall_heard_rarely": mean("recall_heard_rarely"), "recall_untrained": mean("recall_untrained"),
        "imitation_heard_often": mean("imitation_heard_often"), "imitation_heard_rarely": mean("imitation_heard_rarely"), "imitation_untrained": mean("imitation_untrained"),
        "shape_heard_often": mean("shape_heard_often"), "shape_heard_rarely": mean("shape_heard_rarely"), "shape_untrained": mean("shape_untrained"),
        "pitch_heard_often": mean("pitch_heard_often"), "pitch_heard_rarely": mean("pitch_heard_rarely"), "pitch_untrained": mean("pitch_untrained"), "covered_heard_often": mean("covered_heard_often"),
        "repetition_effect_on_recall": mean("recall_heard_often") - mean("recall_heard_rarely"),
        "sounds_recalled_better_when_heard_often": int(sum(r["recall_heard_often"] > r["recall_heard_rarely"] for r in per_sound.values())),
        "bouts": len(parrot.bouts), "imitation_bouts": sum(1 for b in parrot.bouts if b["kind"] == "imitate"), "babble_bouts": sum(1 for b in parrot.bouts if b["kind"] == "babble"), "reinforced_frames": parrot.refinements,
        # the day-A household, for the page and the ladder
        "frequent_recall": float(np.mean([trained[n]["recall"] for n in FREQUENT])), "rare_recall": float(np.mean([trained[n]["recall"] for n in RARE])), "untrained_recall": mean("recall_untrained"),
        "frequent_imitation_similarity": float(np.mean([trained[n]["imitation_similarity"] for n in FREQUENT])), "rare_imitation_similarity": float(np.mean([trained[n]["imitation_similarity"] for n in RARE])), "untrained_imitation_similarity": mean("imitation_untrained"),
    }
    print(f"recall when heard often {summary['recall_heard_often']:.3f}, when heard rarely {summary['recall_heard_rarely']:.3f}, untrained {summary['recall_untrained']:.3f}; imitation {summary['imitation_heard_often']:.3f} / {summary['imitation_heard_rarely']:.3f} / {summary['imitation_untrained']:.3f}; spectral shape {summary['shape_heard_often']:.3f} / {summary['shape_heard_rarely']:.3f} / {summary['shape_untrained']:.3f}; pitch track {summary['pitch_heard_often']:.3f} / {summary['pitch_heard_rarely']:.3f} / {summary['pitch_untrained']:.3f}; {summary['sounds_recalled_better_when_heard_often']} of {len(per_sound)} sounds recalled better when heard often", flush=True)
    for name, r in per_sound.items():
        print(f"  {name:9s} heard {r['heard_often']:3d} / {r['heard_rarely']:2d}: recall {r['recall_heard_often']:.3f} / {r['recall_heard_rarely']:.3f} (untrained {r['recall_untrained']:.3f}), imitation {r['imitation_heard_often']:.3f} / {r['imitation_heard_rarely']:.3f} (untrained {r['imitation_untrained']:.3f}), shape {r['shape_heard_often']:.3f} / {r['shape_heard_rarely']:.3f} (untrained {r['shape_untrained']:.3f}), pitch {r['pitch_heard_often']:.3f} / {r['pitch_heard_rarely']:.3f} (untrained {r['pitch_untrained']:.3f}), {r['imitation_frames_day_a']} frames", flush=True)
    body = {
        "day": {"minutes": minutes, "events": events, "heard": counts, "gap_seconds": GAP, "rare_weight": RARE_WEIGHT, "frequent": list(FREQUENT), "rare": list(RARE)},
        "day_b": {"events": events_b, "heard": counts_b, "frequent": list(swapped_frequent), "rare": list(swapped_rare), "seconds": seconds_b, "bouts": len(parrot_b.bouts), "timeline": timeline_b},
        "chunk_frames": CHUNK, "window_frames": WINDOW,
        "brain": {"owners": int(parrot.brain.wiring.n), "hidden": int(parrot.brain.wiring.n - parrot.brain.inputs - parrot.brain.outputs), "parameters": int(parrot.brain.parameters()), "eta": ETA, "decay": DECAY, "learner": parrot.brain.learner.to_dict()},
        "arousal": {"rate_per_second": AROUSAL_RATE, "noise": AROUSAL_NOISE, "babble_share": BABBLE_SHARE}, "reinforcement": {"on": REFINE, "explore": EXPLORE, "gain": REFINE_GAIN}, "per_sound": per_sound, "summary": summary, "timeline": timeline, "seconds": seconds,
        "boundary": {"learning_rule": "free/nudged contrastive Hebbian, centered, owner-local, quadratic nudges on two output groups of one net, seams decaying every update", "memory_learns_from": "household sound and the 300 ms after it; gated off while the parrot sings (as auditory responses in the song system are)", "mirror_learns_from": "the parrot's own sound against the command issued one frame earlier, including the closed frames that end every bout", "reinforcement": "during imitation bouts the commands carry smooth noise; the command taken is pulled toward in the context that chose it, weighted by how much better than usual the heard frame matched the memory's expectation, or pushed from when worse", "names_never_reach_the_brain": True, "cues": "the first 80 ms after an onset following a quiet spell, up to 24 kept; replay draws among them by familiarity", "two_days": "day B swaps which sounds are frequent, with a fresh parrot and another seed, so each sound is scored once heard often and once heard rarely", "bout_ends": "when the mirror keeps the air sac closed, or the memory expects quiet, for six frames", "imitation_shape": "the same correlation with each frame's mean level removed, so a loud broadband voice does not score on loudness alone", "imitation_pitch": "correlation of the dominant cochlear channel over the frames where both are loud: does the melody go where the original's goes", "evaluation": "recall: the memory replayed from each sound's first 80 ms on its own expectations, against the sound itself (correlation of cochleagrams); imitation: from the same cue, expectation to mirror to syrinx to cochlea to the next context, against the sound; the untrained brain is the control"},
    }
    r = cd.Receipt.build("cadence-examples/10_parrot/v1", body, sources=SOURCES)
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
