"""07 music: the free/nudged rule learns to continue Bach chorales, chord by chord.

Run:  python train.py                      (about an hour; downloads the chorale corpus once)
      python build_page.py                 (embeds net.json into index.html; open it and listen)
      python train.py --verify receipt.json

The corpus is the JSB chorales at quarter-note resolution: 382 four-part chorales as
sequences of chords, each chord the MIDI pitches sounding. The net sees the last eight
chords as eight blocks of pitch owners (one owner per pitch, clamped where a voice sounds)
and predicts the next chord on 54 output owners, one per pitch: a multi-hot target and the
quadratic nudge, so several output owners can be right at once. Training rows are
transposed by up to three semitones either way, which a chorale tolerates. The receipt
records how well the next chord is predicted on held-out chorales (F1 of the pitch set,
bits per chord), next to "repeat the last chord", a per-pitch unigram, and an MLP of the
same shape, and it carries a continuation the net wrote from a held-out opening.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import struct
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

import cadence as cd

HERE = Path(__file__).resolve().parent
SOURCES = [("07_music/train.py", Path(__file__).resolve())]
CORPUS_URL = "https://raw.githubusercontent.com/czhuang/JSB-Chorales-dataset/master/jsb-chorales-quarter.json"
CORPUS_SHA256 = "2db9329f1881a1d3f49703ec556bf1d6f84b4f6c1d702c156536e93cf31e1c91"
LOW, HIGH = 43, 96
PITCHES = HIGH - LOW + 1  # 54
WINDOW = 8
TRANSPOSITIONS = range(-3, 4)
INPUTS = WINDOW * PITCHES
GRID = [{"hidden": 128}, {"hidden": 256}]
GRID_EPOCHS = 4
SCHEDULE = {"epochs": 10, "batch": 256, "decay": 0.8, "eta": 3.0}
CONFIG = cd.LearnerConfig(beta=0.1, eta_bias=0.03, nudge="quadratic", tolerance=3e-3, nudged_steps=12, free_steps=60)
READOUT_TOLERANCE = 1e-4
THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)  # the pitch-set threshold is chosen on validation
CALIBRATION_SLOPES = (2.0, 4.0, 6.0, 8.0, 12.0)  # p = sigmoid(slope * (s - threshold)), for bits per chord
CONTINUE_CHORDS = 32


def load() -> tuple[dict[str, list[list[list[int]]]], dict[str, Any]]:
    cache = HERE / "data"
    cache.mkdir(exist_ok=True)
    path = cache / "jsb-chorales-quarter.json"
    if not path.exists():
        request = urllib.request.Request(CORPUS_URL, headers={"User-Agent": "cadence-examples"})
        with urllib.request.urlopen(request, timeout=120) as response:
            path.write_bytes(response.read())
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != CORPUS_SHA256:
        raise ValueError(f"corpus digest {digest} is not the pinned {CORPUS_SHA256}")
    data = json.loads(raw)
    meta = {"name": "JSB chorales, quarter notes (czhuang/JSB-Chorales-dataset)", "url": CORPUS_URL, "sha256": digest, "chorales": {k: len(v) for k, v in data.items()}, "chords": {k: sum(len(c) for c in v) for k, v in data.items()}, "pitch_range": [LOW, HIGH], "window": WINDOW}
    return data, meta


def multi_hot(chord: list[int], shift: int = 0) -> np.ndarray:
    out = np.zeros(PITCHES)
    for p in chord:
        q = p + shift
        if LOW <= q <= HIGH:
            out[q - LOW] = 1.0
    return out


def rows_of(chorales: list[list[list[int]]], shifts: Any) -> tuple[np.ndarray, np.ndarray]:
    """Every position with a full window before it, for every transposition: (windows, next chord)."""
    xs, ys = [], []
    for chorale in chorales:
        for shift in shifts:
            frames = np.stack([multi_hot(c, shift) for c in chorale])
            for t in range(WINDOW, len(frames)):
                xs.append(frames[t - WINDOW : t].ravel())
                ys.append(frames[t])
    return np.asarray(xs, dtype=np.uint8), np.asarray(ys, dtype=np.uint8)


class ChoraleNet:
    def __init__(self, hidden: int, seed: int, backend: str) -> None:
        self.wiring = cd.layered(INPUTS, hidden, PITCHES, density=1.0, seed=seed)
        self.learner = cd.Learner(cd.Settlement(self.wiring, cd.learning_rule(dt=1.0), backend=backend), self.wiring.sets["output"], CONFIG)  # type: ignore[arg-type]
        self.seed = seed
        self.steps: list[tuple[float, float]] = []

    def drive(self, x: np.ndarray) -> np.ndarray:
        return self.learner.engine.clamp_levels(np.pad(np.asarray(x, dtype=float), ((0, 0), (0, self.wiring.n - INPUTS))))

    def target(self, y: np.ndarray) -> np.ndarray:
        out = np.zeros((len(y), self.wiring.n))
        out[:, self.learner.output_index] = y
        return out

    def fit(self, x: np.ndarray, y: np.ndarray, *, epochs: int, batch: int, decay: float, eta: float, log: str) -> float:
        rng = np.random.default_rng(self.seed)
        learner = self.learner
        t0 = time.perf_counter()
        for epoch in range(epochs):
            learner.config = dataclasses.replace(learner.config, eta=eta * decay**epoch, eta_bias=CONFIG.eta_bias * decay**epoch)
            order = rng.permutation(len(y))
            for s in range(0, len(order), batch):
                idx = order[s : s + batch]
                drive, target = self.drive(x[idx]), self.target(y[idx])
                free = learner.free(drive)
                plus = learner.nudged(drive, free, target)
                minus = learner.nudged(drive, free, target, sign=-1.0)
                learner.update(free, plus, minus)
                self.steps.append((free.steps, plus.steps))
            print(f"{log} epoch {epoch + 1}/{epochs} ({time.perf_counter() - t0:.0f}s)", flush=True)
        return time.perf_counter() - t0

    def outputs(self, x: np.ndarray, batch: int = 1000) -> np.ndarray:
        learner = self.learner
        learning = learner.config
        learner.config = dataclasses.replace(learning, tolerance=READOUT_TOLERANCE)
        try:
            return np.concatenate([learner.free(self.drive(x[s : s + batch])).activation[:, learner.output_index] for s in range(0, len(x), batch)])
        finally:
            learner.config = learning

    def parameters(self) -> int:
        return self.learner.parameters()


class TorchMLP:
    def __init__(self, hidden: int, seed: int) -> None:
        import torch

        torch.manual_seed(seed)
        self.torch = torch
        self.net = torch.nn.Sequential(torch.nn.Linear(INPUTS, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, PITCHES))
        self.seed = seed

    def fit(self, x: np.ndarray, y: np.ndarray, *, epochs: int, batch: int, **_: Any) -> float:
        torch = self.torch
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        rng = np.random.default_rng(self.seed)
        t0 = time.perf_counter()
        for _ in range(epochs):
            order = rng.permutation(len(y))
            for s in range(0, len(order), batch):
                idx = order[s : s + batch]
                logits = self.net(torch.tensor(x[idx], dtype=torch.float32))
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor(y[idx], dtype=torch.float32))
                opt.zero_grad()
                loss.backward()
                opt.step()
        return time.perf_counter() - t0

    def outputs(self, x: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            return self.torch.sigmoid(self.net(self.torch.tensor(np.asarray(x, dtype=np.float32)))).numpy()

    def parameters(self) -> int:
        return sum(p.numel() for p in self.net.parameters())


def score(s: np.ndarray, y: np.ndarray, threshold: float, slope: float) -> dict[str, float]:
    """Pitch-set F1 at the threshold, and bits per chord with p = sigmoid(slope (s - threshold))."""
    guess = s >= threshold
    truth = y > 0
    tp = float((guess & truth).sum())
    precision = tp / max(float(guess.sum()), 1.0)
    recall = tp / max(float(truth.sum()), 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    p = np.clip(1.0 / (1.0 + np.exp(-slope * (s - threshold))), 1e-4, 1 - 1e-4)
    bits = float(-(truth * np.log2(p) + (~truth) * np.log2(1 - p)).sum(axis=1).mean())
    exact = float((guess == truth).all(axis=1).mean())
    return {"f1": f1, "precision": precision, "recall": recall, "bits_per_chord": bits, "exact_chord": exact, "threshold": threshold, "slope": slope}


def calibrate(s: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Threshold with the best validation F1, then the slope with the fewest validation bits at it."""
    threshold = max(THRESHOLDS, key=lambda t: score(s, y, t, 4.0)["f1"])
    slope = min(CALIBRATION_SLOPES, key=lambda k: score(s, y, threshold, k)["bits_per_chord"])
    return threshold, slope


def baselines(x: np.ndarray, y: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, hidden: int, seed: int) -> list[dict[str, Any]]:
    """Every baseline gets the same calibration on validation as the patch net."""
    out = []

    def add(model: str, parameters: int, epochs: int, seconds: float, val: np.ndarray, test: np.ndarray) -> None:
        threshold, slope = calibrate(val, y_val)
        out.append({"model": model, "parameters": parameters, "epochs": epochs, "seconds": seconds, "test": score(test, y_test, threshold, slope)})

    add("repeat the last chord", 0, 0, 0.0, x_val[:, -PITCHES:].astype(float), x_test[:, -PITCHES:].astype(float))
    unigram = y.mean(axis=0)
    add("per-pitch unigram", PITCHES, 1, 0.0, np.broadcast_to(unigram, y_val.shape), np.broadcast_to(unigram, y_test.shape))
    mlp = TorchMLP(hidden, seed)
    seconds = mlp.fit(x, y, **SCHEDULE)
    add(f"MLP {INPUTS}-{hidden}-{PITCHES}, sigmoid outputs, Adam", mlp.parameters(), SCHEDULE["epochs"], seconds, mlp.outputs(x_val), mlp.outputs(x_test))
    for row in out:
        print(f"baseline {row['model']}: F1 {row['test']['f1']:.3f}, {row['test']['bits_per_chord']:.2f} bits/chord, exact {row['test']['exact_chord']:.3f} ({row['seconds']:.0f}s)", flush=True)
    return out


def continue_chorale(net: ChoraleNet, opening: list[list[int]], chords: int, threshold: float) -> list[list[int]]:
    """From an opening of WINDOW chords, write ``chords`` more: at each step the pitches whose output owner clears the threshold, at most four."""
    frames = [multi_hot(c) for c in opening[-WINDOW:]]
    written: list[list[int]] = []
    for _ in range(chords):
        p = net.outputs(np.concatenate(frames[-WINDOW:])[None, :])[0]
        order = np.argsort(-p)
        chosen = [int(k) for k in order[:4] if p[k] >= threshold]
        if not chosen:  # nothing clears the threshold: take the best one, so the music goes on
            chosen = [int(order[0])]
        chord = sorted(LOW + k for k in chosen)
        written.append(chord)
        frames.append(multi_hot(chord))
    return written


def midi_bytes(chords: list[list[int]], quarter_ms: int = 500) -> bytes:
    """A one-track MIDI file, one quarter note per chord, piano."""
    ticks = 480

    def var(n: int) -> bytes:
        out = [n & 0x7F]
        n >>= 7
        while n:
            out.append(0x80 | (n & 0x7F))
            n >>= 7
        return bytes(reversed(out))

    track = b"\x00\xff\x51\x03" + struct.pack(">I", quarter_ms * 1000)[1:] + b"\x00\xc0\x00"
    for chord in chords:
        for i, p in enumerate(chord):
            track += var(0) + bytes([0x90, p, 80])
        for i, p in enumerate(chord):
            track += var(ticks if i == 0 else 0) + bytes([0x80, p, 0])
        if not chord:
            track += var(ticks) + b"\xb0\x7b\x00"
    track += b"\x00\xff\x2f\x00"
    return b"MThd" + struct.pack(">IHHH", 6, 0, 1, ticks) + b"MTrk" + struct.pack(">I", len(track)) + track


def export(net: ChoraleNet, path: Path, meta: dict[str, Any]) -> None:
    engine = net.learner.engine
    rule = engine.rule
    payload = {
        "n": net.wiring.n, "sets": {k: [int(i) for i in v] for k, v in net.wiring.sets.items()},
        "W": [round(float(w), 5) for w in engine.dense().ravel()], "bias": [round(float(v), 5) for v in engine.bias],
        "rule": {"slope": rule.slope, "threshold": rule.threshold, "leak": rule.leak, "dt": rule.dt, "clamp": rule.clamp_amplitude, "rest": rule.rest_emission},
        "music": {"low": LOW, "high": HIGH, "window": WINDOW, "threshold": meta["threshold"]},
        "meta": meta,
    }
    path.write_text(json.dumps(payload, separators=(",", ":")))


def run(seed: int, out: Path, backend: str) -> dict[str, Any]:
    data, meta = load()
    x_train, y_train = rows_of(data["train"], TRANSPOSITIONS)
    x_val, y_val = rows_of(data["valid"], [0])
    x_test, y_test = rows_of(data["test"], [0])
    print(f"{meta['name']}: {len(y_train)} training rows with transposition, {len(y_val)} validation, {len(y_test)} test; backend {backend}", flush=True)

    table = []
    for candidate in GRID:
        net = ChoraleNet(candidate["hidden"], seed, backend)
        seconds = net.fit(x_train, y_train, epochs=GRID_EPOCHS, batch=SCHEDULE["batch"], decay=SCHEDULE["decay"], eta=SCHEDULE["eta"], log=f"validation hidden {candidate['hidden']}")
        val = net.outputs(x_val)
        result = score(val, y_val, *calibrate(val, y_val))
        table.append({**candidate, "parameters": net.parameters(), "seconds": seconds, **{f"validation_{k}": v for k, v in result.items()}})
        print(f"validation: hidden {candidate['hidden']} -> F1 {result['f1']:.3f}, {result['bits_per_chord']:.2f} bits/chord ({net.parameters()} parameters, {seconds:.0f}s)", flush=True)
    best = max(table, key=lambda r: (r["validation_f1"], -r["parameters"]))
    selected = {"hidden": best["hidden"]}

    net = ChoraleNet(selected["hidden"], seed, backend)
    seconds = net.fit(x_train, y_train, log=f"selected hidden {selected['hidden']}", **SCHEDULE)
    steps = np.asarray(net.steps).mean(axis=0)
    threshold, slope = calibrate(net.outputs(x_val), y_val)
    result = score(net.outputs(x_test), y_test, threshold, slope)
    print(f"selected hidden {selected['hidden']}: test F1 {result['f1']:.3f}, precision {result['precision']:.3f}, recall {result['recall']:.3f}, {result['bits_per_chord']:.2f} bits/chord, exact chords {result['exact_chord']:.3f} ({seconds:.0f}s, {net.parameters()} parameters)", flush=True)

    opening = data["test"][0][:WINDOW]
    written = continue_chorale(net, opening, CONTINUE_CHORDS, threshold)
    (HERE / "continuation.mid").write_bytes(midi_bytes(opening + written))
    print("continuation:", written[:8], "...", flush=True)

    engine = net.learner.engine
    cpu_engine = cd.Settlement(net.wiring, engine.rule, edge_scale=engine.edge_scale, log_gain=engine.log_gain, bias=engine.bias)
    conformance = cd.conformance(cpu_engine, net.drive(x_test[:1])[0], steps=60)
    comparison = baselines(x_train, y_train, x_val, y_val, x_test, y_test, selected["hidden"], seed)
    export(net, HERE / "net.json", {"f1": result["f1"], "bits_per_chord": result["bits_per_chord"], "parameters": net.parameters(), "opening": opening, "openings": [c[:WINDOW] for c in data["test"][:12]], "threshold": threshold})

    body = {
        "dataset": {**meta, "transpositions": list(TRANSPOSITIONS), "train_rows": int(len(y_train)), "validation_rows": int(len(y_val)), "test_rows": int(len(y_test))},
        "grid": GRID, "grid_epochs": GRID_EPOCHS, "schedule": SCHEDULE, "config": CONFIG.to_dict(), "readout_tolerance": READOUT_TOLERANCE, "thresholds": list(THRESHOLDS), "calibration_slopes": list(CALIBRATION_SLOPES), "threshold": threshold, "slope": slope,
        "training_backend": {"name": backend, "device": cd.available_backends().get(backend)},
        "validation": table, "selected": selected,
        "wiring": net.wiring.summary(), "rule": engine.rule.to_dict(), "learner": net.learner.to_dict(),
        "training_seconds": seconds, "mean_free_steps": float(steps[0]), "mean_nudged_steps": float(steps[1]),
        "test": result, "continuation": {"opening": opening, "written": written, "chorale": "test[0]"},
        "conformance": conformance, "comparison": comparison,
        "boundary": {"learning_rule": "free/nudged contrastive Hebbian, centered, owner-local; quadratic nudge on a multi-hot target", "goal_enters_only_through_the_nudge": True, "model": "windowed: the last eight chords; no recurrence", "selection_on_validation_chorales_only": True, "test_chorales_read_once": True, "baselines_measured_here_on_the_same_split": True, "transposition_augmentation_on_training_rows_only": True},
    }
    receipt = cd.Receipt.build("cadence-examples/07-music/v1", body, sources=SOURCES)
    receipt.write(out)
    print(f"conformance {conformance['max_abs_deviation']:.1e}; receipt {out} ({receipt.digest[:16]}...)")
    return body


def check(body: dict) -> str | None:
    best = max(body["validation"], key=lambda r: (r["validation_f1"], -r["parameters"]))
    if {"hidden": best["hidden"]} != body["selected"]:
        return "selected configuration is not the best on validation"
    t = body["test"]
    f1 = 2 * t["precision"] * t["recall"] / max(t["precision"] + t["recall"], 1e-12)
    if abs(f1 - t["f1"]) > 1e-9:
        return "F1 does not follow from precision and recall"
    if body["threshold"] not in body["thresholds"] or body["slope"] not in body["calibration_slopes"]:
        return "calibration is not from the declared grids"
    if not body["conformance"]["ledger"]["clean"]:
        return "reference ledger is not clean"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", default="torch" if "torch" in cd.available_backends() else "cpu")
    parser.add_argument("--output", type=Path, default=HERE / "receipt.json")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        ok, message = cd.Receipt.verify(args.verify, sources=SOURCES, check=check)
        print(message)
        return 0 if ok else 1
    run(args.seed, args.output, args.backend)
    return 0


if __name__ == "__main__":
    sys.exit(main())
