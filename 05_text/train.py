"""05 text: the free/nudged rule predicts the next character of Shakespeare from a window of text.

Run:  python train.py                       (a couple of hours on an M-series Mac; downloads the corpus once)
      python train.py --verify receipt.json

The net sees the last k characters as k blocks of one-hot input owners and predicts the
next character with V output owners: classification, exactly as in the digits example,
with the text itself as the label. It is a windowed model, not a recurrent one: nothing is
carried from one character to the next except through the window. The receipt records
bits per character and next-character accuracy on a held-out tail of the corpus, next to
a bigram model, a same-window MLP, and a one-layer transformer trained on the same text,
and a sample the patch net wrote by settling one character at a time.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

import cadence as cd

HERE = Path(__file__).resolve().parent
SOURCES = [("05_text/train.py", Path(__file__).resolve())]
CORPUS_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
WINDOW = 16
GRID = [{"hidden": 256}, {"hidden": 512}]
SCHEDULE = {"epochs": 5, "batch": 512, "decay": 0.8, "eta": 3.0}
CONFIG = cd.LearnerConfig(beta=0.1, eta_bias=0.03, temperature=0.1, tolerance=3e-3, nudged_steps=12, free_steps=60)
READOUT_TOLERANCE = 1e-4
TRAIN_CHARS, VALIDATION_CHARS, TEST_CHARS = 300_000, 50_000, 100_000
GRID_CHARS, GRID_EPOCHS = 100_000, 3
EVAL_TEMPERATURES = (0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5)


def load() -> tuple[np.ndarray, list[str], dict[str, Any]]:
    cache = HERE / "data"
    cache.mkdir(exist_ok=True)
    path = cache / "tinyshakespeare.txt"
    if not path.exists():
        request = urllib.request.Request(CORPUS_URL, headers={"User-Agent": "cadence-examples"})
        with urllib.request.urlopen(request, timeout=60) as response:
            path.write_bytes(response.read())
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    vocabulary = sorted(set(text))
    index = {c: i for i, c in enumerate(vocabulary)}
    codes = np.asarray([index[c] for c in text], dtype=np.int64)
    meta = {"name": "tiny Shakespeare (Karpathy char-rnn)", "url": CORPUS_URL, "sha256": hashlib.sha256(raw).hexdigest(), "characters": len(text), "vocabulary": len(vocabulary), "window": WINDOW}
    return codes, vocabulary, meta


class TextNet:
    def __init__(self, vocabulary: int, hidden: int, seed: int, backend: str) -> None:
        self.v = vocabulary
        self.inputs = WINDOW * vocabulary
        self.wiring = cd.layered(self.inputs, hidden, vocabulary, density=1.0, seed=seed)
        self.learner = cd.Learner(cd.Settlement(self.wiring, cd.learning_rule(dt=1.0), backend=backend), self.wiring.sets["output"], CONFIG)  # type: ignore[arg-type]
        self.seed = seed
        self.steps: list[tuple[float, float]] = []

    def drive(self, windows: np.ndarray) -> np.ndarray:
        """(batch, WINDOW) integer codes -> a clamp with one-hot blocks on the input owners."""
        batch = len(windows)
        out = np.zeros((batch, self.wiring.n))
        rows = np.repeat(np.arange(batch), WINDOW)
        cols = (np.arange(WINDOW) * self.v)[None, :] + windows
        out[rows, cols.ravel()] = self.learner.engine.rule.clamp_amplitude
        return out

    def fit(self, codes: np.ndarray, *, epochs: int, batch: int, decay: float, eta: float, log: str) -> float:
        starts = np.arange(WINDOW, len(codes))
        rng = np.random.default_rng(self.seed)
        learner = self.learner
        t0 = time.perf_counter()
        for epoch in range(epochs):
            learner.config = dataclasses.replace(learner.config, eta=eta * decay**epoch, eta_bias=CONFIG.eta_bias * decay**epoch)
            order = rng.permutation(starts)
            for s in range(0, len(order), batch):
                idx = order[s : s + batch]
                windows = codes[idx[:, None] + np.arange(-WINDOW, 0)[None, :]]
                _, report = learner.step(self.drive(windows), codes[idx])
                self.steps.append((report["free_steps"], report["nudged_steps"]))
            print(f"{log} epoch {epoch + 1}/{epochs} ({time.perf_counter() - t0:.0f}s)", flush=True)
        return time.perf_counter() - t0

    def outputs(self, codes: np.ndarray, batch: int = 1000) -> tuple[np.ndarray, np.ndarray]:
        """Output activations for every position with a full window; returns (activations, targets)."""
        learner = self.learner
        learning = learner.config
        learner.config = dataclasses.replace(learning, tolerance=READOUT_TOLERANCE)
        try:
            starts = np.arange(WINDOW, len(codes))
            out = []
            for s in range(0, len(starts), batch):
                idx = starts[s : s + batch]
                windows = codes[idx[:, None] + np.arange(-WINDOW, 0)[None, :]]
                out.append(learner.free(self.drive(windows)).activation[:, learner.output_index])
            return np.concatenate(out), codes[starts]
        finally:
            learner.config = learning

    def generate(self, prompt: np.ndarray, length: int, temperature: float, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        codes = list(prompt)
        learner = self.learner
        learning = learner.config
        learner.config = dataclasses.replace(learning, tolerance=READOUT_TOLERANCE)
        try:
            for _ in range(length):
                window = np.asarray(codes[-WINDOW:])[None, :]
                s = learner.free(self.drive(window)).activation[0, learner.output_index]
                p = softmax(s / temperature)
                codes.append(int(rng.choice(self.v, p=p)))
        finally:
            learner.config = learning
        return np.asarray(codes[len(prompt) :])


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def score(activations: np.ndarray, targets: np.ndarray, temperature: float) -> dict[str, float]:
    p = softmax(activations / temperature)
    nll = -np.log2(np.clip(p[np.arange(len(targets)), targets], 1e-12, None))
    return {"bits_per_char": float(nll.mean()), "accuracy": float((p.argmax(axis=1) == targets).mean())}


def calibrate(activations: np.ndarray, targets: np.ndarray) -> float:
    """The softmax temperature that minimises bits per character on the validation tail."""
    return min(EVAL_TEMPERATURES, key=lambda t: score(activations, targets, t)["bits_per_char"])


def bigram(train: np.ndarray, test: np.ndarray, v: int) -> dict[str, Any]:
    counts = np.ones((v, v))
    np.add.at(counts, (train[:-1], train[1:]), 1.0)
    p = counts / counts.sum(axis=1, keepdims=True)
    nll = -np.log2(p[test[:-1], test[1:]])
    return {"model": "bigram (Laplace)", "parameters": v * v, "epochs": 1, "seconds": 0.0, "bits_per_char": float(nll.mean()), "accuracy": float((p[test[:-1]].argmax(axis=1) == test[1:]).mean())}


def torch_baselines(train: np.ndarray, validation: np.ndarray, test: np.ndarray, v: int, hidden: int, seed: int, epochs: int) -> list[dict[str, Any]]:
    import torch
    import torch.nn as nn

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(seed)

    def windows_of(codes: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        starts = np.arange(WINDOW, len(codes))
        x = codes[starts[:, None] + np.arange(-WINDOW, 0)[None, :]]
        return torch.tensor(x), torch.tensor(codes[starts])

    class WindowMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Flatten(), nn.Linear(WINDOW * v, hidden), nn.ReLU(), nn.Linear(hidden, v))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(nn.functional.one_hot(x, v).float())

    class TinyTransformer(nn.Module):
        def __init__(self, d: int = 64, heads: int = 4) -> None:
            super().__init__()
            self.embed = nn.Embedding(v, d)
            self.position = nn.Parameter(torch.zeros(WINDOW, d))
            self.block = nn.TransformerEncoderLayer(d, heads, dim_feedforward=4 * d, dropout=0.0, batch_first=True)
            self.norm = nn.LayerNorm(d)
            self.out = nn.Linear(d, v)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.embed(x) + self.position
            mask = torch.triu(torch.full((WINDOW, WINDOW), float("-inf"), device=h.device), diagonal=1)
            h = self.block(h, src_mask=mask)
            return self.out(self.norm(h[:, -1]))

    out = []
    x_tr, y_tr = windows_of(train)
    x_te, y_te = windows_of(test)
    for name, model in (("MLP, same window, Adam", WindowMLP()), ("one-layer transformer, Adam", TinyTransformer())):
        model = model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        t0 = time.perf_counter()
        for _ in range(epochs):
            perm = torch.randperm(len(y_tr))
            for s in range(0, len(perm), SCHEDULE["batch"]):
                idx = perm[s : s + SCHEDULE["batch"]]
                loss = nn.functional.cross_entropy(model(x_tr[idx].to(device)), y_tr[idx].to(device))
                opt.zero_grad()
                loss.backward()
                opt.step()
        seconds = time.perf_counter() - t0
        with torch.no_grad():
            logits = torch.cat([model(x_te[s : s + 4096].to(device)).cpu() for s in range(0, len(y_te), 4096)])
            logp = torch.log_softmax(logits, dim=1)
            bits = float(-logp[torch.arange(len(y_te)), y_te].mean() / np.log(2))
            accuracy = float((logits.argmax(dim=1) == y_te).float().mean())
        out.append({"model": name, "parameters": sum(p.numel() for p in model.parameters()), "epochs": epochs, "seconds": seconds, "bits_per_char": bits, "accuracy": accuracy})
        print(f"baseline {name}: {bits:.3f} bits/char, accuracy {accuracy:.3f} ({out[-1]['parameters']} parameters, {seconds:.0f}s)", flush=True)
    return out


def run(seed: int, out: Path, backend: str) -> dict[str, Any]:
    codes, vocabulary, meta = load()
    v = len(vocabulary)
    train = codes[:TRAIN_CHARS]
    validation = codes[TRAIN_CHARS : TRAIN_CHARS + VALIDATION_CHARS]
    test = codes[-TEST_CHARS:]
    print(f"{meta['name']}: {meta['characters']} characters, vocabulary {v}, window {WINDOW}; train {len(train)}, validation {len(validation)}, test {len(test)}; backend {backend}", flush=True)

    table = []
    for candidate in GRID:
        net = TextNet(v, candidate["hidden"], seed, backend)
        seconds = net.fit(train[:GRID_CHARS], epochs=GRID_EPOCHS, batch=SCHEDULE["batch"], decay=SCHEDULE["decay"], eta=SCHEDULE["eta"], log=f"validation hidden {candidate['hidden']}")
        activations, targets = net.outputs(validation)
        temperature = calibrate(activations, targets)
        table.append({**candidate, "parameters": net.learner.parameters(), "seconds": seconds, "temperature": temperature, **{f"validation_{k}": val for k, val in score(activations, targets, temperature).items()}})
        print(f"validation: hidden {candidate['hidden']} -> {table[-1]['validation_bits_per_char']:.3f} bits/char, accuracy {table[-1]['validation_accuracy']:.3f} at T {temperature} ({table[-1]['parameters']} parameters, {seconds:.0f}s)", flush=True)
    best = min(table, key=lambda row: (row["validation_bits_per_char"], row["parameters"]))
    selected = {"hidden": best["hidden"]}

    net = TextNet(v, selected["hidden"], seed, backend)
    seconds = net.fit(train, log=f"selected hidden {selected['hidden']}", **SCHEDULE)
    steps = np.asarray(net.steps).mean(axis=0)
    val_activations, val_targets = net.outputs(validation)
    temperature = calibrate(val_activations, val_targets)
    activations, targets = net.outputs(test)
    result = score(activations, targets, temperature)
    print(f"selected hidden {selected['hidden']}: test {result['bits_per_char']:.3f} bits/char, accuracy {result['accuracy']:.3f} at T {temperature} ({seconds:.0f}s, {net.learner.parameters()} parameters)", flush=True)

    prompt = test[:WINDOW]
    sample = net.generate(prompt, 400, temperature, seed)
    text = "".join(vocabulary[c] for c in sample)
    print("sample:\n" + text, flush=True)

    engine = net.learner.engine
    cpu_engine = cd.Settlement(net.wiring, engine.rule, edge_scale=engine.edge_scale, log_gain=engine.log_gain, bias=engine.bias)
    conformance = cd.conformance(cpu_engine, net.drive(test[None, :WINDOW])[0], steps=60)

    comparison = [bigram(train, test, v)] + torch_baselines(train, validation, test, v, selected["hidden"], seed, SCHEDULE["epochs"])
    body = {
        "dataset": {**meta, "train": len(train), "validation": len(validation), "test": len(test), "split": "first 300k / next 50k / last 100k characters"},
        "grid": GRID, "grid_chars": GRID_CHARS, "grid_epochs": GRID_EPOCHS, "schedule": SCHEDULE, "config": CONFIG.to_dict(),
        "readout_tolerance": READOUT_TOLERANCE, "eval_temperatures": list(EVAL_TEMPERATURES),
        "training_backend": {"name": backend, "device": cd.available_backends().get(backend)},
        "validation": table, "selected": selected,
        "wiring": net.wiring.summary(), "rule": engine.rule.to_dict(), "learner": net.learner.to_dict(),
        "training_seconds": seconds, "mean_free_steps": float(steps[0]), "mean_nudged_steps": float(steps[1]),
        "temperature": temperature, "test": result, "sample": {"prompt": "".join(vocabulary[c] for c in prompt), "text": text, "temperature": temperature},
        "conformance": conformance, "comparison": comparison,
        "boundary": {"learning_rule": "free/nudged contrastive Hebbian, centered, owner-local", "goal_enters_only_through_the_nudge": True, "model": "windowed: the last 16 characters as one-hot clamps; no recurrence", "temperature_selected_on_validation_only": True, "test_tail_read_once": True, "baselines_measured_here_on_the_same_text": True},
    }
    receipt = cd.Receipt.build("cadence-examples/05-text/v1", body, sources=SOURCES)
    receipt.write(out)
    print(f"conformance {conformance['max_abs_deviation']:.1e}; receipt {out} ({receipt.digest[:16]}...)")
    return body


def check(body: dict) -> str | None:
    best = min(body["validation"], key=lambda r: (r["validation_bits_per_char"], r["parameters"]))
    if {"hidden": best["hidden"]} != body["selected"]:
        return "selected configuration is not the best on validation"
    if body["temperature"] not in body["eval_temperatures"]:
        return "temperature is not from the declared grid"
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
