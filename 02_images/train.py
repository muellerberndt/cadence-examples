"""02 images: the same rule on MNIST, 784 pixels a picture, trained on the accelerated backend.

Run:  python train.py                       (about 15 minutes on an M-series Mac; downloads MNIST once)
      python train.py --verify receipt.json

Nothing changes from the digits example except scale: 784 input owners, a hidden layer,
10 output owners, one free settlement and two nudged settlements per batch, and every
overlap moving on its own two endpoints. Training runs on the torch backend (float32 on
Apple silicon); the held-out test set is read once, on the float64 CPU backend, and the
receipt records the conformance of the two.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

import cadence as cd

HERE = Path(__file__).resolve().parent
SOURCES = [("02_images/train.py", Path(__file__).resolve())]
INPUTS, CLASSES = 784, 10
GRID = [{"hidden": 128}, {"hidden": 256}]
SCHEDULE = {"epochs": 10, "batch": 256, "decay": 0.8, "eta": 3.0}
CONFIG = cd.LearnerConfig(
    beta=0.1, eta_bias=0.03, temperature=0.1, tolerance=3e-3, nudged_steps=12, free_steps=60
)
READOUT_TOLERANCE = 1e-4
VALIDATION = 10_000


def load() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """MNIST from OpenML, cached under data/; the receipt records the digest of the arrays used."""
    from sklearn.datasets import fetch_openml

    cache = HERE / "data"
    cache.mkdir(exist_ok=True)
    npz = cache / "mnist.npz"
    if npz.exists():
        with np.load(npz) as f:
            x, y = f["x"], f["y"]
    else:
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, data_home=str(cache))
        x = np.asarray(mnist.data, dtype=np.uint8)
        y = np.asarray(mnist.target, dtype=np.int64)
        np.savez_compressed(npz, x=x, y=y)
    digest = hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()
    x = x.astype(np.float64) / 255.0
    return x[:60_000], y[:60_000], x[60_000:], y[60_000:], {"name": "MNIST (OpenML mnist_784 v1)", "digest": digest, "train": 60_000, "test": 10_000}


class Net:
    def __init__(self, hidden: int, eta: float, seed: int, backend: str) -> None:
        self.wiring = cd.layered(INPUTS, hidden, CLASSES, density=1.0, seed=seed)
        self.backend = backend
        self.learner = cd.Learner(
            cd.Settlement(self.wiring, cd.learning_rule(dt=1.0), backend=backend),  # type: ignore[arg-type]
            self.wiring.sets["output"],
            dataclasses.replace(CONFIG, eta=eta),
        )
        self.seed = seed
        self.steps: list[tuple[float, float]] = []

    def drive(self, x: np.ndarray) -> np.ndarray:
        return self.learner.engine.clamp_levels(np.pad(x, ((0, 0), (0, self.wiring.n - INPUTS))))

    def fit(self, x: np.ndarray, y: np.ndarray, *, epochs: int, batch: int, decay: float, eta: float, log: str = "") -> float:
        rng = np.random.default_rng(self.seed)
        learner = self.learner
        eta_bias = learner.config.eta_bias
        t0 = time.perf_counter()
        for epoch in range(epochs):
            learner.config = dataclasses.replace(learner.config, eta=eta * decay**epoch, eta_bias=eta_bias * decay**epoch)
            order = rng.permutation(len(y))
            for start in range(0, len(order), batch):
                idx = order[start : start + batch]
                _, report = learner.step(self.drive(x[idx]), y[idx])
                self.steps.append((report["free_steps"], report["nudged_steps"]))
            if log:
                print(f"{log} epoch {epoch + 1}/{epochs} ({time.perf_counter() - t0:.0f}s)", flush=True)
        return time.perf_counter() - t0

    def accuracy(self, x: np.ndarray, y: np.ndarray, backend: str | None = None) -> float:
        """Free settlements at the readout tolerance, optionally on another backend."""
        learner = self.learner
        engine = learner.engine
        if backend is not None and backend != engine.backend:
            engine = cd.Settlement(self.wiring, engine.rule, backend=backend, edge_scale=engine.edge_scale, log_gain=engine.log_gain, bias=engine.bias)  # type: ignore[arg-type]
        probe = cd.Learner(engine, self.wiring.sets["output"], dataclasses.replace(learner.config, tolerance=READOUT_TOLERANCE))
        return probe.accuracy(self.drive(x), y, batch=1000)


def baselines(x: np.ndarray, y: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, hidden: int, seed: int) -> list[dict]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        logistic = LogisticRegression(max_iter=100).fit(x, y)
        out.append({"model": "logistic regression (lbfgs, 100 iterations)", "parameters": INPUTS * CLASSES + CLASSES, "epochs": int(np.max(logistic.n_iter_)), "seconds": time.perf_counter() - t0, "test_accuracy": float(logistic.score(x_test, y_test))})
        for epochs in (SCHEDULE["epochs"], 30):
            t0 = time.perf_counter()
            mlp = MLPClassifier((hidden,), batch_size=SCHEDULE["batch"], max_iter=epochs, random_state=seed).fit(x, y)
            out.append({"model": f"MLP {INPUTS}-{hidden}-{CLASSES} (adam, batch {SCHEDULE['batch']})", "parameters": INPUTS * hidden + hidden + hidden * CLASSES + CLASSES, "epochs": int(mlp.n_iter_), "seconds": time.perf_counter() - t0, "test_accuracy": float(mlp.score(x_test, y_test))})
            print(f"baseline {out[-1]['model']} {out[-1]['epochs']} epochs: {out[-1]['test_accuracy']:.4f} ({out[-1]['seconds']:.0f}s)", flush=True)
    return out


def run(seed: int, out: Path, backend: str) -> dict[str, Any]:
    x_train, y_train, x_test, y_test, meta = load()
    print(f"{meta['name']}: {meta['train']} training, {meta['test']} test images (digest {meta['digest'][:16]}...); backend {backend}: {cd.available_backends().get(backend)}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y_train))
    fit_idx, val_idx = order[VALIDATION:], order[:VALIDATION]

    table = []
    for candidate in GRID:
        net = Net(candidate["hidden"], SCHEDULE["eta"], seed, backend)
        seconds = net.fit(x_train[fit_idx], y_train[fit_idx], **SCHEDULE, log=f"validation hidden {candidate['hidden']}")
        table.append({**candidate, "parameters": net.learner.parameters(), "validation_accuracy": net.accuracy(x_train[val_idx], y_train[val_idx]), "seconds": seconds})
        print(f"validation: hidden {candidate['hidden']:4d} -> {table[-1]['validation_accuracy']:.4f} ({table[-1]['parameters']} parameters, {seconds:.0f}s)", flush=True)
    best = max(table, key=lambda row: (row["validation_accuracy"], -row["parameters"]))
    selected = {"hidden": best["hidden"]}

    net = Net(selected["hidden"], SCHEDULE["eta"], seed, backend)
    untrained = net.accuracy(x_test, y_test, backend="cpu")
    seconds = net.fit(x_train, y_train, **SCHEDULE, log=f"selected hidden {selected['hidden']}")
    steps = np.asarray(net.steps).mean(axis=0)
    train_accuracy = net.accuracy(x_train[:10_000], y_train[:10_000])
    test_accuracy = net.accuracy(x_test, y_test, backend="cpu")
    test_accuracy_accelerated = net.accuracy(x_test, y_test)
    print(f"selected hidden {selected['hidden']}: test {test_accuracy:.4f} on cpu float64 ({test_accuracy_accelerated:.4f} on {backend}); train (first 10k) {train_accuracy:.4f}; {seconds:.0f}s; {net.learner.parameters()} parameters", flush=True)

    engine = net.learner.engine
    cpu_engine = cd.Settlement(net.wiring, engine.rule, edge_scale=engine.edge_scale, log_gain=engine.log_gain, bias=engine.bias)
    conformance = cd.conformance(cpu_engine, net.drive(x_test[:1])[0], steps=60)
    probe = cd.Learner(cpu_engine, net.wiring.sets["output"], dataclasses.replace(CONFIG, tolerance=READOUT_TOLERANCE))
    predictions = probe.predict(net.drive(x_test[:2000]))
    accelerated_predictions = cd.Learner(engine, net.wiring.sets["output"], dataclasses.replace(CONFIG, tolerance=READOUT_TOLERANCE)).predict(net.drive(x_test[:2000]))
    backend_agreement = float((predictions == accelerated_predictions).mean())
    confusion = np.zeros((CLASSES, CLASSES), dtype=int)
    guesses = np.concatenate([probe.predict(net.drive(x_test[s : s + 1000])) for s in range(0, len(y_test), 1000)])
    for truth, guess in zip(y_test, guesses, strict=True):
        confusion[truth, guess] += 1

    comparison = baselines(x_train, y_train, x_test, y_test, selected["hidden"], seed)
    body = {
        "dataset": {**meta, "validation": VALIDATION, "split_seed": seed},
        "grid": GRID,
        "schedule": SCHEDULE,
        "config": CONFIG.to_dict(),
        "readout_tolerance": READOUT_TOLERANCE,
        "training_backend": {"name": backend, "device": cd.available_backends().get(backend)},
        "validation": table,
        "selected": selected,
        "wiring": net.wiring.summary(),
        "rule": engine.rule.to_dict(),
        "learner": net.learner.to_dict(),
        "untrained_test_accuracy": untrained,
        "training_seconds": seconds,
        "mean_free_steps": float(steps[0]),
        "mean_nudged_steps": float(steps[1]),
        "train_accuracy_first_10k": train_accuracy,
        "test_accuracy": test_accuracy,
        "test_accuracy_on_training_backend": test_accuracy_accelerated,
        "backend_prediction_agreement_first_2k": backend_agreement,
        "confusion": confusion.tolist(),
        "conformance": conformance,
        "comparison": comparison,
        "boundary": {
            "learning_rule": "free/nudged contrastive Hebbian, centered, owner-local",
            "goal_enters_only_through_the_nudge": True,
            "trained_on_float32_accelerator_read_out_on_float64_cpu": True,
            "selection_on_validation_split_of_training_set_only": True,
            "test_set_read_once_after_selection": True,
            "baselines_measured_here_on_the_same_split": True,
        },
    }
    receipt = cd.Receipt.build("cadence-examples/02-images/v1", body, sources=SOURCES)
    receipt.write(out)
    print(f"test accuracy {test_accuracy:.4f}; backend agreement {backend_agreement:.4f}; conformance {conformance['max_abs_deviation']:.1e}; receipt {out} ({receipt.digest[:16]}...)")
    return body


def check(body: dict) -> str | None:
    confusion = np.asarray(body["confusion"])
    if confusion.sum() != body["dataset"]["test"]:
        return "confusion matrix does not cover the test set"
    if abs(np.trace(confusion) / confusion.sum() - body["test_accuracy"]) > 1e-9:
        return "test accuracy does not follow from the confusion matrix"
    best = max(body["validation"], key=lambda r: (r["validation_accuracy"], -r["parameters"]))
    if {"hidden": best["hidden"]} != body["selected"]:
        return "selected configuration is not the best on validation"
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
