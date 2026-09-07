"""01 digits: a patch net learns to read 8x8 digits with the owner-local free/nudged rule.

Run:  python train.py                       (about half a minute on a laptop)
      python train.py --verify receipt.json

The script selects its wiring and learning rate on a validation split carved out of the
training set, retrains the selected configuration on the whole training set, and then
reads the held-out test set once. Everything it tried, and the baselines it measured on
the same split, go into the receipt.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

import cadence as cd

HERE = Path(__file__).resolve().parent
SOURCES = [("01_digits/train.py", Path(__file__).resolve())]
INPUTS, CLASSES = 64, 10

# The declared search: every configuration below is tried on the validation split.
GRID = [
    {"hidden": 16, "eta": 3.0},
    {"hidden": 32, "eta": 3.0},
    {"hidden": 64, "eta": 3.0},
]
VALIDATION_SEEDS = 2  # each candidate is trained twice; the mean validation accuracy selects
SCHEDULE = {"epochs": 20, "batch": 32, "decay": 0.8}
CONFIG = cd.LearnerConfig(
    beta=0.1, eta_bias=0.03, temperature=0.1, tolerance=3e-3, nudged_steps=12, free_steps=100
)
READOUT_TOLERANCE = 1e-4  # settle tighter when reading out than while learning


def load(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The scikit-learn digits: 1797 images of 8x8 pixels in 0..16; a stratified 80/20 split."""
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    digits = load_digits()
    x = digits.data / 16.0  # pixel levels in [0, 1]
    y = digits.target
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed, stratify=y
    )
    return x_train, y_train, x_test, y_test


def split_validation(
    x: np.ndarray, y: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    x_fit, x_val, y_fit, y_val = train_test_split(
        x, y, test_size=0.2, random_state=seed, stratify=y
    )
    return x_fit, y_fit, x_val, y_val


class Digits:
    """One patch net on the digits: a wiring, a learner, and the drive for a pixel array."""

    def __init__(self, hidden: int, eta: float, seed: int) -> None:
        # 64 input owners (one per pixel), hidden owners, 10 output owners (one per class).
        # Forward and feedback overlaps share one scale per seam; no lateral overlaps.
        self.wiring = cd.layered(INPUTS, hidden, CLASSES, density=1.0, seed=seed)
        rule = cd.learning_rule(dt=1.0)  # unit slope, threshold 0, leak 0.1, unit clamp
        self.learner = cd.Learner(
            cd.Settlement(self.wiring, rule),
            self.wiring.sets["output"],
            dataclasses.replace(CONFIG, eta=eta),
        )
        self.seed = seed
        self.steps: list[tuple[float, float]] = []

    def drive(self, x: np.ndarray) -> np.ndarray:
        """A pixel's level is the clamp on its owner; nothing else is clamped."""
        return self.learner.engine.clamp_levels(np.pad(x, ((0, 0), (0, self.wiring.n - INPUTS))))

    def fit(self, x: np.ndarray, y: np.ndarray, *, epochs: int, batch: int, decay: float) -> float:
        """Free phase, nudged phases, local update, for every batch; returns seconds spent."""
        drive = self.drive(x)
        rng = np.random.default_rng(self.seed)
        learner = self.learner
        eta, eta_bias = learner.config.eta, learner.config.eta_bias
        t0 = time.perf_counter()
        for epoch in range(epochs):
            learner.config = dataclasses.replace(
                learner.config, eta=eta * decay**epoch, eta_bias=eta_bias * decay**epoch
            )
            order = rng.permutation(len(y))
            for start in range(0, len(order), batch):
                idx = order[start : start + batch]
                _, report = learner.step(drive[idx], y[idx])
                self.steps.append((report["free_steps"], report["nudged_steps"]))
        return time.perf_counter() - t0

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        learner = self.learner
        learning = learner.config
        learner.config = dataclasses.replace(learning, tolerance=READOUT_TOLERANCE)
        try:
            return learner.accuracy(self.drive(x), y)
        finally:
            learner.config = learning

    def predict(self, x: np.ndarray) -> np.ndarray:
        learner = self.learner
        learning = learner.config
        learner.config = dataclasses.replace(learning, tolerance=READOUT_TOLERANCE)
        try:
            return learner.predict(self.drive(x))
        finally:
            learner.config = learning


def baselines(x: np.ndarray, y: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> list[dict]:
    """Logistic regression and a one-hidden-layer MLP on the same split, for the comparison table."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        logistic = LogisticRegression(max_iter=200).fit(x, y)
        out.append(
            {
                "model": "logistic regression (lbfgs)",
                "parameters": INPUTS * CLASSES + CLASSES,
                "epochs": int(np.max(logistic.n_iter_)),
                "seconds": time.perf_counter() - t0,
                "test_accuracy": float(logistic.score(x_test, y_test)),
            }
        )
        for hidden in (16, 32):
            for epochs in (12, 50):
                t0 = time.perf_counter()
                mlp = MLPClassifier((hidden,), batch_size=32, max_iter=epochs, random_state=0).fit(x, y)
                out.append(
                    {
                        "model": f"MLP {INPUTS}-{hidden}-{CLASSES} (adam, batch 32)",
                        "parameters": INPUTS * hidden + hidden + hidden * CLASSES + CLASSES,
                        "epochs": int(mlp.n_iter_),
                        "seconds": time.perf_counter() - t0,
                        "test_accuracy": float(mlp.score(x_test, y_test)),
                    }
                )
    return out


def run(seed: int, out: Path, seeds: int) -> dict[str, Any]:
    x_train, y_train, x_test, y_test = load(seed)

    # 1. Select on a validation split of the training set; the test set is not touched.
    x_fit, y_fit, x_val, y_val = split_validation(x_train, y_train, seed)
    table = []
    for candidate in GRID:
        accuracies, seconds = [], []
        for k in range(VALIDATION_SEEDS):
            net = Digits(candidate["hidden"], candidate["eta"], seed + k)
            seconds.append(net.fit(x_fit, y_fit, **SCHEDULE))
            accuracies.append(net.accuracy(x_val, y_val))
        table.append(
            {
                **candidate,
                "parameters": net.learner.parameters(),
                "validation_accuracies": accuracies,
                "validation_accuracy": float(np.mean(accuracies)),
                "seconds": float(np.mean(seconds)),
            }
        )
        print(
            f"validation: hidden {candidate['hidden']:3d} eta {candidate['eta']:.1f} -> "
            f"{table[-1]['validation_accuracy']:.3f} ({table[-1]['parameters']} parameters, "
            f"{table[-1]['seconds']:.1f}s per run)"
        )
    best = max(table, key=lambda row: (row["validation_accuracy"], -row["parameters"]))
    selected = {"hidden": best["hidden"], "eta": best["eta"]}

    # 2. Retrain the selected configuration on the whole training set; read the test set once.
    runs = []
    for k in range(seeds):
        net = Digits(selected["hidden"], selected["eta"], seed + k)
        untrained = net.accuracy(x_test, y_test)
        seconds = net.fit(x_train, y_train, **SCHEDULE)
        steps = np.asarray(net.steps).mean(axis=0)
        runs.append(
            {
                "seed": seed + k,
                "untrained_test_accuracy": untrained,
                "train_accuracy": net.accuracy(x_train, y_train),
                "test_accuracy": net.accuracy(x_test, y_test),
                "seconds": seconds,
                "mean_free_steps": float(steps[0]),
                "mean_nudged_steps": float(steps[1]),
            }
        )
        print(
            f"seed {seed + k}: test {runs[-1]['test_accuracy']:.3f} "
            f"(train {runs[-1]['train_accuracy']:.3f}, untrained {untrained:.3f}, {seconds:.1f}s)"
        )
        if k == 0:
            first = net

    # 3. Controls on the first run: a net trained on shuffled labels, and the reference engine.
    shuffled = Digits(selected["hidden"], selected["eta"], seed)
    shuffled.fit(x_train, np.random.default_rng(seed).permutation(y_train), **SCHEDULE)
    label_control = shuffled.accuracy(x_test, y_test)
    conformance = cd.conformance(first.learner.engine, first.drive(x_train[:1])[0], steps=100)

    # 4. Confusion on the test set from the first run's free settlements only.
    predictions = first.predict(x_test)
    confusion = np.zeros((CLASSES, CLASSES), dtype=int)
    for truth, guess in zip(y_test, predictions, strict=True):
        confusion[truth, guess] += 1

    comparison = baselines(x_train, y_train, x_test, y_test)
    body = {
        "dataset": {
            "name": "sklearn digits",
            "train": int(len(y_train)),
            "test": int(len(y_test)),
            "split_seed": seed,
        },
        "grid": GRID,
        "schedule": SCHEDULE,
        "config": CONFIG.to_dict(),
        "readout_tolerance": READOUT_TOLERANCE,
        "validation": table,
        "selected": selected,
        "wiring": first.wiring.summary(),
        "rule": first.learner.engine.rule.to_dict(),
        "learner": first.learner.to_dict(),
        "runs": runs,
        "test_accuracy": runs[0]["test_accuracy"],
        "test_accuracy_mean": float(np.mean([r["test_accuracy"] for r in runs])),
        "test_accuracy_std": float(np.std([r["test_accuracy"] for r in runs])),
        "shuffled_label_control_test_accuracy": label_control,
        "confusion": confusion.tolist(),
        "conformance": conformance,
        "comparison": comparison,
        "boundary": {
            "learning_rule": "free/nudged contrastive Hebbian, centered, owner-local",
            "goal_enters_only_through_the_nudge": True,
            "selection_on_validation_split_of_training_set_only": True,
            "test_set_read_once_per_seed_after_selection": True,
            "baselines_measured_here_on_the_same_split": True,
            "no_claim_beyond_this_dataset_and_split": True,
        },
    }
    receipt = cd.Receipt.build("cadence-examples/01-digits/v2", body, sources=SOURCES)
    receipt.write(out)
    print(
        f"test accuracy {body['test_accuracy_mean']:.3f} ± {body['test_accuracy_std']:.3f} "
        f"over {seeds} seed(s); shuffled-label control {label_control:.3f}; "
        f"conformance {conformance['max_abs_deviation']:.1e}"
    )
    for row in comparison:
        print(
            f"baseline {row['model']:36s} {row['parameters']:5d} parameters, "
            f"{row['epochs']:3d} epochs, {row['seconds']:.2f}s, test {row['test_accuracy']:.3f}"
        )
    print(f"receipt {out} ({receipt.digest[:16]}...)")
    return body


def check(body: dict) -> str | None:
    """The arithmetic a verifier recomputes from the stored readings."""
    confusion = np.asarray(body["confusion"])
    if confusion.sum() != body["dataset"]["test"]:
        return "confusion matrix does not cover the test set"
    if abs(np.trace(confusion) / confusion.sum() - body["test_accuracy"]) > 1e-9:
        return "test accuracy does not follow from the confusion matrix"
    accuracies = [r["test_accuracy"] for r in body["runs"]]
    if abs(float(np.mean(accuracies)) - body["test_accuracy_mean"]) > 1e-9:
        return "mean test accuracy does not follow from the runs"
    best = max(body["validation"], key=lambda r: (r["validation_accuracy"], -r["parameters"]))
    if {"hidden": best["hidden"], "eta": best["eta"]} != body["selected"]:
        return "selected configuration is not the best on validation"
    if not body["conformance"]["ledger"]["clean"]:
        return "reference ledger is not clean"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=3, help="independent runs of the selected net")
    parser.add_argument("--output", type=Path, default=HERE / "receipt.json")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        ok, message = cd.Receipt.verify(args.verify, sources=SOURCES, check=check)
        print(message)
        return 0 if ok else 1
    run(args.seed, args.output, args.seeds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
