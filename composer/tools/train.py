"""Train/evaluate a named brain on fixed MIDI piece splits; save every stage receipt."""

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from composer.brain import Design, build, describe, drives, probabilities
from composer.encoding import SIZES

ROOT = Path(__file__).resolve().parents[1]


def evaluate(learner, data, count=2048):
    rng = np.random.default_rng(77)
    ids = rng.choice(
        len(data["labels"]), min(count, len(data["labels"])), replace=False
    )
    correct = np.zeros(4)
    loss = np.zeros(4)
    counts = 0
    for start in range(0, len(ids), 128):
        ix = ids[start : start + 128]
        ps, _ = probabilities(learner, data["context"][ix], data["extra"][ix])
        y = data["labels"][ix]
        for k, p in enumerate(ps):
            correct[k] += (p.argmax(1) == y[:, k]).sum()
            loss[k] -= np.log(p[np.arange(len(ix)), y[:, k]].clip(1e-12)).sum()
        counts += len(ix)
    return {
        "examples": counts,
        "accuracy": (correct / counts).tolist(),
        "nll": (loss / counts).tolist(),
        "mean_nll": float(loss.mean() / counts),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="pilot")
    p.add_argument("--name", default="small")
    p.add_argument("--backend", default="cpu", choices=["cpu", "torch"])
    p.add_argument("--device", default=None)
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--updates", type=int, default=200)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=17)
    a = p.parse_args()
    if a.backend == "torch":
        import torch

        torch.set_num_threads(2)
    folder = ROOT / "data" / a.dataset
    train = dict(np.load(folder / "train.npz"))
    validation = dict(np.load(folder / "validation.npz"))
    run = ROOT / "runs" / a.name
    run.mkdir(parents=True, exist_ok=True)
    design = Design(a.size, a.size // 2, a.size // 2, a.seed)
    brain = build(design, a.backend, a.device)
    before = evaluate(brain, validation)
    history = []
    rng = np.random.default_rng(a.seed)
    start = time.monotonic()
    baseline = []
    for k, n in enumerate(SIZES):
        counts = np.bincount(train["labels"][:, k], minlength=n) + 1
        p = counts / counts.sum()
        baseline.append(float(-np.log(p[validation["labels"][:, k]]).mean()))
    best = float("inf")
    for step in range(a.updates):
        ix = rng.integers(len(train["labels"]), size=a.batch)
        _, report = brain.step(
            drives(brain, train["context"][ix], train["extra"][ix]), train["labels"][ix]
        )
        if (step + 1) % max(10, a.updates // 10) == 0 or step + 1 == a.updates:
            metrics = evaluate(brain, validation)
            row = {"update": step + 1, "seconds": time.monotonic() - start, **metrics}
            history.append(row)
            if metrics["mean_nll"] < best:
                best = metrics["mean_nll"]
                brain.save(run / "brain.npz")
            (run / "progress.json").write_text(
                json.dumps(
                    {
                        "design": asdict(design),
                        "brain": describe(brain),
                        "before": before,
                        "unigram_nll": baseline,
                        "history": history,
                    },
                    indent=2,
                )
                + "\n"
            )
            print(json.dumps(row), flush=True)
    report = {
        "design": asdict(design),
        "brain": describe(brain),
        "config": brain.config.to_dict(),
        "dataset_manifest_sha256": hashlib.sha256(
            (folder / "manifest.json").read_bytes()
        ).hexdigest(),
        "dataset": a.dataset,
        "updates": a.updates,
        "batch": a.batch,
        "backend": a.backend,
        "device": a.device,
        "seconds": time.monotonic() - start,
        "before": before,
        "unigram_nll": baseline,
        "history": history,
        "best_validation_nll": best,
        "sources": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                ROOT / "composer/brain.py",
                ROOT / "composer/encoding.py",
                ROOT / "tools/prepare.py",
                Path(__file__),
            ]
        },
    }
    (run / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "complete": a.name,
                "best_nll": best,
                "baseline_nll": float(np.mean(baseline)),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
