"""Conventional control with identical polyphonic inputs, targets and held-out rows."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from composer.ensemble import INPUTS, SIZES, Dataset, encode


def measure(net, dataset, device, count=2048):
    ids = np.random.default_rng(77).choice(
        dataset.count, min(count, dataset.count), replace=False
    )
    loss, hits = np.zeros(5), np.zeros(5)
    family, family_hits = np.zeros(8), np.zeros(8)
    with torch.inference_mode():
        for start in range(0, len(ids), 128):
            data = dataset.rows(ids[start : start + 128])
            x = torch.as_tensor(encode(data["context"], data["extra"]), device=device)
            y = torch.as_tensor(data["labels"].astype(np.int64), device=device)
            for k, logits in enumerate(net(x).split(SIZES, dim=1)):
                loss[k] += float(
                    torch.nn.functional.cross_entropy(logits, y[:, k], reduction="sum")
                )
                correct = (logits.argmax(1) == y[:, k]).cpu().numpy()
                hits[k] += correct.sum()
                if k == 3:
                    labels = data["labels"][:, k]
                    family += np.bincount(labels, minlength=8)
                    family_hits += np.bincount(labels, weights=correct, minlength=8)
    return {
        "nll": (loss / len(ids)).tolist(),
        "mean_nll": float(loss.mean() / len(ids)),
        "accuracy": (hits / len(ids)).tolist(),
        "instrument_counts": family.tolist(),
        "instrument_accuracy": (family_hits / np.maximum(1, family)).tolist(),
        "examples": len(ids),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ensemble")
    parser.add_argument("--name", default="ensemble-mlp-1024")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--updates", type=int, default=30000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    a = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(31)
    rng = np.random.default_rng(31)
    folder = ROOT / "data" / a.dataset
    train, validation = Dataset(folder, "train"), Dataset(folder, "validation")
    net = torch.nn.Sequential(
        torch.nn.Linear(INPUTS, a.size),
        torch.nn.ReLU(),
        torch.nn.Linear(a.size, a.size),
        torch.nn.ReLU(),
        torch.nn.Linear(a.size, sum(SIZES)),
    ).to(a.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    out = ROOT / "runs" / a.name
    out.mkdir(parents=True, exist_ok=True)
    history, best = [], float("inf")
    started = time.monotonic()
    for update in range(1, a.updates + 1):
        data = train.sample(rng, a.batch)
        x = torch.as_tensor(encode(data["context"], data["extra"]), device=a.device)
        y = torch.as_tensor(data["labels"].astype(np.int64), device=a.device)
        loss = sum(
            torch.nn.functional.cross_entropy(logits, y[:, k])
            for k, logits in enumerate(net(x).split(SIZES, dim=1))
        ) / len(SIZES)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if update % max(1, a.updates // 10) == 0 or update == a.updates:
            measured = measure(net, validation, a.device)
            row = {"update": update, "seconds": time.monotonic() - started, **measured}
            history.append(row)
            if measured["mean_nll"] < best:
                best = measured["mean_nll"]
                torch.save(net.state_dict(), out / "model.pt")
            print(json.dumps(row), flush=True)
    training_seconds = time.monotonic() - started
    net.load_state_dict(
        torch.load(out / "model.pt", map_location=a.device, weights_only=True)
    )
    test = measure(net, Dataset(folder, "test"), a.device, 8192)
    receipt = {
        "model": "Two-hidden-layer ReLU MLP; Adam; equal-weight five-head cross entropy",
        "parameters": sum(p.numel() for p in net.parameters()),
        "width": a.size,
        "updates": a.updates,
        "batch": a.batch,
        "sampled_events": a.updates * a.batch,
        "training_seconds": training_seconds,
        "device": a.device,
        "gpu": torch.cuda.get_device_name(a.device)
        if a.device.startswith("cuda")
        else None,
        "history": history,
        "best_validation_nll": best,
        "test": test,
        "dataset": a.dataset,
        "dataset_manifest_sha256": hashlib.sha256(
            (folder / "manifest.json").read_bytes()
        ).hexdigest(),
        "sources": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), ROOT / "composer/ensemble.py"]
        },
        "checkpoint_sha256": hashlib.sha256(
            (out / "model.pt").read_bytes()
        ).hexdigest(),
        "limits": "Single seed, no hyperparameter sweep; model sizes and hardware parallelism differ. Same encoded information and seed-77 held-out rows. Prediction quality is not composition quality.",
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps({k: v for k, v in receipt.items() if k != "history"}), flush=True)


if __name__ == "__main__":
    main()
