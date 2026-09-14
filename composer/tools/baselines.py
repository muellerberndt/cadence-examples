"""Same encoded data and held-out rows: bigram and Adam-trained MLP controls."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from composer.encoding import INPUTS, SIZES, features

ROOT = Path(__file__).resolve().parents[1]


def main():
    import torch

    p = argparse.ArgumentParser()
    p.add_argument("--updates", type=int, default=6000)
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(17)
    rng = np.random.default_rng(17)
    folder = ROOT / "data/classical"
    train = dict(np.load(folder / "train.npz"))
    validation = dict(np.load(folder / "validation.npz"))
    ids = np.random.default_rng(77).choice(
        len(validation["labels"]), 2048, replace=False
    )
    x = torch.tensor(
        features(validation["context"][ids], validation["extra"][ids]), device=a.device
    )
    y = torch.tensor(validation["labels"][ids].astype(np.int64), device=a.device)
    bigram = []
    for k, n in enumerate(SIZES):
        count = np.ones((n, n))
        np.add.at(count, (train["context"][:, -1, k], train["labels"][:, k]), 1)
        prob = count / count.sum(1, keepdims=True)
        q = prob[validation["context"][ids, -1, k]]
        bigram.append(
            {
                "nll": float(
                    -np.log(q[np.arange(len(ids)), validation["labels"][ids, k]]).mean()
                ),
                "accuracy": float((q.argmax(1) == validation["labels"][ids, k]).mean()),
            }
        )
    net = torch.nn.Sequential(
        torch.nn.Linear(INPUTS, a.size),
        torch.nn.ReLU(),
        torch.nn.Linear(a.size, a.size),
        torch.nn.ReLU(),
        torch.nn.Linear(a.size, sum(SIZES)),
    ).to(a.device)
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    history = []
    start = time.monotonic()
    best = float("inf")
    for update in range(a.updates):
        ix = rng.integers(len(train["labels"]), size=256)
        tx = torch.tensor(
            features(train["context"][ix], train["extra"][ix]), device=a.device
        )
        ty = torch.tensor(train["labels"][ix].astype(np.int64), device=a.device)
        output = net(tx).split(SIZES, dim=1)
        loss = (
            sum(
                torch.nn.functional.cross_entropy(v, ty[:, k])
                for k, v in enumerate(output)
            )
            / 4
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (update + 1) % max(1, a.updates // 10) == 0:
            with torch.no_grad():
                result = net(x).split(SIZES, dim=1)
                nll = [
                    float(torch.nn.functional.cross_entropy(v, y[:, k]))
                    for k, v in enumerate(result)
                ]
                accuracy = [
                    float((v.argmax(1) == y[:, k]).float().mean())
                    for k, v in enumerate(result)
                ]
            history.append(
                {
                    "update": update + 1,
                    "seconds": time.monotonic() - start,
                    "nll": nll,
                    "accuracy": accuracy,
                    "mean_nll": float(np.mean(nll)),
                }
            )
            best = min(best, float(np.mean(nll)))
            print(json.dumps(history[-1]), flush=True)
    report = {
        "bigram": bigram,
        "mlp": {
            "parameters": sum(v.numel() for v in net.parameters()),
            "updates": a.updates,
            "batch": 256,
            "size": a.size,
            "seconds": time.monotonic() - start,
            "device": a.device,
            "history": history,
            "best_validation_nll": best,
        },
        "dataset_manifest_sha256": hashlib.sha256(
            (folder / "manifest.json").read_bytes()
        ).hexdigest(),
        "seed": 17,
        "validation_rows": 2048,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    path = ROOT / "runs/baselines"
    path.mkdir(exist_ok=True, parents=True)
    (path / "receipt.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
