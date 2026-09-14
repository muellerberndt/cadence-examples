"""Produce the public showcase evidence. No candidate is selected on test results.

python tools/benchmark.py --seeds 3 --steps 1200 --output evidence/evidence.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import cadence as cd
import numpy as np
from model import (
    ROOT,
    MATRIX,
    N,
    OnlineMLP,
    fast_memory,
    key_bank,
    reference,
    residual,
    samples,
    worm_engine,
)


def hashes():
    files = [
        ROOT / n
        for n in (
            "tools/model.py",
            "tools/benchmark.py",
            "worm/worm.json",
            "tools/fetch_worm.py",
            "shared/engine.js",
        )
    ]
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    } | {
        "cadence/" + p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(cd.__file__).parent.glob("*.py"))
    }


def memory_trial(seed, correlation, writes=128):
    rng = np.random.default_rng(seed)
    keys = key_bank(correlation)
    fast = fast_memory()
    additive = fast_memory(rule="hebb")
    nets = {n: OnlineMLP(seed) for n in (1, 10, 100)}
    truth = np.zeros(8, dtype=int)
    seen = set()
    totals = {
        k: 0 for k in ("cadence", "additive", "dictionary", "mlp1", "mlp10", "mlp100")
    }
    elapsed = dict.fromkeys(totals, 0.0)
    count = 0
    for step in range(writes):
        k = step if step < 8 else int(rng.integers(8))
        v = int(rng.integers(4))
        target = np.eye(4)[v]
        truth[k] = v
        seen.add(k)
        for name, memory in [("cadence", fast), ("additive", additive)]:
            start = time.perf_counter()
            memory.observe(keys[k : k + 1], target[None])
            elapsed[name] += time.perf_counter() - start
        for n, net in nets.items():
            start = time.perf_counter()
            net.observe(keys[k], target, n)
            elapsed[f"mlp{n}"] += time.perf_counter() - start
        # One stream: reading another key must NOT resize/reset Cadence's batch.
        for query in sorted(seen):
            totals["cadence"] += int(
                fast.recall(keys[query : query + 1]).argmax() == truth[query]
            )
            totals["additive"] += int(
                additive.recall(keys[query : query + 1]).argmax() == truth[query]
            )
            totals["dictionary"] += 1
            for n, net in nets.items():
                totals[f"mlp{n}"] += int(
                    net.predict(keys[query]).argmax() == truth[query]
                )
            count += 1
    return {
        "seed": seed,
        "correlation": correlation,
        "writes": writes,
        "queries": count,
        "arms": {
            k: {
                "correct": v,
                "accuracy": v / count,
                "update_seconds": elapsed[k],
                "updates": writes * (int(k[3:]) if k.startswith("mlp") else 1),
            }
            for k, v in totals.items()
        },
    }


def worm_trial(seed, steps, trials):
    import torch

    torch.set_num_threads(1)
    drive, mask = samples(seed + 100, 2048, mixed=False)
    vd, vm = samples(seed + 200, 256, lesions=3, mixed=True)
    prep = time.perf_counter()
    y = torch.tensor(reference(drive, mask), dtype=torch.float32)
    vy = torch.tensor(reference(vd, vm), dtype=torch.float32)
    label_seconds = time.perf_counter() - prep

    def features(d, m):
        message = MATRIX.dot((m * np.tanh(d)).T).T
        return torch.tensor(
            np.concatenate((d, m, message), axis=1), dtype=torch.float32
        )

    x, vx = features(drive, mask), features(vd, vm)

    def predict(net, z):
        return z[:, N : 2 * N] * torch.tanh(z[:, :N] + z[:, 2 * N :] + net(z))

    start = time.perf_counter()
    best, candidates = None, []
    for width in (64, 128):
        torch.manual_seed(seed)
        net = torch.nn.Sequential(
            torch.nn.Linear(3 * N, width), torch.nn.Tanh(), torch.nn.Linear(width, N)
        )
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        rng = np.random.default_rng(seed + 300)
        for _ in range(steps):
            ids = torch.tensor(rng.integers(len(x), size=128))
            loss = ((predict(net, x[ids]) - y[ids]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.inference_mode():
            score = float(((predict(net, vx) - vy) ** 2).mean())
        candidates.append({"width": width, "validation_mse": score})
        if best is None or score < best[0]:
            best = score, net, width
    training_seconds = time.perf_counter() - start
    _, net, width = best
    engine = worm_engine()
    rows = []
    for condition, lesions, mixed in [
        ("intact", 0, False),
        ("new_mixtures", 2, True),
        ("many_lesions", 64, True),
    ]:
        d, m = samples(seed + 1000 + lesions, trials, lesions, mixed)
        truth = reference(d, m)
        start = time.perf_counter()
        state = engine.settle_batch(d, mask=m, steps=200, tolerance=1e-10).activation
        sec = time.perf_counter() - start
        start = time.perf_counter()
        unrolled = reference(d, m, 32)
        unroll_sec = time.perf_counter() - start
        tx = features(d, m)
        with torch.inference_mode():
            start = time.perf_counter()
            pred = predict(net, tx).numpy()
            mlp_sec = time.perf_counter() - start

        def reading(a, seconds, truth=truth, d=d, m=m):
            return {
                "mse": float(np.mean((a - truth) ** 2)),
                "relative_rmse": float(
                    np.linalg.norm(a - truth) / max(np.linalg.norm(truth), 1e-12)
                ),
                "max_residual": float(residual(a, d, m).max()),
                "batch_seconds": seconds,
            }

        rows.append(
            {
                "condition": condition,
                "trials": trials,
                "arms": {
                    "cadence": reading(state, sec),
                    "mlp": reading(pred, mlp_sec),
                    "unrolled32": reading(unrolled, unroll_sec),
                },
            }
        )
    # Export the first scheduled seed, not the best test-performing seed.
    exported = {
        "w1": net[0].weight.detach().numpy().T.tolist(),
        "b1": net[0].bias.detach().numpy().tolist(),
        "w2": net[2].weight.detach().numpy().T.tolist(),
        "b2": net[2].bias.detach().numpy().tolist(),
    }
    return {
        "seed": seed,
        "training_examples": len(x),
        "steps_per_candidate": steps,
        "candidates": candidates,
        "chosen_width": width,
        "parameters": sum(p.numel() for p in net.parameters()),
        "training_seconds": training_seconds,
        "label_generation_seconds": label_seconds,
        "rows": rows,
    }, exported


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--trials", type=int, default=128)
    p.add_argument("--output", type=Path, default=ROOT / "evidence/evidence.json")
    a = p.parse_args()
    assert a.seeds > 0 and a.steps > 0 and a.trials > 0
    worms, memories, exported = [], [], None
    for seed in range(7, 7 + a.seeds):
        row, net = worm_trial(seed, a.steps, a.trials)
        worms.append(row)
        if exported is None:
            exported = net
        for c in (0.0, 0.5, 0.9):
            memories.append(memory_trial(seed, c))
        print(
            seed,
            "worm MSE",
            [r["arms"]["mlp"]["mse"] for r in row["rows"]],
            "memory",
            memories[-3]["arms"]["mlp1"]["accuracy"],
            flush=True,
        )
    body = {
        "schema": "cadence.showcase/v1",
        "sources": hashes(),
        "platform": platform.platform(),
        "schedule": {
            "seeds": list(range(7, 7 + a.seeds)),
            "memory_writes": 128,
            "memory_correlations": [0, 0.5, 0.9],
            "mlp_updates": [1, 10, 100],
            "worm_steps": a.steps,
            "worm_trials": a.trials,
            "validation_only_selection": True,
        },
        "worm": worms,
        "memory": memories,
        "browser": {"worm_mlp": exported, "online_mlp": OnlineMLP().export()},
        "boundaries": [
            "Supplied chemical graph, normalized positive weights and tanh dynamics; not worm physiology.",
            "MLP receives drives, masks and an exact first propagation; fixed graph is known to both methods.",
            "Tied unrolling is a feedforward graph that also solves the supplied circuit.",
            "Memory uses explicit eight-dimensional keys and four-dimensional values; dictionary is exact.",
            "MLP gets identical revealed samples; extra update budgets are reported; no general efficiency claim.",
            "Fly is a simplified engineered 2D body with a supplied steering controller, not FlyWire or validated flight.",
        ],
    }
    body["digest"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(body, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
