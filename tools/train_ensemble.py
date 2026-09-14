"""Single- or multi-GPU local-contrast training of the polyphonic event brain.

torchrun workers average each seam's measured contrast before its adaptive update.
No autograd graph or backpropagation is constructed. The distributed check
compares that average with the same complete batch on one GPU.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from composer.brain import describe
from composer.ensemble import SIZES, Dataset, Design, build, drives

ROOT = Path(__file__).resolve().parents[1]


def evaluate(brain, dataset, count=2048):
    ids = np.random.default_rng(77).choice(
        dataset.count, min(count, dataset.count), replace=False
    )
    losses = np.zeros(5)
    correct = np.zeros(5)
    families = np.zeros(8)
    hits = np.zeros(8)
    for start in range(0, len(ids), 128):
        batch = dataset.rows(ids[start : start + 128])
        state = brain.free(drives(brain, batch["context"], batch["extra"]))
        logits = state.activation[:, brain.output_index] / brain.config.temperature
        offset = 0
        for k, width in enumerate(SIZES):
            z = logits[:, offset : offset + width]
            z = z - z.max(1, keepdims=True)
            prob = np.exp(z)
            prob /= prob.sum(1, keepdims=True)
            labels = batch["labels"][:, k]
            prediction = z.argmax(1)
            losses[k] -= np.log(prob[np.arange(len(labels)), labels].clip(1e-12)).sum()
            correct[k] += (prediction == labels).sum()
            if k == 3:
                families += np.bincount(labels, minlength=8)
                hits += np.bincount(labels, weights=prediction == labels, minlength=8)
            offset += width
    return {
        "nll": (losses / len(ids)).tolist(),
        "mean_nll": float(losses.mean() / len(ids)),
        "accuracy": (correct / len(ids)).tolist(),
        "instrument_counts": families.tolist(),
        "instrument_accuracy": (hits / np.maximum(1, families)).tolist(),
        "examples": len(ids),
    }


def contrast(brain, batch):
    drive = drives(brain, batch["context"], batch["extra"])
    free = brain.free(drive)
    target = brain.targets(batch["labels"])
    plus = brain.nudged(drive, free, target)
    minus = brain.nudged(drive, free, target, sign=-1)
    kernel = brain._device_kernel(plus, minus)
    edges, owners = kernel.contrast_tensors(plus.device["s"], minus.device["s"])
    divisor = len(batch["labels"]) * 2 * brain.config.beta
    return kernel, edges / divisor, owners / divisor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="ensemble-pilot")
    p.add_argument("--name", default="ensemble-pilot")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--updates", type=int, default=1500)
    p.add_argument("--batch", type=int, default=128, help="batch per GPU")
    p.add_argument("--max-seconds", type=int, default=7200)
    p.add_argument("--check-distributed", action="store_true")
    a = p.parse_args()
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.set_num_threads(2)
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl")
    device = f"cuda:{local}"
    training = Dataset(ROOT / "data" / a.dataset, "train")
    validation = Dataset(ROOT / "data" / a.dataset, "validation")
    design = Design(a.size)
    brain = build(design, backend="torch", device=device)
    folder = ROOT / "runs" / a.name
    if rank == 0:
        folder.mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()
    if a.check_distributed:
        complete = training.sample(np.random.default_rng(91), world * 8)
        small = {k: v[rank * 8 : (rank + 1) * 8] for k, v in complete.items()}
        _, edge, owner = contrast(brain, small)
        if world > 1:
            dist.all_reduce(edge)
            dist.all_reduce(owner)
            edge /= world
            owner /= world
        _, expected_edge, expected_owner = contrast(brain, complete)
        maximum = max(
            float((edge - expected_edge).abs().max()),
            float((owner - expected_owner).abs().max()),
        )
        assert maximum < 3e-5, maximum
        if rank == 0:
            (folder / "distributed_check.json").write_text(
                json.dumps(
                    {
                        "world_size": world,
                        "max_raw_contrast_error": maximum,
                        "tolerance": 3e-5,
                    },
                    indent=2,
                )
            )
        if world > 1:
            dist.destroy_process_group()
        return
    rng = np.random.default_rng(31 + rank)
    best = float("inf")
    history = []
    receipt = {
        "design": asdict(design),
        "brain": describe(brain),
        "config": brain.config.to_dict(),
        "dataset": a.dataset,
        "dataset_manifest_sha256": hashlib.sha256(
            (ROOT / "data" / a.dataset / "manifest.json").read_bytes()
        ).hexdigest(),
        "world_size": world,
        "batch_per_gpu": a.batch,
        "global_batch": a.batch * world,
        "devices": [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ],
        "sources": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                Path(__file__),
                ROOT / "composer/ensemble.py",
                ROOT / "tools/prepare_ensemble.py",
            ]
        },
        "objective": "Joint prediction of pitch, sounding duration, inter-onset time, instrument family and velocity. Prediction loss is not a musical-quality score.",
    }
    started = time.monotonic()
    for update in range(a.updates):
        batch = training.sample(rng, a.batch)
        kernel, edges, owners = contrast(brain, batch)
        if world > 1:
            dist.all_reduce(edges)
            dist.all_reduce(owners)
            edges /= world
            owners /= world
        edges, owners = brain._adaptive_device(kernel, edges, owners)
        brain._apply_device(
            kernel, brain.config.eta * edges, brain.config.eta_bias * owners
        )
        if (update + 1) % max(100, a.updates // 10) == 0 or update == a.updates - 1:
            stop = torch.zeros(1, device=device)
            if rank == 0:
                measured = evaluate(brain, validation)
                row = {
                    "update": update + 1,
                    "seconds": time.monotonic() - started,
                    **measured,
                }
                history.append(row)
                if row["mean_nll"] < best:
                    best = row["mean_nll"]
                    brain.save(folder / "brain.npz")
                receipt.update(
                    history=history,
                    best_validation_nll=best,
                    updates=update + 1,
                    seconds=time.monotonic() - started,
                    sampled_events=(update + 1) * a.batch * world,
                )
                (folder / "receipt.json").write_text(json.dumps(receipt, indent=2))
                print(json.dumps(row), flush=True)
                if time.monotonic() - started >= a.max_seconds:
                    stop.fill_(1)
            if world > 1:
                dist.broadcast(stop, 0)
            if stop.item():
                break
    if rank == 0:
        receipt["complete"] = True
        (folder / "receipt.json").write_text(json.dumps(receipt, indent=2))
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
