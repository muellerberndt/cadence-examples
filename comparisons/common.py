"""Parts both comparison runners use: the arms, the job pool, timings and the receipt.

An arm is one world model under one replay protocol: the perceptron or the transformer,
each with the ring off and with the ring on. Every other part of the stage stays as the
stage runs it.
"""

from __future__ import annotations

import hashlib
import json
import traceback
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from comparisons.networks import OnlineMLP, OnlineModel, OnlineTransformer, Reading

ARMS = ("mlp", "mlp_replay", "transformer", "transformer_replay")


def network_of(arm: str, reading: Reading, settings: dict[str, Any], *, capacity: int, seed: int) -> OnlineModel:
    """The world model of one arm: its kind from the arm's name, its ring from the suffix."""
    kind = "transformer" if arm.startswith("transformer") else "mlp"
    replay = 1 if arm.endswith("_replay") else 0
    if kind == "mlp":
        return OnlineMLP(reading, hidden=settings["hidden"], lr=settings["lr"], replay=replay, capacity=capacity, seed=seed)
    return OnlineTransformer(
        reading,
        width=settings["width"],
        heads=settings["heads"],
        layers=settings["layers"],
        feedforward=settings["feedforward"],
        lr=settings["lr"],
        replay=replay,
        capacity=capacity,
        seed=seed,
    )


def stage_sources(stage_dir: Path) -> dict[str, Any]:
    """The hashes of the stage under test, so a reader can see which stage code ran."""
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(stage_dir.glob("*.py"))}
    files["config.json"] = hashlib.sha256((stage_dir / "config.json").read_bytes()).hexdigest()
    return {"stage_dir": stage_dir.name, "stage_files": files}


def timings(values: Sequence[float]) -> dict[str, float]:
    """Milliseconds per decision: the mean and two quantiles."""
    if not len(values):
        return {"mean": None, "p50": None, "p95": None, "decisions": 0}
    array = np.asarray(values, dtype=float) * 1000.0
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "decisions": int(array.size),
    }


def ratio(model: Sequence[float], zero: Sequence[float]) -> float | None:
    """The prediction error relative to predicting zero; ``None`` when nothing was scored."""
    if not len(model) or not np.sum(zero):
        return None
    return float(np.sum(model) / np.sum(zero))


def run_jobs(jobs: Sequence[Any], workers: int, run: Callable[[Any], dict], worker: Callable[[Any], Any], worker_args: Any = None) -> Iterator[tuple[Any, Any]]:
    """The jobs in this process, or in ``workers`` processes; yields (job, result or error).

    A spawned process starts empty, so the arguments travel with the job."""
    if workers <= 1 or len(jobs) <= 1:
        for job in jobs:
            yield job, guarded(run, job)
        return
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        futures = {pool.submit(worker, (job, worker_args)): job for job in jobs}
        for future in as_completed(futures):
            yield futures[future], future.result()


def guarded(run: Callable[[Any], dict], job: Any) -> Any:
    try:
        return run(job)
    except Exception:  # noqa: BLE001  a failed trial is reported, never hidden
        return traceback.format_exc()


def sanitize(value: Any) -> Any:
    """The receipt's rule: a reading that is not a finite number becomes null."""
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return sanitize(value.tolist())
    return value


def write_summaries(out: Path, results: Sequence[dict]) -> list[str]:
    """One event-free summary per seed; every arm of that seed sits in it."""
    names = []
    for seed in sorted({r["seed"] for r in results}):
        rows = [r for r in results if r["seed"] == seed]
        path = out / f"summary_seed{seed}.json"
        body = sanitize({"seed": seed, "arms": {r["arm"]: r for r in rows}})
        path.write_text(json.dumps(body, sort_keys=True, indent=1, allow_nan=False) + "\n")
        names.append(path.name)
    return names


def mean_over(results: Iterable[dict], key: Callable[[dict], float | None]) -> float | None:
    values = [key(r) for r in results]
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def by_arm(results: Sequence[dict], key: Callable[[dict], float | None]) -> dict[str, Any]:
    """One number per arm: the value on each seed and their mean."""
    out: dict[str, Any] = {}
    for arm in ARMS:
        rows = [r for r in results if r["arm"] == arm]
        if rows:
            out[arm] = {"per_seed": [key(r) for r in rows], "mean": mean_over(rows, key)}
    return out
