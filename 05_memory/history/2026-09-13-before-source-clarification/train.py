"""Updating a small memory: residual writes, additive writes, exact lookup, transformer.

CPU-only, synthetic data, no downloads. Run python train.py --steps 800 --seeds 3.
All methods receive the same explicit key/value vectors and evaluation episodes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import cadence as cd

HERE = Path(__file__).resolve().parent
KEYS, DIM, VALUES = 8, 9, 8
LENGTHS = (8, 32, 128)
CORRELATIONS = (0.0, 0.5)
TRAIN_LENGTHS = (8, 16, 32)
ARMS = ("delta", "hebb", "dictionary", "exact_attention", "transformer")
CASES = [("random_values", c, length) for c in CORRELATIONS for length in LENGTHS]
CASES += [("balanced_first_pass", c, KEYS) for c in (0.5, 0.9)]


def sources() -> list[tuple[str, Path]]:
    package = Path(cd.__file__).resolve().parent
    return [("05_memory/train.py", Path(__file__).resolve())] + [
        (f"cadence/{name}", package / name) for name in ("stream.py", "receipts.py")
    ]


def keys(correlation: float) -> np.ndarray:
    """Unit keys: independent coordinates plus one shared coordinate."""
    return np.column_stack([np.sqrt(1 - correlation) * np.eye(KEYS),
                            np.full(KEYS, np.sqrt(correlation))])


def episodes(seed: int, count: int, length: int, *, balanced: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    identity = np.stack([rng.permutation(KEYS) for _ in range(count)])
    identity = np.concatenate([identity, rng.integers(KEYS, size=(count, length - KEYS))], axis=1)
    values = (np.stack([rng.permutation(VALUES) for _ in range(count)]) if balanced
              else rng.integers(VALUES, size=(count, length)))
    truth = np.zeros((count, KEYS), dtype=int)
    for t in range(length):
        truth[np.arange(count), identity[:, t]] = values[:, t]
    return identity, values, truth


def memory_read(identity: np.ndarray, values: np.ndarray, bank: np.ndarray, rule: str) -> tuple[np.ndarray, int]:
    memory = cd.FastSeams(np.arange(DIM), np.arange(DIM, DIM + VALUES), rule=rule)
    for t in range(identity.shape[1]):
        memory.observe(bank[identity[:, t]], np.eye(VALUES)[values[:, t]])
    output = np.stack([memory.recall(np.repeat(bank[k:k + 1], len(identity), axis=0)) for k in range(KEYS)], axis=1)
    return output, int(memory.strength.nbytes / len(identity))


def dictionary_read(identity: np.ndarray, values: np.ndarray, bank: np.ndarray) -> np.ndarray:
    out = []
    for ids, vals in zip(identity, values, strict=True):
        memory = {}
        for i, v in zip(ids, vals, strict=True):
            memory[tuple(bank[i])] = int(v)
        out.append([memory[tuple(key)] for key in bank])
    return np.eye(VALUES)[np.asarray(out)]


def exact_attention_read(identity: np.ndarray, values: np.ndarray, bank: np.ndarray) -> np.ndarray:
    """Hard key match with latest-position tie break, no learned parameters.

    All stored vectors are retained. This uses the same explicit parser as the other
    arms and deliberately exposes that exact retrieval already solves this task.
    """
    stored = bank[identity]
    score = np.einsum("btd,kd->bkt", stored, bank)
    match = np.isclose(score, 1.0, atol=1e-12, rtol=0)
    positions = np.arange(identity.shape[1])
    last = np.where(match, positions, -1).max(axis=-1)
    return np.eye(VALUES)[np.take_along_axis(values, last, axis=1)]


def make_transformer(seed: int):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    class Reader(nn.Module):
        def __init__(self):
            super().__init__()
            width = 32
            self.project = nn.Linear(DIM + VALUES + 1, width)
            layer = nn.TransformerEncoderLayer(width, 4, 2 * width, dropout=0,
                                               batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.head = nn.Linear(width, VALUES)

        def forward(self, x):
            h = self.project(x)
            pos = torch.arange(x.shape[1], dtype=h.dtype)[:, None]
            rates = torch.exp(-np.log(10000) * torch.arange(0, h.shape[-1], 2) / h.shape[-1])
            encoding = torch.empty((x.shape[1], h.shape[-1]))
            encoding[:, 0::2] = torch.sin(pos * rates)
            encoding[:, 1::2] = torch.cos(pos * rates)
            return self.head(self.encoder(h + encoding[None]))[:, -KEYS:]

    return Reader()


def tokens(identity: np.ndarray, values: np.ndarray, bank: np.ndarray):
    import torch
    written = np.concatenate([bank[identity], np.eye(VALUES)[values],
                              np.zeros((*identity.shape, 1))], axis=-1)
    query = np.concatenate([bank, np.zeros((KEYS, VALUES)), np.ones((KEYS, 1))], axis=-1)
    return torch.tensor(np.concatenate([written, np.repeat(query[None], len(identity), axis=0)], axis=1), dtype=torch.float32)


def train_transformer(seed: int, steps: int) -> tuple[object, dict]:
    import torch
    model = make_transformer(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0)
    rng = np.random.default_rng(seed + 10000)
    validation = [(c, episodes(seed + 20000 + j, 64, 32)) for j, c in enumerate(CORRELATIONS)]
    history = []
    best_loss, best_state = float("inf"), None
    started = time.perf_counter()
    for step in range(1, steps + 1):
        length = int(rng.choice(TRAIN_LENGTHS))
        correlation = float(rng.choice(CORRELATIONS))
        identity, values, target = episodes(int(rng.integers(2**31)), 32, length)
        logits = model(tokens(identity, values, keys(correlation)))
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, VALUES), torch.tensor(target.ravel()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step in {max(1, steps // 2), steps}:
            model.eval()
            with torch.no_grad():
                val_loss = np.mean([float(torch.nn.functional.cross_entropy(
                    model(tokens(ids, vals, keys(c))).reshape(-1, VALUES), torch.tensor(truth.ravel())))
                    for c, (ids, vals, truth) in validation])
            history.append({"step": step, "validation_loss": float(val_loss)})
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.train()
    model.load_state_dict(best_state)
    model.eval()
    return model, {"parameters": sum(p.numel() for p in model.parameters()), "steps": steps,
                   "episodes": 32 * steps, "seconds": time.perf_counter() - started,
                   "validation": history, "selected_validation_loss": float(best_loss),
                   "architecture": "2 encoder layers, width 32, 4 heads, sinusoidal positions; explicit key/value and query tokens",
                   "training_lengths": list(TRAIN_LENGTHS)}


def readings(output: np.ndarray, truth: np.ndarray, identity: np.ndarray, seconds: float) -> dict:
    prediction = output.argmax(axis=-1)
    newest_key = identity[:, -1]
    newest = prediction[np.arange(len(identity)), newest_key]
    latest_truth = truth[np.arange(len(identity)), newest_key]
    return {"prediction": prediction.tolist(), "latest_prediction": newest.tolist(),
            "correct": int((prediction == truth).sum()), "queries": int(truth.size),
            "accuracy": float((prediction == truth).mean()),
            "latest_accuracy": float((newest == latest_truth).mean()),
            "seconds": seconds}


def run(output: Path, steps: int = 800, seeds: int = 3, trials: int = 100) -> dict:
    import torch
    runs = []
    for seed in range(seeds):
        model, training = train_transformer(seed, steps)
        rows = []
        for mode, correlation, length in CASES:
            bank = keys(correlation)
            identity, values, truth = episodes(seed + 30000 + length, trials, length, balanced=mode == "balanced_first_pass")
            arms = {}
            for arm in ARMS:
                start = time.perf_counter()
                if arm in ("delta", "hebb"):
                    result, memory_bytes = memory_read(identity, values, bank, arm)
                elif arm == "dictionary":
                    result = dictionary_read(identity, values, bank)
                elif arm == "exact_attention":
                    result = exact_attention_read(identity, values, bank)
                else:
                    with torch.inference_mode():
                        result = model(tokens(identity, values, bank)).numpy()
                arms[arm] = readings(result, truth, identity, time.perf_counter() - start)
            row = {"mode": mode, "length": length, "key_correlation": correlation,
                   "length_extrapolation": length > max(TRAIN_LENGTHS),
                   "key_correlation_extrapolation": correlation not in CORRELATIONS,
                   "episode_sha256": hashlib.sha256(identity.tobytes() + values.tobytes()).hexdigest(),
                   "truth": truth.tolist(), "latest_key": identity[:, -1].tolist(),
                   "arms": arms, "fast_matrix_bytes_per_stream": memory_bytes,
                   "attention_key_value_payload_bytes_per_stream": length * (DIM + VALUES) * 8}
            rows.append(row)
            print(f"seed {seed}, {mode}, {length} writes, correlation {correlation}: " +
                  ", ".join(f"{name} {arms[name]['accuracy']:.3f}" for name in ARMS), flush=True)
        runs.append({"seed": seed, "transformer_training": training, "rows": rows})
    body = {"task": {"keys": KEYS, "key_dimensions": DIM, "values": VALUES,
                     "lengths": list(LENGTHS), "correlations": list(CORRELATIONS),
                     "trials_per_condition": trials, "seeds": seeds, "train_steps": steps},
            "runs": runs, "environment": {"python": platform.python_version(),
                "numpy": np.__version__, "torch": torch.__version__, "cadence": cd.__version__,
                "machine": platform.machine(), "device": "cpu", "torch_threads": 1},
            "boundary": {"same_evaluation_episodes": True, "explicit_key_value_parser_for_every_arm": True,
                "delta_and_hebb_have_zero_slow_trained_parameters": True,
                "matrix_entries_are_mutable_memory_not_free_capacity": True,
                "exact_lookup_is_a_stronger_task_specific_baseline": True,
                "not_an_equilibrium_propagation_or_language_model_result": True,
                "timings_include_writes_and_all_reads_for_memory_and_encoding_and_reads_for_transformer": True,
                "transformer_training_time_is_reported_separately": True}}
    cd.Receipt.build("cadence-examples/05-memory/v1", body, sources=sources()).write(output)
    return body


def check(body: dict) -> str | None:
    task = body["task"]
    if task["lengths"] != list(LENGTHS) or task["correlations"] != list(CORRELATIONS):
        return "declared condition schedule differs from the experiment"
    if task["seeds"] < 1 or task["trials_per_condition"] < 1:
        return "empty schedule"
    if [run["seed"] for run in body["runs"]] != list(range(task["seeds"])):
        return "seed schedule differs"
    for run in body["runs"]:
        cells = [(row["mode"], row["key_correlation"], row["length"]) for row in run["rows"]]
        if sorted(cells) != sorted(CASES):
            return "condition schedule differs"
        for row in run["rows"]:
            ids, vals, expected = episodes(run["seed"] + 30000 + row["length"],
                                            task["trials_per_condition"], row["length"],
                                            balanced=row["mode"] == "balanced_first_pass")
            truth = np.asarray(row["truth"])
            latest_key = np.asarray(row["latest_key"])
            if not np.array_equal(truth, expected) or not np.array_equal(latest_key, ids[:, -1]):
                return "targets or latest keys differ from the seeded episodes"
            if hashlib.sha256(ids.tobytes() + vals.tobytes()).hexdigest() != row["episode_sha256"]:
                return "episode digest differs"
            if row["length_extrapolation"] != (row["length"] > max(TRAIN_LENGTHS)):
                return "extrapolation label differs"
            if row["key_correlation_extrapolation"] != (row["key_correlation"] not in CORRELATIONS):
                return "key-correlation extrapolation label differs"
            for arm in ARMS:
                data = row["arms"][arm]
                prediction = np.asarray(data["prediction"])
                if prediction.shape != truth.shape:
                    return f"{arm}: prediction shape differs"
                correct = int((prediction == truth).sum())
                latest = prediction[np.arange(len(truth)), latest_key]
                if correct != data["correct"] or data["queries"] != truth.size:
                    return f"{arm}: query counts differ"
                if abs(correct / truth.size - data["accuracy"]) > 1e-12:
                    return f"{arm}: accuracy differs"
                if latest.tolist() != data["latest_prediction"] or abs(float((latest == truth[np.arange(len(truth)), latest_key]).mean()) - data["latest_accuracy"]) > 1e-12:
                    return f"{arm}: latest-write accuracy differs"
                if arm in ("dictionary", "exact_attention") and correct != truth.size:
                    return f"{arm}: exact baseline did not solve its declared task"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--output", type=Path, default=HERE / "receipt.json")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        ok, message = cd.Receipt.verify(args.verify, sources=sources(), check=check)
        print(message)
        return int(not ok)
    if min(args.steps, args.seeds, args.trials) < 1:
        parser.error("steps, seeds and trials must be positive")
    run(args.output, args.steps, args.seeds, args.trials)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
