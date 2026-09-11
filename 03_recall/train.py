"""03 recall: a memory that is the seams. Associative recall with no trained parameters.

Run:  python train.py                       (about ten minutes on a laptop; the transformer baseline is most of it)
      python train.py --verify receipt.json

A context of key-value pairs, then a query key: answer its value. A patch net does this
without training. Each pair is written as one local Hebbian outer product between the key's
owner and the value's owner, and a query is a settlement with the key clamped: the value
owners come to rest on the answer. One settlement step is exactly an attention read
(Ramsauer et al. 2021); a few more steps clean it. There is no window and no context
length: the memory is the seams. A two-layer transformer trained on the same task in the
same script is the comparison; attention learns recall inside the lengths it saw and does
not carry it beyond them, and in this budget it does not learn it at all.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import cadence as cd

HERE = Path(__file__).resolve().parent
SOURCES = [("03_recall/train.py", Path(__file__).resolve())]
VOCAB = 128
LENGTHS = (4, 8, 16, 32, 48, 64, 96, 128)
TRAIN_MAX = 32  # the transformer trains on contexts of up to this many pairs
TRIALS = 100
D_MODEL, HEADS, LAYERS = 64, 4, 2  # an induction head needs two layers
TRAIN_STEPS = 5_000
CLAMP = 3.0


def episode(rng: np.random.Generator, pairs: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    keys = rng.choice(VOCAB, pairs, replace=False)
    values = rng.integers(0, VOCAB, pairs)
    q = rng.integers(0, pairs)
    return keys, values, int(keys[q]), int(values[q])


class Memory:
    """Key owners and value owners, every key joined to every value by a seam that starts at zero."""

    def __init__(self) -> None:
        k = np.repeat(np.arange(VOCAB), VOCAB)
        v = VOCAB + np.tile(np.arange(VOCAB), VOCAB)
        self.wiring = cd.Wiring.from_edges(
            2 * VOCAB, pre=np.concatenate([k, v]), post=np.concatenate([v, k]), sign=np.zeros(2 * VOCAB * VOCAB),
            sets={"key": range(VOCAB), "value": range(VOCAB, 2 * VOCAB)},
        )
        self.rule = cd.learning_rule(gain=1.0, dt=1.0)
        self.strength = np.zeros(self.wiring.edges)
        self.values = np.asarray(self.wiring.sets["value"])
        self.steps: list[int] = []
        w = self.wiring
        self.forward = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(w.pre, w.post, strict=True))}

    def clear(self) -> None:
        self.strength[:] = 0.0

    def write(self, key: int, value: int, eta: float = 1.0) -> None:
        """One local Hebbian outer product: the key's owner and the value's owner were active together."""
        self.strength[self.forward[(key, VOCAB + value)]] += eta
        self.strength[self.forward[(VOCAB + value, key)]] += eta

    def query(self, key: int, steps: int) -> int:
        engine = cd.Settlement(self.wiring, self.rule, edge_scale=self.strength)
        clamp = np.zeros(self.wiring.n)
        clamp[key] = CLAMP
        state = engine.settle(clamp, steps=steps, tolerance=1e-4 if steps > 1 else None)
        self.steps.append(state.steps)
        return int(np.argmax(state.activation[self.values]))


def recall_patch(seed: int, steps: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    memory = Memory()
    out = {}
    for pairs in LENGTHS:
        hits = 0
        for _ in range(TRIALS):
            keys, values, q, answer = episode(rng, pairs)
            memory.clear()
            for k, v in zip(keys, values, strict=True):
                memory.write(int(k), int(v))
            hits += int(memory.query(q, steps) == answer)
        out[str(pairs)] = hits / TRIALS
    return out


def transformer_arm(seed: int) -> dict[str, Any]:
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    torch.set_num_threads(1)

    class Recall(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token = nn.Embedding(2 * VOCAB + 1, D_MODEL)
            self.position = nn.Embedding(2 * max(max(LENGTHS), TRAIN_MAX) + 2, D_MODEL)
            layer = nn.TransformerEncoderLayer(D_MODEL, HEADS, dim_feedforward=2 * D_MODEL, dropout=0.0, batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, LAYERS)
            self.head = nn.Linear(D_MODEL, VOCAB)

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            pos = torch.arange(tokens.shape[1])[None, :]
            return self.head(self.encoder(self.token(tokens) + self.position(pos))[:, -1])

    def encode(keys: np.ndarray, values: np.ndarray, q: int) -> np.ndarray:
        seq = np.empty(2 * len(keys) + 1, dtype=np.int64)
        seq[0::2][: len(keys)] = keys
        seq[1::2] = VOCAB + values
        seq[-1] = q
        return seq

    rng = np.random.default_rng(seed + 100)
    net = Recall()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, TRAIN_STEPS)
    t0 = time.perf_counter()
    losses = []
    for step in range(TRAIN_STEPS):
        pairs = int(rng.integers(2, TRAIN_MAX + 1))
        batch = [episode(rng, pairs) for _ in range(32)]
        tokens = torch.as_tensor(np.stack([encode(k, v, q) for k, v, q, _ in batch]))
        target = torch.as_tensor([a for _, _, _, a in batch])
        loss = nn.functional.cross_entropy(net(tokens), target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        schedule.step()
        losses.append(float(loss))
        if step % 1000 == 999:
            print(f"  transformer step {step + 1}: loss {np.mean(losses[-200:]):.3f}", flush=True)
    seconds = time.perf_counter() - t0
    rng = np.random.default_rng(seed + 200)
    out = {}
    with torch.no_grad():
        for pairs in LENGTHS:
            hits = 0
            for _ in range(TRIALS):
                keys, values, q, answer = episode(rng, pairs)
                hits += int(int(net(torch.as_tensor(encode(keys, values, q))[None]).argmax(dim=1)) == answer)
            out[str(pairs)] = hits / TRIALS
    return {"model": f"transformer, {LAYERS} layers, d {D_MODEL}, trained {TRAIN_STEPS} steps on 2 to {TRAIN_MAX} pairs", "parameters": sum(p.numel() for p in net.parameters()), "seconds": seconds, "final_loss": float(np.mean(losses[-200:])), "chance_loss": float(np.log(VOCAB)), "accuracy": out}


def run(seed: int, out: Path, seeds: int) -> dict[str, Any]:
    runs = []
    for k in range(seeds):
        s = seed + k
        t0 = time.perf_counter()
        one = recall_patch(s, steps=2)  # the key lights, one transport: an attention read
        settled = recall_patch(s, steps=30)
        patch_seconds = time.perf_counter() - t0
        transformer = transformer_arm(s)
        runs.append({"seed": s, "patch_one_read": one, "patch_settled": settled, "patch_seconds": patch_seconds, "transformer": transformer})
        print(f"seed {s}: one read {one}; settled {settled}; transformer {transformer['accuracy']} ({transformer['seconds']:.0f}s, loss {transformer['final_loss']:.2f} of {transformer['chance_loss']:.2f} chance)", flush=True)
    summary = {arm: {str(L): float(np.mean([(r[arm] if arm != "transformer" else r["transformer"]["accuracy"])[str(L)] for r in runs])) for L in LENGTHS} for arm in ("patch_one_read", "patch_settled", "transformer")}
    memory = Memory()
    engine = cd.Settlement(memory.wiring, memory.rule)
    conformance = cd.conformance(engine, {0: CLAMP}, steps=30)
    body = {
        "task": {"vocabulary": VOCAB, "lengths": list(LENGTHS), "trials_per_length": TRIALS, "transformer_trained_up_to": TRAIN_MAX},
        "patch": {"owners": int(memory.wiring.n), "seams": int(memory.wiring.edges), "trained_parameters": 0, "rule": memory.rule.to_dict(), "clamp": CLAMP, "write": "one Hebbian outer product per pair", "read": "settle with the key clamped; the most active value owner"},
        "runs": runs, "summary": summary, "conformance": conformance,
        "boundary": {"no_parameter_is_trained": True, "baseline_measured_here": True, "no_claim_beyond_this_task": True},
    }
    receipt = cd.Receipt.build("cadence-examples/03-recall/v1", body, sources=SOURCES)
    receipt.write(out)
    print("summary", summary)
    print(f"conformance {conformance['max_abs_deviation']:.1e}; receipt {out} ({receipt.digest[:16]}...)")
    return body


def check(body: dict) -> str | None:
    runs = body["runs"]
    for L in body["task"]["lengths"]:
        if abs(float(np.mean([r["patch_settled"][str(L)] for r in runs])) - body["summary"]["patch_settled"][str(L)]) > 1e-9:
            return "summary does not recompute"
    if body["patch"]["trained_parameters"] != 0:
        return "the patch net trained something"
    if not body["conformance"]["ledger"]["clean"]:
        return "reference ledger is not clean"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=3)
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
