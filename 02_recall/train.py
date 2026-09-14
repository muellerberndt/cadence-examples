"""02 Recall: write, read back, and correct a changing association with residual fast seams.

No offline training is needed. Run python train.py, or open index.html and overwrite a key.
The larger matched transformer comparison lives in example 05.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cadence as cd
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCES = [("02_recall/train.py", Path(__file__).resolve())]
SOURCES += [(f"cadence/{p.name}", p) for p in sorted(Path(cd.__file__).parent.glob("*.py"))]
VOCAB = 128
LENGTHS = (4, 8, 16, 32, 48, 64, 96, 128)
TRIALS = 100


class Memory:
    """A one-hot key addresses a row; the row stores a revisable value, not a vote count."""

    def __init__(self, rule: str = "delta") -> None:
        self.bank = np.eye(VOCAB)
        self.fast = cd.FastSeams(np.arange(VOCAB), np.arange(VOCAB, 2 * VOCAB), rule=rule)

    def clear(self) -> None:
        self.fast.reset(1)

    def write(self, key: int, value: int) -> None:
        self.fast.observe(self.bank[key : key + 1], self.bank[value : value + 1])

    def read(self, key: int) -> np.ndarray:
        return self.fast.recall(self.bank[key : key + 1])[0]

    def query(self, key: int) -> int:
        return int(self.read(key).argmax())


def run(seed: int, out: Path, seeds: int = 3) -> dict:
    rows = []
    for offset in range(seeds):
        rng = np.random.default_rng(seed + offset)
        for length in LENGTHS:
            counts = {name: 0 for name in ("distinct", "revised", "additive_revised", "dictionary")}
            memory, additive = Memory(), Memory("hebb")
            for _ in range(TRIALS):
                memory.clear()
                additive.clear()
                keys = rng.choice(VOCAB, length, replace=False)
                values = rng.integers(VOCAB, size=length)
                exact = {}
                for key, value in zip(keys, values, strict=True):
                    memory.write(int(key), int(value))
                    additive.write(int(key), int(value))
                    exact[int(key)] = int(value)
                key = int(rng.choice(keys))
                old = exact[key]
                counts["distinct"] += memory.query(key) == old
                for _ in range(20):
                    memory.write(key, old)
                    additive.write(key, old)
                new = (old + int(rng.integers(1, VOCAB))) % VOCAB
                memory.write(key, new)
                additive.write(key, new)
                exact[key] = new
                counts["revised"] += memory.query(key) == new
                counts["additive_revised"] += additive.query(key) == new
                counts["dictionary"] += exact[key] == new
            rows.append({"seed": seed + offset, "length": length, "queries": TRIALS,
                         "correct": counts, "accuracy": {k: v / TRIALS for k, v in counts.items()}})
    body = {"task": {"vocabulary": VOCAB, "lengths": list(LENGTHS), "seeds": seeds,
                     "trials": TRIALS, "repeated_old_writes": 20}, "rows": rows,
            "memory": {"rule": "delta", "slow_trained_parameters": 0, "mutable_entries": VOCAB**2},
            "boundary": "Explicit orthogonal keys make this exact lookup. Dictionary also succeeds. No general transformer advantage is claimed."}
    out.parent.mkdir(parents=True, exist_ok=True)
    cd.Receipt.build("cadence-examples/02-recall/v2", body, sources=SOURCES).write(out)
    print({k: float(np.mean([r['accuracy'][k] for r in rows])) for k in rows[0]['accuracy']})
    return body


def check(body: dict) -> str | None:
    if len(body["rows"]) != body["task"]["seeds"] * len(body["task"]["lengths"]):
        return "incomplete schedule"
    for row in body["rows"]:
        for arm, correct in row["correct"].items():
            if not 0 <= correct <= row["queries"] or abs(correct / row["queries"] - row["accuracy"][arm]) > 1e-12:
                return "counts and accuracy disagree"
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--output", type=Path, default=HERE / "receipt.json")
    p.add_argument("--verify", type=Path)
    args = p.parse_args()
    if args.verify:
        ok, message = cd.Receipt.verify(args.verify, sources=SOURCES, check=check)
        print(message)
        return 0 if ok else 1
    if args.seeds < 1:
        p.error("--seeds must be positive")
    run(args.seed, args.output, args.seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
