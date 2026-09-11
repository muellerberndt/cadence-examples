"""Self-play positions for Connect Four, each labelled with the move a deeper search prefers.

Games are played by two epsilon-random depth-2 searchers, so the positions are varied and
plausible; every non-terminal position is then labelled with the depth-4 best move for the
side to move. The dataset is written to ``data/positions.npz`` with a digest, and the
receipt of the training script records that digest.

Run:  python dataset.py [--games 8000] [--seed 0]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from connect4 import COLS, Position, search

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "positions.npz"
LABEL_DEPTH, PLAY_DEPTH, EPSILON = 4, 2, 0.25


def one_game(seed: int) -> list[tuple[int, int, int, np.ndarray, int]]:
    """Play one game; return (mover, other, plies, planes, label) for every position seen."""
    rng = np.random.default_rng(seed)
    pos = Position()
    out = []
    while not pos.terminal():
        label, _ = search(pos, LABEL_DEPTH)
        out.append((pos.mover, pos.other, pos.plies, pos.planes(), label))
        legal = pos.legal()
        if rng.random() < EPSILON:
            move = int(rng.choice(legal))
        else:
            move, _ = search(pos, PLAY_DEPTH)
        pos = pos.play(move)
    return out


def digest_of(x: np.ndarray, y: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(x, dtype=np.uint8).tobytes())
    h.update(np.ascontiguousarray(y, dtype=np.int8).tobytes())
    return h.hexdigest()


def build(games: int, seed: int, workers: int | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    seen: dict[tuple[int, int], int] = {}
    xs, ys, plies = [], [], []
    with Pool(workers) as pool:
        for k, game in enumerate(pool.imap_unordered(one_game, range(seed * 100_000, seed * 100_000 + games), chunksize=4)):
            for mover, other, ply, planes, label in game:
                key = (mover, other)
                if key in seen:
                    continue
                seen[key] = len(xs)
                xs.append(planes)
                ys.append(label)
                plies.append(ply)
            if (k + 1) % 200 == 0:
                print(f"{k + 1} games, {len(xs)} unique positions", flush=True)
    x = np.asarray(xs, dtype=np.uint8)
    y = np.asarray(ys, dtype=np.int8)
    meta = {
        "games": games,
        "seed": seed,
        "positions": int(len(y)),
        "label_depth": LABEL_DEPTH,
        "play_depth": PLAY_DEPTH,
        "epsilon": EPSILON,
        "label_histogram": [int((y == c).sum()) for c in range(COLS)],
        "mean_plies": float(np.mean(plies)),
        "digest": digest_of(x, y),
    }
    return x, y, meta


def load() -> tuple[np.ndarray, np.ndarray, dict]:
    with np.load(DATA, allow_pickle=True) as f:
        x, y, meta = f["x"], f["y"], f["meta"].item()
    if digest_of(x, y) != meta["digest"]:
        raise ValueError("positions.npz does not match its own digest")
    return x, y, meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    x, y, meta = build(args.games, args.seed, args.workers)
    DATA.parent.mkdir(exist_ok=True)
    np.savez_compressed(DATA, x=x, y=y, meta=np.array(meta, dtype=object))
    print(f"{meta['positions']} positions from {args.games} games -> {DATA} (digest {meta['digest'][:16]}...)")
    print(f"label histogram {meta['label_histogram']}, mean plies {meta['mean_plies']:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
