"""Games to learn from: set-up positions played out by the perfect player.

    python connect4/school.py --games 200000 --workers 8 --seed 0 --out runs/connect4/school/s0.npz

A game starts from a set-up position: a number of opening plies drawn uniformly up to
``--setup``, each ply the perfect column with the game's own probability (drawn from
``--strengths``) and a random legal column otherwise. From there Pascal Pons' solver plays
both sides to the end, a best column at every move, ties broken at random. The watcher sees
every position of the play-out right after its stone has landed and, when the game ends, how
it ended for the side that placed that stone and how many plies later. Because both sides
play perfectly from the set-up position on, that outcome is the position's value; no score
of the solver is recorded, only the columns it played.

The file holds one row per watched position: the two boards of its reading (``own``,
``other``), the outcome ``z`` for the side that placed the stone (1 win, 0 draw, -1 loss),
the plies ``n`` from the position to the end and the stones on the board (``plies``).
"""

from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from connect4.bench.pons import INVALID, Solver  # noqa: E402
from connect4.game import Position  # noqa: E402


def perfect_column(solver: Solver, position: Position, rng: np.random.Generator) -> int:
    scores = solver.scores(position.cells())
    best = max(int(v) for v in scores if v != INVALID)
    return int(rng.choice([c for c in range(7) if scores[c] == best]))


def watch(solver: Solver, rng: np.random.Generator, setup: int, strengths: tuple[float, ...], keep: int = 0) -> list[tuple[int, int, int, int, int]]:
    """One game: rows ``(own, other, z, n, plies)`` of the play-out's positions. ``keep`` limits
    the rows to the first that many positions of the play-out (0 keeps them all): the game is
    still played to its end, since the end is what tells the watcher how it went."""
    while True:
        position, strength = Position(), float(rng.choice(strengths))
        for _ in range(int(rng.integers(0, setup + 1))):
            legal = [c for c in position.legal() if not position.wins(c)]
            if not legal:
                break
            column = perfect_column(solver, position, rng) if rng.random() < strength else int(rng.choice(legal))
            if position.wins(column):
                break
            position = position.play(column)
        else:
            if not position.full:
                break
    line = [position] if position.plies else []
    while not (position.lost or position.full):
        position = position.play(perfect_column(solver, position, rng))
        line.append(position)
    last = line[-1]
    winner_moved_at = last.plies if last.lost else None
    rows = []
    for p in line:
        if winner_moved_at is None:
            z = 0
        else:
            z = 1 if (winner_moved_at - p.plies) % 2 == 0 else -1
        own, other = p.key
        rows.append((own, other, z + 1, last.plies - p.plies, p.plies))
    return rows[:keep] if keep else rows


def _work(job: tuple[int, int, int, tuple[float, ...], int]) -> np.ndarray:
    seed, games, setup, strengths, keep = job
    solver, rng, rows = Solver(), np.random.default_rng(seed), []
    for _ in range(games):
        rows.extend(watch(solver, rng, setup, strengths, keep))
    solver.close()
    return np.array(rows, dtype=np.uint64)


def generate(games: int, workers: int, seed: int, setup: int, strengths: tuple[float, ...], keep: int = 0) -> dict[str, np.ndarray]:
    chunk = max(1, min(2000, games // workers))
    jobs = [(seed * 1_000_003 + k, chunk, setup, strengths, keep) for k in range((games + chunk - 1) // chunk)]
    with Pool(workers) as pool:
        parts = pool.map(_work, jobs, chunksize=1)
    rows = np.concatenate(parts)
    return {
        "own": rows[:, 0],
        "other": rows[:, 1],
        "z": rows[:, 2].astype(np.int8) - 1,
        "n": rows[:, 3].astype(np.int8),
        "plies": rows[:, 4].astype(np.int8),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--games", type=int, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--setup", type=int, default=28, help="the most opening plies of a set-up position")
    p.add_argument("--strengths", type=float, nargs="+", default=[0.0, 0.5, 0.8])
    p.add_argument("--keep", type=int, default=0, help="keep only the first that many positions of each play-out (0: all)")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    started = time.time()
    data = generate(a.games, a.workers, a.seed, a.setup, tuple(a.strengths), a.keep)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **data, games=a.games, seed=a.seed, setup=a.setup, keep=a.keep, strengths=np.array(a.strengths))
    z = data["z"]
    print(f"{len(z)} positions from {a.games} games in {time.time() - started:.0f} s: "
          f"win {np.mean(z == 1):.3f} draw {np.mean(z == 0):.3f} loss {np.mean(z == -1):.3f}")


if __name__ == "__main__":
    main()
