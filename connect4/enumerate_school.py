"""Every reachable position of the early game, labeled exactly by the perfect solver.

    python connect4/enumerate_school.py --plies 10 --workers 7 \
        --out runs/connect4/school/enum_p10.npz

A play-out school samples the early game; this walks it. Breadth-first from the empty
board over legal moves, one row per distinct position key (a position and its mirror
image are one reading, as everywhere else), stopping a line where a stone has completed
four or filled the board. A terminal row's outcome is read off the rules. Every other
row is labeled from Pascal Pons' exact score ``s`` for the side to move: the placer's
outcome is ``z = -sign(s)``, and the plies ``n`` to the end under perfect play follow
from his stone-count convention — a winning mover reaches their ``22 - s``-th stone, a
losing mover's opponent reaches their ``22 + s``-th, a draw fills the board. These are
the same ``z`` and ``n`` a watched perfect play-out records, exact instead of sampled;
``--check`` verifies that against play-out rows of an existing school file.

Within the solver's opening book (12 plies) every score is a lookup, so the whole
early game labels in minutes. The file holds the same fields as ``school.py``.
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


def reachable(plies: int) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int, int, int]]]:
    """Distinct position keys by breadth-first walk: ``(own, other, plies)`` for the
    open positions to label, and finished ``(own, other, z, n, plies)`` terminal rows."""
    frontier, seen = [Position()], set()
    open_rows, terminal_rows = [], []
    for depth in range(1, plies + 1):
        nxt = []
        for position in frontier:
            for column in position.legal():
                after = position.play(column)
                key = after.key
                if key in seen:
                    continue
                seen.add(key)
                if after.lost:  # rows carry z + 1, as label() writes them
                    terminal_rows.append((*key, 2, 0, depth))
                elif after.full:
                    terminal_rows.append((*key, 1, 0, depth))
                else:
                    open_rows.append((*key, depth))
                    nxt.append(after)
        frontier = nxt
    return open_rows, terminal_rows


def label(rows: list[tuple[int, int, int]]) -> np.ndarray:
    """``(own, other, z+1, n, plies)`` for open positions, from the mover's exact score."""
    solver, out = Solver(), []
    for own, other, plies in rows:
        # the reading is the placer's; the solver scores for the mover = the other side
        mover_cells = _cells(other, own)
        scores = solver.scores(mover_cells)
        s = max(int(v) for v in scores if v != INVALID)
        if s > 0:  # the mover wins: the placer loses
            a = (22 - s) - plies // 2
            z, n = -1, 2 * a - 1
        elif s < 0:  # the mover loses: the placer wins
            b = (22 + s) - (plies - plies // 2)
            z, n = 1, 2 * b
        else:
            z, n = 0, 42 - plies
        out.append((own, other, z + 1, n, plies))
    solver.close()
    return np.array(out, dtype=np.uint64)


def _cells(mover: int, other: int) -> np.ndarray:
    from connect4.game import _unpack

    own = _unpack(np.array([mover], dtype=np.uint64))[0]
    opp = _unpack(np.array([other], dtype=np.uint64))[0]
    return (own + 2 * opp).astype(np.int64)


def check(data: dict[str, np.ndarray], school_file: Path, plies: int) -> None:
    """Play-out rows of ``school_file`` at these plies must agree exactly on z and n."""
    from connect4.data import keys_of, packed

    school = np.load(school_file)
    mask = school["plies"] <= plies
    school_keys = np.stack((school["own"][mask], school["other"][mask]), axis=1)
    mine = keys_of(data)
    index = {k: i for i, k in enumerate(packed(mine).tolist())}
    z, n = school["z"][mask], school["n"][mask]
    matched = mismatched = missing = 0
    for k, (key, zz, nn) in enumerate(zip(packed(school_keys).tolist(), z, n, strict=True)):
        i = index.get(key)
        if i is None:
            missing += 1
            continue
        if int(data["z"][i]) == int(zz) and int(data["n"][i]) == int(nn):
            matched += 1
        else:
            mismatched += 1
    print(f"check against {school_file.name}: {matched} matched, {mismatched} mismatched, "
          f"{missing} not enumerated (unreachable plies or deeper)")
    if mismatched:
        raise SystemExit("enumerated labels disagree with watched perfect play")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plies", type=int, default=10)
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--check", type=Path, help="a school file whose early rows verify the labels")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    started = time.time()
    open_rows, terminal_rows = reachable(a.plies)
    print(f"{len(open_rows)} open + {len(terminal_rows)} terminal distinct positions "
          f"to {a.plies} plies in {time.time() - started:.0f} s", flush=True)
    chunk = max(1, len(open_rows) // (8 * a.workers))
    jobs = [open_rows[s:s + chunk] for s in range(0, len(open_rows), chunk)]
    with Pool(a.workers) as pool:
        parts = pool.map(label, jobs, chunksize=1)
    rows = np.concatenate(parts + [np.array(terminal_rows, dtype=np.uint64).reshape(-1, 5)])
    data = {"own": rows[:, 0], "other": rows[:, 1], "z": rows[:, 2].astype(np.int8) - 1,
            "n": rows[:, 3].astype(np.int8), "plies": rows[:, 4].astype(np.int8)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **data, games=0, seed=0, setup=a.plies, keep=0,
                        strengths=np.array([1.0]))
    z = data["z"]
    print(f"{len(z)} rows in {time.time() - started:.0f} s: "
          f"win {np.mean(z == 1):.3f} draw {np.mean(z == 0):.3f} loss {np.mean(z == -1):.3f}", flush=True)
    if a.check:
        check(data, a.check, a.plies)


if __name__ == "__main__":
    main()
