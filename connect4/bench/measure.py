"""Measure the brain in real games, with every one of its moves graded by the perfect solver.

    python connect4/bench/measure.py --patch runs/connect4/patch/w256/patch.npz \
        --opponents one_ply minimax_2000 solver_070 solver_100 --games 40 --out runs/connect4/bench/w256

Games are paired: the brain opens the even games and the opponent the odd ones, and the two
games of a pair draw the same opponent seed. Pascal Pons' solver grades each column the brain
chose: optimal (a score-maximal column), slip (worse, the theoretical result kept) or blunder
(the result changes). A blunder is only possible where the position is not already lost, so
the blunder rate is taken over those positions, and reported by the stones on the board: the
search reads the end of the game late, and the value patch decides early.

``--policy`` measures an opponent by name in the brain's place, as a control.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from connect4.bench.pons import INVALID, Solver, move_quality  # noqa: E402
from connect4.brain import Brain  # noqa: E402
from connect4.game import Position  # noqa: E402
from connect4.opponents import make_opponent, play  # noqa: E402
from connect4.patch import ValuePatch  # noqa: E402

BANDS = ((0, 8), (8, 16), (16, 24), (24, 42))


def grade(columns: list[int], subject_first: bool, solver: Solver) -> list[dict]:
    position, out = Position(), []
    for ply, column in enumerate(columns):
        if (ply % 2 == 0) == subject_first:
            scores = solver.scores(position.cells())
            best = max(int(v) for v in scores if v != INVALID)
            out.append({"stones": ply, "quality": move_quality(scores, column), "best": best, "score": int(scores[column])})
        position = position.play(column)
    return out


def rates(moves: list[dict]) -> dict:
    open_ = [m for m in moves if m["best"] >= 0]
    n = len(open_)
    return {"moves": len(moves), "optimal": (sum(m["quality"] == "optimal" for m in moves) / len(moves)) if moves else None,
            "not_lost_moves": n, "blunder_when_not_lost": (sum(m["quality"] == "blunder" for m in open_) / n) if n else None}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patch", type=Path)
    p.add_argument("--policy", type=str)
    p.add_argument("--opponents", nargs="+", default=["one_ply", "minimax_2000", "solver_070", "solver_100"])
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--budget", type=int, default=2000)
    p.add_argument("--late-stones", type=int, default=18)
    p.add_argument("--late-budget", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if bool(a.patch) == bool(a.policy):
        raise SystemExit("give exactly one of --patch or --policy")
    solver, shared = Solver(), {}
    shared["solver"] = solver
    subject = Brain(ValuePatch.load(a.patch), reads=a.budget, late_stones=a.late_stones, late_reads=a.late_budget, seed=a.seed) if a.patch else make_opponent(a.policy, a.seed, shared)
    receipt = {"format": "cadence-examples.connect4.bench/1", "subject": str(a.patch or a.policy), "games_per_opponent": a.games,
               "search": {"budget": a.budget, "late_stones": a.late_stones, "late_budget": a.late_budget} if a.patch else None, "opponents": {}}
    a.out.mkdir(parents=True, exist_ok=True)
    for k_opponent, name in enumerate(a.opponents):
        started, results, moves, games = time.time(), {"first": [0, 0, 0], "second": [0, 0, 0]}, [], []
        for g in range(a.games):
            opponent = make_opponent(name, 1000 * (a.seed + 1) + 100 * k_opponent + g // 2, shared)
            subject_first = g % 2 == 0
            columns, winner = play(subject, opponent) if subject_first else play(opponent, subject)
            side = "first" if subject_first else "second"
            results[side][0 if winner == (0 if subject_first else 1) else 1 if winner == -1 else 2] += 1
            moves.extend(grade(columns, subject_first, solver))
            games.append({"columns": "".join(map(str, columns)), "subject_first": subject_first, "winner": winner})
        entry = {"win_draw_loss": results, "all": rates(moves),
                 "by_stones": {f"{lo}-{hi}": rates([m for m in moves if lo <= m["stones"] < hi]) for lo, hi in BANDS},
                 "seconds": time.time() - started, "games": games}
        receipt["opponents"][name] = entry
        r = entry["all"]
        bands = "  ".join(f"{b}: {v['blunder_when_not_lost']:.3f}" if v["blunder_when_not_lost"] is not None else f"{b}: -" for b, v in entry["by_stones"].items())
        print(f"{name:14s} first {results['first']}  second {results['second']}  optimal {r['optimal']:.3f}  "
              f"blunders when not lost {r['blunder_when_not_lost']:.3f} of {r['not_lost_moves']}  [{bands}]  {entry['seconds']:.0f} s", flush=True)
    (a.out / "receipt.json").write_text(json.dumps(receipt, indent=1))
    solver.close()


if __name__ == "__main__":
    main()
