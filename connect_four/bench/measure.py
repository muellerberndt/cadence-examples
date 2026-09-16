"""Measure a Connect Four brain against perfect play and against public opponents.

    python connect_four/bench/measure.py --brain runs/connect_four/dev/brain_seed0_games001200 \
        --games 100 --opponents random one_ply minimax_1024 solver_050 solver_100 --out runs/connect_four/bench/dev0

Every game is paired (the candidate opens the even games) and every column the candidate
chose is scored by Pascal Pons' solver: optimal (a score-maximal column), slip (worse, the
theoretical result kept), blunder (the result changes). The receipt holds, per opponent, the
paired score and the move-quality rates over all moves and by ply band, and the latency of
the candidate's decisions. ``--policy`` measures a supplied-rules policy instead of a brain
(``minimax_1024``, ``one_ply``, ``random``, ``solver_050``, ``alphazero=<checkpoint>@<sims>``).
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

from agent.life import seed_for  # noqa: E402
from connect_four.bench.pons import INVALID, Solver, move_quality  # noqa: E402
from connect_four.brain import STAGE, Brain  # noqa: E402
from connect_four.env import Board, GameConfig, World  # noqa: E402
from connect_four.evaluate import summarize  # noqa: E402
from connect_four.opponents import make_opponent  # noqa: E402
from connect_four.run import play_brain, play_policy  # noqa: E402

BANDS = ((0, 8), (8, 16), (16, 24), (24, 42))


def score_moves(moves: list[int], candidate_first: bool, solver: Solver) -> list[dict]:
    """The quality of every candidate move of a game, replayed from the opening."""
    board = Board()
    out = []
    for ply, column in enumerate(moves):
        candidate_turn = (ply % 2 == 0) == candidate_first
        if candidate_turn and not board.terminal:
            scores = solver.scores(board.relative(board.to_move))
            best = int(scores[scores != INVALID].max())
            out.append({"ply": ply, "column": int(column), "quality": move_quality(scores, int(column)), "best": best, "score": int(scores[column])})
        board.play(int(column))
    return out


def _rates(records: list[dict]) -> dict:
    n = len(records)
    return {"moves": n, **{q: (sum(1 for r in records if r["quality"] == q) / n if n else None) for q in ("optimal", "slip", "blunder")}}


def rates(records: list[dict]) -> dict:
    """Move-quality rates over all moves, over the moves in positions not already lost (where
    a blunder is possible: ``best`` at least 0), in won positions only, and by ply band."""
    out = _rates(records)
    out["not_lost"] = _rates([r for r in records if r["best"] >= 0])
    out["won"] = _rates([r for r in records if r["best"] > 0])
    out["by_band"] = {f"{lo}-{hi}": _rates([r for r in records if lo <= r["ply"] < hi]) for lo, hi in BANDS}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--brain", type=str, help="checkpoint stem (without .agent.npz / .records.npz)")
    p.add_argument("--policy", type=str, help="a supplied-rules policy instead of a brain")
    p.add_argument("--games", type=int, default=100, help="paired games per opponent")
    p.add_argument("--opponents", nargs="+", default=["random", "one_ply", "minimax_1024", "solver_050", "solver_100"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--learning", action="store_true", help="let the brain keep learning during the games (default frozen)")
    a = p.parse_args()
    if bool(a.brain) == bool(a.policy):
        raise SystemExit("give exactly one of --brain or --policy")
    game = GameConfig()
    solver = Solver()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.brain:
        brain = Brain.load(a.brain)
        if not a.learning:
            brain = brain.frozen()
        subject = {"brain": a.brain, "extended": bool(brain.extended), "learning": bool(a.learning)}
    else:
        policy = make_opponent(a.policy, seed_for(STAGE, "bench", a.seed, 0, 0), game, shared={"solver": solver})
        subject = {"policy": a.policy}
    world = World(game, life_id=f"bench-{a.seed}")
    receipt = {"format": "cadence-connect-four-bench/1", "subject": subject, "games_per_opponent": a.games, "seed": a.seed, "opponents": {}}
    started = time.time()
    for k_op, name in enumerate(a.opponents):
        latencies: list[float] = []
        outcomes: list[int] = []
        quality: list[dict] = []
        t0 = time.time()
        for k in range(a.games):
            first = "candidate" if k % 2 == 0 else "opponent"
            opponent = make_opponent(name, seed_for(STAGE, "bench", a.seed, 100 + k_op, k // 2), game, shared={"solver": solver})
            if a.brain:
                record = play_brain(brain, world, first, opponent, latencies)
            else:
                record = play_policy(policy, world, first, opponent)
            outcomes.append(int(record["outcome"]))
            quality.extend(score_moves(record["moves"], first == "candidate", solver))
        entry = {"paired": summarize(outcomes), "quality": rates(quality), "seconds": time.time() - t0}
        if latencies:
            entry["latency_ms"] = {"p50": float(np.percentile(latencies, 50) * 1000), "p95": float(np.percentile(latencies, 95) * 1000), "max": float(np.max(latencies) * 1000)}
        receipt["opponents"][name] = entry
        q, nl = entry["quality"], entry["quality"]["not_lost"]
        print(f"{name:14s} score {entry['paired']['score']:.3f}  optimal {q['optimal']:.3f} slip {q['slip']:.3f} blunder {q['blunder']:.3f} ({q['moves']} moves)"
              f"  not lost: blunder {nl['blunder'] if nl['blunder'] is not None else float('nan'):.3f} ({nl['moves']} moves)  {entry['seconds']:.0f} s", flush=True)
    receipt["seconds"] = time.time() - started
    receipt["solver_calls"] = solver.calls
    (a.out / "receipt.json").write_text(json.dumps(receipt, indent=1))
    solver.close()


if __name__ == "__main__":
    main()
