"""The look-ahead suite: positions whose only sound move is refuted far away.

A position is a trap of horizon ``h`` when the perfect solver says at least one legal column
keeps the theoretical result and at least one other loses it, and every refutation of the
losing columns needs more than ``h`` plies to arrive. The distance comes from the solver's
score: a position whose score is ``s`` is decided ``42 - stones - 2 * |s| + 1`` plies later
when ``s`` is not zero, which is the length of the winning line with perfect play from both
sides. A player that searches ``h`` plies and evaluates leaves by a flat number cannot tell
the columns apart; a player whose evaluation carries the structure can.

    python connect_four/bench/lookahead.py --brain <stem> --depth 8 --budget 32768 --out <dir>

The suite is drawn once per seed from random play that refuses to complete a line, so it does
not depend on the candidate. Positions are stratified by horizon: 4, 6, 8 and beyond.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from connect_four.bench.pons import INVALID, Solver
from connect_four.brain import Brain
from connect_four.env import Board, GameConfig
from connect_four.opponents import make_opponent

BANDS = (4, 6, 8, 10)


def decided_in(board: Board, score: int) -> int:
    """The plies from ``board`` to the end of the game under perfect play, for a decided score."""
    stones = sum(1 for v in board.cells if v)
    return 42 - stones - 2 * abs(int(score)) + 1


def horizon_of(board: Board, scores: np.ndarray, solver: Solver) -> int | None:
    """How deep a search must look to tell the sound columns from the losing ones, or None when
    every column keeps the result or none does."""
    playable = [c for c in range(7) if scores[c] != INVALID]
    best = max(int(scores[c]) for c in playable)
    sound = [c for c in playable if int(scores[c]) == best or np.sign(scores[c]) == np.sign(best)]
    losing = [c for c in playable if c not in sound]
    if not losing or not sound or best < 0:
        return None
    worst = []
    for c in losing:
        board.play(c)
        try:
            child = solver.scores(board.relative(board.to_move))
            playable_child = [k for k in range(7) if child[k] != INVALID]
            if not playable_child:
                worst.append(1)
            else:
                worst.append(decided_in(board, max(int(child[k]) for k in playable_child)))
        finally:
            board.undo()
    return int(min(worst)) + 1  # the ply of the candidate's own move


def band_of(horizon: int) -> int:
    """The index of the horizon band a trap falls in."""
    return max(k for k, low in enumerate(BANDS) if horizon >= low)


def build(size: int, seed: int, solver: Solver) -> list[dict]:
    """``size`` trap positions with their horizons, drawn from random line-free play, the same
    number in every horizon band (random play yields long traps far more often than short)."""
    rng = np.random.default_rng(seed)
    per_band = size // len(BANDS)
    out: list[list[dict]] = [[] for _ in BANDS]
    seen: set[bytes] = set()
    while any(len(b) < per_band for b in out):
        board = Board()
        while not board.terminal:
            cols = board.legal_columns()
            safe = [c for c in cols if not board.would_connect(c, 0)] or cols
            key = board.key()
            if key not in seen and 6 <= sum(1 for v in board.cells if v) <= 30:
                seen.add(key)
                scores = solver.scores(board.relative(board.to_move))
                h = horizon_of(board, scores, solver)
                if h is not None and h >= BANDS[0] and len(out[band_of(h)]) < per_band:
                    out[band_of(h)].append({"moves": list(board.moves), "horizon": int(h),
                                            "scores": [int(s) for s in scores], "to_move": int(board.to_move)})
            board.play(int(rng.choice(safe)))
    return [case for band in out for case in band]


def judge(cases: list[dict], choose) -> dict:
    """The rate at which ``choose(cells, legal) -> column`` keeps the theoretical result, over
    the suite and per horizon band."""
    rows = []
    for case in cases:
        board = Board.from_moves(case["moves"])
        column = int(choose(board.relative(board.to_move), board.legal_mask()))
        scores = np.array(case["scores"])
        playable = scores[scores != INVALID]
        best = int(playable.max())
        rows.append({"horizon": case["horizon"], "sound": bool(np.sign(scores[column]) == np.sign(best) or int(scores[column]) == best)})
    out = {"cases": len(rows), "sound": float(np.mean([r["sound"] for r in rows])) if rows else None, "by_horizon": {}}
    for k, low in enumerate(BANDS):
        high = BANDS[k + 1] if k + 1 < len(BANDS) else 99
        band = [r for r in rows if low <= r["horizon"] < high]
        out["by_horizon"][f"{low}-{high if high < 99 else 'up'}"] = {"cases": len(band), "sound": float(np.mean([r["sound"] for r in band])) if band else None}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--brain", type=str)
    p.add_argument("--policy", type=str)
    p.add_argument("--cases", type=int, default=300)
    p.add_argument("--suite-seed", type=int, default=99)
    p.add_argument("--depth", type=int)
    p.add_argument("--budget", type=int)
    p.add_argument("--threat-weight", type=float, help="weight of the threat summary in the leaf value")
    p.add_argument("--parity-weight", type=float, help="weight of the threat rows in the threat summary")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    solver = Solver()
    suite_file = HERE / f"lookahead_{a.suite_seed}_{a.cases}.json"
    if suite_file.exists():
        cases = json.loads(suite_file.read_text())["cases"]
    else:
        cases = build(a.cases, a.suite_seed, solver)
        suite_file.write_text(json.dumps({"format": "cadence-connect-four-lookahead/1", "seed": a.suite_seed, "cases": cases}, indent=1))
    a.out.mkdir(parents=True, exist_ok=True)
    if a.brain:
        brain = Brain.load(a.brain).frozen()
        if a.depth is not None:
            brain.extended_depth = brain.depth = int(a.depth)
        if a.budget is not None:
            brain.extended_budget = brain.budget = int(a.budget)
        if a.threat_weight is not None:
            brain.imagination.threat_weight = float(a.threat_weight)
        if a.parity_weight is not None:
            brain.imagination.parity_weight = float(a.parity_weight)
        depth, budget = (brain.extended_depth, brain.extended_budget) if brain.extended else (brain.depth, brain.budget)

        def choose(cells, legal):
            return brain.planner.search(np.asarray(cells, dtype=np.int8).ravel(), np.asarray(legal, bool), depth, budget)[0]

        subject = {"brain": a.brain, "depth": depth, "budget": budget, "threat_weight": brain.imagination.threat_weight}
    else:
        policy = make_opponent(a.policy, 7, GameConfig(), {"solver": solver})
        choose = policy
        subject = {"policy": a.policy}
    result = judge(cases, choose)
    receipt = {"format": "cadence-connect-four-lookahead-result/1", "subject": subject, "suite": {"seed": a.suite_seed, "cases": len(cases)}, **result}
    (a.out / "lookahead.json").write_text(json.dumps(receipt, indent=1))
    bands = "  ".join(f"{k}: {v['sound']:.3f} ({v['cases']})" for k, v in result["by_horizon"].items() if v["sound"] is not None)
    print(f"sound {result['sound']:.3f} over {result['cases']} traps   {bands}", flush=True)


if __name__ == "__main__":
    main()
