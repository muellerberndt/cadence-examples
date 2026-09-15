#!/usr/bin/env python3
"""Record a Connect Four parity fixture: the Python brain of one checkpoint plays games against
a fixed random opponent with learning on, and every step it takes is written down.

    python connect_four/tools/parity_fixture.py --run runs/connect_four/acceptance --seed 10 \
        --games 32 --out runs/connect_four/parity
    python connect_four/tools/parity_fixture.py --train 40 --games 32 --out runs/connect_four/parity

Written into the output directory:

    checkpoint.json  the checkpoint the page loads (the agent snapshot and the brain block)
    fixture.json     the games, and one record per call of ``Brain.step``
    final.json       the record tables of both cortices in float64 after the last game

``--train N`` trains a brain from scratch instead of taking a checkpoint from a run, so a
harness needs no artifacts of its own.

``connect_four/web/parity.mjs`` replays the same games through ``connect_four/web/brain.js``
from the same checkpoint and holds it to these numbers: the chosen column, the predicted
landings and the imagined next board exactly, the read values and the record tables to 1e-9.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (str(ROOT), str(ROOT / "connect_four" / "web")):
    if path not in sys.path:
        sys.path.insert(0, path)

import build_page  # noqa: E402  the exporter of connect_four/web

from agent.life import seed_for  # noqa: E402
from connect_four.brain import CANDIDATE, VIEW, Brain  # noqa: E402
from connect_four.env import STAGE, GameConfig, World  # noqa: E402
from connect_four.opponents import RandomOpponent  # noqa: E402

FORMAT = "cadence-connect-four-parity/1"


def moments_of(brain: Brain) -> dict[str, list[float]]:
    """The moments of every record table after a step: sum, absolute sum, square sum, largest."""
    out: dict[str, list[float]] = {}
    for name, cortex in (("drop", brain.drop.records), ("lines", brain.lines.records)):
        for field, table in cortex.tables.items():
            out[f"{name}/{field}"] = [float(table.sum()), float(np.abs(table).sum()), float((table * table).sum()), float(np.abs(table).max())]
        out[f"{name}/mean"] = [float(cortex.mean.sum()), int(cortex.seen), int(cortex.writes)]
    return out


def b64(array: np.ndarray) -> dict[str, Any]:
    data = np.ascontiguousarray(np.asarray(array).astype("<f8"))
    return {"dtype": "f8", "shape": list(data.shape), "b64": base64.b64encode(data.tobytes()).decode("ascii")}


def record_step(brain: Brain, moment, decision, game: int, extra: dict | None = None) -> dict[str, Any]:
    return {
        "game": game,
        "event": int(moment.event_id),
        "episode": int(moment.episode_id),
        "kind": "intermediate" if (moment.has_feedback and not moment.terminated) else "terminal" if moment.terminated else "truncated" if moment.truncated else "decision",
        "decision": None if decision is None else {"decision_id": int(decision.decision_id), "column": int(decision.action), "controller": decision.controller,
                                                   "budget": {k: int(v) for k, v in decision.budget.items()},
                                                   "prediction": {k: np.asarray(v).tolist() for k, v in decision.prediction.items()},
                                                   **(extra or {})},
        "counts": {k: int(v) for k, v in brain.counts.items()},
        "planner": {"nodes": brain.planner.nodes, "searches": brain.planner.searches, "invalid": brain.planner.invalid},
        "validity": len(brain.validity),
        "extended": bool(brain.extended),
        "moments": moments_of(brain),
    }


def play(brain: Brain, world: World, first: str, opponent, game: int, steps: list[dict]) -> dict[str, Any]:
    """One game, recorded: every moment that enters ``Brain.step`` and what it answered."""
    watch: dict[str, Any] = {}
    original = brain.planner.search

    def search(board, legal, depth, budget):
        columns = np.flatnonzero(np.asarray(legal, bool))
        unchosen, chosen = brain.drop.landings(VIEW[CANDIDATE][board])
        read = brain.imagination.one_step(np.asarray(board, dtype=np.int8), CANDIDATE, columns)
        t0 = time.perf_counter()
        column, info = original(board, legal, depth, budget)
        watch.clear()
        watch.update({"landings": [[int(v) for v in unchosen], [int(v) for v in chosen]], "columns": [int(c) for c in columns],
                      "read": [float(v) for v in read], "search": {"value": float(info["value"]), "depth": int(info["depth"]),
                      "expansions": int(info["expansions"]), "invalid": int(info["invalid"]), "valid": float(info["valid"])},
                      "seconds": time.perf_counter() - t0})
        return column, info

    brain.planner.search = search
    replies: list[int] = []
    opening = None
    m = world.reset(first=first, opponent=opponent)
    if first == "opponent":
        opening = int(world.state()["moves"][0])
    while True:
        d = brain.step(m)
        steps.append(record_step(brain, m, d, game, dict(watch)))
        if d is None:
            break
        mid = world.act(d.decision_id, d.action)
        if brain.step(mid) is not None:
            raise RuntimeError("the intermediate moment produced a decision")
        steps.append(record_step(brain, mid, None, game))
        if mid.terminated:
            break
        m = world.reply()
        replies.append(int(world.state()["moves"][-1]))
    brain.planner.search = original
    state = world.state()
    return {"index": game, "first": first, "opening": opening, "replies": replies, "moves": [int(c) for c in state["moves"]], "outcome": int(world.outcome())}


def train(config_path: Path, seed: int, games: int, out: Path) -> Path:
    """A brain trained from scratch against the random opponent, saved as a checkpoint.

    The first quarter of the games are random legal ones, as the stage's curriculum opens; the
    rest are played by the planner. The checkpoint this writes is the fixture's starting point,
    so a harness needs no run directory."""
    config = json.loads(config_path.read_text())
    game = GameConfig(**config["game"])
    brain = Brain(config, seed)
    world = World(game, life_id=f"parity-train-{seed}")
    opponent = RandomOpponent(seed_for(STAGE, "parity-train", seed, 0, 1))
    explore = max(1, games // 4)
    for k in range(games):
        brain.set_epsilon(1.0 if k < explore else float(config["training"]["epsilon"]))
        m = world.reset(first="candidate" if k % 2 == 0 else "opponent", opponent=opponent)
        while True:
            d = brain.step(m)
            if d is None:
                break
            mid = world.act(d.decision_id, d.action)
            brain.step(mid)
            if mid.terminated:
                break
            m = world.reply()
    brain.set_epsilon(0.0)
    stem = out / f"trained_seed{seed}_games{games:06d}"
    brain.save(stem)
    print(f"trained: {games} games against the random opponent, {brain.counts['transitions']:,} transitions, validity {brain.state()['validity']}")
    return stem


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=None, help="a run directory to take the checkpoint from")
    parser.add_argument("--train", type=int, default=None, help="instead of a run, train a brain from scratch for this many games against the random opponent")
    parser.add_argument("--config", type=Path, default=ROOT / "connect_four" / "config.json")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--checkpoint", type=int, default=None, help="the game count of the checkpoint to start from (default: the last)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (args.run is None) == (args.train is None):
        raise SystemExit("name a run directory with --run, or a number of games to train with --train")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.train is not None:
        seed = 0 if args.seed is None else args.seed
        entry = {"stem": train(args.config, seed, args.train, args.out), "games": args.train, "labels": ["trained"]}
    else:
        seed, entries, _ = build_page.checkpoints_of(args.run, args.seed)
        entry = entries[-1] if args.checkpoint is None else next((e for e in entries if e["games"] == args.checkpoint), None)
        if entry is None:
            raise SystemExit(f"no checkpoint at {args.checkpoint} games in {args.run}")
    payload = build_page.checkpoint_export(entry["stem"], args.out / "checkpoint.json", build_page.label_of(entry), entry["games"])
    (args.out / "cortices.json").write_text(json.dumps(payload.pop("cortices"), separators=(",", ":")), encoding="utf-8")
    print(f"checkpoint: {entry['stem']} ({entry['games']:,} games, {payload['brain']['counts']['transitions']:,} transitions)")
    if payload["expansion_difference"]:
        print(f"  the expansion this life was drawn with differs from the one this machine regenerates by {payload['expansion_difference']:.3e}; the checkpoint's is used")

    brain, _ = build_page.load_brain(entry["stem"])
    game_config = GameConfig(**{k: payload["brain"]["game"][k] for k in ("rows", "cols", "connect")})
    world = World(game_config, life_id=f"parity-{seed}")
    brain.new_world()
    opponent = RandomOpponent(seed_for(STAGE, "parity", seed, 0, 0))
    steps: list[dict] = []
    games = []
    t0 = time.time()
    for k in range(args.games):
        first = "candidate" if k % 2 == 0 else "opponent"
        games.append(play(brain, world, first, opponent, k, steps))
    seconds = time.time() - t0
    fixture = {
        "format": FORMAT, "checkpoint": "checkpoint.json", "seed": seed, "life_id": world.life_id,
        "games": games, "steps": steps, "wall_seconds": seconds,
        "tolerances": {"table": 1e-9, "read": 1e-9, "moment": 1e-9},
    }
    (args.out / "fixture.json").write_text(json.dumps(fixture, separators=(",", ":")), encoding="utf-8")
    final = {
        "format": FORMAT, "counts": {k: int(v) for k, v in brain.counts.items()},
        "planner": brain.planner.stats(), "planner_generator": {k: str(v) for k, v in brain.planner.rng.bit_generator.state["state"].items()},
        "validity": [bool(v) for v in brain.validity], "state": json.loads(json.dumps(brain.state(), default=float)),
        "cortices": {name: {"mean": b64(cortex.mean), "tables": {field: b64(table) for field, table in cortex.tables.items()},
                            "seen": int(cortex.seen), "writes": int(cortex.writes)}
                     for name, cortex in (("drop", brain.drop.records), ("lines", brain.lines.records))},
    }
    (args.out / "final.json").write_text(json.dumps(final, separators=(",", ":")), encoding="utf-8")
    decisions = sum(1 for s in steps if s["decision"] is not None)
    outcomes = [g["outcome"] for g in games]
    print(f"fixture: {len(games)} games, {len(steps)} steps, {decisions} decisions, score {float(np.mean((np.array(outcomes) + 1) / 2)):.3f}, {seconds:.1f} s")
    print(f"written: {args.out / 'fixture.json'} ({(args.out / 'fixture.json').stat().st_size / 1e6:.2f} MB), {args.out / 'final.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
