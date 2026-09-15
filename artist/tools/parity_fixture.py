#!/usr/bin/env python3
"""A fixture for the JavaScript artist: a snapshot of a living artist, the drawings that follow
it, and every moment, decision, intention and record write the Python life produced.

    python artist/tools/parity_fixture.py \
        --checkpoint runs/artist/pilot/artist_seed0_scribble.npz --out runs/artist/parity

Writes ``snapshot.json`` (the web export at the boundary, the records tables in float32),
``records_f64.json`` (the records head's mean and tables in float64, for the harness),
``drawings.json`` (the targets and the body poses the drawings start from), ``stream.jsonl``
(one line per step: the moment the world produced, the intention the slow level committed, the
decision that followed and the counters after it) and ``final.json`` (the records head in
float64, the ledger, the generators and the measures of every drawing).

``artist/web/parity.mjs`` replays the drawings through ``artist/web/artist.js`` with its own
canvas world and holds every moment, decision, intention, prediction and record table to the
tolerances recorded here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from agent.brain import Agent
from agent.web import export, records_export
from artist.brain import ArtistLife
from artist.env import FAMILIES, PREDICTED, drawing
from artist.run import build_world, geometry

FORMAT = "cadence-artist-parity/1"


def moment_record(m) -> dict:
    return {
        "life_id": m.life_id, "episode_id": m.episode_id, "event_id": m.event_id, "tick": m.tick, "dt": m.dt,
        "observation": {k: np.asarray(v, float).tolist() for k, v in m.observation.items()},
        "action_mask": np.asarray(m.action_mask).astype(int).tolist(), "feedback_for": m.feedback_for, "executed": m.executed,
        "reward": float(m.reward), "reward_known": bool(m.reward_known), "terminated": bool(m.terminated), "truncated": bool(m.truncated),
        "final_observation": None if m.final_observation is None else {k: np.asarray(v, float).tolist() for k, v in m.final_observation.items()},
        "goal": None if m.goal is None else np.asarray(m.goal, float).tolist(),
    }


def intention_record(life) -> dict:
    i = life.intention
    return {
        "endpoint": [float(x) for x in np.asarray(i.endpoint, float)], "pen": int(i.pen),
        "direction": [float(x) for x in np.asarray(i.direction, float)], "distance": float(i.distance),
        "score": float(i.score), "valid": bool(i.valid), "countdown": int(life.countdown),
        "goal": [float(x) for x in i.encode(max(life.search.distances), life.intention_gain)],
    }


def drawing_record(target, config) -> dict:
    return {
        "index": int(target.index), "family": target.family, "split": target.split,
        "rotation": float(target.rotation), "scale": float(target.scale),
        "theta": [float(x) for x in np.asarray(target.theta, float)],
        "target": np.asarray(target.target, bool).astype(int).ravel().tolist(),
        "size": int(config.size),
    }


def efficacy_sha(agent: Agent) -> str:
    return hashlib.sha256(np.ascontiguousarray(agent.brain.efficacy).tobytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=HERE.parent / "config.json")
    parser.add_argument("--checkpoint", type=Path, default=None, help="a brain run.py saved; without one the fixture scribbles a fresh life")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--decisions", type=int, default=120, help="the fewest decisions the stream must hold")
    parser.add_argument("--drawings", type=int, default=2, help="the fewest drawings the stream must cross")
    parser.add_argument("--per-drawing", type=int, default=80, help="the most decisions one drawing may take")
    parser.add_argument("--epsilon", type=float, default=None, help="exploration rate of the recorded stream (default: the config's)")
    parser.add_argument("--scribble", type=int, default=400, help="decisions of random scribbling when no checkpoint is given")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    geom = geometry(config)

    life = ArtistLife(config, args.seed)
    if args.checkpoint is not None:
        life.agent = Agent.load(args.checkpoint, planner=lambda *_a, **_k: (0, {}, {}))
        life.agent.planner = life._planner()
        print(f"checkpoint: {args.checkpoint} ({life.agent.ledger.real_transitions:,} decisions lived)")
    else:
        world = build_world(config, geom, args.seed, "scribble")
        life.planning = False
        life.set_epsilon(1.0)
        world.config = replace(geom, max_decisions=100, patience=10**9)
        done, index = 0, 0
        while done < args.scribble:
            target = drawing("development", 10**5 + index, args.seed, world.config)
            index += 1
            life.detach()
            m = world.reset(target)
            life.begin(world.view())
            while True:
                d = life.step(m, world.view())
                if d is None:
                    break
                m = life.feedback(world, d)
                done += 1
        world.config = geom
        life.planning = True
        print(f"scribbled {done} decisions")

    life.set_epsilon(config["actor"]["epsilon"] if args.epsilon is None else args.epsilon)
    life.detach()  # the boundary the page opens on: no pending decision, fresh cursors
    export(
        life.agent, args.out / "snapshot.json",
        extra={
            "stage": "S04", "boundary": True, "next_moment": None, "config": config,
            "canvas": {k: (list(v) if isinstance(v, (tuple, list)) else v) for k, v in vars(geom).items()},
            "intention": config["intention"], "intention_interval": config["intention_interval"], "intention_gain": config["intention_gain"],
            "planner": config["planner"], "body": config["body"], "families": list(FAMILIES),
            "source": None if args.checkpoint is None else str(args.checkpoint),
        },
        records_sidecar_path=args.out / "records_f64.json",
    )

    world = build_world(config, geom, args.seed, "parity")
    capped = replace(geom, max_decisions=args.per_drawing)
    world.config = capped
    targets, k = [], 0
    stream = open(args.out / "stream.jsonl", "w")
    measures, total, index = [], 0, 0
    while total < args.decisions or len(targets) < args.drawings:
        target = drawing("development", 7000 + index, args.seed, capped, families=FAMILIES)
        index += 1
        targets.append(target)
        life.detach()
        m = world.reset(target)
        life.begin(world.view())
        step = 0
        while True:
            before = intention_record(life)
            decision = life.step(m, world.view())
            record = {
                "drawing": len(targets) - 1, "step": step, "moment": moment_record(m),
                "intention_before": before, "intention": intention_record(life), "budget": dict(life.budget),
                "pen_pixel": [float(x) for x in np.asarray(life.pen_pixel, float)],
                "sheet_value": float(life.sheet.value), "discrepancy": float(world.discrepancy),
                "decision": None if decision is None else decision.to_dict(),
                "parameter_version": int(life.agent.parameter_version), "efficacy_sha256": efficacy_sha(life.agent),
                "learning": {kk: (float(vv) if isinstance(vv, float) else vv) for kk, vv in life.agent.last_report.get("learning", {}).items() if not isinstance(vv, (list, dict))},
                "ledger": {"record_writes": int(life.agent.ledger.record_writes), "imagined": int(life.agent.ledger.imagined), "real_transitions": int(life.agent.ledger.real_transitions)},
            }
            stream.write(json.dumps(record) + "\n")
            step += 1
            if decision is None:
                break
            m = life.feedback(world, decision)
            total += 1
        measures.append({"drawing": len(targets) - 1, "decisions": step - 1, **{kk: float(vv) for kk, vv in world.measure().items()}, "discrepancy": float(world.discrepancy)})
        k += 1
    stream.close()

    final = {
        "format": FORMAT,
        "records": records_export(life.agent, "<f8"),
        "ledger": life.agent.ledger.to_dict(),
        "rng": {"action": int(life.agent.rng[0].state), "replay": int(life.agent.replay_rng.state)},
        "parameter_version": int(life.agent.parameter_version),
        "efficacy": life.agent.brain.efficacy.tolist(),
        "bias": life.agent.brain.bias.tolist(),
        "critic_w": life.agent.w_critic.tolist(), "critic_b": float(life.agent.b_critic),
        "context_trace": life.agent.context.trace[0].tolist(),
        "measures": measures,
        "predicted_fields": list(PREDICTED),
        "tolerances": {"moment": 1e-9, "read": 1e-9, "table": 1e-9, "intention": 1e-9, "sheet": 1e-9},
    }
    (args.out / "final.json").write_text(json.dumps(final))
    (args.out / "drawings.json").write_text(json.dumps({"format": FORMAT, "drawings": [drawing_record(t, capped) for t in targets], "per_drawing": args.per_drawing}))
    print(f"fixture written: {args.out}; {len(targets)} drawings, {total} decisions, {life.agent.ledger.record_writes:,} record writes, {life.agent.ledger.imagined:,} imagined reads")
    for row in measures:
        print(f"  drawing {row['drawing']}: {row['decisions']} decisions, chamfer {row['chamfer']:.4f}, f1 {row['f1']:.3f}, {int(row['ink'])} pixels inked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
