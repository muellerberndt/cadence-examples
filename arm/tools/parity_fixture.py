"""A fixture for the JavaScript engine: a snapshot of a living arm agent, the following
moments as the environment produced them, and what the Python agent decided and learned.

    python tools/parity_fixture.py --out runs/arm/parity

Writes ``snapshot.json`` (the web export after babbling, the records tables in float32),
``records_f64.json`` (the records head's mean and tables in float64, for the harness),
``stream.jsonl`` (one moment per line with the Python decision that followed it and the
record-write count) and ``final.json`` (parameter digests and arrays after the stream,
the records head in float64). A port that replays the stream must reproduce every
decision, prediction, the final parameters and the records to the stated tolerances.
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

from agent.brain import Agent, ReplayConfig
from agent.life import seed_for
from agent.web import export, records_export
from arm.brain import ModelPlanner, agent_config
from arm.env import Arm, ArmConfig


def moment_record(m) -> dict:
    return {
        "life_id": m.life_id, "episode_id": m.episode_id, "event_id": m.event_id, "tick": m.tick, "dt": m.dt,
        "observation": {k: np.asarray(v).tolist() for k, v in m.observation.items()},
        "observed": {k: np.asarray(v).astype(int).tolist() for k, v in m.observed.items()},
        "action_mask": m.action_mask.astype(int).tolist(), "feedback_for": m.feedback_for, "executed": m.executed,
        "reward": m.reward, "reward_known": m.reward_known, "terminated": m.terminated, "truncated": m.truncated,
        "final_observation": None if m.final_observation is None else {k: np.asarray(v).tolist() for k, v in m.final_observation.items()},
        "goal": None if m.goal is None else np.asarray(m.goal).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE.parent / "config.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--babble", type=int, default=300)
    parser.add_argument("--stream", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    planner = ModelPlanner(**config["planner"])
    cfg = replace(agent_config(config), replay=ReplayConfig(**config["replay"]))
    agent = Agent(cfg, seed=args.seed, planner=planner.for_agent())
    a = dict(config["arm"]); a["lengths"] = tuple(a["lengths"]); a["target_radius"] = tuple(a["target_radius"])
    env = Arm(ArmConfig(**a), seed=seed_for("S01", "parity", args.seed, 0, 0), life_id="parity")
    agent.config = replace(agent.config, actor=replace(agent.config.actor, epsilon=1.0))
    m = env.reset()
    count = 0
    while count < args.babble:
        d = agent.step(m)
        if d is None:
            m = env.reset(); continue
        m = env.act(d.decision_id, d.action); count += 1
    # half exploration, half planning in the recorded stream
    agent.config = replace(agent.config, actor=replace(agent.config.actor, epsilon=0.5))
    export(agent, args.out / "snapshot.json", extra={"next_moment": moment_record(m), "planner": config["planner"], "arm": config["arm"]}, records_sidecar_path=args.out / "records_f64.json")
    with open(args.out / "stream.jsonl", "w") as f:
        for _ in range(args.stream):
            d = agent.step(m)
            record = {"moment": moment_record(m), "decision": None if d is None else d.to_dict(), "parameter_version": agent.parameter_version,
                      "efficacy_sha256": hashlib.sha256(np.ascontiguousarray(agent.brain.efficacy).tobytes()).hexdigest(),
                      "learning": {k: v for k, v in agent.last_report.get("learning", {}).items()},
                      "record_writes": int(agent.ledger.record_writes), "imagined": int(agent.ledger.imagined)}
            f.write(json.dumps(record) + "\n")
            if d is None:
                m = env.reset()
            else:
                m = env.act(d.decision_id, d.action)
    final = {
        "efficacy": agent.brain.efficacy.tolist(), "bias": agent.brain.bias.tolist(),
        "world_velocity": agent.world.velocity.tolist(), "critic_w": agent.w_critic.tolist(), "critic_b": agent.b_critic,
        "context_trace": agent.context.trace[0].tolist(), "rng": {"action": agent.rng[0].state, "replay": agent.replay_rng.state},
        "parameter_version": agent.parameter_version, "ledger": agent.ledger.to_dict(), "ring_stored": len(agent.ring), "ring_at": agent.ring_at,
        "records": records_export(agent, "<f8"),
        "tolerances": {"activation": 1e-9, "efficacy": 1e-8, "prediction": 1e-8, "records": 1e-10},
    }
    (args.out / "final.json").write_text(json.dumps(final))
    print("fixture written:", args.out, "stream", args.stream, "events; final parameter version", agent.parameter_version, "; record writes", agent.ledger.record_writes, "; imagined", agent.ledger.imagined)
    return 0


if __name__ == "__main__":
    sys.exit(main())
