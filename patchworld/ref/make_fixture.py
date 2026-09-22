"""Produce ref/fixture.json from cadence.RecordPatchNet: the numbers sim/parity.js must reproduce.

Run with the library installed or on the path:
    python3 ref/make_fixture.py [--out file.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from record_plan import plan  # noqa: E402

from cadence import RecordPatchNet  # noqa: E402

CONFIG = dict(inputs=6, hidden=5, outputs=4, seed=3, cells=64, active=4, slowest=16.0)


def fresh() -> RecordPatchNet:
    return RecordPatchNet(**CONFIG)


def params(net) -> dict:
    return {k: v.ravel().tolist() for k, v in net.parameters().items()}


def main() -> None:
    rng = np.random.default_rng(11)
    U1, Y1 = rng.normal(size=(3, 6)), rng.normal(size=(3, 4))
    U2, Y2 = rng.normal(size=(3, 6)), rng.normal(size=(3, 4))
    U3 = rng.normal(size=(4, 6))
    net = fresh()
    out = {
        "config": CONFIG,
        "initial": params(net),
        "scale": net._scale.tolist(),
        "projection": net.records.projection.ravel().tolist(),  # (R, D) row-major
        "offset": net.records.offset.tolist(),
        "streams": {"U1": U1.tolist(), "Y1": Y1.tolist(), "U2": U2.tolist(), "Y2": Y2.tolist(), "U3": U3.tolist()},
    }
    o1 = net.observe(U1[None], Y1[None], rate=0.3)
    out["observe1"] = {
        "output": o1.prediction.output[0].tolist(), "loss": o1.prediction.loss, "slow_loss": o1.prediction.slow_loss,
        "delta": {k: v.ravel().tolist() for k, v in o1.delta.items()}, "params": params(net),
        "table": net.records.tables["y"].ravel().tolist(), "mean": net.records.mean.tolist(), "seen": net.records.seen, "writes": o1.writes,
        "state": net.state[0].tolist(),
    }
    o2 = net.observe(U2[None], Y2[None], rate=0.3)
    out["observe2"] = {
        "output": o2.prediction.output[0].tolist(), "loss": o2.prediction.loss, "slow_loss": o2.prediction.slow_loss,
        "params": params(net), "table": net.records.tables["y"].ravel().tolist(), "mean": net.records.mean.tolist(), "seen": net.records.seen,
        "state": net.state[0].tolist(),
    }
    im = net.imagine(U3[None])
    out["imagine"] = {"output": im.output[0].tolist(), "hidden_final": im.final_state[0].tolist(), "read": im.read[0].tolist()}

    net2 = fresh()
    o = net2.observe(U1[None], Y1[None], rate=400.0, backtrack=True)
    out["backtrack"] = {"updated": o.updated, "reason": o.reason, "accepted_rate": o.accepted_rate, "replay_losses": list(o.replay_losses),
                        "replays": o.replay_calls, "final_loss": o.final_loss, "params": params(net2)}

    net3 = fresh()
    o = net3.observe(U1[None], Y1[None], rate=0.3, write=False)
    out["nowrite"] = {"table_abs_sum": float(np.abs(net3.records.tables["y"]).sum()), "params": params(net3), "writes": o.writes}

    net4 = fresh()
    net4.observe(U1[None], Y1[None], rate=0.3)
    net4.observe(U2[None], Y2[None], rate=0.3)
    goal = np.full((4, 4), 0.5)
    weights = np.array([1.0, 1.0, 0.0, 1.0])
    p = plan(net4, U3, goal, weights, controls=[1, 4], lower=-1.0, upper=1.0, feedback=[(0, 3)], rate=0.5, max_steps=6, max_backtracks=8)
    out["plan"] = {"state": net4.state[0].tolist(), "params": params(net4), "table": net4.records.tables["y"].ravel().tolist(), "mean": net4.records.mean.tolist(), "seen": net4.records.seen,
                   "inputs": p.inputs.tolist(), "output": p.output.tolist(), "losses": list(p.losses), "steps": list(p.step_sizes), "replays": p.replays,
                   "reason": p.reason, "converged": p.converged, "residual": p.residual}
    goal_add = np.full((4, 4), -0.3)
    p3 = plan(net4, U3, goal_add, weights, controls=[1, 4], lower=-1.0, upper=1.0, feedback=[(0, 3), (2, 5)], feedback_add=True, rate=2.0, max_steps=6, max_backtracks=8)
    out["plan_add"] = {"inputs": p3.inputs.tolist(), "losses": list(p3.losses), "steps": list(p3.step_sizes), "replays": p3.replays, "reason": p3.reason}
    lo = np.full(6, -1.0); hi = np.full(6, 1.0); lo[1] = U3[0, 1] - 0.2; hi[1] = U3[0, 1] + 0.2; lo[4] = -0.1; hi[4] = 0.1
    p4 = plan(net4, U3, goal, weights, controls=[1, 4], lower=lo, upper=hi, rate=2.0, max_steps=6, max_backtracks=8)
    out["plan_bounds"] = {"lo": lo.tolist(), "hi": hi.tolist(), "inputs": p4.inputs.tolist(), "losses": list(p4.losses), "steps": list(p4.step_sizes), "replays": p4.replays, "reason": p4.reason}
    p5 = plan(net4, U3, goal, weights, controls=[1, 4], lower=-1.0, upper=1.0, feedback=[(0, 3)], feedback_add=True, fixed_read=True, rate=2.0, max_steps=6, max_backtracks=8)
    out["plan_fixed"] = {"inputs": p5.inputs.tolist(), "output": p5.output.tolist(), "losses": list(p5.losses), "steps": list(p5.step_sizes), "replays": p5.replays, "reason": p5.reason}
    p2 = plan(net4, U3, goal, weights, controls=[1, 4], lower=-1.0, upper=1.0, rate=0.5, max_steps=6, max_backtracks=8)
    out["plan_nofeedback"] = {"inputs": p2.inputs.tolist(), "losses": list(p2.losses), "steps": list(p2.step_sizes), "replays": p2.replays, "reason": p2.reason}
    # sleep: after one observation, dream a cue from rest, teach the fixed dream once, re-reference the store once
    net5 = fresh()
    net5.observe(U1[None], Y1[None], rate=0.3)
    dream = net5.dream(U3[None])
    night = net5.sleep([U3[None]], passes=1, rate=0.3, dawn_passes=1)
    out["sleep"] = {"dream": dream[0].tolist(), "params": params(net5), "table": net5.records.tables["y"].ravel().tolist(),
                    "mean": net5.records.mean.tolist(), "seen": net5.records.seen, "report": night}

    target = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else Path(__file__).with_name("fixture.json")
    target.write_text(json.dumps(out))
    print("fixture written:", {k: (v if not isinstance(v, (dict, list)) else "...") for k, v in out["plan"].items() if k in ("losses", "reason", "replays")})


if __name__ == "__main__":
    main()
