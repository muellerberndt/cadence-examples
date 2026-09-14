"""Evaluate an exported browser paddle: points won, lost and drawn, and balls returned.

python evaluate.py --net net_imitation.json --opponent-skill 1 --points 1000
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cadence as cd
import numpy as np
from train import evaluate


class ExportedPolicy:
    """Read the rounded browser weights, using the browser's settlement tolerance."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.net = net = json.loads(path.read_text())
        n = net["n"]
        weights = np.asarray(net["W"]).reshape(n, n)
        pre, post = np.nonzero(weights)
        wiring = cd.Wiring.from_edges(n, pre=pre, post=post, sign=weights[pre, post], sets=net["sets"])
        r = net["rule"]
        rule = cd.GradedRule(dt=r["dt"], slope=r["slope"], threshold=r["threshold"],
                             gain=1.0, clamp_amplitude=r["clamp"], leak=r["leak"])
        if abs(rule.rest_emission - r["rest"]) > 1e-12:
            raise ValueError("exported rest emission does not match the rule")
        self.engine = cd.Settlement(wiring, rule, bias=np.asarray(net["bias"]))
        cfg = net.get("trace")
        self.trace = cd.Afterglow(wiring, source="input", decay=cfg["decay"],
                                 focus=cfg["focus"], amplitude=cfg["amplitude"]) if cfg else None

    def reset(self, batch: int, rows: np.ndarray | None = None) -> None:
        if self.trace is not None:
            self.trace.reset(batch, rows=rows)

    def observation(self, env):
        return env.frames() if self.trace is not None else env.observation()

    def act(self, frames: np.ndarray, greedy: bool = True) -> np.ndarray:
        drive = np.zeros((len(frames), self.net["n"]))
        drive[:, self.net["sets"]["input"]] = frames * self.net["rule"]["clamp"]
        if self.trace is not None:
            drive = self.trace.clamp(drive)
        settled = self.engine.settle_batch(drive, steps=100, tolerance=1e-4)
        if self.trace is not None:
            self.trace.update(settled)
        return settled.activation[:, self.net["sets"]["output"]].argmax(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--net", type=Path, default=Path(__file__).parent / "net.json")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--points", type=int, default=1000)
    parser.add_argument("--opponent-skill", type=float, default=0.7,
                        help="probability the scripted opponent tracks the ball each step (0 to 1)")
    args = parser.parse_args()
    if args.points < 1 or not 0 <= args.opponent_skill <= 1:
        parser.error("--points must be positive and --opponent-skill must be in [0, 1]")
    result = evaluate(ExportedPolicy(args.net), args.seed, args.points, opponent_skill=args.opponent_skill)
    print(json.dumps({"net": str(args.net), "sha256": hashlib.sha256(args.net.read_bytes()).hexdigest(),
                      "seed": args.seed, **result}, indent=2))


if __name__ == "__main__":
    main()
