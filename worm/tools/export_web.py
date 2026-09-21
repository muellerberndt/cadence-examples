#!/usr/bin/env python3
"""Export what the browser needs and the parity cases that bind it to cadence.

  web/data/brain.json       the newborn brain: connectome weights after the
                            prenatal reflex lessons, as masked entries
  web/data/connectome.json  neurons, positions, transmitters, synapses
  web/data/params.json      the world and learning constants
  tests/parity_cases.json   paths, targets and the library's own results
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cadence  # noqa: E402
from worm.brain import WormBrain  # noqa: E402


def entries(net):
    m, P = net.masks, net.parameters()
    r, c = np.nonzero(m["A"])
    B = [[int(n), int(k), float(P["B"][n, k])] for n, k in zip(*np.nonzero(m["B"]))]
    C = [[int(o), int(n), float(P["C"][o, n])] for o, n in zip(*np.nonzero(m["C"]))]
    return {"rows": r.tolist(), "cols": c.tolist(), "vals": P["A"][r, c].tolist()}, B, C


def spec(brain):
    net = brain.net
    A, B, C = entries(net)
    return {"H": brain.H, "I": len(brain.inputs), "O": len(brain.outputs), "names": brain.names,
            "inputs": brain.inputs, "outputs": brain.outputs, "A": A, "B": B, "C": C,
            "tolerance": net.tolerance, "max_iterations": net.max_iterations,
            "max_backtracks": net.max_backtracks, "max_damping_trials": net.max_damping_trials,
            "cadence": cadence.__version__}


def main() -> None:
    web = ROOT / "web" / "data"
    web.mkdir(parents=True, exist_ok=True)
    brain = WormBrain(seed=0)
    prenatal = brain.born()
    s = spec(brain)
    s["prenatal"] = {"lessons": len(prenatal), "admitted": int(sum(prenatal)), "growth": brain.growth()}
    (web / "brain.json").write_text(json.dumps(s, separators=(",", ":")))
    for name in ("connectome.json", "params.json"):
        shutil.copy(ROOT / "data" / name, web / name)

    # Parity: the library's own answers on paths the worm could live through.
    rng = np.random.default_rng(3)
    cases = []
    for T, warm in ((5, False), (8, True), (16, True)):
        b = WormBrain(seed=0)
        b.born()
        warmup = rng.uniform(0, 1, (6, b.p and len(b.inputs))) * (rng.uniform(0, 1, (6, 4)) < 0.4)
        if warm:
            b.net.advance(warmup[None])
        u = rng.uniform(0, 1, (T, 4)) * (rng.uniform(0, 1, (T, 4)) < 0.5)
        y = b.net.imagine(u[None]).output[0].copy()
        y[-4:, 1] = 0.8
        y[-4:, 0] = 0.0
        before = b.net.parameters()
        r = b.net.observe(u[None], y[None], beta=b.p["beta"], rate=b.p["rate"], backtrack=True)
        after = b.net.parameters()
        A, _, _ = entries(b.net)
        m = b.net.masks["A"]
        cases.append({
            "T": T, "warmup": warmup.tolist() if warm else [], "inputs": u.tolist(), "target": y.tolist(),
            "free_output": r.free.output[0].tolist(),
            "plus_hidden": r.plus.hidden[0].tolist(), "minus_hidden": r.minus.hidden[0].tolist(),
            "updated": bool(r.updated), "step": float(r.accepted_rate) if r.updated else 0.0,
            "delta_A": r.delta["A"][m].tolist(),
            "after_A": after["A"][m].tolist(), "before_A": before["A"][m].tolist(),
            "growth_after": b.growth(),
        })
    (ROOT / "tests").mkdir(exist_ok=True)
    (ROOT / "tests" / "parity_cases.json").write_text(json.dumps(cases))

    # The body: scripted crawls, reversals, omega turns and wall turns; web/life.js must lay the same track.
    from worm.rng import Rng
    from worm.world import World
    bodies = []
    for seed, script in (
        (1, [["step", 200], ["reverse", 2.0, -2.4], ["step", 160], ["steer", 0.5], ["step", 120], ["reverse", 1.2, 1.7], ["step", 90]]),
        (2, [["move", 0.5, 0.6], ["step", 400], ["reverse", 3.0, -3.0], ["step", 300]]),
        (3, [["move", 11.6, 7.7], ["step", 250], ["reverse", 0.6, 2.9], ["step", 8], ["reverse", 0.6, -2.9], ["step", 200]]),
    ):
        w = World(brain.p, Rng(seed * 7919 + 17))
        for what, *args in script:
            if what == "step":
                for _ in range(args[0]):
                    w.physics(brain.p["physics_step"], False)
            elif what == "reverse":
                w.reverse(*args)
            elif what == "steer":
                w.body.heading += args[0]
            elif what == "move":
                dx, dy = args[0] - w.body.x, args[1] - w.body.y
                w.body.trail = [(x + dx, y + dy) for x, y in w.body.trail]
                w.body.x, w.body.y = args
        bodies.append({"seed": seed, "script": script, "trail": [list(q) for q in w.body.trail]})
    (ROOT / "tests" / "body_cases.json").write_text(json.dumps(bodies))
    print(f"brain.json: {len(s['A']['vals'])} synapses, prenatal {s['prenatal']}; {len(cases)} parity cases")


if __name__ == "__main__":
    main()
