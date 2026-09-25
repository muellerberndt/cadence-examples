#!/usr/bin/env python3
"""The T-maze reversal, both ways, on the lesson's setup: the receipt that the discrimination is learnable.

One decision per trial at the fruit (the page's protocol), the library's actor-critic on the Kenyon-cell-
to-MBON seam, on the olfactory sub-net with the dictionary's class gains, the seam started naive and the
two readouts calibrated (fruitfly/lessons.py). Sugar at the first odour, then blows there while the sugar
sits at the second, then the second is found; then blows at the first again. A direction is reversed when
the fly ends up approaching the second odour and avoiding the first. Both directions are run; a shuffled
wiring runs the same.

    python tools/tmaze.py            # writes receipts/g4_tmaze_reversal.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cadence  # noqa: E402
from cadence import Learner, LearnerConfig  # noqa: E402
from cadence.plasticity import ActorCritic, ActorCriticConfig  # noqa: E402
from cadence.protocol import shuffled  # noqa: E402
from fruitfly.banc import MANIFEST_PATH  # noqa: E402
from fruitfly.brain import CLASS_LOG_GAIN, load_fly  # noqa: E402
from fruitfly.lessons import OUTPUTS, decision_drives, setup_lessons  # noqa: E402
from fruitfly.subnet import recruit  # noqa: E402
from tools.learn_odour import SEEDS  # noqa: E402

NAMES = ("fruit", "yeast")
PHASES = (("sugar", 0, 4), ("blows", 0, 12), ("sugar", 1, 16), ("blows", 0, 6))  # (outcome, which odour: 0 first / 1 second, trials)


def run(C, setup, *, eta: float, temperature: float, seed: int, backend: str) -> dict:
    cfg = LearnerConfig(beta=0.1, temperature=temperature, tolerance=1e-3, free_steps=100, nudged_steps=10, scale_cap=3.0)
    D = decision_drives(C, 3.0); kc = list(C.populations["kc"]); out = {}
    for first in (0, 1):
        odours = (first, 1 - first)
        brain = setup.brain(C, backend=backend)
        learner = Learner(brain, list(setup.outputs), cfg, plastic_synapses=setup.plastic, plastic_neurons=np.zeros(C.n, bool), reciprocal=False)
        ac = ActorCritic(learner, kc, ActorCriticConfig(gamma=0.95, lam=0.9, eta=eta, eta_bias=0.0, eta_critic=0.05, dopamine_cap=1.0), seed=seed)
        probe = lambda: [float(ac.probabilities(ac.settle(D[k:k + 1]))[0][0]) for k in range(2)]
        t0 = time.time(); trace = [{"phase": "naive", "p_approach": probe()}]
        for outcome, which, trials in PHASES:
            k = odours[which]; landed = 0
            for _ in range(trials):
                act = int(ac.act(D[k:k + 1])[0]); landed += act == 0
                r = (1.0 if outcome == "sugar" else -1.0) if act == 0 else 0.0
                ac.learn(np.array([r]), np.array([True]), D[k:k + 1])
            trace.append({"phase": f"{outcome} at {NAMES[k]} x{trials}", "landed": landed, "p_approach": probe()})
        p = trace[-1]["p_approach"]; a, b = p[first], p[1 - first]
        out[f"{NAMES[first]} first"] = {"trace": trace, "final": {"first": a, "second": b}, "reversed": bool(b > 0.5 and a < 0.5), "seconds": round(time.time() - t0, 1)}
        print(f"  {NAMES[first]} first: " + " | ".join(f"{t['phase']} -> {t['p_approach'][0]:.2f}/{t['p_approach'][1]:.2f}" + (f" ({t['landed']} landed)" if "landed" in t else "") for t in trace) + f" | {'REVERSED' if out[f'{NAMES[first]} first']['reversed'] else 'not reversed'}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--eta", type=float, default=1.0); ap.add_argument("--temperature", type=float, default=0.3); ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--backend", default="torch"); ap.add_argument("--no-shuffled", action="store_true"); args = ap.parse_args()
    fly = load_fly(); sub = recruit(fly.connectome, budget=12000, hops=2, min_count=5.0, seeds=SEEDS); C = sub.connectome
    receipt = {"tool": "tools/tmaze.py", "library": cadence.__version__, "fixture": json.loads(MANIFEST_PATH.read_text())["fixture_sha256"], "class_log_gain": CLASS_LOG_GAIN, "outputs": list(OUTPUTS),
               "subnet": {"neurons": int(C.n), "classes": int(C.synapses), "budget": 12000, "hops": 2, "min_count": 5.0, "seeds": list(SEEDS)},
               "constants": {"eta": args.eta, "eta_critic": 0.05, "gamma": 0.95, "lam": 0.9, "temperature": args.temperature, "beta": 0.1, "nudged_steps": 10, "scale_cap": 3.0, "seed": args.seed},
               "phases": [{"outcome": o, "odour": "first" if w == 0 else "second", "trials": n} for o, w, n in PHASES], "wirings": {}}
    for name, connectome in [("measured", C)] + ([] if args.no_shuffled else [("shuffled:1", shuffled(C, 1))]):
        t0 = time.time(); setup = setup_lessons(connectome, backend=args.backend)
        print(f"{name}: setup in {time.time() - t0:.0f} s; seam {setup.report['seam']['classes_onto_outputs']}; readout bias {setup.report['readout_bias']}", flush=True)
        receipt["wirings"][name] = {"setup": setup.report, "directions": run(connectome, setup, eta=args.eta, temperature=args.temperature, seed=args.seed, backend=args.backend)}
    receipt["both_directions_reversed"] = all(d["reversed"] for d in receipt["wirings"]["measured"]["directions"].values())
    (ROOT / "receipts" / "g4_tmaze_reversal.json").write_text(json.dumps(receipt, indent=1))
    print("both directions reversed on the measured wiring:", receipt["both_directions_reversed"], "; written receipts/g4_tmaze_reversal.json")


if __name__ == "__main__":
    main()
