#!/usr/bin/env python3
"""The Kenyon cell code under the two odours of the arena, across gain and odour level.

A real fly's Kenyon cells are sparse (about 5 percent of them answer an odour; Turner, Bazhenov
and Laurent 2008; Honegger, Campbell and Turner 2011) and odour-specific. This tool measures,
on the learning sub-net, how many Kenyon cells answer each odour alone and both, and where the
four mushroom body output neurons of the lessons sit, for a grid of gains and receptor levels,
and writes the receipt the level of the lessons is selected on: the smallest level at which the
code is sparse and specific and the output neurons are neither silent nor saturated.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cadence import Brain, NeuronModel  # noqa: E402
from cadence.learning import LearnerConfig  # noqa: E402
from fruitfly.brain import load_fly  # noqa: E402
from fruitfly.subnet import recruit  # noqa: E402
from tools.learn_odour import OUTPUTS, SEEDS  # noqa: E402

GAINS = (0.03, 0.02, 0.015, 0.01)
LEVELS = (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0)
ACTIVE = 0.05
SPARSE_BAND = (0.02, 0.15)  # the fraction of Kenyon cells an odour may recruit
OUTPUT_BAND = (0.05, 0.9)  # where an output neuron can still be moved


def main() -> None:
    fly = load_fly()
    sub = recruit(fly.connectome, budget=12000, hops=2, min_count=5.0, seeds=SEEDS)
    C = sub.connectome; P = C.populations; cfg = LearnerConfig()
    kc = np.array(P["kc"]); outputs = [int(P[n][0]) for n in OUTPUTS]
    sets = [[list(P[f"orn:{o}:{s}"]) for s in ("left", "right")] for o in ("decaying_fruit", "yeasty")]
    rows = []
    for gain in GAINS:
        brain = Brain(C, NeuronModel(gain=gain), backend="torch")
        for level in LEVELS:
            act = []
            for k in range(2):
                d = np.zeros((1, C.n))
                for idx in sets[k]:
                    d[0, idx] = brain.neuron_model.stimulus_amplitude * level
                act.append(brain.settle_batch(d, steps=cfg.free_steps, tolerance=cfg.tolerance).activation[0])
            on = [a[kc] > ACTIVE for a in act]
            fruit_only, yeast_only, both = int((on[0] & ~on[1]).sum()), int((on[1] & ~on[0]).sum()), int((on[0] & on[1]).sum())
            frac = [float(o.mean()) for o in on]
            outs = [[float(a[i]) for i in outputs] for a in act]
            pn = [float((a[list(P["pn"])] > ACTIVE).mean()) for a in act] if "pn" in P else [None, None]
            sparse = all(SPARSE_BAND[0] <= f <= SPARSE_BAND[1] for f in frac)
            specific = both < 0.5 * max(1, fruit_only + yeast_only + both)
            movable = all(OUTPUT_BAND[0] <= x <= OUTPUT_BAND[1] for o in outs for x in o)
            rows.append({"gain": gain, "level": level, "kc_fraction": frac, "fruit_only": fruit_only, "yeast_only": yeast_only, "both": both, "pn_fraction": pn, "outputs_fruit": outs[0], "outputs_yeast": outs[1], "sparse": sparse, "specific": specific, "movable": movable})
            print(f"gain {gain:<6} level {level:<5} KC fraction {frac[0]:.3f}/{frac[1]:.3f} fruit-only {fruit_only:5d} yeast-only {yeast_only:5d} both {both:5d} | PN {pn[0]:.2f}/{pn[1]:.2f} | outputs fruit {[round(x, 2) for x in outs[0]]} yeast {[round(x, 2) for x in outs[1]]} {'SPARSE' if sparse else ''} {'SPECIFIC' if specific else ''} {'MOVABLE' if movable else ''}", flush=True)
    chosen = [r for r in rows if r["sparse"] and r["specific"] and r["movable"]]
    out = {"tool": "tools/odour_code.py", "subnet": {"neurons": int(C.n), "classes": int(C.synapses), "seeds": list(SEEDS), "budget": 12000, "hops": 2, "min_count": 5.0}, "active_level": ACTIVE, "sparse_band": SPARSE_BAND, "output_band": OUTPUT_BAND, "rows": rows, "passing": chosen,
           "sources": ["Turner, Bazhenov and Laurent 2008 J Neurophysiol 99:734 (about 5 percent of Kenyon cells answer an odour)", "Honegger, Campbell and Turner 2011 J Neurosci 31:11772 (sparse, odour-specific Kenyon cell responses)"]}
    (ROOT / "receipts" / "odour_code.json").write_text(json.dumps(out, indent=1))
    print("passing:", [(r["gain"], r["level"]) for r in chosen])


if __name__ == "__main__":
    main()
