#!/usr/bin/env python3
"""The Kenyon cell code under the two odours of the arena, across gain and odour level.

A real fly's Kenyon cells are sparse (about 5 percent of them answer an odour; Turner, Bazhenov
and Laurent 2008; Honegger, Campbell and Turner 2011) and odour-specific. This tool measures,
on the learning sub-net, how many Kenyon cells answer each odour alone and both, and where the
four mushroom body output neurons of the lessons sit, for a grid of gains and receptor levels,
and writes the receipt the level of the lessons is selected on: the smallest level at which the
code is sparse and specific and the output neurons are neither silent nor saturated.

Changes against the first version of this scan:
- the brain is the lesson's (fruitfly/lessons.py: class gains, naive seam, calibrated readouts), so the
  "movable" column reads the output neurons the lessons actually start from;
- "specific" needs both codes non-empty; before, a level where fruit recruited nothing and yeast
  recruited 2,352 cells was flagged SPECIFIC (odours told apart by strength, not identity);
- the codes are compared by cosine over the activations, not only by the fraction active: equal
  projection-neuron fractions (0.48/0.48) do not by themselves mean the same neurons;
- --ln scans the antennal lobe local neurons' log gain, the knob that sets the lobe's ignition
  (fruitfly.brain.CLASS_LOG_GAIN; tools/class_gains.py selects it on the whole brain).

    python tools/odour_code.py                       # the dictionary's local-neuron gain
    python tools/odour_code.py --ln 0 -1.5 -3 -4.6   # 0 is the old, igniting lobe
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cadence import Brain, NeuronModel  # noqa: E402
from cadence.learning import LearnerConfig  # noqa: E402
from fruitfly.brain import CLASS_LOG_GAIN, load_fly  # noqa: E402
import fruitfly.brain as fly_brain  # noqa: E402
from fruitfly.lessons import setup_lessons  # noqa: E402
from fruitfly.subnet import recruit  # noqa: E402
from tools.learn_odour import OUTPUTS, SEEDS  # noqa: E402

MAX_SHARED = 0.25  # the library's specific_max: the two codes share at most this fraction of their union

GAINS = (0.03, 0.02, 0.015, 0.01)
LEVELS = (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0)
ACTIVE = 0.05
SPARSE_BAND = (0.02, 0.15)  # the fraction of Kenyon cells an odour may recruit
OUTPUT_BAND = (0.05, 0.9)  # where an output neuron can still be moved


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ln", type=float, nargs="*", default=[CLASS_LOG_GAIN.get("ln", 0.0)], help="log gains of the antennal lobe local neurons to scan")
    ap.add_argument("--gains", type=float, nargs="*", default=list(GAINS)); ap.add_argument("--levels", type=float, nargs="*", default=list(LEVELS))
    ap.add_argument("--backend", default="torch"); ap.add_argument("--no-calibrate", action="store_true", help="the raw brain, as the first version of the scan")
    args = ap.parse_args()
    fly = load_fly()
    sub = recruit(fly.connectome, budget=12000, hops=2, min_count=5.0, seeds=SEEDS)
    C = sub.connectome; P = C.populations; cfg = LearnerConfig()
    kc = np.array(P["kc"]); pn = np.array(P.get("pn", ()), dtype=int); outputs = [int(P[n][0]) for n in OUTPUTS]
    sets = [[list(P[f"orn:{o}:{s}"]) for s in ("left", "right")] for o in ("decaying_fruit", "yeasty")]
    rows = []
    for ln in args.ln:
        fly_brain.CLASS_LOG_GAIN["ln"] = ln  # setup_lessons reads the dictionary through log_gain_for
        for gain in args.gains:
            setup = setup_lessons(C, backend=args.backend, gain=gain, calibrate=not args.no_calibrate)
            brain = setup.brain(C, backend=args.backend, gain=gain)
            for level in args.levels:
                act = []
                for k in range(2):
                    d = np.zeros((1, C.n))
                    for idx in sets[k]:
                        d[0, idx] = brain.neuron_model.stimulus_amplitude * level
                    act.append(np.asarray(brain.settle_batch(d, steps=cfg.free_steps, tolerance=cfg.tolerance).activation[0], float))
                on = [a[kc] > ACTIVE for a in act]
                fruit_only, yeast_only, both = int((on[0] & ~on[1]).sum()), int((on[1] & ~on[0]).sum()), int((on[0] & on[1]).sum())
                frac = [float(o.mean()) for o in on]
                outs = [[float(a[i]) for i in outputs] for a in act]
                pn_frac = [float((a[pn] > ACTIVE).mean()) for a in act] if len(pn) else [float("nan")] * 2
                kc_cos = cosine(act[0][kc], act[1][kc]); pn_cos = cosine(act[0][pn], act[1][pn]) if len(pn) else float("nan")
                union = fruit_only + yeast_only + both
                sparse = all(SPARSE_BAND[0] <= f <= SPARSE_BAND[1] for f in frac)
                specific = min(frac) > 0 and both <= MAX_SHARED * union  # both odours must recruit a code of their own
                movable = all(OUTPUT_BAND[0] <= x <= OUTPUT_BAND[1] for o in outs for x in o)
                rows.append({"ln_log_gain": ln, "gain": gain, "level": level, "kc_fraction": frac, "fruit_only": fruit_only, "yeast_only": yeast_only, "both": both,
                             "kc_cosine": kc_cos, "pn_fraction": pn_frac, "pn_cosine": pn_cos, "outputs_fruit": outs[0], "outputs_yeast": outs[1], "sparse": sparse, "specific": specific, "movable": movable})
                print(f"ln {ln:<5} gain {gain:<6} level {level:<5} KC {frac[0]:.3f}/{frac[1]:.3f} fruit-only {fruit_only:5d} yeast-only {yeast_only:5d} both {both:5d} cos {kc_cos:.2f} | PN {pn_frac[0]:.2f}/{pn_frac[1]:.2f} cos {pn_cos:.2f} | outputs fruit {[round(x, 2) for x in outs[0]]} yeast {[round(x, 2) for x in outs[1]]} {'SPARSE' if sparse else ''} {'SPECIFIC' if specific else ''} {'MOVABLE' if movable else ''}", flush=True)
    chosen = [r for r in rows if r["sparse"] and r["specific"] and r["movable"]]
    out = {"tool": "tools/odour_code.py", "subnet": {"neurons": int(C.n), "classes": int(C.synapses), "seeds": list(SEEDS), "budget": 12000, "hops": 2, "min_count": 5.0}, "active_level": ACTIVE, "sparse_band": SPARSE_BAND, "output_band": OUTPUT_BAND, "max_shared": MAX_SHARED,
           "calibrated": not args.no_calibrate, "rows": rows, "passing": chosen,
           "sources": ["Turner, Bazhenov and Laurent 2008 J Neurophysiol 99:734 (about 5 percent of Kenyon cells answer an odour)", "Honegger, Campbell and Turner 2011 J Neurosci 31:11772 (sparse, odour-specific Kenyon cell responses)"]}
    (ROOT / "receipts" / "odour_code.json").write_text(json.dumps(out, indent=1))
    print("passing:", [(r["ln_log_gain"], r["gain"], r["level"]) for r in chosen])


if __name__ == "__main__":
    main()
