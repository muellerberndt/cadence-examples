#!/usr/bin/env python3
"""Export the flight sub-net for the browser and the parity cases that bind web/brain.js to cadence.

  web/data/brain.json    the sub-net: CSR by receiving neuron, weights, populations, member indices,
                         the neuron model and the gain from the gate-2 receipt
  tests/parity_cases.json  stimuli and the library's per-step readouts on the same sub-net
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cadence  # noqa: E402
from cadence import Brain, NeuronModel  # noqa: E402
from fruitfly.banc import MANIFEST_PATH  # noqa: E402
from fruitfly.brain import load_fly  # noqa: E402
from fruitfly.subnet import recruit  # noqa: E402


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=60000)
    ap.add_argument("--hops", type=int, default=3)
    ap.add_argument("--min-count", type=float, default=6.0)
    ap.add_argument("--gain", type=float, default=None, help="default: the gain of receipts/g2_reflex_facts.json")
    args = ap.parse_args()
    gain = args.gain
    receipt = ROOT / "receipts" / "g2_reflex_facts.json"
    if gain is None:
        gain = float(json.loads(receipt.read_text())["body"]["connectome"]["gain"])
    fly = load_fly()
    sub = recruit(fly.connectome, budget=args.budget, hops=args.hops, min_count=args.min_count)
    C = sub.connectome
    model = NeuronModel(gain=gain)
    brain = Brain(C, model)
    weights = brain.weights  # gain * count * exp(log_gain[pre]) * sign, in (post, pre) order
    row_ptr = np.zeros(C.n + 1, dtype=np.int32)
    np.cumsum(np.bincount(C.post, minlength=C.n), out=row_ptr[1:])
    payload = {
        "format": "cadence-fruitfly.brain/1",
        "library": {"version": cadence.__version__},
        "manifest": json.loads(MANIFEST_PATH.read_text())["fixture_sha256"],
        "n": int(C.n), "edges": int(C.synapses), "synapses": int(C.count.sum()),
        "whole": {"neurons": int(fly.connectome.n), "edges": int(fly.connectome.synapses)},
        "recruitment": {"budget": args.budget, "hops": args.hops, "min_count": args.min_count, "hop_counts": np.bincount(sub.hops).tolist()},
        "model": model.to_dict() | {"gain": gain},
        "populations": {k: list(v) for k, v in C.populations.items()},
        "arrays": {"row_ptr": b64(row_ptr), "pre": b64(C.pre.astype(np.int32)), "weight": b64(weights.astype(np.float64)), "members": b64(sub.members.astype(np.int32)),
                   "count": b64(np.minimum(C.count, 65535).astype(np.uint16)), "sign": b64(C.sign.astype(np.int8))},
    }
    (ROOT / "web" / "data").mkdir(parents=True, exist_ok=True)
    out = ROOT / "web" / "data" / "brain.json"
    out.write_text(json.dumps(payload, separators=(",", ":")))
    # parity: the library's per-step readouts under a few stimuli
    readouts = ["mn:wing:b1:left", "mn:wing:b1:right", "steering:left", "steering:right", "power:left", "power:right", "neck_mn", "dn", "mn:haltere:left", "gf", "dn:landing", "dn:DNa02:left", "dn:DNa02:right", "dn:DNp09", "mn9", "kc", "mbon"]
    readouts = [r for r in readouts if r in C.populations]
    cases = []
    for name, stimulus in (("haltere_both_half", {"haltere:left": 0.5, "haltere:right": 0.5}), ("ocelli", {"ocelli": 1.0}), ("wing_right", {"wing_sense:right": 1.0}), ("loom", {"vis:LC4": 1.0, "vis:LPLC2": 1.0}), ("fruit_left", {"orn:decaying_fruit:left": 1.0, "orn:decaying_fruit:right": 0.3}), ("sugar", {"grn:sugar:labellum": 1.0})):
        drive = np.zeros(C.n)
        for pop, level in stimulus.items():
            drive[list(C.populations[pop])] = np.maximum(drive[list(C.populations[pop])], model.stimulus_amplitude * level)
        state = brain.settle_batch(drive[None, :], steps=40, trajectory=True)
        per_step = [[float(state.trajectory[t, 0, list(C.populations[r])].mean()) for r in readouts] for t in range(state.trajectory.shape[0])]
        cases.append({"name": name, "stimulus": stimulus, "readouts": readouts, "per_step_means": per_step, "final_active": int((state.activation[0] >= 0.5).sum())})
    (ROOT / "tests").mkdir(exist_ok=True)
    (ROOT / "tests" / "parity_cases.json").write_text(json.dumps({"gain": gain, "cases": cases}, indent=1))
    print(f"sub-net {C.n} neurons, {C.synapses} synapse classes, gain {gain}; payload {out.stat().st_size/1e6:.1f} MB; parity cases {len(cases)}")


if __name__ == "__main__":
    main()
