#!/usr/bin/env python3
"""How far the flight sub-net's settled state is from the whole brain's under the flight stimuli."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cadence import Brain, NeuronModel  # noqa: E402
from cadence.receipts import Receipt  # noqa: E402
from fruitfly.banc import MANIFEST_PATH  # noqa: E402
from fruitfly.brain import load_fly, log_gain_for  # noqa: E402
from fruitfly.subnet import recruit  # noqa: E402

STIMULI = {"haltere_both": {"haltere:left": 0.5, "haltere:right": 0.5}, "wing_left": {"wing_sense:left": 1.0}, "ocelli": {"ocelli": 1.0}, "vs_left": {"lptc:vs:left": 1.0}, "hs_left": {"lptc:hs:left": 1.0},
           "jo_e": {"jo:E:left": 0.7, "jo:E:right": 0.7}, "loom": {"vis:LC4": 1.0, "vis:LPLC2": 1.0}, "loom_lplc2": {"vis:LPLC2": 1.0},
           "fruit_left": {"orn:decaying_fruit:left": 1.0, "orn:decaying_fruit:right": 0.3}, "yeast_right": {"orn:yeasty:right": 1.0, "orn:yeasty:left": 0.3},
           "sugar": {"grn:sugar:labellum": 1.0, "grn:sugar:front_leg": 1.0}, "touch": {"leg_touch": 0.5}}
READOUTS = ["mn:wing:b1:left", "steering:left", "steering:right", "power:left", "power:right", "neck_mn", "dn", "gf", "dn:landing", "dn:DNa02:left", "dn:DNa02:right", "dn:DNp09", "mn9", "mbon", "kc"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=60000); ap.add_argument("--hops", type=int, default=3); ap.add_argument("--min-count", type=float, default=6.0)
    args = ap.parse_args()
    gain = float(json.loads((ROOT / "receipts" / "g2_reflex_facts.json").read_text())["body"]["connectome"]["gain"])
    fly = load_fly(); W = fly.connectome
    sub = recruit(W, budget=args.budget, hops=args.hops, min_count=args.min_count); S = sub.connectome
    model = NeuronModel(gain=gain); whole, part = Brain(W, model, log_gain=log_gain_for(W)), Brain(S, model, log_gain=log_gain_for(S))
    rows = []; worst_read = 0.0; worst_member = 0.0; worst_outsider = 0.0
    for name, stim in STIMULI.items():
        dw = np.zeros(W.n); ds = np.zeros(S.n)
        for pop, level in stim.items():
            dw[list(W.populations[pop])] = np.maximum(dw[list(W.populations[pop])], model.stimulus_amplitude * level)
            ds[list(S.populations[pop])] = np.maximum(ds[list(S.populations[pop])], model.stimulus_amplitude * level)
        sw = whole.settle_batch(dw[None, :], steps=60).activation[0]; ss = part.settle_batch(ds[None, :], steps=60).activation[0]
        member = float(np.abs(ss - sw[sub.members]).max())
        outside = np.ones(W.n, bool); outside[sub.members] = False
        outsider = float(sw[outside].max()) if outside.any() else 0.0
        reads = {}
        for r in READOUTS:
            if r in S.populations:
                a = float(sw[list(W.populations[r])].mean()); b = float(ss[list(S.populations[r])].mean()); reads[r] = {"whole": a, "subnet": b}; worst_read = max(worst_read, abs(a - b))
        worst_member = max(worst_member, member); worst_outsider = max(worst_outsider, outsider)
        rows.append({"stimulus": name, "member_deviation": member, "outsider_peak": outsider, "active_whole": int((sw >= 0.5).sum()), "active_subnet": int((ss >= 0.5).sum()), "readouts": reads})
        print(f"{name}: member dev {member:.2e}, outsider peak {outsider:.3f}, active whole {int((sw>=0.5).sum())} sub {int((ss>=0.5).sum())}")
    body = {"gain": gain, "recruitment": {"budget": args.budget, "hops": args.hops, "min_count": args.min_count, "n": int(S.n), "edges": int(S.synapses)}, "rows": rows,
            "worst": {"readout": worst_read, "member": worst_member, "outsider_peak": worst_outsider}, "tolerance": {"readout": 1e-3, "member": 1e-2}}
    body["passed"] = bool(worst_read <= 1e-3)  # the page reads populations; the member-level deviation is reported per stimulus
    body["member_passed"] = bool(worst_member <= 1e-2)
    rec = Receipt.build("cadence-fruitfly.subnet-closure/1", body, sources=[("fruitfly/subnet.py", ROOT / "fruitfly" / "subnet.py"), ("fruitfly/brain.py", ROOT / "fruitfly" / "brain.py"), ("fruitfly/fixtures/banc_888_manifest.json", MANIFEST_PATH), ("tools/subnet_closure.py", Path(__file__))])
    out = rec.write(ROOT / "receipts" / "subnet_closure.json")
    print("closure", "PASS" if body["passed"] else "FAIL", json.dumps(body["worst"]), out.name, rec.digest[:16])


if __name__ == "__main__":
    main()
