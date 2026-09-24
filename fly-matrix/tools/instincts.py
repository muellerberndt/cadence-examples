#!/usr/bin/env python3
"""Gate 2, the reflex facts: connectome versus shuffled under one gain each, with a receipt."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cadence import Brain, NeuronModel  # noqa: E402
from cadence.protocol import select_gain, shuffled  # noqa: E402
from cadence.receipts import Receipt  # noqa: E402
from fruitfly.banc import MANIFEST_PATH  # noqa: E402
from fruitfly.brain import load_fly  # noqa: E402
from fruitfly.instincts import SOURCES, instinct_protocol  # noqa: E402

GRID = (0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025, 0.03, 0.04)


def run(connectome, protocol, label):
    t = time.time()
    gain, table = select_gain(lambda g: Brain(connectome, NeuronModel(gain=g)), protocol, GRID, sparsity_cap=0.05)
    score = protocol.score(Brain(connectome, NeuronModel(gain=gain)))
    print(f"{label}: gain {gain}, training {score['training_passed']}/{len(score['training'])}, rows {score['passed']}/{score['total']} in {time.time()-t:.1f} s")
    for r in score["rows"]:
        print(f"   {r['id']} {'pass' if r['passed'] else 'FAIL'} {r['stimulus']} -> {r['readout']} {r['predicate']} mean {r['reading']['mean']:.3f} ref {r['reference']['mean']:.3f}")
    return {"gain": gain, "gain_table": table, "score": score}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "receipts" / "g3_instinct_facts.json"))
    args = ap.parse_args()
    fly = load_fly()
    protocol = instinct_protocol()
    body = {"protocol": protocol.to_dict(), "sources": SOURCES, "gain_grid": GRID, "neurons": fly.connectome.n, "synapse_classes": fly.connectome.synapses,
            "connectome": run(fly.connectome, protocol, "connectome"), "shuffled": []}
    for seed in range(args.seeds):
        body["shuffled"].append({"seed": seed, **run(shuffled(fly.connectome, seed), protocol, f"shuffled seed {seed}")})
    body["summary"] = {
        "connectome_rows": body["connectome"]["score"]["passed"],
        "shuffled_rows": [s["score"]["passed"] for s in body["shuffled"]],
        "total_rows": body["connectome"]["score"]["total"],
    }
    receipt = Receipt.build("cadence-fruitfly.g3-instinct-facts/1", body, sources=[
        ("fruitfly/banc.py", ROOT / "fruitfly" / "banc.py"), ("fruitfly/brain.py", ROOT / "fruitfly" / "brain.py"),
        ("fruitfly/instincts.py", ROOT / "fruitfly" / "instincts.py"), ("tools/instincts.py", Path(__file__)),
        ("fruitfly/fixtures/banc_888_manifest.json", MANIFEST_PATH)])
    out = receipt.write(Path(args.out))
    print("receipt", out, receipt.digest[:16], json.dumps(body["summary"]))


if __name__ == "__main__":
    main()
