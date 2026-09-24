#!/usr/bin/env python3
"""Map a learned checkpoint (tools/learn_odour.py) onto the page's sub-net: web/data/<name>.json,
and write the lessons' constants for the page: web/data/lessons.json.

The learning ran on the olfactory sub-net; the page settles a larger one. Every synapse whose
efficacy moved is looked up by its whole-brain (pre, post) pair in the page's synapse order. The
constants (outputs, temperature, cap, eta, the tonic drive) come from the run's JSON beside the
checkpoint, so the page runs the receipted configuration.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from fruitfly.brain import load_fly  # noqa: E402
from fruitfly.subnet import recruit  # noqa: E402
from learn_odour import SEEDS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint"); ap.add_argument("--name", default="learned")
    ap.add_argument("--budget", type=int, default=12000); ap.add_argument("--hops", type=int, default=2); ap.add_argument("--min-count", type=float, default=5.0)
    ap.add_argument("--page-budget", type=int, default=60000); ap.add_argument("--page-hops", type=int, default=3); ap.add_argument("--page-min-count", type=float, default=6.0)
    args = ap.parse_args()
    fly = load_fly()
    learn = recruit(fly.connectome, budget=args.budget, hops=args.hops, min_count=args.min_count, seeds=SEEDS)
    page = recruit(fly.connectome, budget=args.page_budget, hops=args.page_hops, min_count=args.page_min_count)
    z = np.load(args.checkpoint)
    eff = z["efficacy"]
    if not np.array_equal(z["members"], learn.members):
        raise SystemExit("the checkpoint's members differ from the recruited learning sub-net")
    L = learn.connectome
    moved = np.flatnonzero(np.abs(eff - L.sign) > 1e-9)
    pairs = {(int(page.members[p]), int(page.members[q])): k for k, (p, q) in enumerate(zip(page.connectome.pre, page.connectome.post))}
    edges, values, missing = [], [], 0
    for e in moved:
        key = (int(learn.members[L.pre[e]]), int(learn.members[L.post[e]]))
        k = pairs.get(key)
        if k is None:
            missing += 1; continue
        edges.append(k); values.append(float(eff[e]))
    out = ROOT / "web" / "data" / f"{args.name}.json"
    out.write_text(json.dumps({"format": "cadence-fruitfly.learned/1", "checkpoint": Path(args.checkpoint).name, "moved": int(len(moved)), "mapped": len(edges), "missing": missing, "edges": edges, "efficacy": values}, separators=(",", ":")))
    print(f"{len(moved)} synapses moved, {len(edges)} mapped onto the page's sub-net, {missing} outside it -> {out} ({out.stat().st_size/1e3:.0f} kB)")
    run_json = Path(args.checkpoint).with_suffix(".json")
    if run_json.exists():
        r = json.loads(run_json.read_text()); a = r["args"]; sm = r["summary"]
        config = {"outputs": sm["outputs"], "actions": sm.get("output_action", [0, 1]), "plastic": a["plastic"], "critic": "kc", "beta": a["beta"], "temperature": a["temperature"], "nudgedSteps": 10, "tolerance": a["tolerance"],
                  "gamma": a["gamma"], "lam": a["lam"], "eta": a["eta"], "etaCritic": a["eta_critic"], "cap": a["scale_cap"], "dopamineCap": 1.0, "tonic": ({sm["outputs"][1]: sm["tonic_avoid"]} if sm.get("tonic_avoid") else {})}
        lessons = ROOT / "web" / "data" / "lessons.json"
        lessons.write_text(json.dumps({"format": "cadence-fruitfly.lessons/1", "run": run_json.name, "gain": a["gain"], "config": config, "held_out": r["held_out"].get("none"), "probe": r.get("probe"), "naive_probe": r.get("naive_probe")}, indent=1))
        print(f"lessons for the page -> {lessons}: {config}")


if __name__ == "__main__":
    main()
