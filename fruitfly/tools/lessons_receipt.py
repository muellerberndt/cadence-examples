#!/usr/bin/env python3
"""Gate 4's receipt: the campaign's runs (runs/learn_odour/*.json) as one table and one JSON.

    python tools/lessons_receipt.py            # writes receipts/g4_lessons.json, prints the table

Rows: every run by kind and seed with its naive food rate, the food rate of the last training
window, the held-out food rate (drawn and greedy policies on fresh arenas), the probe (the
approach probability under each odour alone, before and after), the lesions of the connectome
runs, the direction of the synaptic change per Kenyon cell class and target, and the reversal.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "learn_odour"


def f(x, nd=2):
    return "" if x is None or x != x else f"{x:.{nd}f}"


def main() -> None:
    runs = sorted(RUNS.glob("*.json"))
    if not runs:
        sys.exit("no runs")
    rows, receipts = [], {}
    for p in runs:
        r = json.loads(p.read_text()); sm = r["summary"]; kind = sm["kind"]; seed = sm["seed"]
        held = r["held_out"].get("none", {}); last = r["curve"][-1] if r["curve"] else {}
        probe, naive_probe = r.get("probe") or {}, r.get("naive_probe") or {}
        row = {"run": p.stem, "kind": kind, "seed": seed, "swap_at": r.get("swapped_at"), "final_meaning": r.get("final_meaning"),
               "naive_food": (r.get("naive") or {}).get("drawn", {}).get("food"), "last_window_food": last.get("food"), "last_window_timeout": last.get("timeout"),
               "held_drawn_food": held.get("drawn", {}).get("food"), "held_greedy_food": held.get("greedy", {}).get("food"), "held_greedy_steps": held.get("greedy", {}).get("steps_to_food"),
               "p_approach_fruit_before": naive_probe.get("decaying_fruit", {}).get("approach"), "p_approach_yeast_before": naive_probe.get("yeasty", {}).get("approach"),
               "p_approach_fruit_after": probe.get("decaying_fruit", {}).get("approach"), "p_approach_yeast_after": probe.get("yeasty", {}).get("approach"),
               "synapses_moved": r.get("synapses_moved"), "mean_abs_change": r.get("mean_abs_scale_change"), "wall_s": r.get("wall_seconds"),
               "lesions": {k: {"drawn": v["drawn"]["food"], "greedy": v["greedy"]["food"]} for k, v in r["held_out"].items()} if kind != "mlp" else None,
               "change_by_class": r.get("change_by_class"), "preference": r.get("preference")}
        rows.append(row); receipts[p.stem] = {"args": r.get("args"), "summary": sm, "cadence": r.get("cadence"), "fixture_sha256": r.get("fixture_sha256")}
    out = {"gate": "G4 the lessons", "tool": "tools/learn_odour.py", "rows": rows, "runs": receipts}
    (ROOT / "receipts" / "g4_lessons.json").write_text(json.dumps(out, indent=1))
    print("| run | naive food | last window food | held-out drawn | held-out greedy | p(approach) fruit before→after | p(approach) yeast before→after | synapses moved |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for w in rows:
        print(f"| {w['run']} | {f(w['naive_food'])} | {f(w['last_window_food'])} | {f(w['held_drawn_food'], 3)} | {f(w['held_greedy_food'], 3)} | {f(w['p_approach_fruit_before'])}→{f(w['p_approach_fruit_after'])} | {f(w['p_approach_yeast_before'])}→{f(w['p_approach_yeast_after'])} | {w['synapses_moved'] or ''} |")
    for w in rows:
        if w["lesions"] and w["kind"] == "connectome":
            print(f"\nlesions of {w['run']} (held-out greedy food): " + ", ".join(f"{k} {f(v['greedy'], 2)}" for k, v in w["lesions"].items()))
        if w["change_by_class"] and w["kind"] == "connectome":
            for target, by in w["change_by_class"].items():
                if isinstance(by, dict) and "fruit KCs" in by:
                    print(f"  change onto {target}: " + ", ".join(f"{k} {f(v['mean_change'], 3)} (n={v['synapses']})" for k, v in by.items()))
    print(f"\nreceipt: receipts/g4_lessons.json ({len(rows)} runs)")


if __name__ == "__main__":
    main()
