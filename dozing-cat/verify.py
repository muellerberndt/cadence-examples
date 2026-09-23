#!/usr/bin/env python3
"""Recheck the dozing cat: the receipts, the numbers the README states, and the export the page runs.

    python dozing-cat/verify.py

- Every receipt's digest recomputes over its canonical body; every receipt names the belief
  the page runs (the sha256 of ``pretrained/cat_belief.npz``), the library's commit and the
  digest of ``cat.py``'s belief module, and no path of the machine it was made on.
- The lives receipts: every mean in a summary is the mean of its five lives, and every
  life's fitness is its catch rate minus the priced compute minus the priced surprise.
- The evolution receipts: the held-out fitness of every genome is the mean of its five lives,
  the winner is the best of the three, the hand-set genome is the one ``cat.py`` declares, and
  the best fitness never fell across the generations.
- Every number the README states is the receipts' number at the README's rounding.
- ``web/data/brain.json`` is the export of ``pretrained/cat_belief.npz`` and the receipts, and
  ``web/brain_scan.js`` is the viewer the export recorded; the parity recording was made with
  that belief and holds a kept and an undone learn call.

Exits non-zero on any failure. The page's arithmetic against the library's is checked by
``tests/parity.mjs``; the sill's invariants by ``tests/chaos.mjs``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import cat  # noqa: E402
import export  # noqa: E402
from cadence.receipts import canonical_json  # noqa: E402

RECEIPTS = ("cat_evolution_patch.json", "cat_evolution_threshold.json", "cat_lives_hand_set.json", "cat_lives_evolved.json", "cat_timing.json")
PRICES = {"moments": cat.PRICE_MOMENTS, "surprise": cat.PRICE_SURPRISE}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fitness_of(row: dict) -> float:
    return row["catch_rate"] - PRICES["moments"] * row["moments_per_decision"] - PRICES["surprise"] * row["surprise_term"]


def main() -> int:
    problems: list[str] = []
    bodies: dict[str, dict] = {}
    belief_sha = sha(HERE / "pretrained" / "cat_belief.npz")
    for name in RECEIPTS:
        body = json.loads((HERE / "receipts" / name).read_text())
        bodies[name] = body
        sealed = body.pop("receipt_sha256", None)
        if hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest() != sealed:
            problems.append(f"{name}: the digest does not recompute")
        text = canonical_json(body)
        if "/Users/" in text or "/home/" in text:
            problems.append(f"{name}: the receipt names a path of the machine it was made on")
        if len(str(body["revision"]["library"].get("commit", ""))) != 40 or len(str(body["copied"].get("original_receipt_sha256", ""))) != 64:
            problems.append(f"{name}: the library commit or the original digest is not recorded")
        if body["revision"]["library"]["belief_py_sha256"] != bodies[RECEIPTS[0]]["revision"]["library"]["belief_py_sha256"]:
            problems.append(f"{name}: the belief module differs from the other receipts'")
        if "belief" in body and body["belief"]["sha256"] != belief_sha:
            problems.append(f"{name}: the receipt names another belief than pretrained/cat_belief.npz")
        if body["prices"] != PRICES if "prices" in body else False:
            problems.append(f"{name}: the prices differ from cat.py's")

    # the lives: summaries from their rows
    for name in ("cat_lives_hand_set.json", "cat_lives_evolved.json"):
        body = bodies[name]
        for arm, rows in body["lives"].items():
            if len(rows) != len(body["seeds"]) or [r["seed"] for r in rows] != body["seeds"]:
                problems.append(f"{name}, {arm}: the lives are not the five seeds")
            for r in rows:
                if abs(r["fitness"]["fitness"] - fitness_of({"catch_rate": r["metrics"]["catch_rate"], "moments_per_decision": r["compute"]["moments_per_decision"], "surprise_term": r["metrics"]["surprise_term"]})) > 1e-9:
                    problems.append(f"{name}, {arm}, seed {r['seed']}: the fitness is not the priced catch rate")
                if abs(r["fitness"]["moments_per_decision"] - r["compute"]["moments_per_decision"]) > 1e-9:
                    problems.append(f"{name}, {arm}, seed {r['seed']}: the compute in the fitness is not the life's")
            summary = body["summary"][arm]
            for key, entry in summary.items():
                if not isinstance(entry, dict) or "values" not in entry:
                    continue
                values = [r["metrics"].get(key) for r in rows] if key != "fitness" else [r["fitness"]["fitness"] for r in rows]
                if values != entry["values"]:
                    problems.append(f"{name}, {arm}: the per-seed values of {key} are not the lives'")
                got = [v for v in values if v is not None]
                mean = float(np.mean(got)) if got else None
                if (mean is None) != (entry["mean"] is None) or (mean is not None and abs(mean - entry["mean"]) > 1e-9):
                    problems.append(f"{name}, {arm}: the mean of {key} is not the mean of the lives")
            for kind in ("assimilate", "imagine", "observe", "refit", "governor"):
                if abs(summary["moments_by_kind"][kind] - float(np.mean([r["compute"][kind] for r in rows]))) > 1e-6:
                    problems.append(f"{name}, {arm}: the moments of kind {kind} are not the mean of the lives")

    # the evolutions: held-out means, the winner, the hand-set start, the climb
    for which, name in (("patch", "cat_evolution_patch.json"), ("threshold", "cat_evolution_threshold.json")):
        body = bodies[name]
        if body["which"] != which or body["hand_set"] != cat.HAND_SETS[which]:
            problems.append(f"{name}: the hand-set genome is not the one cat.py declares")
        if set(body["space"]) != set(cat.SPACES[which]):
            problems.append(f"{name}: the gene space is not the one cat.py declares")
        for genome_name, held in body["held_out"].items():
            rows = held["rows"]
            if [r["seed"] for r in rows] != body["test_seeds"]:
                problems.append(f"{name}, {genome_name}: the held-out lives are not the test seeds")
            for key in ("fitness", "catch_rate", "moments_per_decision", "surprise_term", "awake_share"):
                if abs(held[key] - float(np.mean([r[key] for r in rows]))) > 1e-9:
                    problems.append(f"{name}, {genome_name}: {key} is not the mean of the held-out lives")
            for r in rows:
                if abs(r["fitness"] - fitness_of(r)) > 1e-9:
                    problems.append(f"{name}, {genome_name}, seed {r['seed']}: the fitness is not the priced catch rate")
        winner = max(body["held_out"], key=lambda k: body["held_out"][k]["fitness"])
        if body["winner"] != winner:
            problems.append(f"{name}: the winner is {body['winner']}, the held-out lives say {winner}")
        best = [h["best_fitness"] for h in body["evolved"]["history"]]
        if len(best) != body["generations"] or any(b < a - 1e-12 for a, b in zip(best, best[1:])):
            problems.append(f"{name}: the best fitness fell across the generations")
        if abs(body["evolved"]["train_fitness"] - best[-1]) > 1e-9:
            problems.append(f"{name}: the evolved genome's training fitness is not the last generation's best")
        scores = body["random_search"]["scores"]
        if len(scores) != body["generations"] * body["population"] or abs(max(scores) - body["random_search"]["train_fitness"]) > 1e-9:
            problems.append(f"{name}: the random search's best is not the best of its draws")
        if body["random_search"]["genomes"][int(np.argmax(scores))] != body["random_search"]["genome"]:
            problems.append(f"{name}: the random search's genome is not its best draw")

    # the numbers the README states, at the README's rounding
    ep, et = bodies["cat_evolution_patch.json"], bodies["cat_evolution_threshold.json"]
    lh, le = bodies["cat_lives_hand_set.json"]["summary"], bodies["cat_lives_evolved.json"]["summary"]
    timing = bodies["cat_timing.json"]["pieces"]
    ctx = json.loads((HERE / "pretrained" / "cat_belief.json").read_text())
    stated = [
        ("patch evolved, held-out fitness", ep["held_out"]["evolved"]["fitness"], 0.853, 3), ("patch evolved, catch rate", ep["held_out"]["evolved"]["catch_rate"], 0.961, 3),
        ("patch evolved, moments per decision", ep["held_out"]["evolved"]["moments_per_decision"], 15.3, 1), ("patch evolved, awake share", ep["held_out"]["evolved"]["awake_share"], 0.238, 3),
        ("patch hand-set, held-out fitness", ep["held_out"]["hand_set"]["fitness"], 0.782, 3), ("patch random search, held-out fitness", ep["held_out"]["random_search"]["fitness"], 0.642, 3),
        ("patch, best fitness of the last generation", ep["evolved"]["history"][-1]["best_fitness"], 0.906, 3), ("patch, best fitness of the first generation", ep["evolved"]["history"][0]["best_fitness"], 0.824, 3),
        ("thresholds evolved, held-out fitness", et["held_out"]["evolved"]["fitness"], 0.795, 3), ("thresholds evolved, catch rate", et["held_out"]["evolved"]["catch_rate"], 0.909, 3),
        ("thresholds evolved, moments per decision", et["held_out"]["evolved"]["moments_per_decision"], 17.5, 1), ("thresholds evolved, awake share", et["held_out"]["evolved"]["awake_share"], 0.269, 3),
        ("thresholds hand-set, held-out fitness", et["held_out"]["hand_set"]["fitness"], 0.783, 3), ("thresholds random search, held-out fitness", et["held_out"]["random_search"]["fitness"], 0.788, 3),
        ("hand-set lives, patch catch rate", lh["patch"]["catch_rate"]["mean"], 0.977, 3), ("hand-set lives, thresholds catch rate", lh["threshold"]["catch_rate"]["mean"], 0.931, 3),
        ("hand-set lives, always awake catch rate", lh["always_awake"]["catch_rate"]["mean"], 0.962, 3), ("hand-set lives, never wakes catch rate", lh["never_wakes"]["catch_rate"]["mean"], 0.249, 3),
        ("hand-set lives, patch moments", lh["patch"]["moments_per_decision"]["mean"], 37.3, 1), ("hand-set lives, thresholds moments", lh["threshold"]["moments_per_decision"]["mean"], 26.2, 1),
        ("hand-set lives, always awake moments", lh["always_awake"]["moments_per_decision"]["mean"], 71.3, 1), ("hand-set lives, never wakes moments", lh["never_wakes"]["moments_per_decision"]["mean"], 1.0, 1),
        ("hand-set lives, patch awake share", lh["patch"]["awake_share"]["mean"], 0.365, 3), ("hand-set lives, thresholds awake share", lh["threshold"]["awake_share"]["mean"], 0.271, 3),
        ("hand-set lives, never wakes misses", lh["never_wakes"]["misses"]["mean"], 12.0, 1), ("hand-set lives, patch catches", lh["patch"]["catches"]["mean"], 26.4, 1),
        ("hand-set lives, patch detection", lh["patch"]["detection_latency"]["mean"], 99.6, 1), ("hand-set lives, thresholds detection", lh["threshold"]["detection_latency"]["mean"], 158.0, 1),
        ("evolved lives, patch awake share with a dot", le["patch"]["awake_share_with_dot"]["mean"], 0.80, 2), ("evolved lives, patch awake share without a dot", le["patch"]["awake_share_without_dot"]["mean"], 0.03, 2),
        ("evolved lives, patch wake latency", le["patch"]["wake_latency_mean"]["mean"], 1.0, 1), ("evolved lives, patch learn calls", le["patch"]["learn_calls"]["mean"], 0.0, 1),
        ("evolved lives, patch false wakes", le["patch"]["false_wakes"]["mean"], 14.0, 1), ("evolved lives, thresholds false wakes", le["threshold"]["false_wakes"]["mean"], 5.4, 1),
        ("timing, governor synapses", timing["governor_synapses"], 23, 0), ("timing, governor steps", timing["governor_steps_median"], 20, 0), ("timing, imagination moments", timing["imagination_moments"], 60, 0),
        ("timing, a learn call's moments", timing["learn_calls"][0]["moments"], 1824, 0), ("moment multiply-accumulates", timing["moment_macs"], 96488, 0),
        ("parameters", ctx["parameters"], 6853, 0), ("paw explained", ctx["with_store"]["explained_one_step"]["paw"], 0.94, 2), ("motion explained", ctx["with_store"]["explained_one_step"]["position_motion"], 0.58, 2),
        ("routine surprise, median", ctx["routine"]["median"], 0.016, 3), ("imagination eight decisions ahead", ctx["with_store"]["imagination"]["8"], 0.39, 2),
    ]
    for label, actual, wanted, places in stated:
        if actual is None or round(float(actual), places) != wanted:
            problems.append(f"the README states {wanted} for {label}; the receipts say {actual}")

    # the export the page runs, the viewer, the recording
    brain = json.loads((HERE / "web" / "data" / "brain.json").read_text())
    if export.strip(export.build(None)) != export.strip(brain):
        problems.append("web/data/brain.json is not the export of pretrained/cat_belief.npz and the receipts")
    if brain["belief"]["npz_sha256"] != belief_sha:
        problems.append("web/data/brain.json names another belief than pretrained/cat_belief.npz")
    if brain["viewer"]["sha256"] != sha(HERE / "web" / "brain_scan.js"):
        problems.append("web/brain_scan.js is not the viewer the export recorded")
    if any(brain["belief"]["records"]["table_y"]):
        problems.append("the exported record store is not empty")
    for which in ("patch", "threshold"):
        if brain["genomes"][which]["evolved"] != bodies[f"cat_evolution_{which}.json"]["evolved"]["genome"] or brain["default_genome"][which] != bodies[f"cat_evolution_{which}.json"]["winner"]:
            problems.append(f"the page's {which} genomes are not the receipt's")
    cases = json.loads((HERE / "tests" / "parity_cases.json").read_text())
    if cases.get("belief_npz_sha256") != belief_sha or len(cases.get("cases", [])) < 6:
        problems.append("the parity recording was made with another belief, or holds fewer than six cases")
    kept = [l["entry"]["kept"] for c in cases["cases"] for l in c.get("learns", [])]
    if True not in kept or False not in kept:
        problems.append("the parity recording does not hold both a kept and an undone learn call")
    if any(c["genome"] != brain["genomes"][c["which"]][("evolved" if "evolved" in c["name"] else "hand_set")] for c in cases["cases"]):
        problems.append("a parity case's genome is not the page's")

    for problem in problems:
        print("FAIL:", problem)
    if not problems:
        print(f"ok: {len(RECEIPTS)} receipts re-sealed and recomputed, {len(stated)} numbers of the README checked, web/data/brain.json is the export of pretrained/cat_belief.npz, the parity recording is bound to it")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
