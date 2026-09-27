#!/usr/bin/env python3
"""Export the retained connectome and parity cases that bind web/brain.js to cadence.

Compilation here preserves individual recurrent rate units; it does not infer a
complete behavioral policy from anatomy. banc.py filters/signs the chemical graph,
brain.py names population masks, optional subnet.py recruitment selects an induced graph, lessons.py
supplies class gains/naive KC->MBON efficacies/output calibration, and payload_of
serializes incoming-edge CSR. No teacher action table is inserted in the graph.
The demo's substantial pilot/behavior assistance is separate runtime code.
See docs/FLY_MATRIX_BRAIN for exact parameters, input forcing and claim boundaries.

The default includes every neuron and edge in the retained fixture, including
disconnected cells and weak retained edges. --budget explicitly selects the older
recruited subset. --output-dir stages all generated files under a separate root.

  web/data/brain_full.json    the full graph: CSR by receiving neuron, weights, populations, member indices,
                         the neuron model and the gain from the gate-2 receipt, the dictionary's class
                         gains (log_gain), the plastic seam started naive (efficacy) and the two
                         readouts calibrated (bias): fruitfly/lessons.py
  web/data/lessons_full.json  the lesson setup's report for the page
  receipts/lessons_setup_full.json  the same, with graph selection and the source fixture
  tests/parity_full_cases.json  stimuli and the library's per-step readouts on the same graph

Explicit --budget uses the historical brain.json, lessons.json, lessons_setup.json
and parity_cases.json names. Full export leaves those historical artifacts intact.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cadence import Connectome, NeuronModel  # noqa: E402
from fruitfly.banc import MANIFEST_PATH  # noqa: E402
from fruitfly.brain import CLASS_LOG_GAIN, load_fly  # noqa: E402
from fruitfly.lessons import setup_lessons  # noqa: E402
from fruitfly.subnet import recruit  # noqa: E402

sys.path.insert(0, str(ROOT / "tools"))
from brain_payload import payload_of  # noqa: E402  (a copy of cadence-examples/engine/export.py)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    selection = ap.add_mutually_exclusive_group()
    selection.add_argument("--full", action="store_true", help="export every retained neuron and edge (default)")
    selection.add_argument("--budget", type=int, help="opt into recruited-subset export with this neuron budget")
    ap.add_argument("--hops", type=int, help="subset recruitment hops (default: 3; requires --budget)")
    ap.add_argument("--min-count", type=float, help="subset recruitment contact threshold (default: 6; requires --budget)")
    ap.add_argument("--gain", type=float, default=None, help="default: the gain of receipts/g2_reflex_facts.json")
    ap.add_argument("--output-dir", type=Path, default=ROOT, help="artifact root containing web/data, receipts and tests")
    args = ap.parse_args(argv)
    if args.budget is None and (args.hops is not None or args.min_count is not None):
        ap.error("--hops and --min-count require --budget; full export does not recruit")
    if args.budget is not None and args.budget < 1:
        ap.error("--budget must be positive")
    if args.hops is not None and args.hops < 0:
        ap.error("--hops must be nonnegative")
    if args.min_count is not None and (not np.isfinite(args.min_count) or args.min_count < 0):
        ap.error("--min-count must be finite and nonnegative")
    args.hops = 3 if args.hops is None else args.hops
    args.min_count = 6.0 if args.min_count is None else args.min_count
    return args


def select_connectome(whole: Connectome, *, budget: int | None = None, hops: int = 3,
                      min_count: float = 6.0) -> tuple[Connectome, np.ndarray, dict, dict | None]:
    """Select topology only; full export never applies recruitment or a new edge floor."""
    if budget is None:
        connectome, members, recruitment = whole, np.arange(whole.n, dtype=np.int64), None
    else:
        sub = recruit(whole, budget=budget, hops=hops, min_count=min_count)
        connectome, members = sub.connectome, sub.members
        recruitment = {"budget": budget, "hops": hops, "min_count": min_count,
                       "hop_counts": np.bincount(sub.hops).tolist()}
    selected = {"mode": "full_retained" if budget is None else "recruited_subset",
                "neurons": int(connectome.n), "classes": int(connectome.synapses),
                "all_retained_neurons": bool(connectome.n == whole.n),
                "all_retained_edges": bool(connectome.synapses == whole.synapses)}
    return connectome, members, selected, recruitment


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    gain = args.gain
    receipt = ROOT / "receipts" / "g2_reflex_facts.json"
    if gain is None:
        gain = float(json.loads(receipt.read_text())["body"]["connectome"]["gain"])
    fly = load_fly()
    C, members, selection, recruitment = select_connectome(
        fly.connectome, budget=args.budget, hops=args.hops, min_count=args.min_count)
    model = NeuronModel(gain=gain)
    setup = setup_lessons(C, gain=gain)
    brain = setup.brain(C, gain=gain)
    manifest = json.loads(MANIFEST_PATH.read_text())["fixture_sha256"]
    payload = payload_of(brain, members=members, extra={
        "manifest": manifest,
        "whole": {"neurons": int(fly.connectome.n), "edges": int(fly.connectome.synapses)},
        "selection": selection, "recruitment": recruitment,
        "class_log_gain": CLASS_LOG_GAIN, "lessons_setup": setup.report,
    })
    artifact_root = args.output_dir
    names = ({"brain": "brain_full.json", "lessons": "lessons_full.json",
              "setup": "lessons_setup_full.json", "parity": "parity_full_cases.json"}
             if args.budget is None else
             {"brain": "brain.json", "lessons": "lessons.json",
              "setup": "lessons_setup.json", "parity": "parity_cases.json"})
    paths = {"brain": artifact_root / "web" / "data" / names["brain"],
             "lessons": artifact_root / "web" / "data" / names["lessons"],
             "setup": artifact_root / "receipts" / names["setup"],
             "parity": artifact_root / "tests" / names["parity"]}
    (artifact_root / "receipts").mkdir(parents=True, exist_ok=True)
    setup_receipt = {"tool": "tools/export_web.py", "fixture": manifest, "gain": gain,
                     "class_log_gain": CLASS_LOG_GAIN, "selection": selection,
                     "recruitment": recruitment, **setup.report}
    if recruitment is not None:
        setup_receipt["subnet"] = {"neurons": int(C.n), "classes": int(C.synapses),
                                   "budget": args.budget, "hops": args.hops, "min_count": args.min_count}
    paths["setup"].write_text(json.dumps(setup_receipt, indent=1))
    (artifact_root / "web" / "data").mkdir(parents=True, exist_ok=True)
    paths["lessons"].write_text(json.dumps({"config": {}, "setup": setup.report}, indent=1))
    out = paths["brain"]
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
    (artifact_root / "tests").mkdir(parents=True, exist_ok=True)
    paths["parity"].write_text(json.dumps({"gain": gain, "cases": cases}, indent=1))
    print(f"{selection['mode']}: {C.n} neurons, {C.synapses} synapse classes, gain {gain}, class gains {CLASS_LOG_GAIN}; payload {out.stat().st_size/1e6:.1f} MB; parity cases {len(cases)}")
    print("lesson setup:", json.dumps(setup.report["seam"]), json.dumps(setup.report["readout_bias"]), json.dumps(setup.report["naive_outputs"]))
    for kind, path in paths.items():
        print(f"{kind}: {path}")


if __name__ == "__main__":
    main()
