"""Verify an S01 run from its causal event logs, independently of the agent's code.

    python verify.py runs/acceptance

Recomputes the candidate's held-out success from the logged terminal outcomes, the
displacement-prediction gain of the babbling phase from logged predictions and
observed accelerations, checks the receipt digest, source hashes, seed schedule,
finite values and pass flags, and fails closed on an incomplete run.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def fail(message: str) -> int:
    print("FAIL:", message)
    return 1


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    run = Path(argv[1])
    path = run / "receipt.json"
    if not path.exists():
        return fail("no receipt")
    body = json.loads(path.read_text())
    claimed = body.pop("receipt_sha256", None)
    if claimed != hashlib.sha256(json.dumps(body, sort_keys=True, indent=1, allow_nan=False).encode()).hexdigest():
        return fail("receipt digest does not match its content")
    if body.get("schema") != "cadence-experience-run/v1" or body.get("stage") != "S01":
        return fail("unknown schema or stage")
    if body["status"] != "complete":
        return fail(f"run status is {body['status']}")
    if body["seeds"]["failed"] or set(body["seeds"]["completed"]) != set(body["seeds"]["scheduled"]):
        return fail("seed schedule incomplete")
    stage_dir = Path(__file__).resolve().parents[1] / body["sources"]["stage_dir"]
    for name, sha in body["sources"]["stage_files"].items():
        file = stage_dir / name
        if not file.exists() or hashlib.sha256(file.read_bytes()).hexdigest() != sha:
            return fail(f"source {name} changed since the run")
    config = body["config"]
    if hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest() != body["config_sha256"]:
        return fail("config digest mismatch")
    per_seed = {r["seed"]: r for r in body["metrics"]["per_seed"]}
    for artifact in body["artifacts"]["events"]:
        file = run / artifact["path"]
        if hashlib.sha256(file.read_bytes()).hexdigest() != artifact["sha256"]:
            return fail(f"event log {artifact['path']} changed")
        seed = int(artifact["path"].split("seed")[1].split(".")[0])
        r = per_seed[seed]
        heldout: dict[str, list[float]] = defaultdict(list)
        babble_pred: dict[tuple[str, int], np.ndarray] = {}
        model_err, zero_err = [], []
        for line in file.read_text().splitlines():
            e = json.loads(line)
            if e["phase"] == "heldout":
                name = e["life"].split("-")[1] + "-" + e.get("mode", "adapting")
                if e["terminated"] or e["truncated"]:
                    heldout[name].append(float(e["terminated"]))
            elif e["life"] == "candidate":
                # a babbling prediction is scored against the outcome record that follows it in
                # the same episode; the last one is logged under the phase that follows
                key = (e["life"], e["episode"])
                if e["feedback_for"] is not None and key in babble_pred:
                    model_err.append(float(np.mean((babble_pred[key] - np.asarray(e["dd_hand"])) ** 2)))
                    zero_err.append(float(np.mean(np.square(e["dd_hand"]))))
                if e["phase"] == "babbling" and e["decision"] is not None and e["decision"].get("prediction"):
                    babble_pred[key] = np.asarray(e["decision"]["prediction"]["dd_hand"])
                else:
                    babble_pred.pop(key, None)
        for name, claimed_success in (("candidate-adapting", r["heldout"]["success"]), ("candidate-frozen", r["heldout_frozen"]["success"]), ("changed-adapting", r["changed_body"]["success"]), ("returned-adapting", r["returned_body"]["success"])):
            if not heldout[name]:
                return fail(f"seed {seed}: no held-out episodes for {name}")
            if abs(float(np.mean(heldout[name])) - claimed_success) > 1e-9:
                return fail(f"seed {seed}: held-out success of {name} recomputes to {np.mean(heldout[name])}, receipt says {claimed_success}")
            if len(heldout[name]) != r["budget"]["heldout_targets"]:
                return fail(f"seed {seed}: {len(heldout[name])} held-out episodes for {name}, budget says {r['budget']['heldout_targets']}")
        if len(model_err) < 500:
            return fail(f"seed {seed}: too few babbling transitions logged")
        gain = 1 - float(np.mean(model_err[-500:])) / float(np.mean(zero_err[-500:]))
        if abs(gain - body["metrics"]["displacement_gain"][body["seeds"]["completed"].index(seed)]) > 1e-9:
            return fail(f"seed {seed}: displacement gain recomputes to {gain}")
    for p in body["acceptance"]["predicates"]:
        if p["value"] is None or not np.isfinite(p["value"]):
            return fail(f"predicate {p['name']} has no finite value")
    passed = all(p["passed"] for p in body["acceptance"]["predicates"])
    print(f"receipt {body['run_id']}: {len(per_seed)} seeds, {len(body['acceptance']['predicates'])} predicates, {'all passed' if passed else 'some failed'}")
    for p in body["acceptance"]["predicates"]:
        print(f"  {'PASS' if p['passed'] else 'FAIL'} {p['name']}: {p['value']:.4f} vs {p['threshold']}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
