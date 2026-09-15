"""Verify an S02 run from its causal event logs, independently of the agent's code.

    python verify.py runs/acceptance

Recomputes the held-out request success, correction and grounding rates from the logged
terminal outcomes, the consequence accuracies from logged predictions and outcomes,
checks the receipt digest, source hashes, seed schedule, finite values and pass flags,
and fails closed on an incomplete run.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CONSEQUENCES = ("dx", "dy", "outcome", "here_object", "carrying")


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
    if body.get("schema") != "cadence-experience-run/v1" or body.get("stage") != "S02":
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
    if hashlib.sha256(json.dumps(body["config"], sort_keys=True, separators=(",", ":")).encode()).hexdigest() != body["config_sha256"]:
        return fail("config digest mismatch")
    per_seed = {r["seed"]: r for r in body["metrics"]["per_seed"]}
    for artifact in body["artifacts"]["events"]:
        file = run / artifact["path"]
        if hashlib.sha256(file.read_bytes()).hexdigest() != artifact["sha256"]:
            return fail(f"event log {artifact['path']} changed")
        seed = int(artifact["path"].split("seed")[1].split(".")[0])
        r = per_seed[seed]
        outcomes: dict[tuple[str, str], list[float]] = defaultdict(list)
        for line in file.read_text().splitlines():
            e = json.loads(line)
            # a request's outcome is the closing moment of the request episode; the wander that
            # precedes it is closed as truncated by the task and is not an outcome
            if e["life"].startswith("heldout-") and (e["terminated"] or e["truncated"]) and e.get("request"):
                outcomes[(e["life"].split("heldout-")[1], e["phase"])].append(float(e["terminated"]))
        checks = (("candidate", "remember-32", r["heldout"]["remember_32"]), ("candidate", "remember-16-visible", r["heldout"]["correction_visible"]),
                  ("candidate", "word", r["heldout"]["words"]), ("erased-places", "remember-32", r["controls"]["erased_places"]["remember_32"]),
                  ("erased-words", "word", r["controls"]["erased_words"]["words"]), ("shuffled", "remember-32", r["controls"]["shuffled_pairing"]["remember_32"]))
        for life, phase, claimed_rate in checks:
            got = outcomes.get((life, phase))
            if not got:
                return fail(f"seed {seed}: no held-out episodes for {life} {phase}")
            if abs(float(np.mean(got)) - claimed_rate) > 1e-9:
                return fail(f"seed {seed}: {life} {phase} recomputes to {np.mean(got)}, receipt says {claimed_rate}")
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
