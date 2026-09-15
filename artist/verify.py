"""Verify an S04 run from its causal event log, independently of the brain's code.

    python verify.py runs/pilot

Recomputes the held-out discrepancy and F1 per family from the logged closing events, the ink
and displacement prediction gains from the logged scribble predictions, the decisions per
drawing and the body steps of the life, checks the receipt digest, the source hashes, the seed
schedule and every predicate's value, and fails closed on an incomplete run.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SIMPLE = ("segment", "two_segments")
LIFE_PHASES = ("scribble", "segment", "stroke", "multi", "changed-body", "returned-body", "offset-canvas")


def fail(message: str) -> int:
    print("FAIL:", message)
    return 1


def close(a: float, b: float, tolerance: float = 1e-6) -> bool:
    return bool(abs(float(a) - float(b)) <= tolerance)


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
    if body.get("schema") != "cadence-experience-run/v1" or body.get("stage") != "S04":
        return fail("unknown schema or stage")
    if body["status"] != "complete":
        return fail(f"run status is {body['status']}")
    if body["seeds"]["failed"] or set(body["seeds"]["completed"]) != set(body["seeds"]["scheduled"]):
        return fail("seed schedule incomplete")
    stage_dir = Path(__file__).resolve().parent
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
        result = per_seed[seed]
        closing: dict[tuple[str, str], list[dict]] = defaultdict(list)
        model: dict[str, list[float]] = defaultdict(list)
        zero: dict[str, list[float]] = defaultdict(list)
        life_steps = 0
        decisions_per_drawing: list[int] = []
        for line in file.read_text().splitlines():
            event = json.loads(line)
            if event.get("final") is not None:
                key = (str(event.get("life")), str(event.get("mode", "")))
                closing[key].append({"family": event.get("family"), **event["final"], "decisions": int(event["tick"])})
                if event.get("phase") == "heldout" and event.get("life") == "candidate" and event.get("mode") == "adapting":
                    decisions_per_drawing.append(int(event["tick"]))
            if event.get("life") == "candidate" and event.get("phase") in LIFE_PHASES and event.get("action") is not None:
                life_steps += 1
            if event.get("phase") == "scribble" and event.get("life") == "candidate" and event.get("predicted"):
                for name in ("dd_hand", "mark"):
                    if name in event["predicted"] and name in event.get("observed", {}):
                        predicted = np.asarray(event["predicted"][name], float)
                        observed = np.asarray(event["observed"][name], float)
                        model[name].append(float(np.mean((predicted - observed) ** 2)))
                        zero[name].append(float(np.mean(observed**2)))
        for key, claim in (
            (("candidate", "adapting"), result["heldout"]),
            (("candidate", "frozen"), result["heldout_frozen"]),
            (("changed", "adapting"), result["changed_body"]),
            (("returned", "adapting"), result["returned_body"]),
            (("offset-after", "adapting"), result["offset_after"]),
            (("offset-no-readback", "adapting"), result["offset_no_readback"]),
        ):
            rows = closing[key]
            if not rows:
                return fail(f"seed {seed}: no closing events for {key}")
            if len(rows) != claim["drawings"]:
                return fail(f"seed {seed}: {len(rows)} drawings logged for {key}, receipt says {claim['drawings']}")
            if not close(float(np.mean([r["f1"] for r in rows])), claim["f1"], 1e-5):
                return fail(f"seed {seed}: F1 of {key} recomputes to {np.mean([r['f1'] for r in rows])}, receipt says {claim['f1']}")
            if not close(float(np.mean([r["chamfer"] for r in rows])), claim["chamfer"], 1e-5):
                return fail(f"seed {seed}: chamfer of {key} recomputes to {np.mean([r['chamfer'] for r in rows])}")
            simple = [r for r in rows if r["family"] in SIMPLE]
            if simple and not close(float(np.mean([r["f1"] for r in simple])), claim["f1_simple"], 1e-5):
                return fail(f"seed {seed}: simple-family F1 of {key} recomputes to {np.mean([r['f1'] for r in simple])}")
            for family, claimed_family in claim["per_family"].items():
                rows_f = [r for r in rows if r["family"] == family]
                if len(rows_f) != claimed_family["count"] or not close(float(np.mean([r["f1"] for r in rows_f])), claimed_family["f1"], 1e-5):
                    return fail(f"seed {seed}: family {family} of {key} does not recompute")
        if decisions_per_drawing and max(decisions_per_drawing) * config["body"]["substeps"] > config["gates"]["steps_per_drawing"]:
            return fail(f"seed {seed}: a held-out drawing used more body steps than the cap")
        if life_steps * config["body"]["substeps"] > config["budget"]["body_steps_cap"]:
            return fail(f"seed {seed}: the life used more body steps than the cap")
        if not close(life_steps * config["body"]["substeps"], result["body_steps"], 5 * config["body"]["substeps"]):
            return fail(f"seed {seed}: logged body steps {life_steps * config['body']['substeps']} against the receipt's {result['body_steps']}")
        window = int(result["scribble_window"])
        for name, key in (("mark", "mark"), ("dd_hand", "dd_hand")):
            if len(model[name]) < window:
                return fail(f"seed {seed}: {len(model[name])} logged scribble predictions for {name}, the window is {window}")
            gain = 1 - float(np.mean(model[name][-window:])) / float(np.mean(zero[name][-window:]))
            if not close(gain, result["scribble"][key]["gain"], 2e-3):  # the log rounds to six decimals
                return fail(f"seed {seed}: {name} prediction gain recomputes to {gain}, receipt says {result['scribble'][key]['gain']}")
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
