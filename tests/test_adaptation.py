"""Source-bound task probes and causal observation boundaries."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def test_forager_observation_and_exploration_boundaries():
    subprocess.run(["node", "--test", "fly/forager.test.mjs"], cwd=ROOT, check=True)


def test_adaptation_receipt_preserves_schedule_and_tradeoffs():
    receipt = json.loads((ROOT / "evidence/adaptation_evidence.json").read_text())
    for name, digest in receipt["sources"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    assert len(receipt["fly"]) == 12
    rows = receipt["fly"]
    previous = [r for r in rows if r["condition"] == "previous_selection"]
    current = [r for r in rows if r["condition"] == "aging_observations"]
    assert [r["seed"] for r in previous] == [r["seed"] for r in current]
    assert sum(r["after"]["correct"] for r in current) >= 46
    assert sum(r["after"]["correct"] for r in current) > sum(r["after"]["correct"] for r in previous)
    assert len(receipt["arm"]) == 3
    assert all(r["coverage"] >= .95 for r in receipt["arm"])
