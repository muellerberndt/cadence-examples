"""Check local training custody without downloading or changing any result."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
errors = []
checked = []
for p in sorted((ROOT / "runs").glob("*/receipt.json")):
    r = json.loads(p.read_text())
    if "sources" not in r or "dataset_manifest_sha256" not in r:
        continue
    for name, expected in r["sources"].items():
        candidates = [
            ROOT / name,
            ROOT / "runs/training_source" / name,
            ROOT / "runs/four-gpu-source" / name,
            ROOT / "runs/four-gpu-training-source" / name,
            ROOT / "runs/ensemble-pilot-source" / name,
        ]
        if not any(
            q.exists() and hashlib.sha256(q.read_bytes()).hexdigest() == expected
            for q in candidates
        ):
            errors.append(f"{p.parent.name}: source mismatch {name}")
    manifest = ROOT / "data" / r["dataset"] / "manifest.json"
    if (
        hashlib.sha256(manifest.read_bytes()).hexdigest()
        != r["dataset_manifest_sha256"]
    ):
        errors.append(f"{p.parent.name}: dataset changed")
    checkpoint = p.parent / ("model.pt" if "model" in r else "brain.npz")
    if not checkpoint.exists():
        errors.append(f"{p.parent.name}: checkpoint absent")
    elif (
        r.get("checkpoint_sha256")
        and hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        != r["checkpoint_sha256"]
    ):
        errors.append(f"{p.parent.name}: checkpoint changed")
    checked.append(p.parent.name)
assert not errors, errors
print(json.dumps({"training_receipts_checked": checked, "errors": errors}))
