"""Verify current receipts, or explicitly identify a pinned historical receipt.

Historical verification checks the preserved source bytes at their Git commit. It does
not certify the current implementation or bind the historical Cadence dependency.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import cadence as cd

ROOT = Path(__file__).resolve().parents[1]


def verify(name: str) -> tuple[bool, str]:
    receipt = ROOT / name / "receipt.json"
    raw = json.loads(receipt.read_text())
    spec = importlib.util.spec_from_file_location(f"receipt_{name}", ROOT / name / "train.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / name))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    # The producer declares required sources. Never let the receipt choose what is checked.
    sources = module.SOURCES if hasattr(module, "SOURCES") else module.sources()
    ok, message = cd.Receipt.verify(receipt, sources=sources, check=module.check)
    if ok:
        return True, "source bytes and digest agree; this alone does not re-run measurements"
    ok, message = cd.Receipt.verify(receipt, check=module.check)
    if not ok:
        return False, message
    history = json.loads((ROOT / "tools/receipt_history.json").read_text()).get(name)
    if not history or history["receipt_digest"] != raw["digest"]:
        return False, "source changed and receipt has no matching historical provenance"
    for row in raw["source"]["files"]:
        content = subprocess.check_output(
            ["git", "show", f"{history['commit']}:{row['path']}"], cwd=ROOT
        )
        if hashlib.sha256(content).hexdigest() != row["sha256"]:
            return False, "historical source digest differs"
    return True, f"HISTORICAL at {history['commit'][:12]}; not a result of current code"


def main() -> int:
    names = sys.argv[1:] or [p.parent.name for p in sorted(ROOT.glob("*/receipt.json"))]
    failed = False
    for name in names:
        ok, message = verify(name)
        print(f"{name}: {'ok' if ok else 'FAIL'} — {message}")
        failed |= not ok
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
