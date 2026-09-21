#!/usr/bin/env python3
"""Recheck the amen example: the receipt's sources, the shipped brain, and the browser engine.

    python amen/verify.py

For each runs/<name>/receipt.json: the schema and shape are the expected ones, the Cadence
commit is pinned, and every source file exists and hashes to the recorded value. The model
files are checked against model.json (sizes follow from the declared shapes). When node is
installed, parity.mjs is run: the browser engine must reproduce the archived run's sixteen
bars from silence, every slice, note and change point. Exits non-zero on any failure.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    problems = []
    receipts = sorted((HERE / "runs").glob("*/receipt.json"))
    if not receipts:
        problems.append("no receipts under runs/")
    for path in receipts:
        body = json.loads(path.read_text())
        name = path.parent.name
        if body.get("schema") != f"cadence-examples.amen.{name}/v1":
            problems.append(f"{name}: unexpected schema {body.get('schema')!r}")
        for key in ("question", "environment", "sources", "results", "verified"):
            if key not in body:
                problems.append(f"{name}: {key} is missing")
        if len(str(body.get("environment", {}).get("library_commit", ""))) != 40:
            problems.append(f"{name}: the cadence commit is not pinned")
        for relative, expected in body.get("sources", {}).items():
            file = HERE / relative
            if not file.exists():
                problems.append(f"{name}: {relative} is missing")
            elif sha(file) != expected:
                problems.append(f"{name}: {relative} does not match its recorded hash")
        if body.get("verified") is not True:
            problems.append(f"{name}: the receipt is not marked verified")
    index = json.loads((HERE / "web/models/index.json").read_text())
    for entry in index["brains"]:
        folder = HERE / "web/models" / entry["name"]
        model = json.loads((folder / "model.json").read_text())
        expected = sum(math.prod(shape) for _, shape in model["params_order"]) * 8
        if (folder / model["files"]["params"]).stat().st_size != expected:
            problems.append(f"{entry['name']}: the parameter file does not have the declared size")
        if (folder / model["files"]["records_y"]).stat().st_size != model["records"]["cells"] * model["outputs"] * 4:
            problems.append(f"{entry['name']}: the record table does not have the declared size")
        if (folder / model["files"]["records_mean"]).stat().st_size != model["records"]["reading"] * 8:
            problems.append(f"{entry['name']}: the record mean does not have the declared size")
        if not (HERE / "runs" / entry["run"] / "receipt.json").exists():
            problems.append(f"{entry['name']}: no receipt for run {entry['run']}")
    parity = "skipped (node is not installed)"
    if shutil.which("node"):
        result = subprocess.run(["node", str(HERE / "parity.mjs")], capture_output=True, text=True)
        parity = "; ".join(line.split(":")[0] for line in result.stdout.strip().splitlines()) if result.stdout.strip() else result.stderr.strip()
        if result.returncode != 0:
            problems.append("the browser engine does not reproduce the archived run: " + parity)
    for problem in problems:
        print("FAIL:", problem)
    if not problems:
        print(f"amen: {len(receipts)} receipt, every source matches; parity: {parity}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
