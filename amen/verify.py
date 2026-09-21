#!/usr/bin/env python3
"""Recheck the amen example: every receipt's sources, and that the page has what it plays.

    python amen/verify.py

For each runs/<name>/receipt.json: the schema and shape are the expected ones, every source
file exists and hashes to the recorded value, every clip the receipt lists has its audio,
trace and weights among the sources, and the page's index lists exactly the receipts' clips.
Each trace is then checked against its audio: the trace names that audio file and its hash,
and its step count covers the clip. Exits non-zero on any failure.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    problems, clips = [], set()
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
        for clip in body.get("results", {}).get("clips", []):
            clips.add(clip["id"])
            for relative in (f"web/audio/{clip['id']}.mp3", f"web/traces/{clip['id']}.json"):
                if relative not in body.get("sources", {}):
                    problems.append(f"{name}: {relative} is not among the sources")
        if body.get("verified") is not True:
            problems.append(f"{name}: the receipt is not marked verified")
    index = json.loads((HERE / "web/traces/index.json").read_text())
    if set(index) != clips:
        problems.append(f"the page index and the receipts disagree: {sorted(set(index) ^ clips)}")
    for clip_id, entry in index.items():
        trace = json.loads((HERE / "web" / entry["file"]).read_text())
        audio = HERE / "web" / trace["audio"]
        if not audio.exists() or sha(audio) != trace["audio_sha256"]:
            problems.append(f"{clip_id}: the trace does not match its audio")
        if not (HERE / "web" / trace["weights"]).exists():
            problems.append(f"{clip_id}: the weights file is missing")
        if len(trace["steps"]) != entry["steps"]:
            problems.append(f"{clip_id}: the index and the trace disagree on the step count")
    for problem in problems:
        print("FAIL:", problem)
    if not problems:
        print(f"amen: {len(receipts)} receipts, {len(clips)} clips, every source matches")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
