"""Check the published gallery: every example's receipt, its predicates and its sources.

    python tools/verify_gallery.py

For each example directory (arm, world): the page and the README exist; the receipt's digest
matches its content; the schema, stage and status are the accepted ones; every scheduled
acceptance seed completed and none failed; every predicate passed; every stage file hashed in
the receipt matches the file in the repository; and the library commit the receipt ran against
is the commit the README installs. Prints the accepted examples and exits non-zero on any
failure.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = {"arm": "S01", "world": "S02", "connect_four": "S03"}
COMPARISONS = {"comparisons/arm": "S01-comparison", "comparisons/world": "S02-comparison"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(name: str, stage: str, pinned: str | None, *, page: bool = True) -> list[str]:
    """The problems of one receipt folder: a stage folder carries its page, README and receipt;
    a comparison folder carries the receipt alone, with its stage files under comparisons/."""
    problems = []
    folder = ROOT / name
    for required in ("index.html", "README.md", "receipt.json") if page else ("receipt.json",):
        if not (folder / required).exists():
            problems.append(f"{name}: {required} is missing")
    if problems:
        return problems
    body = json.loads((folder / "receipt.json").read_text())
    claimed = body.pop("receipt_sha256", None)
    content = json.dumps(body, sort_keys=True, indent=1, allow_nan=False).encode()
    if claimed != hashlib.sha256(content).hexdigest():
        problems.append(f"{name}: the receipt digest does not match its content")
    if body.get("schema") != "cadence-experience-run/v1" or body.get("stage") != stage:
        problems.append(f"{name}: schema {body.get('schema')} stage {body.get('stage')}")
    if body.get("status") != "complete":
        problems.append(f"{name}: status {body.get('status')}")
    seeds = body.get("seeds", {})
    scheduled, completed, arms = seeds.get("scheduled", []), seeds.get("completed", []), seeds.get("arms")
    if arms:  # a comparison completes every (seed, arm) pair
        complete = len(completed) == len(scheduled) * len(arms) and {(int(s), a) for s, a in completed} == {(int(s), a) for s in scheduled for a in arms}
    else:
        complete = sorted(completed) == sorted(scheduled)
    if seeds.get("split") != "acceptance" or seeds.get("failed") or not complete:
        problems.append(f"{name}: the acceptance seed schedule is incomplete")
    predicates = body.get("acceptance", {}).get("predicates", [])
    if not predicates or not all(p.get("passed") for p in predicates) or not body["acceptance"].get("passed"):
        failed = [p["name"] for p in predicates if not p.get("passed")]
        problems.append(f"{name}: predicates not passed {failed}")
    for file, digest in body.get("sources", {}).get("stage_files", {}).items():
        path = (folder if page else ROOT / "comparisons") / file
        if not path.exists() or sha256_file(path) != digest:
            problems.append(f"{name}: {file} differs from the file the receipt ran")
    commit = body.get("sources", {}).get("core", {}).get("repo", {}).get("commit")
    if pinned is not None and commit != pinned:
        problems.append(f"{name}: the receipt ran against cadence {commit}, the README installs {pinned}")
    return problems


def main() -> int:
    readme = (ROOT / "README.md").read_text()
    match = re.search(r"cadence\.git@([0-9a-f]{40})", readme)
    pinned = match.group(1) if match else None
    if pinned is None:
        print("FAIL: the README does not install cadence at a pinned commit")
        return 1
    problems = [p for name, stage in EXAMPLES.items() for p in check(name, stage, pinned)]
    problems += [p for name, stage in COMPARISONS.items() for p in check(name, stage, pinned, page=False)]
    for problem in problems:
        print("FAIL:", problem)
    if not problems:
        print(f"Accepted examples: {len(EXAMPLES)} ({', '.join(EXAMPLES)}); comparisons: {len(COMPARISONS)}; cadence {pinned[:12]}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
