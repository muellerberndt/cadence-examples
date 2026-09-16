"""Merge the memories of schooling runs into a checkpoint.

    python connect_four/tools/school_merge.py --stem runs/page/brain_seed10_games001200 \
        --school runs/school/*.json --out runs/page_schooled/brain_seed10_games001200 [--max-stones 24] [--limit 60000]

Every proven position of every school file joins the checkpoint's memory; the records, the
agent and the counters are unchanged. Proven results cannot disagree, and a disagreement
stops the merge. Positions with more than --max-stones stones are left out and the memory
is cut to --limit positions, the ones with the fewest stones kept, since the search re-proves
the late positions cheapest and the page carries the memory. The receipt beside the stem, if
any, is copied beside the output.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from connect_four.brain import Brain  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stem", type=Path, required=True)
    p.add_argument("--school", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-stones", type=int, default=24)
    p.add_argument("--limit", type=int, default=60000)
    p.add_argument("--wins-limit", type=int, default=80000, help="boards with winners' columns kept: the fewest stones, then the most games")
    a = p.parse_args()
    brain = Brain.load(a.stem)
    before = len(brain.planner.memory)
    games = {"win": 0, "draw": 0, "loss": 0}
    for file in a.school:
        s = json.loads(file.read_text())
        boards = np.asarray(s["boards"], dtype=np.int8)
        for i in range(int(s["entries"])):
            key = (boards[i].tobytes(), int(s["sides"][i]))
            value = float(s["results"][i])
            old = brain.planner.memory.get(key)
            if old is not None and old != value:
                raise SystemExit(f"{file}: entry {i} disagrees with the memory ({old} against {value})")
            brain.planner.memory[key] = value
        for view, column, count in s.get("wins", []):
            counts = brain.planner.wins.setdefault(np.asarray(view, dtype=np.int8).tobytes(), {})
            counts[int(column)] = counts.get(int(column), 0) + int(count)
        for g in s.get("games", []):
            games[g["result"]] += 1
    stones = {k: int(np.frombuffer(k[0], dtype=np.int8).astype(bool).sum()) for k in brain.planner.memory}
    kept = sorted((k for k in brain.planner.memory if stones[k] <= a.max_stones), key=lambda k: (stones[k], k))[: a.limit]
    merged = len(brain.planner.memory)
    brain.planner.memory = {k: brain.planner.memory[k] for k in kept}
    wins_merged = len(brain.planner.wins)
    order = sorted(brain.planner.wins, key=lambda k: (int(np.frombuffer(k, dtype=np.int8).astype(bool).sum()), -sum(brain.planner.wins[k].values()), k))
    brain.planner.wins = {k: brain.planner.wins[k] for k in order[: a.wins_limit]}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    written = brain.save(a.out)
    receipt = a.stem.parent / "receipt.json"
    if receipt.exists():
        shutil.copy(receipt, a.out.parent / "receipt.json")
    print(f"memory {before} -> {merged} merged -> {len(brain.planner.memory)} kept (<= {a.max_stones} stones, limit {a.limit}); winners' columns on {wins_merged} -> {len(brain.planner.wins)} boards (limit {a.wins_limit}); from {len(a.school)} school files ({games}); wrote {', '.join(w.name for w in written)}")


if __name__ == "__main__":
    main()
