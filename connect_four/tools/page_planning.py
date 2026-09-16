"""Re-save a run's checkpoints of one seed with the planning block the page searches with.

    python connect_four/tools/page_planning.py --run runs/connect_four/acceptance_threat --seed 10 \
        --depth 10 --budget 131072 --out runs/connect_four/page_threat

The records, the agent and every counter are copied unchanged; only the planner's extended
depth and budget change, so the page's brain reads the same tables and searches deeper. The
receipt is copied beside the checkpoints for the page builder, which lists them from it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from connect_four.brain import Brain  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--depth", type=int, required=True, help="the page's extended search depth")
    p.add_argument("--budget", type=int, required=True, help="the page's extended budget of imagined transitions")
    p.add_argument("--late-stones", type=int, help="from this many stones on, the extended search goes --late-depth within --late-budget")
    p.add_argument("--late-depth", type=int)
    p.add_argument("--late-budget", type=int)
    p.add_argument("--threat-weight", type=float, help="set the threat summary's weight in the saved brain config")
    p.add_argument("--parity-weight", type=float, help="set the threat rows' weight in the saved brain config")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    receipt = json.loads((a.run / "receipt.json").read_text())
    entries = [e for e in receipt.get("artifacts", {}).get("checkpoints", []) if int(e["seed"]) == a.seed]
    if not entries:
        raise SystemExit(f"{a.run}: the receipt lists no checkpoint of seed {a.seed}")
    a.out.mkdir(parents=True, exist_ok=True)
    for entry in sorted(entries, key=lambda e: int(e["games"])):
        stem = f"brain_seed{a.seed}_games{int(entry['games']):06d}"
        brain = Brain.load(a.run / stem)
        brain.config["planning"] = {**brain.config["planning"], "extended_depth": int(a.depth), "extended_budget": int(a.budget)}
        brain.extended_depth, brain.extended_budget = int(a.depth), int(a.budget)
        if a.late_stones:
            brain.config["planning"].update({"late_stones": int(a.late_stones), "late_depth": int(a.late_depth), "late_budget": int(a.late_budget)})
            brain.late_stones, brain.late_depth, brain.late_budget = int(a.late_stones), int(a.late_depth), int(a.late_budget)
        if a.threat_weight is not None:
            brain.cfg["planner"]["threat_weight"] = brain.imagination.threat_weight = float(a.threat_weight)
        if a.parity_weight is not None:
            brain.cfg["planner"]["parity_weight"] = brain.imagination.parity_weight = float(a.parity_weight)
        written = brain.save(a.out / stem)
        print(f"{stem}: planning {brain.config['planning']}, threat {brain.imagination.threat_weight}, parity {brain.imagination.parity_weight} -> {', '.join(w.name for w in written)}")
    shutil.copy(a.run / "receipt.json", a.out / "receipt.json")
    print(f"receipt copied to {a.out / 'receipt.json'}")


if __name__ == "__main__":
    main()
