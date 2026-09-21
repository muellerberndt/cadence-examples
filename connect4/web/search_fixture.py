"""Positions searched in Python, for the page's engine to reproduce: column, value, depth, reads."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from connect4.brain import Brain  # noqa: E402
from connect4.game import Position  # noqa: E402
from connect4.patch import ValuePatch  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--patch", type=Path, required=True); p.add_argument("--positions", type=int, default=40)
p.add_argument("--budget", type=int, default=400); p.add_argument("--late-stones", type=int, default=24); p.add_argument("--late-budget", type=int, default=3000); p.add_argument("--visits", type=int, default=2000)
p.add_argument("--out", type=Path, required=True); a = p.parse_args()
brain = Brain(ValuePatch.load(a.patch), reads=a.budget, late_stones=a.late_stones, late_reads=a.late_budget, visits=a.visits)
rng, rows = np.random.default_rng(5), []
while len(rows) < a.positions:
    position, columns = Position(), []
    for _ in range(int(rng.integers(0, 34))):
        legal = [c for c in position.legal() if not position.wins(c)]
        if not legal: break
        c = int(rng.choice(legal)); columns.append(c); position = position.play(c)
    if position.full or position.lost or not position.legal(): continue
    t = brain.think(position)
    rows.append({"columns": columns, "column": t.column, "value": t.value, "depth": t.depth, "reads": t.reads, "visits": brain._visits, "proven": t.proven, "line": t.line})
a.out.mkdir(parents=True, exist_ok=True)
(a.out / "search.json").write_text(json.dumps({"budget": a.budget, "late_stones": a.late_stones, "late_budget": a.late_budget, "visits": a.visits, "positions": rows}))
print("searched", len(rows), "positions; proven", sum(r["proven"] for r in rows), "; stopped by the visit bound", sum(r["visits"] >= a.visits for r in rows))
