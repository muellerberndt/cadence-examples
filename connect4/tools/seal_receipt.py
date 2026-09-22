#!/usr/bin/env python3
"""Seal an arena receipt for the page.

    python connect4/tools/seal_receipt.py --arena runs/connect4/web_v2/receipt.json --web connect4/web \
        --alphazero runs/connect_four/alphazero/recipe/best_iter24.pth.tar --library-commit <40 hex> \
        --school-positions 4444571 --passes 6 --unseen 0.9058 --out connect4/web/receipt.json

The arena measures the page's engine and writes the games and grades. Sealing binds the receipt
to the page's brain and engine files by hash, names the AlphaZero checkpoint by its file name and
hash instead of a path, pins the library commit and records the school the brain was taught
with. `verify.py` checks the bindings and that no path of this machine remains.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

RECIPE = "alpha-zero-general, 32 iterations of 100 self-play games at 25 simulations; the model of iteration 24"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arena", type=Path, required=True)
    p.add_argument("--web", type=Path, required=True)
    p.add_argument("--alphazero", type=Path, required=True)
    p.add_argument("--recipe", default=RECIPE)
    p.add_argument("--library-commit", required=True)
    p.add_argument("--school-positions", type=int, required=True)
    p.add_argument("--passes", type=int, required=True)
    p.add_argument("--unseen", type=float, required=True, help="sign accuracy of the deployed slow readout on unseen held-out positions")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if len(a.library_commit) != 40:
        raise SystemExit("--library-commit must be the full 40-character commit")
    r = json.loads(a.arena.read_text())
    stem = a.alphazero.name.split(".")[0]
    r["opponents"] = {(f"alphazero={stem}@{k.split('@')[1]}" if k.startswith("alphazero=") else k): v for k, v in r["opponents"].items()}
    sealed = {"format": r["format"], "subject": r["subject"], "brain": "brain.json", "search": r["search"],
              "games_per_opponent": r["games_per_opponent"],
              "alphazero": {"checkpoint": a.alphazero.name, "sha256": sha(a.alphazero), "recipe": a.recipe},
              "library": {"name": "cadence", "commit": a.library_commit},
              "training": {"school_positions": a.school_positions, "passes": a.passes, "unseen_sign_accuracy": a.unseen, "records_at_deployment": 0},
              "bound_to": {name: sha(a.web / name) for name in ("brain.json", "brain.js", "patch.js")},
              "opponents": r["opponents"]}
    for k, v in r.items():
        sealed.setdefault(k, v)
    text = json.dumps(sealed, indent=1)
    if "/Users/" in text or "/home/" in text:
        raise SystemExit("the receipt still names a path")
    a.out.write_text(text)
    print(f"sealed {a.out}: {len(sealed['opponents'])} opponents, {sealed['games_per_opponent']} games each, bound to {list(sealed['bound_to'])}")


if __name__ == "__main__":
    main()
