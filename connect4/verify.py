#!/usr/bin/env python3
"""Recheck the Connect Four example: the receipt, the games in it, and the brain the page plays.

    python connect4/verify.py

- The receipt is bound to the page's brain and engine: their hashes are the recorded ones.
- Every game in the receipt, and in the receipt of the rules-only control, is replayed with the rules: each column was playable, the game
  ended with its last stone and not before, and it ended as recorded. The table of wins,
  draws and losses is recounted from the replays.
- The brain the page loads is the deployed patch in ``brain/``, exported again with the
  library: the same arrays, number for number, and an empty record store.
- The receipt names no path of the machine it was made on, and pins the library's commit.
- The record-drowning receipt holds three runs, and in each the slow readout alone reads better
  than with the record read, on positions never seen and on positions witnessed. Its numbers are
  bound to the hashes of the patches and position files; recomputing them needs those files.
- The record-address receipt holds three store sizes; in each, sibling moves share far more
  cells than chance and anchoring protects them. `tools/record_address.py` recomputes it from
  this directory and the pinned library alone.

Exits non-zero on any failure. The page's arithmetic against the library's is checked by
``web/parity.mjs`` and ``web/search_parity.mjs``.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from connect4.game import Position  # noqa: E402


def replay(columns: str) -> int:
    """The winner of a game given as its columns: 0 the first player, 1 the second, -1 a draw."""
    position = Position()
    for ply, ch in enumerate(columns):
        if position.lost or position.full:
            raise ValueError(f"a stone was played after the game had ended (ply {ply})")
        column = int(ch)
        if not position.playable(column):
            raise ValueError(f"column {column} was full at ply {ply}")
        position = position.play(column)
    if position.lost:
        return (position.plies - 1) % 2
    if position.full:
        return -1
    raise ValueError("the game did not end")


def main() -> int:
    problems: list[str] = []
    receipt = json.loads((HERE / "web/receipt.json").read_text())
    if receipt.get("format") != "cadence-examples.connect4.arena/1":
        problems.append(f"unexpected receipt format {receipt.get('format')!r}")
    for name, expected in receipt.get("bound_to", {}).items():
        if hashlib.sha256((HERE / "web" / name).read_bytes()).hexdigest() != expected:
            problems.append(f"web/{name} is not the file the receipt was measured with")
    if set(receipt.get("bound_to", {})) != {"brain.json", "brain.js", "patch.js"}:
        problems.append("the receipt is not bound to the brain and both engine files")
    text = json.dumps(receipt)
    if "/Users/" in text or "/home/" in text:
        problems.append("the receipt names a path of the machine it was made on")
    if len(str(receipt.get("library", {}).get("commit", ""))) != 40:
        problems.append("the library commit is not pinned")
    games = 0
    control = json.loads((HERE / "receipts/control_rules_only.json").read_text())
    if control.get("brain") is not None or "/Users/" in json.dumps(control):
        problems.append("the control receipt names a brain or a path")
    entries = [(f"control, {name}", entry, control["games_per_opponent"]) for name, entry in control["opponents"].items()]
    entries += [(name, entry, receipt["games_per_opponent"]) for name, entry in receipt["opponents"].items()]
    for opponent, entry, expected_games in entries:
        counted = {"first": [0, 0, 0], "second": [0, 0, 0]}
        for game in entry["games"]:
            games += 1
            try:
                winner = replay(game["columns"])
            except ValueError as error:
                problems.append(f"{opponent}: {game['columns']}: {error}")
                continue
            if winner != game["winner"]:
                problems.append(f"{opponent}: {game['columns']} ended {winner}, recorded {game['winner']}")
            mine = 0 if game["subject_first"] else 1
            counted["first" if game["subject_first"] else "second"][1 if winner == -1 else 0 if winner == mine else 2] += 1
        if counted != entry["win_draw_loss"]:
            problems.append(f"{opponent}: the replays count {counted}, the receipt says {entry['win_draw_loss']}")
        if len(entry["games"]) != expected_games:
            problems.append(f"{opponent}: {len(entry['games'])} games, the receipt says {expected_games}")

    drown = json.loads((HERE / "receipts/records_drown.json").read_text())
    if "/Users/" in json.dumps(drown) or len(drown.get("runs", {})) < 3:
        problems.append("the record-drowning receipt names a path or holds fewer than three runs")
    for name, run in drown.get("runs", {}).items():
        for where in ("never_seen", "witnessed"):
            if not run[where]["slow_readout_alone"] > run[where]["with_the_record_read"]:
                problems.append(f"records_drown, {name}, {where}: the slow readout alone does not read better than with the records")

    address = json.loads((HERE / "receipts/record_address.json").read_text())
    for store in address.get("stores", []):
        where = f"record_address, {store['cells']} cells, {store['active']} active"
        if not store["sibling_cells_shared"] > 10 * store["active"] / store["cells"]:
            problems.append(f"{where}: sibling moves do not share more cells than chance")
        if not store["anchored"]["siblings_change_by"] < store["naive"]["siblings_change_by"]:
            problems.append(f"{where}: anchoring does not protect the sibling moves")
    if len(address.get("stores", [])) < 3 or "/Users/" in json.dumps(address):
        problems.append("the record-address receipt holds fewer than three stores or names a path")

    from connect4.patch import ValuePatch
    from connect4.web.export import brain_state

    patch = ValuePatch.load(HERE / "brain/v2.npz")
    if brain_state(patch) != json.loads((HERE / "web/brain.json").read_text()):
        problems.append("web/brain.json is not the export of brain/v2.npz")
    if any(patch.net.records.tables["y"].reshape(-1)):
        problems.append("the deployed record store is not empty")

    for problem in problems:
        print("FAIL:", problem)
    if not problems:
        print(f"ok: {games} games replayed ({len(receipt['opponents'])} opponents of the brain, {len(control['opponents'])} of the rules-only control); the receipt is bound to the page's brain "
              f"and engine; web/brain.json is the export of brain/v2.npz with an empty record store")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
