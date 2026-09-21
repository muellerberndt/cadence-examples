"""Measure how the record store addresses Connect Four positions, from the committed brain alone.

    python connect4/tools/record_address.py --out connect4/receipts/record_address.json

Sibling moves are the afterstates of one position, one per column: the positions a search
chooses between. They differ by one stone, and one stone can decide a game. The tool
measures, for three sizes of record store over the deployed slow parameters:

- the share of active record cells two sibling moves have in common;
- how far one write of "this move loses" moves the written move and its siblings;
- the same with the siblings anchored: the written move is observed together with its
  siblings at their current values, for four sweeps (rate 0, records written in order);
- how many of 1,000 unrelated positions move by more than 0.05.

Positions come from random legal play under a fixed seed, so the receipt is reproduced from
this repository and the pinned library, with no solver and no school file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from cadence import RecordPatchNet

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))
from connect4.game import Position, readings  # noqa: E402
from connect4.patch import ValuePatch  # noqa: E402


def random_position(rng: np.random.Generator, lo: int, hi: int) -> Position:
    while True:
        p = Position()
        for _ in range(int(rng.integers(lo, hi))):
            legal = [c for c in p.legal() if not p.wins(c)]
            if not legal:
                break
            p = p.play(int(rng.choice(legal)))
        else:
            if not p.full:
                return p


def family(rng: np.random.Generator) -> list[Position]:
    while True:
        p = random_position(rng, 4, 20)
        kids = [p.play(c) for c in p.legal() if not p.wins(c)]
        if len(kids) >= 4 and not any(k.full for k in kids):
            return kids


def fresh(source: RecordPatchNet, cells: int, active: int, settle: np.ndarray) -> RecordPatchNet:
    """The deployed slow parameters over an empty store of the given size, its reading statistics settled."""
    net = RecordPatchNet(source.inputs, source.hidden, 1, seed=0, cells=cells, active=active, slowest=source.slowest)
    net.set_parameters(source.parameters())
    for s in range(0, len(settle), 256):
        x = settle[s:s + 256][:, None, :]
        net.reset()
        net.observe(x, net.imagine(x, state=np.zeros((len(x), net.hidden))).slow_output, rate=0.0, write=True)
    assert not np.count_nonzero(net.records.tables["y"])
    return net


def value(net: RecordPatchNet, x: np.ndarray) -> np.ndarray:
    return net.imagine(x[:, None, :], state=np.zeros((len(x), net.hidden))).output[:, 0, 0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--families", type=int, default=30)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    source = ValuePatch.load(HERE / "brain/v1.npz").net
    rng = np.random.default_rng(7)
    settle = readings(random_position(rng, 1, 36).key for _ in range(40 * 256))
    far = readings(random_position(rng, 1, 36).key for _ in range(1000))
    families = [family(rng) for _ in range(a.families)]
    stores = []
    for cells, active in ((4096, 32), (16384, 32), (16384, 8)):
        base = fresh(source, cells, active, settle)
        shared, rows = [], {"naive": [], "anchored": []}
        for kids in families:
            x = readings(k.key for k in kids)
            hidden = base.imagine(x[:, None, :], state=np.zeros((len(x), base.hidden))).hidden
            code = base.records.code(base._readings(x[:, None, :], hidden).reshape(len(x), -1), valued=False)[0] > 0
            shared += [float((code[0] & code[j]).sum() / active) for j in range(1, len(kids))]
            for mode, sweeps in (("naive", 1), ("anchored", 4)):
                net = RecordPatchNet.restore(base.snapshot())
                before, far_before = value(net, x), value(net, far)
                target = before.copy()
                target[0] = -1.0
                take = slice(0, 1) if mode == "naive" else slice(None)
                for _ in range(sweeps):
                    net.reset()
                    net.observe(x[take][:, None, :], target[take][:, None, None], rate=0.0, write=True)
                after, far_after = value(net, x), value(net, far)
                rows[mode].append((after[0] - before[0], float(np.mean(np.abs(after[1:] - before[1:]))), float(np.mean(np.abs(far_after - far_before) > 0.05))))
        entry = {"cells": cells, "active": active, "sibling_cells_shared": float(np.mean(shared))}
        for mode, values in rows.items():
            moved, siblings, unrelated = (float(np.mean([v[k] for v in values])) for k in range(3))
            entry[mode] = {"written_move_changes_by": moved, "siblings_change_by": siblings, "unrelated_positions_moved": unrelated}
        stores.append(entry)
        print(f"cells {cells:6d} active {active:3d}: siblings share {entry['sibling_cells_shared']:.0%} | one write: move {entry['naive']['written_move_changes_by']:+.3f}, siblings {entry['naive']['siblings_change_by']:.3f} | "
              f"anchored, 4 sweeps: move {entry['anchored']['written_move_changes_by']:+.3f}, siblings {entry['anchored']['siblings_change_by']:.3f}, unrelated moved {entry['anchored']['unrelated_positions_moved']:.1%}", flush=True)
    a.out.write_text(json.dumps({"format": "cadence-examples.connect4.record-address/1", "families": a.families, "seed": 7,
                                 "question": "What do two moves that differ by one stone share in the record store, and what does a write to one do to the other?",
                                 "stores": stores}, indent=1))


if __name__ == "__main__":
    main()
