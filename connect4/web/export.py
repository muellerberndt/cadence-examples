"""Write the value patch for the page, and a parity fixture the page's engine must reproduce.

    python connect4/web/export.py --patch runs/connect4/patch/w256/patch.npz --out runs/connect4/web

``brain.json`` holds the sizes, the record settings and the learned arrays; the projection and
the offsets are not in it, because the page draws them from the seed as the library does.
``parity.json`` holds positions with what ``RecordPatchNet.imagine`` says of them: the value,
the slow readout alone, and the active record cells with their activities.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from connect4.game import Position, readings  # noqa: E402
from connect4.patch import ValuePatch  # noqa: E402


def brain_state(patch: ValuePatch) -> dict:
    net, records = patch.net, patch.net.records
    config, state = records.to_dict(), records.state()
    if config["pathways"] or config["tasks"] or config["fan_in"] or config["averaging"] or config["homeostasis"]:
        raise SystemExit("the page reads the plain record store only")
    parameters = net.parameters()
    return {
        "format": "cadence-examples.connect4.brain/1",
        "sizes": {"inputs": net.inputs, "hidden": net.hidden, "cells": records.cells, "active": records.active,
                  "seed": records.seed, "bias": records.bias, "rate": records.rate,
                  "habituation": records.habituation, "slowest": net.slowest},
        **{k: parameters[k].reshape(-1).tolist() for k in ("G", "g", "B", "b", "C", "c")},
        "mean": state["mean"].tolist(), "table": state["table_y"].reshape(-1).tolist(),
        "seen": int(state["seen"]), "input_norm": float(net.snapshot()["input_norm"]),
    }


def fixture(patch: ValuePatch, count: int, seed: int) -> dict:
    rng, keys = np.random.default_rng(seed), []
    while len(keys) < count:
        position = Position()
        for _ in range(int(rng.integers(1, 38))):
            legal = [c for c in position.legal() if not position.wins(c)]
            if not legal:
                break
            position = position.play(int(rng.choice(legal)))
        if position.plies:
            keys.append(position.key)
    x = readings(keys)
    net = patch.net
    path = net.imagine(x[:, None, :], state=np.zeros((len(x), net.hidden)))
    codes = net.records.code(net._readings(x[:, None, :], path.hidden).reshape(len(x), -1), valued=False)[0]
    rows = []
    for k in range(len(x)):
        cells = np.flatnonzero(codes[k])
        rows.append({"reading": x[k].astype(int).tolist(), "value": float(path.output[k, 0, 0]),
                     "slow": float(path.slow_output[k, 0, 0]), "cells": cells.tolist(), "activity": codes[k][cells].tolist()})
    return {"format": "cadence-examples.connect4.parity/1", "positions": rows}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patch", type=Path, required=True)
    p.add_argument("--positions", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    patch = ValuePatch.load(a.patch)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "brain.json").write_text(json.dumps(brain_state(patch)))
    (a.out / "parity.json").write_text(json.dumps(fixture(patch, a.positions, a.seed)))
    print(f"wrote {a.out / 'brain.json'} ({(a.out / 'brain.json').stat().st_size / 1e6:.2f} MB) and {a.positions} parity positions")


if __name__ == "__main__":
    main()
