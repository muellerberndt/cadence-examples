"""Extract training-only Connect Four anchors for persistent browser lessons.

Run after 03_connect_four/train.py; uses the receipt's split seed and saved learner.
"""

import json
import sys
from pathlib import Path

import cadence as cd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / "03_connect_four"
    sys.path.insert(0, str(folder))
    from train import prepare

    body = json.loads((folder / "receipt.json").read_text())["body"]
    x, y, _, _, _ = prepare(body["dataset"]["split_seed"])
    learner = cd.Learner.load(folder / "learner.npz", backend="cpu")
    pick = np.linspace(0, len(y) - 1, min(512, len(y)), dtype=int)
    drives = learner.engine.clamp_levels(
        np.pad(x[pick], ((0, 0), (0, learner.engine.wiring.n - x.shape[1])))
    )
    np.savez_compressed(folder / "rehearsal.npz", x=drives, y=y[pick])
    print(f"Saved {len(pick)} training-only anchors")


if __name__ == "__main__":
    main()
