"""The brain as it is deployed: the school's slow parameters, an empty record store, settled reading statistics.

    python connect4/deploy.py --patch runs/connect4/patch/v1/patch.npz --school runs/connect4/school/s1.npz \
        --out runs/connect4/deployed/v1.npz

The school teaches the slow parameters and writes no records (written in bulk they drown: see
the README). The records are for the games the brain plays afterwards. Their running mean and
input norm still have to be settled before the first write, or the first writes would move the
mean and with it every code. They are settled through the library's own calls: positions are
observed with the slow readout's own prediction as the target at rate 0, so the residual that
is written is exactly zero and the table stays empty.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from cadence import RecordPatchNet

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from connect4.game import readings  # noqa: E402
from connect4.patch import ValuePatch  # noqa: E402


def deploy(source: RecordPatchNet, keys: np.ndarray, *, cells: int, active: int, settle: int, seed: int) -> RecordPatchNet:
    net = RecordPatchNet(source.inputs, source.hidden, 1, seed=seed, cells=cells, active=active, slowest=source.slowest)
    net.set_parameters(source.parameters())
    rng = np.random.default_rng(seed)
    for _ in range(settle):
        x = readings(map(tuple, keys[rng.choice(len(keys), 256)].tolist()))[:, None, :]
        net.reset()
        net.observe(x, net.imagine(x, state=np.zeros((256, net.hidden))).slow_output, rate=0.0, write=True)
    net.reset()
    if np.count_nonzero(net.records.tables["y"]):
        raise SystemExit("the record store is not empty after settling")
    return net


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patch", type=Path, required=True)
    p.add_argument("--school", type=Path, required=True)
    p.add_argument("--cells", type=int, default=4096)
    p.add_argument("--active", type=int, default=32)
    p.add_argument("--settle", type=int, default=200, help="batches of 256 witnessed positions")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    d = np.load(a.school)
    net = deploy(ValuePatch.load(a.patch).net, np.stack((d["own"], d["other"]), 1), cells=a.cells, active=a.active, settle=a.settle, seed=a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    net.save(a.out)
    state = net.records.state()
    print(f"deployed {a.out}: records empty, {int(state['seen'])} readings witnessed, input norm {float(net.snapshot()['input_norm']):.3f}")


if __name__ == "__main__":
    main()
