"""The wiring the worm is born with, read from data/connectome.json."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data"


def params() -> dict:
    return json.loads((DATA / "params.json").read_text())


@dataclass(frozen=True)
class Wiring:
    names: list[str]
    A: np.ndarray        # (post, pre) inherited recurrent weights
    B: np.ndarray        # (neuron, sense)
    C: np.ndarray        # (readout, neuron)
    masks: dict[str, np.ndarray]
    inputs: list[str]
    outputs: list[str]


def load() -> dict:
    return json.loads((DATA / "connectome.json").read_text())


def wiring(p: dict | None = None, lesion: tuple[str, ...] = ()) -> Wiring:
    """Inherited weights: log synapse count, sign from the transmitter's receptor
    class (GABA inhibitory), gap junctions symmetric, then one global scale so
    the recurrent spectral radius is p['radius'], plus a self term for membrane
    persistence. Learning may change every permitted entry, including signs."""
    p = p or params()
    c = load()
    names = [n["name"] for n in c["neurons"]]
    index = {n: i for i, n in enumerate(names)}
    H = len(names)
    A = np.zeros((H, H))
    mA = np.zeros((H, H), bool)
    for pre, post, count, sign, _ in c["chemical"]:
        A[post, pre] += sign * np.log1p(count)
        mA[post, pre] = True
    for a, b, count in c["gap"]:
        A[b, a] += np.log1p(count)
        A[a, b] += np.log1p(count)
        mA[b, a] = mA[a, b] = True
    rho = float(max(abs(np.linalg.eigvals(A))))
    A *= p["radius"] / rho
    if p["persistence"] > 0:
        A[np.diag_indices(H)] += p["persistence"]
        np.fill_diagonal(mA, True)

    inputs, outputs = list(p["inputs"]), list(p["outputs"])
    B = np.zeros((H, len(inputs)))
    for k, sense in enumerate(inputs):
        for cell in p["inputs"][sense]:
            if cell not in lesion:
                B[index[cell], k] = 1.0
    C = np.zeros((len(outputs), H))
    mC = np.zeros_like(C, bool)
    for k, readout in enumerate(outputs):
        for cell in p["outputs"][readout]:
            mC[k, index[cell]] = True
    return Wiring(names, A, B, C, {"A": mA, "B": B != 0, "C": mC}, inputs, outputs)
