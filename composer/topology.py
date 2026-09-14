"""Full owner/seam indexing, with float32 weights for the browser map."""

import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_cached = None


def topology(engine):
    global _cached
    if _cached is not None and _cached[0] is engine:
        return _cached[1]
    w = engine.wiring
    selected = np.concatenate([np.asarray(ids, dtype=int) for ids in w.sets.values()])
    if len(selected) != w.n or len(np.unique(selected)) != w.n:
        raise ValueError("The full map requires disjoint regions covering every owner.")
    inverse = np.empty(w.n, dtype=int)
    inverse[selected] = np.arange(w.n)
    data = np.column_stack((inverse[w.pre], inverse[w.post], engine.weights)).astype(
        "<f4"
    )
    payload = data.tobytes()
    digest = hashlib.sha256(payload + selected.tobytes()).hexdigest()
    folder = ROOT / "runs/topology"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (digest + ".bin")
    if not path.exists():
        path.write_bytes(payload)
    regions = {}
    cursor = 0
    for name, ids in w.sets.items():
        regions[name] = {"start": cursor, "shown": len(ids), "owners": len(ids)}
        cursor += len(ids)
    result = {
        "id": digest,
        "url": f"/graph/{digest}.bin",
        "format": "float32le triples: source, target, weight",
        "owners": w.n,
        "seams": w.edges,
        "owner_ids": selected.tolist(),
        "regions": regions,
    }
    _cached = (engine, result)
    return result


def weight_changes(changes):
    data = np.asarray(changes, dtype="<f4").tobytes()
    digest = hashlib.sha256(data).hexdigest()
    folder = ROOT / "runs/topology"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / (digest + ".bin")).write_bytes(data)
    return f"/graph/{digest}.bin"
