"""Full neuron/synapse indexing, with float32 weights for the browser map."""

import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_cached = None


def topology(brain, limit=None):
    """Every neuron, and every directed synapse unless ``limit`` is below their number, in
    which case the ``limit`` strongest synapses by |weight| are exported and both counts are
    reported. Activity frames always cover every neuron."""
    global _cached
    if _cached is not None and _cached[0] is brain and _cached[2] == limit:
        return _cached[1]
    w = brain.connectome
    selected = np.concatenate(
        [np.asarray(ids, dtype=int) for ids in w.populations.values()]
    )
    if len(selected) != w.n or len(np.unique(selected)) != w.n:
        raise ValueError(
            "The full map requires disjoint regions covering every neuron."
        )
    inverse = np.empty(w.n, dtype=int)
    inverse[selected] = np.arange(w.n)
    weights = brain.weights
    keep = slice(None)
    if limit is not None and w.synapses > limit:
        keep = np.sort(np.argpartition(np.abs(weights), w.synapses - limit)[w.synapses - limit :])
    data = np.column_stack((inverse[w.pre[keep]], inverse[w.post[keep]], weights[keep])).astype(
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
    for name, ids in w.populations.items():
        regions[name] = {"start": cursor, "shown": len(ids), "neurons": len(ids)}
        cursor += len(ids)
    result = {
        "id": digest,
        "url": f"/graph/{digest}.bin",
        "format": "float32le triples: source, target, weight",
        "neurons": w.n,
        "synapses": int(len(data)),
        "synapses_total": int(w.synapses),
        "neuron_ids": selected.tolist(),
        "regions": regions,
    }
    _cached = (brain, result, limit)
    return result


def weight_changes(changes):
    data = np.asarray(changes, dtype="<f4").tobytes()
    digest = hashlib.sha256(data).hexdigest()
    folder = ROOT / "runs/topology"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / (digest + ".bin")).write_bytes(data)
    return f"/graph/{digest}.bin"
