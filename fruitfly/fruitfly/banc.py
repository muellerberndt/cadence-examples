"""Custody of the BANC connectome (Brain And Nerve Cord, adult female Drosophila, v888).

Three public files, exported by the Lee lab from the BANC release into a public bucket,
carry the custody. Each is pinned to a SHA-256. The derived fixture lives beside the raw
files under ``data/`` (never committed) and a small tracked manifest records every digest.

The dictionary is declared here: one neuron per proofread BANC neuron that is not glia,
trachea, or a non-neuronal object; one signed synapse-class per connected pair at
``MIN_SYNAPSES`` or more synapses; the sign from the presynaptic neuron's predicted
transmitter (acetylcholine excitatory; GABA, glutamate and histamine inhibitory;
the neuromodulators dopamine, octopamine, serotonin and tyramine, and an unknown
transmitter, neutral). Soma positions are the release's root positions in micrometres.
"""

from __future__ import annotations

import csv
import json
import urllib.request
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Final

import numpy as np

__all__ = [
    "SOURCES", "MIN_SYNAPSES", "SIGN_OF", "DEFAULT_ROOT", "MANIFEST_PATH",
    "Neurons", "fetch_sources", "build_fixture", "load_fixture", "verify_sources",
]

_BUCKET: Final = "https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888/"

SOURCES: Final[dict[str, dict[str, str]]] = {
    "meta": {
        "file": "banc_888_meta.feather",
        "url": _BUCKET + "banc_888_meta.feather",
        "sha256": "86ccf5df0c67419f8c5f43e93a7ed38d23a080e9f7fde26737290252f3780098",
        "citation": (
            "Lee lab and the BANC community (2025). The Brain-And-Nerve-Cord connectome of an "
            "adult female Drosophila, release 888: neuron annotations (side, region, class, type, "
            "function, peripheral target, soma position)."
        ),
    },
    "edges": {
        "file": "banc_888_edgelist_simple_v3.feather",
        "url": _BUCKET + "banc_888_edgelist_simple_v3.feather",
        "sha256": "8c296e946f3c69a8c7222f30ad75fa8a98eeb189124fec6df829c9125f4be64b",
        "citation": "BANC release 888, neuron-to-neuron edge list v3: synapse count per connected pair.",
    },
    "transmitters": {
        "file": "banc_888_neurotransmitter_prediction_v2.csv",
        "url": _BUCKET + "banc_888_neurotransmitter_prediction_v2.csv",
        "sha256": "bb0f4afa48a05d90008c6d801e053ff2667fb1ba68e29501c92f49514540da1d",
        "citation": "BANC release 888, per-neuron neurotransmitter predictions v2 (Eckstein et al. 2024 method).",
    },
}

MIN_SYNAPSES: Final = 5
SIGN_OF: Final[dict[str, float]] = {
    "acetylcholine": 1.0, "gaba": -1.0, "glutamate": -1.0, "histamine": -1.0,
    "dopamine": 0.0, "octopamine": 0.0, "serotonin": 0.0, "tyramine": 0.0,
}
EXCLUDED_SUPER_CLASSES: Final = ("glia", "not_a_neuron", "trachea")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT: Final = ROOT / "data" / "banc"
MANIFEST_PATH: Final = Path(__file__).resolve().parent / "fixtures" / "banc_888_manifest.json"
FIXTURE_NAME: Final = "banc_888_min5_v2.npz"

TEXT_FIELDS: Final = (
    "banc_888_id", "side", "region", "super_class", "cell_class", "cell_sub_class",
    "cell_type", "cell_function", "body_part_sensory", "body_part_effector",
    "peripheral_target_type", "nerve", "neuromere", "flow", "neurotransmitter_predicted",
    "hemilineage", "cell_function_detailed",
)


def _digest(path: Path) -> str:
    h = sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_sources(root: Path = DEFAULT_ROOT) -> dict[str, Path]:
    """Download any missing source file and verify every digest."""
    root.mkdir(parents=True, exist_ok=True)
    out = {}
    for key, src in SOURCES.items():
        path = root / src["file"]
        if not path.exists():
            urllib.request.urlretrieve(src["url"], path)
        out[key] = path
    verify_sources(root)
    return out


def verify_sources(root: Path = DEFAULT_ROOT) -> None:
    for key, src in SOURCES.items():
        path = root / src["file"]
        if not path.exists():
            raise FileNotFoundError(f"{key}: {path} is missing; run fetch_sources()")
        got = _digest(path)
        if got != src["sha256"]:
            raise ValueError(f"{key}: digest {got} differs from the pinned {src['sha256']}")


@dataclass(frozen=True)
class Neurons:
    """The kept neurons in fixture order, with their annotations as string arrays."""

    fields: dict[str, np.ndarray]
    position_um: np.ndarray  # (n, 3) float32, NaN where the release has no root position

    @property
    def n(self) -> int:
        return len(self.position_um)

    def where(self, **fields: str | tuple[str, ...]) -> np.ndarray:
        """Indices whose annotation fields equal the given values (a tuple means any of)."""
        keep = np.ones(self.n, dtype=bool)
        for name, value in fields.items():
            column = self.fields[name]
            if isinstance(value, tuple):
                keep &= np.isin(column, value)
            else:
                keep &= column == value
        return np.flatnonzero(keep)

    def contains(self, name: str, needle: str) -> np.ndarray:
        column = self.fields[name]
        return np.flatnonzero(np.char.find(column.astype(str), needle) >= 0)


def build_fixture(root: Path = DEFAULT_ROOT, *, min_synapses: int = MIN_SYNAPSES) -> Path:
    """Derive the fixture from the pinned sources; returns its path and writes the manifest."""
    import pyarrow.feather as feather

    verify_sources(root)
    meta = feather.read_table(root / SOURCES["meta"]["file"]).to_pandas()
    for col in TEXT_FIELDS:
        meta[col] = meta[col].fillna("").astype(str)
    keep = (meta["proofread"] == "TRUE") & ~meta["super_class"].isin(EXCLUDED_SUPER_CLASSES)
    kept = meta.loc[keep].reset_index(drop=True)
    ids = kept["banc_888_id"].to_numpy()
    index_of = {i: k for k, i in enumerate(ids)}

    # transmitters: the meta column is the release's own; the v2 csv is the source it came from
    nt = kept["neurotransmitter_predicted"].to_numpy()
    sign_by_neuron = np.array([SIGN_OF.get(t, 0.0) for t in nt], dtype=np.float32)

    edges = feather.read_table(root / SOURCES["edges"]["file"]).to_pandas()
    edges = edges[edges["count"] >= min_synapses]
    pre = edges["pre"].map(index_of)
    post = edges["post"].map(index_of)
    ok = pre.notna() & post.notna()
    pre = pre[ok].to_numpy().astype(np.int64)
    post = post[ok].to_numpy().astype(np.int64)
    count = edges.loc[ok, "count"].to_numpy().astype(np.int32)
    order = np.lexsort((pre, post))  # sorted by (post, pre) as cadence.Connectome wants
    pre, post, count = pre[order], post[order], count[order]
    sign = sign_by_neuron[pre]

    pos = np.full((len(kept), 3), np.nan, dtype=np.float32)
    for k, text in enumerate(kept["root_position_nm"].fillna("").astype(str)):
        if text:
            pos[k] = [float(v) / 1000.0 for v in text.split(",")]

    arrays: dict[str, Any] = {
        "pre": pre.astype(np.int32), "post": post.astype(np.int32), "count": count, "sign": sign,
        "position_um": pos,
    }
    for col in TEXT_FIELDS:
        arrays["f_" + col] = kept[col].to_numpy().astype(str)
    path = root / FIXTURE_NAME
    np.savez_compressed(path, **arrays)

    manifest = {
        "format": "cadence-fruitfly.banc-fixture/2",
        "fixture": FIXTURE_NAME,
        "fixture_sha256": _digest(path),
        "min_synapses": min_synapses,
        "excluded_super_classes": list(EXCLUDED_SUPER_CLASSES),
        "sign_of": SIGN_OF,
        "neurons": int(len(kept)),
        "synapse_classes": int(len(pre)),
        "synapses": int(count.sum()),
        "neurons_in_release": int(len(meta)),
        "edges_in_release": int(len(edges) + int((~ok).sum())),
        "sources": {k: {"file": v["file"], "sha256": v["sha256"], "url": v["url"]} for k, v in SOURCES.items()},
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def load_fixture(root: Path = DEFAULT_ROOT) -> tuple[Neurons, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The fixture as (neurons, pre, post, count, sign); the digest must match the manifest."""
    path = root / FIXTURE_NAME
    manifest = json.loads(MANIFEST_PATH.read_text())
    if _digest(path) != manifest["fixture_sha256"]:
        raise ValueError("fixture digest differs from the tracked manifest; rebuild it")
    z = np.load(path, allow_pickle=False)
    fields = {k[2:]: z[k] for k in z.files if k.startswith("f_")}
    neurons = Neurons(fields=fields, position_um=z["position_um"])
    return neurons, z["pre"].astype(np.int64), z["post"].astype(np.int64), z["count"].astype(np.float64), z["sign"].astype(np.float64)


if __name__ == "__main__":
    print(build_fixture())
    print(MANIFEST_PATH.read_text())
