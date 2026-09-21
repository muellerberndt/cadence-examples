"""Measure what bulk writing does to the record store: the slow readout alone against the slow
readout plus the record read, on positions the patch never saw and on positions it witnessed.

    python connect4/tools/records_drown.py --school runs/connect4/school/s1.npz --heldout runs/connect4/school/test.npz \
        --patch name=path/to/patch.npz ... --out connect4/receipts/records_drown.json

Each patch was trained on the school file with every witnessed position written into its
records. Held-out positions are those of ``--heldout`` that occur nowhere in the school;
witnessed ones are those that do. The receipt records the hashes of the patches and of both
position files, so the numbers are bound to bytes; the files themselves are regenerated from
their seeds by ``school.py`` and ``train.py --write``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from connect4.patch import ValuePatch  # noqa: E402
from connect4.train import keys_of, load, packed  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signs(patch: ValuePatch, keys: np.ndarray, z: np.ndarray) -> dict[str, float]:
    both = [patch.values(map(tuple, keys[s:s + 2048].tolist()), parts=True) for s in range(0, len(keys), 2048)]
    full, slow = np.concatenate([b[0] for b in both]), np.concatenate([b[1] for b in both])
    decided = z != 0
    return {"positions": int(decided.sum()), "slow_readout_alone": float(np.mean(np.sign(slow[decided]) == z[decided])),
            "with_the_record_read": float(np.mean(np.sign(full[decided]) == z[decided]))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--school", type=Path, required=True)
    p.add_argument("--heldout", type=Path, required=True)
    p.add_argument("--patch", nargs="+", required=True, help="name=path, with an optional note after a second =")
    p.add_argument("--sample", type=int, default=30000)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    school, held = load([a.school]), load([a.heldout])
    hk = keys_of(held)
    seen = np.isin(packed(hk), packed(keys_of(school)))
    _, first = np.unique(packed(hk), return_index=True)
    distinct = np.zeros(len(hk), dtype=bool)
    distinct[first] = True
    rng = np.random.default_rng(12345)
    unseen_rows = np.sort(rng.permutation(np.flatnonzero(distinct & ~seen))[: a.sample])
    seen_rows = np.flatnonzero(distinct & seen)
    runs = {}
    for spec in a.patch:
        name, path, *note = spec.split("=")
        patch = ValuePatch.load(path)
        runs[name] = {"note": note[0] if note else "", "patch_sha256": sha(path), "hidden": patch.hidden,
                      "record_cells": patch.net.records.cells, "records_written": int(patch.net.records.writes),
                      "never_seen": signs(patch, hk[unseen_rows], held["z"][unseen_rows]),
                      "witnessed": signs(patch, hk[seen_rows], held["z"][seen_rows])}
        r = runs[name]
        print(f"{name:8s} never seen: slow {r['never_seen']['slow_readout_alone']:.3f} with records {r['never_seen']['with_the_record_read']:.3f} | "
              f"witnessed: slow {r['witnessed']['slow_readout_alone']:.3f} with records {r['witnessed']['with_the_record_read']:.3f}", flush=True)
    sampling = float(np.sqrt(0.8 * 0.2 / min(r["never_seen"]["positions"] for r in runs.values())))
    a.out.write_text(json.dumps({"format": "cadence-examples.connect4.records-drown/1",
                                 "question": "Does writing every schooled position into the record store help the value it reads?",
                                 "measure": "share of decided positions whose winner the value names",
                                 "school": {"file_sha256": sha(a.school), "positions": int(len(school["z"]))},
                                 "heldout": {"file_sha256": sha(a.heldout), "never_seen": int(len(unseen_rows)), "witnessed": int(len(seen_rows))},
                                 "sampling_error_never_seen": sampling, "runs": runs}, indent=1))


if __name__ == "__main__":
    main()
