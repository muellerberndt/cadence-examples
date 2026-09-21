"""Teach the value patch from watched games, and measure it on positions it has not seen.

    python connect4/train.py --school runs/connect4/school/s1.npz runs/connect4/school/s2.npz \
        --heldout runs/connect4/school/test.npz --hidden 256 --passes 2 --out runs/connect4/patch/h256

Positions are drawn so that every band of stones on the board is witnessed equally often:
the school's games spend most of their plies late in the game, where the search reads the
end of the game itself, and the value patch is needed early. Held-out positions are those of
``--heldout`` that occur in no school file; positions of ``--heldout`` that do occur in one
are reported separately as seen, since the records are there for what was witnessed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from connect4.patch import ValuePatch, value_target  # noqa: E402

BANDS = ((0, 8), (8, 16), (16, 24), (24, 43))


def load(files: list[Path]) -> dict[str, np.ndarray]:
    parts = [np.load(f) for f in files]
    return {k: np.concatenate([p[k] for p in parts]) for k in ("own", "other", "z", "n", "plies")}


def keys_of(data: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack((data["own"], data["other"]), axis=1)


def packed(keys: np.ndarray) -> np.ndarray:
    """Each key as one structured value, for set operations."""
    return np.ascontiguousarray(keys).view([("a", keys.dtype), ("b", keys.dtype)]).reshape(-1)


def score(patch: ValuePatch, keys: np.ndarray, z: np.ndarray, target: np.ndarray) -> dict[str, float]:
    if not len(keys):
        return {"positions": 0}
    both = [patch.values(map(tuple, keys[s:s + 2048].tolist()), parts=True) for s in range(0, len(keys), 2048)]
    y, slow = np.concatenate([b[0] for b in both]), np.concatenate([b[1] for b in both])
    decided = z != 0
    classes = np.where(y > 0.33, 1, np.where(y < -0.33, -1, 0))
    return {"positions": int(len(keys)), "sign": float(np.mean(np.sign(y[decided]) == z[decided])),
            "slow_sign": float(np.mean(np.sign(slow[decided]) == z[decided])),
            "three_way": float(np.mean(classes == z)), "mse": float(np.mean((y - target) ** 2))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--school", type=Path, nargs="+", required=True)
    p.add_argument("--heldout", type=Path, required=True)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--cells", type=int, default=4096)
    p.add_argument("--active", type=int, default=32)
    p.add_argument("--rate", type=float, default=0.3, help="the slow rate up to hidden 256; a wider patch steps in proportion smaller")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--passes", type=float, default=1.0, help="witnessed positions, in multiples of the school's size")
    p.add_argument("--report", type=int, default=400, help="batches between measurements")
    p.add_argument("--sample", type=int, default=20000, help="held-out positions scored at each report; all of them at the end")
    p.add_argument("--write", action="store_true", help="also write every witnessed position into the records (measured: it drowns them; the school leaves them empty for the games the brain plays later)")
    p.add_argument("--decay", type=float, default=1.0, help="the rate at the last batch, as a share of the first (1 keeps it fixed). Measured over one pass at hidden 256: 0.1 plateaus at 0.809 where the fixed rate reaches 0.829; the slow readout was never the unstable part")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", type=Path, help="continue from a saved patch instead of starting from a fresh one")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    school, held = load(a.school), load([a.heldout])
    keys, hkeys = keys_of(school), keys_of(held)
    seen = np.isin(packed(hkeys), packed(keys))
    _, first = np.unique(packed(hkeys), return_index=True)
    distinct = np.zeros(len(hkeys), dtype=bool)
    distinct[first] = True
    unseen_rows, seen_rows = np.flatnonzero(distinct & ~seen), np.flatnonzero(distinct & seen)
    target, htarget = value_target(school["z"], school["n"]), value_target(held["z"], held["n"])
    band = np.digitize(school["plies"], [hi for _, hi in BANDS[:-1]])
    weight = (1.0 / np.bincount(band, minlength=len(BANDS)))[band]
    weight /= weight.sum()
    print(f"school {len(keys)} positions; held-out unseen {len(unseen_rows)}, seen {len(seen_rows)}", flush=True)

    patch = ValuePatch.load(a.resume) if a.resume else ValuePatch(a.hidden, a.cells, a.active, a.seed)
    # Measured: 0.3 is stable up to hidden 256 (64 diverges at 1.0) and diverges at 1024, where
    # 0.075 to 0.15 learn. A wider context steps smaller; a narrower one does not step larger.
    rate = a.rate * min(1.0, 256.0 / patch.hidden)
    rng = np.random.default_rng(a.seed)
    batches = int(a.passes * len(keys) / a.batch)
    a.out.mkdir(parents=True, exist_ok=True)
    curve, started = [], time.time()
    # The unseen held-out positions are split once: the validation half chooses the checkpoint,
    # the reported half is scored only at the end, by the checkpoint validation chose.
    probe = np.random.default_rng(12345)
    shuffled = probe.permutation(unseen_rows)
    validation, unseen_rows = shuffled[: len(shuffled) // 2], np.sort(shuffled[len(shuffled) // 2:])
    quick_unseen = validation[: a.sample]
    quick_seen = probe.choice(seen_rows, size=min(a.sample, len(seen_rows)), replace=False)
    best_validation, kept = -1.0, None
    for k in range(1, batches + 1):
        rows = rng.choice(len(keys), size=a.batch, p=weight)
        step = rate * a.decay ** (k / batches)
        seen_loss = patch.learn(map(tuple, keys[rows].tolist()), target[rows], rate=step, write=a.write).prediction.slow_loss
        if seen_loss is None or not seen_loss < 10.0:
            raise SystemExit(f"the slow readout diverged at batch {k} (loss {seen_loss}); lower --rate")
        if k % a.report == 0 or k == batches:
            entry = {"batches": k, "witnessed": k * a.batch, "seconds": time.time() - started, "rate": step,
                     "validation": score(patch, hkeys[quick_unseen], held["z"][quick_unseen], htarget[quick_unseen]),
                     "seen": score(patch, hkeys[quick_seen], held["z"][quick_seen], htarget[quick_seen])}
            curve.append(entry)
            v = entry["validation"]
            chosen_by = v["sign"] if a.write else v["slow_sign"]
            if chosen_by > best_validation:
                best_validation, kept = chosen_by, k
                patch.save(a.out / "patch.npz")
            print(f"{k:7d} batches {entry['seconds']:6.0f} s  validation sign {v['sign']:.3f} (slow alone {v['slow_sign']:.3f}) "
                  f"3-way {v['three_way']:.3f}  seen sign {entry['seen']['sign']:.3f} (slow alone {entry['seen']['slow_sign']:.3f})"
                  f"{'  kept' if kept == k else ''}", flush=True)
    patch = ValuePatch.load(a.out / "patch.npz")
    final = {"kept_at_batch": kept, "validation_sign": best_validation,
             "unseen": score(patch, hkeys[unseen_rows], held["z"][unseen_rows], htarget[unseen_rows]),
             "seen": score(patch, hkeys[seen_rows], held["z"][seen_rows], htarget[seen_rows])}
    print(f"kept batch {kept}: reported unseen {final['unseen']}  seen {final['seen']}", flush=True)
    by_band = {}
    for lo, hi in BANDS:
        rows = unseen_rows[(held["plies"][unseen_rows] >= lo) & (held["plies"][unseen_rows] < hi)]
        by_band[f"{lo}-{hi}"] = score(patch, hkeys[rows], held["z"][rows], htarget[rows])
        print(f"  unseen, {lo}-{hi} stones: {by_band[f'{lo}-{hi}']}", flush=True)
    (a.out / "training.json").write_text(json.dumps({"arguments": {k: str(v) for k, v in vars(a).items()}, "curve": curve, "kept": final, "unseen_by_stones": by_band}, indent=1))


if __name__ == "__main__":
    main()
