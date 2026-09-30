"""Teach the deep value patch from watched games, and measure it on positions it has
not seen.

    python connect4/train_deep.py --school runs/connect4/school/big_s0.npz \
        --heldout runs/connect4/school/big_test.npz --layers 96 48 24 --head 4 \
        --passes 1 --out runs/connect4/deep/l96_48_24

Positions are drawn so that every band of stones on the board is witnessed equally often,
as in ``train.py``. Each draw is one ``observe_batch`` admission: private activities, one
shared parameter anchor, all or nothing. A batch the solver refuses is retried once with
twice the sweep budget and otherwise skipped and recorded; a long run of refusals ends
the run, since then the numerical policy is not holding. The unseen held-out positions
are split once: the validation half chooses the checkpoint, the reported half is scored
only at the end, by the checkpoint validation chose. Validation reads go through the
NumPy twin, which is parity-checked against ``Brain.settle`` at every kept checkpoint.
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

from connect4.data import BANDS, keys_of, load, packed  # noqa: E402
from connect4.deep import DeepValuePatch, value_target  # noqa: E402
from connect4.engine import parity  # noqa: E402


def score(patch: DeepValuePatch, keys: np.ndarray, z: np.ndarray, target: np.ndarray) -> dict:
    if not len(keys):
        return {"positions": 0}
    y = np.concatenate([patch.values(map(tuple, keys[s:s + 4096].tolist()))
                        for s in range(0, len(keys), 4096)])
    decided = z != 0
    classes = np.where(y > 0.33, 1, np.where(y < -0.33, -1, 0))
    return {"positions": int(len(keys)), "sign": float(np.mean(np.sign(y[decided]) == z[decided])),
            "three_way": float(np.mean(classes == z)), "mse": float(np.mean((y - target) ** 2))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--school", type=Path, nargs="+", required=True)
    p.add_argument("--heldout", type=Path, required=True)
    p.add_argument("--layers", type=int, nargs="+", default=[96, 48, 24])
    p.add_argument("--head", type=int, default=4)
    p.add_argument("--prior", type=float, default=0.1, help="parameter_prior: how conservatively one batch moves the retained relations")
    p.add_argument("--initial-scale", type=float, default=0.3)
    p.add_argument("--budget", type=int, default=2048, help="settle_budget: accepted repair sweeps per solve")
    p.add_argument("--tolerance", type=float, default=1e-6)
    p.add_argument("--device", type=str, default="cpu", help="python, cpu, mps, cuda (float64 everywhere but mps)")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--passes", type=float, default=1.0, help="witnessed positions, in multiples of the school's size")
    p.add_argument("--report", type=int, default=200, help="batches between measurements")
    p.add_argument("--sample", type=int, default=20000, help="held-out positions scored at each report; all of them at the end")
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
    unseen_rows = np.flatnonzero(distinct & ~seen)
    target, htarget = value_target(school["z"], school["n"]), value_target(held["z"], held["n"])
    band = np.digitize(school["plies"], [hi for _, hi in BANDS[:-1]])
    weight = (1.0 / np.bincount(band, minlength=len(BANDS)))[band]
    weight /= weight.sum()
    print(f"school {len(keys)} positions; held-out unseen {len(unseen_rows)}", flush=True)

    patch = DeepValuePatch.load(a.resume, device=a.device) if a.resume else DeepValuePatch(
        a.layers, a.head, seed=a.seed, parameter_prior=a.prior, initial_scale=a.initial_scale,
        settle_budget=a.budget, tolerance=a.tolerance, device=a.device)
    info = patch.brain.inspect()
    print(f"levels {patch.layers}: {info['patches']} patches, {info['connections']} connections", flush=True)
    rng = np.random.default_rng(a.seed)
    batches = int(a.passes * len(keys) / a.batch)
    a.out.mkdir(parents=True, exist_ok=True)
    curve, refused, streak, started = [], [], 0, time.time()
    probe = np.random.default_rng(12345)
    shuffled = probe.permutation(unseen_rows)
    validation, unseen_rows = shuffled[: len(shuffled) // 2], np.sort(shuffled[len(shuffled) // 2:])
    quick_unseen = validation[: a.sample]
    best_validation, kept = np.inf, None
    admitted = 0
    for k in range(1, batches + 1):
        rows = rng.choice(len(keys), size=a.batch, p=weight)
        drawn = list(map(tuple, keys[rows].tolist()))
        lesson = patch.learn(drawn, target[rows])
        if not lesson["accepted"]:
            lesson = patch.learn(drawn, target[rows], budget=2 * a.budget)
        if not lesson["accepted"]:
            refused.append({"batch": k, "reason": lesson["reason"],
                            "stationarity": lesson["stationarity"]})
            streak += 1
            print(f"{k:7d} batches: refused ({lesson['reason']}, "
                  f"stationarity {lesson['stationarity']:.2e})", flush=True)
            if streak >= 25:
                raise SystemExit(f"{streak} refused batches in a row at batch {k} "
                                 f"({len(refused)} in all); the numerical policy is not holding")
            continue
        streak = 0
        admitted += lesson["batch_size"]
        if k % a.report == 0 or k == batches:
            entry = {"batches": k, "witnessed": admitted, "seconds": time.time() - started,
                     "sweeps": lesson["sweeps"], "refused": len(refused),
                     "validation": score(patch, hkeys[quick_unseen], held["z"][quick_unseen],
                                         htarget[quick_unseen])}
            curve.append(entry)
            v = entry["validation"]
            patch.save(a.out / "last.json")
            if v["mse"] < best_validation:
                best_validation, kept = v["mse"], k
                patch.save(a.out / "patch.json")
                check = parity(patch.brain, patch.twin(),
                               patch.boards(map(tuple, hkeys[quick_unseen[:32]].tolist())))
                entry["parity"] = {"rows": check["rows"], "worst_output": check["worst_output"]}
            print(f"{k:7d} batches {entry['seconds']:6.0f} s  validation sign {v['sign']:.3f} "
                  f"3-way {v['three_way']:.3f} mse {v['mse']:.4f}  sweeps {lesson['sweeps']}"
                  f"{'  kept' if kept == k else ''}", flush=True)
    patch = DeepValuePatch.load(a.out / "patch.json", device=a.device)
    final = {"kept_at_batch": kept, "validation_mse": best_validation,
             "unseen": score(patch, hkeys[unseen_rows], held["z"][unseen_rows], htarget[unseen_rows])}
    print(f"kept batch {kept}: reported unseen {final['unseen']}", flush=True)
    by_band = {}
    for lo, hi in BANDS:
        rows = unseen_rows[(held["plies"][unseen_rows] >= lo) & (held["plies"][unseen_rows] < hi)]
        by_band[f"{lo}-{hi}"] = score(patch, hkeys[rows], held["z"][rows], htarget[rows])
        print(f"  unseen, {lo}-{hi} stones: {by_band[f'{lo}-{hi}']}", flush=True)
    (a.out / "training.json").write_text(json.dumps(
        {"arguments": {k: str(v) for k, v in vars(a).items()}, "curve": curve, "refused": refused,
         "kept": final, "unseen_by_stones": by_band}, indent=1))


if __name__ == "__main__":
    main()
