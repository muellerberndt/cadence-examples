"""Teach the value patch by day and by night: records written by day, slow parameters taught from dreams by night.

    python connect4/train_sleep.py --school runs/connect4/school/s1.npz --heldout runs/connect4/school/test.npz \
        --hidden 256 --cells 4096 --nights 3 --out runs/connect4/sleep/c4096

A day is one pass over the school's positions, each written once into the records with its
outcome (``write=True`` at slow rate 0): the slow parameters do not move. A night is
``RecordPatchNet.sleep``: the store's completion of every cue is dreamed once, the slow
parameters learn those fixed dreams by ``observe(write=False)``, and at dawn the dreams are
written back so the store holds what the slow parameters did not take. The cues are the
school's readings, drawn as ``train.py`` draws its batches (every band of stones equally
often), each a path of one moment from rest, ``--batch`` rows to a cue. Held-out positions
are split as in ``train.py``; after each night the validation half is scored with the record
store emptied and with it present, the checkpoint is chosen by the store-off score, and the
reported half is scored at the end by that checkpoint and by the last night's patch.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from cadence import RecordPatchNet

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from connect4.game import readings  # noqa: E402
from connect4.patch import ValuePatch, value_target  # noqa: E402
from connect4.train import BANDS, keys_of, load, packed  # noqa: E402


def emptied(patch: ValuePatch) -> ValuePatch:
    """A copy of the patch whose record store holds nothing: the slow readout answers alone."""
    copy = RecordPatchNet.restore(patch.net.snapshot())
    copy.records.tables["y"][:] = 0.0
    return ValuePatch(net=copy)


def metrics(y: np.ndarray, z: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """The scores of ``train.score``, for values ``y``."""
    decided = z != 0
    classes = np.where(y > 0.33, 1, np.where(y < -0.33, -1, 0))
    return {"positions": int(len(y)), "sign": float(np.mean(np.sign(y[decided]) == z[decided])),
            "three_way": float(np.mean(classes == z)), "mse": float(np.mean((y - target) ** 2))}


def values(patch: ValuePatch, keys: np.ndarray, chunk: int) -> tuple[np.ndarray, np.ndarray]:
    both = [patch.values(map(tuple, keys[s:s + chunk].tolist()), parts=True) for s in range(0, len(keys), chunk)]
    return np.concatenate([b[0] for b in both]), np.concatenate([b[1] for b in both])


def score_both(patch: ValuePatch, keys: np.ndarray, z: np.ndarray, target: np.ndarray, chunk: int) -> dict[str, dict]:
    """The scores with the store present (``on``) and with it emptied (``off``). The emptied copy's
    values are checked against the slow part of the present patch's read."""
    if not len(keys):
        return {"on": {"positions": 0}, "off": {"positions": 0}}
    y, slow = values(patch, keys, chunk)
    y_off, _ = values(emptied(patch), keys, chunk)
    if not np.allclose(y_off, slow, atol=1e-9):
        raise SystemExit("the emptied store does not read as the slow readout alone")
    return {"on": metrics(y, z, target), "off": metrics(y_off, z, target)}


def store_state(net: RecordPatchNet) -> dict[str, float]:
    table = net.records.tables["y"]
    return {"cells_nonzero": int(np.count_nonzero(np.any(table != 0.0, axis=1))), "rms": float(np.sqrt(np.mean(table ** 2))),
            "max_abs": float(np.max(np.abs(table))), "writes": int(net.records.writes), "table_mb": table.nbytes / 1e6}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--school", type=Path, nargs="+", required=True)
    p.add_argument("--heldout", type=Path, required=True)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--cells", type=int, default=4096)
    p.add_argument("--active", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--nights", type=int, default=1)
    p.add_argument("--passes", type=int, default=1, help="passes of a night over its fixed dreams; slow updates per night are passes times cues")
    p.add_argument("--rate", type=float, default=0.3, help="the slow rate of a night up to hidden 256, as in train.py")
    p.add_argument("--backtrack", action="store_true")
    p.add_argument("--dawn-passes", type=int, default=2)
    p.add_argument("--batch", type=int, default=256, help="rows in a day's write batch and in a cue")
    p.add_argument("--cues", type=int, default=0, help="rows dreamed in a night (0: one pass, the school's size rounded down to whole cues)")
    p.add_argument("--cue-sampling", choices=("banded", "all"), default="banded",
                   help="banded draws the cues as train.py draws its batches; all dreams every school position once, in random order")
    p.add_argument("--days", choices=("each", "first"), default="each", help="a day before every night, or before the first night only")
    p.add_argument("--shards", type=int, default=0,
                   help="days over consecutive slices of the school, the games in the order they were played: day k writes the k-th slice and "
                        "night k dreams cues drawn from that slice, night k+shards returns to the first; 0 writes the whole school every day")
    p.add_argument("--settle", type=int, default=100, help="batches of readings observed at rate 0 with the slow readout's own prediction before the first day, as deploy.py settles the reading statistics; the table stays empty")
    p.add_argument("--record-rate", type=float, default=0.5, help="the write rate of the records (the patch's default 0.5 makes a cell hold its last few writers)")
    p.add_argument("--record-averaging", action="store_true", help="each cell's write rate is the larger of --record-rate and one over the code mass written into it, so it averages its writers (record_averaging of the library); at 0.5 every cell here is written hundreds of times a day and the floor applies")
    p.add_argument("--sample", type=int, default=20000, help="validation positions scored after each day and night; all reported positions at the end")
    p.add_argument("--chunk", type=int, default=512, help="rows read per call when scoring")
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
    probe = np.random.default_rng(12345)
    shuffled = probe.permutation(unseen_rows)
    validation, unseen_rows = shuffled[: len(shuffled) // 2], np.sort(shuffled[len(shuffled) // 2:])
    quick_unseen = validation[: a.sample]
    quick_seen = probe.choice(seen_rows, size=min(a.sample, len(seen_rows)), replace=False)

    patch = ValuePatch(a.hidden, a.cells, a.active, a.seed, record_rate=a.record_rate, record_averaging=a.record_averaging)
    rate = a.rate * min(1.0, 256.0 / patch.hidden)
    rng = np.random.default_rng(a.seed)
    shards = [np.arange(len(keys))] if a.shards <= 0 else np.array_split(np.arange(len(keys)), a.shards)
    cues = a.cues or (len(shards[0]) // a.batch) * a.batch
    a.out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    def quick(tag: str) -> dict:
        return {"validation": score_both(patch, hkeys[quick_unseen], held["z"][quick_unseen], htarget[quick_unseen], a.chunk),
                "seen": score_both(patch, hkeys[quick_seen], held["z"][quick_seen], htarget[quick_seen], a.chunk),
                "store": store_state(patch.net), "tag": tag, "seconds": time.time() - started}

    for _ in range(a.settle):
        rows = rng.choice(len(keys), a.batch)
        x = readings(map(tuple, keys[rows].tolist()))[:, None, :]
        patch.net.reset()
        patch.net.observe(x, patch.net.imagine(x, state=np.zeros((len(rows), patch.hidden))).slow_output, rate=0.0, write=True)
    patch.net.reset()
    if np.count_nonzero(patch.net.records.tables["y"]):
        raise SystemExit("the record store is not empty after settling")
    curve = [quick("start")]
    v = curve[-1]["validation"]
    print(f"start: validation sign off {v['off']['sign']:.3f} on {v['on']['sign']:.3f}  input norm {float(patch.net.snapshot()['input_norm']):.3f}", flush=True)

    slow_updates, best_validation, kept = 0, -1.0, None
    for night in range(1, a.nights + 1):
        shard = shards[(night - 1) % len(shards)]
        entry: dict = {"night": night, "shard": int((night - 1) % len(shards)), "shard_rows": int(len(shard))}
        if a.days == "each" or night == 1:
            t = time.time()
            order = shard[rng.permutation(len(shard))]
            for s in range(0, len(order), a.batch):
                rows = order[s:s + a.batch]
                patch.learn(map(tuple, keys[rows].tolist()), target[rows], rate=0.0, write=True)
            patch.net.reset()
            entry["day"] = {"rows_written": int(len(order)), "seconds": time.time() - t}
            entry["bedtime"] = quick("bedtime")
            v = entry["bedtime"]["validation"]
            print(f"night {night} day: {len(order)} rows written in {entry['day']['seconds']:.0f} s; "
                  f"validation sign off {v['off']['sign']:.3f} on {v['on']['sign']:.3f}", flush=True)
        t = time.time()
        n_cues = min(cues, (len(shard) // a.batch) * a.batch)
        if a.cue_sampling == "banded":
            rows = shard[rng.choice(len(shard), size=n_cues, p=weight[shard] / weight[shard].sum())]
        else:
            rows = shard[rng.permutation(len(shard))[:n_cues]]
        x = readings(map(tuple, keys[rows].tolist()))
        batches = [x[s:s + a.batch, None, :] for s in range(0, len(x), a.batch)]
        patch.net.reset()
        result = patch.net.sleep(batches, passes=a.passes, rate=rate, backtrack=a.backtrack, dawn_passes=a.dawn_passes)
        if not np.isfinite(result["dream_loss_after"]) or result["dream_loss_after"] > 10.0:
            raise SystemExit(f"the slow readout diverged in night {night} (dream loss {result['dream_loss_after']}); lower --rate")
        slow_updates += int(result["updates"])
        entry["sleep"] = {**result, "cue_batches": len(batches), "rows_dreamed": int(len(x)), "rate": rate, "seconds": time.time() - t}
        entry["slow_updates"] = slow_updates
        entry["dawn"] = quick("dawn")
        curve.append(entry)
        v = entry["dawn"]["validation"]
        if v["off"]["sign"] > best_validation:
            best_validation, kept = v["off"]["sign"], night
            patch.save(a.out / "patch.npz")
        print(f"night {night}: {len(batches)} cues, {int(result['updates'])} updates in {entry['sleep']['seconds']:.0f} s, dream loss "
              f"{result['dream_loss_before']:.4f} -> {result['dream_loss_after']:.4f}; validation sign off {v['off']['sign']:.3f} "
              f"on {v['on']['sign']:.3f}  3-way off {v['off']['three_way']:.3f}  seen sign off {entry['dawn']['seen']['off']['sign']:.3f} "
              f"on {entry['dawn']['seen']['on']['sign']:.3f}{'  kept' if kept == night else ''}", flush=True)
    patch.save(a.out / "patch_last.npz")

    def final(p: ValuePatch) -> dict:
        """The reported half and the seen positions, each row read once with the store present and once emptied;
        the scores by stones on the board are taken from the same reads."""
        y, slow = values(p, hkeys[unseen_rows], a.chunk)
        y_off, _ = values(emptied(p), hkeys[unseen_rows], a.chunk)
        if not np.allclose(y_off, slow, atol=1e-9):
            raise SystemExit("the emptied store does not read as the slow readout alone")
        z, t = held["z"][unseen_rows], htarget[unseen_rows]
        out = {"unseen": {"on": metrics(y, z, t), "off": metrics(y_off, z, t)},
               "seen": score_both(p, hkeys[seen_rows], held["z"][seen_rows], htarget[seen_rows], a.chunk), "unseen_by_stones": {}}
        for lo, hi in BANDS:
            rows = (held["plies"][unseen_rows] >= lo) & (held["plies"][unseen_rows] < hi)
            out["unseen_by_stones"][f"{lo}-{hi}"] = {"on": metrics(y[rows], z[rows], t[rows]), "off": metrics(y_off[rows], z[rows], t[rows])}
        return out

    kept_patch = ValuePatch.load(a.out / "patch.npz")
    kept_final = final(kept_patch)
    last = kept_final if kept == a.nights else final(patch)
    for name, f in (("kept", kept_final), ("last", last)):
        u = f["unseen"]
        print(f"{name}: reported unseen sign off {u['off']['sign']:.4f} on {u['on']['sign']:.4f}  3-way off {u['off']['three_way']:.4f} "
              f"on {u['on']['three_way']:.4f}  mse off {u['off']['mse']:.4f} on {u['on']['mse']:.4f}  seen sign off {f['seen']['off']['sign']:.4f} on {f['seen']['on']['sign']:.4f}", flush=True)
        for b, s in f["unseen_by_stones"].items():
            print(f"  unseen, {b} stones: off {s['off']['sign']:.3f} on {s['on']['sign']:.3f} of {s['off']['positions']}", flush=True)
    (a.out / "training.json").write_text(json.dumps({
        "arguments": {k: str(v) for k, v in vars(a).items()}, "school_positions": int(len(keys)),
        "heldout": {"unseen_validation": int(len(validation)), "unseen_reported": int(len(unseen_rows)), "seen": int(len(seen_rows))},
        "rate": rate, "cues_per_night": cues, "shards": len(shards), "slow_updates": slow_updates, "net_update_counter": int(patch.net.updates),
        "kept_at_night": kept, "validation_sign_off": best_validation, "seconds": time.time() - started,
        "curve": curve, "kept": kept_final, "last": last}, indent=1))


if __name__ == "__main__":
    main()
