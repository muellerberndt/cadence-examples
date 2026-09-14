"""Learn transposition-invariant harmony/rhythm/interval expectations from MIDI.

Counts include sustained notes in each harmonic window. Rhythm uses inter-onset
intervals from a selected melodic track, not the duration of an arbitrary voice.
Only training families update memory. Held-out relationship likelihoods are a
diagnostic; they do not measure beauty, originality or whole-piece coherence.
"""

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import pairwise
from pathlib import Path

import mido
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from composer.encoding import DURATIONS
from composer.intuition import Intuition

ROOT = Path(__file__).resolve().parents[1]


def observations(meta):
    try:
        path = ROOT / "data" / meta["mid"]
        midi = mido.MidiFile(path)
        tracks = []
        for track in midi.tracks:
            tick = 0
            active = {}
            notes = []
            for m in track:
                tick += m.time
                if m.type == "note_on" and m.velocity and m.channel != 9:
                    active[(m.channel, m.note)] = tick
                elif m.type == "note_off" or m.type == "note_on" and not m.velocity:
                    start = active.pop((m.channel, m.note), None)
                    if start is not None and tick > start:
                        notes.append(
                            (
                                start / midi.ticks_per_beat * 4,
                                tick / midi.ticks_per_beat * 4,
                                m.note - meta["key"],
                            )
                        )
            if notes:
                tracks.append(np.array(notes))
        if not tracks:
            return None
        all_notes = np.concatenate(tracks)
        # A transparent melody-track heuristic, rather than switching voices at every onset.
        candidates = [t for t in tracks if len(t) >= 16]
        if not candidates:
            return None
        melody = max(candidates, key=lambda t: np.median(t[:, 2]))
        onsets = {}
        for start, end, pitch in melody:
            step = round(start)
            if step not in onsets or pitch > onsets[step]:
                onsets[step] = int(pitch)
        times = np.array(sorted(onsets))
        pitches = np.array([onsets[t] for t in times])
        iois = np.diff(times)
        bins = np.argmin(abs(iois[:, None] - DURATIONS), axis=1)
        arousal = 0 if np.median(iois) >= 4 else 2 if np.median(iois) <= 1 else 1
        intervals = np.clip(np.diff(pitches), -12, 12) + 12
        rhythms = [
            (arousal, int(times[i] % 4), int(bins[i - 1]), int(bins[i]))
            for i in range(1, len(bins))
        ]
        pairs = list(zip(intervals[:-1].tolist(), intervals[1:].tolist()))
        # Sixteen sixteenths per bar; overlap weights retain notes sustained into this bar.
        bar_count = min(1024, int(np.ceil(all_notes[:, 1].max() / 16)))
        histogram = np.zeros((bar_count, 12))
        for start, end, pitch in all_notes:
            for bar in range(
                max(0, int(start // 16)), min(bar_count, int(np.ceil(end / 16)))
            ):
                overlap = max(0, min(end, (bar + 1) * 16) - max(start, bar * 16))
                histogram[bar, int(pitch) % 12] += overlap
        templates = np.full((24, 12), -0.5)
        for chord in range(24):
            root = chord % 12
            templates[
                chord, [root, (root + (3 if chord >= 12 else 4)) % 12, (root + 7) % 12]
            ] = 1
        chords = np.argmax(histogram @ templates.T, axis=1)
        valid = histogram.sum(1) > 0
        harmonic = [
            (meta["mode"], int(a), int(b))
            for i, (a, b) in enumerate(pairwise(chords))
            if valid[i] and valid[i + 1]
        ]
        return {
            "split": meta["split"],
            "chords": harmonic[:: max(1, len(harmonic) // 256)],
            "rhythm": rhythms[:: max(1, len(rhythms) // 256)],
            "intervals": pairs[:: max(1, len(pairs) // 256)],
        }
    except Exception:  # noqa: BLE001 — malformed corpus input is counted and quarantined
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    manifest = ROOT / "data/classical/manifest.json"
    pieces = json.loads(manifest.read_text())["pieces"]
    if a.limit:
        pieces = pieces[: a.limit]
    shapes = {"chords": (2, 24, 24), "rhythm": (3, 4, 12, 12), "intervals": (25, 25)}
    train = {k: np.zeros(s) for k, s in shapes.items()}
    held = {
        split: {k: np.zeros(s) for k, s in shapes.items()}
        for split in ["validation", "test"]
    }
    counts = {"train": 0, "validation": 0, "test": 0, "skipped": 0}
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i, r in enumerate(pool.map(observations, pieces, chunksize=8)):
            if r is None:
                counts["skipped"] += 1
                continue
            counts[r["split"]] += 1
            target = train if r["split"] == "train" else held[r["split"]]
            for name in shapes:
                rows = np.array(r[name], dtype=int)
                if len(rows):
                    np.add.at(target[name], tuple(rows.T), 1)
            if (i + 1) % 1000 == 0:
                print(json.dumps({"pieces": i + 1, "counts": counts}), flush=True)
    out = ROOT / "checkpoints/intuition"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "music.npz", **train)
    for split, bucket in held.items():
        np.savez_compressed(out / f"{split}.npz", **bucket)
    evaluation = {}
    for split, bucket in held.items():
        evaluation[split] = {}
        for name, counts_table in bucket.items():
            prob = Intuition.probability(train[name])
            total = counts_table.sum()
            # Conditioning removed: same corpus marginal, so the contrast tests relationships.
            marginal = Intuition.probability(
                train[name].reshape(-1, train[name].shape[-1]).sum(0)
            )
            evaluation[split][name] = {
                "observations": int(total),
                "conditional_nll": float(
                    -(counts_table * np.log(prob)).sum() / max(1, total)
                ),
                "marginal_nll": float(
                    -(counts_table * np.log(marginal)).sum() / max(1, total)
                ),
            }
    receipt = {
        "piece_counts": counts,
        "evaluation": evaluation,
        "learned_associations": sum(t.size for t in train.values()),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "sources": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), ROOT / "composer/intuition.py"]
        },
        "representation": "Aggregated key-relative chord transitions, interval transitions and onset intervals; no complete score stored.",
        "limits": "Melody-track and triad extraction are supplied heuristics; likelihood is not musical quality.",
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
