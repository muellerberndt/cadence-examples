"""License-filter, family-split and encode real MIDI note events before training."""

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mido
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from composer.encoding import DURATIONS, HISTORY, chord, music_features, tonal_center

ROOT = Path(__file__).resolve().parents[1]


def parse_piece(item):
    path, meta = item
    try:
        midi = mido.MidiFile(path)
        notes = []
        for track in midi.tracks:
            tick = 0
            active = {}
            program = {}
            for m in track:
                tick += m.time
                if m.type == "program_change":
                    program[m.channel] = m.program
                if m.type == "note_on" and m.velocity and m.channel != 9:
                    active[(m.channel, m.note)] = (
                        tick,
                        m.velocity,
                        program.get(m.channel, 0),
                    )
                elif m.type == "note_off" or m.type == "note_on" and not m.velocity:
                    key = (m.channel, m.note)
                    if key in active:
                        start, velocity, prog = active.pop(key)
                        if tick > start:
                            notes.append(
                                (
                                    start / midi.ticks_per_beat,
                                    (tick - start) / midi.ticks_per_beat,
                                    m.note,
                                    velocity,
                                    prog,
                                )
                            )
        if len(notes) < 64:
            return None
        key, mode = tonal_center([n[2] for n in notes], [n[1] for n in notes])
        grouped = defaultdict(list)
        for start, duration, pitch, velocity, program in notes:
            grouped[round(start * 4)].append(
                (round(duration * 4), pitch - key, velocity, program)
            )
        events = []
        times = []
        for step, group in sorted(grouped.items()):
            melody = max(group, key=lambda n: n[1])
            bass = min(group, key=lambda n: n[1])
            if not 36 <= melody[1] <= 96:
                continue
            event = [
                melody[1] - 36,
                int(np.argmin(abs(DURATIONS - max(1, melody[0])))),
                chord([n[1] for n in group]),
                int(np.clip(bass[1] - 24, 0, 48)),
            ]
            events.append(event)
            times.append(step)
        if len(events) < 32:
            return None
        events = np.array(events, dtype=np.uint8)
        stepdiff = np.diff(times)
        density = float(np.median(stepdiff))
        arousal = 2 if density <= 1 else 0 if density >= 4 else 1
        style = 1 if len({n[4] for n in notes}) >= 3 else 0
        if (
            "bach"
            in (meta.get("composer_name", "") + meta.get("artist_name", "")).lower()
        ):
            style = 2
        # Bound per-piece weight so a huge score cannot dominate the corpus.
        selected = np.linspace(
            HISTORY, len(events) - 1, min(1024, len(events) - HISTORY), dtype=int
        )
        contexts = np.stack([events[i - HISTORY : i] for i in selected])
        extras = np.stack(
            [
                music_features(events[:i], times[i], mode, style, arousal)
                for i in selected
            ]
        )
        return (
            contexts,
            extras,
            events[selected],
            {
                **meta,
                "events": len(events),
                "examples": len(selected),
                "key": key,
                "mode": mode,
                "style": style,
                "notes": len(notes),
                "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            },
        )
    except Exception as error:  # noqa: BLE001 — quarantine malformed third-party MIDI files
        return {
            "path": str(path),
            "error": type(error).__name__ + ": " + str(error)[:120],
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=2000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--name", default="pilot")
    a = p.parse_args()
    files = {p.stem: p for p in (ROOT / "data/mid").rglob("*.mid")}
    if not files:
        files = {p.stem: p for p in (ROOT / "data").rglob("*.mid")}
    chosen = []
    reject = defaultdict(int)
    with (ROOT / "data/PDMX.csv").open() as stream:
        for row in csv.DictReader(stream):
            if (
                row.get("license") != "publicdomain"
                or row.get("license_conflict", "").lower() != "false"
                or row.get("subset:no_license_conflict", "").lower() != "true"
            ):
                reject["license"] += 1
                continue
            if row.get("subset:deduplicated", "").lower() != "true":
                reject["duplicate"] += 1
                continue
            title = " ".join(
                row.get(k, "")
                for k in [
                    "composer_name",
                    "artist_name",
                    "song_name",
                    "title",
                    "genres",
                    "tags",
                ]
            ).lower()
            if not (
                "classical" in title
                or any(
                    c in title
                    for c in [
                        "bach",
                        "beethoven",
                        "mozart",
                        "chopin",
                        "debussy",
                        "rachman",
                        "schubert",
                        "tchaikov",
                        "haydn",
                        "schumann",
                        "brahms",
                    ]
                )
            ):
                reject["not_classical"] += 1
                continue
            path = files.get(Path(row.get("mid", "")).stem)
            if not path:
                reject["missing_midi"] += 1
                continue
            family = row.get("best_path") or row.get("song_name") or row["path"]
            digest = hashlib.sha256(family.encode()).hexdigest()
            split = (
                "test"
                if int(digest[:8], 16) % 20 == 0
                else "validation"
                if int(digest[:8], 16) % 20 == 1
                else "train"
            )
            keep = {
                k: row.get(k, "")
                for k in [
                    "path",
                    "mid",
                    "song_name",
                    "composer_name",
                    "artist_name",
                    "license",
                    "license_url",
                    "license_conflict",
                    "best_path",
                    "n_tracks",
                ]
            }
            keep.update(
                family=family,
                split=split,
                order=hashlib.sha256(row["path"].encode()).hexdigest(),
            )
            chosen.append((str(path), keep))
    chosen.sort(key=lambda i: i[1]["order"])
    available = len(chosen)
    chosen = chosen[: a.limit]
    target = ROOT / "data" / a.name
    target.mkdir(exist_ok=True)
    accum = {
        s: {"context": [], "extra": [], "labels": []}
        for s in ["train", "validation", "test"]
    }
    manifest = []
    errors = []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i, result in enumerate(pool.map(parse_piece, chosen, chunksize=8)):
            if isinstance(result, dict):
                errors.append(result)
            elif result is not None:
                c, x, y, meta = result
                bucket = accum[meta["split"]]
                for k, v in [("context", c), ("extra", x), ("labels", y)]:
                    bucket[k].append(v)
                manifest.append(meta)
            if (i + 1) % 100 == 0:
                print(
                    f"Parsed {i + 1}/{len(chosen)}; accepted {len(manifest)}; {time.monotonic() - started:.1f}s",
                    flush=True,
                )
    counts = {}
    for split, bucket in accum.items():
        if not bucket["labels"]:
            continue
        arrays = {k: np.concatenate(v) for k, v in bucket.items()}
        np.savez_compressed(target / f"{split}.npz", **arrays)
        counts[split] = len(arrays["labels"])
    report = {
        "source": json.loads((ROOT / "data/source.json").read_text()),
        "selection": "publicdomain, no license conflict, deduplicated, classical metadata",
        "available_pieces": available,
        "requested_pieces": a.limit,
        "accepted_pieces": len(manifest),
        "samples": counts,
        "rejected": dict(reject),
        "parse_errors": errors,
        "pieces": manifest,
    }
    (target / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in ["available_pieces", "accepted_pieces", "samples", "rejected"]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
