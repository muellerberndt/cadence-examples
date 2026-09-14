"""Prepare longer-context, simultaneous multi-instrument events with family-level splits.

PDMX uploader license metadata is retained verbatim. It does not independently
establish underlying composition rights. Artist transcriptions stay in private
research artifacts; this tool does not grant a public checkpoint release license.
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mido
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from composer.encoding import DURATIONS, tonal_center
from composer.ensemble import DELTAS, EXTRA_RAW, HISTORY, family

ROOT = Path(__file__).resolve().parents[1]


def parse(item):
    meta, cap = item
    try:
        path = ROOT / "data" / meta["mid"]
        midi = mido.MidiFile(path)
        notes = []
        for track in midi.tracks:
            active, program = {}, {}
            tick = 0
            for m in track:
                tick += m.time
                if m.type == "program_change":
                    program[m.channel] = m.program
                elif m.type == "note_on" and m.velocity:
                    active[(m.channel, m.note)] = (
                        tick,
                        m.velocity,
                        family(program.get(m.channel, 0), m.channel),
                    )
                elif m.type == "note_off" or m.type == "note_on" and not m.velocity:
                    v = active.pop((m.channel, m.note), None)
                    if v is not None and tick > v[0]:
                        notes.append(
                            (
                                round(v[0] * 4 / midi.ticks_per_beat),
                                max(1, round((tick - v[0]) * 4 / midi.ticks_per_beat)),
                                m.note,
                                v[1],
                                v[2],
                            )
                        )
        if len(notes) < 64:
            return None
        pitched = [n for n in notes if n[4] != 7]
        key, mode = (
            tonal_center([n[2] for n in pitched], [n[1] for n in pitched])
            if pitched
            else (0, 0)
        )
        notes = [
            (s, d, p - key if f != 7 else p, v, f)
            for s, d, p, v, f in notes
            if f == 7 or 24 <= p - key <= 96
        ]
        # Fixed ordering of simultaneous events avoids a training-only random order.
        notes.sort(key=lambda n: (n[0], n[4], -n[2]))
        if len(notes) < 32:
            return None
        style = 1 if len({n[4] for n in notes}) >= 3 else 0
        if "bach" in meta["composer_name"].lower():
            style = 2
        positive = np.diff(sorted({n[0] for n in notes}))
        arousal = (
            0 if np.median(positive) >= 4 else 2 if np.median(positive) <= 1 else 1
        )
        tokens, extras = [], []
        held = []
        pcs = np.zeros(12)
        previous_step = notes[0][0]
        for start, duration, pitch, velocity, fam in notes:
            # Conditioning only uses events already heard; no target instrument/pitch or future onset leaks.
            active_pc = np.zeros((8, 12))
            for end, p, f in held:
                if end > previous_step:
                    active_pc[f, p % 12] += 1
            extra = np.zeros(EXTRA_RAW, np.uint8)
            extra[:5] = [
                mode,
                style,
                arousal,
                previous_step % 16,
                (previous_step // 16) % 16,
            ]
            extra[5:17] = np.round(pcs / max(1, pcs.sum()) * 255).astype(np.uint8)
            extra[17:] = (np.minimum(1, active_pc).ravel() * 255).astype(np.uint8)
            extras.append(extra)
            tokens.append(
                [
                    int(np.clip(pitch - 24, 0, 72)),
                    int(np.argmin(abs(DURATIONS - duration))),
                    int(np.argmin(abs(DELTAS - (start - previous_step)))),
                    fam,
                    min(7, velocity // 16),
                ]
            )
            held = [(e, p, f) for e, p, f in held if e > start]
            held.append((start + duration, pitch, fam))
            pcs *= 0.96
            if fam != 7:
                pcs[pitch % 12] += 1
            previous_step = start
        tokens = np.asarray(tokens, np.uint8)
        selected = np.unique(
            np.linspace(
                HISTORY, len(tokens) - 1, min(cap, len(tokens) - HISTORY), dtype=int
            )
        )
        return (
            np.stack([tokens[i - HISTORY : i] for i in selected]),
            np.asarray(extras)[selected],
            tokens[selected],
            {
                **meta,
                "notes": len(notes),
                "examples": len(selected),
                "instrument_families": sorted({n[4] for n in notes}),
                "midi_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            },
        )
    except Exception as error:  # noqa: BLE001 — quarantine malformed external scores
        return {"mid": meta["mid"], "error": type(error).__name__}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=100000)
    p.add_argument("--per-piece", type=int, default=512)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--name", default="ensemble")
    a = p.parse_args()
    selected = []
    for row in csv.DictReader((ROOT / "data/PDMX.csv").open()):
        if not (
            row.get("license") in ("publicdomain", "cc-zero")
            and row.get("license_conflict", "").lower() == "false"
            and row.get("subset:no_license_conflict", "").lower() == "true"
            and row.get("subset:deduplicated", "").lower() == "true"
        ):
            continue
        family_key = row.get("best_path") or row.get("song_name") or row["path"]
        split_id = int(hashlib.sha256(family_key.encode()).hexdigest()[:8], 16) % 20
        meta = {
            k: row.get(k, "")
            for k in [
                "mid",
                "path",
                "song_name",
                "composer_name",
                "artist_name",
                "license",
                "license_url",
                "best_path",
            ]
        }
        meta.update(
            family=family_key,
            split="test"
            if split_id == 0
            else "validation"
            if split_id == 1
            else "train",
        )
        selected.append(meta)
    selected.sort(key=lambda r: hashlib.sha256(r["path"].encode()).hexdigest())
    selected = selected[: a.limit]
    out = ROOT / "data" / a.name
    out.mkdir(parents=True, exist_ok=True)
    buckets = {s: defaultdict(list) for s in ["train", "validation", "test"]}
    shards = {s: [] for s in buckets}
    pieces, errors = [], []

    def flush(split):
        b = buckets[split]
        if not b["labels"]:
            return
        target = out / split
        target.mkdir(exist_ok=True)
        row = {}
        for k in ["context", "extra", "labels"]:
            file = f"{len(shards[split]):04}-{k}.npy"
            data = np.concatenate(b[k])
            np.save(target / file, data)
            row[k] = file
            row["rows"] = len(data)
        shards[split].append(row)
        b.clear()

    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i, r in enumerate(
            pool.map(parse, ((m, a.per_piece) for m in selected), chunksize=8)
        ):
            if isinstance(r, dict):
                errors.append(r)
            elif r is not None:
                context, extra, labels, meta = r
                pieces.append(meta)
                b = buckets[meta["split"]]
                for k, v in [
                    ("context", context),
                    ("extra", extra),
                    ("labels", labels),
                ]:
                    b[k].append(v)
            if (i + 1) % 1000 == 0:
                for split in buckets:
                    flush(split)
                print(
                    json.dumps({"parsed": i + 1, "accepted": len(pieces)}), flush=True
                )
    for split in buckets:
        flush(split)
    artists = [
        r
        for r in pieces
        if "john williams" in (r["composer_name"] + r["artist_name"]).lower()
        and "thomas" not in (r["composer_name"] + r["artist_name"]).lower()
    ]
    report = {
        "shards": shards,
        "pieces": pieces,
        "errors": errors,
        "selected": len(selected),
        "samples": {s: sum(r["rows"] for r in rows) for s, rows in shards.items()},
        "artist_metadata_records": len(artists),
        "artist_examples": sum(r["examples"] for r in artists),
        "rights_status": "PDMX uploader metadata retained; underlying composition rights are not independently verified. Private research corpus; no public checkpoint release clearance implied.",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out / "manifest.json").write_text(json.dumps(report, separators=(",", ":")))
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "samples",
                    "selected",
                    "artist_metadata_records",
                    "artist_examples",
                ]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
