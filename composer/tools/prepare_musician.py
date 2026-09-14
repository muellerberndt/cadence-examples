"""Prepare whole pieces as event streams for the musician: tokens, causal senses, mood,
and (for design version 3) the bar profiles of every piece.

Each accepted PDMX score becomes one stream: its events in fixed order, the sense row
that describes the moment before each event (computed by the same ``Senses`` the
composer uses when it plays), seven measured mood classes of the whole piece, the
absolute bar of every event and the eight-class profile of every bar
(``composer/form.py``). PDMX uploader license metadata is retained verbatim; it does not
independently establish underlying composition rights. Private research corpus.

    python tools/prepare_musician.py --workers 44 --name musician              # deduplicated PD/CC0 pool
    python tools/prepare_musician.py --workers 44 --name musician-full --pool full   # every PD/CC0 file without a license conflict
    python tools/prepare_musician.py --composers all --name musician-focus     # a named-composer subset
"""

import argparse
import csv
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mido
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from composer.encoding import tonal_center
from composer.form import PLAN_WIDTHS, bar_profiles
from composer.musician import (
    DELTAS,
    DURATIONS,
    SENSE_RAW,
    WINDOW,
    Senses,
    family,
    mood_classes,
)

ROOT = Path(__file__).resolve().parents[1]


def parse(item):
    meta, cap = item
    try:
        path = ROOT / "data" / meta["mid"]
        midi = mido.MidiFile(path)
        notes = []
        tempo = None
        for track in midi.tracks:
            active, program = {}, {}
            tick = 0
            for m in track:
                tick += m.time
                if m.type == "set_tempo" and tempo is None:
                    tempo = mido.tempo2bpm(m.tempo)
                elif m.type == "program_change":
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
        mood = mood_classes(notes, tempo or 120.0, key, mode)
        notes = [
            (s, d, p - key if f != 7 else p, v, f)
            for s, d, p, v, f in notes
            if f == 7 or 24 <= p - key <= 96
        ]
        notes.sort(key=lambda n: (n[0], n[4], -n[2]))
        if len(notes) < 32:
            return None
        tokens = []
        previous = notes[0][0]
        for start, duration, pitch, velocity, fam in notes:
            tokens.append(
                [
                    int(np.clip(pitch - 24, 0, 72)),
                    int(np.argmin(abs(DURATIONS - duration))),
                    int(np.argmin(abs(DELTAS - (start - previous)))),
                    fam,
                    min(7, velocity // 16),
                ]
            )
            previous = start
        tokens = np.asarray(tokens, np.uint8)[:cap]
        total = int(DELTAS[tokens[:, 2]].sum() + DURATIONS[tokens[-1, 1]])
        senses = Senses(total)
        raw = np.empty((len(tokens), SENSE_RAW), np.uint8)
        event_bar = np.empty(len(tokens), np.uint16)
        for i, token in enumerate(tokens):
            raw[i] = senses.raw()
            event_bar[i] = min(senses.bar, 65535)
            senses.observe(token)
        profiles = bar_profiles(tokens, mode=int(mode), bars=int(event_bar.max()) + 1)
        return (
            tokens,
            raw,
            mood,
            event_bar,
            profiles,
            {
                **meta,
                "notes": len(notes),
                "events": int(len(tokens)),
                "bars": int(len(profiles)),
                "key": int(key),
                "mode": int(mode),
                "tempo_bpm": float(tempo or 120.0),
                "mood": mood.tolist(),
                "instrument_families": sorted({int(n[4]) for n in notes}),
                "midi_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            },
        )
    except Exception as error:  # noqa: BLE001 — quarantine malformed external scores
        return {"mid": meta["mid"], "error": type(error).__name__}


FOCUS = {
    "williams": r"john williams",
    "silvestri": r"silvestri",
    "horner": r"james horner",
    "holst": r"holst",
    "wagner": r"wagner",
    "tchaikovsky": r"tchaikovsk|tschaikow|chaikovsk",
}


def selection(limit, composers=None, pool="deduplicated"):
    """The public-domain pool (``deduplicated``: one file per composition family; ``full``:
    every file without a license conflict, arrangements included), or with ``composers``
    every deduplicated score whose composer, artist or title matches one of the FOCUS
    patterns (a private research subset: uploader license labels of arrangements are not a
    rights clearance)."""
    import re

    selected = []
    for row in csv.DictReader((ROOT / "data/PDMX.csv").open()):
        if composers:
            text = " ".join(row.get(k, "") for k in ("composer_name", "artist_name", "title")).lower()
            match = [name for name in composers if re.search(FOCUS[name], text)]
            if not match or row.get("subset:deduplicated", "").lower() != "true":
                continue
            row = {**row, "focus": match[0]}
        else:
            clean = (
                row.get("license") in ("publicdomain", "cc-zero")
                and row.get("license_conflict", "").lower() == "false"
                and row.get("subset:no_license_conflict", "").lower() == "true"
            )
            if not clean or (pool == "deduplicated" and row.get("subset:deduplicated", "").lower() != "true"):
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
                "genres",
                "n_tracks",
                "focus",
            ]
        }
        meta.update(
            family=family_key,
            split="test" if split_id == 0 else "validation" if split_id == 1 else "train",
        )
        selected.append(meta)
    selected.sort(key=lambda r: hashlib.sha256(r["path"].encode()).hexdigest())
    return selected[:limit]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=300000)
    p.add_argument("--per-piece", type=int, default=4096)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--name", default="musician")
    p.add_argument("--pool", choices=("deduplicated", "full"), default="deduplicated")
    p.add_argument("--composers", nargs="*", help="focus subset: names from FOCUS, or 'all'")
    a = p.parse_args()
    composers = list(FOCUS) if a.composers == ["all"] else a.composers
    selected = selection(a.limit, composers, a.pool)
    out = ROOT / "data" / a.name
    out.mkdir(parents=True, exist_ok=True)
    parts = {s: {"tokens": [], "sense": [], "mood": [], "event_bar": [], "bars": [], "pieces": []} for s in ["train", "validation", "test"]}
    errors = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i, r in enumerate(pool.map(parse, ((m, a.per_piece) for m in selected), chunksize=8)):
            if isinstance(r, dict):
                errors.append(r)
            elif r is not None:
                tokens, raw, mood, event_bar, profiles, meta = r
                part = parts[meta["split"]]
                part["tokens"].append(tokens)
                part["sense"].append(raw)
                part["mood"].append(mood)
                part["event_bar"].append(event_bar)
                part["bars"].append(profiles)
                part["pieces"].append(meta)
            if (i + 1) % 2000 == 0:
                print(
                    json.dumps({"parsed": i + 1, "accepted": sum(len(v["pieces"]) for v in parts.values())}),
                    flush=True,
                )
    report = {"splits": {}, "errors": errors, "selected": len(selected), "window": WINDOW, "composers": composers, "pool": a.pool, "plan_widths": list(PLAN_WIDTHS)}
    for split, part in parts.items():
        if not part["pieces"]:
            continue
        folder = out / split
        folder.mkdir(exist_ok=True)
        offsets = np.concatenate([[0], np.cumsum([len(t) for t in part["tokens"]])]).astype(np.int64)
        bar_offsets = np.concatenate([[0], np.cumsum([len(b) for b in part["bars"]])]).astype(np.int64)
        np.save(folder / "tokens.npy", np.concatenate(part["tokens"]))
        np.save(folder / "sense.npy", np.concatenate(part["sense"]))
        np.save(folder / "mood.npy", np.stack(part["mood"]))
        np.save(folder / "offsets.npy", offsets)
        np.save(folder / "event_bar.npy", np.concatenate(part["event_bar"]))
        np.save(folder / "bars.npy", np.concatenate(part["bars"]))
        np.save(folder / "bar_offsets.npy", bar_offsets)
        (folder / "pieces.json").write_text(json.dumps(part["pieces"], separators=(",", ":")))
        returns = np.concatenate(part["bars"])[:, 7] > 0
        report["splits"][split] = {"pieces": len(part["pieces"]), "events": int(offsets[-1]), "bars": int(bar_offsets[-1]), "return_share": float(returns.mean())}
    report["rights_status"] = (
        "PDMX uploader metadata retained; underlying composition rights are not independently "
        "verified. Private research corpus; no public checkpoint release clearance implied."
    )
    report["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["musician_sha256"] = hashlib.sha256((ROOT / "composer/musician.py").read_bytes()).hexdigest()
    report["form_sha256"] = hashlib.sha256((ROOT / "composer/form.py").read_bytes()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(report, separators=(",", ":")))
    print(json.dumps({k: report[k] for k in ["splits", "selected", "pool"]}), flush=True)


if __name__ == "__main__":
    main()
