"""Look for exact interval/rhythm windows in a declared training corpus.

This is a bounded copying screen, not an originality or non-memorization proof.
Polyphonic MIDI is projected to the highest newly attacked pitched note per
sixteenth-note onset. Sustained upper voices and inner parts are not compared.
"""

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import pairwise
from pathlib import Path

import mido

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LENGTHS = (8, 16)


def melody(path):
    midi = mido.MidiFile(path)
    notes = {}
    for track in midi.tracks:
        tick = 0
        for event in track:
            tick += event.time
            if event.type == "note_on" and event.velocity and event.channel != 9:
                step = round(tick * 4 / midi.ticks_per_beat)
                notes[step] = max(event.note, notes.get(step, -1))
    return sorted(notes.items())


def windows(events, length, rhythm=True):
    intervals = tuple(b[1] - a[1] for a, b in pairwise(events))
    times = tuple(b[0] - a[0] for a, b in pairwise(events)) if rhythm else ()
    for start in range(len(events) - length + 1):
        pitches = intervals[start : start + length - 1]
        timing = times[start : start + length - 1]
        yield start, (pitches, timing) if rhythm else pitches


QUERIES = {}


def initialize(queries):
    global QUERIES
    QUERIES = queries


def compare(meta):
    try:
        events = melody(ROOT / "data" / meta["mid"])
        hits = []
        for (length, rhythm), index in QUERIES.items():
            found = set()
            for position, signature in windows(events, length, rhythm):
                for query, start in index.get(signature, ()):
                    key = (query, start)
                    if key not in found:
                        hits.append((length, rhythm, query, start, position))
                        found.add(key)
        return meta, hits, None
    except Exception as error:  # noqa: BLE001 — retain failed corpus entries
        return meta, [], type(error).__name__


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pieces", nargs="+", help="Composition JSON or MIDI files")
    parser.add_argument("--manifest", default="data/ensemble/manifest.json")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="runs/similarity-audit.json")
    args = parser.parse_args()
    queries, results = {}, {}
    for filename in args.pieces:
        path = Path(filename)
        if path.suffix == ".json":
            source = json.loads(path.read_text())
            events = [(e["step"], e["pitch"]) for e in source["events"]]
        else:
            events = melody(path)
        name = str(path)
        results[name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "events": len(events),
            "windows": {},
        }
        for length in LENGTHS:
            for rhythm in (True, False):
                index = queries.setdefault((length, rhythm), defaultdict(list))
                for start, signature in windows(events, length, rhythm):
                    index[signature].append((name, start))
                label = f"{length}_notes_" + (
                    "interval_and_rhythm" if rhythm else "interval_only"
                )
                results[name]["windows"][label] = {
                    "total": max(0, len(events) - length + 1),
                    "matched_query_positions": set(),
                    "matching_training_pieces": 0,
                    "examples": [],
                }
    manifest = Path(args.manifest)
    meta = json.loads(manifest.read_text())
    training = [p for p in meta["pieces"] if p["split"] == "train"]
    errors = []
    started = time.monotonic()
    with ProcessPoolExecutor(
        args.workers, initializer=initialize, initargs=(queries,)
    ) as pool:
        for count, (piece, hits, error) in enumerate(
            pool.map(compare, training, chunksize=16), 1
        ):
            if error:
                errors.append({"mid": piece["mid"], "error": error})
            seen = set()
            for length, rhythm, query, start, position in hits:
                label = f"{length}_notes_" + (
                    "interval_and_rhythm" if rhythm else "interval_only"
                )
                record = results[query]["windows"][label]
                record["matched_query_positions"].add(start)
                if (query, label) not in seen:
                    record["matching_training_pieces"] += 1
                    seen.add((query, label))
                    if len(record["examples"]) < 5:
                        record["examples"].append(
                            {
                                "mid": piece["mid"],
                                "song_name": piece.get("song_name"),
                                "composer_name": piece.get("composer_name"),
                                "query_position": start,
                                "source_position": position,
                                "source_midi_sha256": piece.get("midi_sha256"),
                            }
                        )
            if count % 2000 == 0:
                print(json.dumps({"scanned": count, "of": len(training)}), flush=True)
    for result in results.values():
        for record in result["windows"].values():
            record["matched_query_positions"] = sorted(
                record["matched_query_positions"]
            )
    receipt = {
        "results": results,
        "training_pieces": len(training),
        "errors": errors,
        "seconds": time.monotonic() - started,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "method": "8/16-note exact interval windows, with and without exact sixteenth-note onset intervals; transposition invariant; training split only",
        "limits": "Highest-new-attack melody projection only. Common scales and repeated-note patterns can match by chance. Missing, transformed, partial, inner-voice and non-symbolic copying can escape this screen. A clean result does not prove originality or non-memorization.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2))
    print(
        json.dumps(
            {"receipt": str(out), "pieces": len(training), "errors": len(errors)}
        )
    )


if __name__ == "__main__":
    main()
