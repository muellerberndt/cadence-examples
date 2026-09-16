#!/usr/bin/env python3
"""Build the S06 composer page from the stage's public bundle: one HTML file with the standard
renderer, the records-cortex view, the style and the page script inlined, and the audio, the
brain frames and the records activity beside it under ``assets/``.

    python composer/web/build_page.py \
        --bundle ../cadence-paper/experiments/experience/s06_composer/public \
        --receipt ../cadence-paper/experiments/experience/s06_composer/runs/acceptance/receipt.json \
        --out composer/index.html

One command rebuilds the page from a regenerated bundle of the same layout
(``s06-public-bundle/1``); nothing about the page is hand-edited between brains.

What the build reads, file by file and field by field, is in ``web/README.md``. In short: it
copies every file the page fetches to ``composer/assets/`` under a flat name, reads each WAV for
its length and its level block by block, reads each frames file to find which frames belong to
which track, takes the gate table from the run receipt, and writes the page with that index
inlined and everything heavy beside it:

    composer/index.html          the page, under --max-bytes (3 MB by default)
    composer/assets/…            the atlas, the WAVs, the frames files
    composer/checkpoints.json    the manifest of those assets, with a sha256 of every file
    composer/receipt.json        the run receipt the gate table comes from

``tools/checkpoint_assets.py prepare composer <release>`` names the release the Pages workflow
downloads them from, and ``verify`` checks what it downloaded against the manifest. The assets
stay out of the repository, as the other stages' checkpoints do.

A page whose assets are beside it has to be served: a page opened from ``file://`` cannot fetch
its neighbours. The source files stay separate for development (serve the repository root and
open ``composer/web/index.html``; page.js then loads the page index named by
``<body data-bundle>``, which ``--page-index`` writes).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import re
import shutil
import struct
import sys
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WEB = ROOT / "web"  # the shared renderer and records view
BUNDLE_FORMAT = "s06-public-bundle/1"
FRAMES_FORMAT = "s06-public-frames/1"
PAGE_FORMAT = "cadence-composer-page/1"
ASSET_FORMAT = "cadence-composer-assets/1"
STAGE = "composer"
BLOCK_MS = 20  # the stage's block clock
TRACKS = ("demonstration", "imitation", "continuation")
TITLES = {
    "a-b-a-form": "A-B-A form",
    "fast-arpeggio": "Fast arpeggio",
    "one-voice-scale": "One voice, a scale",
    "ostinato-in-c": "Ostinato in C",
    "ostinato-in-c-a-third-up": "Ostinato in C, a third up",
    "sparse-answer": "Voices that rest apart",
    "walking-bassline": "Walking bass line",
}

IMPORT_RE = re.compile(r"^\s*import\s[^;]*?\bfrom\s+[\"'][^\"']+[\"'];?[ \t]*$", re.MULTILINE)
EXPORT_RE = re.compile(r"^export\s+(?=(?:const|let|var|function|class|async)\b)", re.MULTILINE)
DECL_RE = re.compile(r"^(?:export\s+)?(?:async\s+)?(?:function\*?|class|const|let|var)\s+([A-Za-z_$][\w$]*)", re.MULTILINE)

# The receipt's predicate and measure names, as the page groups and names them. A name outside
# this map is shown under "controls" with its own name, so every name the run states reaches the page.
GATES: dict[str, tuple[str, str, str]] = {
    # name: (section, label, note)
    "illegal_fields": ("instrument", "fields outside the mask", "the codec's own mask, over the held-out split"),
    "unexpected_silence": ("instrument", "unexpected silence", "blocks that should sound by the envelope model and do not"),
    "reward_mismatch": ("instrument", "reward mismatch", "rewards that do not follow from their own terms"),
    "nonfinite_blocks": ("instrument", "values that are not finite", ""),
    "renderer_faults": ("instrument", "renderer faults", "a note that attacks, is unfiltered and makes no sound"),
    "pitch_within_a_semitone": ("imitation", "pitch within a semitone", "share of the blocks a demonstrated voice sounds"),
    "onset_f1": ("imitation", "onset F1", "attacks paired inside 40 ms"),
    "continuation_surprise_nats": ("continuation", "continuation surprise", "nats per note event under the event model fitted on the training split; the threshold is what the corpus's own continuations cost under it"),
    "continuation_pitch_within_a_semitone": ("continuation", "continuation pitch within a semitone", "against what the piece actually does there"),
    "imitation_margin_over_procedural": ("controls", "margin over the procedural composer", ""),
    "imitation_margin_over_shuffled_pairing": ("controls", "margin over shuffled pairing", ""),
    "imitation_margin_over_born_frozen": ("controls", "margin over a brain frozen at birth", ""),
    "latency_p95_ms": ("latency", "p95 decision", "milliseconds, inside the 20 ms block"),
    "emitted_blocks_cap": ("latency", "blocks emitted", "the compute cap of the packet"),
    "instruction_success": ("instruction", "instruction success", "judged phrases at or above the score the request asks for"),
    "instruction_margin_over_procedural": ("instruction", "margin over the procedural composer", ""),
    "instruction_margin_over_born_frozen": ("instruction", "margin over a brain frozen at birth", ""),
    "instruction_margin_over_scrambled_request": ("instruction", "margin over a scrambled request", ""),
    "theme_recall": ("theme recall", "theme recall", "how much of the first phrase returns in the third"),
    "theme_recall_erasure_loss": ("theme recall", "loss when the phrase store is erased", ""),
    "revision_success": ("revision", "revision success", "revisions that moved the attribute asked for and kept the others"),
}
AT_MOST = {"illegal_fields", "unexpected_silence", "reward_mismatch", "nonfinite_blocks", "renderer_faults",
           "latency_p95_ms", "emitted_blocks_cap", "continuation_surprise_nats"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inline_module(name: str, source: str) -> str:
    """The module's code as a script fragment: static imports dropped, export keywords removed."""
    if re.search(r"^\s*export\s*\{", source, re.MULTILINE) or re.search(r"^\s*export\s+default\b", source, re.MULTILINE):
        raise SystemExit(f"{name}: only `export const/function/class` declarations can be inlined")
    out = IMPORT_RE.sub("", source)
    if re.search(r"^\s*import\s", out, re.MULTILINE):
        raise SystemExit(f"{name}: an import statement could not be inlined (write one-line imports)")
    return f"// ---- {name}\n{EXPORT_RE.sub('', out).strip()}\n"


def check_collisions(modules: list[tuple[str, str]]) -> None:
    seen: dict[str, str] = {}
    for name, source in modules:
        for match in DECL_RE.finditer(source):
            ident = match.group(1)
            if ident in seen and seen[ident] != name:
                raise SystemExit(f"top-level name {ident!r} is declared in both {seen[ident]} and {name}; rename one before inlining")
            seen.setdefault(ident, name)


# ---------------------------------------------------------------------------- the run's gates
def gate_rows(receipt: dict) -> list[dict]:
    """The run's own predicates and open measures as the page's gate rows: the gated set from
    ``acceptance.predicates``, the open set from ``open.measures``. Nothing is computed here; the
    value, the threshold and the verdict are the receipt's."""
    rows: list[dict] = []
    for predicate in (receipt.get("acceptance") or {}).get("predicates", []):
        name = str(predicate.get("name"))
        section, label, note = GATES.get(name, ("controls", name.replace("_", " "), ""))
        rows.append({"section": section, "name": label, "key": name, "value": predicate.get("value"),
                     "threshold": predicate.get("threshold"), "rule": "at most" if name in AT_MOST else "at least",
                     "passed": bool(predicate.get("passed")), "open": False,
                     "note": note or str(predicate.get("aggregation") or "")})
    for measure in (receipt.get("open") or {}).get("measures", []):
        name = str(measure.get("name"))
        section, label, note = GATES.get(name, ("controls", name.replace("_", " "), ""))
        rows.append({"section": section, "name": label, "key": name, "value": measure.get("value"),
                     "threshold": measure.get("threshold"), "rule": "at most" if name in AT_MOST else "at least",
                     "passed": bool(measure.get("met")), "open": True,
                     "note": note or str(measure.get("aggregation") or "")})
    return rows


# ------------------------------------------------------------------------------- the audio
def wav_blocks(path: Path, block_ms: int = BLOCK_MS) -> dict:
    """A WAV's length in blocks, and its level block by block: the RMS over -60 dBFS to 0 as a
    byte, and the peak beside it. This is what the roll draws until the bundle carries the notes
    of each track; it is measured off the file the page plays and nothing else."""
    with wave.open(str(path), "rb") as handle:
        channels, width, rate, count = handle.getnchannels(), handle.getsampwidth(), handle.getframerate(), handle.getnframes()
        raw = handle.readframes(count)
    if width != 2:
        raise SystemExit(f"{path}: {width * 8} bit audio; the page reads 16 bit PCM")
    samples = struct.unpack(f"<{len(raw) // 2}h", raw)
    if channels > 1:
        samples = samples[::channels]
    per = max(1, int(round(rate * block_ms / 1000)))
    blocks = max(1, math.ceil(len(samples) / per))
    level, peak = [], []
    for b in range(blocks):
        window = samples[b * per:(b + 1) * per]
        if not window:
            level.append(0)
            peak.append(0)
            continue
        rms = math.sqrt(sum(float(s) * s for s in window) / len(window)) / 32768.0
        top = max(abs(s) for s in window) / 32768.0
        db = 20 * math.log10(max(rms, 1e-6))
        level.append(max(0, min(255, int(round((db + 60) / 60 * 255)))))
        peak.append(max(0, min(255, int(round(top * 255)))))
    return {"rate": rate, "samples": len(samples), "seconds": len(samples) / rate, "blocks": blocks, "level": level, "peak": peak}


# ------------------------------------------------------------------------------- the frames
def frame_ranges(frames: dict, demonstration_blocks: int, attempt_blocks: int) -> dict:
    """Which frames of one episode belong to which track.

    An episode runs ``listen``, ``gap``, ``attempt``, one frame per block. The attempt frames are
    what the brain played, which is the imitation or the continuation audio. The listen frames are
    the phrase played to it once per voice and then whole, so the last pass, one phrase long, is
    what the demonstration audio carries. A listen phase that is not a whole number of passes
    leaves the demonstration without frames; nothing is guessed.
    """
    phases = [str(block.get("phase")) for block in frames["blocks"]]
    listen = [i for i, phase in enumerate(phases) if phase == "listen"]
    attempt = [i for i, phase in enumerate(phases) if phase == "attempt"]
    out: dict[str, dict] = {}
    if attempt:
        start = attempt[-1] + 1 - attempt_blocks if attempt_blocks and len(attempt) >= attempt_blocks else attempt[0]
        out["attempt"] = {"start": start, "end": attempt[-1] + 1, "phase": "attempt"}
    if listen and demonstration_blocks and len(listen) >= demonstration_blocks and len(listen) % demonstration_blocks == 0:
        out["listen"] = {"start": listen[-1] + 1 - demonstration_blocks, "end": listen[-1] + 1, "phase": "listen",
                         "passes": len(listen) // demonstration_blocks}
    return out


def read_frames(path: Path) -> dict:
    body = json.loads(path.read_text())
    if body.get("format") != FRAMES_FORMAT:
        raise SystemExit(f"{path} is {body.get('format')!r}, not {FRAMES_FORMAT}")
    for key in ("frames", "blocks"):
        if key not in body:
            raise SystemExit(f"{path} carries no {key!r}")
    return body


def decode_u2(spec: dict) -> list[int]:
    data = base64.b64decode(spec["b64"])
    return list(struct.unpack(f"<{len(data) // 2}H", data))


def reading_groups(atlas: dict, reading: list[int]) -> list[dict]:
    """The reading's port groups, from the region each of its neurons belongs to: the records
    view labels the strip under the granule raster with them."""
    region_index = decode_u2(atlas["region"])
    names = [region["name"] for region in atlas["regions"]]
    groups: list[dict] = []
    for position, neuron in enumerate(reading):
        name = names[region_index[neuron]] if neuron < len(region_index) else "?"
        if groups and groups[-1]["name"] == name:
            groups[-1]["end"] = position + 1
        else:
            groups.append({"name": name, "start": position, "end": position + 1})
    return groups


# --------------------------------------------------------------------------------- the build
def load_bundle(folder: Path) -> dict:
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest.get("format") != BUNDLE_FORMAT:
        raise SystemExit(f"{folder / 'manifest.json'} is {manifest.get('format')!r}, not {BUNDLE_FORMAT}")
    for key in ("atlas", "pieces"):
        if key not in manifest:
            raise SystemExit(f"the bundle manifest carries no {key!r}")
    if not (folder / manifest["atlas"]["file"]).exists():
        raise SystemExit(f"the bundle's atlas is missing: {manifest['atlas']['file']}")
    for piece in manifest["pieces"]:
        for name, entry in (piece.get("files") or {}).items():
            if not (folder / entry["path"]).exists():
                raise SystemExit(f"{piece['name']}: the {name} file {entry['path']} is missing")
    return manifest


def build_pieces(folder: Path, manifest: dict, take) -> list[dict]:
    """Every piece, with its audio, its frame ranges and its level lanes, as the page reads it."""
    block_ms = int(manifest.get("block_ms") or BLOCK_MS)
    pieces = []
    for piece in manifest["pieces"]:
        name = piece["name"]
        files = piece.get("files") or {}
        title = TITLES.get(name, name.replace("-", " "))
        audio: dict[str, str] = {}
        level: dict[str, list[int]] = {}
        seconds: dict[str, float] = {}
        blocks: dict[str, int] = {}
        for track in TRACKS:
            entry = files.get(track)
            if not entry:
                continue
            measured = wav_blocks(folder / entry["path"], block_ms)
            audio[track] = take(entry["path"], f"{title}: the {track}, rendered by the emulator")
            level[track] = measured["level"]
            seconds[track] = measured["seconds"]
            blocks[track] = measured["blocks"]
        frames: dict[str, dict] = {}
        for episode, played in (("imitation_frames", "imitation"), ("continuation_frames", "continuation")):
            entry = files.get(episode)
            if not entry:
                continue
            body = read_frames(folder / entry["path"])
            asset = take(entry["path"], f"{title}: the settling frames and the records of the {played} episode")
            ranges = frame_ranges(body, blocks.get("demonstration", 0), blocks.get(played, 0))
            if "attempt" in ranges:
                frames[played] = {"file": asset, **ranges["attempt"]}
            # the demonstration's own brain frames are the last listening pass of the imitation
            # episode: the whole phrase, once, which is what the demonstration audio plays
            if played == "imitation" and "listen" in ranges:
                frames["demonstration"] = {"file": asset, **ranges["listen"]}
            print(f"  {name}/{played}: {body['frames']['steps']} frames, ranges {json.dumps(ranges)}")
        pieces.append({
            "id": name, "title": title, "name": piece.get("piece", name), "split": piece.get("split", ""),
            "family": piece.get("family", ""), "blocks": int(piece.get("blocks") or 0),
            "licence": "public domain", "source": "written for this stage",
            "audio": audio, "frames": frames, "level": level, "seconds": seconds, "track_blocks": blocks,
            "numbers": {k: v for k, v in (piece.get("numbers") or {}).items() if v is not None},
        })
    return pieces


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, required=True, help="the stage's public/ directory (s06-public-bundle/1)")
    parser.add_argument("--out", type=Path, required=True, help="the page to write; the assets go beside it")
    parser.add_argument("--receipt", type=Path, default=None, help="the run receipt the gate table comes from (default: receipt.json inside the bundle)")
    parser.add_argument("--template", type=Path, default=HERE / "index.html")
    parser.add_argument("--brain-scan", type=Path, default=WEB / "brain_scan.js")
    parser.add_argument("--release", default=None, help="the release tag the asset manifest names")
    parser.add_argument("--keep-assets", action="store_true", help="rebuild the page against the files already under assets/: nothing there is copied, removed or rewritten, and the asset manifest is left as it stands")
    parser.add_argument("--page-index", type=Path, default=None, help="also write the page index as JSON, for opening web/index.html over http in development")
    parser.add_argument("--max-bytes", type=int, default=3_000_000, help="the page's own size limit")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    folder = args.bundle.resolve()
    bundle = load_bundle(folder)
    receipt_path = Path(args.receipt) if args.receipt else (folder / "receipt.json")
    if not receipt_path.exists():
        raise SystemExit(f"no receipt: {receipt_path} (pass --receipt)")
    receipt = json.loads(receipt_path.read_text())
    gates = gate_rows(receipt)
    print(f"  gates: {sum(1 for g in gates if not g['open'])} gated and {sum(1 for g in gates if g['open'])} open, from {receipt.get('run_id')}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    assets = args.out.parent / "assets"
    if not args.keep_assets:
        if assets.exists():
            shutil.rmtree(assets)
        assets.mkdir(parents=True)
    elif not assets.exists():
        raise SystemExit(f"--keep-assets: {assets} does not exist; build once without it")
    entries: list[dict] = []
    taken: set[str] = set()
    missing: list[str] = []

    def take(source: str, label: str) -> str:
        """The flat name one bundle file has beside the page. Without --keep-assets the file is
        copied there and listed; with it the file already there is left exactly as it is."""
        flat = "-".join(Path(source).parts)
        while flat in taken:
            flat = f"x-{flat}"
        taken.add(flat)
        if args.keep_assets:
            if not (assets / flat).exists():
                missing.append(flat)
            return flat
        shutil.copy2(folder / source, assets / flat)
        entries.append({"id": Path(flat).name.split(".")[0], "label": label, "file": f"assets/{flat}",
                        "bytes": (assets / flat).stat().st_size, "sha256": sha256_file(assets / flat), "source": source})
        return flat

    atlas_body = json.loads((folder / bundle["atlas"]["file"]).read_text())
    atlas_asset = take(bundle["atlas"]["file"], "the brain's atlas")
    pieces = build_pieces(folder, bundle, take)
    reading = [int(i) for i in bundle["atlas"].get("reading", [])]
    numerics = ((receipt.get("numerics") or {}).get("brain") or {}).get("agent", {}).get("records", {})
    page_manifest = {
        "format": PAGE_FORMAT,
        "stage": bundle.get("stage", "S06"),
        "bundle": {"format": bundle["format"], "source": str(folder), "listening": bundle.get("listening", "")},
        "run": {"run_id": receipt.get("run_id"), "receipt_sha256": receipt.get("receipt_sha256"),
                "seed": (bundle.get("brain") or {}).get("seed"), "checkpoint": (bundle.get("brain") or {}).get("checkpoint"),
                "started_utc": receipt.get("started_utc"), "seeds": len((receipt.get("seeds") or {}).get("completed", [])),
                "emulator_sha256": ((receipt.get("sources") or {}).get("emulator") or {}).get("digest")},
        "clock": {"sample_rate": int(bundle.get("sample_rate_hz") or 22050), "block_ms": BLOCK_MS,
                  "sixteenth_blocks": 6, "bar_blocks": 96, "phrase_blocks": int(bundle.get("phrase_blocks") or 192)},
        "brain": {"atlas": atlas_asset, "neurons": int(bundle["atlas"].get("neurons") or atlas_body.get("n") or 0),
                  "synapses": int(atlas_body.get("synapses") or 0),
                  "reading": reading,
                  "records": {"cells": int(numerics.get("cells") or 0), "active": int(numerics.get("active") or 0),
                              "inputs": len(reading), "rate": float(numerics.get("rate") or 0.2),
                              "reward_rate": float(numerics.get("reward_rate") or 1.0),
                              "groups": reading_groups(atlas_body, reading)}},
        "gates": gates,
        "pieces": pieces,
    }

    modules = [
        ("brain_scan.js", args.brain_scan.read_text()),
        ("records_view.js", (WEB / "records_view.js").read_text()),
        ("page.js", (HERE / "page.js").read_text()),
    ]
    check_collisions(modules)
    bundle_js = "\n".join(inline_module(name, source) for name, source in modules)
    payload = {"manifest": page_manifest, "base": "assets/", "atlas": None}
    encoded = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/").replace("<!--", "<\\!--").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    html = args.template.read_text()
    if '<link rel="stylesheet" href="style.css">' not in html or '<script type="module" src="page.js"></script>' not in html:
        raise SystemExit("the template must link style.css and load page.js as a module")
    html = html.replace('<link rel="stylesheet" href="style.css">', f"<style>\n{(HERE / 'style.css').read_text()}</style>")
    html = html.replace('<script type="module" src="page.js"></script>', f"<script>window.__COMPOSER__ = {encoded};</script>\n<script type=\"module\">\n{bundle_js}</script>")
    html = re.sub(r'\s*data-bundle="[^"]*"', "", html, count=1)
    if args.title:
        html = re.sub(r"<title>.*?</title>", f"<title>{args.title}</title>", html, count=1)
    args.out.write_text(html, encoding="utf-8")
    size = args.out.stat().st_size
    if size > args.max_bytes:
        raise SystemExit(f"the page is {size / 1e6:.2f} MB, over the {args.max_bytes / 1e6:.2f} MB limit")
    if args.page_index:
        args.page_index.parent.mkdir(parents=True, exist_ok=True)
        args.page_index.write_text(json.dumps(page_manifest, separators=(",", ":")), encoding="utf-8")

    if missing:
        raise SystemExit(f"--keep-assets: {len(missing)} files the page needs are not under {assets}: {missing[:4]}")
    if args.keep_assets:
        print(f"page written: {args.out} ({size / 1e6:.2f} MB, {len(modules)} modules inlined, {len(pieces)} pieces)")
        print(f"assets kept: {assets} ({len(taken)} files referenced, none copied or rewritten)")
        return 0

    shutil.copy2(receipt_path, args.out.parent / "receipt.json")
    asset_manifest = {
        "format": ASSET_FORMAT, "page": args.out.name,
        "bundle": {"source": str(folder), "format": bundle["format"],
                   "manifest_sha256": hashlib.sha256((folder / "manifest.json").read_bytes()).hexdigest(),
                   "pieces": len(pieces), "run": receipt.get("run_id")},
        "release": args.release, "checkpoints": entries,
    }
    (args.out.parent / "checkpoints.json").write_text(json.dumps(asset_manifest, indent=1) + "\n", encoding="utf-8")
    total = sum(entry["bytes"] for entry in entries)
    print(f"page written: {args.out} ({size / 1e6:.2f} MB, {len(modules)} modules inlined, {len(pieces)} pieces)")
    print(f"assets written: {assets} ({len(entries)} files, {total / 1e6:.2f} MB)")
    print(f"receipt copied: {args.out.parent / 'receipt.json'} ({receipt.get('run_id')})")
    print(f"asset manifest: {args.out.parent / 'checkpoints.json'}" + (f" (release {args.release})" if args.release else " (no release named yet)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
