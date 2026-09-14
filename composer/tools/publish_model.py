"""Publish a trained model as GitHub release assets and record it in ``models.json``.

    python tools/publish_model.py --name maestro-1 --folder checkpoints/maestro-1 --tag maestro-1 \
        --summary "the musician, design version 2, PD/CC0 pool" [--part-bytes 1900000000]

The checkpoint ``brain.npz`` is split into parts below the release asset size limit, each
part is uploaded with ``gh release upload`` and deleted right after (so the disk holds one
part at a time), and every other file in the folder (the design, the receipt, a selection
note) is uploaded under ``<name>.<file>``. ``tools/models.json`` receives the entry that
``tools/fetch_model.py`` reads: every asset's URL, size and SHA-256, and the SHA-256 of
the joined checkpoint. Needs the ``gh`` CLI, signed in with write access to the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = Path(__file__).with_name("models.json")
REPO = "muellerberndt/cadence-examples"


def sha256_of(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def gh(*args: str) -> str:
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout


def ensure_release(tag: str, title: str, notes: str) -> None:
    exists = subprocess.run(["gh", "release", "view", tag, "-R", REPO], capture_output=True, text=True).returncode == 0
    if not exists:
        gh("release", "create", tag, "-R", REPO, "--title", title, "--notes", notes)
        print(f"created release {tag}", flush=True)


def upload(tag: str, path: Path) -> dict:
    size, digest = path.stat().st_size, sha256_of(path)
    print(f"  uploading {path.name} ({size / 1e6:,.0f} MB)", flush=True)
    gh("release", "upload", tag, str(path), "-R", REPO, "--clobber")
    return {"asset": path.name, "url": f"https://github.com/{REPO}/releases/download/{tag}/{path.name}", "bytes": size, "sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--folder", required=True, help="the checkpoint folder: brain.npz and its companions")
    parser.add_argument("--tag", required=True, help="the release tag")
    parser.add_argument("--summary", default="")
    parser.add_argument("--part-bytes", type=int, default=1_900_000_000)
    parser.add_argument("--notes", default="Pretrained composer model. Fetch it with `python composer/tools/fetch_model.py`; the data sheet is composer/MODEL_CARD.md.")
    args = parser.parse_args()
    folder = Path(args.folder)
    checkpoint = folder / "brain.npz"
    if not checkpoint.exists():
        raise SystemExit(f"{checkpoint} is missing")
    ensure_release(args.tag, f"Composer model {args.name}", args.notes)
    entry = {
        "summary": args.summary,
        "tag": args.tag,
        "checkpoint": {"bytes": checkpoint.stat().st_size, "sha256": sha256_of(checkpoint)},
        "parts": [],
        "files": [],
    }
    with checkpoint.open("rb") as source:
        index = 0
        while True:
            chunk = source.read(args.part_bytes)
            if not chunk:
                break
            part = folder / f"{args.name}.npz.part{index:02d}"
            part.write_bytes(chunk)
            try:
                record = upload(args.tag, part)
            finally:
                part.unlink()
            entry["parts"].append({"name": part.name, **record})
            index += 1
    for extra in sorted(folder.iterdir()):
        if extra.name == "brain.npz" or extra.name.startswith(args.name + ".npz.part") or extra.suffix == ".part":
            continue
        published = folder / f"{args.name}.{extra.name}"
        published.write_bytes(extra.read_bytes())
        try:
            record = upload(args.tag, published)
        finally:
            published.unlink()
        entry["files"].append({"name": extra.name, **record})
    models = json.loads(SPEC.read_text()) if SPEC.exists() else {}
    models[args.name] = entry
    SPEC.write_text(json.dumps(models, indent=1) + "\n")
    print(json.dumps({"name": args.name, "parts": len(entry["parts"]), "files": [f["name"] for f in entry["files"]], "checkpoint_sha256": entry["checkpoint"]["sha256"]}), flush=True)


if __name__ == "__main__":
    main()
