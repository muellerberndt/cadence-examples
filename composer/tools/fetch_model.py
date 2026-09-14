"""Download a pretrained composer model into ``checkpoints/<name>/``.

    python tools/fetch_model.py [--model maestro-1] [--force]

``models.json`` next to this file lists every published model: the GitHub release
assets (a checkpoint split into parts below the release size limit, its design and its
training receipt), the SHA-256 and size of every asset, and the SHA-256 of the joined
checkpoint. Every byte is verified before the model is used; a mismatch deletes the
download. Only the standard library is needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = Path(__file__).with_name("models.json")
ORIGIN = "https://github.com/muellerberndt/cadence-examples/releases/download/"


def sha256_of(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def download(url: str, target: Path, expected_bytes: int, expected_sha256: str) -> None:
    if not url.startswith(ORIGIN):
        raise ValueError(f"unexpected download origin: {url}")
    if target.exists() and target.stat().st_size == expected_bytes and sha256_of(target) == expected_sha256:
        print(f"  {target.name}: already present and verified", flush=True)
        return
    print(f"  {target.name}: downloading {expected_bytes / 1e6:,.0f} MB", flush=True)
    partial = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as file:
        shutil.copyfileobj(response, file, length=1 << 20)
    if partial.stat().st_size != expected_bytes or sha256_of(partial) != expected_sha256:
        partial.unlink()
        raise ValueError(f"{target.name} does not match its published size or SHA-256")
    partial.replace(target)


def fetch(name: str, force: bool = False) -> Path:
    models = json.loads(SPEC.read_text())
    if name not in models:
        raise SystemExit(f"unknown model {name!r}; published models: {', '.join(models)}")
    spec = models[name]
    folder = ROOT / "checkpoints" / name
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / "brain.npz"
    if checkpoint.exists() and not force and checkpoint.stat().st_size == spec["checkpoint"]["bytes"] and sha256_of(checkpoint) == spec["checkpoint"]["sha256"]:
        print(f"{name}: checkpoint present and verified at {checkpoint}", flush=True)
    else:
        print(f"{name}: {len(spec['parts'])} checkpoint parts", flush=True)
        parts = []
        for part in spec["parts"]:
            target = folder / part["name"]
            download(part["url"], target, part["bytes"], part["sha256"])
            parts.append(target)
        with checkpoint.open("wb") as joined:
            for part in parts:
                with part.open("rb") as source:
                    shutil.copyfileobj(source, joined, length=1 << 20)
        if checkpoint.stat().st_size != spec["checkpoint"]["bytes"] or sha256_of(checkpoint) != spec["checkpoint"]["sha256"]:
            checkpoint.unlink()
            raise ValueError("the joined checkpoint does not match its published SHA-256")
        for part in parts:
            part.unlink()
        print(f"{name}: checkpoint joined and verified at {checkpoint}", flush=True)
    for extra in spec.get("files", []):
        download(extra["url"], folder / extra["name"], extra["bytes"], extra["sha256"])
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="maestro-1")
    parser.add_argument("--force", action="store_true", help="download again even if the checkpoint verifies")
    parser.add_argument("--list", action="store_true", help="list the published models")
    args = parser.parse_args()
    if args.list:
        for name, spec in json.loads(SPEC.read_text()).items():
            print(f"{name}: {spec.get('summary', '')} ({spec['checkpoint']['bytes'] / 1e9:.2f} GB)")
        return
    try:
        fetch(args.model, args.force)
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
