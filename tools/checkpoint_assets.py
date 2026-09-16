"""The checkpoint snapshots of a stage's page as release assets.

``prepare <stage> <release>`` adds the sha256, the size on disk, the compressed size a page
downloads and the release tag to ``<stage>/checkpoints.json`` and prints the upload command;
the files themselves stay out of the repository. ``verify <stage> <directory>`` checks
downloaded files against the manifest, which the Pages workflow runs after
``gh release download``."""
import gzip
import hashlib
import json
import sys
from pathlib import Path

try:  # the size a brotli-serving host sends; gzip when the encoder is absent
    import brotli
except ImportError:
    brotli = None

ROOT = Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compressed_size(path: Path) -> int:
    """The bytes a page downloads: the file under the encoding a static host serves it with."""
    body = path.read_bytes()
    return len(brotli.compress(body, quality=5)) if brotli is not None else len(gzip.compress(body, 9))


def prepare(stage: str, release: str) -> int:
    manifest_path = ROOT / stage / "checkpoints.json"
    manifest = json.loads(manifest_path.read_text())
    uploads = []
    for entry in manifest["checkpoints"]:
        path = ROOT / stage / entry["file"]
        entry["sha256"] = sha256_file(path)
        entry["bytes"] = path.stat().st_size
        entry["download_bytes"] = compressed_size(path)
        entry["asset"] = f"{stage}-{Path(entry['file']).name}"
        uploads.append(f"{path}#{entry['asset']}")
    manifest["release"] = release
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"gh release upload {release} " + " ".join(f'"{u}"' for u in uploads) + " --clobber")
    return 0


def verify(stage: str, directory: str) -> int:
    manifest = json.loads((ROOT / stage / "checkpoints.json").read_text())
    problems = []
    for entry in manifest["checkpoints"]:
        path = Path(directory) / Path(entry["file"]).name
        if not path.exists():
            problems.append(f"{stage}: {entry['asset']} was not downloaded")
        elif sha256_file(path) != entry["sha256"]:
            problems.append(f"{stage}: {entry['asset']} differs from the manifest")
    for problem in problems:
        print("FAIL:", problem)
    if not problems:
        print(f"{stage}: {len(manifest['checkpoints'])} checkpoints match the manifest")
    return 1 if problems else 0


if __name__ == "__main__":
    command, stage, argument = sys.argv[1:4]
    sys.exit(prepare(stage, argument) if command == "prepare" else verify(stage, argument))
