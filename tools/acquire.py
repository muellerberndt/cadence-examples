"""Acquire the pinned PDMX v9 metadata and MIDI archive; verify publisher MD5s."""

import hashlib
import json
import tarfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RECORD = "15571083"


def main():
    DATA.mkdir(exist_ok=True)
    record = json.load(
        urllib.request.urlopen(f"https://zenodo.org/api/records/{RECORD}", timeout=60)
    )
    selected = [f for f in record["files"] if f["key"] in ("PDMX.csv", "mid.tar.gz")]
    for f in selected:
        path = DATA / f["key"]
        expected = f["checksum"].split(":")[1]
        if not path.exists() or hashlib.md5(path.read_bytes()).hexdigest() != expected:
            temporary = path.with_suffix(path.suffix + ".partial")
            request = urllib.request.Request(
                f["links"]["self"],
                headers={"User-Agent": "Cadence-Composer-Research/1.0"},
            )
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                temporary.open("wb") as out,
            ):
                size, stamp = 0, time.monotonic()
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
                    size += len(chunk)
                    if time.monotonic() - stamp > 15:
                        print(
                            f"{path.name}: {size / 1e6:.1f}/{f['size'] / 1e6:.1f} MB",
                            flush=True,
                        )
                        stamp = time.monotonic()
            if hashlib.md5(temporary.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Checksum mismatch: {path.name}")
            temporary.replace(path)
        print(f"Verified {path.name}", flush=True)
    (DATA / "source.json").write_text(
        json.dumps(
            {
                "record": RECORD,
                "doi": record["doi"],
                "files": selected,
                "dataset_url": f"https://zenodo.org/records/{RECORD}",
                "selection_policy": "publicdomain + no_license_conflict + deduplicated; per-piece manifest produced before training",
            },
            indent=2,
        )
        + "\n"
    )
    count = 0
    with tarfile.open(DATA / "mid.tar.gz", "r:gz") as archive:
        for member in archive:
            if not member.isfile() or Path(member.name).suffix.lower() not in (
                ".mid",
                ".midi",
            ):
                continue
            target = (DATA / member.name).resolve()
            if not target.is_relative_to(DATA.resolve()):
                raise ValueError("Unsafe archive path")
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())
            count += 1
    print(f"Extracted {count} MIDI files", flush=True)


if __name__ == "__main__":
    main()
