"""Build GitHub Pages with the hash-pinned, unlisted flight recording.

Large replay data lives in a prerelease asset, outside ordinary source checkouts.
Only this site's build downloads it. No links are added to the official gallery.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
PREVIEW = Path('previews/1943')
LIMIT = 950_000_000  # headroom below the 1 GB GitHub Pages site limit


def sha256(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def unpack_recording(archive, destination, spec):
    if archive.stat().st_size != spec['bytes'] or sha256(archive) != spec['sha256']:
        raise ValueError('Recording archive does not match its pinned digest and size')
    destination = Path(destination)
    with tarfile.open(archive, 'r:') as tar:
        members = tar.getmembers()
        seen, size = set(), 0
        for member in members:
            path = PurePosixPath(member.name)
            allowed = member.name == 'gameplay.mp4' or (
                len(path.parts) >= 2 and path.parts[0] == 'recording'
                and (path.suffix in ('.json', '.png') or member.name.endswith('.bin.gz'))
            )
            if not member.isfile() or not allowed or path.is_absolute() or '..' in path.parts or member.name in seen:
                raise ValueError('Unsafe or unexpected recording archive entry')
            seen.add(member.name)
            size += member.size
        if size > LIMIT or len(members) != spec['files']:
            raise ValueError('Unexpected recording size or file count')
        for member in members:
            target = destination/member.name
            if target.exists():
                raise ValueError('Recording must not overwrite site source files')
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, target.open('xb') as output:
                shutil.copyfileobj(source, output)
    hashes = json.loads((destination/'recording/sha256.json').read_text())
    for name, expected in hashes.items():
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Unsafe recording checksum path')
        if sha256(destination/'recording'/name) != expected:
            raise ValueError(f'Recording checksum mismatch: {name}')
    manifest_path = destination/'recording/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if sha256(manifest_path) != spec['manifest_sha256'] or not manifest['complete'] or len(manifest['events']) != spec['events'] or len(manifest['frames']) != spec['frames'] or sum(e.get('steps', 0) for e in manifest['events']) != spec['iterations']:
        raise ValueError('Recording chronology differs from the pinned capture')
    return size


def build(output: Path, archive: Path | None = None):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Copy tracked web assets only. This excludes local recordings, ROMs, caches,
    # credentials and generated output even in a developer's populated checkout.
    files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    for name in filter(None, files):
        path = Path(name)
        if any(part.startswith('.') for part in path.parts) or path.parts[0] in ('tests', 'tools'):
            continue
        if path.suffix not in ('.html', '.js', '.mjs', '.css', '.json', '.png', '.svg', '.woff2'):
            continue
        target = output/path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/path, target)
    spec = json.loads((ROOT/PREVIEW/'recording.json').read_text())
    with tempfile.TemporaryDirectory(prefix='cadence-flight-') as temporary:
        if archive is None:
            archive = Path(temporary)/'recording.tar'
            url = spec['url']
            if not url.startswith('https://github.com/muellerberndt/cadence-examples/releases/download/'):
                raise ValueError('Unexpected recording download origin')
            print('Downloading the pinned flight recording…', flush=True)
            with urllib.request.urlopen(url, timeout=120) as response, archive.open('wb') as file:
                shutil.copyfileobj(response, file)
        unpack_recording(archive, output/PREVIEW, spec)
    (output/'.nojekyll').touch()
    size = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    if size > LIMIT:
        raise ValueError(f'Pages site exceeds its size budget: {size} bytes')
    print(f'Pages site: {size:,} bytes; all recording hashes verified', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'runs/pages')
    parser.add_argument('--recording', type=Path, help='Use a local copy of the pinned archive')
    args = parser.parse_args()
    build(args.output, args.recording)
