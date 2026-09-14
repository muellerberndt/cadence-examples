"""Build the GitHub Pages site: the gallery and the official examples.

    python tools/build_pages.py [--output runs/pages]

Only tracked web assets are copied. Local recordings, models, datasets, caches,
credentials and generated output stay out of the site even in a populated checkout.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 950_000_000  # headroom below the 1 GB GitHub Pages site limit
WEB = ('.html', '.js', '.mjs', '.css', '.json', '.png', '.svg', '.woff2', '.md')


def build(output: Path):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    for name in filter(None, files):
        path = Path(name)
        if any(part.startswith('.') for part in path.parts) or path.parts[0] in ('tests', 'tools'):
            continue
        if path.suffix not in WEB or not (ROOT/path).is_file():
            continue
        target = output/path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/path, target)
    (output/'.nojekyll').touch()
    size = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    if size > LIMIT:
        raise ValueError(f'Pages site exceeds its size budget: {size} bytes')
    print(f'Pages site: {size:,} bytes', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'runs/pages')
    build(parser.parse_args().output)
