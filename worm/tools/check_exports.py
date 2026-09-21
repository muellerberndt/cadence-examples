#!/usr/bin/env python3
"""After a rebuild, compare the exports in the working tree with the committed ones.

Files without computed floats must be identical. Files that carry the library's
arithmetic (the newborn brain, the parity cases, the body cases) differ between
machines in the last digits, through BLAS and libm, so they must have the same
structure and the same numbers to a tolerance.

  python worm/tools/check_exports.py
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXACT = ["worm/data/connectome.json", "worm/web/data/connectome.json", "worm/web/data/params.json"]
CLOSE = ["worm/web/data/brain.json", "worm/tests/parity_cases.json", "worm/tests/body_cases.json"]
TOLERANCE = 1e-6


def committed(path: str) -> bytes:
    return subprocess.run(["git", "show", f"HEAD:{path}"], cwd=ROOT, capture_output=True, check=True).stdout


def number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def compare(a, b, where: str, worst: list) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            raise SystemExit(f"{where}: keys differ: {sorted(a.keys() ^ b.keys())}")
        for k in a:
            compare(a[k], b[k], f"{where}.{k}", worst)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            raise SystemExit(f"{where}: {len(a)} entries committed, {len(b)} rebuilt")
        if a and all(number(x) for x in a) and all(number(x) for x in b):
            scale = max(1e-300, max(abs(x) for x in a), max(abs(x) for x in b))      # a vector is judged at its own scale
            gap = max(abs(x - y) for x, y in zip(a, b)) / scale
            worst[0] = max(worst[0], gap)
            if gap > TOLERANCE:
                raise SystemExit(f"{where}: differs by {gap:.3g} of its scale")
        else:
            for k, (x, y) in enumerate(zip(a, b)):
                compare(x, y, f"{where}[{k}]", worst)
    elif number(a) and number(b):
        gap = abs(a - b) / max(1.0, abs(a), abs(b))
        worst[0] = max(worst[0], gap)
        if gap > TOLERANCE:
            raise SystemExit(f"{where}: {a} committed, {b} rebuilt")
    elif a != b:
        raise SystemExit(f"{where}: {a!r} committed, {b!r} rebuilt")


def main() -> None:
    for path in EXACT:
        if committed(path) != (ROOT / path).read_bytes():
            raise SystemExit(f"{path}: the rebuilt file is not the committed one")
        print(f"{path}: identical")
    for path in CLOSE:
        worst = [0.0]
        compare(json.loads(committed(path)), json.loads((ROOT / path).read_text()), path, worst)
        print(f"{path}: same structure, largest relative difference {worst[0]:.2e}")
    print("EXPORTS OK")


if __name__ == "__main__":
    sys.exit(main())
