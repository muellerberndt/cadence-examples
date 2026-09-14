"""Copy the standard brain-scan renderer shipped with the cadence library into shared/.

    python tools/sync_brain_scan.py [--source PATH]

The renderer is `cadence.brain_scan_script()` from an installed cadence-net that ships it;
`--source` copies a checkout's `src/cadence/brain_scan.js` instead. The copies (the shared demos and the 1943
preview) are committed so the pages need no build step and the site stays static.
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [ROOT / "shared" / "brain_scan.js", ROOT / "previews" / "1943" / "brain_scan.js"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, help="a checkout's src/cadence/brain_scan.js")
    args = parser.parse_args()
    if args.source:
        script = args.source.read_text(encoding="utf-8")
    else:
        import cadence

        script = cadence.brain_scan_script()
    version = next(line for line in script.splitlines() if "VERSION" in line)
    for target in TARGETS:
        target.write_text(script, encoding="utf-8")
        print(f"wrote {target.relative_to(ROOT)} ({len(script)} bytes): {version.strip()}")


if __name__ == "__main__":
    main()
