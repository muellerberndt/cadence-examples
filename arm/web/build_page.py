#!/usr/bin/env python3
"""Build the self-contained S01 arm page: one HTML file with the snapshot, the standard
renderer, the engine, the records-cortex view, the planner, the arm and the page script
inlined, so it runs from file:// and on GitHub Pages.

    python arm/web/build_page.py \
        --snapshot runs/arm/parity/snapshot.json --out runs/arm/web/index.html

Other points of the same arm's life can travel beside the page as files the visitor loads
from the brain selector:

    python arm/web/build_page.py \
        --snapshot runs/arm/final/snapshot.json --out runs/arm/web/index.html \
        --snapshot-label "the final brain" \
        --checkpoint "after 500 babbling decisions=runs/arm/ck/babble500.json" \
        --checkpoint "after 5,000 babbling decisions=runs/arm/ck/babble5000.json"

Each checkpoint is copied to ``checkpoints/<id>.json`` next to the page and listed in
``checkpoints.json`` with its label, file and size; the page fetches that manifest by
relative URL and falls back to the inlined snapshot alone when it is not there (a page
opened from file:// cannot fetch its neighbours).

Any snapshot ``agent/web.py`` exports for the arm is accepted. When it carries no
``extra.planner`` or ``extra.arm`` (an export written without them), both are taken from
``--config``, so the page always knows the planner and the body it runs.

The source files stay separate for development (open index.html over http from the
repository root; page.js then fetches the snapshot named by ``<body data-snapshot>``). The
renderer is the verbatim copy of cadence's brain_scan.js in web/; ``--brain-scan`` names
another copy (the cadence source tree, say).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEB = HERE.parents[1] / "web"  # the shared engine, renderer and records view
SNAPSHOT_FORMAT = "cadence-experience-web/1"
CHECKPOINT_FORMAT = "cadence-arm-checkpoints/1"

IMPORT_RE = re.compile(r"^\s*import\s[^;]*?\bfrom\s+[\"'][^\"']+[\"'];?[ \t]*$", re.MULTILINE)
EXPORT_RE = re.compile(r"^export\s+(?=(?:const|let|var|function|class|async)\b)", re.MULTILINE)
DECL_RE = re.compile(r"^(?:export\s+)?(?:async\s+)?(?:function\*?|class|const|let|var)\s+([A-Za-z_$][\w$]*)", re.MULTILINE)


def inline_module(name: str, source: str) -> str:
    """The module's code as a script fragment: static imports dropped, export keywords removed."""
    if re.search(r"^\s*export\s*\{", source, re.MULTILINE) or re.search(r"^\s*export\s+default\b", source, re.MULTILINE):
        raise SystemExit(f"{name}: only `export const/function/class` declarations can be inlined")
    out = IMPORT_RE.sub("", source)
    if re.search(r"^\s*import\s", out, re.MULTILINE):
        raise SystemExit(f"{name}: an import statement could not be inlined (write one-line imports)")
    out = EXPORT_RE.sub("", out)
    return f"// ---- {name}\n{out.strip()}\n"


def check_collisions(modules: list[tuple[str, str]]) -> None:
    seen: dict[str, str] = {}
    for name, source in modules:
        for match in DECL_RE.finditer(source):
            ident = match.group(1)
            if ident in seen and seen[ident] != name:
                raise SystemExit(f"top-level name {ident!r} is declared in both {seen[ident]} and {name}; rename one before inlining")
            seen.setdefault(ident, name)


def load_snapshot(path: Path) -> dict:
    snapshot = json.loads(path.read_text())
    if snapshot.get("format") != SNAPSHOT_FORMAT:
        raise SystemExit(f"{path} is not a {SNAPSHOT_FORMAT} export")
    return snapshot


def fill_extras(snapshot: dict, config: dict | None, path: Path) -> dict:
    """The page needs the planner settings and the body; a snapshot without them takes the config's."""
    extra = snapshot.setdefault("extra", {})
    for key in ("planner", "arm"):
        if extra.get(key) is None:
            if config is None or key not in config:
                raise SystemExit(f"{path} carries no extra.{key} and the config has none either; pass --config")
            extra[key] = config[key]
            print(f"  {path.name}: extra.{key} taken from the config")
    return snapshot


def slug(label: str, taken: set[str]) -> str:
    base = re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", label.lower())).strip("-") or "brain"
    name, k = base, 2
    while name in taken:
        name, k = f"{base}-{k}", k + 1
    taken.add(name)
    return name


def decisions_of(snapshot: dict) -> int:
    return int((snapshot.get("ledger") or {}).get("real_transitions") or 0)


def write_checkpoints(pairs: list[str], out: Path, config: dict | None, inlined: dict, inlined_label: str, inlined_source: Path) -> list[dict]:
    """Copy every checkpoint snapshot next to the page and write the manifest the page fetches."""
    entries: list[dict] = []
    taken: set[str] = set()
    folder = out.parent / "checkpoints"
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--checkpoint takes label=path, not {pair!r}")
        label, _, raw = pair.partition("=")
        label, source = label.strip(), Path(raw.strip())
        snapshot = fill_extras(load_snapshot(source), config, source)
        folder.mkdir(parents=True, exist_ok=True)
        name = f"{slug(label, taken)}.json"
        target = folder / name
        target.write_text(json.dumps(snapshot, separators=(",", ":")), encoding="utf-8")
        entries.append({"id": target.stem, "label": label, "file": f"checkpoints/{name}", "bytes": target.stat().st_size, "decisions": decisions_of(snapshot), "source": str(source)})
        print(f"  checkpoint {label!r}: {target} ({target.stat().st_size / 1e6:.2f} MB, {entries[-1]['decisions']:,} decisions)")
    manifest = {
        "format": CHECKPOINT_FORMAT,
        "page": out.name,
        "inlined": {"id": "inlined", "label": inlined_label, "decisions": decisions_of(inlined), "bytes": inlined_source.stat().st_size, "source": str(inlined_source)},
        "checkpoints": entries,
    }
    (out.parent / "checkpoints.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=HERE / "index.html")
    parser.add_argument("--brain-scan", type=Path, default=WEB / "brain_scan.js")
    parser.add_argument("--config", type=Path, default=HERE.parent / "config.json", help="the stage config the planner and arm settings come from when a snapshot carries none")
    parser.add_argument("--checkpoint", action="append", default=[], metavar="LABEL=PATH", help="another snapshot of the same arm, copied beside the page and offered in the brain selector")
    parser.add_argument("--snapshot-label", default="this page's brain", help="how the inlined snapshot is named in the brain selector")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()
    config = json.loads(args.config.read_text()) if args.config and args.config.exists() else None
    snapshot = fill_extras(load_snapshot(args.snapshot), config, args.snapshot)
    modules = [
        ("brain_scan.js", args.brain_scan.read_text()),
        ("engine.js", (WEB / "engine.js").read_text()),
        ("records_view.js", (WEB / "records_view.js").read_text()),
        ("planner.js", (HERE / "planner.js").read_text()),
        ("arm.js", (HERE / "arm.js").read_text()),
        ("page.js", (HERE / "page.js").read_text()),
    ]
    check_collisions(modules)
    bundle = "\n".join(inline_module(name, source) for name, source in modules)
    payload = json.dumps(snapshot, separators=(",", ":")).replace("</", "<\\/").replace("<!--", "<\\!--").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    css = (HERE / "style.css").read_text()
    html = args.template.read_text()
    if '<link rel="stylesheet" href="style.css">' not in html or '<script type="module" src="page.js"></script>' not in html:
        raise SystemExit("the template must link style.css and load page.js as a module")
    html = html.replace('<link rel="stylesheet" href="style.css">', f"<style>\n{css}</style>")
    html = html.replace('<script type="module" src="page.js"></script>', f"<script>window.__SNAPSHOT__ = {payload};</script>\n<script type=\"module\">\n{bundle}</script>")
    html = re.sub(r'\s*data-snapshot="[^"]*"', "", html, count=1)
    if args.title:
        html = re.sub(r"<title>.*?</title>", f"<title>{args.title}</title>", html, count=1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"page written: {args.out} ({args.out.stat().st_size / 1e6:.2f} MB; snapshot {args.snapshot}, {len(modules)} modules inlined)")
    if args.checkpoint:
        entries = write_checkpoints(args.checkpoint, args.out, config, snapshot, args.snapshot_label, args.snapshot)
        print(f"checkpoints written: {args.out.parent / 'checkpoints.json'} ({len(entries)} beside the inlined brain)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
