#!/usr/bin/env python3
"""Build the self-contained S01 arm page: one HTML file with the snapshot, the standard
renderer, the engine, the records-cortex view, the planner, the arm and the page script
inlined, so it runs from file:// and on GitHub Pages.

    python arm/web/build_page.py \
        --snapshot runs/arm/parity/snapshot.json --out runs/arm/web/index.html

The source files stay separate for development (open index.html over http from the
repository root; page.js then fetches the snapshot named by ``<body data-snapshot>``). The
renderer is the verbatim copy of cadence's brain_scan.js in web/;
``--brain-scan`` names another copy (the cadence source tree, say).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEB = HERE.parents[1] / "web"  # the shared engine, renderer and records view

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=HERE / "index.html")
    parser.add_argument("--brain-scan", type=Path, default=WEB / "brain_scan.js")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text())
    if snapshot.get("format") != "cadence-experience-web/1":
        raise SystemExit("the snapshot is not a cadence-experience-web/1 export")
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
