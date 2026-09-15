#!/usr/bin/env python3
"""Build the self-contained S04 artist page: one HTML file with the snapshot, the standard
renderer, the engine, the records-cortex view, the S01 planner, the artist and the page script
inlined, so it runs from file:// and on GitHub Pages.

From a run directory, which is what a visitor sees: the final brain is inlined and one
checkpoint per phase of the life travels beside the page in the brain selector.

    python artist/web/build_page.py --run runs/artist/pilot --out artist/index.html

Any run directory works. The checkpoints are the ``artist_seed<seed>*.npz`` files the receipt
lists under ``artifacts.checkpoints``, with their sha256 checked against it; by default one is
taken per phase (after the scribbling, after the segments, after the strokes, after the
compositions, the final brain) and ``--all-checkpoints`` keeps every file the run saved. The
receipt's numbers for that seed travel in ``extra.receipt`` and the page prints them.

From one exported snapshot, which is what the parity fixture writes:

    python artist/web/build_page.py \\
        --snapshot runs/artist/parity/snapshot.json --out runs/artist/web/index.html

Each checkpoint is written to ``checkpoints/<id>.json`` next to the page and listed in
``checkpoints.json`` with its label, file, size and the decisions it had lived; the page fetches
that manifest by relative URL and falls back to the inlined snapshot alone when it is not there
(a page opened from file:// cannot fetch its neighbours). ``tools/checkpoint_assets.py prepare``
adds the sha256 and the release tag the Pages workflow downloads them from.

The source files stay separate for development (open index.html over http from the repository
root; page.js then fetches the snapshot named by ``<body data-snapshot>``). The renderer is the
verbatim copy of cadence's brain_scan.js in web/; ``--brain-scan`` names another copy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WEB = ROOT / "web"  # the shared engine, renderer and records view
ARM = ROOT / "arm" / "web"  # the S01 planner the fast level's leaf objective comes from
sys.path.insert(0, str(ROOT))

SNAPSHOT_FORMAT = "cadence-experience-web/1"
CHECKPOINT_FORMAT = "cadence-artist-checkpoints/1"
STAGE = "S04"

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
    if snapshot.get("extra", {}).get("stage") != STAGE:
        raise SystemExit(f"{path} is not an S04 snapshot (extra.stage)")
    return snapshot


def decisions_of(snapshot: dict) -> int:
    return int((snapshot.get("ledger") or {}).get("real_transitions") or 0)


def slug(label: str, taken: set[str]) -> str:
    base = re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", label.lower())).strip("-") or "brain"
    name, k = base, 2
    while name in taken:
        name, k = f"{base}-{k}", k + 1
    taken.add(name)
    return name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# -- a run directory: the brains run.py saved, one per phase of the life


PHASES = (
    ("scribble", "after the scribbling"),
    ("segment", "after the segments"),
    ("stroke", "after the strokes"),
    ("multi", "after the compositions"),
    ("trained", "after the compositions"),
    ("final", "the final brain"),
)


def phase_of(name: str, seed: int) -> str:
    rest = Path(name).stem.split(f"artist_seed{seed}", 1)[-1].lstrip("_")
    return rest.split("_")[0] if rest else "final"


def label_of(name: str, seed: int, decisions: int) -> str:
    phase = phase_of(name, seed)
    label = dict(PHASES).get(phase, phase.replace("_", " "))
    return label if phase in ("scribble", "final") else f"{label} ({decisions:,} decisions)"


def select(names: list[str], seed: int, every: bool) -> list[str]:
    """One checkpoint per phase, the last the run saved in it; ``every`` keeps them all.

    ``trained`` is the brain at the end of the composition phase, so it stands for that phase
    whenever the run saved it and the last ``multi`` file stands for it when it did not."""
    if every:
        return names
    by_phase: dict[str, str] = {}
    for name in names:
        by_phase[phase_of(name, seed)] = name
    if "trained" in by_phase:
        by_phase.pop("multi", None)
    out = [by_phase[phase] for phase, _ in PHASES if phase in by_phase]
    return out or names


def receipt_numbers(receipt: dict, seed: int) -> dict:
    """The numbers of this run for the page's receipt line: the gates' metrics at this seed."""
    completed = list((receipt.get("seeds") or {}).get("completed") or [])
    at = completed.index(seed) if seed in completed else 0
    metrics = receipt.get("metrics") or {}
    controls = receipt.get("controls") or {}
    per_seed = metrics.get("per_seed") or []
    seed_row = per_seed[at] if at < len(per_seed) and isinstance(per_seed[at], dict) else {}

    def value(name: str) -> float | None:
        series = metrics.get(name)
        return None if not isinstance(series, list) or at >= len(series) else float(series[at])

    def control(name: str) -> float | None:
        row = (controls.get(name) or {}).get("f1")
        return None if not isinstance(row, list) or at >= len(row) else float(row[at])

    families = metrics.get("per_family") or []
    weakest = None
    if at < len(families) and isinstance(families[at], dict) and families[at]:
        weakest = min(float(v["f1"]) for v in families[at].values())
    latency = (receipt.get("resources") or {}).get("latency_ms") or []
    return {
        "run_id": receipt.get("run_id", "this run"),
        "split": (receipt.get("seeds") or {}).get("split", "development"),
        "seed": seed,
        "decisions": int(((seed_row.get("ledger") or {}).get("real_transitions")) or 0),
        "drawings": int(((seed_row.get("heldout") or {}).get("drawings")) or 0),
        "f1": value("heldout_f1"), "chamfer": value("heldout_chamfer"), "f1_family": weakest,
        "f1_simple": value("f1_simple"), "chamfer_simple": value("chamfer_simple"),
        "changed_body": value("changed_body_f1"), "returned_body": value("returned_body_f1"),
        "born_frozen": control("born_frozen"), "corrupted_model": control("corrupted_model"),
        "single_intention": control("single_intention"), "readback_removed": control("readback_removed"),
        "online_mlp": control("online_mlp"), "oracle": control("oracle"), "random": control("random"),
        "latency_p95_ms": None if at >= len(latency) else float(latency[at]["p95"]),
        "gates": [{"name": p["name"], "value": p["value"], "threshold": p["threshold"], "passed": bool(p["passed"])} for p in ((receipt.get("acceptance") or {}).get("predicates") or [])],
    }


def snapshots_from_run(run: Path, seed: int | None, out: Path, taken: set[str], every: bool) -> tuple[list[dict], dict]:
    """Every brain the run saved for one seed, exported as page snapshots in the order it saved them."""
    from agent.brain import Agent
    from agent.web import export
    from artist.env import FAMILIES
    from artist.run import geometry

    receipt_path = run / "receipt.json"
    if not receipt_path.exists():
        raise SystemExit(f"{run} carries no receipt.json; name the snapshots with --checkpoint instead")
    receipt = json.loads(receipt_path.read_text())
    config = receipt["config"]
    completed = list((receipt.get("seeds") or {}).get("completed") or [])
    if seed is None:
        seed = completed[0] if completed else 0
    listed = [c["path"] for c in ((receipt.get("artifacts") or {}).get("checkpoints") or []) if isinstance(c, dict)]
    digests = {c["path"]: c.get("sha256") for c in ((receipt.get("artifacts") or {}).get("checkpoints") or []) if isinstance(c, dict)}
    names = [n for n in listed if n.startswith(f"artist_seed{seed}") and (run / n).exists()]
    if not names:
        names = sorted(p.name for p in run.glob(f"artist_seed{seed}*.npz"))
        if not names:
            raise SystemExit(f"{run} holds no artist_seed{seed}*.npz checkpoint")
        print(f"  the receipt lists no checkpoint for seed {seed}; taking the {len(names)} file(s) in the directory")
    names = select(names, seed, every)
    final = f"artist_seed{seed}_final.npz"
    names = [n for n in names if n != final] + ([final] if final in names or (run / final).exists() else [])
    geom = geometry(config)
    numbers = receipt_numbers(receipt, seed)
    folder = out.parent / "checkpoints"
    folder.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    for name in names:
        source = run / name
        expected = digests.get(name)
        if expected and sha256_file(source) != expected:
            raise SystemExit(f"{source} does not match the sha256 the receipt records")
        agent = Agent.load(source, planner=lambda *_a, **_k: (0, {}, {}))  # the controller is "planner"; the page installs the life's own search
        lived = int(agent.ledger.real_transitions)
        extra = {
            "stage": STAGE, "boundary": True, "next_moment": None, "config": config,
            "canvas": {k: (list(v) if isinstance(v, (tuple, list)) else v) for k, v in vars(geom).items()},
            "intention": config["intention"], "intention_interval": config["intention_interval"], "intention_gain": config["intention_gain"],
            "planner": config["planner"], "body": config["body"], "families": list(FAMILIES),
            "receipt": numbers, "decisions": lived,
            "checkpoint": {"file": name, "run": str(run), "seed": seed, "split": numbers["split"], "sha256": expected},
        }
        label = label_of(name, seed, lived)
        path = out.parent / f"{Path(name).stem}.json" if name == final else folder / f"{slug(label, taken)}.json"
        export(agent, path, extra=extra)
        entries.append({"label": label, "path": path, "decisions": lived, "source": str(source)})
        print(f"  {name}: {path.name} ({path.stat().st_size / 1e6:.1f} MB, {lived:,} decisions lived) · {label}")
    return entries, numbers


def write_manifest(entries: list[dict], out: Path, inlined: dict, inlined_label: str, inlined_source: str, taken: set[str]) -> list[dict]:
    """Copy every checkpoint snapshot next to the page and write the manifest the page fetches."""
    listed: list[dict] = []
    folder = out.parent / "checkpoints"
    for entry in entries:
        folder.mkdir(parents=True, exist_ok=True)
        source = Path(entry["path"])
        name = source.name if source.parent.resolve() == folder.resolve() else f"{slug(entry['label'], taken)}.json"
        target = folder / name
        if source.resolve() != target.resolve():
            target.write_text(json.dumps(load_snapshot(source), separators=(",", ":")), encoding="utf-8")
        listed.append({"id": target.stem, "label": entry["label"], "file": f"checkpoints/{name}", "bytes": target.stat().st_size, "decisions": entry["decisions"], "source": entry.get("source", str(entry["path"]))})
    manifest = {
        "format": CHECKPOINT_FORMAT,
        "page": out.name,
        "inlined": {"id": "inlined", "label": inlined_label, "decisions": decisions_of(inlined), "source": inlined_source},
        "checkpoints": listed,
    }
    (out.parent / "checkpoints.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return listed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=None, help="a run directory: the final brain is inlined and one checkpoint per phase travels beside the page")
    parser.add_argument("--seed", type=int, default=None, help="which seed of the run (default: the first completed one)")
    parser.add_argument("--snapshot", type=Path, default=None, help="one exported snapshot to inline instead of a run")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=HERE / "index.html")
    parser.add_argument("--brain-scan", type=Path, default=WEB / "brain_scan.js")
    parser.add_argument("--checkpoint", action="append", default=[], metavar="LABEL=PATH", help="another snapshot of the same life, copied beside the page and offered in the brain selector")
    parser.add_argument("--all-checkpoints", action="store_true", help="keep every checkpoint the run saved, not one per phase")
    parser.add_argument("--snapshot-label", default=None, help="how the inlined snapshot is named in the brain selector")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()
    if (args.run is None) == (args.snapshot is None):
        raise SystemExit("build from a run directory (--run) or from one snapshot (--snapshot)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    taken: set[str] = set()
    if args.run is not None:
        print(f"run: {args.run}")
        exported, numbers = snapshots_from_run(args.run, args.seed, args.out, taken, args.all_checkpoints)
        shown = [(k, numbers[k]) for k in ("f1", "chamfer", "f1_family", "born_frozen", "corrupted_model", "online_mlp", "oracle", "latency_p95_ms") if numbers.get(k) is not None]
        print(f"  receipt: {numbers['run_id']} (seed {numbers['seed']}, {numbers['decisions']:,} decisions) · " + " · ".join(f"{k} {v:.4g}" for k, v in shown))
        inlined_source = exported[-1]["source"]
        snapshot = load_snapshot(Path(exported[-1]["path"]))
        label = args.snapshot_label or exported[-1]["label"]
        entries = exported[:-1]
    else:
        snapshot = load_snapshot(args.snapshot)
        inlined_source = str(args.snapshot)
        label = args.snapshot_label or "this page's brain"
    for pair in args.checkpoint:
        if "=" not in pair:
            raise SystemExit(f"--checkpoint takes label=path, not {pair!r}")
        name, _, raw = pair.partition("=")
        path = Path(raw.strip())
        entries.append({"label": name.strip(), "path": path, "decisions": decisions_of(load_snapshot(path)), "source": str(path)})
    modules = [
        ("brain_scan.js", args.brain_scan.read_text()),
        ("engine.js", (WEB / "engine.js").read_text()),
        ("records_view.js", (WEB / "records_view.js").read_text()),
        ("planner.js", (ARM / "planner.js").read_text()),
        ("artist.js", (HERE / "artist.js").read_text()),
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
    args.out.write_text(html, encoding="utf-8")
    print(f"page written: {args.out} ({args.out.stat().st_size / 1e6:.2f} MB; {label}, {decisions_of(snapshot):,} decisions lived, {len(modules)} modules inlined)")
    if entries:
        listed = write_manifest(entries, args.out, snapshot, label, inlined_source, taken)
        print(f"checkpoints written: {args.out.parent / 'checkpoints.json'} ({len(listed)} beside the inlined brain)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
