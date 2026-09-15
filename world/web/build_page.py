#!/usr/bin/env python3
"""Build the self-contained S02 page: one HTML file with the snapshot, the standard renderer,
the engine, the records-cortex view, the world, the stores, the planner and the page script
inlined, so it runs from file:// and on GitHub Pages.

From a run directory, which is what a visitor sees: the final brain is inlined and every
checkpoint ``run.py`` saved travels beside the page in the brain selector.

    python world/web/build_page.py --run runs/world/dev --out runs/world/web/index.html

Any run directory works: the checkpoints are the ``world_seed<seed>*.npz`` files the receipt
lists under ``artifacts.checkpoints`` (explore at a tenth, a half and all of the explore
budget, then after the remembered requests, after the cue task, after the doors locked, and
the final brain), in the order the run saved them, with their sha256 checked against the
receipt. The world comes back from the life's own seed, so every brain of the run meets the
map it learned; the checkpoints carry no stores (``Agent.save`` holds the brain alone), so a
brain loaded from the selector starts with nothing remembered and fills its stores by
wandering.

From one exported snapshot, which is what the parity fixture writes:

    python world/web/build_page.py \\
        --snapshot runs/world/parity/snapshot.json --out runs/world/web/index.html

Each checkpoint is written to ``checkpoints/<id>.json`` next to the page and listed in
``checkpoints.json`` with its label, file and size; the page fetches that manifest by relative
URL and falls back to the inlined snapshot alone when it is not there (a page opened from
file:// cannot fetch its neighbours). ``--checkpoint "label=path"`` names snapshots directly.

The source files stay separate for development (open index.html over http from the repository
root; page.js then fetches the snapshot named by ``<body data-snapshot>``). The renderer is the
verbatim copy of cadence's brain_scan.js in web/; ``--brain-scan`` names another copy (the
cadence source tree, say).
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
sys.path.insert(0, str(ROOT))

SNAPSHOT_FORMAT = "cadence-experience-web/1"
CHECKPOINT_FORMAT = "cadence-world-checkpoints/1"
STAGE = "S02"

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
        raise SystemExit(f"{path} is not an S02 snapshot (extra.stage)")
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


# -- a run directory: the brains run.py saved, each with the world of its own life


def label_of(name: str, seed: int) -> str:
    """The name of a checkpoint file as a visitor reads it (run.py `checkpoint`)."""
    rest = Path(name).stem.split(f"world_seed{seed}", 1)[-1].lstrip("_")
    if not rest:
        return "the final brain"
    if rest.startswith("explore_"):
        return f"after {int(rest.split('_')[1]):,} explore steps"
    return {"remember": "after the remembered requests", "cue": "after the cue task", "doors": "after the doors locked"}.get(rest, rest.replace("_", " "))


def receipt_numbers(receipt: dict, seed: int) -> dict:
    """The numbers of this run for the page's receipt paragraph: the gates' metrics at this seed."""
    completed = list((receipt.get("seeds") or {}).get("completed") or [])
    at = completed.index(seed) if seed in completed else 0
    metrics = receipt.get("metrics") or {}
    controls = receipt.get("controls") or {}

    def value(name: str) -> float | None:
        series = metrics.get(name)
        return None if not isinstance(series, list) or at >= len(series) else float(series[at])

    def control(name: str, key: str = "remember_32") -> float | None:
        series = controls.get(name)
        if not isinstance(series, list) or at >= len(series) or not isinstance(series[at], dict):
            return None
        return None if series[at].get(key) is None else float(series[at][key])

    per_seed = (metrics.get("per_seed") or [{}])[at] if isinstance(metrics.get("per_seed"), list) and at < len(metrics.get("per_seed")) else {}
    return {
        "run_id": receipt.get("run_id", "this run"),
        "split": (receipt.get("seeds") or {}).get("split", "development"),
        "seed": seed,
        "steps": int(per_seed.get("real_steps") or 0),
        "metrics": {
            "consequence_accuracy": value("consequence_accuracy"), "request_success_32": value("request_success_32"),
            "correction_visible": value("correction_visible"), "cue_delay8": value("cue_delay8"),
            "grounding": value("grounding"), "door_recovery": value("door_recovery"),
            "erased_places": control("erased_places"), "born_frozen": control("born_frozen"),
        },
    }


def snapshots_from_run(run: Path, seed: int | None, out: Path, taken: set[str]) -> tuple[list[dict], dict]:
    """Every brain the run saved for one seed, exported as page snapshots in the order it saved them.

    Returns the entries (label, snapshot path, decisions, source) and the run's receipt numbers.
    """
    from agent.brain import Agent
    from agent.life import seed_for
    from agent.web import export
    from world.brain import Brain
    from world.env import World, WorldConfig
    from world.tasks import Curriculum
    from world.tools.parity_fixture import (
        bookkeeping_record,
        planner_record,
        stores_record,
        world_state,
    )

    receipt_path = run / "receipt.json"
    if not receipt_path.exists():
        raise SystemExit(f"{run} carries no receipt.json; name the snapshots with --checkpoint instead")
    receipt = json.loads(receipt_path.read_text())
    config = receipt["config"]
    split = (receipt.get("seeds") or {}).get("split", "development")
    completed = list((receipt.get("seeds") or {}).get("completed") or [])
    if seed is None:
        seed = completed[0] if completed else 0
    listed = [c["path"] for c in ((receipt.get("artifacts") or {}).get("checkpoints") or []) if isinstance(c, dict)]
    digests = {c["path"]: c.get("sha256") for c in ((receipt.get("artifacts") or {}).get("checkpoints") or []) if isinstance(c, dict)}
    names = [n for n in listed if n.startswith(f"world_seed{seed}") and (run / n).exists()]
    if not names:
        names = sorted(p.name for p in run.glob(f"world_seed{seed}*.npz"))
        if not names:
            raise SystemExit(f"{run} holds no world_seed{seed}*.npz checkpoint")
        print(f"  the receipt lists no checkpoint for seed {seed}; taking the {len(names)} file(s) in the directory")
    final = f"world_seed{seed}.npz"
    names = [n for n in names if n != final] + ([final] if final in names else [])  # the final brain last: the page inlines it
    # the world of this life, at the state a page opens on: the objects scattered, the agent placed
    world = World(WorldConfig(**config["world_config"]), seed=seed_for(STAGE, split, seed, 0, 0), life_id="candidate")
    cur = Curriculum(world, seed=seed_for(STAGE, split, seed, 0, 1))
    cur.explore()
    world.reset()
    fresh = Brain(config, seed)  # a life's planner and stores at birth: a saved brain carries neither
    numbers = receipt_numbers(receipt, seed)
    folder = out.parent / "checkpoints"
    folder.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    for name in names:
        source = run / name
        expected = digests.get(name)
        if expected and sha256_file(source) != expected:
            raise SystemExit(f"{source} does not match the sha256 the receipt records")
        agent = Agent.load(source, planner=fresh._plan)  # the controller is "planner": a saved brain is loaded with the life's own search
        extra = {
            "stage": STAGE, "boundary": True, "next_moment": None,
            "config": config, "planner": planner_record(fresh), "brain": bookkeeping_record(fresh), "stores": stores_record(fresh),
            "recall": {"last": [0.0] * len(agent.ports.recall), "gain": config["graph"].get("recall_gain", 2.0), "width": len(agent.ports.recall)},
            "world": world_state(world), "curriculum": {"cue_rule": int(cur.cue_rule), "seed": int(seed_for(STAGE, split, seed, 0, 1))},
            "world_seed": int(world.seed), "steps_before": int(agent.ledger.real_transitions),
            "receipt": numbers, "checkpoint": {"file": name, "run": str(run), "seed": seed, "split": split, "sha256": expected},
        }
        label = label_of(name, seed)
        path = out.parent / f"{Path(name).stem}.json" if name == final else folder / f"{slug(label, taken)}.json"
        export(agent, path, extra=extra)
        entries.append({"label": label, "path": path, "decisions": int(agent.ledger.real_transitions), "source": str(source)})
        print(f"  {name}: {path.name} ({path.stat().st_size / 1e6:.1f} MB, {entries[-1]['decisions']:,} decisions lived) · {entries[-1]['label']}")
    return entries, numbers


def write_manifest(entries: list[dict], out: Path, inlined: dict, inlined_label: str, inlined_source: str, taken: set[str]) -> list[dict]:
    """Copy every checkpoint snapshot next to the page and write the manifest the page fetches."""
    listed: list[dict] = []
    folder = out.parent / "checkpoints"
    for entry in entries:
        folder.mkdir(parents=True, exist_ok=True)
        source = Path(entry["path"])
        name = source.name if source.parent.resolve() == folder.resolve() else f"{slug(entry['label'], taken)}.json"  # a snapshot exported from a run is already in place
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
    parser.add_argument("--run", type=Path, default=None, help="a run directory: the final brain is inlined and every checkpoint of --seed travels beside the page")
    parser.add_argument("--seed", type=int, default=None, help="which seed of the run (default: the first completed one)")
    parser.add_argument("--snapshot", type=Path, default=None, help="one exported snapshot to inline instead of a run")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=HERE / "index.html")
    parser.add_argument("--brain-scan", type=Path, default=WEB / "brain_scan.js")
    parser.add_argument("--checkpoint", action="append", default=[], metavar="LABEL=PATH", help="another snapshot of the same life, copied beside the page and offered in the brain selector")
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
        exported, numbers = snapshots_from_run(args.run, args.seed, args.out, taken)
        print(f"  receipt: {numbers['run_id']} (seed {numbers['seed']}, {numbers['steps']:,} real steps) · " + " · ".join(f"{k} {v:.2f}" for k, v in numbers["metrics"].items() if v is not None))
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
        ("world.js", (HERE / "world.js").read_text()),
        ("stores.js", (HERE / "stores.js").read_text()),
        ("planner.js", (HERE / "planner.js").read_text()),
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
