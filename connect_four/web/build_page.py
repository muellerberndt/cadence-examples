#!/usr/bin/env python3
"""Build the Connect Four page from a run directory: one HTML file with the brain of the last
checkpoint inlined, the other checkpoints beside it, and the run's receipt numbers on the page.

    python connect_four/web/build_page.py --run runs/connect_four/page_pilot
    python connect_four/web/build_page.py --run runs/connect_four/acceptance --seed 10 \
        --out runs/connect_four/page/index.html

Every checkpoint the receipt lists for that seed (the scheduled game counts and the ends of the
phases) is exported as one file: the S00 agent through ``agent/web.py`` (the engine of
``web/engine.js`` rebuilds the whole brain from it) and the two record cortices as a block that
mirrors it, holding the configuration and the seed the expansion is regenerated from, the
habituation mean, every record table in float64, the planner's generator state and the validity
record. The page reads the inlined checkpoint from ``window.__CHECKPOINT__`` and the manifest
(every checkpoint, the receipt's numbers) from ``window.__MANIFEST__``; picking another
checkpoint fetches its file, which a page opened from file:// cannot do.

The source files stay separate for development: open connect_four/web/index.html over http from
the repository root and it fetches the checkpoint named by ``<body data-checkpoint>``.

Written beside the page: ``checkpoints/<id>.json`` (one per checkpoint), ``cortices.json`` (the
fixed cells both cortices were drawn with, shared by every checkpoint of the life and inlined in
the page) and ``checkpoints.json`` (the manifest, also inlined). To publish the page,
.github/workflows/pages.yml needs

    mkdir -p _site/connect_four
    cp connect_four/index.html connect_four/receipt.json connect_four/checkpoints.json connect_four/cortices.json _site/connect_four/
    cp -r connect_four/checkpoints _site/connect_four/checkpoints

and tools/verify_gallery.py needs ``"connect_four": "S03"`` in EXAMPLES.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WEB = ROOT / "web"  # the shared engine, renderer and records view
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.web import export as export_agent  # noqa: E402
from connect_four.brain import Brain, checkpoint_files  # noqa: E402

CHECKPOINT_FORMAT = "cadence-connect-four-checkpoint/1"
CORTICES_FORMAT = "cadence-connect-four-cortices/1"
EXPANSION_TOLERANCE = 1e-12  # the last bits log and cos round differently on another platform
BRAIN_FORMAT = "cadence-connect-four-web/1"
SNAPSHOT_FORMAT = "cadence-experience-web/1"
MANIFEST_FORMAT = "cadence-connect-four-checkpoints/1"

IMPORT_RE = re.compile(r"^\s*import\s[^;]*?\bfrom\s+[\"'][^\"']+[\"'];?[ \t]*$", re.MULTILINE)
EXPORT_RE = re.compile(r"^export\s+(?=(?:const|let|var|function|class|async)\b)", re.MULTILINE)
DECL_RE = re.compile(r"^(?:export\s+)?(?:async\s+)?(?:function\*?|class|const|let|var)\s+([A-Za-z_$][\w$]*)", re.MULTILINE)
STEM_RE = re.compile(r"^brain_seed(\d+)_games(\d+)$")


def safe(text: str) -> str:
    """JSON that can sit inside a script element of an HTML page."""
    return text.replace("</", "<\\/").replace("<!--", "<\\!--").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def inline_module(name: str, source: str) -> str:
    """The module's code as a script fragment: static imports dropped, export keywords removed."""
    if re.search(r"^\s*export\s*\{", source, re.MULTILINE) or re.search(r"^\s*export\s+default\b", source, re.MULTILINE):
        raise SystemExit(f"{name}: only `export const/function/class` declarations can be inlined")
    out = IMPORT_RE.sub("", source)
    if re.search(r"^\s*import\s", out, re.MULTILINE):
        raise SystemExit(f"{name}: an import statement could not be inlined (write one-line imports)")
    return f"// ---- {name}\n{EXPORT_RE.sub('', out).strip()}\n"


def check_collisions(modules: list[tuple[str, str]]) -> None:
    seen: dict[str, str] = {}
    for name, source in modules:
        for match in DECL_RE.finditer(source):
            ident = match.group(1)
            if ident in seen and seen[ident] != name:
                raise SystemExit(f"top-level name {ident!r} is declared in both {seen[ident]} and {name}; rename one before inlining")
            seen.setdefault(ident, name)


@contextmanager
def recorded_expansion(stem: Path):
    """Load a checkpoint whose expansion was drawn on another machine.

    ``Brain.load`` requires the fixed projection to rebuild from its seed exactly. A life
    recorded on another platform draws it a few last bits apart, because log and cos round
    differently there, and the brain then refuses to load at all. Inside this context that
    equality holds when the difference stays under ``EXPANSION_TOLERANCE``; the caller installs
    the projection and the offsets the checkpoint carries, so the brain that comes out is the
    one the run used, not a rebuilt near-copy. The largest difference is reported."""
    original = np.array_equal
    seen = {"max": 0.0, "tolerated": 0}

    def close(a, b, **kwargs):
        if original(a, b, **kwargs):
            return True
        first, second = np.asarray(a), np.asarray(b)
        if first.shape != second.shape or first.dtype.kind != "f":
            return False
        difference = float(np.abs(first - second).max())
        if difference > EXPANSION_TOLERANCE:
            return False
        seen["max"] = max(seen["max"], difference)
        seen["tolerated"] += 1
        return True

    np.array_equal = close
    try:
        yield seen
    finally:
        np.array_equal = original


def load_brain(stem: Path) -> tuple[Brain, dict[str, Any]]:
    """The brain of one checkpoint, with the fixed expansion the checkpoint carries."""
    with recorded_expansion(stem) as seen:
        brain = Brain.load(stem)
    _, records_file = checkpoint_files(stem)
    with np.load(records_file, allow_pickle=False) as data:
        for name, cortex in (("drop", brain.drop.records), ("lines", brain.lines.records)):
            cortex.projection = data[f"{name}/projection"].copy()
            cortex.offset = data[f"{name}/offset"].copy()
    brain.drop._known[:] = False
    brain.lines._fresh = False
    return brain, seen


def b64(array: np.ndarray, dtype: str = "<f8") -> dict[str, Any]:
    data = np.ascontiguousarray(np.asarray(array).astype(dtype))
    return {"dtype": data.dtype.str.lstrip("<>|="), "shape": list(data.shape), "b64": base64.b64encode(data.tobytes()).decode("ascii")}


def cortex_block(cortex, extra: dict[str, Any]) -> dict[str, Any]:
    """One record cortex as the page rebuilds it: the fixed cells from the seed, the learned tables."""
    return {
        **extra,
        "config": cortex.to_dict(),
        "seed": int(cortex.seed),
        "inputs": int(cortex.inputs),
        "cells": int(cortex.cells),
        "active": int(cortex.active),
        "fields": [{"name": name, "width": int(width)} for name, width in cortex.fields.items()],
        "valued": sorted(cortex.valued),
        "mean": b64(cortex.mean),
        "tables": {name: b64(table) for name, table in cortex.tables.items()},
        "seen": int(cortex.seen),
        "writes": int(cortex.writes),
        # the page regenerates the projection and the offsets from the seed; these hold it to the Python ones
        "probe": {
            "projection": [float(cortex.projection[0, 0]), float(cortex.projection[-1, -1]), float(cortex.projection.sum())],
            "offset": [float(cortex.offset[0]), float(cortex.offset[-1]), float(cortex.offset.sum())],
        },
    }


def pack_boards(boards: np.ndarray) -> np.ndarray:
    """Boards of ternary cells packed five cells to a byte, nine bytes a board (42 cells)."""
    boards = np.asarray(boards, dtype=np.int64).reshape(len(boards), -1)
    cells = boards.shape[1]
    width = -(-cells // 5)
    padded = np.zeros((len(boards), width * 5), dtype=np.int64)
    padded[:, :cells] = boards
    powers = 3 ** np.arange(5)
    return (padded.reshape(len(boards), width, 5) * powers).sum(axis=2).astype(np.uint8)


def memory_file(brain: Brain) -> dict[str, Any]:
    """The brain's two memories as the page fetches them beside the checkpoints: the proven
    positions and the columns winners played, boards packed five cells to a byte."""
    memory = brain.planner.memory
    boards = np.array([np.frombuffer(k[0], dtype=np.int8) for k in memory], dtype=np.int8).reshape(len(memory), brain.game.cells)
    wins = [(k, c, n) for k, counts in brain.planner.wins.items() for c, n in counts.items()]
    wboards = np.array([np.frombuffer(k, dtype=np.int8) for k, _, _ in wins], dtype=np.int8).reshape(len(wins), brain.game.cells)
    return {
        "format": "cadence-connect-four-memory/1", "cells": brain.game.cells, "packed": 5,
        "proofs": {"entries": len(memory), "boards": b64(pack_boards(boards), "|u1"), "sides": b64(np.array([k[1] for k in memory], dtype=np.int8), "|i1"),
                   "results": b64(np.array(list(memory.values()), dtype=np.float32), "<f4")},
        "wins": {"entries": len(wins), "boards": b64(pack_boards(wboards), "|u1"), "columns": b64(np.array([c for _, c, _ in wins], dtype=np.int8), "|i1"),
                 "counts": b64(np.array([min(n, 65535) for _, _, n in wins], dtype=np.uint16), "<u2")},
    }


def memory_block(brain: Brain) -> dict[str, Any]:
    """The positions the brain's searches proved, as the page reads them back."""
    memory = brain.planner.memory
    boards = np.array([np.frombuffer(k[0], dtype=np.int8) for k in memory], dtype=np.int8).reshape(len(memory), brain.game.cells)
    return {"entries": len(memory), "boards": b64(boards, "|i1"), "sides": b64(np.array([k[1] for k in memory], dtype=np.int8), "|i1"),
            "results": b64(np.array(list(memory.values()), dtype=np.float32), "<f4")}


def wins_block(brain: Brain) -> dict[str, Any]:
    """The columns winners played on the boards the brain has seen, as the page reads them back."""
    wins = [(k, c, n) for k, counts in brain.planner.wins.items() for c, n in counts.items()]
    boards = np.array([np.frombuffer(k, dtype=np.int8) for k, _, _ in wins], dtype=np.int8).reshape(len(wins), brain.game.cells)
    return {"entries": len(wins), "boards": b64(boards, "|i1"), "columns": b64(np.array([c for _, c, _ in wins], dtype=np.int8), "|i1"),
            "counts": b64(np.array([n for _, _, n in wins], dtype=np.int32), "<i4")}


def brain_block(brain: Brain) -> dict[str, Any]:
    """The record cortices, the planner and the life's counters of one checkpoint."""
    state = brain.planner.rng.bit_generator.state
    if state.get("bit_generator") != "PCG64":
        raise SystemExit(f"the planner's generator is {state.get('bit_generator')}, and the page ports PCG64")
    return {
        "format": BRAIN_FORMAT,
        "seed": int(brain.seed),
        "learning": bool(brain.learning),
        "is_copy": bool(brain.is_copy),
        "game": {"rows": brain.game.rows, "cols": brain.game.cols, "connect": brain.game.connect},
        "planning": dict(brain.config["planning"]),
        "validity_gate": float(brain.validity_gate),
        "brain": brain.cfg,
        "counts": dict(brain.counts),
        "snapshots": int(brain.snapshots),
        "planner": brain.planner.stats(),
        "planner_generator": {"state": str(state["state"]["state"]), "inc": str(state["state"]["inc"]),
                              "has_uint32": int(state["has_uint32"]), "uinteger": int(state["uinteger"])},
        "validity": [bool(v) for v in brain.validity],
        "extended": bool(brain.extended),
        "memory": {"entries": 0, "file": "checkpoints/memory.json"},  # the memories travel in their own file, fetched by the page
        "wins": {"entries": 0, "file": "checkpoints/memory.json"},
        "parameters": brain.parameters(),
        "cortices": {
            "drop": cortex_block(brain.drop.records, {"chosen_gain": float(brain.drop.gain),
                                                      "reading": "one column: a one-hot (empty, the side to move, the other side) per cell from the bottom, then the chosen flag times chosen_gain",
                                                      "groups": [{"name": f"row {r}", "start": 3 * r, "end": 3 * (r + 1)} for r in range(brain.game.rows)]
                                                                + [{"name": "chosen", "start": 3 * brain.game.rows, "end": 3 * brain.game.rows + 1}]}),
            "lines": cortex_block(brain.lines.records, {"reading": "one window: a one-hot (empty, the side that just moved, the other side) per cell in window order",
                                                        "groups": [{"name": f"cell {k}", "start": 3 * k, "end": 3 * (k + 1)} for k in range(brain.game.connect)]}),
        },
    }


def cortices_export(brain: Brain) -> dict[str, Any]:
    """The fixed cells of this life: the projection and the offsets of both cortices, which every
    checkpoint of the life shares. A page regenerates them from the seed and holds the numbers
    it draws to these (the last bits differ between platforms)."""
    return {
        "format": CORTICES_FORMAT, "seed": int(brain.seed),
        "drop": {"seed": int(brain.drop.records.seed), "projection": b64(brain.drop.records.projection), "offset": b64(brain.drop.records.offset)},
        "lines": {"seed": int(brain.lines.records.seed), "projection": b64(brain.lines.records.projection), "offset": b64(brain.lines.records.offset)},
    }


def checkpoint_export(stem: Path, out: Path, label: str, games: int) -> dict[str, Any]:
    """One checkpoint as one file: the agent snapshot of agent/web.py and the brain block."""
    brain, expansion = load_brain(stem)
    agent_path = out.with_name(out.name + ".agent.json")
    export_agent(brain.agent, agent_path, extra={"connect_four": {"games": games, "label": label}})
    agent = json.loads(agent_path.read_text())
    agent_path.unlink()
    if agent.get("format") != SNAPSHOT_FORMAT:
        raise SystemExit(f"{stem}: the agent export is not a {SNAPSHOT_FORMAT} snapshot")
    payload = {"format": CHECKPOINT_FORMAT, "games": games, "label": label, "agent": agent, "brain": brain_block(brain),
               "cortices_file": "cortices.json", "expansion_difference": expansion["max"]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    payload["cortices"] = cortices_export(brain)  # kept beside the page, not in the checkpoint file
    return payload


def receipt_numbers(receipt: dict, seed: int) -> dict[str, Any]:
    """What the page states about the run this brain comes from."""
    per_seed = [r for r in receipt.get("metrics", {}).get("per_seed", []) if int(r.get("seed", -1)) == seed]
    if not per_seed:
        raise SystemExit(f"the receipt holds no metrics for seed {seed}")
    r = per_seed[0]
    predicates = {p["name"]: p for p in receipt.get("acceptance", {}).get("predicates", [])}
    return {
        "run_id": receipt.get("run_id"),
        "stage": receipt.get("stage"),
        "status": receipt.get("status"),
        "pilot": bool(receipt.get("pilot")),
        "seed": seed,
        "split": receipt.get("seeds", {}).get("split"),
        "seeds": receipt.get("seeds", {}).get("completed", []),
        "games": r.get("games", {}),
        "decisions": r.get("decisions"),
        "tactical": {"success": r["tactical"]["success"], "cases": r["tactical"]["cases"], "strata": r["tactical"]["strata"]},
        "paired": {k: v["score"] for k, v in r.get("paired_mixture", {}).items()},
        "paired_adaptation": {k: v["score"] for k, v in r.get("paired_adaptation", {}).items()},
        "no_planning": {k: v["score"] for k, v in r.get("no_planning", {}).get("paired", {}).items()},
        "planning_gain": r.get("planning_gain"),
        "corrupted_loss": r.get("corrupted_loss"),
        "validity": r.get("validity", {}).get("exact"),
        "terminal": r.get("terminal", {}).get("balanced_accuracy"),
        "latency_ms": r.get("latency_ms", {}),
        "transitions": r.get("after_adaptation", {}).get("transitions"),
        "imagined": r.get("after_adaptation", {}).get("planner", {}).get("nodes"),
        "record_writes": {k: r.get("after_adaptation", {}).get(k) for k in ("drop_writes", "line_writes", "value_writes")},
        "parameters": receipt.get("resources", {}).get("parameters", {}),
        "cadence": receipt.get("sources", {}).get("core", {}).get("repo", {}).get("commit"),
        "predicates": [{"name": p["name"], "value": p["value"], "threshold": p["threshold"], "passed": p["passed"]} for p in predicates.values()],
    }


def checkpoints_of(run: Path, seed: int | None) -> tuple[int, list[dict[str, Any]], dict]:
    """The checkpoints of one seed in a run directory, in game order, with the receipt."""
    receipt_path = run / "receipt.json"
    if not receipt_path.exists():
        raise SystemExit(f"{run} holds no receipt.json")
    receipt = json.loads(receipt_path.read_text())
    entries = receipt.get("artifacts", {}).get("checkpoints", [])
    seeds = sorted({int(e["seed"]) for e in entries})
    if not seeds:
        raise SystemExit(f"{run}: the receipt lists no checkpoints")
    if seed is None:
        if len(seeds) > 1:
            raise SystemExit(f"{run} holds the seeds {seeds}; name one with --seed")
        seed = seeds[0]
    if seed not in seeds:
        raise SystemExit(f"{run} holds no checkpoint of seed {seed} (it holds {seeds})")
    out = []
    for entry in sorted((e for e in entries if int(e["seed"]) == seed), key=lambda e: int(e["games"])):
        games = int(entry["games"])
        stem = run / f"brain_seed{seed}_games{games:06d}"
        agent_file, records_file = checkpoint_files(stem)
        if not agent_file.exists() or not records_file.exists():
            raise SystemExit(f"{stem}: the checkpoint files the receipt lists are not in {run}")
        out.append({"games": games, "labels": list(entry.get("labels", [])), "stem": stem})
    return seed, out, receipt


LABELS = {"scheduled": "", "after_explore": "the end of the random games", "after_mixture": "the end of the mixture", "final": "the end of the life"}


def label_of(entry: dict) -> str:
    """How one checkpoint is named in the selector: its game count, and the phase it ends."""
    games = entry["games"]
    phase = next((LABELS[name] for name in entry["labels"] if LABELS.get(name)), "")
    return f"{games:,} games" + (f" · {phase}" if phase else "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="the run directory: the checkpoints and the receipt of one life")
    parser.add_argument("--seed", type=int, default=None, help="which seed's life, when the run holds several")
    parser.add_argument("--out", type=Path, default=ROOT / "connect_four" / "index.html")
    parser.add_argument("--template", type=Path, default=HERE / "index.html")
    parser.add_argument("--brain-scan", type=Path, default=WEB / "brain_scan.js")
    parser.add_argument("--inline", default="last", help="which checkpoint the page carries: 'last', 'first' or a game count")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    seed, entries, receipt = checkpoints_of(args.run, args.seed)
    numbers = receipt_numbers(receipt, seed)
    folder = args.out.parent / "checkpoints"
    folder.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    payloads: dict[str, dict] = {}
    cortices: dict[str, Any] | None = None
    for entry in entries:
        ident = f"games{entry['games']:06d}"
        path = folder / f"{ident}.json"
        payload = checkpoint_export(entry["stem"], path, label_of(entry), entry["games"])
        fixed = payload.pop("cortices")
        if cortices is None:
            cortices = fixed
        elif any(cortices[name]["projection"]["b64"] != fixed[name]["projection"]["b64"] for name in ("drop", "lines")):
            raise SystemExit(f"{entry['stem']}: the fixed expansion differs from the one of the first checkpoint of this life")
        payloads[ident] = payload
        manifest.append({
            "id": ident, "label": label_of(entry), "file": f"checkpoints/{path.name}", "games": entry["games"],
            "labels": entry["labels"], "bytes": path.stat().st_size,
            "transitions": payload["brain"]["counts"]["transitions"], "drop_writes": payload["brain"]["counts"]["drop_writes"],
            "line_writes": payload["brain"]["counts"]["line_writes"], "value_writes": payload["brain"]["counts"]["value_writes"],
            "extended": payload["brain"]["extended"], "planner": payload["brain"]["planner"], "source": str(entry["stem"]),
        })
        print(f"  checkpoint {ident}: {path} ({path.stat().st_size / 1e6:.2f} MB, {entry['games']:,} games, {payload['brain']['counts']['transitions']:,} transitions)")
    (args.out.parent / "cortices.json").write_text(json.dumps(cortices, separators=(",", ":")), encoding="utf-8")
    print(f"  cortices: {args.out.parent / 'cortices.json'} ({(args.out.parent / 'cortices.json').stat().st_size / 1e6:.2f} MB, the fixed cells of this life)")
    chosen = manifest[-1]["id"] if args.inline == "last" else manifest[0]["id"] if args.inline == "first" else f"games{int(args.inline):06d}"
    if chosen not in payloads:
        raise SystemExit(f"--inline {args.inline}: no such checkpoint ({', '.join(payloads)})")
    remembered = Brain.load(next(e["stem"] for e in entries if f"games{int(e['games']):06d}" == chosen))
    memory_path = folder / "memory.json"
    memory_path.write_text(json.dumps(memory_file(remembered), separators=(",", ":")), encoding="utf-8")
    manifest.append({"id": "memory", "label": "what the brain remembers", "file": "checkpoints/memory.json", "games": int(remembered.counts.get("games_written", 0)),
                     "labels": ["memory"], "bytes": memory_path.stat().st_size, "proofs": len(remembered.planner.memory), "wins": len(remembered.planner.wins),
                     "source": str(next(e["stem"] for e in entries if f"games{int(e['games']):06d}" == chosen))})
    print(f"  memory: {memory_path} ({memory_path.stat().st_size / 1e6:.2f} MB, {len(remembered.planner.memory):,} proven positions, {len(remembered.planner.wins):,} boards with winners' columns)")
    body = {"format": MANIFEST_FORMAT, "seed": seed, "run": str(args.run), "inlined": chosen, "checkpoints": manifest, "receipt": numbers}
    (args.out.parent / "checkpoints.json").write_text(json.dumps(body, indent=1), encoding="utf-8")

    modules = [
        ("brain_scan.js", args.brain_scan.read_text()),
        ("engine.js", (WEB / "engine.js").read_text()),
        ("records_view.js", (WEB / "records_view.js").read_text()),
        ("game.js", (HERE / "game.js").read_text()),
        ("brain.js", (HERE / "brain.js").read_text()),
        ("page.js", (HERE / "page.js").read_text()),
    ]
    check_collisions(modules)
    bundle = "\n".join(inline_module(name, source) for name, source in modules)
    payload = safe(json.dumps(payloads[chosen], separators=(",", ":")))
    css = (HERE / "style.css").read_text()
    html = args.template.read_text()
    if '<link rel="stylesheet" href="style.css">' not in html or '<script type="module" src="page.js"></script>' not in html:
        raise SystemExit("the template must link style.css and load page.js as a module")
    html = html.replace('<link rel="stylesheet" href="style.css">', f"<style>\n{css}</style>")
    html = html.replace(
        '<script type="module" src="page.js"></script>',
        f"<script>window.__CHECKPOINT__ = {payload};\nwindow.__CORTICES__ = {safe(json.dumps(cortices, separators=(',', ':')))};\nwindow.__MANIFEST__ = {safe(json.dumps(body, separators=(',', ':')))};</script>\n<script type=\"module\">\n{bundle}</script>",
    )
    html = re.sub(r'\s*data-(?:checkpoint|cortices)="[^"]*"', "", html)
    if args.title:
        html = re.sub(r"<title>.*?</title>", f"<title>{args.title}</title>", html, count=1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"page written: {args.out} ({args.out.stat().st_size / 1e6:.2f} MB; seed {seed}, {len(manifest)} checkpoints, {chosen} inlined, {len(modules)} modules)")
    print(f"manifest written: {args.out.parent / 'checkpoints.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
