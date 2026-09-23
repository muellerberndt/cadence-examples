#!/usr/bin/env python3
"""Export what the page needs: the pretrained belief, the genomes and the world's constants as
``web/data/brain.json``, and the library's viewer as ``web/brain_scan.js``.

    python dozing-cat/export.py                       # web/data/brain.json and web/brain_scan.js beside this file
    python dozing-cat/export.py --out /tmp/dozing-cat # brain.json there, the viewer left alone
    python dozing-cat/export.py --check               # rebuild in memory and compare with the committed brain.json

``brain.json`` holds the belief patch's slow parameters, output code and precision, its input
norm and its record store's learned state (the projection and the offsets are not in it: the
page draws them from the seed as the library does); the motion scale and the routine floors
the governors compare with; the machinery every arm shares; the world's constants (the eight
directions, the retina's grid) as the floats numpy computes them; and the genomes of both
governors: hand-set (with the habit the pretraining fitted, as the experiment's demo runs it),
evolved and best of random search, from the evolution receipts, with the held-out winner as
the default. The viewer is copied from the library checkout as it stands, with its commit and
the sha256 of the copy recorded in ``brain.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cadence  # noqa: E402
import cat  # noqa: E402
from cadence import BeliefPatch  # noqa: E402
from cadence.atlas import brain_scan_script  # noqa: E402

FORMAT = "cadence-examples.dozing-cat.brain/1"
PROVENANCE = ("library", "viewer")  # the fields a rebuild on another machine may differ in


def git(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def library_info() -> dict:
    root = Path(cadence.__file__).resolve().parents[2]
    commit = git(root, "rev-parse", "HEAD") if (root / ".git").exists() else None
    return {"version": cadence.__version__, "commit": commit, "belief_py_sha256": hashlib.sha256(Path(cadence.__file__).with_name("belief.py").read_bytes()).hexdigest()}


def viewer_info(text: str) -> dict:
    root = Path(cadence.__file__).resolve().parents[2]
    source = Path(cadence.__file__).with_name("brain_scan.js")
    dirty = git(root, "status", "--porcelain", "--", str(source)) if (root / ".git").exists() else None
    version = None
    for line in text.splitlines():
        if line.startswith("export const VERSION"):
            version = line.split('"')[1]
            break
    return {"file": "brain_scan.js", "version": version, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "commit": git(root, "rev-parse", "HEAD") if (root / ".git").exists() else None, "uncommitted_changes": bool(dirty) if dirty is not None else None, "style": "brain" if "brainLayout" in text else "scan"}


def belief_state(patch: BeliefPatch) -> dict:
    snapshot = patch.snapshot()
    meta = json.loads(str(snapshot["meta"]))
    records = patch.records
    config = records.to_dict()
    if config["pathways"] or config["tasks"] or config["fan_in"] or config["averaging"] or config["homeostasis"]:
        raise SystemExit("the page reads the plain record store only")
    if len(meta["port"]["blocks"]) != 1 or meta["port"]["blocks"][0]["kind"] != "dense" or meta["port"].get("broadcast"):
        raise SystemExit("the page reads a single dense port only")
    state = records.state()
    P = patch.parameters()
    return {
        "format": meta["format"],
        "inputs": patch.inputs, "encoded": patch.encoded, "actions": patch.actions, "belief": patch.belief, "outputs": patch.outputs,
        "iterations": patch.iterations, "damping": patch.damping, "record_width": patch.record_width, "updates": patch.updates,
        **{k: P[k].reshape(-1).tolist() for k in ("E", "e_b", "T", "t_b", "G", "g_b", "F", "f_b", "C", "c")},
        "output_precision": snapshot["output_precision"].tolist(), "output_code": snapshot["output_code"].reshape(-1).tolist(),
        "input_norm": float(snapshot["input_norm"]),
        "records": {"inputs": records.inputs, "cells": records.cells, "active": records.active, "seed": records.seed, "bias": records.bias, "rate": records.rate, "habituation": records.habituation, "width": records.fields["y"],
                    "mean": state["mean"].tolist(), "boost": state["boost"].tolist(), "table_y": state["table_y"].reshape(-1).tolist(), "seen": int(state["seen"]), "writes": int(state["writes"])},
    }


def genomes(ctx: dict) -> tuple[dict, dict, dict]:
    """The genomes the page offers, as the experiment's demo builds them: the hand-set genome
    with the habit the pretraining fitted, the evolved and the random-search genomes of each
    evolution receipt, and the held-out winner as the default."""
    out = {"patch": {"hand_set": dict(cat.PATCH_HAND_SET)}, "threshold": {"hand_set": dict(cat.THRESHOLD_HAND_SET)}}
    default = {"patch": "hand_set", "threshold": "hand_set"}
    sources = {}
    for which in ("patch", "threshold"):
        out[which]["hand_set"].update({k: float(ctx["habit_fitted"][k]) for k in cat.HABIT_KEYS})
        receipt = HERE / "receipts" / f"cat_evolution_{which}.json"
        body = json.loads(receipt.read_text())
        out[which]["evolved"] = dict(body["evolved"]["genome"])
        out[which]["random_search"] = dict(body["random_search"]["genome"])
        default[which] = str(body["winner"])
        sources[which] = {"file": f"receipts/{receipt.name}", "receipt_sha256": body.get("receipt_sha256"), "winner": body["winner"]}
    return out, default, sources


def build(viewer_text: str | None) -> dict:
    belief_path = HERE / "pretrained" / "cat_belief.npz"
    patch = BeliefPatch.load(belief_path)
    ctx = json.loads(belief_path.with_suffix(".json").read_text())
    gen, default, sources = genomes(ctx)
    schedule = {**cat.SCHEDULE, "change": "none"}
    body = {
        "format": FORMAT,
        "library": library_info(),
        "viewer": viewer_info(viewer_text) if viewer_text is not None else None,
        "belief": {"npz_sha256": hashlib.sha256(belief_path.read_bytes()).hexdigest(), **belief_state(patch)},
        "scale": [float(v) for v in ctx["scale"]],
        "floors": {k: float(v) for k, v in ctx["routine"].items()},
        "parameters": int(sum(v.size for v in patch.parameters().values())),
        "moment_macs": cat.moment_macs(patch),
        "pretraining": {k: ctx[k] for k in ("seed", "episodes", "length", "batch", "rate", "epochs", "belief", "encoded", "clip", "fill_store")},
        "held_out": {"explained_one_step": ctx["with_store"]["explained_one_step"], "imagination": ctx["with_store"]["imagination"]},
        "machinery": dict(cat.MACHINERY),
        "prices": {"moments": cat.PRICE_MOMENTS, "surprise": cat.PRICE_SURPRISE},
        "world": {"grid": cat.GRID, "readings": cat.READINGS, "paw_speed": cat.PAW_SPEED, "catch_radius": cat.CATCH_RADIUS, "catch_hold": cat.CATCH_HOLD, "effort": cat.EFFORT, "clip": cat.CLIP,
                  "faster": cat.FASTER, "gravity": cat.GRAVITY, "vmax": cat.VMAX, "schedule": schedule,
                  "retina_grid": cat._GRID.tolist(), "retina_sigma": float(cat._SIGMA), "directions": cat.DIRECTIONS.tolist()},
        "governor": {"readback": list(cat.READBACK), "cortex": cat.CORTEX, "motor": list(cat.MOTOR), "neuron": cat.GOVERNOR_MODEL.to_dict(), "settle": {"budget": 200, "chunk": 10, "tolerance": 1e-3}},
        "genomes": gen,
        "default_genome": default,
        "genome_sources": sources,
    }
    return body


def strip(body: dict) -> dict:
    return {k: v for k, v in body.items() if k not in PROVENANCE}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=None, help="the directory for brain.json (default: web/data beside this file)")
    p.add_argument("--no-viewer", action="store_true", help="leave web/brain_scan.js alone")
    p.add_argument("--check", action="store_true", help="rebuild in memory and compare with web/data/brain.json, provenance aside")
    a = p.parse_args()
    viewer_text = brain_scan_script()
    body = build(viewer_text)
    committed = HERE / "web" / "data" / "brain.json"
    if a.check:
        current = json.loads(committed.read_text())
        if strip(current) != strip(body):
            mine, theirs = strip(body), strip(current)
            for key in sorted(set(mine) | set(theirs)):
                if mine.get(key) != theirs.get(key):
                    print(f"FAIL: brain.json differs at {key!r}")
            return 1
        print(f"ok: web/data/brain.json is the export of pretrained/cat_belief.npz and the receipts ({committed.stat().st_size / 1e6:.2f} MB)")
        return 0
    out = a.out or committed.parent
    out.mkdir(parents=True, exist_ok=True)
    (out / "brain.json").write_text(json.dumps(body, separators=(",", ":")))
    print(f"wrote {out / 'brain.json'} ({(out / 'brain.json').stat().st_size / 1e6:.2f} MB): {body['parameters']} parameters, genomes {', '.join(f'{w}={d}' for w, d in body['default_genome'].items())}")
    if not a.no_viewer:
        target = HERE / "web" / "brain_scan.js"
        target.write_text(viewer_text, encoding="utf-8")
        v = body["viewer"]
        print(f"copied brain_scan.js ({v['version']}, style {v['style']}, sha256 {v['sha256'][:12]}, library commit {v['commit']}{' with uncommitted changes' if v['uncommitted_changes'] else ''})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
