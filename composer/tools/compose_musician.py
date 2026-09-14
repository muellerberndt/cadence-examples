"""Compose with the musician from the command line.

    python tools/compose_musician.py --checkpoint runs/<run>/brain.npz --mood "sad slow piano" --bars 32 --record --render
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from composer.musician import Musician
from composer.perform import perform

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(ROOT / "checkpoints/maestro-1/brain.npz"))
    p.add_argument("--mood", default="calm piano")
    p.add_argument("--bars", type=int, default=32)
    p.add_argument("--futures", type=int, default=8)
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--edits", type=int, default=2)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--tempo", type=int, default=100)
    p.add_argument("--key", type=int, default=0)
    p.add_argument("--target-surprise", type=float, default=None, help="defaults to the checkpoint's best held-out surprise")
    p.add_argument("--target-plan-surprise", type=float, default=None, help="defaults to the checkpoint's best held-out plan surprise (version 3)")
    p.add_argument("--record", action="store_true", help="record every settling iteration for the studio")
    p.add_argument("--render", action="store_true")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top", type=int, default=0, help="sample among the top-k expected choices per attribute (0: all)")
    p.add_argument("--settle", type=int, default=96, help="settling iterations per event when playing")
    p.add_argument("--detune", type=float, default=0.0, help="std of noise added to the cortices while imagining (0: none)")
    p.add_argument("--out", default=None, help="output folder (default: runs/musician-compositions/<time>-<seed>)")
    p.add_argument("--backend", default="cpu")
    p.add_argument("--device", default=None)
    a = p.parse_args()
    musician = Musician.load(a.checkpoint, backend=a.backend, device=a.device)
    musician.settle_steps = a.settle
    target = a.target_surprise
    receipt = Path(a.checkpoint).with_name("receipt.json")
    target_plan = a.target_plan_surprise
    if receipt.exists():
        training = json.loads(receipt.read_text())
        target = training.get("best_validation_nll", 1.0) if target is None else target
        target_plan = training.get("best_plan_nll") if target_plan is None else target_plan
    folder = Path(a.out) if a.out else ROOT / "runs/musician-compositions" / f"{time.time_ns()}-{a.seed}"
    def progress(event):
        keys = ("stage", "step", "total", "bar", "accepted", "before", "after", "seconds")
        print(json.dumps({k: v for k, v in event.items() if k in keys}), flush=True)

    result = perform(
        musician,
        a.mood,
        folder=folder,
        key=a.key,
        render=a.render,
        record=a.record,
        checkpoint=a.checkpoint,
        target_surprise=target or 1.0,
        target_plan_surprise=target_plan,
        progress=progress,
        bars=a.bars,
        futures=a.futures,
        horizon=a.horizon,
        edits=a.edits,
        seed=a.seed,
        tempo=a.tempo,
        temperature=a.temperature,
        top=a.top,
        detune=a.detune,
    )
    print(json.dumps({"id": result["id"], "events": len(result["events"]), "draft": result["draft_score"]["score"], "final": result["final_score"]["score"], "edits": [(e["bar"], e["accepted"]) for e in result["edits"]], "seconds": result["seconds"]}))


if __name__ == "__main__":
    main()
