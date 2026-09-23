#!/usr/bin/env python3
"""Record the Python brain's decisions along a scripted dot path, for the browser twin to replay.

    python dozing-cat/tests/record_parity.py                       # writes tests/parity_cases.json
    python dozing-cat/tests/record_parity.py --out /tmp/cases.json

Every case is one life of ``cat.Life`` on a mouse-mode sill whose dot is the scripted path (the
laser off for a while, a slow figure that rests once in four seconds, the laser off again, a
fast circle), with one governor and one genome. Per decision it logs the dot, the reading, the
action, the mode, the belief after the moment, the expectation, the residual, the surprise, the
baseline, the slow average, the readback the governor read, the governor's settled activation
and steps, the imagination's candidate costs, the paw and the flags. One case forces a learn
call after the path and logs the window's losses, the update, the refitted habit and the
random starts the refit drew, so the twin can refit from the same starts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import cadence  # noqa: E402
import cat  # noqa: E402
from cadence import BeliefPatch  # noqa: E402

RATE = 25.0


def path(t: int) -> list[float] | None:
    """The scripted dot per decision: off, a slow figure resting one second in four, off, a fast circle."""
    s = t / RATE
    if t < 10:
        return None
    if t < 170:
        u = min(s, 4.0 * math.floor(s / 4.0) + 3.0)  # every fourth second the dot rests where it was
        return [0.5 + 0.38 * math.sin(2 * math.pi * 0.08 * u), 0.5 + 0.34 * math.cos(2 * math.pi * 0.05 * u)]
    if t < 185:
        return None
    return [0.5 + 0.3 * math.cos(2 * math.pi * 0.4 * s), 0.5 + 0.3 * math.sin(2 * math.pi * 0.4 * s)]


class RecordingLife(cat.Life):
    """The experiment's life with the imagination's candidate costs kept for the log."""

    costs: np.ndarray | None = None

    def _imagine(self, r: np.ndarray, z: np.ndarray | None) -> tuple[np.ndarray, int]:
        h = int(self.m["horizon"])
        candidates = np.clip(np.concatenate([float(self.m["spread"]) * cat.DIRECTIONS, cat.habit_act(self.habit, r)[None]]), -1.0, 1.0)
        n = len(candidates)
        zz = np.zeros((n, self.patch.belief)) if z is None else np.repeat(z, n, axis=0)
        rr = np.repeat(r[None], n, axis=0)
        a = candidates.copy()
        cost = np.zeros(n)
        for _ in range(h):
            p = self.patch.imagine(a[:, None, :], state=zz)
            rr[:, cat.PREDICTED] += p.output[:, 0] * self.scale
            zz = p.final_state
            cost += cat.task_cost(rr, a)
            if not self.m["hold"]:
                a = cat.habit_act(self.habit, rr)
        self.costs = cost
        return candidates[int(np.argmin(cost))], n * h


def lst(x) -> list:
    """Twelve significant digits: three orders under the parity tolerance, a third of the bytes."""
    return [float(f"{float(v):.12g}") for v in np.asarray(x).reshape(-1)]


def num(v: float | None) -> float | None:
    return None if v is None else float(f"{float(v):.12g}")


def run_case(name: str, arm: str, which: str, genome: dict, decisions: int, spec: dict, learn_at: tuple[int, ...] = ()) -> dict:
    belief_path = ROOT / "pretrained" / "cat_belief.npz"
    ctx = json.loads(belief_path.with_suffix(".json").read_text())
    patch = BeliefPatch.load(belief_path)
    schedule = {**cat.SCHEDULE, "decisions": 10**9, "change": "none"}
    world = cat.World(schedule, [], np.random.default_rng(0), source="mouse")
    life = RecordingLife(patch, world, genome, np.array(ctx["scale"]), ctx["routine"], arm=arm, seed=0, keep_log=False)
    starts_drawn: list[np.ndarray] = []
    original = cat.random_starts

    def capture(rng: np.random.Generator, n: int) -> np.ndarray:
        out = original(rng, n)
        starts_drawn.append(out)
        return out

    cat.random_starts = capture
    steps = []
    learns = []
    try:
        for t in range(decisions):
            if t in learn_at:
                before = life.patch.parameters()
                starts_drawn.clear()
                entry = life._learn()
                after = life.patch.parameters()
                learns.append({
                    "t": t, "entry": {k: (num(entry[k]) if k in ("loss_before", "loss_after", "update_size") else entry[k]) for k in ("t", "window", "loss_before", "loss_after", "update_size", "valid", "kept", "passes_kept", "refit", "moments")},
                    "habit_before": entry["habit_before"], "habit_after": entry.get("habit_after"), "imagined_cost": num(entry.get("imagined_cost")), "refit_moments": entry.get("refit_moments"),
                    "delta": {k: lst(after[k] - before[k]) for k in after}, "starts": [lst(s) for s in (starts_drawn[-1] if starts_drawn else [])],
                    "baseline_after": num(life.baseline), "state_after": lst(life.patch.state[0]) if life.patch.state is not None else None,
                })
            mouse = path(t)
            world.mouse = None if mouse is None else np.array(mouse)
            life.costs = None
            out = life.decide()
            gov = life.governor
            steps.append({
                "t": t, "mouse": mouse, "reading": lst(life.o[-1]) if t % 8 == 0 else None, "action": lst(out["action"]), "mode": out["mode"],
                "belief": lst(life.patch.state[0]), "expected": lst(out["expected"]), "residual": num(out["residual"]), "surprise": num(out["surprise"]), "baseline": num(out["baseline"]), "slow": num(life.slow),
                "readback": lst(out["readback"]), "governor": None if gov is None else {"activation": lst(gov.activation), "steps": out["governor_steps"], "synapses": gov.synapses},
                "costs": None if life.costs is None else lst(life.costs), "paw": lst(world.paw), "dot": None if world.dot is None else lst(world.dot),
                "flags": {k: bool(v) for k, v in out["flags"].items()}, "moments": num(out["moments"]), "learn": out["learn"] is not None, "imagine_left": life.imagine_left, "above": life.above,
                "target": lst(life.y[-1]),
            })
    finally:
        cat.random_starts = original
    return {"name": name, "arm": arm, "which": which, "genome": genome, "decisions": decisions, "steps": steps, "learns": learns, "totals": {k: v for k, v in life.totals.items() if k not in ("ms",)}, "final_habit": life.habit}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=HERE / "parity_cases.json")
    p.add_argument("--decisions", type=int, default=240)
    p.add_argument("--data", type=Path, default=ROOT / "web" / "data" / "brain.json", help="the export whose genomes and belief the cases use")
    p.add_argument("--learn-at", type=int, nargs="+", default=[120, 200], help="the decisions at which the last case forces a learn call")
    a = p.parse_args()
    spec = json.loads(a.data.read_text())
    G = spec["genomes"]
    cases = [
        run_case("patch evolved, the scripted path", "patch", "patch", G["patch"]["evolved"], a.decisions, spec),
        run_case("thresholds evolved, the scripted path", "threshold", "threshold", G["threshold"]["evolved"], a.decisions, spec),
        run_case("patch hand-set, the scripted path", "patch", "patch", G["patch"]["hand_set"], a.decisions, spec),
        run_case("always awake (thresholds hand-set), the scripted path", "always_awake", "threshold", G["threshold"]["hand_set"], 120, spec),
        run_case("never wakes, the scripted path", "never_wakes", "threshold", G["threshold"]["hand_set"], 120, spec),
        run_case("patch hand-set, learn calls forced at decisions " + ", ".join(str(t) for t in a.learn_at), "patch", "patch", G["patch"]["hand_set"], a.decisions, spec, tuple(a.learn_at)),
    ]
    body = {"format": "cadence-examples.dozing-cat.parity/1", "library": {"version": cadence.__version__}, "belief_npz_sha256": spec["belief"]["npz_sha256"], "rate": RATE, "cases": cases}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(body, separators=(",", ":")))
    for c in cases:
        modes = {m: sum(1 for s in c["steps"] if s["mode"] == m) for m in ("habit", "imagine", "learn")}
        catches = sum(1 for s in c["steps"] if s["flags"]["catch"])
        forced = "; ".join(f"forced learn at {l['t']}: kept={l['entry']['kept']} loss {l['entry']['loss_before']:.4f}->{l['entry']['loss_after']:.4f}" for l in c["learns"])
        print(f"{c['name']}: {c['decisions']} decisions, modes {modes}, catches {catches}, learn calls {c['totals']['learn_calls']}" + (f", {forced}" if forced else ""))
    print(f"wrote {a.out} ({a.out.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
