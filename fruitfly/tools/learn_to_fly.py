#!/usr/bin/env python3
"""The fly learns its steering reflex before birth, on the measured wiring, from a teacher.

The rate model carries no spike timing, so the innate haltere reflex is out of its reach
(tools/reflex_loop.py). What a settling brain can do is learn: the hand-written pilot flies
the body through kicks; at every lesson moment the brain settles free under its senses (the
haltere tone, the two ocelli, HS and VS), then nudged toward the wing motor-neuron pattern
the pilot's controls call for through the inverse of the motor dictionary, and every
synapse of the flight sub-net moves on the centred contrast of its two endpoints (the
library's free/nudged rule). Only existing synapses change. After the first episodes the
brain flies while the pilot only teaches, so its own mistakes are in the lessons. The
control is the same protocol on the shuffled wiring; the evaluation is the closed loop of
tools/reflex_loop.py on kicks the lessons never used.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from cadence import Brain, NeuronModel  # noqa: E402
from cadence.learning import Learner, LearnerConfig  # noqa: E402
from cadence.protocol import shuffled  # noqa: E402
from cadence.receipts import Receipt  # noqa: E402
from fruitfly.banc import MANIFEST_PATH  # noqa: E402
from fruitfly.body import DT, Flight, HandPilot, hover_trim  # noqa: E402
from fruitfly.brain import load_fly  # noqa: E402
from fruitfly.motor import GAIN_AMPLITUDE, GAIN_SHIFT, GAIN_TILT, GROUPS_PER_SIDE  # noqa: E402
from fruitfly.subnet import recruit  # noqa: E402
import reflex_loop as loop  # noqa: E402

LESSON_MS = 5.0  # a lesson every 5 ms of flight
EPISODE_S = 0.4
KICK_AT = 0.1
TEACHER_EPISODES = 5  # the pilot flies these; afterwards the brain flies and the pilot teaches
TAKEOVER_TILT = math.radians(60.0)  # the pilot takes the body back when the brain lets it tilt this far


def inverse_dictionary(controls: dict[str, float]) -> dict[str, float]:
    """Motor-neuron group deltas that the dictionary maps to these controls (hover trim removed)."""
    trim = hover_trim(); out: dict[str, float] = {}
    for side, a, b, s in (("left", "aL", "betaL", "sL"), ("right", "aR", "betaR", "sR")):
        da = (controls[a] - trim[a]) / GAIN_AMPLITUDE
        out[f"mn:wing:b1:{side}"], out[f"mn:wing:b3:{side}"] = max(0.0, da), max(0.0, -da)
        db = (controls[b] - trim[b]) / GAIN_TILT
        out[f"mn:wing:b2:{side}"], out[f"mn:wing:i2:{side}"] = max(0.0, db), max(0.0, -db)
        ds = (controls[s] - trim[s]) * 1e3 / GAIN_SHIFT
        out[f"mn:wing:iii1:{side}"], out[f"mn:wing:iii3:{side}"] = max(0.0, ds), max(0.0, -ds)
    return {k: min(1.0, v) for k, v in out.items()}


class Teaching:
    def __init__(self, brain: Brain, sub, cfg: LearnerConfig, rng: np.random.Generator) -> None:
        self.pilot = loop.BrainPilot(brain, sub)
        C = brain.connectome
        outputs = sorted({i for name in self.pilot.readout_names for i in C.populations[name]})
        self.learner = Learner(brain, outputs, cfg, reciprocal=False, plastic_neurons=np.zeros(C.n, dtype=bool))
        self.rng = rng
        self.groups = {name: np.array(list(C.populations[name])) for name in self.pilot.readout_names}
        self.log: list[dict] = []

    def lesson(self, fl: Flight, teacher_controls: dict[str, float]) -> dict[str, float]:
        L = self.learner; self.pilot.brain = L.brain
        drive = self.pilot.drive_for(fl)[None, :]
        free = L.free(drive, warm=self.pilot.state)
        target = free.activation[0].copy()
        deltas = inverse_dictionary(teacher_controls)
        for name, idx in self.groups.items():
            target[idx] = np.clip(target[idx] + deltas.get(name, 0.0), 0.0, 1.0)
        nudged = L.nudged(drive, free, target)
        opposite = L.nudged(drive, free, target, sign=-1.0)
        before = L.brain.efficacy
        report = L.update(free, nudged, opposite)
        moved = np.abs(L.brain.efficacy - before)
        report["max_step"] = float(moved.max()); report["moved_synapses"] = int((moved > 1e-4).sum())
        self.pilot.brain = L.brain
        self.pilot.state = free  # the running state continues from the free phase
        err = float(np.abs(target[L.output_index] - free.activation[0][L.output_index]).mean())
        report.update({"target_error": err, "free_steps": free.steps, "nudged_steps": nudged.steps})
        return report

    def episode(self, k: int) -> dict:
        fl = Flight((2.0, 1.5, 1.2), 0.0)
        teacher = HandPilot((2.0, 1.5, 1.2), 0.0)
        brain_flies = k >= TEACHER_EPISODES
        self.pilot.rest(fl)
        axis = int(self.rng.integers(3)); sign = float(self.rng.choice([-1.0, 1.0])); mag = float(self.rng.uniform(5.0, 20.0))
        steps = int(round(EPISODE_S / DT)); per_brain = int(round(loop.BRAIN_MS * 1e-3 / DT)); per_lesson = int(round(LESSON_MS * 1e-3 / DT))
        controls = hover_trim(); reports = []; taken_over = False; kick_step = int(round(KICK_AT / DT))
        for step in range(steps):
            if step == kick_step:
                kick = [0.0, 0.0, 0.0]; kick[axis] = sign * mag; fl.kick(*kick)
            teacher.target = list(fl.p)  # the teacher only holds attitude, rates and velocity
            tc = teacher.controls(fl)
            if step % per_lesson == 0 and step >= kick_step - per_lesson:
                reports.append(self.lesson(fl, tc))
            if step % per_brain == 0:
                bc = loop.to_body_controls(self.pilot.step(fl))
            if brain_flies and not taken_over and loop.tilt_of(fl) > TAKEOVER_TILT:
                taken_over = True
            controls = bc if (brain_flies and not taken_over) else tc
            fl.step(DT, controls)
        return {"episode": k, "axis": "roll pitch yaw".split()[axis], "sign": sign, "magnitude": mag, "brain_flies": brain_flies, "taken_over": taken_over,
                "lessons": len(reports), "target_error": float(np.mean([r["target_error"] for r in reports])), "scale_step": float(np.mean([r["scale_step"] for r in reports])),
                "max_step": float(np.max([r["max_step"] for r in reports])), "moved_synapses": float(np.mean([r["moved_synapses"] for r in reports])), "target_error_max": float(np.max([r["target_error"] for r in reports])),
                "tilt_end": loop.tilt_of(fl), "efficacy_abs_mean": float(np.abs(self.learner.brain.efficacy).mean())}


def evaluate(brain: Brain, sub, magnitudes=(15.0, 8.0, 20.0)) -> dict:
    pilot = loop.BrainPilot(brain, sub)
    rows = []
    for mag in magnitudes:
        for axis in (0, 1, 2):
            for sign in (1.0, -1.0):
                pilot.state = None
                saved = loop.KICK; loop.KICK = mag
                try:
                    r = loop.run("brain", pilot, axis, sign, loop.BRAIN_MS)
                finally:
                    loop.KICK = saved
                r["magnitude"] = mag; r.pop("trace", None); r.pop("samples", None); rows.append(r)
    return {"tumbled": int(sum(r["tumbled"] for r in rows)), "total": len(rows), "mean_tilt_end_deg": float(np.degrees(np.mean([r["tilt_end"] for r in rows]))),
            "mean_rate_end": float(np.mean([r["rate_end"] for r in rows])), "by_magnitude": {str(m): int(sum(r["tumbled"] for r in rows if r["magnitude"] == m)) for m in magnitudes}, "runs": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30); ap.add_argument("--eta", type=float, default=0.1); ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--conditions", nargs="+", default=["connectome", "shuffled:0"]); ap.add_argument("--eval-every", type=int, default=5); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "runs" / "learn")); ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    gain = float(json.loads((ROOT / "receipts" / "g2_reflex_facts.json").read_text())["body"]["connectome"]["gain"])
    fly = load_fly(); sub = recruit(fly.connectome, budget=30000, hops=3, min_count=8.0)
    model = NeuronModel(gain=gain)
    cfg = LearnerConfig(beta=args.beta, eta=args.eta, eta_bias=0.0, centered=True, free_steps=20, nudged_steps=40, tolerance=1e-4, nudge="quadratic")
    results = {}
    for cond in args.conditions:
        C = sub.connectome if cond == "connectome" else shuffled(sub.connectome, int(cond.split(":")[1]))
        teaching = Teaching(Brain(C, model), sub, cfg, np.random.default_rng(args.seed))
        t0 = time.time(); curve = [{"episode": 0, **{k: v for k, v in evaluate(teaching.learner.brain, sub, (15.0,)).items() if k != "runs"}}]
        print(f"[{cond}] untrained: tumbled {curve[0]['tumbled']}/{curve[0]['total']} tilt {curve[0]['mean_tilt_end_deg']:.0f} deg", flush=True)
        episodes = []
        for k in range(args.episodes if not args.quick else 3):
            ep = teaching.episode(k); episodes.append(ep)
            print(f"[{cond}] ep {k:3d} {ep['axis']:5s} {ep['sign']:+.0f} {ep['magnitude']:4.1f} rad/s {'brain' if ep['brain_flies'] else 'pilot'}{' takeover' if ep['taken_over'] else ''}: lessons {ep['lessons']}, target err mean {ep['target_error']:.4f} max {ep['target_error_max']:.3f}, max step {ep['max_step']:.2e}, moved {ep['moved_synapses']:.0f}, |eff| {ep['efficacy_abs_mean']:.3f}, tilt end {math.degrees(ep['tilt_end']):.0f} deg, {time.time()-t0:.0f} s", flush=True)
            if (k + 1) % args.eval_every == 0 or args.quick:
                ev = evaluate(teaching.learner.brain, sub, (15.0,)); curve.append({"episode": k + 1, **{kk: v for kk, v in ev.items() if kk != "runs"}})
                print(f"[{cond}] after {k+1} episodes: tumbled {ev['tumbled']}/{ev['total']}, mean tilt {ev['mean_tilt_end_deg']:.0f} deg, mean rate {ev['mean_rate_end']:.1f}", flush=True)
        final = evaluate(teaching.learner.brain, sub)
        print(f"[{cond}] FINAL held-out (8, 15, 20 rad/s): tumbled {final['tumbled']}/{final['total']} by magnitude {final['by_magnitude']}, mean tilt {final['mean_tilt_end_deg']:.0f} deg", flush=True)
        results[cond] = {"curve": curve, "episodes": episodes, "final": final, "config": cfg.to_dict(), "seconds": time.time() - t0}
        np.savez_compressed(out / f"{cond.replace(':', '_')}_brain.npz", efficacy=teaching.learner.brain.efficacy, bias=teaching.learner.brain.bias, members=sub.members)
    body = {"gain": gain, "lesson_ms": LESSON_MS, "episode_s": EPISODE_S, "teacher_episodes": TEACHER_EPISODES, "takeover_tilt": TAKEOVER_TILT, "subnet": {"n": int(sub.n), "edges": int(sub.connectome.synapses)}, "results": results}
    rec = Receipt.build("cadence-fruitfly.learn-to-fly/1", body, sources=[("fruitfly/body.py", ROOT / "fruitfly" / "body.py"), ("fruitfly/senses.py", ROOT / "fruitfly" / "senses.py"), ("fruitfly/motor.py", ROOT / "fruitfly" / "motor.py"), ("fruitfly/subnet.py", ROOT / "fruitfly" / "subnet.py"), ("fruitfly/brain.py", ROOT / "fruitfly" / "brain.py"), ("tools/reflex_loop.py", ROOT / "tools" / "reflex_loop.py"), ("tools/learn_to_fly.py", Path(__file__)), ("fruitfly/fixtures/banc_888_manifest.json", MANIFEST_PATH)])
    path = rec.write(out / ("receipt_quick.json" if args.quick else "receipt.json")); print("receipt", path, rec.digest[:16])


if __name__ == "__main__":
    main()
