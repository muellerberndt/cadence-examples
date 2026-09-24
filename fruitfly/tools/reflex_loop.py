#!/usr/bin/env python3
"""Gate 2, the closed loop: does the measured nervous system damp a tumble?

The body (fruitfly.body) flies at hover trim; at 100 ms a kick sets a body rate about one
axis; the senses (halteres by rotation magnitude, ocelli by attitude, HS and VS by the
rotation rates) drive the afferents of the flight sub-net; the sub-net settles one step per
millisecond of simulated time with its state carried; the motor dictionary turns the steering
and power motor neurons into the wing controls. Conditions: open loop (hover trim, nothing
fed back), the measured wiring, the wiring with postsynaptic endpoints permuted (three
seeds), and the hand-written pilot. The wing controls are the hover trim plus the motor
dictionary applied to the change of each motor-neuron group's activation from its resting
state at level hover (the settled state under the level-flight senses), so the standing
ocellar activation of level flight is the trim and only departures from it steer. Everything
is declared before the run; the settle timescale (milliseconds per settling step) is reported
over a small declared set rather than chosen.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cadence import Brain, NeuronModel  # noqa: E402
from cadence.protocol import shuffled  # noqa: E402
from cadence.receipts import Receipt  # noqa: E402
from fruitfly.banc import MANIFEST_PATH  # noqa: E402
from fruitfly.body import DT, Flight, HandPilot, hover_trim  # noqa: E402
from fruitfly.brain import load_fly  # noqa: E402
from fruitfly.motor import GROUPS_PER_SIDE, wing_controls  # noqa: E402
from fruitfly.senses import haltere_tone, ocelli_lr, optic_flow_drive  # noqa: E402
from fruitfly.subnet import recruit  # noqa: E402

BRAIN_MS = 1.0  # milliseconds of simulated time per settling step (declared timescale)
TIMESCALES = (1.0, 0.5)
KICK = 15.0  # rad/s
KICK_AT = 0.1  # s
HORIZON = 0.4  # s
TUMBLE_RATE = 60.0  # rad/s
TUMBLE_TILT = math.pi / 2


def tilt_of(fl: Flight) -> float:
    R = fl.rotation()
    return math.acos(max(-1.0, min(1.0, R[8])))  # angle between body z and world z


def to_body_controls(c: dict[str, float]) -> dict[str, float]:
    return {"aL": c["amplitude_left"], "aR": c["amplitude_right"], "betaL": c["tilt_left"], "betaR": c["tilt_right"],
            "sL": c["shift_left"] * 1e-3, "sR": c["shift_right"] * 1e-3, "f": c["frequency"]}


class BrainPilot:
    def __init__(self, brain: Brain, sub, *, vision: bool = True, halteres: bool = True) -> None:
        self.brain, self.sub = brain, sub
        self.vision, self.halteres = vision, halteres
        self.C = brain.connectome
        self.state = None
        self.amp = brain.neuron_model.stimulus_amplitude
        self.readout_names = [f"{g}:{s}" for g in GROUPS_PER_SIDE for s in ("left", "right") if f"{g}:{s}" in self.C.populations]
        self.members = {name: list(self.C.populations[name]) for name in self.readout_names}
        self.last = wing_controls({})
        self.baseline: dict[str, float] = {}
        self.trace: list = []

    def rest(self, fl: Flight) -> None:
        """The resting activation at level hover: the settled state under the level-flight senses."""
        drive = self.drive_for(fl)
        state = self.brain.settle_batch(drive[None, :], steps=200)
        s = state.activation[0]
        self.baseline = {name: float(s[idx].mean()) for name, idx in self.members.items()}
        self.state = state

    def drive_for(self, fl: Flight) -> np.ndarray:
        drive = np.zeros(self.C.n)
        for name, level in self.stimuli(fl).items():
            idx = list(self.C.populations.get(name, ()))
            if idx:
                drive[idx] = np.maximum(drive[idx], self.amp * level)
        return drive

    def stimuli(self, fl: Flight) -> dict[str, float]:
        hl, hr = haltere_tone(self.halteres)
        out = {"haltere:left": hl, "haltere:right": hr}
        if self.vision:
            ol, orr = ocelli_lr(fl.rotation())
            flow = (fl.w[2], fl.w[0], fl.w[1])
            out.update({"ocelli:left": ol, "ocelli:right": orr,
                        "lptc:hs:left": optic_flow_drive(*flow, "left"), "lptc:vs:left": optic_flow_drive(*flow, "left"),
                        "lptc:hs:right": optic_flow_drive(*flow, "right"), "lptc:vs:right": optic_flow_drive(*flow, "right")})
        return out

    def step(self, fl: Flight) -> dict[str, float]:
        drive = self.drive_for(fl)
        self.state = self.brain.settle_batch(drive[None, :], steps=1, state=self.state)
        s = self.state.activation[0]
        delta = {name: float(s[idx].mean()) - self.baseline.get(name, 0.0) for name, idx in self.members.items()}
        self.last = wing_controls(delta)
        return self.last


def run(condition: str, pilot, axis: int, sign: float, brain_ms: float = BRAIN_MS) -> dict:
    fl = Flight((2.0, 1.5, 1.2), 0.0)
    hand = HandPilot((2.0, 1.5, 1.2), 0.0) if condition == "hand" else None
    controls = hover_trim()
    steps = int(round(HORIZON / DT)); kick_step = int(round(KICK_AT / DT)); per_brain = max(1, int(round(brain_ms * 1e-3 / DT)))
    samples = []; peak_after = 0.0; tumbled = False; trace = []
    if pilot is not None:
        pilot.rest(fl)
    for k in range(steps):
        if k == kick_step:
            kick = [0.0, 0.0, 0.0]; kick[axis] = sign * KICK; fl.kick(*kick)
        if condition == "hand":
            controls = hand.controls(fl)
        elif condition != "open" and k % per_brain == 0:
            controls = to_body_controls(pilot.step(fl))
        fl.step(DT, controls)
        rate = math.sqrt(fl.w[0] ** 2 + fl.w[1] ** 2 + fl.w[2] ** 2); tilt = tilt_of(fl)
        if k % 20 == 0 and condition not in ("open", "hand"):
            e = fl.euler(); c = controls
            trace.append([round(fl.t, 4), round(e[0], 3), round(e[1], 3), round(e[2], 3), round(c["aL"], 3), round(c["aR"], 3), round(c["sL"] * 1e3, 3), round(c["sR"] * 1e3, 3), round(c["betaL"], 3), round(c["betaR"], 3)])
        if k > kick_step + per_brain * 5:
            peak_after = max(peak_after, rate)
        if rate > TUMBLE_RATE or tilt > TUMBLE_TILT:
            tumbled = True
        if k % 40 == 0:
            samples.append((round(fl.t, 4), round(rate, 3), round(tilt, 4)))
    rate_end = math.sqrt(sum(x * x for x in fl.w)); tilt_end = tilt_of(fl)
    return {"condition": condition, "axis": "roll pitch yaw".split()[axis], "sign": sign, "peak_rate_after_50ms": peak_after, "rate_end": rate_end,
            "tilt_end": tilt_end, "tumbled": tumbled, "samples": samples, "brain_ms": brain_ms,
            "trace_columns": ["t", "roll", "pitch", "yaw", "aL", "aR", "sL_mm", "sR_mm", "betaL", "betaR"], "trace": trace}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--budget", type=int, default=30000); ap.add_argument("--hops", type=int, default=3); ap.add_argument("--min-count", type=float, default=8.0)
    ap.add_argument("--timescales", type=float, nargs="+", default=list(TIMESCALES)); ap.add_argument("--trace", action="store_true", help="print the pitch traces of the brain condition")
    args = ap.parse_args()
    gain = float(json.loads((ROOT / "receipts" / "g2_reflex_facts.json").read_text())["body"]["connectome"]["gain"])
    fly = load_fly(); sub = recruit(fly.connectome, budget=args.budget, hops=args.hops, min_count=args.min_count)
    model = NeuronModel(gain=gain)
    measured = Brain(sub.connectome, model)
    conditions = [("open", None), ("hand", None), ("brain", BrainPilot(measured, sub)),
                  ("brain:no_vision", BrainPilot(measured, sub, vision=False)), ("brain:no_halteres", BrainPilot(measured, sub, halteres=False))]
    for seed in range(args.seeds):
        conditions.append((f"shuffled:{seed}", BrainPilot(Brain(shuffled(sub.connectome, seed), model), sub)))
    results = []
    for name, pilot in conditions:
        for brain_ms in (args.timescales if pilot is not None else [args.timescales[0]]):
            rows = []
            for axis in (0, 1, 2):
                for sign in (1.0, -1.0):
                    if pilot is not None:
                        pilot.state = None
                    rows.append(run(name, pilot, axis, sign, brain_ms))
            tumbles = sum(r["tumbled"] for r in rows); mean_end = float(np.mean([r["rate_end"] for r in rows])); mean_tilt = float(np.mean([r["tilt_end"] for r in rows]))
            label = name if pilot is None else f"{name}@{brain_ms}ms"
            print(f"{label:16s} tumbled {tumbles}/6, mean rate at {HORIZON*1000:.0f} ms {mean_end:6.2f} rad/s, mean tilt {math.degrees(mean_tilt):6.1f} deg :: " + " ".join(f"{r['axis'][0]}{'+' if r['sign']>0 else '-'}={r['rate_end']:.1f}/{math.degrees(r['tilt_end']):.0f}" for r in rows))
            if args.trace and name == "brain":
                for r in rows:
                    if r["axis"] == "pitch":
                        print(f"   pitch {'+' if r['sign']>0 else '-'} trace (t, roll, pitch, yaw, aL, aR, sL, sR, betaL, betaR):")
                        for row in r["trace"][8:40:2]: print("     ", row)
            results.append({"condition": name, "brain_ms": brain_ms, "tumbled": tumbles, "mean_rate_end": mean_end, "mean_tilt_end": mean_tilt, "runs": rows})
    body = {"gain": gain, "kick_rad_s": KICK, "kick_at_s": KICK_AT, "horizon_s": HORIZON, "brain_step_ms": BRAIN_MS, "tumble": {"rate": TUMBLE_RATE, "tilt": TUMBLE_TILT},
            "subnet": {"budget": args.budget, "hops": args.hops, "min_count": args.min_count, "n": int(sub.n), "edges": int(sub.connectome.synapses)}, "results": results}
    rec = Receipt.build("cadence-fruitfly.g2-reflex-loop/1", body, sources=[("fruitfly/body.py", ROOT / "fruitfly" / "body.py"), ("fruitfly/senses.py", ROOT / "fruitfly" / "senses.py"), ("fruitfly/motor.py", ROOT / "fruitfly" / "motor.py"), ("fruitfly/subnet.py", ROOT / "fruitfly" / "subnet.py"), ("fruitfly/brain.py", ROOT / "fruitfly" / "brain.py"), ("fruitfly/fixtures/banc_888_manifest.json", MANIFEST_PATH), ("tools/reflex_loop.py", Path(__file__))])
    out = rec.write(ROOT / "receipts" / "g2_reflex_loop.json"); print("receipt", out.name, rec.digest[:16])


if __name__ == "__main__":
    main()
