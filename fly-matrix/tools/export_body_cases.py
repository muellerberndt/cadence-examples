"""Scripted flights through the Python body, written to tests/body_cases.json for web/body.js.

Each case names a start state, a pilot or fixed controls, a list of timed events and the state
after every `every` steps. tests/body.mjs replays the same script through the JavaScript twin
and compares. The file also carries samples of the twin sine, cosine and arctangent.

    python tools/export_body_cases.py             write tests/body_cases.json
    python tools/export_body_cases.py --measure   print the tumble, damping, saccade and speed figures
"""

from __future__ import annotations

import json
import sys
from math import acos, degrees, pi
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import fruitfly.body as B  # noqa: E402
from fruitfly.body import DT, Flight, HandPilot, LAP_ROUTE, hover_trim  # noqa: E402

OUT = ROOT / "tests" / "body_cases.json"


def run_case(case):
    """The script of one case through the body. Returns the samples."""
    st = case["start"]
    fl = Flight(st["position"], st.get("heading", 0.0))
    if "velocity" in st:
        fl.v = list(st["velocity"])
    if "attitude" in st:
        fl.set_attitude(*st["attitude"])
    pilot = None
    if case.get("pilot") is not None:
        pc = case["pilot"]
        pilot = HandPilot(pc["target"], pc.get("heading", 0.0), pc.get("speed", 0.3))
        if pc.get("route") is not None:
            pilot.fly_route(pc["route"])
    fixed = dict(case.get("controls") or hover_trim())
    events = {}
    for ev in case.get("events", []):
        events.setdefault(ev["step"], []).append(ev)
    samples = []
    every = case["every"]
    for k in range(case["steps"]):
        for ev in events.get(k, []):
            if "saccade" in ev:
                pilot.saccade(ev["saccade"])
            if "swat" in ev:
                fl.swat(*ev["swat"])
            if "kick" in ev:
                fl.kick(*ev["kick"])
            if "target" in ev:
                pilot.target = list(ev["target"])
            if "controls" in ev:
                fixed = dict(ev["controls"])
        c = pilot.controls(fl) if pilot is not None else fixed
        fl.step(DT, c)
        if (k + 1) % every == 0:
            samples.append([k + 1, list(fl.p), list(fl.q), list(fl.v), list(fl.w)])
    return samples


def cases():
    hover = {"target": [2.0, 1.5, 1.2], "heading": 0.0, "speed": 0.3}
    return [
        {"name": "hover", "start": {"position": [2.0, 1.5, 1.2]}, "pilot": hover,
         "steps": 10000, "every": 100, "events": []},
        {"name": "saccade", "start": {"position": [2.0, 1.5, 1.2]}, "pilot": hover,
         "steps": 2000, "every": 20, "events": [{"step": 1000, "saccade": pi / 2}]},
        {"name": "lap", "start": {"position": [1.0, 0.8, 1.3]},
         "pilot": {"target": [1.0, 0.8, 1.3], "heading": 0.0, "speed": 0.3, "route": [list(w) for w in LAP_ROUTE]},
         "steps": 40000, "every": 200, "events": []},
        {"name": "swat", "start": {"position": [2.0, 1.5, 1.2]}, "pilot": hover,
         "steps": 4000, "every": 40,
         "events": [{"step": 1000, "swat": [[0.8, 0.3, -0.4], [2.0, 1.0, -1.0], [20.0, -30.0, 10.0]]}]},
        {"name": "wall", "start": {"position": [3.9, 1.5, 1.2], "velocity": [1.0, 0.0, 0.0]}, "pilot": None,
         "controls": hover_trim(), "steps": 4000, "every": 40, "events": []},
        {"name": "landing", "start": {"position": [2.0, 1.5, 0.05]}, "pilot": None,
         "controls": {**hover_trim(), "f": 0.0}, "steps": 2000, "every": 40, "events": []},
        {"name": "tumble", "start": {"position": [2.0, 1.5, 1.2], "attitude": [0.0, 0.05, 0.0]}, "pilot": None,
         "controls": hover_trim(), "steps": 1000, "every": 20, "events": []},
        {"name": "burst", "start": {"position": [0.5, 1.5, 1.3]},
         "pilot": {"target": [3.5, 1.5, 1.3], "heading": 0.0, "speed": 1.0},
         "steps": 8000, "every": 100, "events": []},
    ]


def math_samples():
    """Arguments and values of the twin functions, for a bit-for-bit check."""
    xs = [-40.0 + 80.0 * k / 1999.0 for k in range(2000)] + [-0.5 + k / 1000.0 for k in range(1001)]
    pairs = [(B.sin_(0.7 * k) * (1.0 + 0.001 * k), B.cos_(1.3 * k) * (1.0 + 0.002 * k)) for k in range(1000)]
    pairs += [(0.0, 1.0), (0.0, -1.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 0.0), (1e-300, 1.0), (-1.0, -1.0)]
    us = [-1.0 + 2.0 * k / 999.0 for k in range(1000)]
    return {
        "x": xs, "sin": [B.sin_(x) for x in xs], "cos": [B.cos_(x) for x in xs],
        "yx": [list(p) for p in pairs], "atan2": [B.atan2_(y, x) for y, x in pairs],
        "u": us, "asin": [B.asin_(u) for u in us],
    }


# ---------------------------------------------------------------------------------------------
# Measurements for the report.

def tilt_deg(fl):
    R = fl.rotation()
    c = R[8]
    return degrees(acos(1.0 if c > 1.0 else (-1.0 if c < -1.0 else c)))


def measure_tumble(disturbance=0.05, thresholds=(30, 45, 60, 90)):
    fl = Flight((2.0, 1.5, 1.2))
    fl.set_attitude(0.0, disturbance, 0.0)
    c = hover_trim()
    hit = {}
    for _ in range(int(1.0 / DT)):
        fl.step(DT, c)
        tilt = tilt_deg(fl)
        for th in thresholds:
            if th not in hit and tilt > th:
                hit[th] = fl.t
    return hit


def measure_yaw_damping(w0=20.0):
    fl = Flight()
    fl.w[2] = w0
    c = hover_trim()
    for _ in range(int(0.3 / DT)):
        fl.step(DT, c)
        if abs(fl.w[2]) < 0.05 * w0:
            return fl.t
    return None


def measure_saccade(angle=pi / 2):
    fl = Flight((2.0, 1.5, 1.2))
    pt = HandPilot((2.0, 1.5, 1.2), 0.0, 0.3)
    for _ in range(int(0.5 / DT)):
        fl.step(DT, pt.controls(fl))
    t0 = fl.t
    pt.saccade(angle)
    log = []
    peak = 0.0
    for _ in range(int(0.3 / DT)):
        fl.step(DT, pt.controls(fl))
        log.append((fl.t - t0, abs(B.wrap_(pt.heading - fl.euler()[2]))))
        peak = max(peak, abs(fl.w[2]))
    within = None
    for t, e in reversed(log):
        if e > 5.0 * pi / 180.0:
            break
        within = t
    return within, peak


def measure_hover(seconds=5.0):
    fl = Flight((2.0, 1.5, 1.2))
    pt = HandPilot((2.0, 1.5, 1.2), 0.0, 0.3)
    dev = 0.0
    for _ in range(int(seconds / DT)):
        fl.step(DT, pt.controls(fl))
        dev = max(dev, abs(fl.p[2] - 1.2))
    return dev


def measure_kick(rate=40.0):
    fl = Flight((2.0, 1.5, 1.2))
    pt = HandPilot((2.0, 1.5, 1.2), 0.0, 0.3)
    for _ in range(int(0.5 / DT)):
        fl.step(DT, pt.controls(fl))
    fl.kick(0.0, rate, 0.0)
    t0 = fl.t
    worst = 0.0
    settled = None
    for _ in range(int(1.0 / DT)):
        fl.step(DT, pt.controls(fl))
        worst = max(worst, tilt_deg(fl))
        if settled is None and fl.t - t0 > 0.02 and tilt_deg(fl) < 2.0 and abs(fl.w[1]) < 1.0:
            settled = fl.t - t0
    return worst, settled


def measure_laps(seconds=24.0):
    tr = B.fly_laps(seconds, every=4)
    speeds = sorted(B.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) for _, _, _, v, _ in tr)
    zs = [p[2] for _, p, _, _, _ in tr]
    return speeds[-1], speeds[len(speeds) // 2], min(zs), max(zs)


def measure_burst():
    fl = Flight((0.5, 1.5, 1.3))
    pt = HandPilot((3.5, 1.5, 1.3), 0.0, 1.0)
    vmax = fmax = amax = 0.0
    for _ in range(int(4.0 / DT)):
        c = pt.controls(fl)
        fl.step(DT, c)
        vmax = max(vmax, fl.speed())
        fmax = max(fmax, c["f"])
        amax = max(amax, c["aL"], c["aR"])
    return vmax, fmax, amax


def measure_landing():
    fl = Flight((2.0, 1.5, 0.05))
    c = {**hover_trim(), "f": 0.0}
    for _ in range(int(2.0 / DT)):
        fl.step(DT, c)
        if fl.landed:
            return fl.t, fl.p[2]
    return None, fl.p[2]


def measure():
    kw, kf = B.K_WING, B.K_FCT[1]
    margin = kf * (2.0 * kw / B.MASS + (2.0 * kw * B.HINGE_Z * B.HINGE_Z + kf) / B.INERTIA[1]) / (B.GRAVITY * B.HINGE_Z * B.MASS)
    print(f"K_WING = {kw:g} N s/m, K_FCT = {B.K_FCT} N m s/rad")
    print(f"linear hover pitch stability margin: {margin:.3f} (below 1 is unstable)")
    hit = measure_tumble()
    print("open-loop tumble after a 0.05 rad pitch disturbance: " + ", ".join(f"{k} deg at {v * 1000:.0f} ms" for k, v in sorted(hit.items())))
    print(f"yaw spin 20 rad/s to 1 rad/s: {measure_yaw_damping() * 1000:.1f} ms")
    within, peak = measure_saccade()
    print(f"90 degree saccade: inside 5 degrees from {within * 1000:.1f} ms on, peak yaw rate {peak:.1f} rad/s")
    print(f"hover altitude deviation over 5 s: {measure_hover() * 1000:.3f} mm")
    worst, settled = measure_kick()
    print(f"40 rad/s pitch kick: worst tilt {worst:.1f} deg, level again after {settled * 1000:.0f} ms")
    vmax, vmed, zmin, zmax = measure_laps()
    print(f"laps at speed 0.3: top speed {vmax:.3f} m/s, median {vmed:.3f} m/s, altitude {zmin:.4f} to {zmax:.4f} m")
    vmax, fmax, amax = measure_burst()
    print(f"burst at speed 1.0: top speed {vmax:.3f} m/s, wingbeat up to {fmax:.0f} Hz, amplitude up to {amax:.2f}")
    t, z = measure_landing()
    print(f"landing from 5 cm with the wings off: at rest after {t * 1000:.0f} ms at z = {z * 1000:.3f} mm")


def main():
    if "--measure" in sys.argv:
        measure()
        return
    out = {"dt": DT, "math": math_samples(), "cases": []}
    for case in cases():
        samples = run_case(case)
        out["cases"].append({**case, "samples": samples})
        print(f"{case['name']:8s} {case['steps']:6d} steps, {len(samples)} samples")
    OUT.write_text(json.dumps(out))
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} kB)")


if __name__ == "__main__":
    main()
