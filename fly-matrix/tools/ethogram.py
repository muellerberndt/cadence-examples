#!/usr/bin/env python3
"""Gate 3b: the fly's ethogram on the page against the real fly's numbers (docs/REAL_FLY.md).

    python tools/ethogram.py --seconds 240

Serves web/ on a free port and runs the page in Playwright's Chromium, headless on SwiftShader as the
reversal scenario does (--headed for a window on Metal), for the given simulated seconds under each
condition of the switch (brain, shuffled, instincts), each from a fresh page after a warm-up, and reads
the life layer's counters: saccades per second of flight, mean flight bout, mean sit, landings, grooming
and feeding episodes, the fraction of time sitting, odour decisions, sugar visits. Each summary is set
against the bands of docs/REAL_FLY.md. Writes receipts/g3b_ethogram.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
REAL = {"saccade_rate_per_s": [0.4, 1.0], "bout_s": [30, 90], "sit_s": [2, 60], "sitting_fraction": [0.3, 0.9], "sources": "docs/REAL_FLY.md"}
BANDS = {k: v for k, v in REAL.items() if k != "sources"}
READY = "window.__app && window.__app.S.ready && window.__app.S.learnReady"
RESET = """() => { const L = window.__app.life(); L.ethogram = { saccades: 0, microSaccades: 0, bouts: [], sits: [], grooms: 0, feeds: 0, landings: 0,
  flying_s: 0, sitting_s: 0, boutStart: L.clock, sitStart: L.mode === "flying" ? null : L.clock, decisions: 0 }; return L.clock; }"""
STATE = """() => { const app = window.__app, L = app.life(); return { clock: L.clock, mode: L.mode, events: L.events.slice(-40), fps: app.S.fps }; }"""
READ = """() => { const app = window.__app, L = app.life(); return JSON.parse(JSON.stringify({ ...L.ethogram, mode: L.mode, sugar: L.sugar, hunger: L.hunger,
  visits: L.visits, lastP: L.lastP, decisions: app.S.decisions, lessons: app.S.lessonStats || null, fps: app.S.fps, brainMs: app.S.brainMs, active: app.S.active })); }"""


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def commit() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def mean(xs):
    return (sum(xs) / len(xs)) if xs else None


def in_band(value, band):
    return None if value is None else bool(band[0] <= value <= band[1])


def run_condition(browser, url, cond, args, log):
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)[:300]))
    page.goto(f"{url}?seed={args.seed}&noscan=1&nobloom=1&noviews=1")
    page.wait_for_function(READY, timeout=300000)
    page.evaluate(f"window.__app.setPilot('{cond}'); window.__app.S.speed = {args.speed};")
    t_start = page.evaluate("window.__app.life().clock")
    while page.evaluate("window.__app.life().clock") - t_start < args.warmup:  # the switch's transients pass before the counters start
        time.sleep(1)
    mode0 = page.evaluate("window.__app.life().mode")
    t0 = page.evaluate(RESET)
    wall0 = time.time()
    seen, events, last = set(), [], t0
    while last - t0 < args.seconds:
        time.sleep(2)
        st = page.evaluate(STATE)
        last = st["clock"]
        for t, text in st["events"]:
            if (round(t, 3), text) not in seen:
                seen.add((round(t, 3), text)); events.append((t, text))
    e = page.evaluate(READ)
    wall = time.time() - wall0
    fly = max(1e-9, e["flying_s"])
    bouts = e["bouts"][1:] if mode0 == "flying" else e["bouts"]  # the bout or sit under way at the reset is partial and is left out of the means
    sits = e["sits"][1:] if mode0 != "flying" else e["sits"]
    summary = {
        "saccade_rate_per_s": e["saccades"] / fly, "micro_saccade_rate_per_s": e["microSaccades"] / fly,
        "bout_s": mean(bouts), "sit_s": mean(sits), "bouts": len(bouts), "sits": len(sits), "landings": e["landings"], "grooms": e["grooms"], "feeds": e["feeds"],
        "sitting_fraction": e["sitting_s"] / max(1e-9, e["flying_s"] + e["sitting_s"]), "flying_s": e["flying_s"], "sitting_s": e["sitting_s"],
        "odour_decisions": e["decisions"], "visits": e["visits"], "p_approach": e["lastP"], "fps": e["fps"], "brain_ms_per_step": e["brainMs"],
    }
    summary["in_band"] = {k: in_band(summary[k], band) for k, band in BANDS.items()}
    timeouts = sum(1 for _, x in events if "did not answer" in x)
    out = {"mode_at_reset": mode0, "seconds": last - t0, "wall_s": wall, "summary": summary, "raw": e, "brain_timeouts": timeouts, "page_errors": errors[:5], "events_tail": events[-30:]}
    log(f"{cond}: {json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary.items() if k not in ('visits', 'p_approach', 'in_band')})} in band {summary['in_band']}; "
        f"{len(events)} events, {timeouts} brain timeouts, {len(errors)} page errors, {wall:.0f} s wall")
    page.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None, help="a served page; by default web/ is served on a free port")
    ap.add_argument("--web", default=str(ROOT / "web"))
    ap.add_argument("--seconds", type=float, default=240); ap.add_argument("--warmup", type=float, default=10); ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--conditions", default="brain,shuffled,instincts"); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--headed", action="store_true"); ap.add_argument("--receipt", default=str(ROOT / "receipts" / "g3b_ethogram.json"))
    args = ap.parse_args()
    log = lambda *a: print(*a, flush=True)
    srv = None; url = args.url
    if url is None:
        s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
        srv = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"], cwd=args.web, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.8); url = f"http://127.0.0.1:{port}/index.html"
    data = Path(args.web) / "data"
    out = {"tool": "tools/ethogram.py", "page": commit(), "payload": {p.name: sha(p) for p in sorted(data.glob("*.json"))}, "seed": args.seed, "speed": args.speed,
           "seconds": args.seconds, "warmup_s": args.warmup, "headless": not args.headed, "real": REAL, "conditions": {}}
    try:
        with sync_playwright() as p:
            chrome_args = ["--use-angle=metal", "--window-size=1400,900"] if args.headed else ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--window-size=1400,900"]
            browser = p.chromium.launch(headless=not args.headed, args=chrome_args)
            for cond in args.conditions.split(","):
                try:
                    out["conditions"][cond] = run_condition(browser, url, cond, args, log)
                except Exception as exc:  # a lost page loses one condition, not the receipt
                    out["conditions"][cond] = {"error": str(exc)[:300]}
                    log(cond, "failed:", str(exc)[:200])
            browser.close()
    finally:
        if srv: srv.terminate()
    Path(args.receipt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.receipt).write_text(json.dumps(out, indent=1))
    log(f"receipt: {os.path.relpath(args.receipt, ROOT)}")


if __name__ == "__main__":
    main()
