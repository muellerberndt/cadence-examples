#!/usr/bin/env python3
"""Gate 3b: the fly's ethogram on the page against the real fly's numbers (docs/REAL_FLY.md).

    python tools/ethogram.py --seconds 240 --url http://127.0.0.1:8813/

Runs the page headed (WebGL needs it) for the given simulated seconds under each condition of
the switch (brain, shuffled, instincts) and reads the life layer's counters: saccades per second
of flight, mean flight bout, mean sit, landings, grooming and feeding episodes, the fraction of
time sitting, odour decisions, sugar visits. Writes receipts/g3b_ethogram.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
REAL = {"saccade_rate_per_s": [0.4, 1.0], "bout_s": [30, 90], "sit_s": [2, 60], "sitting_fraction": [0.3, 0.9], "sources": "docs/REAL_FLY.md"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8813/"); ap.add_argument("--seconds", type=float, default=240); ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--conditions", default="brain,shuffled,instincts"); ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    out = {"tool": "tools/ethogram.py", "seconds": args.seconds, "speed": args.speed, "real": REAL, "conditions": {}}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, args=["--use-angle=metal", "--window-size=1400,900"])
        for cond in args.conditions.split(","):
          try:
              page = browser.new_page(viewport={"width": 1400, "height": 900})
              page.goto(f"{args.url}?seed={args.seed}")
              for _ in range(120):
                  if page.evaluate("window.__app && window.__app.S && window.__app.S.ready"):
                      break
                  time.sleep(0.5)
              page.evaluate(f"window.__app.setPilot('{cond}'); window.__app.S.speed = {args.speed};")
              page.evaluate("window.__app.life().ethogram = { saccades: 0, microSaccades: 0, bouts: [], sits: [], grooms: 0, feeds: 0, landings: 0, flying_s: 0, sitting_s: 0, boutStart: window.__app.life().clock, sitStart: null, decisions: 0 }")
              t0 = page.evaluate("window.__app.life().clock")
              while page.evaluate("window.__app.life().clock") - t0 < args.seconds:
                  time.sleep(2)
              e = page.evaluate("JSON.parse(JSON.stringify({ ...window.__app.life().ethogram, visits: window.__app.life().visits, decisions: window.__app.S.decisions, lessons: window.__app.S.lessonStats, fps: window.__app.S.fps, brainMs: window.__app.S.brainMs }))")
              fly = max(1e-9, e["flying_s"])
              summary = {"saccade_rate_per_s": e["saccades"] / fly, "micro_saccade_rate_per_s": e["microSaccades"] / fly, "bout_s": (sum(e["bouts"]) / len(e["bouts"])) if e["bouts"] else None, "sit_s": (sum(e["sits"]) / len(e["sits"])) if e["sits"] else None,
                         "landings": e["landings"], "grooms": e["grooms"], "feeds": e["feeds"], "sitting_fraction": e["sitting_s"] / max(1e-9, e["flying_s"] + e["sitting_s"]), "odour_decisions": e["decisions"], "visits": e["visits"], "lessons": e["lessons"], "fps": e["fps"], "brain_ms_per_step": e["brainMs"]}
              out["conditions"][cond] = {"raw": e, "summary": summary}
              print(cond, json.dumps(summary), flush=True)
              page.close()
          except Exception as exc:  # a closed page loses one condition, not the receipt
            out["conditions"][cond] = {"error": str(exc)[:300]}
            print(cond, "failed:", str(exc)[:200], flush=True)
        browser.close()
    (ROOT / "receipts" / "g3b_ethogram.json").write_text(json.dumps(out, indent=1))
    print("receipt: receipts/g3b_ethogram.json")


if __name__ == "__main__":
    main()
