#!/usr/bin/env python3
"""Open the built S02 page in headless Chromium, run events through it, save screenshots,
and fail on any page or console error.

    /Users/muellerberndt/Projects/oph-meta/cadence-artist/.venv/bin/python \
        world/web/check_page.py runs/world/web/index.html --events 60

Uses SwiftShader so WebGL2 works without a GPU. Selects the phase named by --phase before
the first event (remembered requests at delay 8 by default: wander, close, request), steps
--events events through window.__page.step(), writes page_start.png, page_mid.png,
page_after.png (the whole viewport), brain_after.png, world_after.png, stores_after.png and
records_after.png (clips; the records clip only when the snapshot has a records head) next
to the page unless --out names another directory, then plays, switches back to
the curriculum, steps --phase-events more, writes page_phase.png, and prints a JSON report
with the renderer snapshot and the page's stats. Every event costs a few seconds here: the
software renderer draws every synapse.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright


def clip_of(page, selector: str) -> dict | None:
    box = page.locator(selector).bounding_box()
    if box is None or not box["width"] or not box["height"]:
        return None
    return {"x": max(0, box["x"]), "y": max(0, box["y"]), "width": box["width"], "height": box["height"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("page")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--events", type=int, default=60)
    parser.add_argument("--phase", default="remember-8", help="the phase selected before the first event (the curriculum otherwise starts with 100 explore steps)")
    parser.add_argument("--phase-events", type=int, default=6, help="events stepped after switching back to the curriculum")
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=1500)
    args = parser.parse_args()
    page_path = Path(args.page).resolve()
    out = args.out or page_path.parent
    out.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    warnings: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": args.width, "height": args.height})
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        page.on("console", lambda m: errors.append(f"console.error: {m.text}") if m.type == "error" else warnings.append(f"console.{m.type}: {m.text}") if m.type == "warning" else None)
        page.goto(page_path.as_uri())
        page.wait_for_function("window.__page && window.__page.ready", timeout=300000)
        time.sleep(0.5)
        if args.phase:
            page.select_option("#phaseSelect", args.phase)
        renderer = page.evaluate("window.__page.scan.snapshot()")
        start = page.evaluate("window.__page.stats")
        viewport = {"x": 0, "y": 0, "width": args.width, "height": args.height}
        page.screenshot(path=str(out / "page_start.png"), clip=viewport, timeout=180000)
        stats = None
        first = None
        timings = []
        for k in range(args.events):
            t0 = time.time()
            stats = page.evaluate("window.__page.step()")
            timings.append(time.time() - t0)
            if k == 0:
                first = {key: stats.get(key) for key in ("last_decision", "controller", "phase", "parameter_version", "events")}
            if k == args.events // 2 - 1:
                time.sleep(0.3)
                page.screenshot(path=str(out / "page_mid.png"), clip=viewport, timeout=180000)
        time.sleep(0.3)
        page.screenshot(path=str(out / "page_after.png"), clip=viewport, timeout=180000)
        page.screenshot(path=str(out / "brain_after.png"), clip=clip_of(page, "#scan"), timeout=180000)
        page.screenshot(path=str(out / "world_after.png"), clip=clip_of(page, "#world"), timeout=180000)
        page.screenshot(path=str(out / "stores_after.png"), clip=clip_of(page, "#places"), timeout=180000)
        records_clip = clip_of(page, "#recordsPanel")
        if records_clip is not None:
            page.screenshot(path=str(out / "records_after.png"), clip=records_clip, timeout=180000)
        # the controls: play a little, switch back to the curriculum, step on, and make sure nothing throws
        page.click("#play")
        time.sleep(1.5)
        page.select_option("#speed", "0")
        page.select_option("#imagine", "1")
        time.sleep(1.5)
        page.click("#play")
        page.select_option("#phaseSelect", "curriculum")
        page.select_option("#view", "change")
        page.click("#fit")
        for _ in range(args.phase_events):
            played = page.evaluate("window.__page.step()")
        phase = page.inner_text("#phase")
        status = page.inner_text("#status")
        phase_label = page.inner_text("#phaseLabel")
        after = page.evaluate("window.__page.scan.snapshot()")
        time.sleep(0.3)
        page.screenshot(path=str(out / "page_phase.png"), clip=viewport, timeout=180000)
        browser.close()
    stats_after = dict(stats or {})
    stats_after.pop("renderer", None)
    played.pop("renderer", None)
    report = {
        "page": str(page_path), "renderer": renderer, "renderer_after": after, "stats_start": {k: v for k, v in start.items() if k != "renderer"},
        "first_event": first, "stats_after_events": stats_after, "stats_after_controls": played, "events": args.events,
        "seconds_per_event": {"mean": sum(timings) / max(1, len(timings)), "max": max(timings) if timings else None},
        "phase_label": phase, "curriculum_label": phase_label, "status": status, "errors": errors, "warnings": warnings[:10],
        "screenshots": [str(out / n) for n in ("page_start.png", "page_mid.png", "page_after.png", "brain_after.png", "world_after.png", "stores_after.png") + (("records_after.png",) if records_clip is not None else ()) + ("page_phase.png",)],
    }
    print(json.dumps(report, indent=1))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
