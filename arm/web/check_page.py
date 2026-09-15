#!/usr/bin/env python3
"""Open the built S01 arm page in headless Chromium, run events through it, save screenshots,
and fail on any page or console error.

    /Users/muellerberndt/Projects/oph-meta/cadence-artist/.venv/bin/python \
        arm/web/check_page.py runs/arm/web/index.html --events 50

Uses SwiftShader so WebGL2 works without a GPU. Writes page_start.png, page_mid.png,
page_after.png (the whole viewport), brain_after.png, arm_after.png and records_after.png
(clips of the canvases; the records clip only when the snapshot has a records head) next to
the page unless --out names another directory, and prints a JSON report with the renderer
snapshot and the page's stats after the events.
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
    parser.add_argument("--events", type=int, default=50)
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=1250)
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
                first = {key: stats.get(key) for key in ("last_decision", "controller", "continued", "parameter_version", "events")}
            if k == args.events // 2 - 1:
                time.sleep(0.3)
                page.screenshot(path=str(out / "page_mid.png"), clip=viewport, timeout=180000)
        time.sleep(0.3)
        page.screenshot(path=str(out / "page_after.png"), clip=viewport, timeout=180000)
        page.screenshot(path=str(out / "brain_after.png"), clip=clip_of(page, "#scan"), timeout=180000)
        page.screenshot(path=str(out / "arm_after.png"), clip=clip_of(page, "#arm"), timeout=180000)
        records_clip = clip_of(page, "#recordsPanel")
        if records_clip is not None:
            page.screenshot(path=str(out / "records_after.png"), clip=records_clip, timeout=180000)
        # the controls: play a little, change the target and the body, and make sure nothing throws
        page.click("#play")
        time.sleep(1.5)
        page.click("#target")
        page.click("#lengthen")
        time.sleep(1.5)
        page.click("#play")
        page.select_option("#mode", "circle")
        page.click("#resetBody")
        played = page.evaluate("window.__page.step()")
        phase = page.inner_text("#phase")
        status = page.inner_text("#status")
        after = page.evaluate("window.__page.scan.snapshot()")
        browser.close()
    stats_after = dict(stats or {})
    stats_after.pop("renderer", None)
    played.pop("renderer", None)
    report = {
        "page": str(page_path), "renderer": renderer, "renderer_after": after, "stats_start": {k: v for k, v in start.items() if k != "renderer"},
        "first_event": first, "stats_after_events": stats_after, "stats_after_controls": played, "events": args.events,
        "seconds_per_event": {"mean": sum(timings) / max(1, len(timings)), "max": max(timings) if timings else None},
        "phase_label": phase, "status": status, "errors": errors, "warnings": warnings[:10],
        "screenshots": [str(out / n) for n in ("page_start.png", "page_mid.png", "page_after.png", "brain_after.png", "arm_after.png") + (("records_after.png",) if records_clip is not None else ())],
    }
    print(json.dumps(report, indent=1))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
