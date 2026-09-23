#!/usr/bin/env python3
"""The headless check of the page, served from web/ or live: no console errors, no horizontal
overflow at 1280 and at 390 wide, the cat drawn before any click, the laser switched on by a
click and off by the next, the dot led along a figure and walked into the paw so that a catch
happens, the brain canvas drawing, every brain of the selector run, and a screenshot at each
width. Needs playwright with its Chromium (python -m playwright install chromium); it is not a
dependency of the checks.

    python dozing-cat/tools/check_page.py --dir dozing-cat/web --shots /tmp/catshots --out dozing-cat/receipts/page_check.json
    python dozing-cat/tools/check_page.py --url https://floatingpragma.io/cadence-examples/dozing-cat/
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import math
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
ARMS = ("patch", "threshold", "never_wakes", "always_awake")
BOX = "() => { const c = document.getElementById('view'); const r = c.getBoundingClientRect(); const B = boxOf(r.width, r.height); return { left: r.left, top: r.top, x: B.x, y: B.y, s: B.s }; }"
SAMPLE = """() => { const c = document.getElementById('view'); const img = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; let bright = 0, red = 0;
  for (let i = 0; i < img.length; i += 16) { const R = img[i], G = img[i + 1], B = img[i + 2]; if (R > 60 || G > 60 || B > 80) bright++; if (R > 200 && G < 120) red++; }
  return { width: c.width, height: c.height, bright, red, laser: S.laser, decisions: S.decisions, catches: S.catches, mode: S.frame ? S.frame.out.mode : null, arm: S.arm,
    awake_share: S.recentAwake.length ? S.recentAwake.reduce((a, b) => a + b, 0) / S.recentAwake.length : 0, moments: S.recentMoments.length ? S.recentMoments.reduce((a, b) => a + b, 0) / S.recentMoments.length : 0,
    learn_calls: S.life.totals.learn_calls, kept: S.life.totals.kept, viewer: S.scan.snapshot(), regions: S.scan.atlas.regions.map(r => r.name),
    life: { decisions: S.life.totals.decisions, awake_share: 1 - S.life.totals.decisions.habit / Math.max(1, S.life.totals.decisions.habit + S.life.totals.decisions.imagine + S.life.totals.decisions.learn) } }; }"""


def serve(directory: Path) -> http.server.ThreadingHTTPServer:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    handler.log_message = lambda *a, **k: None  # noqa: E731
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def move(page, box: dict, x: float, y: float) -> None:
    page.mouse.move(box["left"] + box["x"] + x * box["s"], box["top"] + box["y"] + (1 - y) * box["s"])


def click_sill(page, box: dict) -> None:
    page.mouse.click(box["left"] + box["x"] + 0.5 * box["s"], box["top"] + box["y"] + 0.5 * box["s"])
    time.sleep(0.2)


def lit_pixels(page, selector: str) -> int:
    """Bright pixels of an element, from a screenshot of it (a WebGL canvas reads back blank after its frame is shown)."""
    from io import BytesIO

    from PIL import Image

    shot = page.locator(selector).screenshot()
    image = Image.open(BytesIO(shot)).convert("RGB").resize((160, 160))
    return sum(1 for r, g, b in image.getdata() if r > 40 or g > 40 or b > 40)


def drive(page, seconds: float) -> dict:
    """The laser on, the dot led along a slow figure resting once in four seconds, then walked
    into the paw and held still, which the catch rule turns into a catch; the laser stays on."""
    box = page.evaluate(BOX)
    if not page.evaluate("() => S.laser"):
        click_sill(page, box)
    t0 = time.time()
    while time.time() - t0 < seconds:
        s = time.time() - t0
        if int(s) % 4 != 3:
            move(page, box, 0.5 + 0.36 * math.sin(2 * math.pi * 0.08 * s), 0.5 + 0.34 * math.cos(2 * math.pi * 0.05 * s))
        time.sleep(0.05)
    for k in range(20):
        cur, paw = page.evaluate("() => [S.pointer || S.world.paw, S.world.paw]")
        move(page, box, cur[0] + (paw[0] - cur[0]) * (k + 1) / 20, cur[1] + (paw[1] - cur[1]) * (k + 1) / 20)
        time.sleep(0.05)
    paw = page.evaluate("() => S.world.paw")
    move(page, box, paw[0], paw[1])
    time.sleep(0.6)
    return page.evaluate(SAMPLE)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, help="serve this folder (web/)")
    p.add_argument("--url", help="or check a live page")
    p.add_argument("--shots", type=Path, default=Path("/tmp/catshots"))
    p.add_argument("--out", type=Path, default=None, help="write the receipt here")
    p.add_argument("--seconds", type=float, default=8.0)
    a = p.parse_args()
    a.shots.mkdir(parents=True, exist_ok=True)
    server = serve(a.dir) if a.dir else None
    base = a.url or f"http://127.0.0.1:{server.server_port}/"
    errors: list[str] = []
    report: dict = {"kind": "cadence-examples/dozing-cat-page-check/v1", "written": time.strftime("%Y-%m-%dT%H:%M:%S"), "url": base if a.url else "web/ served locally", "problems": []}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=ARGS)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append("console: " + m.text) if m.type == "error" else None)
        page.goto(base, wait_until="networkidle", timeout=120000)
        page.wait_for_function("() => window.S && S.life && S.decisions > 5", timeout=60000)
        time.sleep(1.0)
        report["before_click"] = page.evaluate(SAMPLE)
        report["scroll_width"] = {"1280": page.evaluate("() => document.documentElement.scrollWidth")}
        report["arms"] = {}
        for arm in ARMS:
            page.click(f'button[data-arm="{arm}"]', force=True)
            time.sleep(0.5)
            got = drive(page, a.seconds if arm == "patch" else 3.0)
            got["brain_canvas_lit"] = lit_pixels(page, "#scan")
            got["tiles"] = page.evaluate("() => Array.from(document.querySelectorAll('#tiles .stat')).map(e => e.textContent)")
            got["account"] = page.evaluate("() => document.getElementById('what').textContent.slice(0, 160)")
            report["arms"][arm] = got
            if arm == "patch":
                page.screenshot(path=str(a.shots / "page_1280.png"), full_page=True)
                page.evaluate("() => { document.getElementById('card').open = true; }")
                time.sleep(0.5)
                page.screenshot(path=str(a.shots / "page_1280_card.png"), full_page=True)
                page.evaluate("() => { document.getElementById('card').open = false; }")
        box = page.evaluate(BOX)
        click_sill(page, box)
        report["laser_after_second_click"] = page.evaluate("() => S.laser")
        page.close()
        phone = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=2, is_mobile=True, has_touch=True)
        phone.on("pageerror", lambda e: errors.append("phone: " + str(e)))
        phone.goto(base, wait_until="networkidle", timeout=120000)
        phone.wait_for_function("() => window.S && S.life && S.decisions > 5", timeout=60000)
        report["scroll_width"]["390"] = phone.evaluate("() => document.documentElement.scrollWidth")
        box = phone.evaluate(BOX)
        click_sill(phone, box)
        for k in range(20):
            move(phone, box, 0.5 + 0.3 * math.sin(k / 5), 0.5 + 0.3 * math.cos(k / 5))
            time.sleep(0.08)
        report["phone_390"] = phone.evaluate(SAMPLE)
        phone.screenshot(path=str(a.shots / "page_390.png"), full_page=True)
        phone.close()
        browser.close()
    report["console_errors"] = errors
    arms = report["arms"]
    checks = [
        (errors, "console errors"),
        (report["scroll_width"]["1280"] > 1280 or report["scroll_width"]["390"] > 390, "horizontal overflow"),
        (report["before_click"]["bright"] == 0, "the sill is blank before any click"),
        (report["before_click"]["laser"], "the laser was on before any click"),
        (not arms["patch"]["laser"], "the click did not switch the laser on"),
        (arms["patch"]["red"] == 0, "no laser dot was drawn"),
        (arms["patch"]["catches"] == 0, "the dot walked into the paw and was not caught"),
        (any(v["brain_canvas_lit"] == 0 for v in arms.values()), "the brain canvas is dark"),
        (any(v["viewer"]["neurons"] not in (620, 634) for v in arms.values()), "the viewer does not hold the whole brain"),
        (arms["patch"]["regions"] != ["senses", "evidence", "belief", "records", "prediction", "habit", "readback", "governor cortex", "governor motor"], "the governor patch's regions are not drawn"),
        (arms["never_wakes"]["life"]["awake_share"] != 0 or arms["always_awake"]["life"]["awake_share"] != 1, "the controls are not what they say"),
        (arms["patch"]["viewer"]["view"] != {"yaw": 0.5, "pitch": 0.25}, "the brain view moved on its own"),
        (report["laser_after_second_click"], "the second click did not switch the laser off"),
        (not report["phone_390"]["laser"], "the tap did not switch the laser on at 390"),
    ]
    report["problems"] = [what for bad, what in checks if bad]
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({k: v for k, v in report.items() if k != "arms"}, indent=1, default=str))
    for arm, got in arms.items():
        print(f"{arm}: {got['decisions']} decisions on the page, this life awake {got['life']['awake_share']:.2f}, catches {got['catches']}, moments {got['moments']:.1f}, learn calls {got['learn_calls']} ({got['kept']} kept), viewer {got['viewer']['style']} {got['viewer']['neurons']} neurons, canvas lit {got['brain_canvas_lit']}")
    return 1 if report["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
