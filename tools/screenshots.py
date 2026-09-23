"""Photograph the five example pages in action, one screenshot.png per example.

    python tools/screenshots.py                 # all five
    python tools/screenshots.py worm amen       # only these

Each page is served from its web/ folder, driven to a moment worth showing (the worm mid-lesson,
Connect Four mid-thought, Amen playing a dub, Patch World with a population, the cat mid-chase
with a catch on the board), rendered at 1280 x 800 (the cat at 1280 x 1000: its sill and brain
row is taller) at twice the resolution and reduced. Needs playwright with its Chromium
(python -m playwright install chromium) and pillow; neither is a dependency of the checks.
"""
from __future__ import annotations

import functools
import http.server
import io
import math
import sys
import threading
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SIZE = (1280, 800)
SIZES = {"dozing-cat": (1280, 1000)}


def serve(directory: Path) -> http.server.ThreadingHTTPServer:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    handler.log_message = lambda *a, **k: None  # noqa: E731
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def worm(page) -> None:
    page.wait_for_timeout(7000)                       # the worm has crawled; the network is warm
    page.click("#treat")                              # food arrives: a lesson, drawn in green
    page.wait_for_timeout(1200)


def connect4(page) -> None:
    ready = "document.getElementById('state').textContent === 'your move'"
    page.wait_for_function(ready, timeout=60000)
    for column in (3, 3, 2, 4):
        page.locator(".col").nth(column).click()
        page.wait_for_function(ready, timeout=60000)
    page.wait_for_timeout(500)
    page.locator(".col").nth(1).click()
    page.wait_for_timeout(450)                        # mid-thought: the reading, the context and the record cells are lit


def amen(page) -> None:
    page.wait_for_function("!document.getElementById('cut').disabled", timeout=120000)
    page.click("#cut")
    status = "document.getElementById('status').textContent.trim()"
    page.wait_for_function(f"{status} !== 'ready'", timeout=30000)
    page.wait_for_function(f"['playing', 'press play', 'ready'].includes({status})", timeout=240000)
    page.wait_for_timeout(3000)                       # the waveform and the brain in time with it


def patchworld(page) -> None:
    page.wait_for_timeout(12000)                      # the first births
    page.click("#pickEldest")
    page.wait_for_timeout(1500)


def dozing_cat(page) -> None:
    page.wait_for_function("() => window.S && S.life && S.decisions > 10", timeout=60000)
    box = page.evaluate("() => { const c = document.getElementById('view'); const r = c.getBoundingClientRect(); const B = boxOf(r.width, r.height); return { left: r.left, top: r.top, x: B.x, y: B.y, s: B.s }; }")
    move = lambda x, y: page.mouse.move(box["left"] + box["x"] + x * box["s"], box["top"] + box["y"] + (1 - y) * box["s"])  # noqa: E731
    page.mouse.click(box["left"] + box["x"] + 0.5 * box["s"], box["top"] + box["y"] + 0.5 * box["s"])  # the laser on
    t0 = time.time()
    while time.time() - t0 < 5.0:                     # the dot led along a figure: the cat wakes and chases
        s = time.time() - t0
        move(0.5 + 0.36 * math.sin(2 * math.pi * 0.08 * s), 0.5 + 0.34 * math.cos(2 * math.pi * 0.05 * s))
        time.sleep(0.04)
    for k in range(20):                               # then walked into the paw: a catch
        cur, paw = page.evaluate("() => [S.pointer || S.world.paw, S.world.paw]")
        move(cur[0] + (paw[0] - cur[0]) * (k + 1) / 20, cur[1] + (paw[1] - cur[1]) * (k + 1) / 20)
        time.sleep(0.04)
    page.wait_for_function("() => S.catches > 0", timeout=15000)
    t0 = time.time()
    while time.time() - t0 < 2.0:                     # and led away again, so the picture shows the chase
        s = time.time() - t0
        move(0.5 + 0.36 * math.sin(2 * math.pi * 0.2 * s + 1.0), 0.5 + 0.34 * math.cos(2 * math.pi * 0.15 * s))
        time.sleep(0.04)


PAGES = {"worm": worm, "connect4": connect4, "amen": amen, "patchworld": patchworld, "dozing-cat": dozing_cat}


def main(names: list[str]) -> None:
    names = names or list(PAGES)
    unknown = [n for n in names if n not in PAGES]
    if unknown:
        raise SystemExit(f"no such example: {', '.join(unknown)}; choose from {', '.join(PAGES)}")
    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--autoplay-policy=no-user-gesture-required",
                                           "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        for name in names:
            server = serve(ROOT / name / "web")
            size = SIZES.get(name, SIZE)
            page = browser.new_page(viewport={"width": size[0], "height": size[1]}, device_scale_factor=2)
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"http://127.0.0.1:{server.server_port}/index.html", wait_until="networkidle")
            PAGES[name](page)
            shot = page.screenshot(type="png")
            page.close()
            server.shutdown()
            if errors:
                raise SystemExit(f"{name}: the page raised {errors[0]}")
            out = ROOT / name / "screenshot.png"
            Image.open(io.BytesIO(shot)).convert("RGB").resize(size, Image.LANCZOS).save(out, "PNG", optimize=True)
            print(f"{out.relative_to(ROOT)}: {size[0]} x {size[1]}, {out.stat().st_size / 1000:.0f} kB")
        browser.close()


if __name__ == "__main__":
    main(sys.argv[1:])
