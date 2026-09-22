"""Photograph the four example pages in action, one screenshot.png per example.

    python tools/screenshots.py                 # all four
    python tools/screenshots.py worm amen       # only these

Each page is served from its web/ folder, driven to a moment worth showing (the worm mid-lesson,
Connect Four mid-thought, Amen playing a dub, Patch World with a population), rendered at
1280 x 800 at twice the resolution and reduced. Needs playwright with its Chromium
(python -m playwright install chromium) and pillow; neither is a dependency of the checks.
"""
from __future__ import annotations

import functools
import http.server
import io
import sys
import threading
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SIZE = (1280, 800)


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


PAGES = {"worm": worm, "connect4": connect4, "amen": amen, "patchworld": patchworld}


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
            page = browser.new_page(viewport={"width": SIZE[0], "height": SIZE[1]}, device_scale_factor=2)
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
            Image.open(io.BytesIO(shot)).convert("RGB").resize(SIZE, Image.LANCZOS).save(out, "PNG", optimize=True)
            print(f"{out.relative_to(ROOT)}: {SIZE[0]} x {SIZE[1]}, {out.stat().st_size / 1000:.0f} kB")
        browser.close()


if __name__ == "__main__":
    main(sys.argv[1:])
