"""Draw the page's social card with the page itself: a game in progress, the brain caught mid-thought.

    python connect4/tools/make_card.py            # writes connect4/web/card.jpg (1200 x 630)

The page is served from ``connect4/web`` and driven in a headless browser. The card's layout
is a style sheet injected here, so nothing of it ships in the page. It is rendered at twice
the size and reduced, and saved as a JPEG of modest size: a large PNG makes X fall back to
its small card without an image.
"""
from __future__ import annotations

import functools
import http.server
import io
import threading
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

WEB = Path(__file__).resolve().parents[1] / "web"
CARD = """
  body { padding: 30px 34px 0 !important; overflow: hidden; }
  .masthead { padding: 0 2px 16px !important; align-items: center !important; }
  h1 { font-size: 40px !important; letter-spacing: -0.02em !important; }
  .lede { font-size: 17.5px !important; max-width: none !important; color: #b9c8d6 !important; margin-top: 6px !important; }
  .lede a { border: 0 !important; }
  .facts, .toolbar, .legend, footer, .panel-head .score { display: none !important; }
  :root { --stage: 388px !important; }
  .layout { gap: 14px !important; }
  .panel { padding: 14px !important; }
  .mind .scan { min-height: 0 !important; }
  .columns { height: 64px !important; }
"""
HUMAN = (3, 3, 2, 4, 1)          # the visitor's columns; the brain answers each, the last answer is the one pictured


def main() -> None:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(WEB))
    handler.log_message = lambda *a, **k: None  # noqa: E731
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1200, "height": 630}, device_scale_factor=2)
        page.goto(f"http://127.0.0.1:{server.server_port}/index.html", wait_until="networkidle")
        page.wait_for_function("document.getElementById('state').textContent === 'your move'", timeout=60000)
        page.add_style_tag(content=CARD)
        page.evaluate("document.querySelector('.lede').textContent = 'A Cadence record patch, schooled by a perfect player. Play it and watch it read the boards it imagines.'")
        for column in HUMAN[:-1]:
            page.locator(".col").nth(column).click()
            page.wait_for_function("document.getElementById('state').textContent === 'your move'", timeout=60000)
        time.sleep(0.6)
        page.locator(".col").nth(HUMAN[-1]).click()
        time.sleep(0.5)                                  # mid-thought: the retinas, the context and the record cells are lit
        state = page.locator("#state").text_content()
        shot = page.screenshot(type="png")
        browser.close()
    server.shutdown()
    if state != "the brain is thinking":
        raise SystemExit(f"the picture was not taken mid-thought (the page said {state!r}); rerun")
    image = Image.open(io.BytesIO(shot)).convert("RGB").resize((1200, 630), Image.LANCZOS)
    out = WEB / "card.jpg"
    image.save(out, "JPEG", quality=88, optimize=True, progressive=True)
    print(f"{out}: {image.size[0]} x {image.size[1]}, {out.stat().st_size / 1000:.0f} kB")


if __name__ == "__main__":
    main()
