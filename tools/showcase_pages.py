"""Real-browser interactions for all five demos, including taught tasks and image input."""

import functools
import http.server
import json
import threading
from pathlib import Path

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def main():
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(Quiet, directory=str(ROOT))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as p:
            try:
                b = p.chromium.launch()
            except PlaywrightError:
                b = p.chromium.launch(
                    executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
                )
            page = b.new_page(viewport={"width": 1440, "height": 1050})
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"http://127.0.0.1:{server.server_address[1]}/")
            page.wait_for_function("window.showcase !== undefined")
            assert page.evaluate("showcase.snapshot().mode") == "mouse"
            page.locator("#task-cue").select_option("3")
            page.locator("#perform-task").click()
            assert "unfamiliar" in page.locator("#lesson-status").inner_text()
            page.locator("#teach-goal").select_option("3")
            page.locator("#teach-task").click()
            page.locator("#perform-task").click()
            assert [3, 3] in page.evaluate("showcase.snapshot().body.lessons")
            page.locator("#new-maze").click()
            assert [3, 3] in page.evaluate("showcase.snapshot().body.lessons")
            page.reload()
            page.wait_for_function("window.showcase !== undefined")
            assert [3, 3] in page.evaluate("showcase.snapshot().body.lessons")
            page.locator("#task-cue").select_option("3")
            page.locator("#teach-goal").select_option("2")
            page.locator("#teach-task").click()
            lessons = page.evaluate("showcase.snapshot().body.lessons")
            assert all(r in lessons for r in [[0, 0], [1, 1], [2, 2], [3, 2]])
            page.locator('[data-tab="worm"]').click()
            before = page.evaluate("showcase.snapshot().result.state")
            page.locator("#cut-many").click()
            after = page.evaluate("showcase.snapshot()")
            assert (
                sum(v == 0 for v in after["mask"]) == 32
                and before != after["result"]["state"]
            )
            assert after["result"]["residual"] < 1e-8
            page.locator('[data-tab="memory"]').click()
            page.locator("#value").select_option("3")
            page.locator("#teach").click()
            page.locator("#value").select_option("1")
            page.locator("#teach").click()
            assert page.evaluate("showcase.snapshot().fast[0][1]") == 1
            page.locator("#correlation").select_option("0.9")
            assert page.evaluate("showcase.snapshot().writeCount") == 0
            page.locator('[data-tab="fly"]').click()
            page.wait_for_function(
                "showcase.snapshot().agents[0].encounters > 0", timeout=20000
            )
            page.locator("#change-nectar").click()
            page.locator("#pause").click()
            before = page.evaluate("showcase.snapshot().agents")
            page.wait_for_timeout(150)
            assert before == page.evaluate("showcase.snapshot().agents")
            page.locator('[data-tab="arm"]').click()
            page.wait_for_function("showcase.snapshot().body.ink > 0")
            page.locator("#disturb").click()
            page.locator("#feedback").click()
            assert not page.evaluate("showcase.snapshot().body.feedback")
            page.locator("#restart-arm").click()
            assert page.evaluate("showcase.snapshot().body.feedback")
            # A generated raster is an upload fixture, not a browser implementation shortcut.
            png = page.evaluate(
                """() => {const c=document.createElement('canvas');c.width=c.height=64;const x=c.getContext('2d');x.fillStyle='white';x.fillRect(0,0,64,64);x.strokeStyle='black';x.lineWidth=4;x.strokeRect(12,12,40,40);return c.toDataURL().split(',')[1];}"""
            )
            import base64

            page.locator("#image-upload").set_input_files(
                {
                    "name": "square.png",
                    "mimeType": "image/png",
                    "buffer": base64.b64decode(png),
                }
            )
            page.wait_for_function(
                "document.querySelector('#image-status').textContent.includes('Image received')"
            )
            for mode in ["mouse", "arm", "fly", "worm", "memory"]:
                page.locator(f'[data-tab="{mode}"]').click()
                page.set_viewport_size({"width": 390, "height": 844})
                page.wait_for_timeout(80)
                assert not page.evaluate(
                    "document.documentElement.scrollWidth > innerWidth"
                ), mode
                assert page.locator("canvas").bounding_box()["height"] >= 300
                page.set_viewport_size({"width": 1440, "height": 1050})
            assert not errors, errors
            print(
                json.dumps(
                    {
                        "demos": 5,
                        "task_teaching_and_persistence": True,
                        "image_upload": True,
                        "mobile_layout": True,
                        "browser_errors": errors,
                    }
                )
            )
            b.close()
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
