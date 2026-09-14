"""Real-browser interactions for all five demos, including taught tasks and image input."""

import functools
import http.server
import json
import threading
from pathlib import Path

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import expect, sync_playwright

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
            page.wait_for_function("showcase.snapshot().brain.owners > 0")
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
            page.locator('[data-worm-view="circuit"]').click()
            before = page.evaluate("showcase.snapshot().result.state")
            page.locator("#cut-many").click()
            after = page.evaluate("showcase.snapshot()")
            assert (
                sum(v == 0 for v in after["mask"]) == 32
                and before != after["result"]["state"]
            )
            assert after["result"]["residual"] < 1e-8
            page.locator('[data-worm-view="habitat"]').click()
            page.locator("#worm-pause").click()
            page.locator("#worm-clear-food").click()
            page.locator("#scene").focus()
            page.keyboard.press("ArrowRight")
            page.keyboard.press("Space")
            assert len(page.evaluate("showcase.snapshot().body.food")) == 1
            page.locator("#worm-smell").click()
            page.locator("#worm-pause").click()
            before = page.evaluate("showcase.snapshot().body.moves")
            page.wait_for_timeout(500)
            assert page.evaluate("showcase.snapshot().body.moves") == before
            page.locator("#worm-smell").click()
            page.wait_for_function("showcase.snapshot().body.eaten > 0")
            page.locator("#worm-pause").click()
            # Draw a continuous wall across multiple input events, then erase it.
            bounds = page.locator("#scene").bounding_box()
            size = min((bounds["width"] - 32) / 31, (bounds["height"] - 42) / 21)
            left = bounds["x"] + (bounds["width"] - 31 * size) / 2
            top = bounds["y"] + (bounds["height"] - 21 * size) / 2
            page.locator('[data-worm-tool="wall"]').click()
            page.mouse.move(left + 8.5 * size, top + 3.5 * size)
            page.mouse.down()
            page.mouse.move(left + 15.5 * size, top + 3.5 * size, steps=8)
            page.mouse.up()
            walls = page.evaluate("showcase.snapshot().body.walls")
            assert all(walls[3 * 31 + x] for x in range(8, 16))
            page.locator('[data-worm-tool="erase"]').click()
            page.mouse.click(left + 9.5 * size, top + 3.5 * size)
            assert not page.evaluate("showcase.snapshot().body.walls[3*31+9]")
            page.locator('[data-worm-view="circuit"]').click()
            page.locator('[data-worm-view="habitat"]').click()
            assert page.evaluate("showcase.snapshot().body.walls[3*31+8]")
            assert not page.evaluate("showcase.snapshot().body.walls[3*31+9]")
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
                page.wait_for_function("showcase.snapshot().brain.owners > 0")
                brain = page.evaluate("showcase.snapshot().brain")
                assert (
                    brain["owners"]
                    == {"mouse": 259, "arm": 4, "fly": 12, "worm": 297, "memory": 12}[
                        mode
                    ]
                )
                assert page.locator("#brain-replay").is_enabled() == (
                    mode in ["mouse", "arm", "worm"]
                )
                if mode == "memory":
                    page.locator("#value").select_option("2")
                    page.locator("#teach").click()
                    page.wait_for_function("showcase.snapshot().brain.writes > 0")
                elif mode == "worm":
                    page.locator("#brain-replay").click()
                    snapshot = page.evaluate("showcase.snapshot().brain")
                    assert (
                        max(
                            abs(a - b)
                            for a, b in zip(snapshot["last"], snapshot["state"])
                        )
                        < 1e-8
                    )
                    page.locator("#brain-release").click()
                    page.wait_for_function("showcase.snapshot().brain.frame > 0")
                    assert page.evaluate("showcase.snapshot().brain.release")
                    page.locator("#brain-live").click()
                page.locator("#brain-signal").select_option("input")
                page.wait_for_function("showcase.snapshot().brain.signal === 'input'")
                assert (
                    len(page.evaluate("showcase.snapshot().brain.input"))
                    == brain["owners"]
                )
                if mode in ["memory", "fly"]:
                    assert page.evaluate(
                        "showcase.snapshot().brain.input.slice(8)"
                    ) == [0, 0, 0, 0]
                if not page.evaluate("showcase.snapshot().brain.heatmap"):
                    page.locator("#brain-heat").click()
                page.wait_for_function(
                    "!document.querySelector('#brain-legend').hidden"
                )
                expect(page.locator("#brain-legend-label")).to_contain_text(
                    "Input drive"
                )
                page.locator("#brain-heat").click()
                page.wait_for_function("document.querySelector('#brain-legend').hidden")
                page.locator("#brain-heat").click()
                page.locator("#brain-signal").select_option("plasticity")
                page.wait_for_function("document.querySelector('#brain-legend').hidden")
                page.locator("#brain-signal").select_option("activity")
                guide = page.locator("#demo-guide")
                assert guide.get_attribute("data-demo") == mode
                assert guide.locator("h3").count() == 3
                assert guide.locator("h3").last.inner_text() == "Why Cadence fits"
                assert (
                    "Inside the feedback loop"
                    in page.locator(".mechanism summary").inner_text()
                )
                page.set_viewport_size({"width": 390, "height": 844})
                page.wait_for_timeout(80)
                assert not page.evaluate(
                    "document.documentElement.scrollWidth > innerWidth"
                ), mode
                assert page.locator("#scene").bounding_box()["height"] >= 300
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
