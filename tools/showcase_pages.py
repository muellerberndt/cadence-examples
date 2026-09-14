"""Real-browser interactions for the four browser demos, including image input."""

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


def choose(page, mode):
    page.locator(f'[data-tab="{mode}"]').click()
    page.wait_for_function(
        "mode => window.showcase?.snapshot().mode === mode && window.showcase.snapshot().brain.neurons > 0",
        arg=mode,
    )


def main():
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(Quiet, directory=str(ROOT))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    screenshots = ROOT / "runs" / "browser"
    screenshots.mkdir(parents=True, exist_ok=True)
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
            page.goto(f"http://127.0.0.1:{server.server_address[1]}/eye-arm/")
            page.wait_for_function("window.showcase !== undefined")
            assert page.evaluate("showcase.snapshot().mode") == "arm"
            page.wait_for_function("showcase.snapshot().brain.neurons > 0")
            topology=page.evaluate("showcase.snapshot().brain.topology")
            assert topology["neurons"]==page.evaluate("showcase.snapshot().brain.neurons")
            assert topology["allEdgesSubmitted"]
            page.screenshot(path=str(screenshots / "arm-initial.png"))
            brain_bounds=page.locator("#brain-scene").bounding_box()
            page.mouse.move(brain_bounds["x"]+brain_bounds["width"]*.5,brain_bounds["y"]+brain_bounds["height"]*.5)
            page.mouse.wheel(0,-400)
            page.wait_for_function("showcase.snapshot().brain.topology.zoom > 1")
            page.locator("#brain-fit").click()
            assert page.evaluate("showcase.snapshot().brain.topology.zoom")==1
            # The composer is linked from every demo's navigation and from the gallery.
            assert page.locator('.tabs a[href="composer/"]').count() == 1
            page.goto(f"http://127.0.0.1:{server.server_address[1]}/composer/")
            expect(page.locator("h1")).to_contain_text("Hear")
            page.goto(f"http://127.0.0.1:{server.server_address[1]}/eye-arm/")
            page.wait_for_function("window.showcase?.snapshot().brain.neurons > 0")
            choose(page, "worm")
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
            # No cue: the controller rests, and the canvas must not replay the last step.
            page.locator("#worm-clear-food").click()
            page.wait_for_timeout(700)
            idle_moves = page.evaluate("showcase.snapshot().body.moves")
            idle_image = page.locator("#scene").screenshot()
            page.wait_for_timeout(450)
            assert page.evaluate("showcase.snapshot().body.moves") == idle_moves
            assert page.locator("#scene").screenshot() == idle_image
            page.locator("#worm-pause").click()
            # Draw a continuous wall across multiple input events, then erase it.
            page.locator('[data-worm-tool="wall"]').click()
            page.locator("#scene").scroll_into_view_if_needed()
            bounds = page.locator("#scene").bounding_box()
            size = min((bounds["width"] - 32) / 31, (bounds["height"] - 42) / 21)
            left = bounds["x"] + (bounds["width"] - 31 * size) / 2
            top = bounds["y"] + (bounds["height"] - 21 * size) / 2
            page.mouse.move(left + 8.5 * size, top + 3.5 * size)
            page.mouse.down()
            page.mouse.move(left + 15.5 * size, top + 3.5 * size, steps=8)
            page.mouse.up()
            walls = page.evaluate("showcase.snapshot().body.walls")
            assert all(walls[3 * 31 + x] for x in range(8, 16))
            page.locator('[data-worm-tool="erase"]').click()
            page.locator("#scene").scroll_into_view_if_needed()
            bounds = page.locator("#scene").bounding_box()
            left = bounds["x"] + (bounds["width"] - 31 * size) / 2
            top = bounds["y"] + (bounds["height"] - 21 * size) / 2
            page.mouse.click(left + 9.5 * size, top + 3.5 * size)
            assert not page.evaluate("showcase.snapshot().body.walls[3*31+9]")
            page.locator('[data-worm-view="circuit"]').click()
            page.locator('[data-worm-view="habitat"]').click()
            assert page.evaluate("showcase.snapshot().body.walls[3*31+8]")
            assert not page.evaluate("showcase.snapshot().body.walls[3*31+9]")
            choose(page, "fly")
            page.wait_for_function(
                "showcase.snapshot().agents[0].encounters > 0", timeout=20000
            )
            page.locator("#change-nectar").click()
            page.locator("#pause").click()
            expect(page.locator("#pause")).to_have_text("Resume")
            assert page.evaluate("showcase.snapshot().paused")
            before = page.evaluate("showcase.snapshot().agents")
            page.wait_for_timeout(150)
            after = page.evaluate("showcase.snapshot()")
            page.screenshot(path=str(screenshots / "fly-paused.png"))
            (screenshots / "fly-pause.json").write_text(json.dumps({
                "before": before, "after": after["agents"], "paused": after["paused"]}))
            assert before == after["agents"], {"before": before, "after": after}
            choose(page, "arm")
            page.wait_for_function("showcase.snapshot().body.ink > 0", timeout=20000)
            page.wait_for_function(
                "showcase.snapshot().body.phase === 'done'", timeout=60000
            )
            assert page.evaluate("showcase.snapshot().body.coverage") == 1
            assert page.evaluate("showcase.snapshot().body.strokes") == 1
            page.locator("#erase-copy").click()
            page.wait_for_function("showcase.snapshot().body.missing > 0")
            page.wait_for_function(
                "showcase.snapshot().body.phase === 'done' && showcase.snapshot().body.coverage === 1",
                timeout=30000,
            )
            # A freehand mark is read back through the retina, including pencil lift.
            page.locator("#pause").click()
            page.locator("#brain-think").click()
            expect(page.locator(".primary-controls #clear-pad")).to_have_text("Clear")
            page.get_by_role("button", name="Clear", exact=True).click()
            page.wait_for_function("showcase.snapshot().body.targets === 0 && showcase.snapshot().body.ink === 0 && showcase.snapshot().brain.neurons === 17")
            assert not page.evaluate("showcase.snapshot().body.paused")
            assert page.locator("#target-image").input_value() == ""
            assert page.locator("#image-upload").input_value() == ""
            expect(page.locator("#brain-think")).to_have_attribute("aria-pressed", "false")
            page.wait_for_timeout(250)
            assert page.evaluate("showcase.snapshot().body.targets") == 0
            page.screenshot(path=str(screenshots / "arm-cleared.png"))
            page.locator("#scene").scroll_into_view_if_needed()
            bounds = page.locator("#scene").bounding_box()
            size = min(bounds["width"] * 0.40 - 18, bounds["height"] - 88)
            x, y = bounds["x"] + 18, bounds["y"] + 58
            page.mouse.move(x + size * 0.3, y + size * 0.3)
            page.mouse.down()
            page.mouse.move(x + size * 0.7, y + size * 0.3, steps=12)
            page.mouse.move(x + size * 0.7, y + size * 0.7, steps=12)
            page.mouse.up()
            page.wait_for_function("showcase.snapshot().body.targets > 3")
            page.wait_for_function("showcase.snapshot().body.ink > 0", timeout=20000)
            assert page.evaluate("showcase.snapshot().body.retina.some(v=>v>.23)")
            # Install the lesion atomically with reset: two browser round trips
            # can otherwise allow a real motor step before the lesion is applied.
            page.evaluate("document.querySelector('#restart-arm').click(); document.querySelector('#pencil-motors').click()")
            page.wait_for_timeout(600)
            assert page.evaluate("showcase.snapshot().body.ink") == 0
            assert page.evaluate("showcase.snapshot().body.z") == 1
            page.locator("#pencil-motors").click()
            page.wait_for_function("showcase.snapshot().body.ink > 0", timeout=20000)
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
            choose(page, "game")
            # Pondering is on by default, without placing stones or blocking the human.
            page.wait_for_function("showcase.snapshot().body.result?.depth >= 4")
            thought = page.evaluate("showcase.snapshot().body")
            assert thought["evaluated"] > 0
            page.locator("#brain-future").fill("0")
            page.locator("#brain-step-next").click()
            page.wait_for_function("showcase.snapshot().brain.displayed.frame === 1")
            assert page.evaluate("showcase.snapshot().brain.displayed.phase.startsWith('Recorded future')")
            page.screenshot(path=str(screenshots / "game-future-inspection.png"))
            page.locator("#brain-live").click()
            assert thought["background"] and not thought["busy"]
            assert thought["board"] == [0] * 42 and thought["turn"] == 1
            assert page.locator('[data-column="3"]').is_enabled()
            page.locator("#background-thought").click()
            paused = page.evaluate("showcase.snapshot().body")
            assert not paused["background"] and not paused["pondering"]
            page.wait_for_timeout(200)
            assert page.evaluate("showcase.snapshot().body") == paused
            page.locator("#background-thought").click()
            page.wait_for_function("showcase.snapshot().body.result?.depth === 6 && !showcase.snapshot().body.pondering")
            page.locator("#watch-thought").click()
            page.locator('[data-column="3"]').click()
            page.wait_for_function("showcase.snapshot().body.result?.depth >= 4")
            page.wait_for_function(
                "document.querySelector('#game-status').textContent.startsWith('Considering')"
            )
            assert page.evaluate("showcase.snapshot().body.result.cacheHits") > 0
            before = page.evaluate("showcase.snapshot().body.board")
            assert sum(v != 0 for v in before) == 1
            page.locator("#game-futures button").last.click()
            page.locator("#future-step").fill("0")
            assert page.evaluate("showcase.snapshot().body.board") == before
            page.locator("#play-thought").click()
            assert (
                sum(v != 0 for v in page.evaluate("showcase.snapshot().body.board"))
                == 2
            )
            page.locator("#new-game").click()
            assert page.evaluate("showcase.snapshot().body.board") == [0] * 42
            page.locator("#self-monitor").click()
            page.locator("#cadence-first").click()
            page.wait_for_function(
                "document.querySelector('#game-status').textContent.startsWith('Considering')"
            )
            assert page.evaluate("showcase.snapshot().body.result.depth") == 4
            assert page.evaluate("showcase.snapshot().body.board") == [0] * 42
            page.locator("#play-thought").click()
            assert (
                sum(v != 0 for v in page.evaluate("showcase.snapshot().body.board"))
                == 1
            )
            for mode in ["arm", "fly", "worm", "game"]:
                choose(page, mode)
                page.wait_for_function("showcase.snapshot().brain.neurons > 0")
                page.locator("#brain-options").evaluate("el => el.open = true")
                brain = page.evaluate("showcase.snapshot().brain")
                assert (
                    brain["neurons"]
                    == {
                        "arm": (
                            page.evaluate("showcase.snapshot().body.targets * 3 + 17")
                            if mode == "arm" else None
                        ),
                        "fly": 18,
                        "worm": 309,
                        "game": 19,
                    }[mode]
                )
                assert page.locator("#brain-replay").is_enabled() == (
                    mode in ["arm", "worm", "fly", "game"]
                )
                assert brain["equilibrium"]["converged"], (mode, brain["equilibrium"])
                assert (
                    brain["equilibrium"]["residual"]
                    <= brain["equilibrium"]["tolerance"]
                )
                # The iteration controls traverse the whole captured state, one step at a time.
                page.locator("#brain-replay").click()
                page.locator("#brain-timeline").fill("0")
                page.wait_for_function("showcase.snapshot().brain.displayed.frame === 0")
                page.locator("#brain-step-next").click()
                page.wait_for_function("showcase.snapshot().brain.displayed.frame === 1")
                assert len(page.evaluate("showcase.snapshot().brain.displayed.state")) == brain["neurons"]
                page.locator("#brain-live").click()
                if mode == "worm":
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
                    == brain["neurons"]
                )
                if mode == "fly":
                    assert page.evaluate(
                        "showcase.snapshot().brain.input.slice(8,12)"
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
                expect(page.locator("#brain-legend")).to_be_visible()
                expect(page.locator("#brain-legend-label")).to_contain_text(
                    "Total synaptic strength · gold strengthens, blue weakens"
                )
                page.locator("#brain-signal").select_option("activity")
                assert (
                    brain["displayed"]["regions"]
                    == {
                        "arm": [
                            "Retina · reference marks",
                            "Target & proprioception",
                            "Visual / height error",
                            "Joint coordination",
                            "Motor neurons",
                            "Retina · actual ink",
                            "Missing ink",
                        ],
                        "worm": [
                            "Chemical sensory input",
                            "Interneurons",
                            "Chemical motor output",
                            "Directional odor readback",
                            "Body direction motors",
                        ],
                        "fly": [
                            "Flower cue",
                            "Nectar memory",
                            "Visual bearing / approach",
                            "Turn / propulsion motors",
                        ],
                        "game": [
                            "Threat features",
                            "Value evaluator",
                            "Compared futures",
                            "Own activity readback",
                            "Self-monitor / budget",
                        ],
                    }[mode]
                )
                page.locator("#brain-options").evaluate("el => el.open = false")
                page.set_viewport_size({"width": 1366, "height": 768})
                page.evaluate("scrollTo(0,0)")
                scene = page.locator("#scene").bounding_box()
                circuit = page.locator("#brain-scene").bounding_box()
                assert scene["x"] + scene["width"] < circuit["x"]
                for bounds in [scene, circuit, page.locator("#brain-waves").bounding_box(), page.locator("#brain-equilibrium").bounding_box()]:
                    assert bounds["y"] >= 0 and bounds["y"] + bounds["height"] <= 768, (
                        mode,
                        bounds,
                    )
                guide = page.locator("#demo-guide")
                assert guide.get_attribute("data-demo") == mode
                assert guide.locator("h3").count() == 3
                assert guide.locator("h3").last.inner_text() == "Why Cadence fits"
                assert (
                    "Inside the feedback loop"
                    in page.locator(".mechanism summary").inner_text()
                )
                page.screenshot(path=str(screenshots / f"{mode}-desktop.png"))
                page.set_viewport_size({"width": 390, "height": 844})
                page.wait_for_timeout(80)
                assert not page.evaluate(
                    "document.documentElement.scrollWidth > innerWidth"
                ), mode
                assert page.locator("#scene").bounding_box()["height"] >= 300
                for control in page.locator(
                    "#controls button, #controls select, #controls input"
                ).all():
                    if control.is_visible():
                        bounds = control.bounding_box()
                        assert (
                            bounds["x"] >= 0 and bounds["x"] + bounds["width"] <= 390
                        ), (mode, control.get_attribute("id"), bounds)
                page.screenshot(path=str(screenshots / f"{mode}-mobile.png"), full_page=True)
                page.set_viewport_size({"width": 1440, "height": 1050})
            # Real touch events follow the same retinal input path on a phone.
            mobile = b.new_context(
                viewport={"width": 390, "height": 844}, has_touch=True
            )
            touch_page = mobile.new_page()
            touch_page.goto(f"http://127.0.0.1:{server.server_address[1]}/eye-arm/")
            touch_page.wait_for_function("window.showcase?.snapshot().brain.neurons > 0")
            touch_page.locator("#clear-pad").click()
            touch_page.locator("#scene").scroll_into_view_if_needed()
            bounds = touch_page.locator("#scene").bounding_box()
            size = min(bounds["width"] * 0.40 - 18, bounds["height"] - 88)
            x, y = bounds["x"] + 18, bounds["y"] + 58
            cdp = mobile.new_cdp_session(touch_page)
            cdp.send(
                "Input.dispatchTouchEvent",
                {
                    "type": "touchStart",
                    "touchPoints": [{"x": x + 0.25 * size, "y": y + 0.3 * size}],
                },
            )
            for step in range(1, 11):
                cdp.send(
                    "Input.dispatchTouchEvent",
                    {
                        "type": "touchMove",
                        "touchPoints": [
                            {"x": x + (0.25 + 0.05 * step) * size, "y": y + 0.3 * size}
                        ],
                    },
                )
            cdp.send(
                "Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []}
            )
            touch_page.wait_for_function("showcase.snapshot().body.targets > 3")
            mobile.close()
            assert not errors, errors
            print(
                json.dumps(
                    {
                        "demos": 4,
                        "image_upload": True,
                        "mobile_layout": True,
                        "desktop_body_and_circuit_visible": True,
                        "motor_ablation_and_pixel_drawing": True,
                        "isolated_futures_and_monitor_budget": True,
                        "default_pondering_pause_and_cache_reuse": True,
                        "worm_idle_canvas_is_stationary": True,
                        "whole_brain_iteration_controls": True,
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
