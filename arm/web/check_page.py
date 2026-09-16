#!/usr/bin/env python3
"""Open the built S01 arm page in headless Chromium, draw a figure on it through pointer
events, let the arm copy it twice, and fail on any page or console error, on a layout that
does not fit, or when the hand's mean tracking error stays above the declared bound.

    /Users/muellerberndt/Projects/oph-meta/cadence-artist/.venv/bin/python \
        arm/web/check_page.py runs/arm/web/index.html

Uses SwiftShader so WebGL2 works without a GPU. The page is loaded with its demo off
(``window.__ARM_PAGE__ = {autoplay: false}``) and paused, so every event is computed through
``window.__page.run()`` and the measurement repeats. The check:

1. measures the layout at 1280 by 800: the arm canvas and the brain scan must both lie inside
   the viewport, side by side, with nothing scrolling sideways;
2. draws a figure eight with the mouse; the page turns the strokes into a path when the
   pointer is released and copies it (page_draw.png halfway through);
3. presses Copy again for a second copy, the brain having learned from the first;
4. recomputes each copy's mean tracking error from the page's log of hand and target
   positions, compares it with the page's own readout, and requires the last copy to track
   below --max-error arm lengths;
5. saves page_after.png, brain_after.png, records_after.png and arm_after.png;
6. exercises the rest of the page: two strokes (a travel segment with the pen up), an example
   figure, Clear, the reaching tab with a placed target and the body change and its return,
   the brain selector when a checkpoint manifest sits beside the page, and the live loop;
7. loads the page at 390 by 844 and requires the arm above the brain with no sideways scroll
   (page_phone.png).

The bound is judged only when the brain has lived at least --tracking-from decisions, because
a brain with a few hundred babbled decisions has no world model worth planning through and
cannot copy anything: on the 2026-09-15 parity fixture (300 babbled decisions) the two copies
of the check figure measure 0.42 and 0.44 arm lengths, against 0.21 for a hand that never
moves. On the same page built from a snapshot babbled for 5,000 decisions they measure 0.063
and 0.067 with the ink 0.010 from the drawing, which is the lag of a hand following a target
that moves at 0.12 arm lengths per second. Everything else the check measures is judged on
every snapshot.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import math
import socketserver
import sys
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

MAX_TRACKING_ERROR = 0.12  # arm lengths, on the last copy; the arm's full reach is 1.0
TRACKING_FROM = 2000  # decisions of experience: below this a brain has no world model to plan through


def check_figure(points: int = 56) -> list[list[float]]:
    """The figure the check draws: a figure eight in the upper half of the workspace, in arm lengths."""
    out = []
    for k in range(points + 1):
        t = 2 * math.pi * k / points
        out.append([0.32 * math.sin(t), 0.55 + 0.16 * math.sin(2 * t)])
    return out


def two_strokes() -> list[list[list[float]]]:
    """Two strokes, so the page has to travel between them with the pen up."""
    wave = [[-0.3 + 0.6 * k / 24, 0.72 + 0.06 * math.sin(math.pi * k / 6)] for k in range(25)]
    line = [[-0.22 + 0.44 * k / 12, 0.38] for k in range(13)]
    return [wave, line]


def draw_strokes(page, strokes: list[list[list[float]]]) -> None:
    """Draw every stroke through pointer events; the positions are mapped first, so the strokes of
    one figure follow each other inside the page's window for adding another stroke."""
    mapped = page.evaluate("(all) => all.map((s) => s.map((p) => window.__page.toClient(p[0], p[1])))", strokes)
    for client in mapped:
        page.mouse.move(client[0][0], client[0][1])
        page.mouse.down()
        for x, y in client[1:]:
            page.mouse.move(x, y)
        page.mouse.up()


def draw_stroke(page, world_points: list[list[float]]) -> None:
    draw_strokes(page, [world_points])


def box_of(page, selector: str) -> dict | None:
    box = page.locator(selector).bounding_box()
    if box is None or not box["width"] or not box["height"]:
        return None
    return {"x": max(0, box["x"]), "y": max(0, box["y"]), "width": box["width"], "height": box["height"]}


def geometry(page, ids: list[str]) -> dict:
    return page.evaluate(
        """(ids) => {
            const out = {viewport: {width: innerWidth, height: innerHeight}, scrollWidth: document.documentElement.scrollWidth, scrollHeight: document.documentElement.scrollHeight};
            for (const id of ids) { const el = document.getElementById(id); if (!el) continue; const r = el.getBoundingClientRect();
                out[id] = {x: r.x, y: r.y, width: r.width, height: r.height, right: r.right, bottom: r.bottom}; }
            return out;
        }""",
        ids,
    )


def inside(box: dict, view: dict, slack: float = 0.5) -> bool:
    return box["x"] >= -slack and box["y"] >= -slack and box["right"] <= view["width"] + slack and box["bottom"] <= view["height"] + slack


def overlap(a: dict, b: dict, slack: float = 0.5) -> bool:
    return not (a["right"] <= b["x"] + slack or b["right"] <= a["x"] + slack or a["bottom"] <= b["y"] + slack or b["bottom"] <= a["y"] + slack)


def mean_error(log: list[list[float]]) -> float | None:
    if not log:
        return None
    return sum(math.hypot(row[0] - row[2], row[1] - row[3]) for row in log) / len(log)


def still_hand(log: list[list[float]]) -> float | None:
    """What a hand that never moved from the first logged position would have measured."""
    if not log:
        return None
    x, y = log[0][0], log[0][1]
    return sum(math.hypot(x - row[2], y - row[3]) for row in log) / len(log)


def serve(folder: Path) -> tuple[str, socketserver.TCPServer]:
    """A page built with checkpoints has to be served, because a file:// page cannot fetch its neighbours."""
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args) -> None:  # the check's own output stays readable
            pass

    handler = functools.partial(Quiet, directory=str(folder))
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{httpd.server_address[1]}", httpd


ACTIVE = "['queued','approach','copying','finishing'].includes(window.__page.copy.state)"


def wait_for_copy(page, timeout: int = 10000) -> None:
    """A copy is on its way: queued for the next episode boundary, or already running."""
    page.wait_for_function(ACTIVE, timeout=timeout)


STRIP_WIDTH = 276  # below this the renderer's own region labels and its caption overlap in the strip


def clipped(page) -> list[str]:
    """Every leaf of the brain panel's controls and instrument row whose text does not fit its box."""
    return page.evaluate(
        """() => {
            const out = [];
            for (const el of document.querySelectorAll("#brain .toolbar *, #brain .readouts *")) {
                if (el.children.length) continue;
                const text = (el.textContent || "").trim();
                if (!text) continue;
                const box = el.getBoundingClientRect();
                if (!box.width || !box.height) continue;
                if (el.scrollWidth > el.clientWidth + 1 || el.scrollHeight > el.clientHeight + 1)
                    out.push(`${el.id || el.className || el.tagName.toLowerCase()} "${text.slice(0, 44)}" needs ${el.scrollWidth} by ${el.scrollHeight} px in ${el.clientWidth} by ${el.clientHeight}`);
            }
            return out;
        }"""
    )


def strip_width(page) -> float:
    return page.evaluate("() => { const s = document.getElementById('strip'); return s ? s.getBoundingClientRect().width : 0; }")


def note(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def run_copy(page, max_events: int, batch: int = 25, midway=None) -> dict:
    """Compute events until the running copy is over; returns how many ran and how long they took."""
    ran, seconds, shot = 0, 0.0, midway is None
    while ran < max_events:
        state = page.evaluate("window.__page.copy")
        running = state.get("running")
        note(f"  copy: {state['state']} · {ran} events · progress {running['progress']:.2f} · mean {running['mean']}" if running else f"  copy: {state['state']} · {ran} events")
        if state["state"] in ("done", "idle"):
            break
        if not shot and state.get("running") and state["running"]["progress"] >= 0.45:
            midway(page)
            shot = True
        t0 = time.time()
        out = page.evaluate(f"window.__page.run({batch})")
        seconds += time.time() - t0
        if not out["ran"]:
            break
        ran += out["ran"]
    return {"events": ran, "seconds_per_event": seconds / max(1, ran)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("page")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--events", type=int, default=500, help="the most events one copy may take")
    parser.add_argument("--max-error", type=float, default=MAX_TRACKING_ERROR, help="the bound on the last copy's mean tracking error, in arm lengths")
    parser.add_argument("--tracking-from", type=int, default=TRACKING_FROM, help="judge the tracking bound only when the brain has lived this many decisions")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--skip-phone", action="store_true")
    args = parser.parse_args()
    page_path = Path(args.page).resolve()
    out = args.out or page_path.parent
    out.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    warnings: list[str] = []
    problems: list[str] = []
    report: dict = {"page": str(page_path), "bound": args.max_error}
    viewport = {"x": 0, "y": 0, "width": args.width, "height": args.height}
    manifest = page_path.parent / "checkpoints.json"
    server = None
    if manifest.exists():
        base, server = serve(page_path.parent)
        url = f"{base}/{page_path.name}"
    else:
        url = page_path.as_uri()
    report["url"] = url

    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": args.width, "height": args.height})
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        page.on("console", lambda m: errors.append(f"console.error: {m.text}") if m.type == "error" else warnings.append(f"console.{m.type}: {m.text}") if m.type == "warning" else None)
        page.add_init_script("window.__ARM_PAGE__ = {autoplay: false};")
        page.goto(url)
        page.wait_for_function("window.__page && window.__page.ready", timeout=300000)
        page.evaluate("window.__page.pause()")
        time.sleep(0.4)
        note(f"page ready at {url}")
        report["renderer"] = page.evaluate("window.__page.scan.snapshot()")
        report["stats_start"] = {k: v for k, v in page.evaluate("window.__page.stats").items() if k != "renderer"}
        experience = report["stats_start"].get("experience", 0)
        judge = experience >= args.tracking_from
        report["experience"] = experience
        report["tracking_judged"] = judge
        note(f"the brain has lived {experience} decisions; the tracking bound is {'judged' if judge else 'reported only'}")

        # 1. the layout: both canvases inside the first screen, side by side
        desktop = geometry(page, ["arm", "scan", "records", "ledger", "armKey"])
        report["layout_desktop"] = desktop
        view = desktop["viewport"]
        if not inside(desktop["arm"], view):
            problems.append(f"the arm canvas is not inside the {view['width']}x{view['height']} viewport: {desktop['arm']}")
        if not inside(desktop["scan"], view):
            problems.append(f"the brain scan is not inside the {view['width']}x{view['height']} viewport: {desktop['scan']}")
        if overlap(desktop["arm"], desktop["scan"]):
            problems.append("the arm canvas and the brain scan overlap")
        if desktop["arm"]["right"] > desktop["scan"]["x"] + 0.5:
            problems.append("the brain scan is not beside the arm canvas")
        if desktop["scrollWidth"] > view["width"] + 1:
            problems.append(f"the page scrolls sideways at {view['width']} px: scrollWidth {desktop['scrollWidth']}")
        cut = clipped(page)
        report["clipped_desktop"] = cut
        if cut:
            problems.append(f"text is cut in the brain panel at {view['width']} px: " + "; ".join(cut))
        report["strip_width"] = strip_width(page)
        if report["strip_width"] < STRIP_WIDTH:
            problems.append(f"the strip chart is {report['strip_width']:.0f} px wide, under the {STRIP_WIDTH} px its labels and caption need")
        page.screenshot(path=str(out / "page_start.png"), clip=viewport, timeout=180000)

        # 2. draw the figure and copy it
        note("layout measured; drawing the figure")
        draw_stroke(page, check_figure())
        wait_for_copy(page)
        report["figure"] = page.evaluate("window.__page.copy")
        first = run_copy(page, args.events, midway=lambda pg: pg.screenshot(path=str(out / "page_draw.png"), clip=viewport, timeout=180000))
        log_first = page.evaluate("window.__page.log('last')")
        result_first = page.evaluate("window.__page.copy.last")
        if result_first is None:
            problems.append(f"the first copy did not finish within {args.events} events")
        report["copy_1"] = {**first, "page_mean": result_first["mean"] if result_first else None, "measured_mean": mean_error(log_first), "still_hand": still_hand(log_first), "decisions": len(log_first)}

        # 3. the same figure again, with everything the brain learned from the first copy
        note("copy 1 done; copy again")
        page.click("#again")
        wait_for_copy(page)
        second = run_copy(page, args.events)
        log_second = page.evaluate("window.__page.log('last')")
        last = page.evaluate("window.__page.copy.last") or {}
        report["copy_2"] = {**second, "page_mean": last.get("mean"), "measured_mean": mean_error(log_second), "still_hand": still_hand(log_second), "decisions": len(log_second), "ink": last.get("ink"), "max": last.get("max")}
        measured = mean_error(log_second)
        if measured is None or len(log_second) < 20 or "mean" not in last:
            problems.append(f"the second copy did not finish within {args.events} events")
        else:
            if abs(measured - last["mean"]) > 1e-9:
                problems.append(f"the page reads {last['mean']:.6f} where the log says {measured:.6f}")
            if judge and measured > args.max_error:
                problems.append(f"the mean tracking error of the last copy is {measured:.4f}, above the bound {args.max_error}")
            if judge and measured >= still_hand(log_second):
                problems.append(f"the copy tracked no better than a hand that never moved ({measured:.4f} against {still_hand(log_second):.4f})")

        # 4. the screenshots of the finished copy
        note("copy 2 done; screenshots")
        page.screenshot(path=str(out / "page_after.png"), clip=viewport, timeout=180000)
        page.screenshot(path=str(out / "page_full.png"), full_page=True, timeout=180000)
        page.screenshot(path=str(out / "arm_after.png"), clip=box_of(page, "#arm"), timeout=180000)
        page.screenshot(path=str(out / "brain_after.png"), clip=box_of(page, "#scan"), timeout=180000)
        records_clip = box_of(page, "#recordsPanel")
        if records_clip is not None:
            page.screenshot(path=str(out / "records_after.png"), clip=records_clip, timeout=180000)

        # 5. the rest of the page
        note("the rest of the controls")
        draw_strokes(page, two_strokes())
        wait_for_copy(page)
        multi = page.evaluate("window.__page.copy")
        report["two_strokes"] = {"strokes": multi["strokes"], "points": multi["points"], "pen_up_points": multi["penUp"]}
        if multi["strokes"] != 2:
            problems.append(f"two strokes became {multi['strokes']} strokes")
        if not multi["penUp"]:
            problems.append("the travel between the two strokes has no pen-up points")
        page.evaluate("window.__page.run(60)")
        page.click(".chip[data-figure='star']")
        wait_for_copy(page)
        page.evaluate("window.__page.run(40)")
        page.click("#clear")
        page.evaluate("window.__page.run(5)")
        report["after_clear"] = page.evaluate("window.__page.copy")["state"]
        page.click("#modeReach")
        page.evaluate("window.__page.run(4)")
        client = page.evaluate("window.__page.toClient(0.45, -0.35)")
        page.mouse.click(client[0], client[1])
        page.evaluate("window.__page.run(30)")
        page.click("#lengthen")
        page.evaluate("window.__page.run(20)")
        page.click("#resetBody")
        page.click("#target")
        page.select_option("#mode", "circle")
        page.evaluate("window.__page.run(20)")
        report["reach_stats"] = {k: v for k, v in page.evaluate("window.__page.stats").items() if k in ("events", "decisions", "episodes", "task", "mode", "lengths", "tick")}
        # the brain selector, when the page was built with a checkpoint manifest beside it
        checkpoints = page.evaluate("window.__page.copy.checkpoints")
        report["checkpoints"] = checkpoints
        if checkpoints:
            page.click("#modeDraw")
            page.select_option("#checkpoint", checkpoints[0])
            page.wait_for_function("window.__page.stats.checkpoint !== 'inlined'", timeout=120000)
            page.click(".chip[data-figure='circle']")  # the brain just loaded copies a figure of its own
            wait_for_copy(page)
            page.evaluate("window.__page.run(40)")
            report["checkpoint_stats"] = {k: v for k, v in page.evaluate("window.__page.stats").items() if k in ("checkpoint", "events", "decisions")}
            page.select_option("#checkpoint", "inlined")
            page.wait_for_function("window.__page.stats.checkpoint === 'inlined'", timeout=120000)
        # the live loop, the way a visitor leaves it running
        page.click("#modeDraw")
        page.click(".chip[data-figure='spiral']")
        page.evaluate("window.__page.play()")
        time.sleep(2.5)
        page.evaluate("window.__page.pause()")
        report["stats_after"] = {k: v for k, v in page.evaluate("window.__page.stats").items() if k != "renderer"}
        report["renderer_after"] = page.evaluate("window.__page.scan.snapshot()")
        report["results"] = page.evaluate("window.__page.copy.results")
        report["phase_label"] = page.inner_text("#phase")
        report["status"] = page.inner_text("#status")

        # 6. phone width: the arm above the brain, nothing scrolling sideways
        if not args.skip_phone:
            note("phone width")
            phone = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=2)
            phone.on("pageerror", lambda e: errors.append(f"phone pageerror: {e}"))
            phone.on("console", lambda m: errors.append(f"phone console.error: {m.text}") if m.type == "error" else None)
            phone.add_init_script("window.__ARM_PAGE__ = {autoplay: false};")
            phone.goto(url)
            phone.wait_for_function("window.__page && window.__page.ready", timeout=300000)
            phone.evaluate("window.__page.pause()")
            draw_stroke(phone, check_figure(40))
            wait_for_copy(phone)
            phone.evaluate("window.__page.run(120)")
            time.sleep(0.4)
            small = geometry(phone, ["arm", "scan", "records"])
            report["layout_phone"] = small
            if small["arm"]["bottom"] > small["scan"]["y"] + 1:
                problems.append("at 390 px the brain scan does not sit below the arm canvas")
            if small["scrollWidth"] > small["viewport"]["width"] + 1:
                problems.append(f"the page scrolls sideways at 390 px: scrollWidth {small['scrollWidth']}")
            cut = clipped(phone)
            report["clipped_phone"] = cut
            if cut:
                problems.append("text is cut in the brain panel at 390 px: " + "; ".join(cut))
            phone.screenshot(path=str(out / "page_phone.png"), full_page=True, timeout=180000)
            phone.close()
        browser.close()
    if server is not None:
        server.shutdown()

    report["errors"] = errors
    report["warnings"] = warnings[:6]
    report["problems"] = problems
    report["screenshots"] = [str(out / name) for name in ("page_start.png", "page_draw.png", "page_after.png", "page_full.png", "arm_after.png", "brain_after.png", "records_after.png", "page_phone.png") if (out / name).exists()]
    report["passed"] = not errors and not problems
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
