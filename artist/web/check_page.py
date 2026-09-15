#!/usr/bin/env python3
"""Open the built S04 artist page in headless Chromium, draw a figure on it through pointer
events, let the artist draw it twice, and fail on any page or console error, on a layout that
does not fit, on a decision slower than the page's budget, or when the drawing the artist
produces stays below the declared F1 bar.

    /Users/muellerberndt/Projects/oph-meta/cadence-artist/.venv/bin/python \
        artist/web/check_page.py runs/artist/web/index.html

Uses SwiftShader so WebGL2 works without a GPU. The page is loaded with its demo off
(``window.__ARTIST_PAGE__ = {autoplay: false}``) and paused, so every decision is computed
through ``window.__page.run()`` and the measurement repeats. The check:

1. measures the layout at 1280 by 800: the canvas and the brain scan must both lie inside the
   viewport, side by side, with nothing scrolling sideways;
2. draws a figure with the mouse; the page turns the strokes into a target when the pointer is
   released and the artist starts drawing it (page_draw.png part of the way through);
3. presses Draw again for a second drawing, the brain having learned from the first;
4. recomputes the Chamfer distance and the F1 of each drawing from the page's own canvas and
   target, compares them with the page's readout, and requires the better of the two to reach
   --min-f1, which defaults to the largest F1 bar the run's receipt reports (its f1_simple gate);
5. saves page_after.png, brain_after.png, records_after.png and canvas_after.png;
6. exercises the rest of the page: a two-stroke figure, every target family, Clear, the brain
   selector when a checkpoint manifest sits beside the page, and the live loop;
7. loads the page at 390 by 844 and requires the canvas above the brain with no sideways scroll
   (page_phone.png).

The F1 bar is judged only when the brain has lived at least --judge-from decisions, because a
brain that has only scribbled has no ink model worth planning through and marks the canvas
almost at random. The decision budget is judged on every page: a decision of this stage has
100 ms in Python and the page's own budget (--max-decision-ms) is what a visitor waits.
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

MIN_F1 = 0.70  # the fallback bar when the page carries no receipt
JUDGE_FROM = 3000  # decisions of experience: below this a brain has no ink model to plan through
MAX_DECISION_MS = 400.0  # what a visitor waits for one decision on this machine


def check_figure() -> list[list[list[float]]]:
    """The figure the check draws: a wide open V in canvas pixel coordinates (row, column)."""
    top, low, left, right = 9.0, 23.0, 7.0, 25.0
    first = [[top + (low - top) * k / 16, left + (16.0 - left) * k / 16] for k in range(17)]
    second = [[low + (top - low) * k / 16, 16.0 + (right - 16.0) * k / 16] for k in range(17)]
    return [first + second[1:]]


def two_strokes() -> list[list[list[float]]]:
    """Two strokes, so the artist has to lift the pen and travel between them."""
    line = [[9.0, 7.0 + 18.0 * k / 16] for k in range(17)]
    arc = [[20.0 + 4.0 * math.sin(math.pi * k / 16), 9.0 + 14.0 * k / 16] for k in range(17)]
    return [line, arc]


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


# -- the measures, recomputed here from the page's own canvas and target


def points_of(image: list[int], size: int) -> list[tuple[int, int]]:
    return [(k // size, k % size) for k, v in enumerate(image) if v]


def nearest(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> list[float]:
    return [min(math.dist(p, q) for q in b) for p in a]


def measures(log: dict, size: int = 32) -> dict:
    drawn, target = points_of(log["canvas"], size), points_of(log["target"], size)
    if not drawn or not target:
        return {"chamfer": 0.0 if not drawn and not target else 1.0, "f1": 1.0 if not drawn and not target else 0.0, "ink": len(drawn), "target": len(target)}
    d, t = nearest(drawn, target), nearest(target, drawn)
    diagonal = math.hypot(size, size)
    precision = sum(1 for x in d if x <= 1.0 + 1e-9) / len(d)
    recall = sum(1 for x in t if x <= 1.0 + 1e-9) / len(t)
    total = precision + recall
    return {
        "chamfer": 0.5 * (sum(d) / len(d) + sum(t) / len(t)) / diagonal,
        "f1": (2 * precision * recall / total) if total else 0.0,
        "precision": precision, "recall": recall, "ink": len(drawn), "target": len(target),
    }


def gate_bar(receipt: dict | None) -> float:
    """The F1 bar the run's receipt reports: the largest threshold of its F1 predicates."""
    gates = (receipt or {}).get("gates") or []
    thresholds = [float(g["threshold"]) for g in gates if str(g.get("name", "")).startswith("f1") and g.get("threshold") is not None]
    return max(thresholds) if thresholds else MIN_F1


def serve(folder: Path) -> tuple[str, socketserver.TCPServer]:
    """A page built with checkpoints has to be served, because a file:// page cannot fetch its neighbours."""
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

    handler = functools.partial(Quiet, directory=str(folder))
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{httpd.server_address[1]}", httpd


ACTIVE = "['queued','drawing'].includes(window.__page.drawing.state)"


def wait_for_drawing(page, timeout: int = 15000) -> None:
    """A drawing is on its way: queued for the next event, or already running."""
    page.wait_for_function(ACTIVE, timeout=timeout)


def note(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def run_drawing(page, max_events: int, batch: int = 10, midway=None) -> dict:
    """Compute decisions until the running drawing is over; returns how many ran and how long they took."""
    ran, seconds, shot = 0, 0.0, midway is None
    while ran < max_events:
        state = page.evaluate("window.__page.drawing")
        running = state.get("running")
        note(f"  drawing: {state['state']} · {ran} decisions · F1 {running['f1']:.3f} · Chamfer {running['chamfer']:.4f}" if running else f"  drawing: {state['state']} · {ran} decisions")
        if state["state"] in ("done", "idle"):
            break
        if not shot and running and ran >= 30:
            midway(page)
            shot = True
        t0 = time.time()
        out = page.evaluate(f"window.__page.run({batch})")
        seconds += time.time() - t0
        if not out["ran"]:
            break
        ran += out["ran"]
    return {"decisions": ran, "seconds_per_decision": seconds / max(1, ran)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("page")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--events", type=int, default=240, help="the most decisions one drawing may take")
    parser.add_argument("--min-f1", type=float, default=None, help="the bar the better of the two drawings must reach (default: the receipt's weakest family)")
    parser.add_argument("--judge-from", type=int, default=JUDGE_FROM, help="judge the F1 bar only when the brain has lived this many decisions")
    parser.add_argument("--max-decision-ms", type=float, default=MAX_DECISION_MS, help="the page's budget for one decision, measured here")
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
    report: dict = {"page": str(page_path)}
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
        page.set_default_navigation_timeout(300000)  # the page carries an 18 MB snapshot
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        page.on("console", lambda m: errors.append(f"console.error: {m.text}") if m.type == "error" else warnings.append(f"console.{m.type}: {m.text}") if m.type == "warning" else None)
        page.add_init_script("window.__ARTIST_PAGE__ = {autoplay: false};")
        page.goto(url)
        page.wait_for_function("window.__page && window.__page.ready", timeout=300000)
        page.evaluate("window.__page.pause()")
        time.sleep(0.4)
        note(f"page ready at {url}")
        report["renderer"] = page.evaluate("window.__page.scan.snapshot()")
        report["stats_start"] = {k: v for k, v in page.evaluate("window.__page.stats").items() if k not in ("renderer", "drawing")}
        receipt = page.evaluate("(window.__SNAPSHOT__ && window.__SNAPSHOT__.extra && window.__SNAPSHOT__.extra.receipt) || null")
        report["receipt"] = receipt
        bar = args.min_f1 if args.min_f1 is not None else gate_bar(receipt)
        report["bar"] = bar
        experience = report["stats_start"].get("experience", 0)
        judge = experience >= args.judge_from
        report["experience"] = experience
        report["f1_judged"] = judge
        note(f"the brain has lived {experience} decisions; the F1 bar {bar:.3f} is {'judged' if judge else 'reported only'}")

        # 1. the layout: both canvases inside the first screen, side by side
        desktop = geometry(page, ["canvas", "scan", "records", "ledger", "canvasKey", "receipt", "body", "brain"])
        report["layout_desktop"] = desktop
        view = desktop["viewport"]
        if not inside(desktop["canvas"], view):
            problems.append(f"the canvas is not inside the {view['width']}x{view['height']} viewport: {desktop['canvas']}")
        if not inside(desktop["scan"], view):
            problems.append(f"the brain scan is not inside the {view['width']}x{view['height']} viewport: {desktop['scan']}")
        if overlap(desktop["canvas"], desktop["scan"]):
            problems.append("the canvas and the brain scan overlap")
        if desktop["canvas"]["right"] > desktop["scan"]["x"] + 0.5:
            problems.append("the brain scan is not beside the canvas")
        if desktop["scrollWidth"] > view["width"] + 1:
            problems.append(f"the page scrolls sideways at {view['width']} px: scrollWidth {desktop['scrollWidth']}")
        for name in ("body", "brain"):
            if desktop[name]["bottom"] > view["height"] + 0.5:
                problems.append(f"the {name} panel falls below the first screen: {desktop[name]['bottom']:.0f} px of {view['height']}")
        page.screenshot(path=str(out / "page_start.png"), clip=viewport, timeout=180000)

        # 2. draw the figure and let the artist draw it
        note("layout measured; drawing the figure")
        draw_strokes(page, check_figure())
        wait_for_drawing(page)
        first = run_drawing(page, args.events, midway=lambda pg: pg.screenshot(path=str(out / "page_draw.png"), clip=viewport, timeout=180000))
        log_first = page.evaluate("window.__page.log()")
        result_first = page.evaluate("window.__page.drawing.last")
        if result_first is None:
            problems.append(f"the first drawing did not finish within {args.events} decisions")
        report["drawing_1"] = {**first, "page": result_first, "measured": measures(log_first)}

        # 3. the same figure again, with everything the brain learned from the first
        note("drawing 1 done; draw again")
        page.click("#again")
        wait_for_drawing(page)
        second = run_drawing(page, args.events)
        log_second = page.evaluate("window.__page.log()")
        last = page.evaluate("window.__page.drawing.last") or {}
        report["drawing_2"] = {**second, "page": last, "measured": measures(log_second)}
        measured = measures(log_second)
        if not last:
            problems.append(f"the second drawing did not finish within {args.events} decisions")
        else:
            if abs(measured["f1"] - last["f1"]) > 1e-9 or abs(measured["chamfer"] - last["chamfer"]) > 1e-9:
                problems.append(f"the page reads F1 {last['f1']:.6f} and Chamfer {last['chamfer']:.6f} where the canvas says {measured['f1']:.6f} and {measured['chamfer']:.6f}")
            best = max(report["drawing_1"]["measured"]["f1"], measured["f1"])
            report["best_f1"] = best
            if judge and best < bar:
                problems.append(f"the better of the two drawings reaches F1 {best:.3f}, below the bar {bar:.3f}")
        for name in ("drawing_1", "drawing_2"):
            if report[name]["seconds_per_decision"] * 1000 > args.max_decision_ms:
                problems.append(f"{name}: {report[name]['seconds_per_decision'] * 1000:.0f} ms per decision, above the budget {args.max_decision_ms:.0f} ms")

        # 4. the screenshots of the finished drawing
        note("drawing 2 done; screenshots")
        page.screenshot(path=str(out / "page_after.png"), clip=viewport, timeout=180000)
        page.screenshot(path=str(out / "page_full.png"), full_page=True, timeout=180000)
        page.screenshot(path=str(out / "canvas_after.png"), clip=box_of(page, "#canvas"), timeout=180000)
        page.screenshot(path=str(out / "brain_after.png"), clip=box_of(page, "#scan"), timeout=180000)
        records_clip = box_of(page, "#recordsPanel")
        if records_clip is not None:
            page.screenshot(path=str(out / "records_after.png"), clip=records_clip, timeout=180000)

        # 5. the rest of the page
        note("the rest of the controls")
        draw_strokes(page, two_strokes())
        wait_for_drawing(page)
        page.evaluate("window.__page.run(60)")
        multi = page.evaluate("window.__page.drawing")
        report["two_strokes"] = {"state": multi["state"], "strokes": multi["strokes"], "ink": multi["ink"], "target": measures(page.evaluate("window.__page.log()"))["target"]}
        if multi["strokes"] < 1:
            problems.append("the artist left no stroke on the two-stroke figure")
        families = {}
        for family in ("segment", "two_segments", "polygon", "curve", "composition"):
            page.click(f".chip[data-family='{family}']")
            wait_for_drawing(page)
            page.evaluate("window.__page.run(24)")
            state = page.evaluate("window.__page.drawing")
            families[family] = {"state": state["state"], "ink": state["ink"], "running": bool(state["running"])}
            if not state["running"]:
                problems.append(f"the {family} family did not start a drawing")
        report["families"] = families
        page.click("#clear")
        page.evaluate("window.__page.run(3)")
        report["after_clear"] = page.evaluate("window.__page.drawing")["state"]
        if report["after_clear"] != "idle":
            problems.append(f"Clear left the page {report['after_clear']}")
        # the brain selector, when the page was built with a checkpoint manifest beside it
        checkpoints = page.evaluate("window.__page.drawing.checkpoints")
        report["checkpoints"] = checkpoints
        if checkpoints:
            page.select_option("#checkpoint", checkpoints[0])
            page.wait_for_function("window.__page.stats.checkpoint !== 'inlined'", timeout=180000)
            page.click(".chip[data-family='segment']")  # the brain just loaded draws a figure of its own
            wait_for_drawing(page)
            page.evaluate("window.__page.run(24)")
            report["checkpoint_stats"] = {k: v for k, v in page.evaluate("window.__page.stats").items() if k in ("checkpoint", "events", "decisions", "experience")}
            if report["checkpoint_stats"]["checkpoint"] == "inlined":
                problems.append("the brain selector did not load another brain")
            page.select_option("#checkpoint", "inlined")
            page.wait_for_function("window.__page.stats.checkpoint === 'inlined'", timeout=180000)
        # the live loop, the way a visitor leaves it running
        page.click(".chip[data-family='curve']")
        page.evaluate("window.__page.play()")
        time.sleep(3.0)
        page.evaluate("window.__page.pause()")
        report["stats_after"] = {k: v for k, v in page.evaluate("window.__page.stats").items() if k not in ("renderer", "drawing")}
        report["renderer_after"] = page.evaluate("window.__page.scan.snapshot()")
        report["results"] = page.evaluate("window.__page.drawing.results")
        report["phase_label"] = page.inner_text("#phase")
        report["status"] = page.inner_text("#status")
        report["receipt_line"] = page.inner_text("#receipt")

        # 6. phone width: the canvas above the brain, nothing scrolling sideways
        if not args.skip_phone:
            note("phone width")
            page.close()  # one 18 MB page and its regenerated expansion at a time
            phone = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=2)
            phone.set_default_navigation_timeout(300000)
            phone.on("pageerror", lambda e: errors.append(f"phone pageerror: {e}"))
            phone.on("console", lambda m: errors.append(f"phone console.error: {m.text}") if m.type == "error" else None)
            phone.add_init_script("window.__ARTIST_PAGE__ = {autoplay: false};")
            phone.goto(url)
            phone.wait_for_function("window.__page && window.__page.ready", timeout=300000)
            phone.evaluate("window.__page.pause()")
            draw_strokes(phone, check_figure())
            wait_for_drawing(phone)
            phone.evaluate("window.__page.run(40)")
            time.sleep(0.4)
            small = geometry(phone, ["canvas", "scan", "records"])
            report["layout_phone"] = small
            if small["canvas"]["bottom"] > small["scan"]["y"] + 1:
                problems.append("at 390 px the brain scan does not sit below the canvas")
            if small["scrollWidth"] > small["viewport"]["width"] + 1:
                problems.append(f"the page scrolls sideways at 390 px: scrollWidth {small['scrollWidth']}")
            phone.screenshot(path=str(out / "page_phone.png"), full_page=True, timeout=180000)
            phone.close()
        browser.close()
    if server is not None:
        server.shutdown()

    report["errors"] = errors
    report["warnings"] = warnings[:6]
    report["problems"] = problems
    report["screenshots"] = [str(out / name) for name in ("page_start.png", "page_draw.png", "page_after.png", "page_full.png", "canvas_after.png", "brain_after.png", "records_after.png", "page_phone.png") if (out / name).exists()]
    report["passed"] = not errors and not problems
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
