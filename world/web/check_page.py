#!/usr/bin/env python3
"""Open the built S02 page in headless Chromium, ask the agent for an object, move one behind
its back, teach a word, run the cue task, lock the doors, switch brains, and fail on any page
or console error, on a layout that does not fit, or on a decision that takes longer than the
bound.

    /Users/muellerberndt/Projects/oph-meta/cadence-artist/.venv/bin/python \
        world/web/check_page.py runs/world/web/index.html

Uses SwiftShader so WebGL2 works without a GPU. The page is loaded with its demo off
(``window.__WORLD_PAGE__ = {autoplay: false}``) and paused, so every event is computed through
``window.__page.run()`` and the measurement repeats. The check:

1. measures the layout at 1280 by 800: the world canvas and the brain scan must both lie inside
   the viewport, side by side, with nothing scrolling sideways;
2. wanders, so the agent sees objects and its place store fills, and measures the time an event
   takes (page_start.png, page_wander.png);
3. asks for an object whose place the agent remembers, runs the request to its end and reads the
   result off the page (page_ask.png);
4. drags that object to another cell with the mouse, checks that the world moved and the record
   did not, and asks again;
5. teaches a word and asks by it, runs the cue task at delay 16, locks and unlocks the doors,
   and runs one of run.py's own phases;
6. switches the brain selector to a checkpoint when a manifest sits beside the page, decides
   with it, and switches back;
7. saves page_after.png, page_full.png, world_after.png, brain_after.png and records_after.png;
8. loads the page at 390 by 844 and requires the world above the brain with no sideways scroll
   (page_phone.png).

The decision bound is the mean compute time of one event, everything the page does for it
included: the settling phases, the search through the imagined consequences, the record write
and the frames the scan plays back. A software renderer on a laptop measures 16 ms per event
on the 2026-09-15 brain (16,000 cells, 80 active, at most 128 expansions), 26 ms in the first
batch, before the browser has warmed up.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import socketserver
import sys
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

MAX_DECISION_MS = 100.0  # the mean time one event may take, the whole page's work included


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


def serve(folder: Path) -> tuple[str, socketserver.TCPServer]:
    """A page built with checkpoints has to be served, because a file:// page cannot fetch its neighbours."""
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args) -> None:  # the check's own output stays readable
            pass

    handler = functools.partial(Quiet, directory=str(folder))
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{httpd.server_address[1]}", httpd


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


def run(page, count: int) -> dict:
    """`count` events, computed and shown; returns how long each one took."""
    out = page.evaluate(f"window.__page.run({count})")
    return {"events": out["ran"], "ms_per_event": 1000 * out["seconds"] / max(1, out["ran"])}


def run_task(page, limit: int = 400) -> dict:
    """Events until the running task records a result."""
    out = page.evaluate(f"window.__page.runTask({limit})")
    note(f"  task: {'finished' if out['finished'] else 'unfinished'} after {out['ran']} events ({1000 * out['seconds'] / max(1, out['ran']):.0f} ms each) · {out['last']}")
    return out


def known_place(page) -> dict | None:
    """An object the agent has seen and remembers, with the cell it remembers."""
    places = page.evaluate("window.__page.task.places")
    for row in places:
        if row["where"] and row["known"] >= 0.5:
            return row
    return None


def free_cell(page, keep: list[int]) -> list[int] | None:
    """A cell with no object in it and not the agent's own, for the drag."""
    task = page.evaluate("window.__page.task")
    taken = {tuple(c) for c in task["objects"].values() if c} | {tuple(task["agent"])}
    for y in range(6):
        for x in range(6):
            if (x, y) not in taken and [x, y] != keep:
                return [x, y]
    return None


def drag(page, source: list[int], target: list[int]) -> None:
    a = page.evaluate("(c) => window.__page.toClient(c)", source)
    b = page.evaluate("(c) => window.__page.toClient(c)", target)
    page.mouse.move(a[0], a[1])
    page.mouse.down()
    page.mouse.move((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    page.mouse.move(b[0], b[1])
    page.mouse.up()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("page")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--wander", type=int, default=40, help="wandering events before the first request")
    parser.add_argument("--events", type=int, default=400, help="the most events one task may take")
    parser.add_argument("--max-decision-ms", type=float, default=MAX_DECISION_MS)
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
    report: dict = {"page": str(page_path), "decision_bound_ms": args.max_decision_ms}
    viewport = {"x": 0, "y": 0, "width": args.width, "height": args.height}
    manifest = page_path.parent / "checkpoints.json"
    server = None
    if manifest.exists():
        base, server = serve(page_path.parent)
        url = f"{base}/{page_path.name}"
    else:
        url = page_path.as_uri()
    report["url"] = url
    timings: list[float] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": args.width, "height": args.height})
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        page.on("console", lambda m: errors.append(f"console.error: {m.text}") if m.type == "error" else warnings.append(f"console.{m.type}: {m.text}") if m.type == "warning" else None)
        page.add_init_script("window.__WORLD_PAGE__ = {autoplay: false};")
        page.goto(url)
        page.wait_for_function("window.__page && window.__page.ready", timeout=300000)
        page.evaluate("window.__page.pause()")
        time.sleep(0.4)
        note(f"page ready at {url}")
        report["renderer"] = page.evaluate("window.__page.scan.snapshot()")
        report["stats_start"] = {k: v for k, v in page.evaluate("window.__page.stats").items() if k != "renderer"}
        report["experience"] = report["stats_start"].get("experience", 0)

        # 1. the layout: both canvases inside the first screen, side by side
        desktop = geometry(page, ["world", "scan", "records", "stores", "worldKey", "readouts", "body", "brain"])
        report["layout_desktop"] = desktop
        view = desktop["viewport"]
        if not inside(desktop["world"], view):
            problems.append(f"the world canvas is not inside the {view['width']}x{view['height']} viewport: {desktop['world']}")
        if not inside(desktop["scan"], view):
            problems.append(f"the brain scan is not inside the {view['width']}x{view['height']} viewport: {desktop['scan']}")
        if desktop.get("records") and not inside(desktop["records"], view):
            problems.append(f"the records cortex is not inside the viewport: {desktop['records']}")
        if overlap(desktop["world"], desktop["scan"]):
            problems.append("the world canvas and the brain scan overlap")
        if desktop["world"]["right"] > desktop["scan"]["x"] + 0.5:
            problems.append("the brain scan is not beside the world canvas")
        for name, panel in (("world", desktop["body"]), ("brain", desktop["brain"])):
            if panel["bottom"] > view["height"] + 0.5:
                problems.append(f"the {name} panel ends {panel['bottom'] - view['height']:.0f} px below the fold: its controls need scrolling")
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

        # 2. wandering: the agent sees objects and writes down where they were
        note("the wander the page opens with")
        opening = page.evaluate("window.__page.wander()")
        report["opening_wander"] = opening
        if opening["remembered"] < 4:
            problems.append(f"the wander the page opens with left {opening['remembered']} of 4 objects remembered")
        page.evaluate("window.__page.pause()")
        note("wandering")
        wander = {"events": 0, "ms_per_event": 0.0}
        for _ in range(5):  # random wandering on a 6 by 6 world takes a while to walk onto an object
            batch = run(page, args.wander)
            wander = {"events": wander["events"] + batch["events"], "ms_per_event": batch["ms_per_event"]}
            timings.append(batch["ms_per_event"])
            if known_place(page) is not None:
                break
        report["wander"] = {**wander, "places": page.evaluate("window.__page.task.places")}
        page.screenshot(path=str(out / "page_wander.png"), clip=viewport, timeout=180000)
        remembered = known_place(page)
        report["remembered"] = remembered
        if remembered is None:
            problems.append(f"after {wander['events']} wandering events the agent remembers no object at all")

        # 3. ask for an object it remembers
        if remembered is not None:
            k = remembered["object"]
            note(f"asking for object {k}, remembered at {remembered['where']}")
            page.click(f".chip[data-object='{k}']")
            first = run_task(page, args.events)
            timings.append(1000 * first["seconds"] / max(1, first["ran"]))
            report["ask"] = {"object": k, "remembered": remembered["where"], **{key: first[key] for key in ("ran", "finished", "last")}}
            if not first["finished"]:
                problems.append(f"the request for object {k} did not end within {args.events} events")
            page.screenshot(path=str(out / "page_ask.png"), clip=viewport, timeout=180000)

            # 4. move it behind the agent's back and ask again
            task = page.evaluate("window.__page.task")
            where = task["objects"].get(str(k)) or task["objects"].get(k)
            target = free_cell(page, where)
            note(f"dragging object {k} from {where} to {target}")
            if where and target:
                drag(page, where, target)
                moved = page.evaluate("window.__page.task")
                now = moved["objects"].get(str(k)) or moved["objects"].get(k)
                record = next((r for r in moved["places"] if r["object"] == k), None)
                report["move"] = {"object": k, "from": where, "to": target, "now": now, "record": record}
                if now != target:
                    problems.append(f"the drag did not move object {k}: the world says {now}, the drop was {target}")
                if record and record["where"] == target and where != target:
                    problems.append("the place record followed the object without the agent seeing it")
                page.click(f".chip[data-object='{k}']")
                second = run_task(page, args.events)
                timings.append(1000 * second["seconds"] / max(1, second["ran"]))
                report["ask_after_move"] = {key: second[key] for key in ("ran", "finished", "last")}
                if not second["finished"]:
                    problems.append("the request after the move did not end")
            else:
                problems.append("no free cell to drag an object into")

        # 5. the other tasks: a word taught and asked by, the cue task, the doors, run.py's phases
        note("teaching a word, then asking by it")
        page.click("#word")
        word = run_task(page, args.events)
        timings.append(1000 * word["seconds"] / max(1, word["ran"]))
        report["word"] = {key: word[key] for key in ("ran", "finished", "last")}
        if not word["finished"]:
            problems.append("the word task did not end")
        note("the cue task at delay 16")
        page.select_option("#delay", "16")
        page.click("#cue")
        cue = run_task(page, args.events)
        timings.append(1000 * cue["seconds"] / max(1, cue["ran"]))
        report["cue"] = {key: cue[key] for key in ("ran", "finished", "last")}
        if not cue["finished"]:
            problems.append("the cue task did not end")
        page.screenshot(path=str(out / "page_after.png"), clip=viewport, timeout=180000)
        note("locking the doors")
        page.click("#doors")
        run(page, 4)
        locked = page.evaluate("window.__page.task.doors")
        page.click("#doors")
        run(page, 2)
        unlocked = page.evaluate("window.__page.task.doors")
        report["doors"] = {"after_locking": locked, "after_unlocking": unlocked, "label": page.inner_text("#doors")}
        if locked != "locked" or unlocked != "toggle":
            problems.append(f"the door control reads {locked} then {unlocked}")
        note("run.py's own phase")
        page.select_option("#phaseSelect", "remember-8")
        run(page, 12)
        report["phase"] = {"task": page.evaluate("window.__page.task.kind"), "state": page.inner_text("#taskState")}
        if report["phase"]["task"] != "remember-8":
            problems.append(f"the run.py phase select left the page in {report['phase']['task']}")
        page.select_option("#phaseSelect", "free")
        run(page, 2)

        # 6. the brain controls and the brain selector
        note("the brain controls")
        page.select_option("#view", "change")
        page.click("#fit")
        page.select_option("#imagine", "1")
        page.select_option("#speed", "0")
        page.click("#step")
        page.evaluate("window.__page.play()")
        time.sleep(1.5)
        page.evaluate("window.__page.pause()")
        page.select_option("#view", "activity")
        page.select_option("#imagine", "4")
        page.select_option("#speed", "64")
        checkpoints = page.evaluate("window.__page.task.checkpoints")
        report["checkpoints"] = checkpoints
        if checkpoints:
            note(f"the brain selector: {checkpoints[0]}")
            page.select_option("#checkpoint", checkpoints[0])
            page.wait_for_function("window.__page.stats.checkpoint !== 'inlined'", timeout=180000)
            other = run(page, 12)
            timings.append(other["ms_per_event"])
            report["checkpoint"] = {"id": page.evaluate("window.__page.stats.checkpoint"), "experience": page.evaluate("window.__page.stats.experience"), **other}
            page.select_option("#checkpoint", "inlined")
            page.wait_for_function("window.__page.stats.checkpoint === 'inlined'", timeout=180000)
            run(page, 4)

        # 7. the screenshots of the finished page
        note("screenshots")
        page.screenshot(path=str(out / "page_full.png"), full_page=True, timeout=180000)
        page.screenshot(path=str(out / "world_after.png"), clip=box_of(page, "#worldStage"), timeout=180000)
        page.screenshot(path=str(out / "brain_after.png"), clip=box_of(page, "#scan"), timeout=180000)
        records_clip = box_of(page, "#recordsPanel")
        if records_clip is not None:
            page.screenshot(path=str(out / "records_after.png"), clip=records_clip, timeout=180000)
        report["stats_after"] = {k: v for k, v in page.evaluate("window.__page.stats").items() if k != "renderer"}
        report["renderer_after"] = page.evaluate("window.__page.scan.snapshot()")
        report["task_after"] = page.evaluate("window.__page.task")
        report["state"] = page.inner_text("#taskState")
        report["phase_label"] = page.inner_text("#phase")
        report["status"] = page.inner_text("#status")
        report["facts"] = page.inner_text("#counts")
        report["receipt_note"] = page.inner_text("#receiptNote")

        # 8. phone width: the world above the brain, nothing scrolling sideways
        if not args.skip_phone:
            note("phone width")
            phone = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=2)
            phone.on("pageerror", lambda e: errors.append(f"phone pageerror: {e}"))
            phone.on("console", lambda m: errors.append(f"phone console.error: {m.text}") if m.type == "error" else None)
            phone.add_init_script("window.__WORLD_PAGE__ = {autoplay: false};")
            phone.goto(url)
            phone.wait_for_function("window.__page && window.__page.ready", timeout=300000)
            phone.evaluate("window.__page.pause()")
            phone.evaluate("window.__page.run(20)")
            phone.click(".chip[data-object='0']")
            phone.evaluate("window.__page.run(6)")
            time.sleep(0.4)
            small = geometry(phone, ["world", "scan", "records"])
            report["layout_phone"] = small
            if small["world"]["bottom"] > small["scan"]["y"] + 1:
                problems.append("at 390 px the brain scan does not sit below the world canvas")
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

    mean = sum(timings) / max(1, len(timings))
    report["ms_per_event"] = {"mean": mean, "max": max(timings) if timings else None, "batches": [round(t, 1) for t in timings]}
    if mean > args.max_decision_ms:
        problems.append(f"an event takes {mean:.0f} ms on average, above the bound {args.max_decision_ms:.0f} ms")
    report["errors"] = errors
    report["warnings"] = warnings[:6]
    report["problems"] = problems
    report["screenshots"] = [str(out / name) for name in ("page_start.png", "page_wander.png", "page_ask.png", "page_after.png", "page_full.png", "world_after.png", "brain_after.png", "records_after.png", "page_phone.png") if (out / name).exists()]
    report["passed"] = not errors and not problems
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
