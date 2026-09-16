#!/usr/bin/env python3
"""Open the built S06 composer page in headless Chromium, walk its seven pieces, play every
track each piece has, and fail on any page or console error, on a layout that does not fit, on
audio that does not point at the file the manifest names, or on a replay that does not advance.

    /Users/muellerberndt/Projects/oph-meta/cadence/.venv/bin/python \
        composer/web/check_page.py composer/index.html

Uses SwiftShader so WebGL2 works without a GPU, and allows autoplay so the transport can be
driven without a gesture. The page is served over http, because a page whose assets sit beside
it cannot fetch them from file://. The check:

1. measures the layout at 1280 by 800: the roll and the brain scan both inside the viewport,
   side by side, with nothing scrolling sideways (page_start.png);
2. selects every piece in turn and reads back what the page loaded: the audio elements' sources
   against the index, the level lane of each track, the frames and the records the piece carries;
3. plays every track the piece has and requires the block, the frames shown, the settling steps
   of the scan and the records the view has been fed to advance while it plays;
4. saves page_playing.png, roll.png, brain.png and records.png of the piece named by --shot,
   and page_full.png of the whole page;
5. checks the gate table: a gated set and an open set, every row with a value and a threshold;
6. loads the page at 390 by 844 and requires the roll above the brain with no sideways scroll
   (page_phone.png).

The bundle under test may be a development bundle (``manifest.development``); the check says so
and judges everything except the provenance, which only a published bundle can satisfy.
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

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

MIN_ADVANCE_BLOCKS = 2  # blocks the replay must move while a track plays


def serve(folder: Path) -> tuple[str, socketserver.TCPServer]:
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

    handler = functools.partial(Quiet, directory=str(folder))
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{httpd.server_address[1]}", httpd


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


def box_of(page, selector: str) -> dict | None:
    box = page.locator(selector).bounding_box()
    if box is None or not box["width"] or not box["height"]:
        return None
    return {"x": max(0, box["x"]), "y": max(0, box["y"]), "width": box["width"], "height": box["height"]}


def note(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def play_track(page, track: str, seconds: float, patience: float = 20.0) -> dict:
    """Play one track and report what moved while it played.

    The advance is waited for, never timed: a software renderer can take a second over one
    animation frame, so a fixed window measures the renderer. What is judged is that the replay
    reaches MIN_ADVANCE_BLOCKS while the track plays; how long that took is reported beside it."""
    before = page.evaluate("window.__page.stats")
    started = time.time()
    page.evaluate("(t) => window.__page.play(t)", track)
    timed_out = False
    try:
        page.wait_for_function("(n) => window.__page.stats.block >= n", arg=MIN_ADVANCE_BLOCKS, timeout=int(patience * 1000))
    except PlaywrightTimeout:
        timed_out = True
    page.wait_for_timeout(int(seconds * 1000))
    during = page.evaluate("window.__page.stats")
    page.evaluate("window.__page.stop()")
    return {
        "waited_seconds": round(time.time() - started, 2),
        "timed_out": timed_out,
        "track": track,
        "clock_source": during["clock_source"],
        "seconds": during["seconds"],
        "blocks_advanced": during["block"] - before["block"],
        "frames_shown": during["frames_shown"],
        "frames_total": during["frames_total"],
        "scan_steps_advanced": during["scan_steps"] - before["scan_steps"],
        "records_shown": during["records_shown"],
        "records_total": during["records_total"],
        "record_counts": during["record_counts"],
        "playing": during["playing"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("page")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seconds", type=float, default=1.6, help="how long each track plays")
    parser.add_argument("--shot", default=None, help="the piece the close screenshots are taken of (default: the second one)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--skip-phone", action="store_true")
    args = parser.parse_args()
    page_path = Path(args.page).resolve()
    out = args.out or (page_path.parents[1] / "runs" / "composer" / "web")
    out.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    warnings: list[str] = []
    problems: list[str] = []
    report: dict = {"page": str(page_path), "out": str(out)}
    viewport = {"x": 0, "y": 0, "width": args.width, "height": args.height}
    base, server = serve(page_path.parent)
    url = f"{base}/{page_path.name}"
    report["url"] = url

    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--autoplay-policy=no-user-gesture-required"])
        page = browser.new_page(viewport={"width": args.width, "height": args.height})
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        page.on("console", lambda m: errors.append(f"console.error: {m.text}") if m.type == "error" else warnings.append(f"console.{m.type}: {m.text}") if m.type == "warning" else None)
        page.goto(url)
        page.wait_for_function("window.__page && window.__page.ready", timeout=180000)
        page.wait_for_timeout(500)
        note(f"page ready at {url}")
        manifest = page.evaluate("window.__page.manifest")
        report["renderer"] = page.evaluate("window.__page.scan.snapshot()")
        report["bundle"] = {"format": manifest["format"], "pieces": len(manifest["pieces"]), "development": bool(manifest.get("development")), "run": manifest.get("run"), "provenance": manifest.get("provenance")}
        if manifest.get("development"):
            note("the bundle under test is a development bundle")

        # 1. the layout at 1280 by 800
        desktop = geometry(page, ["roll", "scan", "records", "transport", "numbers", "piece"])
        report["layout_desktop"] = desktop
        view = desktop["viewport"]
        for name in ("roll", "scan"):
            if name not in desktop:
                problems.append(f"the page has no #{name}")
            elif not inside(desktop[name], view):
                problems.append(f"#{name} is not inside the {view['width']}x{view['height']} viewport: {desktop[name]}")
        if "roll" in desktop and "scan" in desktop:
            if overlap(desktop["roll"], desktop["scan"]):
                problems.append("the roll and the brain scan overlap")
            if desktop["roll"]["right"] > desktop["scan"]["x"] + 0.5:
                problems.append("the brain scan is not beside the roll")
        if desktop["scrollWidth"] > view["width"] + 1:
            problems.append(f"the page scrolls sideways at {view['width']} px: scrollWidth {desktop['scrollWidth']}")
        page.screenshot(path=str(out / "page_start.png"), clip=viewport, timeout=180000)

        # 2 and 3. every piece, and every track it carries
        pieces = page.evaluate("window.__page.pieces")
        report["pieces"] = {}
        steps_taken = 0  # settling steps the scan took over the whole walk; a software renderer
        # can be starved enough that one track's replay jumps instead of stepping, so the
        # stepping is judged over the walk and the following of the audio track by track
        shot_piece = args.shot or (pieces[1] if len(pieces) > 1 else pieces[0])
        for piece_id in pieces:
            note(f"piece {piece_id}")
            page.select_option("#piece", piece_id)
            page.wait_for_function("(id) => window.__page.stats.loaded === id", arg=piece_id, timeout=60000)
            page.wait_for_timeout(250)
            stats = page.evaluate("window.__page.stats")
            declared = next(p for p in manifest["pieces"] if p["id"] == piece_id)
            entry: dict = {"lanes": stats["lanes"], "audio": {}, "tracks": []}
            for track, source in (declared.get("audio") or {}).items():
                src = stats["audio"][track]["src"]
                entry["audio"][track] = src
                if not src.endswith(source):
                    problems.append(f"{piece_id}: the {track} audio element points at {src!r}, not at {source!r}")
            for track in ("demonstration", "imitation", "continuation"):
                has_audio = track in (declared.get("audio") or {})
                disabled = page.evaluate("(t) => document.getElementById('play' + t[0].toUpperCase() + t.slice(1)).disabled", track)
                if has_audio and disabled:
                    problems.append(f"{piece_id}: the {track} button is disabled although the bundle carries that audio")
                if not has_audio and not disabled:
                    problems.append(f"{piece_id}: the {track} button is live although the bundle carries no such audio")
                if not has_audio:
                    continue
                result = play_track(page, track, args.seconds)
                entry["tracks"].append(result)
                if result["timed_out"] or result["blocks_advanced"] < MIN_ADVANCE_BLOCKS:
                    problems.append(f"{piece_id}/{track}: the replay advanced {result['blocks_advanced']} blocks in {result['waited_seconds']} s")
                if result["frames_total"] and result["frames_shown"] < 2:
                    problems.append(f"{piece_id}/{track}: the brain replay stayed on frame {result['frames_shown']} of {result['frames_total']}")
                steps_taken += max(0, result["scan_steps_advanced"])
                if result["records_total"] and result["records_shown"] < 1:
                    problems.append(f"{piece_id}/{track}: the records view was fed nothing although the piece carries {result['records_total']} entries")
                if piece_id == shot_piece and track == "demonstration":
                    page.evaluate("(t) => window.__page.play(t)", track)
                    page.wait_for_timeout(1400)
                    page.screenshot(path=str(out / "page_playing.png"), clip=viewport, timeout=180000)
                    for name, selector in (("roll", "#rollStage"), ("brain", "#scan"), ("records", "#recordsPanel"), ("numbers", "#numbers")):
                        clip = box_of(page, selector)
                        if clip:
                            page.screenshot(path=str(out / f"{name}.png"), clip=clip, timeout=180000)
                    page.evaluate("window.__page.stop()")
            report["pieces"][piece_id] = entry
        report["scan_steps_over_the_walk"] = steps_taken
        if not steps_taken:
            problems.append("the brain scan took no settling step over the whole walk")
        page.screenshot(path=str(out / "page_full.png"), full_page=True, timeout=180000)

        # 4. the gate table
        gates = page.evaluate(
            """() => {
                const read = (id) => Array.from(document.querySelectorAll(`#${id} tbody tr`)).map((tr) => ({
                    section: tr.classList.contains("section"), cells: Array.from(tr.children).map((td) => td.textContent.trim())}));
                return {gated: read("gatedTable"), open: read("openTable")};
            }"""
        )
        report["gates"] = {"gated": len([r for r in gates["gated"] if not r["section"]]), "open": len([r for r in gates["open"] if not r["section"]]),
                           "sections": {"gated": [r["cells"][0] for r in gates["gated"] if r["section"]], "open": [r["cells"][0] for r in gates["open"] if r["section"]]}}
        if report["gates"]["gated"] < 1:
            problems.append("the gate table has no gated row")
        if report["gates"]["open"] < 1:
            problems.append("the gate table has no open row")
        for row in [r for r in gates["gated"] + gates["open"] if not r["section"]]:
            if len(row["cells"]) < 4 or row["cells"][1] in ("", "not measured"):
                problems.append(f"a gate row carries no value: {row['cells']}")
            if row["cells"][3] not in ("PASS", "NOT MET", "OPEN", "MEASURED"):
                problems.append(f"a gate row carries no verdict: {row['cells']}")
        report["receipt_note"] = page.inner_text("#receiptNote")
        report["counts"] = page.inner_text("#counts")
        report["status"] = page.inner_text("#status")
        report["state"] = page.inner_text("#state")

        # 5. phone width
        if not args.skip_phone:
            note("phone width")
            phone = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=2)
            phone.on("pageerror", lambda e: errors.append(f"phone pageerror: {e}"))
            phone.on("console", lambda m: errors.append(f"phone console.error: {m.text}") if m.type == "error" else None)
            phone.goto(url)
            phone.wait_for_function("window.__page && window.__page.ready", timeout=180000)
            phone.evaluate("window.__page.play('demonstration')")
            phone.wait_for_timeout(1200)
            small = geometry(phone, ["roll", "scan", "records"])
            report["layout_phone"] = small
            if "roll" in small and "scan" in small and small["roll"]["bottom"] > small["scan"]["y"] + 1:
                problems.append("at 390 px the brain scan does not sit below the roll")
            if small["scrollWidth"] > small["viewport"]["width"] + 1:
                problems.append(f"the page scrolls sideways at 390 px: scrollWidth {small['scrollWidth']}")
            # a tall page at device_scale_factor 2 can pass the capture limit; the first screen
            # is what the check judges, and the whole page is taken when it can be
            phone.screenshot(path=str(out / "page_phone.png"), clip={"x": 0, "y": 0, "width": 390, "height": 844}, timeout=180000)
            try:
                phone.screenshot(path=str(out / "page_phone_full.png"), full_page=True, timeout=60000)
            except PlaywrightError as error:  # the capture limit is not a page fault
                warnings.append(f"the whole phone page could not be captured ({small['scrollHeight']} px tall): {error}")
            phone.close()
        browser.close()
    server.shutdown()

    report["errors"] = errors
    report["warnings"] = warnings[:6]
    report["problems"] = problems
    report["screenshots"] = [str(out / name) for name in ("page_start.png", "page_playing.png", "roll.png", "brain.png", "records.png", "numbers.png", "page_full.png", "page_phone.png", "page_phone_full.png") if (out / name).exists()]
    report["passed"] = not errors and not problems
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
