#!/usr/bin/env python3
"""Open the built Connect Four page in headless Chromium, play a game against the brain at every
checkpoint with random legal moves, and fail on any page or console error, on a layout that does
not fit, on a brain view that does not render, or on a move the brain takes longer than the
bound to answer.

    /Users/muellerberndt/Projects/oph-meta/cadence/.venv/bin/python \
        connect_four/web/check_page.py runs/connect_four/page/index.html

Uses SwiftShader so WebGL2 works without a GPU. The check:

1. measures the layout at 1280 by 800: the board and the brain scan must both lie inside the
   viewport, side by side, with nothing scrolling sideways (page_start.png);
2. plays one game with the mouse: a random legal column each move, clicked on the board
   (page_play.png halfway through, page_after.png when the game is over), and holds every
   answer of the brain under --max-ms, measuring the exchange (the brain reads your move,
   searches, moves and reads its own move) and the search alone;
3. checks the brain view: the scan drew its steps, both record cortices lit cells and took
   writes, and the search line names the columns the brain imagined;
4. plays one game at every other checkpoint from the selector, with the same bound;
5. plays a game with learning frozen and checks that nothing is written;
6. saves board_after.png, brain_after.png, records_after.png;
7. loads the page at 390 by 844 and requires the board above the brain with no sideways
   scroll (page_phone.png).
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import random
import socketserver
import sys
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

MAX_MS = 100.0  # the bound on one exchange: the brain reads the move, searches, answers and reads its own move


def serve(folder: Path) -> tuple[str, socketserver.TCPServer]:
    """The page fetches its other checkpoints, which a file:// page cannot do."""
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


def note(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def click_column(page, column: int) -> None:
    point = page.evaluate("(c) => window.__page.toClient(c)", column)
    page.mouse.move(point[0], point[1])
    page.mouse.down()
    page.mouse.up()


def settle(page, timeout: int = 20000) -> None:
    """Wait until the brain view has played back everything the last moves queued."""
    try:
        page.wait_for_function("window.__page.stats.queued === 0", timeout=timeout)
    except Exception:  # noqa: BLE001  a slow machine: drain the rest at once and go on
        page.evaluate("window.__page.drainAll()")


def play_game(page, rng: random.Random, label: str, midway=None, use_mouse: bool = True, max_moves: int = 60) -> dict:
    """One game of random legal moves; returns the moves, the latencies and the outcome."""
    page.evaluate("window.__page.newGame()")
    moves, latencies, searches = 0, [], []
    shot = midway is None
    while moves < max_moves:
        board = page.evaluate("window.__page.board")
        if board["over"] or not board["legal"]:
            break
        column = rng.choice(board["legal"])
        if use_mouse:
            click_column(page, column)
        else:
            page.evaluate("(c) => window.__page.drop(c)", column)
        stats = page.evaluate("window.__page.stats")
        moves += 1
        if stats["move_ms"]:
            latencies.append(stats["move_ms"])
            searches.append(stats["decide_ms"])
        if not shot and moves >= 3:
            midway(page)
            shot = True
    settle(page)
    final = page.evaluate("window.__page.board")
    stats = page.evaluate("window.__page.stats")
    return {"label": label, "moves": moves, "board_moves": final["moves"], "winner": final["winner"], "over": final["over"],
            "exchange_ms": {"max": max(latencies, default=0.0), "mean": sum(latencies) / max(1, len(latencies))},
            "search_ms": {"max": max(searches, default=0.0), "mean": sum(searches) / max(1, len(searches))},
            "imagined": page.evaluate("window.__page.imagined"), "stats": {k: stats[k] for k in ("decisions", "transitions", "records", "imagined", "extended", "learning")}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("page")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-ms", type=float, default=MAX_MS, help="the bound on one exchange, in milliseconds")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip-phone", action="store_true")
    args = parser.parse_args()
    page_path = Path(args.page).resolve()
    out = args.out or page_path.parent
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    errors: list[str] = []
    warnings: list[str] = []
    problems: list[str] = []
    report: dict = {"page": str(page_path), "bound_ms": args.max_ms}
    viewport = {"x": 0, "y": 0, "width": args.width, "height": args.height}
    base, server = serve(page_path.parent)
    url = f"{base}/{page_path.name}"
    report["url"] = url

    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": args.width, "height": args.height})
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        page.on("console", lambda m: errors.append(f"console.error: {m.text}") if m.type == "error" else warnings.append(f"console.{m.type}: {m.text}") if m.type == "warning" else None)
        page.goto(url)
        page.wait_for_function("window.__page && window.__page.ready", timeout=300000)
        time.sleep(0.4)
        note(f"page ready at {url}")
        report["renderer"] = page.evaluate("window.__page.scan.snapshot()")
        report["stats_start"] = {k: v for k, v in page.evaluate("window.__page.stats").items() if k != "renderer"}
        report["checkpoints"] = page.evaluate("window.__page.checkpoints")

        # 1. the layout: the board and the brain side by side, both on the first screen
        desktop = geometry(page, ["board", "scan", "strip", "card"])
        report["layout_desktop"] = desktop
        view = desktop["viewport"]
        for name in ("board", "scan"):
            if not inside(desktop[name], view):
                problems.append(f"the {name} is not inside the {view['width']}x{view['height']} viewport: {desktop[name]}")
        if overlap(desktop["board"], desktop["scan"]):
            problems.append("the board and the brain scan overlap")
        if desktop["board"]["right"] > desktop["scan"]["x"] + 0.5:
            problems.append("the brain scan is not beside the board")
        if desktop["scrollWidth"] > view["width"] + 1:
            problems.append(f"the page scrolls sideways at {view['width']} px: scrollWidth {desktop['scrollWidth']}")
        page.screenshot(path=str(out / "page_start.png"), clip=viewport, timeout=180000)

        # 2. one game against the brain of this page, played with the mouse
        note("playing a game with random legal moves")
        first = play_game(page, rng, page.evaluate("window.__page.stats.label"), midway=lambda pg: pg.screenshot(path=str(out / "page_play.png"), clip=viewport, timeout=180000))
        report["game"] = first
        if not first["over"]:
            problems.append("the game did not end within the move cap")
        if first["exchange_ms"]["max"] > args.max_ms:
            problems.append(f"an exchange took {first['exchange_ms']['max']:.1f} ms, over the {args.max_ms:.0f} ms bound")
        page.screenshot(path=str(out / "page_after.png"), clip=viewport, timeout=180000)

        # 3. the brain view drew, the cortices lit and were written, the search line is there
        after = page.evaluate("window.__page.scan.snapshot()")
        report["renderer_after"] = after
        if after["steps"] <= report["renderer"]["steps"]:
            problems.append(f"the brain scan drew no settling steps ({after['steps']} vs {report['renderer']['steps']})")
        if after["neurons"] < 1 or after["regions"] < 1:
            problems.append(f"the brain scan holds no neurons or regions: {after}")
        imagined = first["imagined"]
        if not imagined or not imagined["columns"] or len(imagined["read"]) != len(imagined["columns"]):
            problems.append(f"the page does not carry what the brain imagined: {imagined}")
        report["state"] = page.inner_text("#state")
        report["card_facts"] = page.text_content("#cardFacts")
        if "neurons" not in report["card_facts"]:
            problems.append("the model card states no neuron count")
        for name, selector in (("board_after.png", "#boardStage"), ("brain_after.png", "#brainRow")):
            page.locator(selector).screenshot(path=str(out / name), timeout=180000)

        # 4. every other checkpoint: a game each, the same bound
        games = [first]
        for ident in report["checkpoints"]:
            if ident == page.evaluate("window.__page.stats.checkpoint"):
                continue
            note(f"checkpoint {ident}")
            page.select_option("#checkpoint", ident)
            page.wait_for_function(f"window.__page.stats.checkpoint === '{ident}'", timeout=180000)
            games.append(play_game(page, rng, ident, use_mouse=False))
            if games[-1]["exchange_ms"]["max"] > args.max_ms:
                problems.append(f"{ident}: an exchange took {games[-1]['exchange_ms']['max']:.1f} ms, over the {args.max_ms:.0f} ms bound")
            if not games[-1]["over"]:
                problems.append(f"{ident}: the game did not end within the move cap")
        report["games"] = games
        report["latency_ms"] = {g["label"]: {"exchange_max": g["exchange_ms"]["max"], "exchange_mean": g["exchange_ms"]["mean"], "search_max": g["search_ms"]["max"], "search_mean": g["search_ms"]["mean"]} for g in games}

        # 5. the frozen brain writes nothing
        note("learning frozen")
        page.evaluate("window.__page.setLearning(false)")
        before = page.evaluate("window.__page.stats.records")
        frozen = play_game(page, rng, "frozen", use_mouse=False)
        writes = page.evaluate("window.__page.stats.records")
        report["frozen"] = {"before": before, "after": writes, "moves": frozen["moves"]}
        if writes != before:
            problems.append(f"the frozen brain wrote records: {before} then {writes}")
        page.evaluate("window.__page.setLearning(true)")

        report["stats_after"] = {k: v for k, v in page.evaluate("window.__page.stats").items() if k != "renderer"}
        report["phase_label"] = page.inner_text("#phase")
        page.screenshot(path=str(out / "page_full.png"), full_page=True, timeout=180000)

        # 6. phone width: the board above the brain, nothing scrolling sideways
        if not args.skip_phone:
            note("phone width")
            phone = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=2)
            phone.on("pageerror", lambda e: errors.append(f"phone pageerror: {e}"))
            phone.on("console", lambda m: errors.append(f"phone console.error: {m.text}") if m.type == "error" else None)
            phone.goto(url)
            phone.wait_for_function("window.__page && window.__page.ready", timeout=300000)
            for column in (3, 2, 4):
                phone.evaluate("(c) => window.__page.drop(c)", column)
            time.sleep(0.5)
            small = geometry(phone, ["board", "scan"])
            report["layout_phone"] = small
            if small["board"]["bottom"] > small["scan"]["y"] + 1:
                problems.append("at 390 px the brain scan does not sit below the board")
            if small["scrollWidth"] > small["viewport"]["width"] + 1:
                problems.append(f"the page scrolls sideways at 390 px: scrollWidth {small['scrollWidth']}")
            phone.screenshot(path=str(out / "page_phone.png"), full_page=True, timeout=180000)
            phone.close()
        browser.close()
    server.shutdown()

    report["errors"] = errors
    report["warnings"] = warnings[:6]
    report["problems"] = problems
    report["screenshots"] = [str(out / name) for name in ("page_start.png", "page_play.png", "page_after.png", "page_full.png", "board_after.png", "brain_after.png", "records_after.png", "page_phone.png") if (out / name).exists()]
    report["passed"] = not errors and not problems
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
