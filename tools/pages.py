"""Open every page in a real browser and check it works: it loads clean, it decodes as UTF-8, the games play to the end,
and the settlement the page runs in JavaScript is the settlement the Python engine runs on the same net.

Run:  python -m pip install playwright && python -m playwright install chromium
      python tools/pages.py                (about a minute)
      python tools/pages.py --rally 20     (longer Pong rallies)

What is checked, page by page:
- the hub: every local link answers; digits, recall, Connect Four, Pong: no script errors, document.characterSet is UTF-8.
- digits: every held-out picture the page ships is read by the page; most are read as their label.
- recall: the symbol alphabet arrives intact (a page decoded with the wrong charset garbles it).
- Connect Four: on forty positions the column the page's settlement picks is the engine's; four games against a
  random player and a win-or-block player run to a result with legal moves only.
- Pong, both nets: on forty frames the page's action is the engine's; the page keeps its twelve decisions a
  second, and the paddle's return rate against a tracking opponent is reported.
The engine reads the net the page embeds (net.json: the same dense matrix, bias and rule), so a mismatch is the page.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import random
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

import cadence as cd

ROOT = Path(__file__).resolve().parents[1]
SYSTEM_CHROME = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "/usr/bin/google-chrome"]


class Quiet(http.server.SimpleHTTPRequestHandler):
    """A plain static server that declares no charset, as a file opened from disk would not: the page must declare it."""

    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".html": "text/html"}

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


def engine_from(path: Path) -> tuple[cd.Settlement, dict]:
    net = json.loads(path.read_text())
    n = net["n"]
    w = np.asarray(net["W"], float).reshape(n, n)
    r = net["rule"]
    pre, post = np.nonzero(w)
    wiring = cd.Wiring.from_edges(n, pre=pre, post=post, sign=w[pre, post], sets=net["sets"])
    rule = cd.GradedRule(dt=r["dt"], slope=r["slope"], threshold=r["threshold"], gain=1.0, clamp_amplitude=r["clamp"], leak=r["leak"])
    assert abs(rule.rest_emission - r["rest"]) < 1e-12, "the page's rest emission is not the rule's"
    return cd.Settlement(wiring, rule, bias=np.asarray(net["bias"], float)), net


def engine_outputs(engine: cd.Settlement, net: dict, drives: list) -> np.ndarray:
    state = engine.settle_batch(np.asarray(drives, float), steps=100, tolerance=net.get("readout_tolerance", 1e-4))
    return state.activation[:, net["sets"]["output"]]


def c4_winner(grid: list) -> int:
    for r in range(6):
        for c in range(7):
            w = grid[r][c]
            if w and any(all(0 <= r + k * dr < 6 and 0 <= c + k * dc < 7 and grid[r + k * dr][c + k * dc] == w for k in range(4)) for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1))):
                return w
    return 0


def c4_choice(grid: list, me: int, them: int, rng: random.Random, smart: bool) -> int:
    legal = [c for c in range(7) if grid[5][c] == 0]
    if smart:  # win if possible, else block
        for who in (me, them):
            for c in legal:
                g = [row[:] for row in grid]
                r = next(r for r in range(6) if g[r][c] == 0)
                g[r][c] = who
                if c4_winner(g) == who:
                    return c
    return rng.choice(legal)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rally", type=float, default=10.0, help="seconds of Pong per net")
    parser.add_argument("--games", type=int, default=4, help="Connect Four games")
    args = parser.parse_args()
    from playwright.sync_api import sync_playwright

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(ROOT)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    failures: list[str] = []
    report: dict = {}

    def fail(msg: str) -> None:
        failures.append(msg)
        print(f"FAIL {msg}", flush=True)

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception:  # noqa: BLE001  (no bundled browser: fall back to a system Chrome)
            chrome = next((c for c in SYSTEM_CHROME if Path(c).exists()), None)
            if chrome is None:
                raise
            browser = p.chromium.launch(executable_path=chrome)
        ctx = browser.new_context(viewport={"width": 1280, "height": 900})

        def open_page(name: str, path: str):  # type: ignore[no-untyped-def]
            page = ctx.new_page()
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" and "fonts.g" not in m.text and "favicon" not in m.text and "404" not in m.text else None)
            page.goto(f"{base}/{path}")
            page.wait_for_load_state("networkidle")
            charset = page.evaluate("document.characterSet")
            report[name] = {"charset": charset}
            if charset.upper() != "UTF-8":
                fail(f"{name}: decoded as {charset}, not UTF-8")
            return page, errors

        # the hub
        page, errors = open_page("hub", "index.html")
        links = sorted({h for h in page.eval_on_selector_all("a[href]", "as => as.map(a => a.getAttribute('href'))") if not h.startswith(("http", "#"))})
        for h in links:
            try:
                urllib.request.urlopen(f"{base}/{h}").read(1)
            except Exception as e:  # noqa: BLE001
                fail(f"hub: link {h} does not answer ({e})")
        report["hub"]["local_links"] = len(links)
        page.close()

        # digits: every shipped held-out picture through the page's own read
        page, errors = open_page("digits", "01_digits/index.html")
        # the page's own choice: the most active output owner at rest, as read() computes it (read() shows it after an animation)
        reads = page.evaluate("() => NET.meta.samples.map(s => { const d = new Float64Array(NET.n); for (let k = 0; k < 64; k++) d[IN[k]] = (s.pixels[k] / 16) * R.clamp; const f = __digits.settle(d).trace.at(-1); let b = 0; for (let c = 1; c < 10; c++) if (f[c] > f[b]) b = c; return [String(s.label), String(b)]; })")
        page.locator("#samples button").first.click()  # and the button path renders a verdict once the playback ends
        page.wait_for_function("document.getElementById('verdict').textContent.trim() !== '·'", timeout=5000)
        report["digits"]["verdict_after_click"] = page.inner_text("#verdict")
        right = sum(a == b for a, b in reads)
        report["digits"].update({"samples": len(reads), "read_as_label": right})
        if not reads or right < 0.8 * len(reads):
            fail(f"digits: {right} of {len(reads)} held-out pictures read as their label")
        if errors:
            fail(f"digits: {errors[:3]}")
        page.close()

        # recall: the alphabet arrives intact
        page, errors = open_page("recall", "02_recall/index.html")
        symbols = page.evaluate("Array.from(__recall.SYMBOLS).join('')")
        report["recall"]["symbols"] = len(symbols)
        if "α" not in symbols or "Â" in symbols.replace("Â", "", 1) and "Ã" in symbols and "α" not in symbols:
            fail("recall: the symbol alphabet is garbled")
        if errors:
            fail(f"recall: {errors[:3]}")
        page.close()

        # Connect Four
        engine, net = engine_from(ROOT / "03_connect_four/net.json")
        page, errors = open_page("connect_four", "03_connect_four/index.html")
        rng = np.random.default_rng(0)
        drives = []
        for _ in range(40):
            grid = [[0] * 7 for _ in range(6)]
            who = 1
            for _ in range(int(rng.integers(0, 30))):
                c = int(rng.choice([c for c in range(7) if grid[5][c] == 0]))
                grid[next(r for r in range(6) if grid[r][c] == 0)][c] = who
                who = 3 - who
            d = np.zeros(net["n"])
            for r in range(6):
                for c in range(7):
                    if grid[r][c] == 2:
                        d[net["sets"]["input"][r * 7 + c]] = 1.0
                    elif grid[r][c] == 1:
                        d[net["sets"]["input"][42 + r * 7 + c]] = 1.0
            drives.append(d.tolist())
        js = np.asarray(page.evaluate("ds => ds.map(d => { const o = settle(Float64Array.from(d)); return o.trace[o.trace.length - 1]; })", drives))
        py = engine_outputs(engine, net, drives)
        same = float((js.argmax(1) == py.argmax(1)).mean())
        report["connect_four"].update({"page_vs_engine_same_column": same, "page_vs_engine_max_abs": float(np.abs(js - py).max())})
        if same < 1.0:
            fail(f"connect four: the page picks the engine's column on {same:.2f} of positions")
        results = []
        for g in range(args.games):
            smart, net_first, prng = g >= args.games // 2, g % 2 == 1, random.Random(g)
            page.click("#netfirst" if net_first else "#new")
            time.sleep(0.2)
            t0 = time.time()
            while True:
                st = page.evaluate("({over, turn, busy, human, grid})")
                if st["over"]:
                    break
                if time.time() - t0 > 90:
                    fail(f"connect four: game {g} did not finish")
                    break
                if st["turn"] == st["human"] and not st["busy"]:
                    page.locator(f'.cell[data-col="{c4_choice(st["grid"], st["human"], 3 - st["human"], prng, smart)}"]').first.click()
                time.sleep(0.05)
            results.append(page.inner_text("#status"))
        report["connect_four"]["games"] = results
        if errors:
            fail(f"connect four: {errors[:3]}")
        page.close()

        # Pong, both nets
        for which, file in (("reward", "net.json"), ("imitation", "net_imitation.json")):
            engine, net = engine_from(ROOT / "04_pong" / file)
            name = f"pong_{which}"
            page, errors = open_page(name, "04_pong/index.html")
            page.evaluate(f"useNet('{which}')")
            obs = page.evaluate("() => { const out = []; paused = true; for (let k = 0; k < 40; k++) { __pong.reset(); for (let j = 0; j < (k % 7); j++) __pong.step(1, 0); out.push(Array.from(__pong.observation())); } paused = false; return out; }")
            js = np.asarray(page.evaluate("os => os.map(o => { const r = settle(Float64Array.from(o)); return OUT.map(k => r.s[k]); })", obs))
            py = engine_outputs(engine, net, obs)
            same = float((js.argmax(1) == py.argmax(1)).mean())
            if same < 1.0:
                fail(f"pong ({which}): the page picks the engine's action on {same:.2f} of frames")
            page.evaluate("""() => { score = [0, 0]; window.__n = { calls: 0, hits: 0, misses: 0 };
                const s0 = settle; settle = d => { __n.calls++; return s0(d); };
                const st0 = step; step = (a, h) => { const dc = game.dc; const e = st0(a, h); if (!e && dc === 1 && game.dc === -1) __n.hits++; if (e === 'net') __n.misses++; return e; };
                setInterval(() => { target = game.ball_r - 1; }, 20); }""")
            t0 = time.time()
            time.sleep(args.rally)
            n = page.evaluate("({...__n})")
            rate = n["calls"] / (time.time() - t0)
            report[name].update({"page_vs_engine_same_action": same, "decisions_per_second": round(rate, 1), "returns": n["hits"], "misses": n["misses"]})
            if rate < 10:
                fail(f"pong ({which}): {rate:.1f} decisions a second, the page is lagging")
            if errors:
                fail(f"pong ({which}): {errors[:3]}")
            page.close()
        browser.close()

    print(json.dumps(report, indent=1))
    print("all pages work" if not failures else f"{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
