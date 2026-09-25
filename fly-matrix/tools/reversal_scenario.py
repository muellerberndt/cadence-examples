# The reversal scenario on the real page, as a visitor would run it: sugar on the banana until the fly feeds
# there, then the sugar on the bread and a blow whenever the fly sits on the banana (the first through a real
# click on the fly, the rest through the page's strike), until it finds the sugar on the bread; then a spell of
# free behaviour to count where it lands and what it found. Playwright's own Chromium (headless, SwiftShader)
# against a local static server of web/; PASS when the fly finds the sugar on the bread.
#   python tools/reversal_scenario.py [--seed 1] [--speed 2] [--headed] [--first banana|bread]
# Each direction's numbers go into receipts/page_reversal.json (--receipt), one entry per first fruit.
# At speed 2 a run takes 10 to 20 minutes of wall time: the worker's brain step dominates.
import argparse, hashlib, json, os, subprocess, sys, time, socket
from playwright.sync_api import sync_playwright
ap = argparse.ArgumentParser()
ap.add_argument("--web", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web"))
ap.add_argument("--seed", type=int, default=1); ap.add_argument("--speed", type=float, default=2.0)
ap.add_argument("--phase1-max", type=float, default=300); ap.add_argument("--phase2-max", type=float, default=900); ap.add_argument("--after", type=float, default=240)
ap.add_argument("--headed", action="store_true"); ap.add_argument("--pilot", default="brain")
ap.add_argument("--receipt", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "receipts", "page_reversal.json"))
ap.add_argument("--first", default="banana", choices=["banana", "bread"], help="the fruit the sugar starts on; the other is the reversal")
a = ap.parse_args()
s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
srv = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"], cwd=a.web, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(0.8)
STATE = """() => { const app = window.__app, L = app.life(); return { clock: L.clock, mode: L.mode, on: L.onWhich || null, visits: L.visits, sweet: L.visits.sweet, hunger: L.hunger, lastP: L.lastP, search: L.search ? { fruit: L.search.fruit, asked: !!L.search.asked, landed: L.search.landed || null } : null, events: L.events.slice(-40), decisions: app.S.decisions, lessons: app.S.lessons.slice(), fps: app.S.fps, sugar: L.sugar, smelled: L.smelled, pending: app.S.pending }; }"""
PROJECT = """async () => { const THREE = await import("three"); const app = window.__app, p = app.flight().p; const v = new THREE.Vector3(p[0], p[2], -p[1]).project(app.rig.camera); return [(v.x + 1) / 2 * innerWidth, (1 - v.y) / 2 * innerHeight]; }"""
seen = set()
def tick(page, log):
    st = page.evaluate(STATE)
    for t, text in st["events"]:
        key = (round(t, 3), text)
        if key not in seen: seen.add(key); log.append((t, text))
    return st
def count(log, needle, since=0): return sum(1 for t, x in log if x.startswith(needle) and t >= since)
try:
    with sync_playwright() as p:
        args = ["--use-angle=metal", "--window-size=1400,900"] if a.headed else ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--window-size=1400,900"]
        browser = p.chromium.launch(headless=not a.headed, args=args)
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        errors = []; page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"http://127.0.0.1:{port}/index.html?seed={a.seed}&noscan=1&nobloom=1&noviews=1")
        page.wait_for_function("window.__app && window.__app.S.ready && window.__app.S.learnReady", timeout=180000)
        A, B = a.first, ("bread" if a.first == "banana" else "banana")  # the first fruit, and the other
        page.evaluate(f"window.__app.setCam('follow'); window.__app.setPilot('{a.pilot}'); window.__app.S.speed = {a.speed}; window.__app.setSugar('{A}');")
        log = []; t0 = time.time()
        # phase 1: sugar on the first fruit until the fly has fed there
        st = tick(page, log); assert st["sugar"] == A, st["sugar"]
        while st["clock"] < a.phase1_max and count(log, f"lands on the {A}: sugar") < 1:
            time.sleep(0.25); st = tick(page, log)
        fed1 = count(log, f"lands on the {A}: sugar"); t_fed1 = st["clock"]; wall1 = time.time() - t0; lessons1 = len(st["lessons"])
        print(f"phase 1: {'fed on the ' + A if fed1 else 'never fed'} at {st['clock']:.0f} s sim ({time.time() - t0:.0f} s wall, {st['fps']:.0f} fps); lessons so far {len(st['lessons'])}, dopamine {[round(d, 2) for d in st['lessons']]}", flush=True)
        # phase 2: the sugar on the other fruit; a blow whenever the fly sits on the first
        page.evaluate(f"window.__app.setSugar('{B}')")
        t_switch = st["clock"]; blows = 0; clicked = False; last_blow_clock = -10
        while st["clock"] - t_switch < a.phase2_max and count(log, f"lands on the {B}: sugar") < 1:
            time.sleep(0.2); st = tick(page, log)
            if st["mode"] in ("landed", "grooming", "feeding") and st["on"] == A and st["clock"] - last_blow_clock > 1.0:
                if not clicked:  # the first blow through the real click path
                    x, y = page.evaluate(PROJECT); page.mouse.click(x, y); time.sleep(0.3); st2 = tick(page, log)
                    clicked = count(log, f"struck at the {A}") >= 1
                    if not clicked: print(f"  the click at ({x:.0f},{y:.0f}) did not strike (mode {st2['mode']}, on {st2['on']}); falling back to the page's strike()", flush=True); page.evaluate("window.__app.strike()")
                else: page.evaluate("window.__app.strike()")
                blows += 1; last_blow_clock = st["clock"]; st = tick(page, log)
                print(f"  blow {blows} at {st['clock']:.0f} s: p(approach) banana {st['lastP']['banana']} bread {st['lastP']['bread']}; dopamine so far {[round(d, 2) for d in st['lessons'][-6:]]}", flush=True)
        found = count(log, f"lands on the {B}: sugar") >= 1; t_found = st["clock"]; wall2 = time.time() - t0
        p2_first, p2_other = count(log, f"lands on the {A}", t_switch), count(log, f"lands on the {B}", t_switch)  # the phase's own counts, before the free spell
        print(f"phase 2: {'found the sugar on the ' + B if found else 'did not find the ' + B} after {blows} blows, {st['clock'] - t_switch:.0f} s sim ({time.time() - t0:.0f} s wall); {A} landings {count(log, f'lands on the {A}', t_switch)}, {B} landings {count(log, f'lands on the {B}', t_switch)}", flush=True)
        # afterwards: free behaviour with the sugar on the bread, blows continue at the banana
        t_after = st["clock"]; blows2 = 0
        while st["clock"] - t_after < a.after:
            time.sleep(0.25); st = tick(page, log)
            if st["mode"] in ("landed", "grooming", "feeding") and st["on"] == A and st["clock"] - last_blow_clock > 1.0:
                page.evaluate("window.__app.strike()"); blows2 += 1; last_blow_clock = st["clock"]
        st = tick(page, log)
        print(f"after: in {a.after:.0f} s with the sugar on the {B}: {A} landings {count(log, f'lands on the {A}', t_after)} ({blows2} blows), {B} landings {count(log, f'lands on the {B}', t_after)}, sugar found {count(log, f'lands on the {B}: sugar', t_after)} times; p(approach) banana {st['lastP']['banana']} bread {st['lastP']['bread']}; {len(st['lessons'])} lessons in all", flush=True)
        print("timeline:"); [print(f"  {t:6.1f} s  {x}") for t, x in log if any(k in x for k in ("sugar", "struck", "approaches", "avoids", "lands", "smells", "left the", "dopamine", "moved", "picked"))]
        print("page errors:", errors[:5])
        print("RESULT", "PASS" if found else "FAIL")
        rec = {"seed": a.seed, "speed": a.speed, "pilot": a.pilot, "first": A, "other": B,
               "phase1": {"fed": fed1 >= 1, "at_s": t_fed1, "wall_s": wall1, "lessons": lessons1},
               "phase2": {"found": found, "blows": blows, "seconds": t_found - t_switch, "wall_s": wall2 - wall1, "first_landings": p2_first, "other_landings": p2_other},
               "after": {"seconds": a.after, "first_landings": count(log, f"lands on the {A}", t_after), "blows": blows2, "other_landings": count(log, f"lands on the {B}", t_after), "sugar_found": count(log, f"lands on the {B}: sugar", t_after)},
               "p_approach": {"banana": st["lastP"]["banana"], "bread": st["lastP"]["bread"]}, "lessons": len(st["lessons"]), "brain_timeouts": sum(1 for _, x in log if "did not answer" in x), "fps": st["fps"], "wall_s": time.time() - t0, "page_errors": errors[:5], "result": "PASS" if found else "FAIL"}
        try: page_commit = subprocess.check_output(["git", "-C", a.web, "rev-parse", "HEAD"], text=True).strip()
        except Exception: page_commit = None
        data = os.path.join(a.web, "data"); payload = {f: hashlib.sha256(open(os.path.join(data, f), "rb").read()).hexdigest() for f in sorted(os.listdir(data)) if f.endswith(".json")}
        receipt = json.load(open(a.receipt)) if os.path.exists(a.receipt) else {}
        receipt.update({"tool": "tools/reversal_scenario.py", "page": page_commit, "payload": payload}); receipt.setdefault("directions", {})[f"{A} first, seed {a.seed}"] = dict(rec, page=page_commit)
        os.makedirs(os.path.dirname(a.receipt), exist_ok=True); json.dump(receipt, open(a.receipt, "w"), indent=1)
        print("receipt:", os.path.relpath(a.receipt))
        browser.close()
finally:
    srv.terminate()
