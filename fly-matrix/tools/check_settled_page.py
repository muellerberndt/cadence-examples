#!/usr/bin/env python3
"""Exercise actual browser workers, mode isolation, failures and strict visual authority.

Run with Playwright Chromium installed; optionally pass --url to an existing local server.
No historical receipt is rewritten. A temporary loopback server is otherwise used.
"""
from __future__ import annotations

import argparse
import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
READ = """() => {const a=window.__app; return {mode:a.S.pilot, source:a.life().source,
  state:a.life().commandStatus, controls:a.life().controls, time:a.S.simTime,
  position:a.flight().p, camera:a.rig.camera.position.toArray(), closeup:a.rig.closeup.position.toArray(), learning:a.S.learnReady, generation:a.S.generation};}"""


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


def check(url, full=False):
    checks = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url + ("?controller=brain" if full else "?controller=brain&noscan=1&nobloom=1&noviews=1"), wait_until="domcontentloaded")
        page.wait_for_function("window.__app && window.__app.S.ready", timeout=60000)
        start = page.evaluate("({wall:performance.now(), time:window.__app.S.simTime, p:window.__app.flight().p.slice()})")
        page.wait_for_function("window.__app.S.simTime >= .5 && window.__app.S.controlReplies >= 1", timeout=60000)
        motion = page.evaluate("({wall:performance.now(), time:window.__app.S.simTime, p:window.__app.flight().p.slice(), replies:window.__app.S.controlReplies, late:window.__app.S.controlLate, frames:window.__frames})")
        assert motion["time"] - start["time"] >= .35
        assert motion["p"][2] < start["p"][2] - .35
        assert motion["wall"] - start["wall"] < 15000, motion
        checks.append("full physical body moves while asynchronous neural solves run")
        state = page.evaluate(READ)
        assert state["mode"] == "brain" and state["source"] == "patchnet" and not state["learning"]
        assert state["position"][2] < 1.2  # no supplied hover
        checks.append("unassisted actual worker reports settled or failed outcomes without learner or pilot")

        # Pause during an outstanding request: delayed worker replies cannot advance the body.
        page.wait_for_function("window.__app.S.pending", timeout=15000)
        page.keyboard.press("Space")
        page.wait_for_function("frame => window.__frames >= frame + 2", arg=page.evaluate("window.__frames"), timeout=15000)  # finish displaying the final physical pose
        paused = page.evaluate(READ)
        page.wait_for_timeout(1800)
        after_pause = page.evaluate(READ)
        assert after_pause == paused, {"paused": paused, "later": after_pause}
        assert all(value == 0 for value in paused["controls"].values())
        checks.append("pause revokes power and rejects an in-flight worker reply")

        # Replace a worker and inspect synchronously before any new ready message.
        initial = page.evaluate("""() => {window.__app.setPilot('brain'); return {
          ready:window.__app.S.ready, p:window.__app.flight().p, t:window.__app.S.simTime};}""")
        assert not initial["ready"] and initial["p"] == [1.2, 1, 1.2] and initial["t"] == 0
        page.keyboard.press("Space")
        page.wait_for_function("window.__app.S.simTime >= .02", timeout=60000)
        checks.append("reset holds the declared initial condition while loading a fresh worker")

        page.evaluate("window.__app.setPilot('motoroff')")
        page.wait_for_function("window.__app.S.simTime >= .1", timeout=60000)
        off = page.evaluate(READ)
        assert all(value == 0 for value in off["controls"].values())
        assert off["position"][2] < 1.2 and off["source"] == "patchnet"
        checks.append("motor-off mode falls under passive physics with zero active commands")

        page.evaluate("window.__app.setPilot('legacy')")
        page.wait_for_function("window.__app.S.learnReady && window.__app.S.ready", timeout=60000)
        assert page.evaluate("window.__app.life().constructor.name") == "Life"
        page.evaluate("window.__oldWorker = window.__app.worker; window.__app.setPilot('brain')")
        page.wait_for_function("window.__app.S.simTime >= .02", timeout=60000)
        assert page.evaluate("window.__app.life().constructor.name") == "SettledLife"
        assert not page.evaluate("window.__app.S.learnReady")
        page.evaluate("window.__oldWorker.onerror({message:'late error from terminated comparison'})")
        assert page.evaluate("window.__app.S.brainReady")
        checks.append("legacy comparison is explicit; learner and stale worker errors cannot survive a mode change")

        # Trigger the real page's transport failure handler; no hover fallback can remain.
        page.evaluate("window.__app.worker.onmessageerror(new MessageEvent('messageerror'))")
        failed = page.evaluate(READ)
        assert all(value == 0 for value in failed["controls"].values())
        page.wait_for_function("!window.__app.S.brainReady && !window.__app.S.pending")
        checks.append("transport failure removes motor authority")

        # Synthetic admission fixture, not a claim that the connectome learned to fly:
        # callback arrival cannot retroactively power the preceding physics interval.
        queued = page.evaluate("""async () => {
          const {SETTLED_MOTOR_GROUPS}=await import('./motor.js');
          // Start after the page's current animation callback. Otherwise a RAF
          // timestamp queued before this evaluator can make the next interval zero.
          await new Promise(resolve=>requestAnimationFrame(resolve));
          const a=window.__app,L=a.life();a.worker.terminate();L.revokeControl('test fixture');
          a.S.speed=.5;a.S.clockReset=false;
          const request={requestId:++a.S.requestId,generation:a.S.generation,steps:256,tolerance:1e-6};
          L.expectControl(request);a.S.pending=true;a.S.pendingId=request.requestId;
          a.S.pendingSince=performance.now();a.S.requestDeadline=L.clock+L.commandTTL;
          const oldLate=a.S.controlLate;
          a.worker.onmessage({data:{type:'control',kind:'control',...request,converged:true,
            iterations:1,residual:0,tolerance:1e-6,readouts:Object.fromEntries(SETTLED_MOTOR_GROUPS.map(k=>[k,k.startsWith('power:')?1:0]))}});
          const queued=!!a.S.controlReply && Object.values(L.controls).every(v=>v===0);
          const start=performance.now();while(performance.now()-start<300) {} // one deliberately delayed display frame
          return {queued,oldLate};
        }""")
        assert queued["queued"], "worker callback applied force before advancing elapsed physics"
        page.wait_for_function("old => window.__app.S.controlLate > old", arg=queued["oldLate"], timeout=15000)
        assert all(value == 0 for value in page.evaluate(READ)["controls"].values())
        checks.append("queued replies cannot power past physics or survive a missed deadline")

        # Test renderer separately: a neural pose never samples an idle motion policy.
        visual = page.evaluate("""async () => {
          const {createFly}=await import('./fly.js'), {Flight}=await import('./body.js');
          const fly=createFly(), body=new Flight([1,1,1]), c={aL:0,aR:0,betaL:0,betaR:0,sL:0,sR:0,f:0};
          const snap=()=>{const values=[];fly.group.traverse(o=>{
            values.push(...o.position.toArray(),...o.quaternion.toArray(),...o.scale.toArray());
            if(o.geometry?.attributes?.position) values.push(...o.geometry.attributes.position.array);
          });return JSON.stringify(values);};
          const original=Math.random; Math.random=()=>{throw Error('scripted random motion');};
          try {
            fly.update(body,c,.02,()=>0,false,{neural:true,mode:'flying',t:0,feed:0});
            const first=snap(); body.phase=1;
            fly.update(body,c,.5,()=>0,false,{neural:true,mode:'grooming',groom:'head',t:99,feed:0});
            const still=first===snap();
            c.aL=1;c.f=200;body.phase=0;
            fly.update(body,c,.02,()=>0,false,{neural:true,mode:'flying',feed:0});const wing=snap();
            body.phase=1;fly.update(body,c,.02,()=>0,false,{neural:true,mode:'flying',feed:0});
            return {still, motorMoves:wing!==snap()};
          } finally {Math.random=original;}
        }""")
        assert visual == {"still": True, "motorMoves": True}, visual
        checks.append("strict renderer has no idle behavioral motion; powered wings follow body phase")
        assert not errors, errors
        browser.close()
    return {"passed": True, "full_visualization": full, "motion": {"start": start, "end": motion}, "checks": checks, "page_errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url")
    parser.add_argument("--full", action="store_true", help="test the full atlas, eye, close-up and bloom")
    args = parser.parse_args()
    server = None
    try:
        if args.url:
            url = args.url.rstrip("/") + "/"
        else:
            server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(ROOT / "web")))
            threading.Thread(target=server.serve_forever, daemon=True).start()
            url = f"http://127.0.0.1:{server.server_port}/"
        print(json.dumps(check(url, full=args.full), indent=2))
    finally:
        if server:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    main()
