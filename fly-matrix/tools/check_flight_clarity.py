#!/usr/bin/env python3
"""Bounded real-browser flight and on-demand inspector check; new outputs only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import traceback
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright
from browser_source_binding import ServedSourceBinding, source_hashes

ROOT = Path(__file__).resolve().parents[1]
RUNNER = "tools/check_flight_clarity.py"
RECORD = """(()=>{
 window.__clarityFrames=0;window.__clarityRequests=[];window.__clarityReplies=[];
 const pending=new Map();function frame(){__clarityFrames++;requestAnimationFrame(frame)}requestAnimationFrame(frame);
 const send=Worker.prototype.postMessage;
 Worker.prototype.postMessage=function(m,...args){
  if(m.requestId!=null){
   if(['assist:observe','assist:decide','assist:reward','inspect:retina'].includes(m.type))pending.set(m.generation+':'+m.requestId,{wall:performance.now(),type:m.type});
   __clarityRequests.push({type:m.type,requestId:m.requestId,generation:m.generation,trace:!!m.trace,
    retina:m.retina?{width:m.retina.width,height:m.retina.height,bytes:m.retina.rgba.length}:null});}
  return send.call(this,m,...args);};
 const Native=Worker;window.Worker=new Proxy(Native,{construct(T,args){const w=new T(...args);
  w.addEventListener('message',e=>{const m=e.data,key=m.generation+':'+m.requestId,start=pending.get(key);
   if(start&&({'assist:observe':'control','assist:decide':'assisted_decision','assist:reward':'lesson','inspect:retina':'retina_diagnostic'}[start.type]===m.type)){
   __clarityReplies.push({type:m.type,requestId:m.requestId,generation:m.generation,roundTripMs:performance.now()-start.wall,
    workerMs:m.ms??null,iterations:m.iterations??null,residual:m.residual??null,converged:m.converged??null,
    retinal:m.retinal??null,trace:m.trace?{identity:m.trace.identity,n:m.trace.n,frames:m.trace.frames.length}:null});pending.delete(key);}});return w;}});
})();"""
SAMPLE = """()=>{const a=__app,f=a.flight(),L=a.life(),nav=L.authority.navigation;return {
 wall:performance.now(),simTime:L.clock,frames:__clarityFrames,pageFps:a.S.fps,details:a.details(),
 p:f.p.slice(),v:f.v.slice(),q:f.q.slice(),bodyYaw:f.euler()[2],velocityHeading:Math.atan2(f.v[1],f.v[0]),
 horizontalSpeed:Math.hypot(f.v[0],f.v[1]),requestedHeading:L.heading,facingHeading:L.facingHeading,
 command:structuredClone(nav.command),active:nav.active,framesApplied:nav.framesApplied,
 settleMs:a.S.solveMs,accepted:a.S.controlAccepted,late:a.S.controlLate,failed:a.S.controlFailed,
 observationsAccepted:L.authority.observations.accepted,
 cameraOffset:[a.rig.camera.position.x-f.p[0],-a.rig.camera.position.z-f.p[1],a.rig.camera.position.y-f.p[2]],
 neuralState:a.states().neural.state,pathsEnabled:a.paths().enabled,
 choices:structuredClone(L.authority.choices),lastChoice:L.authority.lastChoice?{
 requestId:L.authority.lastChoice.requestId,searchToken:L.authority.lastChoice.searchToken,
 executedAt:L.authority.lastChoice.executedAt,fruit:L.authority.lastChoice.fruit,action:L.authority.lastChoice.action}:null,
 pending:{requestId:a.S.pendingId,worker:a.S.pending,decision:a.S.assistedDecision??null},
 search:L.search?{pendingUntil:L.search.pendingUntil,neuralToken:L.search.neuralToken,
 choice:L.search.neuralChoice?{requestId:L.search.neuralChoice.requestId,executed:L.search.neuralChoice.executed,
 executedAt:L.search.neuralChoice.executedAt,revoked:L.search.neuralChoice.revoked}:null}:null};}"""
WINDOW = """async source=>{const sample=eval('('+source+')'),rows=[sample()],end=performance.now()+8000;
 while(performance.now()<end){await new Promise(r=>setTimeout(r,200));rows.push(sample());}return rows;}"""
PANEL_IDS = {"brain": "brain", "paths": "path-panel", "instruments": "panel", "options": "options", "about": "card"}


def visibility(page):
    return page.evaluate("""ids=>Object.fromEntries(ids.map(id=>{const e=document.getElementById(id),s=getComputedStyle(e),r=e.getBoundingClientRect();
      return[id,{visible:s.display!=='none'&&s.visibility!=='hidden'&&r.width>0&&r.height>0,
      left:r.left,right:r.right,top:r.top,bottom:r.bottom,width:r.width,height:r.height}];}))""",
                         [*PANEL_IDS.values(), "eye", "inset", "status", "toolbar"])


def panel(page, expected):
    page.wait_for_function("x=>__app.details().panel===x", arg=expected)
    seen = visibility(page)
    for name, id_ in PANEL_IDS.items():
        assert seen[id_]["visible"] == (name == expected), (expected, name, seen[id_])
    return seen


def bounds(page, width, height):
    result = page.evaluate("""()=>Object.fromEntries(['state-hunger','state-activity','state-steps','state-learning','pause','gust','nudge','inspect-brain','more'].map(id=>{
      const e=document.getElementById(id),r=e.getBoundingClientRect(),hit=document.elementFromPoint(r.x+r.width/2,r.y+r.height/2);
      return[id,{left:r.left,right:r.right,top:r.top,bottom:r.bottom,unobscured:!!hit&&(hit===e||e.contains(hit))}];}))""")
    for id_, box in result.items():
        assert 0 <= box["left"] < box["right"] <= width and 0 <= box["top"] < box["bottom"] <= height, (id_, box)
        assert box["unobscured"], (id_, box)
    return result


def wait_panel_transition(page, id_):
    page.wait_for_function("""id=>document.getElementById(id).getAnimations().filter(a=>a instanceof CSSTransition)
      .every(a=>a.playState!=='running')""", arg=id_, timeout=10000)


def record_image(page, path):
    page.screenshot(path=str(path))
    return {"path": str(path.resolve().relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def summary(rows):
    first, last = rows[0], rows[-1]
    wall = (last["wall"] - first["wall"]) / 1000
    wrap = lambda x: (x + math.pi) % (2 * math.pi) - math.pi
    active = [s for s in rows if s["active"]]
    moving = [s for s in active if s["horizontalSpeed"] > .005 and s["simTime"] >= first["simTime"] + .5]
    angles = [abs(wrap(s["bodyYaw"] - s["velocityHeading"])) * 180 / math.pi for s in moving]
    return {"wallSeconds": wall, "simulatedSeconds": last["simTime"] - first["simTime"],
            "simulationPerWall": (last["simTime"] - first["simTime"]) / wall,
            "independentFps": (last["frames"] - first["frames"]) / wall,
            "horizontalDisplacementM": math.hypot(last["p"][0] - first["p"][0], last["p"][1] - first["p"][1]),
            "sampledHorizontalPathLengthM": sum(math.hypot(b["p"][0] - a["p"][0], b["p"][1] - a["p"][1]) for a, b in zip(rows, rows[1:])),
            "sampleCount": len(rows), "activeSamples": len(active), "activeSampleFraction": len(active) / len(rows),
            "neuralActuationSteps": last["framesApplied"] - first["framesApplied"],
            "neuralActuationSimulatedSeconds": (last["framesApplied"] - first["framesApplied"]) * .0005,
            "neuralActuationTimeFraction": (last["framesApplied"] - first["framesApplied"]) * .0005 / (last["simTime"] - first["simTime"]) if last["simTime"] > first["simTime"] else None,
            "medianActiveMovingNoseTravelErrorDeg": statistics.median(angles) if angles else None,
            "activeMovingNoseTravelErrorsDeg": angles,
            "acceptedRepliesDuringWindow": last["accepted"] - first["accepted"],
            "acceptedMotorObservationsDuringWindow": last["observationsAccepted"] - first["observationsAccepted"],
            "bodyYawChangeDeg": wrap(last["bodyYaw"] - first["bodyYaw"]) * 180 / math.pi,
            "maximumCameraOffsetChangeM": max(math.dist(s["cameraOffset"], first["cameraOffset"]) for s in rows),
            "lateRepliesDuringWindow": last["late"] - first["late"],
            "droppedWallSecondsDuringWindow": last["details"]["droppedWallSeconds"] - first["details"]["droppedWallSeconds"]}


def run(args):
    frozen = source_hashes(ROOT, RUNNER)
    result = {"schema": "cadence.flight-clarity-browser/1", "passed": False, "started": datetime.now(timezone.utc).isoformat(),
              "sha256": frozen, "checks": [], "page_errors": [], "protocol": {
                  "url": args.url, "viewport": [1440, 1000], "mobile_viewport": [430, 900], "wall_window_ms": 8000,
                  "sample_interval_ms": 200, "minimum_displacement_m": .05, "maximum_median_nose_travel_error_deg": 30,
                  "default_simulation_speed": .5, "physics_step_seconds": .0005,
                  "windows": "Initial8wallsec immediately after brain/learner ready, includingstartup. Then wait at most30wallsec for persistent actual executed-choice counter, and capture8wallsec. Both retained; the latter is the movement/alignment acceptance window. Applied-step differences measure actual actuation time even when an authority interval begins and expires within one rendering frame.",
                  "angle_scope": "Actual body yaw versus actual horizontal velocity, active route, speed>0.005m/s, at least0.5simsec after first sample.",
                  "limits": "Software WebGL timing is machine-specific. One wiring/layout run does not establish navigation competence or a learning improvement. No neural values, body states, or route commands are fabricated."}}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        result["browser"] = browser.version
        binding = ServedSourceBinding(page.context, args.url, frozen)
        page.on("pageerror", lambda error: result["page_errors"].append(str(error)))
        page.add_init_script(RECORD)
        try:
            page.goto(args.url, wait_until="domcontentloaded")
            page.wait_for_function("window.__app?.S.ready&&__app.S.learnReady", timeout=90000)
            result["default_visibility"] = panel(page, None)
            details = page.evaluate("__app.details()")
            assert details["speed"] == .5 and not details["extraViews"]
            assert not result["default_visibility"]["eye"]["visible"] and not result["default_visibility"]["inset"]["visible"]
            assert page.evaluate("Array.from(document.querySelectorAll('#toolbar>button')).map(e=>e.id)") == ["pause", "gust", "nudge", "inspect-brain", "more"]
            result["checks"].append("plain default has five primary controls, compact metrics, no open diagnostic/path/extra views, and explicit half-speed setting")
            initial_rows = page.evaluate(WINDOW, SAMPLE)
            result["initial_samples"] = initial_rows
            result["initial_summary"] = summary(initial_rows)
            print(json.dumps({"initial_flight_summary": result["initial_summary"]}), flush=True)
            page.wait_for_function("()=>{const s=getComputedStyle(document.getElementById('loading'));return s.opacity==='0'&&s.pointerEvents==='none';}", timeout=10000)
            result["desktop_layout"] = bounds(page, 1440, 1000)
            page.wait_for_function("__app.life().authority.choices.executed>0", timeout=30000)
            result["executed_choice_window_start"] = page.evaluate(SAMPLE)
            rows = page.evaluate(WINDOW, SAMPLE)
            result["samples"] = rows; result["summary"] = summary(rows)
            result["desktop_screenshot"] = record_image(page, args.screenshot)
            result["default_requests"] = page.evaluate("__clarityRequests")
            result["default_replies"] = page.evaluate("__clarityReplies")
            for reply in result["default_replies"]:
                if reply["workerMs"] is not None:
                    assert reply["roundTripMs"] + 1 >= reply["workerMs"], ("invalid request timing", reply)
            observations = [r for r in result["default_requests"] if r["type"] == "assist:observe"]
            assert observations and all(not r["trace"] and r["retina"] == {"width": 64, "height": 32, "bytes": 8192} for r in observations)
            assert any(r["type"] == "control" and r["converged"] and r["retinal"]["neurons"] == 1831 for r in result["default_replies"])
            assert result["summary"]["horizontalDisplacementM"] >= .05, result["summary"]
            assert result["summary"]["acceptedMotorObservationsDuringWindow"] > 0
            assert result["summary"]["neuralActuationSteps"] > 0
            assert result["summary"]["medianActiveMovingNoseTravelErrorDeg"] is not None and result["summary"]["medianActiveMovingNoseTravelErrorDeg"] < 30, result["summary"]
            assert result["summary"]["maximumCameraOffsetChangeM"] < 1e-9, result["summary"]
            result["checks"].append("hidden inspectors preserve real retinal observations and admitted neural motion; actual physical nose follows travel")
            # Publish early flight diagnostics to the caller before optional panels.
            print(json.dumps({"flight_summary": result["summary"]}), flush=True)
            page.locator('#inspect-brain').click(); panel(page, "brain")
            page.wait_for_function("()=>{const v=__app.settlement().view;return v.identity&&__clarityReplies.some(r=>r.type==='control'&&r.trace&&r.trace.n===150802&&r.trace.identity.requestId===v.identity.requestId&&r.trace.identity.generation===v.identity.generation);}", timeout=90000)
            result["inspector"] = page.evaluate("__app.settlement()")
            assert result["inspector"]["view"]["identity"] is not None
            page.locator('#brain [data-close-detail]').click(); panel(page, None)
            for name, button in (("paths", "paths"), ("instruments", "instruments"), ("about", "about")):
                page.locator('#more').click(); panel(page, "options")
                page.locator('#' + button).click(); result[name + "_visibility"] = panel(page, name)
                wait_panel_transition(page, PANEL_IDS[name])
                page.locator('#' + PANEL_IDS[name] + ' [data-close-detail]').click(); panel(page, None)
            result["checks"].append("brain, paths, senses and About open only on demand, one panel at a time, and normal close returns to flight")
            page.locator('#pause').click(); page.wait_for_function('__app.S.paused')
            paused = page.evaluate(SAMPLE)
            page.set_viewport_size({"width": 430, "height": 900})
            result["mobile_layout"] = bounds(page, 430, 900)
            result["mobile_default"] = panel(page, None)
            result["mobile_screenshot"] = record_image(page, args.mobile_screenshot)
            page.locator('#more').click(); panel(page, "options")
            page.locator('#about').click(); panel(page, "about"); wait_panel_transition(page, "card")
            result["mobile_about"] = visibility(page)["card"]
            box = result["mobile_about"]
            assert 0 <= box["left"] < box["right"] <= 430 and 0 <= box["top"] < box["bottom"] <= 900
            page.keyboard.press('Escape'); panel(page, None)
            after = page.evaluate(SAMPLE)
            assert paused["simTime"] == after["simTime"] and paused["q"] == after["q"] and paused["p"] == after["p"]
            page.locator('#more').click(); panel(page, "options")
            generation = page.evaluate('__app.S.generation')
            page.locator('#forget').click()
            page.wait_for_function('g=>__app.S.generation>g&&__app.S.ready&&__app.S.learnReady', arg=generation, timeout=90000)
            result["reset"] = page.evaluate('__app.states()')
            assert result["reset"]["neural"]["sample"] is None and result["reset"]["learningUpdates"] == 0
            result["checks"].append("mobile primary controls and metrics fit unobscured; About closes with Escape; pause freezes physics and reset clears state")
            result["served_sources"] = binding.receipt()
            assert not result["page_errors"]
            assert source_hashes(ROOT, RUNNER) == frozen
            result["passed"] = True
        except Exception as error:
            result["error"] = repr(error); result["traceback"] = traceback.format_exc()
            try:
                result["failure_state"] = page.evaluate(SAMPLE)
                result["failure_visibility"] = visibility(page)
                result["requests"] = page.evaluate('__clarityRequests')
                result["replies"] = page.evaluate('__clarityReplies')
            except Exception as capture_error:
                result["capture_error"] = repr(capture_error)
            try:
                if not args.screenshot.exists():
                    result["failure_screenshot"] = record_image(page, args.screenshot)
            except Exception as capture_error:
                result["screenshot_error"] = repr(capture_error)
        finally:
            result["sources_unchanged"] = source_hashes(ROOT, RUNNER) == frozen
            browser.close()
    result["completed"] = datetime.now(timezone.utc).isoformat()
    result["body_sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8817/')
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--screenshot', type=Path, required=True)
    parser.add_argument('--mobile-screenshot', type=Path, required=True)
    args = parser.parse_args()
    for path in (args.receipt, args.screenshot, args.mobile_screenshot):
        if path.exists():
            parser.error(f'refusing to overwrite {path}')
    result = run(args)
    args.receipt.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: result.get(key) for key in ('passed', 'checks', 'error', 'summary', 'body_sha256')}, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
