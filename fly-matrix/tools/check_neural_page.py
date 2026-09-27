#!/usr/bin/env python3
"""Check rendered pixels, checked neural route execution, and brain-off hover.

This uses the real page/worker and full retained payload. It never injects a
choice, score, retinal frame, route, reward or neural state. Browser instrumentation
only records outgoing sensory packets and incoming solver evidence. A screenshot
and the first actual RGBA frame can be retained. Failed bounded runs are explicitly
marked failed, with diagnostics; receipts and screenshots are never overwritten.
--after-active reports startup latency separately and begins the same motion
window after the first neural choice has actually supplied a route command.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import platform
import traceback
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from playwright.sync_api import sync_playwright
from browser_source_binding import ServedSourceBinding, source_hashes

ROOT = Path(__file__).resolve().parents[1]
MIN_XY_MOTION = .05  # Metres in >=2 simulated seconds; fixed before this follow-up run.
READ = """() => {const a=window.__app,L=a.life(),f=a.flight();return {
 wall:performance.now(),time:a.S.simTime,brain:a.brainInfo(),pilot:a.S.pilot,
 source:L.source,p:f.p.slice(),v:f.v.slice(),w:f.w.slice(),mode:L.mode,
 authority:L.authority,controls:{...L.controls},search:L.search,
 pilotState:{speed:L.pilot?.speed,heading:L.pilot?.heading,target:L.pilot?.target,route:L.pilot?.route},
 status:{pending:a.S.pending,brainReady:a.S.brainReady,learnReady:a.S.learnReady,
 solveMs:a.S.solveMs,failed:a.S.controlFailed,accepted:a.S.controlAccepted},
 lessons:a.S.lessonStats,events:L.events,packets:window.__neuralPackets,
 messages:window.__neuralMessages,firstRetinal:window.__firstRetinal};}"""
RECORD_TRANSPORT = """(() => {
 window.__neuralPackets=[];window.__neuralMessages=[];window.__firstRetinal=null;
 const original=Worker.prototype.postMessage;
 Worker.prototype.postMessage=function(m,...rest){
   if(m && ['assist:observe','assist:decide'].includes(m.type)){
     const r=m.retina;let info=null;
     if(r){let hash=2166136261,sum=0,min=255,max=0;
       for(let i=0;i<r.rgba.length;i++){hash=Math.imul(hash^r.rgba[i],16777619)>>>0;
         if(i%4!==3){sum+=r.rgba[i];min=Math.min(min,r.rgba[i]);max=Math.max(max,r.rgba[i]);}}
       info={width:r.width,height:r.height,origin:r.origin,bytes:r.rgba.length,
         fnv1a32:hash.toString(16).padStart(8,'0'),minRGB:min,maxRGB:max,meanRGB:sum/(r.width*r.height*3)};
       if(!window.__firstRetinal){let bin='';for(const v of r.rgba)bin+=String.fromCharCode(v);
         window.__firstRetinal={...info,rgba_base64:btoa(bin)};}
     }
     window.__neuralPackets.push({type:m.type,requestId:m.requestId,generation:m.generation,
       searchToken:m.searchToken,attention:m.attention??null,selectTarget:m.selectTarget===true,
       steps:m.steps,tolerance:m.tolerance,retinal:info});
   }
   return original.call(this,m,...rest);
 };
 const NativeWorker=window.Worker;
 window.Worker=new Proxy(NativeWorker,{construct(T,args){const w=new T(...args);
   w.addEventListener('message',e=>{const m=e.data;
     if(['control','assisted_decision','lesson'].includes(m.type))window.__neuralMessages.push({
       type:m.type,requestId:m.requestId,generation:m.generation,searchToken:m.searchToken,
       attention:m.attention,accepted:m.accepted,converged:m.converged,iterations:m.iterations,
       residual:m.residual,tolerance:m.tolerance,reason:m.reason,retinal:m.retinal,
       targetSelection:m.targetSelection,decision:m.decision,updates:m.updates,changed:m.changed,
       readouts:m.readouts?Object.fromEntries(Object.entries(m.readouts).filter(([k])=>
         ['dn:DNa02:left','dn:DNa02:right','dn:DNp09','dn:DNp03','dn:landing','mn9',
          'mbon:MBON11:right','mbon:MBON05:left'].includes(k))):undefined});
   });return w;}});
})();"""


def qualified(solve, cap=1024):
    assert solve["converged"] is True, solve
    residual, tolerance, iterations = solve["residual"], solve["tolerance"], solve["iterations"]
    assert type(residual) in (int, float) and math.isfinite(residual) and residual >= 0
    assert type(tolerance) in (int, float) and 0 < tolerance <= 1e-6 and residual <= tolerance
    assert type(iterations) is int and 0 <= iterations <= cap


def validate_target(target):
    candidates = target["candidates"]
    assert [c["fruit"] for c in candidates] == ["banana", "bread"]
    for candidate in candidates:
        qualified(candidate["solve"])
        expected = candidate["readouts"]["mbon:MBON11:right"] - candidate["readouts"]["mbon:MBON05:left"]
        assert math.isfinite(expected) and abs(candidate["score"] - expected) <= 1e-12
    temperature = target["temperature"]
    assert type(temperature) in (int, float) and math.isfinite(temperature) and temperature > 0
    logits = [c["score"] / temperature for c in candidates]
    weights = [math.exp(x - max(logits)) for x in logits]
    expected = [x / sum(weights) for x in weights]
    assert len(target["p"]) == 2 and all(abs(p - q) <= 1e-12 for p, q in zip(target["p"], expected))
    assert 0 <= target["draw"] < 1
    assert target["fruit"] == ("banana" if target["draw"] <= expected[0] else "bread")


def is_neutral(snapshot):
    navigation = snapshot["authority"]["navigation"]
    assert navigation["active"] is False
    assert all(value == 0 for value in navigation["command"].values())
    assert snapshot["pilotState"]["route"] is None
    assert snapshot["pilotState"]["speed"] == 0
    assert snapshot["search"] is None
    assert all(value == 0 for value in snapshot["authority"]["neuralTrim"].values())


def run(url, full=False, screenshot=None, after_active=False):
    frozen = source_hashes(ROOT, "tools/check_neural_page.py")
    result = {"schema": "cadence.neural-page-check/3", "passed": False,
              "started": datetime.now(timezone.utc).isoformat(), "full_views": full,
              "checks": [], "sha256": frozen, "page_errors": [],
              "protocol": {"seed": 1, "minimum_active_simulation_seconds": 2,
                           "window_origin": "first_executed_neural_route" if after_active else "initial_ready_snapshot",
                           "minimum_horizontal_displacement_metres": MIN_XY_MOTION,
                           "maximum_brain_off_horizontal_drift_metres": .01,
                           "followup": "Separate validation after the initial 6.6mm-motion diagnostic and addition of supplied hover-thrust projection; no claim of independent navigation confirmation or learning improvement.",
                           "startup_interpretation": "With --after-active, startup is measured separately; passing the active window does not erase the earlier hidden-view startup-window failure."},
              "claim": "Actual rendered pixels enter the full recurrent graph; qualified neural readouts supply route requests and odor attention; brain-off retains only stabilization.",
              "limits": "One seed and bounded start; supplied retinal mapping, attention protocol, decoders and HandPilot stabilization; no biological retinotopy, learned navigation, flight competence or whole-mind claim."}
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query)); query["seed"] = "1"
    if not full:
        query.update(noscan="1", nobloom="1", noviews="1")
    run_url = urlunsplit((*parts[:3], urlencode(query), parts.fragment))
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        result["runtime"] = {"python": platform.python_version(), "browser": browser.version}
        binding = ServedSourceBinding(page.context, urlunsplit((*parts[:3], "", "")), frozen)
        page.on("pageerror", lambda error: result["page_errors"].append(str(error)))
        page.add_init_script(RECORD_TRANSPORT)
        try:
            page.goto(run_url, wait_until="domcontentloaded")
            page.wait_for_function("window.__app?.S.ready && window.__app.S.learnReady", timeout=60000)
            binding.wait_for_complete(page)
            initial = result["initial"] = page.evaluate(READ)
            assert initial["source"] == "neural-navigation"
            assert initial["brain"]["n"] == initial["brain"]["whole"]["neurons"] == 150802
            assert initial["brain"]["edges"] == initial["brain"]["whole"]["edges"] == 1877099
            baseline = initial
            if after_active:
                page.wait_for_function("""()=>{const L=window.__app.life();return L.authority.navigation.active
                  && L.authority.navigation.framesApplied>0
                  && L.authority.lastChoice?.executed===true;}""", timeout=90000)
                baseline = result["activation_baseline"] = page.evaluate(READ)
                assert baseline["authority"]["navigation"]["framesApplied"] > 0
                first_execution = baseline["authority"]["lastChoice"]["executedAt"]
                result["startup_latency"] = {
                    "origin": "initial ready snapshot, after source binding",
                    "first_execution_simulation_time": first_execution,
                    "first_execution_simulation_seconds": first_execution - initial["time"],
                    "observed_active_simulation_seconds": baseline["time"] - initial["time"],
                    "observed_active_wall_seconds": (baseline["wall"] - initial["wall"]) / 1000,
                }
            page.wait_for_function("""t=>{const a=window.__app,L=a.life();return a.S.simTime>=t+2
              &&L.authority.navigation.framesApplied>0&&L.authority.targetSelection.accepted>0
              &&L.authority.choices.executed>0;}""", arg=baseline["time"], timeout=90000)
            active = result["active"] = page.evaluate(READ)
            assert active["authority"]["autonomous"] is False
            assert active["pilotState"]["route"] is None
            distance = math.dist(active["p"][:2], baseline["p"][:2])
            result["motion"] = {"horizontal_metres": distance,
                                "simulation_seconds": active["time"] - baseline["time"],
                                "wall_seconds": (active["wall"] - baseline["wall"]) / 1000}
            assert distance >= MIN_XY_MOTION, result["motion"]
            assert active["p"][2] > .5
            assert active["authority"]["retinal"]["neurons"] == 1831
            assert active["packets"] and all(m["retinal"] and m["retinal"]["bytes"] == 64*32*4 for m in active["packets"])
            first = active["firstRetinal"]
            rgba = base64.b64decode(first["rgba_base64"], validate=True)
            assert len(rgba) == 64*32*4 and first["origin"] == "bottom-left"
            assert first["maxRGB"] > first["minRGB"], "actual camera frame should contain scene contrast"
            first["sha256"] = hashlib.sha256(rgba).hexdigest()
            decisions = [m for m in active["messages"] if m["type"] == "assisted_decision" and m.get("accepted")]
            assert decisions
            for decision in decisions:
                validate_target(decision["targetSelection"])
                assert set(decision["decision"]["solves"]) == {"free", "plus", "minus"}
                for phase in decision["decision"]["solves"].values():
                    qualified(phase)
            choice = active["authority"]["lastChoice"]
            admitted = next(m for m in decisions if all(m[k] == choice[k] for k in ("requestId", "generation", "searchToken")))
            assert choice["executed"] and choice["fruit"] == admitted["targetSelection"]["fruit"]
            attended = [m for m in active["messages"] if m["type"] == "control" and m.get("converged")
                        and m.get("attention") == choice["fruit"] and m.get("searchToken") == choice["searchToken"]]
            assert attended, "used route must have a qualified observation of selected odor"
            for observed in attended:
                qualified(observed)
            result["checks"].append("rendered pixel packets, checked candidate competition, matching attended observation and actual route execution")
            if screenshot:
                page.screenshot(path=str(screenshot))
                result["screenshot"] = {"path": str(screenshot), "sha256": hashlib.sha256(screenshot.read_bytes()).hexdigest()}
            page.keyboard.press("Space")
            paused = page.evaluate(READ)
            page.wait_for_timeout(1000)
            assert page.evaluate("window.__app.flight().p.slice()") == paused["p"]
            is_neutral(page.evaluate(READ))
            result["checks"].append("pause preserves physical position and immediately revokes route and trim")
            page.keyboard.press("Space")
            page.get_by_role("button", name="brain off", exact=True).click()
            off_start = result["brain_off_baseline"] = page.evaluate(READ)
            page.wait_for_function("t=>window.__app.S.simTime>=t+2", arg=off_start["time"], timeout=60000)
            off = result["brain_off"] = page.evaluate(READ)
            assert off["pilot"] == "assisted-off"
            is_neutral(off)
            assert off["authority"]["observations"]["requested"] == 0
            assert off["authority"]["choices"]["executed"] == 0
            assert off["authority"]["targetSelection"]["requested"] == 0
            assert off["controls"]["f"] > 0 and off["p"][2] > .5
            assert math.dist(off["p"][:2], off_start["p"][:2]) < .01
            result["brain_off_comparison"] = {
                "simulation_seconds": off["time"] - off_start["time"],
                "wall_seconds": (off["wall"] - off_start["wall"]) / 1000,
                "horizontal_metres": math.dist(off["p"][:2], off_start["p"][:2]),
            }
            result["checks"].append("brain-off has no target/search/yaw/translation command and remains in stabilized hover")
            assert not result["page_errors"], result["page_errors"]
            result["served_sources"] = binding.receipt()
            assert source_hashes(ROOT, "tools/check_neural_page.py") == frozen, "source changed during browser run"
            result["passed"] = True
        except Exception as error:
            result["error"] = str(error)
            result["traceback"] = traceback.format_exc()
            try:
                result["failure_state"] = page.evaluate(READ)
            except Exception as diagnostic_error:
                result["diagnostic_error"] = str(diagnostic_error)
            try:
                result["served_sources"] = binding.receipt()
            except Exception as binding_error:
                result["binding_error"] = str(binding_error)
            result["sources_unchanged"] = source_hashes(ROOT, "tools/check_neural_page.py") == frozen
        finally:
            browser.close()
    result["completed"] = datetime.now(timezone.utc).isoformat()
    result["body_sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8817/")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--after-active", action="store_true", help="start the unchanged 2-second motion gate after first executed neural route; report startup separately")
    ap.add_argument("--receipt", type=Path)
    ap.add_argument("--screenshot", type=Path)
    args = ap.parse_args()
    for path in (args.receipt, args.screenshot):
        if path and path.exists():
            ap.error(f"refusing to overwrite {path}")
    outcome = run(args.url, args.full, args.screenshot, args.after_active)
    if args.receipt:
        with args.receipt.open("x") as out:
            json.dump(outcome, out, indent=2, allow_nan=False); out.write("\n")
    print(json.dumps({key: outcome.get(key) for key in ("passed", "checks", "error", "body_sha256")}, indent=2))
    raise SystemExit(0 if outcome["passed"] else 1)
