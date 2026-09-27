#!/usr/bin/env python3
"""Bounded, source-bound verification of the public neural fly and state cards.

One actual worker reply is delayed across a real G gust, then delivered unchanged
to test stale observation rejection. No neural values, actions or rewards are
fabricated. Complete request summaries, actual response measurements, failures
and screenshots are retained in a small receipt. Use a new output path.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import platform
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from playwright.sync_api import sync_playwright
from browser_source_binding import ServedSourceBinding, source_hashes

ROOT = Path(__file__).resolve().parents[1]
RUNNER = "tools/check_public_states.py"
RECORD = """(() => {
 const copy=x=>ArrayBuffer.isView(x)?Array.from(x):Array.isArray(x)?x.map(copy):
   x&&typeof x==='object'?Object.fromEntries(Object.entries(x).map(([k,v])=>[k,copy(v)])):x;
 const hash=bytes=>{let h=2166136261;for(const b of bytes||[])h=Math.imul(h^b,16777619);return(h>>>0).toString(16).padStart(8,'0');};
 window.__publicRequests=[];window.__publicMessages=[];window.__heldControls=[];window.__physicalEvents=[];
 window.__holdControls=false;window.__releasingControl=false;
 const send=Worker.prototype.postMessage;
 Worker.prototype.postMessage=function(m,...rest){
   window.__publicRequests.push(copy({type:m.type,requestId:m.requestId,generation:m.generation,
     senses:m.senses,attention:m.attention,searchToken:m.searchToken,steps:m.steps,tolerance:m.tolerance,
     reward:m.reward,done:m.done,neuralCredit:m.neuralCredit,
     retina:m.retina?{width:m.retina.width,height:m.retina.height,origin:m.retina.origin,
       rgbaFNV1a32:hash(m.retina.rgba)}:undefined}));
   if(window.__publicRequests.length>200)window.__publicRequests.shift();return send.call(this,m,...rest);};
 const Native=window.Worker;
 window.Worker=new Proxy(Native,{construct(T,args){const w=new T(...args);
   w.addEventListener('message',e=>{if(window.__releasingControl)return;const m=e.data;
     let above=0,equal=0,allFinite=true;
     for(const s of m.s||[]){above+=s>.5;equal+=s===.5;if(!Number.isFinite(s))allFinite=false;}
     window.__publicMessages.push(copy({type:m.type,kind:m.kind,requestId:m.requestId,generation:m.generation,
       initialResidual:m.initialResidual,residual:m.residual,tolerance:m.tolerance,converged:m.converged,
       iterations:m.iterations,reason:m.reason,active:m.active,n:m.s?.length,aboveHalf:above,equalHalf:equal,allFinite,
       accepted:m.accepted,lost:m.lost,updates:m.updates,changed:m.changed,reset:m.reset,
       searchToken:m.searchToken,attention:m.attention,
       trace:m.trace?{identity:m.trace.identity,n:m.trace.n,first:m.trace.frames?.[0]?{
         iteration:m.trace.frames[0].iteration,residual:m.trace.frames[0].residual,
         settledCount:m.trace.frames[0].settledCount}:null,final:m.trace.frames?.at(-1)?{
         iteration:m.trace.frames.at(-1).iteration,residual:m.trace.frames.at(-1).residual,
         settledCount:m.trace.frames.at(-1).settledCount}:null}:undefined}));
     if(window.__publicMessages.length>200)window.__publicMessages.shift();
     if(m.type==='control'&&window.__holdControls){window.__heldControls.push({worker:w,message:m});e.stopImmediatePropagation();}
   });return w;}});
 window.__releaseOldControl=()=>{const item=window.__heldControls.shift();if(!item)throw Error('no real reply held');
   window.__releasingControl=true;try{item.worker.dispatchEvent(new MessageEvent('message',{data:item.message}));}
   finally{window.__releasingControl=false;}return{requestId:item.message.requestId,generation:item.message.generation};};
 window.__releaseRemainingControls=()=>{window.__holdControls=false;const ids=[];while(window.__heldControls.length)ids.push(window.__releaseOldControl());return ids;};
 const body=()=>{const a=window.__app,f=a.flight();return{clock:a.life().clock,p:f.p.slice(),v:f.v.slice(),w:f.w.slice(),q:f.q.slice(),states:a.states(),
   issuedRequestId:a.S.requestId,choice:copy(a.life().search?.neuralChoice??null)};};
 const eventWitnesses=new WeakMap();
 const event=(kind,e)=>{if(!window.__app?.states)return;const item={kind,before:body()};
   window.__physicalEvents.push(item);eventWitnesses.set(e,item);};
 addEventListener('keydown',e=>{if(e.key.toLowerCase()==='g')event('gust',e);},true);
 addEventListener('click',e=>{if(e.target instanceof Element&&e.target.closest('#nudge'))event('nudge',e);},true);
 // Install after the page's module handlers: native-event microtasks can run
 // between listeners, so a microtask in the capture listener is not an after
 // witness. These bubble listeners record synchronously after the real handler.
 window.__installPhysicalAfterWitness=()=>{const finish=e=>{const item=eventWitnesses.get(e);if(item)item.after=body();};
   addEventListener('keydown',finish);addEventListener('click',finish);};
})();"""
READ = """()=>{const a=window.__app,L=a.life();return{
 wall:performance.now(),clock:L.clock,generation:a.S.generation,paused:a.S.paused,ready:a.S.ready,
 p:a.flight().p.slice(),v:a.flight().v.slice(),q:a.flight().q.slice(),omega:a.flight().w.slice(),
 source:L.source,neuralEnabled:L.neuralEnabled,hunger:L.hunger,states:a.states(),
 authority:L.authority,controls:{...L.controls},events:L.events.slice(-8),
 controlAccepted:a.S.controlAccepted,lessonStats:a.S.lessonStats,lastLesson:a.S.lastLesson,
 api:{setPilot:typeof a.setPilot},pilotControls:document.querySelectorAll('[data-pilot]').length,
 cards:Object.fromEntries(['hunger','activity','steps','learning','initial','residual','status'].map(k=>
   [k,document.getElementById('state-'+k)?.textContent??null])),
 requests:window.__publicRequests,messages:window.__publicMessages,
 held:window.__heldControls.map(x=>({requestId:x.message.requestId,generation:x.message.generation})),
 physicalEvents:window.__physicalEvents,brain:a.brainInfo(),
 bodyText:document.body.innerText};}"""


def near(a, b, tolerance=1e-12):
    assert type(a) in (int, float) and type(b) in (int, float)
    assert math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tolerance, (a, b)


def check_public(snapshot):
    assert snapshot["source"] == "neural-navigation" and snapshot["neuralEnabled"] is True
    assert snapshot["api"]["setPilot"] == "undefined" and snapshot["pilotControls"] == 0
    forbidden = re.search(r"\b(?:assistance|assisted|unassisted|legacy)\b|\bbrain[-\s]+off\b|\bmotors?[-\s]+off\b", snapshot["bodyText"], re.I)
    assert forbidden is None, forbidden.group(0) if forbidden else ""
    assert all(snapshot["cards"][name] is not None for name in snapshot["cards"])
    assert not any(request["type"] in ("shuffle", "learned", "settle_control") for request in snapshot["requests"])


def check_worker(message):
    assert message["type"] == "control" and message["kind"] == "control"
    assert type(message["initialResidual"]) in (int, float) and math.isfinite(message["initialResidual"]) and message["initialResidual"] >= 0
    trace = message["trace"]
    assert trace["n"] == 150802 and trace["first"]["iteration"] == 0
    assert trace["identity"]["requestId"] == message["requestId"] and trace["identity"]["generation"] == message["generation"]
    near(message["initialResidual"], trace["first"]["residual"], 0)
    near(message["residual"], trace["final"]["residual"], 0)
    assert message["iterations"] == trace["final"]["iteration"] and 0 <= message["iterations"] <= 1024
    if message["converged"]:
        assert 0 <= message["residual"] <= message["tolerance"] <= 1e-6
        assert message["n"] == 150802 and message["allFinite"]
        # The worker counts Float64 s>=.5. Its transported Float32 display copy
        # can round a just-below-threshold value to exactly .5, so these are
        # independent valid bounds, exact when equalHalf is zero.
        assert message["aboveHalf"] <= message["active"] <= message["aboveHalf"] + message["equalHalf"]
    return {key: message.get(key) for key in ("requestId", "generation", "initialResidual", "residual", "iterations", "converged", "active", "aboveHalf", "equalHalf")}


def check_cards(snapshot, require_live=True):
    state, cards = snapshot["states"], snapshot["cards"]
    assert state["generation"] == snapshot["generation"] and state["paused"] == snapshot["paused"]
    near(state["time"], snapshot["clock"], 0); near(state["hunger"], snapshot["hunger"], 0)
    assert 0 <= state["hunger"] <= 1 and state["neurons"] == snapshot["brain"]["n"] == 150802
    near(float(cards["hunger"].removesuffix("%")), 100 * state["hunger"], .51)
    neural = state["neural"]
    if require_live:
        assert neural["available"] and neural["state"] in ("live", "paused"), neural
    if neural["available"]:
        sample = neural["sample"]
        message = next(m for m in reversed(snapshot["messages"]) if m["type"] == "control" and
                       m["requestId"] == sample["requestId"] and m["generation"] == sample["generation"])
        measurement = check_worker(message)
        for key in ("initialResidual", "residual", "iterations", "tolerance", "converged"):
            assert sample[key] == message[key], (key, sample[key], message[key])
        assert sample["neuronCount"] == 150802 and sample["active"] == message.get("active")
        near(neural["raw"], sample["initialResidual"], 0)
        assert 0 <= neural["ageSeconds"] < neural["maxAgeSeconds"]
        near(neural["ageSeconds"], state["time"] - sample["observationTime"])
        assert state["settlingSteps"] == sample["iterations"] == int(cards["steps"])
        for key, name in (("initialResidual", "initial"), ("residual", "residual")):
            value = sample[key]
            near(float(cards[name]), value, max(abs(value) * .00501, 1e-14))
        if sample["converged"]:
            near(state["activityFraction"], sample["active"] / state["neurons"], 0)
            near(float(cards["activity"].removesuffix("%")), 100 * state["activityFraction"], .05001)
        else:
            assert state["activityFraction"] is None and cards["activity"] == "—"
        expected_status = ("RECORDED · " if neural["state"] == "paused" else "") + ("SETTLED" if sample["converged"] else "UNQUALIFIED")
        assert cards["status"] == expected_status
    else:
        measurement = None
        assert state["activityFraction"] is None and state["settlingSteps"] is None
        assert all(cards[key] == "—" for key in ("activity", "steps", "initial", "residual"))
    if state["learningUpdates"] is not None:
        accepted = [m for m in snapshot["messages"] if m["type"] == "lesson" and m.get("accepted") is True and m.get("generation") == snapshot["generation"]]
        expected_updates = max((m["updates"] for m in accepted), default=0)
        assert state["learningUpdates"] == expected_updates == int(cards["learning"].replace(",", ""))
        assert state["changedConnections"] == snapshot["lessonStats"]["changed"]
    return measurement


def matched_snapshot(page, require_live=True, timeout=20):
    """Allow the documented 200ms HUD refresh without conflating old DOM/new data."""
    deadline = time.monotonic() + timeout
    while True:
        snapshot = page.evaluate(READ)
        try:
            check_cards(snapshot, require_live)
            return snapshot
        except (AssertionError, ValueError, StopIteration) as error:
            if time.monotonic() >= deadline:
                raise AssertionError(f"cards did not match a current measured snapshot: {error}") from error
            page.wait_for_timeout(100)


def check_impulse(event):
    before, after = event["before"], event["after"]
    assert before["clock"] == after["clock"], "event witness must precede any subsequent physics frame"
    assert any(before[key] != after[key] for key in ("v", "w", "q")), "no physical impulse was applied"
    assert not after["states"]["neural"]["available"] and after["states"]["neural"]["sample"] is None
    assert after["states"]["settlingSteps"] is None and after["states"]["activityFraction"] is None


def changed_senses(snapshot, old_id, new_id):
    def request(number):
        return next(r for r in snapshot["requests"] if r.get("requestId") == number and r.get("generation") == snapshot["generation"] and r["type"] == "assist:observe")
    old, new = request(old_id), request(new_id)
    changed = {key: [old["senses"].get(key), value] for key, value in new["senses"].items() if old["senses"].get(key) != value}
    assert changed, "physical perturbation did not reach a changed sensory observation"
    return {"oldRequestId": old_id, "newRequestId": new_id, "changed": changed,
            "oldRetina": old.get("retina"), "newRetina": new.get("retina")}


def run(url, screenshot, mobile_screenshot):
    frozen = source_hashes(ROOT, RUNNER)
    companion_path = ROOT / "receipts/settlement_trace_public_states.json"
    companion_bytes = companion_path.read_bytes()
    companion = json.loads(companion_bytes)
    assert companion["passed"] and companion["sources_unchanged"]
    for name, digest in companion["sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, ("stale full-worker companion", name)
    result = {"schema": "cadence.public-state-browser/1", "passed": False,
              "started": datetime.now(timezone.utc).isoformat(), "sha256": frozen,
              "checks": [], "page_errors": [], "protocol": {
                  "seed": 1, "viewport": [1440, 1000], "obsolete_queries": {"controller": "assisted-off", "card": "sit", "pose": "grooming"},
                  "motion_window_simulated_seconds": 2, "minimum_horizontal_displacement_m": .05,
                  "mobile_viewport": [430, 900],
                  "full_worker_arithmetic_companion": {"path": str(companion_path.relative_to(ROOT)), "sha256": hashlib.sha256(companion_bytes).hexdigest(), "body_sha256": companion["body_sha256"]},
                  "fault_injection": "Hold actual control response(s) before page admission, apply actual G gust, release the unchanged old response, then release real newer responses. No synthetic neural value, action or reward.",
                  "limits": "One bounded UI/wiring check. It does not establish trained navigation, biological mental states, convergence guarantees, or a positive learning outcome."}}
    parts = urlsplit(url); query = dict(parse_qsl(parts.query)); query.update(controller="assisted-off", card="sit", pose="grooming", seed="1", style="settling")
    run_url = urlunsplit((*parts[:3], urlencode(query), parts.fragment))
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        result["runtime"] = {"python": platform.python_version(), "browser": browser.version}
        binding = ServedSourceBinding(page.context, urlunsplit((*parts[:3], "", "")), frozen)
        page.on("pageerror", lambda error: result["page_errors"].append(str(error)))
        page.add_init_script(RECORD)
        try:
            page.goto(run_url, wait_until="domcontentloaded")
            page.wait_for_function("window.__app?.S.ready&&window.__app.S.learnReady&&typeof window.__app.states==='function'", timeout=90000)
            page.evaluate("window.__installPhysicalAfterWitness()")
            binding.wait_for_complete(page)
            page.wait_for_function("window.__app.states().neural.available&&window.__app.life().authority.navigation.active&&window.__app.life().search?.neuralChoice?.executed", timeout=90000)
            baseline = result["baseline"] = matched_snapshot(page)
            check_public(baseline); result["baseline_measurement"] = check_cards(baseline)
            result["checks"].append("ignored controller=assisted-off and card=sit/pose=grooming queries; only enabled NeuralLife runs and comparison controls/API are absent")
            page.locator('#about').click()
            page.wait_for_function("document.getElementById('card').classList.contains('open')")
            about = result["about_open"] = page.evaluate(READ); check_public(about)
            page.locator('#about').click()
            result["checks"].append("normal page and open About contain no assistance or disabled/comparison controller wording")
            page.wait_for_function("t=>window.__app.life().clock>=t+2", arg=baseline["clock"], timeout=90000)
            moved = result["motion"] = page.evaluate(READ)
            displacement = math.hypot(*(moved["p"][i] - baseline["p"][i] for i in (0, 1)))
            assert displacement >= .05, displacement
            assert moved["authority"]["navigation"]["framesApplied"] > baseline["authority"]["navigation"]["framesApplied"]
            assert moved["controlAccepted"] > baseline["controlAccepted"]
            result["motion_displacement_m"] = displacement
            result["checks"].append("actual executed neural navigation continues and moves the body at least5cm in a2simulated-second active-start window")
            page.evaluate("()=>{document.activeElement?.blur();window.__holdControls=true;}")
            page.wait_for_function("window.__heldControls.length>0", timeout=90000)
            held = result["held_before_gust"] = page.evaluate(READ)
            old_id = held["held"][0]["requestId"]
            page.keyboard.press('g')
            page.wait_for_function("window.__physicalEvents.at(-1)?.kind==='gust'&&window.__physicalEvents.at(-1).after", timeout=10000)
            after_gust = result["gust_invalidated"] = page.evaluate(READ)
            check_impulse(after_gust["physicalEvents"][-1])
            released = page.evaluate("window.__releaseOldControl()")
            assert released == {"requestId": old_id, "generation": held["generation"]}
            page.wait_for_function("!window.__app.states().neural.available&&document.getElementById('state-initial').textContent==='—'", timeout=10000)
            rejected = result["stale_reply_rejected"] = page.evaluate(READ)
            assert rejected["states"]["neural"]["sample"] is None
            assert rejected["controlAccepted"] == after_gust["controlAccepted"]
            check_cards(rejected, require_live=False)
            page.evaluate("window.__releaseRemainingControls()")
            page.wait_for_function("id=>window.__app.states().neural.available&&window.__app.states().neural.sample.requestId>id", arg=old_id, timeout=90000)
            fresh = result["fresh_after_gust"] = matched_snapshot(page)
            result["gust_measurement"] = check_cards(fresh)
            new_id = fresh["states"]["neural"]["sample"]["requestId"]
            result["gust_input_change"] = changed_senses(fresh, old_id, new_id)
            result["checks"].append("actual G impulse clears telemetry; unchanged delayed pre-gust response cannot refill it; new changed inputs supply measured initial/final residuals")
            page.wait_for_function("window.__app.life().search?.neuralChoice?.executed&&window.__app.life().authority.navigation.active", timeout=90000)
            before_nudge = result["before_nudge"] = matched_snapshot(page)
            page.locator('#nudge').click()
            page.wait_for_function("window.__physicalEvents.at(-1)?.kind==='nudge'&&window.__physicalEvents.at(-1).after", timeout=10000)
            nudged = result["nudge_invalidated"] = page.evaluate(READ)
            event = nudged["physicalEvents"][-1]; check_impulse(event)
            choice = event["before"]["choice"]
            assert choice and choice["executed"]
            reward = next(r for r in reversed(nudged["requests"]) if r["type"] == "assist:reward" and r.get("requestId") == choice["requestId"])
            assert reward["generation"] == choice["generation"] and reward["searchToken"] == choice["searchToken"]
            assert reward["neuralCredit"] is True and reward["done"] is True and reward["reward"] == -1
            page.wait_for_function("id=>window.__publicMessages.some(m=>m.type==='lesson'&&m.requestId===id)", arg=choice["requestId"], timeout=90000)
            page.wait_for_function("id=>window.__app.states().neural.available&&window.__app.states().neural.sample.requestId>id", arg=event["after"]["issuedRequestId"], timeout=90000)
            learned = result["after_nudge"] = matched_snapshot(page)
            lesson = next(m for m in learned["messages"] if m["type"] == "lesson" and m.get("requestId") == choice["requestId"])
            assert lesson["accepted"] is True and lesson["generation"] == choice["generation"] and lesson["searchToken"] == choice["searchToken"]
            assert learned["states"]["learningUpdates"] == before_nudge["states"]["learningUpdates"] + 1
            result["nudge_measurement"] = check_cards(learned)
            result["nudge_input_change"] = changed_senses(learned, before_nudge["states"]["neural"]["sample"]["requestId"], learned["states"]["neural"]["sample"]["requestId"])
            result["checks"].append("actual Nudge changes physical inputs; its matched executed choice receives one real worker learning update and the counter matches")
            page.locator('#pause').click(); page.wait_for_function("window.__app.S.paused")
            paused = result["paused"] = matched_snapshot(page)
            assert paused["states"]["neural"]["state"] == "paused" and not paused["authority"]["navigation"]["active"]
            page.locator('#gust').click(); page.locator('#nudge').click(); page.wait_for_timeout(400)
            still = result["paused_impulses_ignored"] = matched_snapshot(page)
            for key in ("p", "v", "q", "omega", "clock"):
                assert still[key] == paused[key], key
            assert still["states"] == paused["states"]
            check_public(still)
            if screenshot:
                page.screenshot(path=str(screenshot)); result["screenshot"] = {"path": str(screenshot), "sha256": hashlib.sha256(screenshot.read_bytes()).hexdigest()}
            page.set_viewport_size({"width": 430, "height": 900}); page.wait_for_timeout(400)
            mobile = result["mobile"] = page.evaluate(READ); check_public(mobile); check_cards(mobile)
            result["mobile_pause_badge_display"] = page.locator('#paused').evaluate("e=>getComputedStyle(e).display")
            assert result["mobile_pause_badge_display"] == "none", "duplicate paused badge obscures the mobile title"
            layout = page.evaluate("""()=>Object.fromEntries(['state-hunger','state-activity','state-steps','state-learning','pause','gust','nudge','forget'].map(id=>{
              const e=document.getElementById(id),r=e.getBoundingClientRect(),hit=document.elementFromPoint(r.x+r.width/2,r.y+r.height/2);
              return[id,{left:r.left,right:r.right,top:r.top,bottom:r.bottom,unobscured:!!hit&&(hit===e||e.contains(hit)),hit:hit?.id||hit?.tagName}];}))""")
            for name, bounds in layout.items():
                assert 0 <= bounds["left"] < bounds["right"] <= 430 and 0 <= bounds["top"] < bounds["bottom"] <= 900, (name, bounds)
                assert bounds["unobscured"], (name, bounds)
            result["mobile_layout"] = layout
            if mobile_screenshot:
                page.screenshot(path=str(mobile_screenshot)); result["mobile_screenshot"] = {"path": str(mobile_screenshot), "sha256": hashlib.sha256(mobile_screenshot.read_bytes()).hexdigest()}
            assert mobile["p"] == paused["p"] and mobile["clock"] == paused["clock"]
            page.locator('#about').click()
            page.wait_for_function("document.getElementById('card').classList.contains('open')&&getComputedStyle(document.getElementById('card')).display!=='none'")
            mobile_about = result["mobile_about"] = page.evaluate(READ); check_public(mobile_about)
            about_box = page.locator('#card').evaluate("""e=>{const r=e.getBoundingClientRect(),hit=document.elementFromPoint(r.x+r.width/2,r.y+r.height/2);
              return{left:r.left,right:r.right,top:r.top,bottom:r.bottom,unobscured:!!hit&&(hit===e||e.contains(hit))};}""")
            assert 0 <= about_box["left"] < about_box["right"] <= 430 and 0 <= about_box["top"] < about_box["bottom"] <= 900
            assert about_box["unobscured"]
            result["mobile_about_layout"] = about_box
            page.locator('#about').click(); page.wait_for_function("!document.getElementById('card').classList.contains('open')")
            page.set_viewport_size({"width": 1440, "height": 1000})
            result["checks"].append("pause retains labeled recorded metrics and rejects physical impulses; desktop/mobile cards and controls are unobscured, and mobile About opens/closes")
            reset = page.evaluate("()=>{document.getElementById('forget').click();return window.__app.states();}")
            assert reset["generation"] > paused["generation"] and reset["neural"]["sample"] is None
            assert reset["settlingSteps"] is None and reset["activityFraction"] is None and reset["learningUpdates"] is None
            result["reset"] = reset
            page.wait_for_function("window.__app.S.ready&&window.__app.S.learnReady", timeout=90000)
            page.locator('#pause').click()
            page.wait_for_function("window.__app.states().neural.available", timeout=90000)
            final = result["reset_recovered"] = matched_snapshot(page)
            assert final["generation"] == reset["generation"] and final["states"]["learningUpdates"] == 0
            check_public(final); check_cards(final)
            result["checks"].append("Reset clears old measurement/learning identities and new generation produces fresh measured states")
            assert not result["page_errors"], result["page_errors"]
            result["served_sources"] = binding.receipt()
            assert source_hashes(ROOT, RUNNER) == frozen, "source changed during browser run"
            result["passed"] = True
        except Exception as error:
            result["error"] = str(error); result["traceback"] = traceback.format_exc()
            try:
                result["failure_state"] = page.evaluate(READ)
            except Exception as failure:
                result["diagnostic_error"] = str(failure)
            try:
                result["served_sources"] = binding.receipt()
            except Exception as failure:
                result["binding_error"] = str(failure)
            result["sources_unchanged"] = source_hashes(ROOT, RUNNER) == frozen
        finally:
            browser.close()
    result["completed"] = datetime.now(timezone.utc).isoformat()
    result["body_sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8817/")
    parser.add_argument("--receipt", type=Path, required=True, help="New JSON receipt; .gz writes lossless gzip JSON")
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--mobile-screenshot", type=Path)
    args = parser.parse_args()
    for path in (args.receipt, args.screenshot, args.mobile_screenshot):
        if path and path.exists():
            parser.error(f"refusing to overwrite {path}")
    outcome = run(args.url, args.screenshot, args.mobile_screenshot)
    stream = gzip.open(args.receipt, "xt", encoding="utf-8") if args.receipt.suffix == ".gz" else args.receipt.open("x")
    with stream as output:
        json.dump(outcome, output, indent=2, allow_nan=False); output.write("\n")
    print(json.dumps({key: outcome.get(key) for key in ("passed", "checks", "error", "body_sha256")}, indent=2))
    raise SystemExit(0 if outcome["passed"] else 1)
