#!/usr/bin/env python3
"""Bind the real settlement inspector to actual worker traces and retinal inputs.

The separate settlement-trace-worker.mjs assay checks full-state arithmetic and
trace/diagnostic custody. This bounded browser check checks request identity,
rendered replay values, real CSR edges, exact captured pixels and independently
encoded receptor levels, plus paused pixel-contrast/no-actuation behavior.
It never injects a neural state, action, reward, or replacement sensory frame.
"""
from __future__ import annotations

import argparse
from array import array
import base64
import gzip
import hashlib
import json
import math
import platform
import re
import struct
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from playwright.sync_api import sync_playwright
from browser_source_binding import ServedSourceBinding, source_hashes

ROOT = Path(__file__).resolve().parents[1]
RUNNER = "tools/check_settlement_view.py"
RECORD = """(() => {
 const copy=x=>ArrayBuffer.isView(x)?Array.from(x):Array.isArray(x)?x.map(copy):
   x&&typeof x==='object'?Object.fromEntries(Object.entries(x).map(([k,v])=>[k,copy(v)])):x;
 window.__settlementRequests=[];window.__settlementMessages=[];
 window.__dropDiagnostic=false;window.__droppedDiagnostics=0;window.__captureWatchdog=false;window.__watchdog=null;
 const timeout=window.setTimeout;
 window.setTimeout=function(fn,delay,...args){const id=timeout.call(this,fn,delay,...args);
   if(window.__captureWatchdog&&delay===30000)window.__watchdog={id,delay,invoke:()=>fn(...args)};return id;};
 const send=Worker.prototype.postMessage;
 Worker.prototype.postMessage=function(m,...rest){
   if(m?.trace||m?.type==='inspect:retina'){
     window.__settlementRequests.push(copy({type:m.type,requestId:m.requestId,generation:m.generation,
       senses:m.senses,retina:m.retina,steps:m.steps,tolerance:m.tolerance,trace:m.trace,attention:m.attention}));
     if(window.__settlementRequests.length>64)window.__settlementRequests.shift();}
   return send.call(this,m,...rest);};
 const Native=window.Worker;
 window.Worker=new Proxy(Native,{construct(T,args){const w=new T(...args);
   w.addEventListener('message',e=>{const m=e.data;
     if(m.trace||m.type==='retina_diagnostic'){
       const result={type:m.type,kind:m.kind,requestId:m.requestId,generation:m.generation,
         converged:m.converged,iterations:m.iterations,residual:m.residual,tolerance:m.tolerance,
         reason:m.reason,trace:m.trace,diagnosticOnly:m.diagnosticOnly,comparable:m.comparable,
         stateRestored:m.stateRestored,nonreceptor:m.nonreceptor,deltaReadouts:m.deltaReadouts,
         actual:m.actual?{solve:m.actual.solve,trace:m.actual.trace,readouts:m.actual.readouts}:undefined,
         black:m.black?{solve:m.black.solve,trace:m.black.trace,readouts:m.black.readouts}:undefined,
         topLevelControlFields:['s','readouts','decision','targetSelection'].filter(k=>k in m)};
       window.__settlementMessages.push(copy(result));if(window.__settlementMessages.length>32)window.__settlementMessages.shift();
       if(m.type==='retina_diagnostic'&&window.__dropDiagnostic){window.__droppedDiagnostics++;e.stopImmediatePropagation();}
     }});return w;}});
})();"""
READ = """() => {const a=window.__app,L=a.life(),s=a.settlement(),root=document.getElementById('settlement-view');
 return {wall:performance.now(),clock:L.clock,generation:a.S.generation,paused:a.S.paused,
   p:a.flight().p.slice(),brain:a.brainInfo(),pilot:a.S.pilot,settlement:s,
   authority:L.authority,controls:{...L.controls},lessonStats:a.S.lessonStats,
   controlsAccepted:a.S.controlAccepted,decisions:a.S.decisions,
   brainReady:a.S.brainReady,learnReady:a.S.learnReady,
   heading:root.querySelector('h3')?.textContent,
   text:Object.fromEntries(['status','source','frame','global','local','equation','graphnote','retinainfo',
     'contraststatus','contrastbrief','downstream','localbrief','graphlegend'].map(k=>[k,root.querySelector(`[data-sv=${k}]`)?.textContent||''])),
   requests:window.__settlementRequests,messages:window.__settlementMessages};}"""


def near(a, b, tolerance=1e-11):
    assert type(a) in (int, float) and type(b) in (int, float)
    assert math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tolerance, (a, b)


def qualified(solve):
    assert solve["converged"] is True
    residual, tolerance = solve["residual"], solve["tolerance"]
    assert type(residual) in (int, float) and math.isfinite(residual) and 0 <= residual <= tolerance <= 1e-6
    assert type(solve["iterations"]) is int and 0 <= solve["iterations"] <= 1024


def float32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def encode_independently(frame, members):
    ids = sorted(members)
    # JS Math.round for positive numbers, explicitly avoiding Python's ties-to-even.
    rows = max(1, math.floor(math.sqrt(len(ids) / 2) + .5))
    short, extra = divmod(len(ids), rows)
    width, height, rgba = frame["width"], frame["height"], frame["rgba"]
    def lum(x, y):
        k = 4 * (y * width + x)
        return (.2126 * rgba[k] + .7152 * rgba[k + 1] + .0722 * rgba[k + 2]) / 255
    levels = []
    for row in range(rows):
        columns = short + (row < extra)
        for col in range(columns):
            x = (col + .5) / columns * (width - 1)
            v = (row + .5) / rows
            y = (1 - v if frame["origin"] == "bottom-left" else v) * (height - 1)
            x0, y0 = math.floor(x), math.floor(y)
            x1, y1 = min(width - 1, x0 + 1), min(height - 1, y0 + 1)
            dx, dy = x - x0, y - y0
            levels.append(float32((1 - dy) * ((1 - dx) * lum(x0, y0) + dx * lum(x1, y0))
                                  + dy * ((1 - dx) * lum(x0, y1) + dx * lum(x1, y1))))
    return ids, levels


def load_graph():
    payload = json.loads((ROOT / "web/data/brain_full.json").read_text())
    def decode(name, code):
        values = array(code); values.frombytes(base64.b64decode(payload["arrays"][name], validate=True))
        if sys.byteorder != "little":
            values.byteswap()
        return values
    return {"n": payload["n"], "row": decode("row_ptr", "i"), "pre": decode("pre", "i"),
            "weights": decode("weight", "d"), "receptors": payload["populations"]["photoreceptor"]}


def matching_trace(snapshot):
    view = snapshot["settlement"]["view"]
    identity = view["identity"]
    message = next(m for m in reversed(snapshot["messages"])
                   if m["requestId"] == identity["requestId"] and m["generation"] == identity["generation"])
    branch = view.get("diagnosticBranch")
    trace = message[branch]["trace"] if branch else message["trace"]
    request = next(r for r in reversed(snapshot["requests"])
                   if r["requestId"] == identity["requestId"] and r["generation"] == identity["generation"])
    return view, trace, request, message


def verify_replay(snapshot, graph):
    view, trace, request, message = matching_trace(snapshot)
    assert view["replay"] is True and view["n"] == graph["n"] == 150802
    assert view["identity"]["generation"] == snapshot["generation"] == trace["identity"]["generation"]
    assert "Recorded settlement replay" == snapshot["heading"]
    index = view["frameIndex"]
    f = trace["frames"][index]
    assert view["iteration"] == f["iteration"] and view["frameCount"] == len(trace["frames"])
    near(view["globalResidual"], f["residual"]); near(view["localMaxDefect"], f["localMaxDefect"])
    assert view["settledCount"] == f["settledCount"] and 0 <= f["settledCount"] <= graph["n"]
    assert f["residual"] + 1e-12 >= f["localMaxDefect"]
    k = next(i for i, p in enumerate(trace["patches"]) if p["id"] == view["patch"]["id"])
    patch = trace["patches"][k]
    for name in ("potential", "activity", "drive", "bias", "synapticInput", "defect", "deltaV"):
        near(view["patch"][name], f[name][k])
    near(f["defect"][k], f["synapticInput"][k] + (f["drive"][k] + f["bias"][k]) - f["potential"][k])
    if f["iteration"]:
        near(view["patch"]["previousDefect"], f["previousDefect"][k])
        expected = f["previousPotential"][k] + trace["dt"] * f["previousDefect"][k]
        near(f["potential"][k], expected)
        near(f["deltaV"][k], expected - f["previousPotential"][k])
    assert view["patch"]["incomingCount"] == graph["row"][patch["id"] + 1] - graph["row"][patch["id"]]
    assert view["patch"]["displayedIncomingCount"] <= view["patch"]["capturedIncomingCount"] <= view["patch"]["incomingCount"]
    assert all(snapshot["lessonStats"][key] == 0 for key in ("changed", "meanAbsChange", "maxAbsChange")), "graph-edge comparison assumes this fresh unlearned browser episode"
    for edge in trace["edges"]:
        assert edge["pre"] == graph["pre"][edge["id"]]
        assert graph["row"][edge["post"]] <= edge["id"] < graph["row"][edge["post"] + 1]
        near(edge["weight"], graph["weights"][edge["id"]], 0)
    expected_shared = [p["id"] for p in trace["sharedInputs"] if patch["id"] in p["posts"]]
    assert view["patch"]["sharedInputIds"] == expected_shared
    for key, expected in (("global", f["residual"]), ("local", f["localMaxDefect"])):
        displayed = float(snapshot["text"][key])
        near(displayed, expected, max(abs(expected) * .000501, .00005001 if abs(expected) >= .001 else 1e-14))
    assert "all" in snapshot["text"]["source"] or "150,802" in snapshot["text"]["source"]
    assert "not separate copies" in snapshot["text"]["graphnote"]
    retinal = trace["retinal"]
    source_frame = request["retina"]
    expected_frame = dict(source_frame)
    if view.get("diagnosticBranch") == "black":
        expected_frame["rgba"] = [v if i % 4 == 3 else 0 for i, v in enumerate(source_frame["rgba"])]
    assert retinal["frame"] == expected_frame
    ids, levels = encode_independently(expected_frame, graph["receptors"])
    assert retinal["indices"] == ids and retinal["levels"] == levels
    assert len(ids) == 1831
    raw_hash = 2166136261
    for value in expected_frame["rgba"]:
        raw_hash = ((raw_hash ^ value) * 16777619) & 0xffffffff
    assert view["retinal"]["rgbaFNV1a32"] == f"{raw_hash:08x}"
    assert view["retinal"]["firstSourcePixel"] == expected_frame["rgba"][:4]
    chain = view["retinal"].get("receptorChain")
    if chain:
        near(chain["level"], levels[ids.index(chain["id"])] , 0)
        assert chain["selectedPatchId"] == patch["id"] and chain["mapping"] == "supplied-retained-index-grid/v1"
        receptor_index = next((i for i, p in enumerate(trace["patches"]) if p["id"] == chain["id"]), None)
        if receptor_index is not None:
            near(chain["activity"], f["activity"][receptor_index]); near(chain["drive"], f["drive"][receptor_index])
        else:
            edge_index = next(i for i, e in enumerate(trace["edges"]) if e["pre"] == chain["id"] and e["post"] == patch["id"])
            near(chain["activity"], f["edgeActivity"][edge_index]); assert chain["drive"] is None
        assert "encoded luminance" in snapshot["text"]["retinainfo"] and "→ activity" in snapshot["text"]["retinainfo"]
    return {"identity": view["identity"], "iteration": f["iteration"], "frameIndex": index,
            "globalResidual": f["residual"], "localMaxDefect": f["localMaxDefect"], "settledCount": f["settledCount"],
            "selectedPatch": view["patch"], "receptors": len(ids),
            "rgba_sha256": hashlib.sha256(bytes(expected_frame["rgba"])).hexdigest()}, expected_frame


def check_preview(page, frame):
    pixels = page.locator('[data-sv="retina"]').evaluate("""c=>({width:c.width,height:c.height,
      rgba:Array.from(c.getContext('2d').getImageData(0,0,c.width,c.height).data)})""")
    width, height = pixels["width"], pixels["height"]
    scale = min(width / frame["width"], height / frame["height"])
    ox, oy = (width - frame["width"] * scale) / 2, (height - frame["height"] * scale) / 2
    chain = page.evaluate("window.__app.settlement().view.retinal.receptorChain")
    highlight = chain and chain.get("previewHighlight")
    bounds = highlight["bounds"] if highlight else None
    mismatches, checked = 0, 0
    for y in range(frame["height"]):
        source_y = frame["height"] - 1 - y if frame["origin"] == "bottom-left" else y
        for x in range(frame["width"]):
            px, py = math.floor(ox + (x + .5) * scale), math.floor(oy + (y + .5) * scale)
            if bounds and bounds["x0"] <= px <= bounds["x1"] and bounds["y0"] <= py <= bounds["y1"]:
                continue
            target = 4 * (py * width + px)
            source = 4 * (source_y * frame["width"] + x)
            if pixels["rgba"][target:target + 4] != frame["rgba"][source:source + 4]:
                mismatches += 1
            checked += 1
    assert mismatches == 0, f"{mismatches} source pixels do not match nearest-neighbor preview centers"
    assert checked >= .9 * frame["width"] * frame["height"]
    return {"checkedPixelCenters": checked, "mismatches": mismatches, "excludedViewerHighlight": highlight,
            "preview_sha256": hashlib.sha256(bytes(pixels["rgba"])).hexdigest()}


def select_frame(page, index):
    page.locator('[data-sv="scrub"]').evaluate("""(e,i)=>{e.value=String(i);e.dispatchEvent(new Event('input',{bubbles:true}));}""", index)
    page.wait_for_function("""i=>{const v=window.__app.settlement().view;
      return v.frameIndex===i&&!v.playing&&document.querySelector('[data-sv=frame]').textContent.includes(`sample ${i+1}/`);}""", arg=index, timeout=15000)


def neutral(snapshot):
    assert snapshot["paused"]
    n = snapshot["authority"]["navigation"]
    assert n["active"] is False and all(x == 0 for x in n["command"].values())
    assert all(x == 0 for x in snapshot["authority"]["neuralTrim"].values())


MOVE_FRUIT = """()=>{const a=window.__app,L=a.life();
 const keys=['lastHand','loom','handDist','smellC','smelled','smellDrive','onFruit','onSugar','onWhich','fruitC'];
 const cache=()=>JSON.stringify(Object.fromEntries(keys.map(k=>[k,L[k]])));
 const sample=()=>Object.assign(Object.create(Object.getPrototypeOf(L)),L).senses(null,0);
 const old=L.fruits.banana.pos.slice(),before=sample(),cached=cache();
 a.room.placeFruit('banana',[a.room.table.x0+.1,a.room.table.y0+.08]);
 const expected=sample();return{oldPosition:old,newPosition:L.fruits.banana.pos.slice(),before,expected,
   cacheBefore:cached,cacheAfter:cache(),position:a.flight().p.slice(),clock:L.clock};}"""


def run(url, screenshot=None, difference_screenshot=None, expanded_screenshot=None):
    frozen, graph = source_hashes(ROOT, RUNNER), load_graph()
    result = {"schema": "cadence.settlement-browser-check/1", "passed": False,
              "started": datetime.now(timezone.utc).isoformat(), "sha256": frozen, "checks": [], "page_errors": [],
              "protocol": {"seed": 1, "full_views": True, "pixel_contrast_minimum_maximum_difference": 1e-8,
                           "worker_arithmetic_companion": "receipts/settlement_trace_worker.json",
                           "watchdog_fault": "After normal resume, drop one diagnostic reply before the page handler, then explicitly invoke its captured 30000ms timeout callback. This checks the actual failure transition, not elapsed timeout performance.",
                           "limits": "One bounded live browser episode. Global full-state arithmetic and telemetry parity are independently checked by the companion worker assay. No equilibrium uniqueness, biological vision, navigation or learning claim from this viewer."}}
    parts = urlsplit(url); query = dict(parse_qsl(parts.query)); query.update(controller="assisted", seed="1", style="settling")
    run_url = urlunsplit((*parts[:3], urlencode(query), parts.fragment))
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        result["runtime"] = {"python": platform.python_version(), "browser": browser.version}
        binding = ServedSourceBinding(page.context, urlunsplit((*parts[:3], "", "")), frozen)
        page.on("pageerror", lambda e: result["page_errors"].append(str(e))); page.add_init_script(RECORD)
        try:
            page.goto(run_url, wait_until="domcontentloaded")
            page.wait_for_function("window.__app?.S.ready&&window.__app.S.learnReady&&window.__app.settlement().view.hasTrace", timeout=90000)
            binding.wait_for_complete(page)
            page.keyboard.press("Space")
            page.wait_for_function("window.__app.S.paused", timeout=10000)
            select_frame(page, 0)
            initial = result["initial_replay"] = page.evaluate(READ)
            neutral(initial)
            assert initial["settlement"]["mode"] == "settling"
            check, pixels = verify_replay(initial, graph); result["initial_check"] = check
            result["pixel_preview"] = check_preview(page, pixels)
            # This is the real Expand control. Check clipped visibility as well
            # as viewport bounds: a large canvas can otherwise pass while most
            # of it is hidden inside an overflow container or behind another pane.
            assert initial["settlement"]["view"]["retinal"]["receptorChain"] is not None
            page.locator('#brain-big').click()
            page.wait_for_function("document.getElementById('brain').classList.contains('big')&&document.getElementById('brain').getBoundingClientRect().width>=1390")
            page.locator('#settlement-view').evaluate("e=>{e.scrollTop=0;}")
            expanded = page.evaluate("""()=>{
              const inspector=document.getElementById('settlement-view'),ir=inspector.getBoundingClientRect();
              const rect=r=>({left:r.left,top:r.top,right:r.right,bottom:r.bottom,width:r.width,height:r.height});
              const nodes=Object.fromEntries(['retina','graph','localbrief'].map(name=>{
                const e=document.querySelector(`[data-sv=${name}]`),r=e.getBoundingClientRect();
                const points=[[.02,.02],[.98,.02],[.5,.5],[.02,.98],[.98,.98]].map(([u,v])=>{
                  const x=r.left+r.width*u,y=r.top+r.height*v,hit=document.elementFromPoint(x,y);
                  return{x,y,unobscured:!!hit&&(hit===e||e.contains(hit)),hit:hit?{tag:hit.tagName,id:hit.id,sv:hit.dataset.sv||null}:null};});
                return[name,{bounds:rect(r),points}];}));
              const a=window.__app;return{inspector:rect(ir),scrollTop:inspector.scrollTop,
                scrollHeight:inspector.scrollHeight,nodes,legendDisplay:getComputedStyle(document.getElementById('legend')).display,
                clock:a.life().clock,p:a.flight().p.slice(),view:a.settlement().view};}""")
            assert expanded["scrollTop"] == 0 and expanded["legendDisplay"] == "none"
            ir = expanded["inspector"]
            for name, node in expanded["nodes"].items():
                box = node["bounds"]
                assert box["width"] > 0 and box["height"] > 0
                assert max(0, ir["left"]) <= box["left"] and box["right"] <= min(1440, ir["right"]), (name, box, ir)
                assert max(0, ir["top"]) <= box["top"] and box["bottom"] <= min(1000, ir["bottom"]), (name, box, ir)
                assert all(point["unobscured"] for point in node["points"]), (name, node["points"])
            assert expanded["p"] == initial["p"] and expanded["clock"] == initial["clock"]
            assert expanded["view"]["retinal"]["receptorChain"] == initial["settlement"]["view"]["retinal"]["receptorChain"]
            result["expanded_layout"] = expanded
            if expanded_screenshot:
                page.screenshot(path=str(expanded_screenshot))
                result["expanded_screenshot"] = {"path": str(expanded_screenshot), "sha256": hashlib.sha256(expanded_screenshot.read_bytes()).hexdigest()}
            page.locator('#brain-big').click()
            page.wait_for_function("!document.getElementById('brain').classList.contains('big')&&document.getElementById('brain').getBoundingClientRect().width<=561")
            result["checks"].append("Expand shows captured pixels, whole graph and local values inside the inspector without overlay obstruction; anatomy legend remains hidden")
            trace = matching_trace(initial)[1]
            chosen = max(range(len(trace["patches"])), key=lambda k: trace["patches"][k]["sampledIncomingCount"])
            page.locator('[data-sv="patch"]').select_option(str(chosen))
            page.wait_for_function("id=>window.__app.settlement().view.patch.id===id", arg=trace["patches"][chosen]["id"])
            observations = []
            for index in sorted({len(trace["frames"]) // 2, len(trace["frames"]) - 1}):
                select_frame(page, index); snap = page.evaluate(READ); checked, _ = verify_replay(snap, graph); observations.append(checked)
                assert snap["p"] == initial["p"] and snap["clock"] == initial["clock"]
                neutral(snap)
            result["replay_checks"] = observations
            if screenshot:
                page.locator('#settlement-view').evaluate("e=>{e.scrollTop=0;}")
                result["first_viewport"] = {}
                for name in ("retina", "graph"):
                    box = page.locator(f'[data-sv="{name}"]').bounding_box()
                    assert box and box["y"] >= 0 and box["y"] + box["height"] <= 1000, (name, box)
                    result["first_viewport"][name] = box
                page.screenshot(path=str(screenshot))
                result["replay_screenshot"] = {"path": str(screenshot), "sha256": hashlib.sha256(screenshot.read_bytes()).hexdigest()}
            result["checks"].append("matched actual request/trace, exact oriented preview, independent1831-receptor encoding, actual CSR edges, global/local replay and prior-step Euler values")
            # UI-only replay and tab selection must never resume physics or add
            # admitted controls/learning while the actual body is paused.
            before = page.evaluate(READ)
            page.evaluate("window.__app.setBrainView('brain')")
            page.evaluate("window.__app.setBrainView('settling')")
            after = page.evaluate(READ)
            assert after["p"] == before["p"] and after["clock"] == before["clock"]
            assert after["controlsAccepted"] == before["controlsAccepted"] and after["lessonStats"] == before["lessonStats"]
            neutral(after)
            result["checks"].append("scrubbing and inspector tab changes have no control/physics/learning authority")
            moved = result["paused_fruit_move"] = page.evaluate(MOVE_FRUIT)
            assert moved["cacheBefore"] == moved["cacheAfter"]
            assert moved["oldPosition"] != moved["newPosition"]
            assert abs(moved["before"]["orn:decaying_fruit:left"] - moved["expected"]["orn:decaying_fruit:left"]) > 1e-10
            page.locator('[data-sv="contrast"]').click()
            baseline = result["diagnostic_baseline"] = page.evaluate(READ); neutral(baseline)
            page.wait_for_function("""()=>{const s=window.__app.settlement();return !s.pendingDiagnostic&&s.diagnostic&&s.view.contrast.available;}""", timeout=90000)
            diagnostic = result["diagnostic_complete"] = page.evaluate(READ); neutral(diagnostic)
            assert diagnostic["clock"] == baseline["clock"] and diagnostic["p"] == baseline["p"]
            assert diagnostic["controlsAccepted"] == baseline["controlsAccepted"] and diagnostic["lessonStats"] == baseline["lessonStats"]
            d = diagnostic["settlement"]["diagnostic"]
            assert d["diagnosticOnly"] is True and d["stateRestored"] is True and d["comparable"] is True
            assert d["requestId"] == d["identity"]["requestId"] and d["generation"] == diagnostic["generation"]
            message = next(m for m in diagnostic["messages"] if m["requestId"] == d["requestId"] and m["type"] == "retina_diagnostic")
            request = next(r for r in diagnostic["requests"] if r["requestId"] == d["requestId"] and r["type"] == "inspect:retina")
            assert request["senses"] == moved["expected"], "diagnostic must recompute the changed scene's senses while paused"
            cache_after = page.evaluate("""()=>{const L=window.__app.life();return JSON.stringify(Object.fromEntries(
              ['lastHand','loom','handDist','smellC','smelled','smellDrive','onFruit','onSugar','onWhich','fruitC'].map(k=>[k,L[k]])));}""")
            assert cache_after == moved["cacheBefore"], "diagnostic encoder changed live controller caches"
            assert message["kind"] == "diagnostic" and message["topLevelControlFields"] == []
            qualified(d["actual"]["solve"]); qualified(d["black"]["solve"])
            assert d["nonreceptor"]["count"] == 150802 - 1831
            assert 0 <= d["nonreceptor"]["changedCount"] <= d["nonreceptor"]["count"]
            assert d["nonreceptor"]["maxAbsDelta"] > 1e-8
            for name, delta in d["deltaReadouts"].items():
                near(delta, d["actual"]["readouts"][name] - d["black"]["readouts"][name])
            branch_checks = []
            for branch in ("black", "actual"):
                page.locator(f'[data-sv="{branch}"]').click()
                page.wait_for_function("b=>window.__app.settlement().view.diagnosticBranch===b", arg=branch)
                count = page.evaluate("window.__app.settlement().view.frameCount")
                select_frame(page, count - 1)
                snap = page.evaluate(READ); checked, pixels = verify_replay(snap, graph)
                assert snap["settlement"]["view"]["policyAccepted"] is False
                checked["preview"] = check_preview(page, pixels); branch_checks.append(checked)
            result["paired_checks"] = branch_checks
            page.locator('[data-sv="difference"]').click()
            page.wait_for_function("window.__app.settlement().view.graphMode==='final-pixel-difference'&&document.querySelector('[data-sv=graphlegend]').textContent.includes('Log brightness')")
            difference = result["final_difference_view"] = page.evaluate(READ)
            patch_id = difference["settlement"]["view"]["patch"]["id"]
            def final_activity(branch):
                trace = d[branch]["trace"]
                index = next(i for i, patch in enumerate(trace["patches"]) if patch["id"] == patch_id)
                return trace["frames"][-1]["activity"][index]
            near(difference["settlement"]["view"]["patch"]["finalPixelActivityDifference"], final_activity("actual") - final_activity("black"))
            result["checks"].append("fruit moved while paused changes the diagnostic's freshly encoded senses without mutating live sensor caches")
            result["checks"].append("paused actual-pixel/black comparison qualifies both branches, displays matched captures/readout deltas, reports restoration and supplies no control admission")
            if difference_screenshot:
                page.locator('#settlement-view').evaluate("e=>{e.scrollTop=0;}")
                page.screenshot(path=str(difference_screenshot))
                result["difference_screenshot"] = {"path": str(difference_screenshot), "sha256": hashlib.sha256(difference_screenshot.read_bytes()).hexdigest()}
            page.keyboard.press("Space")
            page.wait_for_function("""t=>{const a=window.__app;return !a.S.paused&&a.life().clock>=t+.1
              &&a.life().authority.navigation.active;}""", arg=diagnostic["clock"], timeout=90000)
            resumed = result["resumed"] = page.evaluate(READ)
            assert not resumed["paused"] and resumed["authority"]["navigation"]["active"]
            result["checks"].append("normal qualified neural control resumes after isolated diagnostic")
            page.wait_for_function("window.__app.settlement().view.queued", timeout=30000)
            page.locator('[data-sv="follow"]').click()
            page.wait_for_function("id=>{const v=window.__app.settlement().view;return v.identity.requestId!==id&&v.graphMode==='replay-activity';}", arg=d["requestId"], timeout=15000)
            followed = result["followed_live_trace"] = page.evaluate(READ)
            contrast_id = followed["settlement"]["view"]["contrast"]["identity"]
            assert contrast_id["requestId"] == d["requestId"] and contrast_id["generation"] == d["generation"]
            assert f'Recorded pixel pair {d["generation"]}/{d["requestId"]}' in followed["text"]["contrastbrief"]
            result["checks"].append("Follow latest restores activity replay; the retained pixel contrast identifies its separate historical request")
            page.evaluate("""()=>{window.__dropDiagnostic=true;window.__captureWatchdog=true;
              window.__watchdog=null;window.__app.compareRetinalPixels();}""")
            page.wait_for_function("window.__droppedDiagnostics===1&&window.__app.settlement().pendingDiagnostic", timeout=90000)
            lost = result["dropped_reply"] = page.evaluate(READ); neutral(lost)
            callback = page.evaluate("""()=>{const w=window.__watchdog;if(!w)throw Error('no timeout captured');
              const pending=window.__app.settlement().pendingDiagnostic;w.invoke();
              window.__captureWatchdog=false;window.__dropDiagnostic=false;return{delay:w.delay,pending};}""")
            assert callback["delay"] == 30000
            page.wait_for_function("!window.__app.settlement().pendingDiagnostic&&!window.__app.S.brainReady", timeout=10000)
            failed = result["watchdog_failure"] = page.evaluate(READ); neutral(failed)
            assert failed["p"] == lost["p"] and failed["clock"] == lost["clock"]
            assert not failed["brainReady"] and not failed["learnReady"]
            assert failed["controlsAccepted"] == lost["controlsAccepted"]
            result["watchdog_injection"] = callback
            result["checks"].append("dropped diagnostic reply plus declared deadline-callback injection clears pending request and disables worker without actuation")
            reset = page.evaluate("""()=>{const a=window.__app;document.getElementById('forget').click();
              return {generation:a.S.generation,inspection:a.settlement()};}""")
            assert reset["generation"] > diagnostic["generation"]
            assert not reset["inspection"]["view"]["hasTrace"] and reset["inspection"]["diagnostic"] is None and reset["inspection"]["pendingDiagnostic"] is None
            result["reset"] = reset; result["checks"].append("Reset clears old trace, diagnostic and pending request identity")
            assert not result["page_errors"], result["page_errors"]
            result["served_sources"] = binding.receipt()
            assert source_hashes(ROOT, RUNNER) == frozen, "source changed during browser run"
            result["passed"] = True
        except Exception as error:
            result["error"] = str(error); result["traceback"] = traceback.format_exc()
            try:
                result["failure_state"] = page.evaluate(READ)
            except Exception as error2:
                result["diagnostic_error"] = str(error2)
            try:
                result["served_sources"] = binding.receipt()
            except Exception as error3:
                result["binding_error"] = str(error3)
            result["sources_unchanged"] = source_hashes(ROOT, RUNNER) == frozen
        finally:
            browser.close()
    result["completed"] = datetime.now(timezone.utc).isoformat()
    result["body_sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8817/")
    ap.add_argument("--receipt", type=Path, help="New JSON receipt; a .gz suffix writes the same JSON with lossless gzip compression")
    ap.add_argument("--screenshot", type=Path)
    ap.add_argument("--difference-screenshot", type=Path)
    ap.add_argument("--expanded-screenshot", type=Path)
    args = ap.parse_args()
    for path in (args.receipt, args.screenshot, args.difference_screenshot, args.expanded_screenshot):
        if path and path.exists():
            ap.error(f"refusing to overwrite {path}")
    outcome = run(args.url, args.screenshot, args.difference_screenshot, args.expanded_screenshot)
    if args.receipt:
        receipt_out = gzip.open(args.receipt, "xt", encoding="utf-8") if args.receipt.suffix == ".gz" else args.receipt.open("x")
        with receipt_out as out:
            json.dump(outcome, out, indent=2, allow_nan=False); out.write("\n")
    print(json.dumps({k: outcome.get(k) for k in ("passed", "checks", "error", "body_sha256")}, indent=2))
    raise SystemExit(0 if outcome["passed"] else 1)
