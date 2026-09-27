#!/usr/bin/env python3
"""Bounded real-page audit of measured trails and held-command projections.

This is a display/authority/isolation check, not a navigation competence test.
It observes the real full-graph worker, never injects neural decisions or routes,
and adds a temporary bright annotation solely during synchronous retinal-isolation
captures. For that paired test, room render-time animation hooks are suspended to
hold environmental state fixed; a repeated baseline verifies that condition. An
unhidden copy provides a positive control. All hooks and temporary scene changes
are restored before the event loop resumes.
Source hashes freeze before launch; served bytes are checked; no output overwrites.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from playwright.sync_api import sync_playwright
from browser_source_binding import ServedSourceBinding, source_hashes

ROOT = Path(__file__).resolve().parents[1]
RUNNER = "tools/check_neural_paths.py"
READ = """() => {const a=window.__app,L=a.life(),f=a.flight();return {
 wall:performance.now(),time:a.S.simTime,clock:L.clock,generation:a.S.generation,
 brain:a.brainInfo(),pilot:a.S.pilot,source:L.source,paused:a.S.paused,
 p:f.p.slice(),v:f.v.slice(),heading:L.heading,paths:a.paths(),display:a.pathDisplay(),
 authority:L.authority,controls:{...L.controls},search:L.search,
 events:L.events,eventText:document.getElementById('events').innerText,
 resetButton:document.getElementById('forget').innerText,
 annotationsVisible:a.pathAnnotations.visible,
 rendered:Object.fromEntries(a.pathAnnotations.children.filter(o=>o.geometry).map(o=>[o.type,
   {visible:o.visible,positions:Array.from(o.geometry.getAttribute('position')?.array||[]),
    colors:Array.from(o.geometry.getAttribute('color')?.array||[])}])),
 panel:Object.fromEntries(['path-panel','path-status','path-command','path-choice','path-legend',
   'path-note','path-candidates','path-sampling','path-duration'].map(id=>{
   const e=document.getElementById(id),r=e.getBoundingClientRect(),s=getComputedStyle(e);
   return [id,{text:e.innerText,visible:s.display!=='none'&&s.visibility!=='hidden'&&r.width>0&&r.height>0,
     rect:{x:r.x,y:r.y,width:r.width,height:r.height}}];})),
 decisions:window.__pathDecisions};}"""
RECORD_DECISIONS = """(() => {window.__pathDecisions=[];const Native=window.Worker;
 window.Worker=new Proxy(Native,{construct(T,args){const w=new T(...args);
   w.addEventListener('message',e=>{const m=e.data;if(m.type==='assisted_decision'&&m.accepted)
     window.__pathDecisions.push(JSON.parse(JSON.stringify({requestId:m.requestId,generation:m.generation,
       searchToken:m.searchToken,targetSelection:m.targetSelection,decision:m.decision})));});return w;}});
})();"""


def near(a, b, tolerance=1e-10):
    assert type(a) in (float, int) and type(b) in (float, int)
    assert math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tolerance, (a, b)


def verify_projection(current, projection):
    if not projection:
        return
    assert len(projection) <= 256 and current["active"]
    assert projection[-1]["time"] <= min(current["deadline"], current["time"] + 1) + 1e-12
    for point in projection:
        dt = point["time"] - current["time"]
        assert dt >= 0
        omega, speed = current["command"]["yawRate"], current["command"]["forwardSpeed"]
        heading = current["heading"]
        # Independent closed-form circular arc, with straight-line limit.
        if abs(omega) > 1e-7:
            dx = speed / omega * (math.sin(heading + omega * dt) - math.sin(heading))
            dy = speed / omega * (math.cos(heading) - math.cos(heading + omega * dt))
        else:
            dx, dy = speed * dt * math.cos(heading), speed * dt * math.sin(heading)
        expected = [current["position"][0] + dx, current["position"][1] + dy,
                    current["position"][2] + current["command"]["verticalSpeed"] * dt]
        for a, b in zip(point["position"], expected):
            near(a, b, 1e-8)
        near(point["heading"], heading + omega * dt)


def verify_snapshot(snapshot, require_projection=False):
    paths = snapshot["paths"]
    history = paths["history"]
    current = history["current"]
    assert history["generation"] == snapshot["generation"] == current["generation"]
    near(current["time"], snapshot["clock"])
    for actual, recorded in zip(snapshot["p"], current["position"]):
        near(actual, recorded)
    near(current["heading"], snapshot["heading"])
    navigation = snapshot["authority"]["navigation"]
    assert current["active"] == navigation["active"]
    for key in ("forwardSpeed", "yawRate", "verticalSpeed"):
        near(current["command"][key], navigation["command"][key])
    points = history["points"]
    assert 0 < len(points) <= history["maxPoints"]
    assert all(p["generation"] == history["generation"] for p in points)
    assert all(a["time"] < b["time"] for a, b in zip(points, points[1:]))
    assert points[-1]["time"] <= current["time"]
    assert current["time"] - points[-1]["time"] <= history["interval"] + .001
    projection = paths["projection"]
    if require_projection:
        assert len(projection) > 1 and current["active"]
        assert current["deadline"] > current["time"]
    verify_projection(current, projection)
    display = snapshot["display"]["state"]
    displayed_history = display["history"]
    assert displayed_history["generation"] == history["generation"]
    assert 0 <= current["time"] - displayed_history["current"]["time"] <= .1
    assert displayed_history["current"]["active"] == current["active"]
    for point in displayed_history["points"]:
        assert point in points, "visible trail must use retained actual physical samples"
    current, points, projection = displayed_history["current"], displayed_history["points"], display["projection"]
    verify_projection(current, projection)
    measured = [*points]
    if measured[-1]["time"] != current["time"]:
        measured.append(current)
    vertices = [v for a, b in zip(measured, measured[1:]) for p in (a, b) for v in p["position"]]
    actual_vertices = snapshot["rendered"]["LineSegments"]["positions"]
    assert len(vertices) == len(actual_vertices)
    for a, b in zip(vertices, actual_vertices):
        near(a, b, 1e-6)  # Float32 GPU geometry versus Float64 physical records.
    colors = snapshot["rendered"]["LineSegments"]["colors"]
    assert len(colors) == len(vertices)
    for k, (_, b) in enumerate(zip(measured, measured[1:])):
        rgb = (105, 232, 255) if b["intervalActive"] else (128, 153, 141)
        # Three's vertex Color converts supplied sRGB to the linear working space.
        linear = [v / 255 / 12.92 if v / 255 <= .04045 else ((v / 255 + .055) / 1.055) ** 2.4 for v in rgb]
        for actual, expected in zip(colors[6 * k:6 * k + 6], linear * 2):
            near(actual, expected, 1e-6)
    actual_guide = snapshot["rendered"]["Line"]
    assert actual_guide["visible"] == (len(projection) > 1)
    assert len(actual_guide["positions"]) == 3 * len(projection)
    for a, b in zip(actual_guide["positions"], [v for p in projection for v in p["position"]]):
        near(a, b, 1e-6)


def verify_text(snapshot):
    current = snapshot["display"]["state"]["history"]["current"]
    c = current["command"]
    text = snapshot["panel"]["path-command"]["text"]
    assert f"{abs(c['yawRate'] * 180 / math.pi):.2f}°/s" in text
    assert ("right" if c["yawRate"] < 0 else "left") in text
    for key, positive, negative in (("forwardSpeed", "forward", "backward"), ("verticalSpeed", "climb", "descend")):
        assert f"{abs(c[key] * 100):.2f} cm/s {negative if c[key] < 0 else positive}" in text
    record = snapshot["display"]["state"]["decision"]
    assert record in snapshot["decisions"]
    target, action = record["targetSelection"], record["decision"]
    assert target["fruit"] in snapshot["panel"]["path-choice"]["text"]
    candidates = snapshot["panel"]["path-candidates"]["text"]
    for k, candidate in enumerate(target["candidates"]):
        match = re.search(rf"{candidate['fruit']}:\s*([\d.]+)%\s*·\s*score\s*([-+\deE.]+)", candidates)
        assert match, candidates
        near(float(match[1]) / 100, target["p"][k], .00050001)
        near(float(match[2]), candidate["score"], max(abs(candidate["score"]) * .00050001, 1e-15))
    sampling = snapshot["panel"]["path-sampling"]["text"]
    for label, expected in (("Odor draw", target["draw"]), ("Action draw", action["draw"])):
        match = re.search(rf"{label}\s+([\d.]+)", sampling)
        assert match, sampling
        near(float(match[1]), expected, .00005001)
    for label, expected in zip(("approach", "avoid"), action["p"]):
        match = re.search(rf"{label}\s+([\d.]+)%", sampling)
        assert match, sampling
        near(float(match[1]) / 100, expected, .00050001)
    assert ("→ approach intent" if action["action"] == 0 else "→ avoid intent") in sampling
    if max(target["p"]) - min(target["p"]) < .02:
        assert "Near tie" in snapshot["panel"]["path-choice"]["text"]
    assert snapshot["resetButton"].strip().lower() == "reset"
    logs = [event[1] for event in snapshot["events"]]
    forbidden = r"\b(?:approaches|avoids)\s+(?:the\s+)?(?:banana|bread)\b"
    assert not any(re.search(forbidden, text, re.I) for text in [*logs, snapshot["eventText"]]), "sampled intent must not be described as accomplished fruit-directed motion"
    intent = "approach" if action["action"] == 0 else "avoid"
    expected = f"samples {intent} intent for {target['fruit']}"
    assert any(expected in text for text in logs), (expected, logs)


def verify_decision(snapshot):
    record = snapshot["paths"]["decision"]
    assert record is not None and snapshot["paths"]["decisionCurrent"]
    assert record in snapshot["decisions"], "display decision must exactly copy an accepted worker message"
    target, action = record["targetSelection"], record["decision"]
    assert [c["fruit"] for c in target["candidates"]] == ["banana", "bread"]
    logits = []
    for candidate in target["candidates"]:
        solve = candidate["solve"]
        assert solve["converged"] and 0 <= solve["residual"] <= solve["tolerance"] <= 1e-6
        expected = candidate["readouts"]["mbon:MBON11:right"] - candidate["readouts"]["mbon:MBON05:left"]
        near(candidate["score"], expected)
        logits.append(expected / target["temperature"])
    weights = [math.exp(x - max(logits)) for x in logits]
    for a, b in zip(target["p"], [w / sum(weights) for w in weights]):
        near(a, b)
    assert 0 <= target["draw"] < 1
    assert target["fruit"] == ("banana" if target["draw"] <= target["p"][0] else "bread")
    assert action["accepted"] and not action["greedy"] and 0 <= action["draw"] < 1
    assert len(action["p"]) == 2 and all(0 <= p <= 1 for p in action["p"])
    near(sum(action["p"]), 1)
    assert action["choice"] == (0 if action["draw"] <= action["p"][0] else 1)
    assert action["action"] == action["choice"]
    return record


RETINAL_ISOLATION = """async () => {
 const THREE=await import('three');const a=window.__app,group=a.pathAnnotations;
 let scene=group;while(scene.parent)scene=scene.parent;
 const prior=group.visible,renderer=a.renderer,flight=a.flight();
 const hooks=[];scene.traverse(o=>{if(Object.hasOwn(o,'onBeforeRender'))hooks.push([o,o.onBeforeRender]);});
 const saved={target:renderer.getRenderTarget(),viewport:renderer.getViewport(new THREE.Vector4()).toArray(),
   scissor:renderer.getScissor(new THREE.Vector4()).toArray(),scissorTest:renderer.getScissorTest(),autoClear:renderer.autoClear};
 const geometry=new THREE.PlaneGeometry(1,1),material=new THREE.MeshBasicMaterial({color:0xff00ff,
   side:THREE.DoubleSide,depthTest:false,depthWrite:false,toneMapped:false});
 const marker=new THREE.Mesh(geometry,material);marker.renderOrder=1e6;marker.frustumCulled=false;
 try {
   // room.js animates on every render, even synchronous captures. Freeze only
   // those environment hooks for this paired visibility intervention.
   for(const [object] of hooks)object.onBeforeRender=()=>{};
   group.visible=false;const baseline=a.eye.capture(flight).rgba;
   const repeat=a.eye.capture(flight).rgba;
   const direction=a.eye.camera.getWorldDirection(new THREE.Vector3());
   marker.position.copy(a.eye.camera.position).addScaledVector(direction,.02);
   marker.quaternion.copy(a.eye.camera.quaternion);scene.add(marker);group.attach(marker);group.visible=true;
   const excluded=a.eye.capture(flight).rgba;
   // Positive control: exactly the same visible marker outside the excluded group.
   scene.attach(marker);const positive=a.eye.capture(flight).rgba;
   scene.remove(marker);const restored=a.eye.capture(flight).rgba;
   const different=(x,y)=>x.reduce((n,v,i)=>n+(v!==y[i]?1:0),0);
   const physical={p:flight.p.slice(),clock:a.life().clock,authority:JSON.stringify(a.life().authority)};
   a.setPaths(false);const hidden=a.paths().enabled===false&&group.visible===false;
   const toggleOff=a.eye.capture(flight).rgba;a.setPaths(true);const toggleOn=a.eye.capture(flight).rgba;
   const unchanged=JSON.stringify(physical)===JSON.stringify({p:flight.p.slice(),clock:a.life().clock,
     authority:JSON.stringify(a.life().authority)});
   return {bytes:baseline.length,suspendedRenderHooks:hooks.length,
     repeatedBaselineChangedBytes:different(baseline,repeat),excludedChangedBytes:different(baseline,excluded),
     positiveChangedBytes:different(baseline,positive),restoredChangedBytes:different(baseline,restored),
     toggleChangedBytes:different(toggleOff,toggleOn),toggleHidesAnnotations:hidden,
     togglePreservesPhysicalAndNeuralState:unchanged,
     rendererRestored:renderer.getRenderTarget()===saved.target&&renderer.autoClear===saved.autoClear
       &&renderer.getScissorTest()===saved.scissorTest
       &&JSON.stringify(renderer.getViewport(new THREE.Vector4()).toArray())===JSON.stringify(saved.viewport)
       &&JSON.stringify(renderer.getScissor(new THREE.Vector4()).toArray())===JSON.stringify(saved.scissor),
     baseline:Array.from(baseline),positive:Array.from(positive)};
 } finally {for(const [object,hook] of hooks)object.onBeforeRender=hook;
   marker.removeFromParent();geometry.dispose();material.dispose();group.visible=prior;}
}"""


def run(url, screenshot=None, reduced_views=False):
    frozen = source_hashes(ROOT, RUNNER)
    result = {"schema": "cadence.neural-path-view-check/3", "passed": False,
              "started": datetime.now(timezone.utc).isoformat(), "sha256": frozen,
              "checks": [], "page_errors": [],
              "protocol": {"seed": 1, "measured_window_simulation_seconds": .4,
                           "brain_off_window_simulation_seconds": .25, "reduced_views": reduced_views,
                           "retinal_positive_control": "Temporarily suspend render-time animation hooks to hold environment fixed; require repeated baseline equality; bright camera-facing annotation inside versus outside excluded group; synchronous capture and unconditional cleanup.",
                           "followup": "Preserves the initial moving-background-confounded failure and the passed frozen-scene receipt. Same protocol against final UI text: event logs report sampled intent and the actual Reset button is checked. No control-policy or pixel-threshold change."},
              "claim": "Viewer faithfully separates measured body history from a held-command kinematic projection and cannot supply retinal cues or change control state.",
              "limits": "One bounded UI run. Projection is not planned navigation or a physical trajectory prediction. This does not establish learned navigation or a new brain-level result."}
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query)); query.update(controller="assisted", seed="1")
    if reduced_views:
        query.update(noscan="1", nobloom="1", noviews="1")
    run_url = urlunsplit((*parts[:3], urlencode(query), parts.fragment))
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        result["runtime"] = {"python": platform.python_version(), "browser": browser.version}
        binding = ServedSourceBinding(page.context, urlunsplit((*parts[:3], "", "")), frozen)
        page.on("pageerror", lambda error: result["page_errors"].append(str(error)))
        page.add_init_script(RECORD_DECISIONS)
        try:
            page.goto(run_url, wait_until="domcontentloaded")
            page.wait_for_function("window.__app?.S.ready && window.__app.S.learnReady && window.__app.paths", timeout=60000)
            binding.wait_for_complete(page)
            page.wait_for_function("""()=>{const a=window.__app,p=a.paths(),L=a.life();return p.history.current?.active
              &&p.projection.length>1&&p.decisionCurrent&&L.authority.lastChoice?.executed;}""", timeout=90000)
            initial = result["initial_active"] = page.evaluate(READ)
            assert initial["source"] == "neural-navigation" and initial["brain"]["n"] == 150802
            verify_snapshot(initial, True); verify_decision(initial); verify_text(initial)
            assert page.get_by_role("button", name=re.compile(r"^reset$", re.I)).count() == 1
            page.wait_for_function("""t=>{const a=window.__app,p=a.paths();return a.life().clock>=t+.4
              &&p.history.current?.active&&p.projection.length>1&&p.decisionCurrent;}""", arg=initial["clock"], timeout=60000)
            active = result["active"] = page.evaluate(READ)
            verify_snapshot(active, True); verify_decision(active); verify_text(active)
            for old in initial["paths"]["history"]["points"]:
                assert old in active["paths"]["history"]["points"], "old measured samples must remain exact"
            assert len(active["paths"]["history"]["points"]) > len(initial["paths"]["history"]["points"])
            assert math.dist(active["p"], initial["p"]) > 1e-6, "bounded observation must include actual body motion"
            assert active["paths"]["enabled"] and active["annotationsVisible"]
            assert all(v["visible"] for v in active["panel"].values())
            text = " ".join(v["text"] for v in active["panel"].values()).lower()
            assert "projection" in text and "measured" in text
            assert "not" in text and ("plan" in text or "predict" in text), "projection limitation must be visible"
            result["checks"].append("measured history follows physics; current command matches authority; guide matches independent kinematic integration; accepted probabilities and draws retained exactly")
            isolation = page.evaluate(RETINAL_ISOLATION)
            for key in ("baseline", "positive"):
                pixels = bytes(isolation.pop(key)); isolation[key + "_sha256"] = hashlib.sha256(pixels).hexdigest()
            result["retinal_isolation"] = isolation
            assert isolation["bytes"] == 64 * 32 * 4
            assert isolation["positiveChangedBytes"] > 100, "bright marker must demonstrably enter the camera when not excluded"
            assert isolation["suspendedRenderHooks"] > 0
            assert all(isolation[k] == 0 for k in ("repeatedBaselineChangedBytes", "excludedChangedBytes", "restoredChangedBytes", "toggleChangedBytes"))
            assert isolation["toggleHidesAnnotations"] and isolation["togglePreservesPhysicalAndNeuralState"] and isolation["rendererRestored"]
            result["checks"].append("camera-frustum annotation excluded from raw retina, positive control visible, viewer toggles preserve retina and physical/neural state")
            if screenshot:
                page.screenshot(path=str(screenshot))
                result["screenshot"] = {"path": str(screenshot), "sha256": hashlib.sha256(screenshot.read_bytes()).hexdigest()}
            page.keyboard.press("Space")
            paused = result["paused"] = page.evaluate(READ)
            assert paused["paused"] and not paused["paths"]["projection"] and not paused["display"]["state"]["projection"]
            assert not paused["paths"]["history"]["current"]["active"]
            assert all(v == 0 for v in paused["paths"]["history"]["current"]["command"].values())
            page.wait_for_timeout(300)
            assert page.evaluate("window.__app.flight().p.slice()") == paused["p"]
            page.keyboard.press("Space")
            page.wait_for_function("""()=>{const a=window.__app,p=a.paths();return !a.S.paused
              &&p.history.current?.active&&p.projection.length>1;}""", timeout=60000)
            resumed = result["resumed"] = page.evaluate(READ)
            verify_snapshot(resumed, True)
            assert resumed["generation"] == paused["generation"]
            for old in paused["paths"]["history"]["points"]:
                assert old in resumed["paths"]["history"]["points"]
            result["checks"].append("pause withdraws projection and freezes position; fresh active control resumes without erasing measured history")
            page.get_by_role("button", name="brain off", exact=True).click()
            off_start = result["brain_off_start"] = page.evaluate(READ)
            assert off_start["pilot"] == "assisted-off" and not off_start["paths"]["projection"] and not off_start["display"]["state"]["projection"]
            page.wait_for_function("t=>window.__app.life().clock>=t+.25", arg=off_start["clock"], timeout=60000)
            off = result["brain_off"] = page.evaluate(READ)
            verify_snapshot(off)
            assert not off["paths"]["projection"] and not off["paths"]["history"]["current"]["active"]
            assert all(v == 0 for v in off["paths"]["history"]["current"]["command"].values())
            assert len(off["paths"]["history"]["points"]) > len(off_start["paths"]["history"]["points"])
            assert not off["paths"]["decisionCurrent"]
            result["checks"].append("brain-off removes command projection while actual physical history continues")
            before_generation = off["generation"]
            reset = result["reset"] = page.evaluate("""()=>{const a=window.__app;document.getElementById('forget').click();
              return {generation:a.S.generation,time:a.life().clock,p:a.flight().p.slice(),paths:a.paths()};}""")
            assert reset["generation"] > before_generation and reset["time"] == 0
            assert reset["paths"]["history"]["generation"] == reset["generation"]
            assert len(reset["paths"]["history"]["points"]) <= 1 and len(reset["paths"]["history"]["events"]) <= 1
            assert not reset["paths"]["projection"] and reset["paths"]["decision"] is None
            result["checks"].append("visible Reset button clears old generation trail, command events, projection and decision; neural event text reports sampled intent")
            assert not result["page_errors"], result["page_errors"]
            result["served_sources"] = binding.receipt()
            assert source_hashes(ROOT, RUNNER) == frozen, "source changed during bounded run"
            result["passed"] = True
        except Exception as error:
            result["error"] = str(error); result["traceback"] = traceback.format_exc()
            try:
                result["failure_state"] = page.evaluate(READ)
            except Exception as diagnostic_error:
                result["diagnostic_error"] = str(diagnostic_error)
            try:
                result["served_sources"] = binding.receipt()
            except Exception as binding_error:
                result["binding_error"] = str(binding_error)
            result["sources_unchanged"] = source_hashes(ROOT, RUNNER) == frozen
        finally:
            browser.close()
    result["completed"] = datetime.now(timezone.utc).isoformat()
    result["body_sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8817/")
    ap.add_argument("--reduced-views", action="store_true")
    ap.add_argument("--receipt", type=Path)
    ap.add_argument("--screenshot", type=Path)
    args = ap.parse_args()
    for path in (args.receipt, args.screenshot):
        if path and path.exists():
            ap.error(f"refusing to overwrite {path}")
    outcome = run(args.url, args.screenshot, args.reduced_views)
    if args.receipt:
        with args.receipt.open("x") as out:
            json.dump(outcome, out, indent=2, allow_nan=False); out.write("\n")
    print(json.dumps({k: outcome.get(k) for k in ("passed", "checks", "error", "body_sha256")}, indent=2))
    raise SystemExit(0 if outcome["passed"] else 1)
