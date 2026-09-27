#!/usr/bin/env python3
"""Source-bound real-browser console/eye verification. Refuses existing outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import traceback
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright
from browser_source_binding import ServedSourceBinding, source_hashes

ROOT = Path(__file__).resolve().parents[1]
RUNNER = 'tools/check_brain_console.py'
RECORD = """(()=>{
 window.__consolePackets=[];window.__consoleRequests=[];window.__consoleSubmits=[];window.__consoleHeld=[];
 window.__consoleHold=false;window.__consoleReleasing=false;
 const cp=x=>structuredClone(x), witnesses=new WeakMap();
 const body=()=>{const a=__app,f=a.flight(),L=a.life();return {state:a.states(),details:a.details(),p:f.p.slice(),v:f.v.slice(),q:f.q.slice(),omega:f.w.slice(),
  navigation:cp(L.authority.navigation),observations:cp(L.authority.observations),choices:cp(L.authority.choices),
  requestId:a.S.requestId,zoom:a.rig.zoom,camera:a.rig.mode,sugar:L.sugar,source:L.source};};
 window.__consoleBody=body;
 document.addEventListener('submit',e=>{if(e.target.id!=='console-form'||!window.__app)return;
  const w={raw:document.getElementById('console-input').value,before:body(),beforeId:__app.console.snapshot().entries.at(-1)?.id??0};
  witnesses.set(e,w);__consoleSubmits.push(w);},true);
 window.__installConsoleWitness=()=>document.getElementById('console-form').addEventListener('submit',e=>{const w=witnesses.get(e);if(w)w.after=body();});
 const send=Worker.prototype.postMessage;Worker.prototype.postMessage=function(m,...args){
  __consoleRequests.push({type:m.type,requestId:m.requestId,generation:m.generation,searchToken:m.searchToken,reward:m.reward,
   senses:m.senses,retina:m.retina?{width:m.retina.width,height:m.retina.height,bytes:m.retina.rgba.length}:null});
  return send.call(this,m,...args);};
 const Native=Worker;window.Worker=new Proxy(Native,{construct(T,args){const w=new T(...args);w.addEventListener('message',e=>{
  if(__consoleReleasing)return;const m=e.data;
  __consolePackets.push({type:m.type,kind:m.kind,requestId:m.requestId,generation:m.generation,searchToken:m.searchToken,
   converged:m.converged,iterations:m.iterations,initialResidual:m.initialResidual,residual:m.residual,tolerance:m.tolerance,
   active:m.active,n:m.s?.length,retinal:m.retinal,readouts:m.readouts,decision:m.decision,targetSelection:m.targetSelection,
   accepted:m.accepted,updates:m.updates,changed:m.changed,moved:m.moved,delta:m.delta,
   trace:m.trace?{identity:m.trace.identity,n:m.trace.n}:null});
  if(m.type==='control'&&__consoleHold){__consoleHeld.push({w,m});e.stopImmediatePropagation();}
 });return w;}});
 window.__releaseConsoleControls=()=>{__consoleHold=false;const ids=[];while(__consoleHeld.length){const {w,m}=__consoleHeld.shift();
  __consoleReleasing=true;try{w.dispatchEvent(new MessageEvent('message',{data:m}));ids.push(m.requestId);}finally{__consoleReleasing=false;}}return ids;};
})();"""
READ = """()=>({body:__consoleBody(),console:__app.console.snapshot(),packets:__consolePackets,requests:__consoleRequests,submits:__consoleSubmits})"""


def send(page, command):
    before = page.evaluate('__consoleSubmits.length')
    page.locator('#console-input').fill(command)
    page.locator('#console-input').press('Enter')
    page.wait_for_function("""n=>{const w=__consoleSubmits[n];return w?.after&&
      (w.raw==='/clear'||__app.console.snapshot().entries.some(e=>e.id>w.beforeId&&['reply','error'].includes(e.kind)));}""", arg=before, timeout=10000)
    return page.evaluate("""n=>{const w=__consoleSubmits[n];return {...w,output:__app.console.snapshot().entries.filter(e=>e.id>w.beforeId&&['reply','error'].includes(e.kind))};}""", before)


def check_status(witness):
    s = witness['before']['state']; n = s['neural']; text = '\n'.join(e['text'] for e in witness['output'])
    assert witness['output'] and all(e['kind'] == 'reply' for e in witness['output']), witness
    if n['available']:
        sample = n['sample']; label = 'RECORDED ' if n['state'] == 'paused' else ''
        label += 'SETTLED' if sample['converged'] else 'UNQUALIFIED'
        assert f'Brain {label}' in text and f"{sample['iterations']:,} settling steps" in text
        assert f"input #{sample['requestId']:,}" in text
        if sample['converged']:
            assert f"{sample['active']:,} / {sample['neuronCount']:,} cells" in text
    else:
        assert f"Brain {n['state'].upper()}" in text and 'mismatch unavailable' in text
        assert 'Mismatch ' not in text and ' settling steps\n' not in text
    if s['learningUpdates'] is not None:
        assert f"Learning {s['learningUpdates']:,} updates" in text
    return {'state': n['state'], 'available': n['available'], 'text': text}


def streams(snapshot, required=True):
    entries = snapshot['console']['entries']; packets = snapshot['packets']; body = snapshot['body']
    counts = {k: 0 for k in ('settle', 'attention', 'course', 'learn')}
    for e in entries:
        if e['kind'] not in counts:
            continue
        m = e['meta']; assert m and isinstance(m['generation'], int) and isinstance(m['requestId'], int)
        kind = {'settle': 'control', 'course': 'control', 'attention': 'assisted_decision', 'learn': 'lesson'}[e['kind']]
        p = next(p for p in packets if p['type'] == kind and p.get('generation') == m['generation'] and p.get('requestId') == m['requestId'])
        if kind in ('control', 'assisted_decision'):
            assert p['converged'] is True and 0 <= p['residual'] <= p['tolerance'] <= 1e-6
        if e['kind'] == 'settle':
            assert m['accepted'] is True and m['iterations'] == p['iterations'] and m['residual'] == p['residual']
            assert p['n'] == 150802 and p['retinal']['neurons'] == 1831
        elif e['kind'] == 'attention':
            assert m['fruit'] == p['targetSelection']['fruit'] and m['action'] == p['decision']['action']
        elif e['kind'] == 'course':
            assert 0 < m['framesApplied'] <= body['navigation']['framesApplied']
            assert any(x['kind'] == 'settle' and x['meta']['generation'] == m['generation'] and x['meta']['requestId'] == m['requestId'] and x['id'] < e['id'] for x in entries)
            assert all(isinstance(v, (int, float)) and math.isfinite(v) for v in m['command'].values())
            assert any(m['command'][k] != 0 for k in ('yawRate', 'forwardSpeed', 'verticalSpeed'))
        else:
            assert p['accepted'] is True
            for key in ('updates', 'changed', 'moved', 'delta'):
                assert m[key] == p[key]
        counts[e['kind']] += 1
    if required:
        assert all(counts[k] > 0 for k in ('settle', 'attention', 'course')), counts
    return counts


def geometry(page, ids):
    return page.evaluate("""ids=>Object.fromEntries(ids.map(id=>{const e=document.getElementById(id);if(!e)return[id,null];
      const r=e.getBoundingClientRect(),s=getComputedStyle(e),hit=document.elementFromPoint(r.x+r.width/2,r.y+r.height/2);
      return[id,{left:r.left,right:r.right,top:r.top,bottom:r.bottom,width:r.width,height:r.height,
      visible:s.display!=='none'&&s.visibility!=='hidden'&&r.width>0&&r.height>0,
      unobscured:!!hit&&(hit===e||e.contains(hit))}];}))""", ids)


def layout(page, width, height):
    g = geometry(page, ['brain-console', 'console-input', 'console-toggle', 'eye'])
    for name, box in g.items():
        assert box and box['visible'] and 0 <= box['left'] < box['right'] <= width and 0 <= box['top'] < box['bottom'] <= height, (name, box)
        if name in ('console-input', 'console-toggle', 'eye'):
            assert box['unobscured'], (name, box)
    return g


def save_image(page, path):
    page.screenshot(path=str(path))
    return {'path': str(path.resolve().relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def run(args):
    frozen = source_hashes(ROOT, RUNNER)
    r = {'schema': 'cadence.brain-console-browser/1', 'passed': False, 'started': datetime.now(timezone.utc).isoformat(),
         'sha256': frozen, 'checks': [], 'page_errors': [], 'protocol': {
             'url': args.url, 'desktop': [1440, 1000], 'mobile': [430, 900],
             'freshness': 'Live and paused status checked against exact submit-time telemetry; stale formatting has focused unit coverage.',
             'scope': 'UI, actual neural-event provenance, commands and rendering. No fabricated neural/body states; no new learning-performance or flight-competence claim. Learn entries, if emitted, must join real accepted lessons; this assay does not deliberately train.'}}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
        page = browser.new_page(viewport={'width': 1440, 'height': 1000}); r['browser'] = browser.version
        binding = ServedSourceBinding(page.context, args.url, frozen)
        page.on('pageerror', lambda e: r['page_errors'].append(str(e)))
        page.add_init_script(RECORD)
        try:
            page.goto(args.url, wait_until='domcontentloaded')
            page.wait_for_function('window.__app?.S.ready&&__app.S.learnReady&&__app.console', timeout=90000)
            page.evaluate('__installConsoleWitness()')
            page.wait_for_function("getComputedStyle(document.getElementById('loading')).opacity==='0'", timeout=10000)
            assert page.evaluate("['status','toolbar','options','inset'].every(id=>!document.getElementById(id))")
            assert page.evaluate('__app.details().panel===null&&__app.details().eyeVisible&&!__app.details().eyeMain')
            r['desktop_layout'] = layout(page, 1440, 1000)
            page.wait_for_function("['settle','attention','course'].every(k=>__app.console.snapshot().entries.some(e=>e.kind===k))", timeout=60000)
            r['live'] = page.evaluate(READ); r['stream_counts'] = streams(r['live'])
            r['live_status'] = check_status(send(page, '/status'))
            r['desktop_screenshot'] = save_image(page, args.screenshot)
            eye = geometry(page, ['eye'])['eye']; im = Image.open(args.screenshot).convert('RGB')
            colors = {im.getpixel((x, y)) for y in range(int(eye['top']) + 30, int(eye['bottom']) - 4, 3) for x in range(int(eye['left']) + 4, int(eye['right']) - 4, 3)}
            assert len(colors) > 20, 'eye viewport is blank'
            r['eye_sampled_color_count'] = len(colors)
            r['checks'].append('default is console plus live eye; real settle/attention/course entries join actual qualified packets and physical-step counters')
            print(json.dumps({'startup_streams': r['stream_counts'], 'eye_colors': len(colors)}), flush=True)

            send(page, '/eye off'); assert page.evaluate('!__app.details().eyeVisible&&!__app.details().eyeMain')
            request_id = page.evaluate('__app.S.requestId')
            page.wait_for_function("id=>__consoleRequests.some(m=>m.type==='assist:observe'&&m.requestId>id&&m.retina?.bytes===8192)", arg=request_id, timeout=30000)
            r['eye_off_request'] = page.evaluate("id=>__consoleRequests.find(m=>m.type==='assist:observe'&&m.requestId>id)", request_id)
            send(page, '/eye full'); assert page.evaluate('__app.details().eyeMain')
            send(page, '/eye on'); assert page.evaluate('__app.details().eyeVisible&&!__app.details().eyeMain')
            zoom = page.evaluate('__app.rig.zoom'); page.mouse.move(720, 350); page.mouse.wheel(0, -120)
            page.wait_for_function('z=>__app.rig.zoom<z', arg=zoom); r['zoom'] = {'before': zoom, 'after': page.evaluate('__app.rig.zoom')}
            page.mouse.wheel(0, 120)
            r['checks'].append('eye off keeps retinal packets, full/on modes work, and real mouse wheel zooms the main view')

            page.wait_for_function('__app.states().neural.available', timeout=30000)
            gust = send(page, '/gust'); r['gust'] = gust
            assert any(gust['before'][k] != gust['after'][k] for k in ('q', 'omega', 'v'))
            assert gust['before']['state']['time'] == gust['after']['state']['time'] and not gust['after']['state']['neural']['available']
            page.wait_for_function('__app.states().neural.available', timeout=30000)
            send(page, '/pause'); assert page.evaluate('__app.S.paused')
            r['paused_status'] = check_status(send(page, '/status')); assert r['paused_status']['state'] == 'paused'
            stable = page.evaluate('__consoleBody()')
            page.locator('#console-input').press_sequentially('g w i e 1 2 3 ', delay=10)
            typed = page.evaluate('__consoleBody()')
            for key in ('p', 'v', 'q', 'omega', 'state', 'details', 'camera'):
                assert typed[key] == stable[key], ('typing caused global shortcut', key)
            r['inert_expression'] = send(page, 'window.__consoleEvalProbe = 1')
            assert any(e['kind'] == 'error' for e in r['inert_expression']['output']) and page.evaluate("typeof window.__consoleEvalProbe==='undefined'")
            r['unknown'] = send(page, '/not-a-command'); assert any(e['kind'] == 'error' for e in r['unknown']['output'])
            r['invalid_speed'] = send(page, '/speed 10'); assert any(e['kind'] == 'error' for e in r['invalid_speed']['output'])
            for command in ('/gust', '/nudge'):
                w = send(page, command)
                assert all(w['before'][k] == w['after'][k] for k in ('p', 'v', 'q', 'omega', 'state'))
            r['checks'].append('live/paused status reflects exact submit-time measurements; real gust changes inputs; typed shortcuts and unknown/expression commands are inert')

            send(page, '/help'); page.locator('#console-input').press('ArrowUp'); assert page.locator('#console-input').input_value() == '/help'
            page.locator('#console-input').press('ArrowDown'); assert page.locator('#console-input').input_value() == ''
            page.locator('#console-input').fill('/sta'); page.locator('#console-input').press('Tab'); assert page.locator('#console-input').input_value() == '/status '
            page.locator('#console-input').fill('')
            entries = page.evaluate('__app.console.snapshot().entries'); page.locator('#console-toggle').click()
            assert page.evaluate('__app.console.snapshot().collapsed&&document.getElementById("console-log").hidden')
            assert page.locator('#console-input').is_visible(); assert page.evaluate('__app.console.snapshot().entries') == entries
            page.locator('#console-toggle').click(); assert not page.evaluate('__app.console.snapshot().collapsed')
            for mode in ('3d', 'scan', 'settling'):
                send(page, '/brain ' + mode)
                assert page.evaluate('__app.details().panel') == 'brain'
                assert page.evaluate('__app.settlement().mode') == ('brain' if mode == '3d' else mode)
            send(page, '/brain off'); assert page.evaluate('__app.details().panel') is None
            for cmd, expected in (('/paths on', 'paths'), ('/paths off', None), ('/senses on', 'instruments'), ('/senses off', None), ('/about', 'about'), ('/close', None)):
                send(page, cmd); assert page.evaluate('__app.details().panel') == expected
            for speed in ('0.25', '1', '0.5'):
                send(page, '/speed ' + speed); assert page.evaluate('__app.S.speed') == float(speed)
            for sugar in ('bread', 'none', 'banana'):
                send(page, '/sugar ' + sugar); assert page.evaluate('__app.life().sugar') == (None if sugar == 'none' else sugar)
            for camera in ('room', 'follow'):
                send(page, '/camera ' + camera); assert page.evaluate('__app.rig.mode') == camera
            send(page, '/clear'); assert page.evaluate('__app.console.snapshot().entries.length') == 0
            send(page, '/help'); r['checks'].append('history, completion, collapse, clear and validated view/camera/speed/sugar commands work without evaluating code')

            page.set_viewport_size({'width': 430, 'height': 900})
            r['mobile_layout'] = layout(page, 430, 900)
            r['mobile_screenshot'] = save_image(page, args.mobile_screenshot)
            send(page, '/brain settling')
            panel = geometry(page, ['brain', 'console-input', 'eye']); r['mobile_panel_layout'] = panel
            for name in ('brain', 'console-input'):
                box = panel[name]; assert box['visible'] and 0 <= box['left'] < box['right'] <= 430 and 0 <= box['top'] < box['bottom'] <= 900, (name, box)
            assert panel['console-input']['unobscured']
            send(page, '/close'); send(page, '/about')
            page.wait_for_function("document.getElementById('card').getAnimations().filter(a=>a instanceof CSSTransition).every(a=>a.playState!=='running')", timeout=10000)
            card = geometry(page, ['card'])['card']; assert card['visible'] and 0 <= card['top'] < card['bottom'] <= 900
            send(page, '/close'); generation = page.evaluate('__app.S.generation'); send(page, '/reset')
            page.wait_for_function('g=>__app.S.generation>g&&__app.S.ready&&__app.S.learnReady', arg=generation, timeout=90000)
            r['reset'] = page.evaluate('__app.states()'); assert r['reset']['neural']['sample'] is None and r['reset']['learningUpdates'] == 0
            send(page, '/resume'); page.wait_for_function('__app.states().neural.available', timeout=30000)
            r['recovered'] = page.evaluate('__app.states()')
            r['checks'].append('desktop/mobile input stays accessible, inspectors remain on demand, and typed reset/resume restores a new live generation')
            r['final'] = page.evaluate(READ); r['served_sources'] = binding.receipt()
            assert not r['page_errors'] and source_hashes(ROOT, RUNNER) == frozen
            r['passed'] = True
        except Exception as error:
            r['error'] = repr(error); r['traceback'] = traceback.format_exc()
            try: r['failure'] = page.evaluate(READ)
            except Exception as error: r['capture_error'] = repr(error)
            if not args.screenshot.exists():
                try: r['failure_screenshot'] = save_image(page, args.screenshot)
                except Exception as error: r['screenshot_error'] = repr(error)
        finally:
            r['sources_unchanged'] = source_hashes(ROOT, RUNNER) == frozen
            browser.close()
    r['completed'] = datetime.now(timezone.utc).isoformat()
    r['body_sha256'] = hashlib.sha256(json.dumps(r, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return r


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8817/')
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--screenshot', type=Path, required=True)
    parser.add_argument('--mobile-screenshot', type=Path, required=True)
    args = parser.parse_args()
    for path in (args.receipt, args.screenshot, args.mobile_screenshot):
        if path.exists(): parser.error(f'refusing to overwrite {path}')
    outcome = run(args)
    args.receipt.write_text(json.dumps(outcome, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: outcome.get(k) for k in ('passed', 'checks', 'error', 'body_sha256')}, indent=2))
    raise SystemExit(0 if outcome['passed'] else 1)
