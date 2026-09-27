#!/usr/bin/env python3
"""Bounded source-bound browser check of default scan, recorded repair and motion."""
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
RUNNER = 'tools/check_scan_motion.py'
RECORD = """(()=>{window.__motionFrames=0;window.__motionPackets=[];
 const frame=()=>{__motionFrames++;requestAnimationFrame(frame)};requestAnimationFrame(frame);
 const Native=Worker;window.Worker=new Proxy(Native,{construct(T,args){const w=new T(...args);
 w.addEventListener('message',e=>{const m=e.data;if(m.type!=='control')return;
 const r=m.replay;__motionPackets.push({requestId:m.requestId,generation:m.generation,iterations:m.iterations,
 residual:m.residual,tolerance:m.tolerance,converged:m.converged,retinal:m.retinal,
 replay:r?structuredClone(r):null});if(__motionPackets.length>4)__motionPackets.shift();});return w;}});})();"""
SAMPLE = """()=>{const a=__app,L=a.life(),f=a.flight(),n=L.authority.navigation;return {
 wall:performance.now(),time:L.clock,frames:__motionFrames,fps:a.S.fps,
 position:f.p.slice(),velocity:f.v.slice(),heading:f.euler()[2],speed:Math.hypot(f.v[0],f.v[1]),touching:f.touching,onFloor:f.onFloor,
 freshFrames:n.framesApplied,freshActive:n.active,motion:structuredClone(L.authority.motionSupport),
 observations:structuredClone(L.authority.observations),details:a.details(),replay:a.scanReplay()};}"""
WINDOW = """async source=>{const sample=eval('('+source+')'),rows=[sample()],end=performance.now()+8000;
 while(performance.now()<end){await new Promise(r=>setTimeout(r,200));rows.push(sample());}return rows;}"""


def summary(rows):
    a, b = rows[0], rows[-1]; wall = (b['wall'] - a['wall']) / 1000
    errors = [abs((x['heading']-math.atan2(x['velocity'][1],x['velocity'][0])+math.pi)%(2*math.pi)-math.pi)*180/math.pi for x in rows if x['speed'] > .1]
    return {'wallSeconds':wall,'simSeconds':b['time']-a['time'],'fps':(b['frames']-a['frames'])/wall,
            'netHorizontalM':math.hypot(b['position'][0]-a['position'][0],b['position'][1]-a['position'][1]),
            'minSpeedMps':min(x['speed'] for x in rows),'medianSpeedMps':statistics.median(x['speed'] for x in rows),
            'fractionAbovePointOne':sum(x['speed']>.1 for x in rows)/len(rows),
            'contactSamples':sum(x['touching']>0 for x in rows),
            'freshSteps':b['freshFrames']-a['freshFrames'],
            'supportSteps':{k:b['motion'][k]-a['motion'][k] for k in ('appliedFrames','freshFrames','delayedFrames','heldFrames')},
            'medianNoseTravelDegrees':statistics.median(errors) if errors else None}


def geometry(page, width, height):
    boxes=page.evaluate("""()=>Object.fromEntries(['brain','scan','eye','brain-console','console-input','scan-playback'].map(id=>{
      const e=document.getElementById(id),r=e.getBoundingClientRect(),s=getComputedStyle(e),h=document.elementFromPoint(r.x+r.width/2,r.y+r.height/2);
      return[id,{left:r.left,right:r.right,top:r.top,bottom:r.bottom,width:r.width,height:r.height,
      visible:s.display!=='none'&&r.width>0&&r.height>0,unobscured:h===e||e.contains(h)}];}))""")
    for name, b in boxes.items():
        assert b['visible'] and 0 <= b['left'] < b['right'] <= width and 0 <= b['top'] < b['bottom'] <= height, (name,b)
        assert b['unobscured'], (name,b)
    assert boxes['brain']['bottom'] <= boxes['eye']['top']
    assert boxes['brain']['width'] <= (280 if width > 760 else 170)
    assert boxes['scan']['height'] >= (250 if width > 760 else 100)
    return boxes


def command(page, raw):
    page.locator('#console-input').fill(raw);page.locator('#console-input').press('Enter')


def image(page, path):
    page.screenshot(path=str(path));return {'path':str(path.relative_to(ROOT)), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def run(args):
    frozen=source_hashes(ROOT,RUNNER)
    r={'schema':'cadence.scan-motion-browser/1','passed':False,'started':datetime.now(timezone.utc).isoformat(),
       'sha256':frozen,'errors':[],'checks':[],'protocol':{'url':args.url,'desktop':[1440,1000],'mobile':[430,900],
       'windowWallSeconds':8,'sampleMilliseconds':200,'speedThresholdMps':.1,'minimumMovingFraction':.5,'maximumMedianNoseTravelDegrees':30,
       'stopThresholdMps':.02,'stopWindow':'Across both retained windows, airborne noncontact samples after2 simulated seconds from the first admitted motion-support source. All contacts retained and reported; at least3 eligible samples required. Late window may be entirely at a wall.',
       'scope':'One UI and motion-support wiring check. Supplied motion support is separated from fresh neural actuation; no navigation competence or learning claim.'}}
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--use-angle=swiftshader','--enable-unsafe-swiftshader'])
        page=browser.new_page(viewport={'width':1440,'height':1000});r['browser']=browser.version
        binding=ServedSourceBinding(page.context,args.url,frozen)
        page.on('pageerror',lambda e:r['errors'].append(str(e)));page.add_init_script(RECORD)
        try:
            page.goto(args.url,wait_until='domcontentloaded')
            page.wait_for_function('window.__app?.S.ready&&__app.S.learnReady',timeout=90000)
            r['initial']=page.evaluate(WINDOW,SAMPLE);r['initialSummary']=summary(r['initial'])
            print(json.dumps({'initial':r['initialSummary']}),flush=True)
            assert page.evaluate("__app.details().panel==='brain'&&__app.settlement().mode==='scan'&&__app.details().eyeVisible&&__app.details().speed===1")
            assert page.evaluate("['status','toolbar','options','inset'].every(id=>!document.getElementById(id))")
            page.wait_for_function("getComputedStyle(document.getElementById('loading')).opacity==='0'",timeout=10000)
            r['desktopLayout']=geometry(page,1440,1000)
            if not any(x['motion']['source'] and x['speed']>.1 for x in r['initial']):
                page.wait_for_function('__app.life().authority.motionSupport.source&&Math.hypot(...__app.flight().v.slice(0,2))>.1',timeout=30000)
            r['steady']=page.evaluate(WINDOW,SAMPLE);r['steadySummary']=summary(r['steady'])
            r['desktopImage']=image(page,args.screenshot.resolve())
            all_rows=r['initial']+r['steady']
            first_source=min(x['motion']['source']['acceptedAt'] for x in all_rows if x['motion']['source'])
            air=[x for x in all_rows if not x['touching'] and not x['onFloor'] and x['time']>=first_source+2]
            assert len(air)>=3 and sum(x['speed']>.1 for x in air)/len(air) > .5,r['steadySummary']
            assert all(x['speed']>=.02 for x in air),r['steadySummary']
            errors=[abs((x['heading']-math.atan2(x['velocity'][1],x['velocity'][0])+math.pi)%(2*math.pi)-math.pi)*180/math.pi for x in air if x['speed']>.1]
            r['airborneAssessment']={'firstSupportAcceptedAt':first_source,'samples':len(air),'movingFraction':sum(x['speed']>.1 for x in air)/len(air),
                'medianSpeedMps':statistics.median(x['speed'] for x in air),'minimumSpeedMps':min(x['speed'] for x in air),
                'medianNoseTravelDegrees':statistics.median(errors) if errors else None,'contactSamples':sum(x['touching']>0 for x in all_rows)}
            assert errors and statistics.median(errors)<30,r['airborneAssessment']
            support_before=all_rows[0]['motion'];support_after=all_rows[-1]['motion']
            assert support_after['delayedFrames']+support_after['heldFrames']>support_before['delayedFrames']+support_before['heldFrames']
            r['checks'].append('default narrow scan, eye and console visible; actual supported motion sustained with fresh/delayed/held counters separate')
            shown=[x['replay'] for x in r['initial']+r['steady'] if x['replay'] and x['replay']['displayed']]
            assert any(a['identity']==b['identity'] and a['frameIndex']<b['frameIndex'] and a['iteration']<b['iteration'] for a,b in zip(shown,shown[1:])), 'No measured replay advancement observed'
            page.wait_for_function('__app.scanReplay()?.displayed',timeout=15000)
            r['replay']=page.evaluate("""()=>{const d=__app.scanReplay(),m=__motionPackets.find(m=>m.generation===d.identity.generation&&m.requestId===d.identity.requestId);
              if(!m?.replay)throw Error('Displayed replay has no actual worker packet');
              const f=m.replay.frames[d.frameIndex],scan=__app.scan();return{display:d,rendered:d.samples.map(s=>({activity:scan.activation[s.atlasIndex],heat:scan.heat[s.atlasIndex]})),packet:{requestId:m.requestId,generation:m.generation,
              residual:m.residual,converged:m.converged,iterations:m.iterations,replay:{schema:m.replay.schema,n:m.replay.n,
              identity:m.replay.identity,frameCount:m.replay.frames.length,iteration:f.iteration,residual:f.residual,
              peakDelta:m.replay.maxAbsDeltaV,
              samples:d.samples.map(s=>({neuronIndex:s.neuronIndex,activity:f.activity[s.neuronIndex],deltaV:f.deltaV[s.neuronIndex]}))}}};}""")
            d=r['replay']['display'];rp=r['replay']['packet']['replay']
            assert d['iteration']==rp['iteration'] and d['residual']==rp['residual'] and rp['n']==150802
            assert len(d['samples'])>=5
            for a,b,rendered in zip(d['samples'],rp['samples'],r['replay']['rendered']):
                assert a['neuronIndex']==b['neuronIndex'] and a['activity']==b['activity'] and a['deltaV']==b['deltaV']
                assert rendered['activity']==a['activity'] and rendered['heat']==a['heat']
                heat=math.log1p(99*abs(b['deltaV'])/rp['peakDelta'])/math.log(100) if rp['peakDelta'] else 0
                assert abs(a['heat']-heat)<1e-7
            r['checks'].append('displayed recorded repair identity, iteration, residual and neuron activity/delta witnesses equal the actual worker frame')
            command(page,'/pause');page.wait_for_function('__app.S.paused')
            page.set_viewport_size({'width':430,'height':900})
            r['mobileLayout']=geometry(page,430,900);r['mobileImage']=image(page,args.mobile_screenshot.resolve())
            command(page,'/brain settling');page.wait_for_function("__app.settlement().mode==='settling'")
            command(page,'/close');page.wait_for_function("__app.details().panel==='brain'&&__app.settlement().mode==='scan'")
            r['checks'].append('mobile scan and eye fit above console; optional settlement inspector closes back to default scan')
            r['served']=binding.receipt();assert not r['errors'] and source_hashes(ROOT,RUNNER)==frozen
            r['passed']=True
        except Exception as e:
            r['error']=repr(e);r['traceback']=traceback.format_exc()
            try:r['failure']=page.evaluate(SAMPLE)
            except Exception:pass
            if not args.screenshot.exists():
                try:r['failureImage']=image(page,args.screenshot.resolve())
                except Exception:pass
        finally:
            r['sourcesUnchanged']=source_hashes(ROOT,RUNNER)==frozen;browser.close()
    r['completed']=datetime.now(timezone.utc).isoformat()
    r['body_sha256']=hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return r


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--url',default='http://127.0.0.1:8817/')
    for name in ('receipt','screenshot','mobile-screenshot'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    for path in (args.receipt,args.screenshot,args.mobile_screenshot):
        if path.exists():p.error(f'refusing to overwrite {path}')
    r=run(args);args.receipt.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:r.get(k) for k in ('passed','checks','error','body_sha256')},indent=2))
    raise SystemExit(0 if r['passed'] else 1)
