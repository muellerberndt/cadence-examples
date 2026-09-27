#!/usr/bin/env python3
"""Measure the actual demo with the browser's normal hardware renderer."""
import argparse, hashlib, json, math, platform, statistics, traceback
from datetime import datetime, timezone
from pathlib import Path
from playwright.sync_api import sync_playwright
from browser_source_binding import ServedSourceBinding, source_hashes
ROOT=Path(__file__).resolve().parents[1]
RUNNER='tools/check_goal_performance.py'
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--url',default='http://127.0.0.1:8817/')
p.add_argument('--receipt',type=Path,required=True)
p.add_argument('--screenshot',type=Path,required=True)
p.add_argument('--seconds',type=float,default=60)
p.add_argument('--sugar',choices=['banana','bread'])
p.add_argument('--check-startup',action='store_true',help='Require bounded, uncredited startup motion in the unchanged initial scene')
a=p.parse_args()
for name in (a.receipt,a.screenshot):
    if name.exists():p.error(f'refusing to overwrite {name}')
pins=source_hashes(ROOT,RUNNER)
r={'schema':'cadence.goal-browser-performance/1','passed':False,'started':datetime.now(timezone.utc).isoformat(),
   'sourceSha256':pins,'platform':platform.platform(),'protocol':{'url':a.url,'seconds':a.seconds,'viewport':[1440,1000],
   'browser':'Chromium new headless, normal hardware renderer; no forced software rendering','sugarCommand':a.sugar,'threshold':'average >=45 FPS; p95 frame interval <=50ms; real-time ratio >=.95',
   'checkStartup':a.check_startup,
   'scope':'Single-machine foreground measurement including the full connectome worker, physical body, default scan and eye. No hidden views, pose or reward injection.'},'errors':[]}
with sync_playwright() as pw:
    browser=pw.chromium.launch(headless=True,channel='chromium')
    page=browser.new_page(viewport={'width':1440,'height':1000});r['browser']=browser.version
    binding=ServedSourceBinding(page.context,a.url,pins);page.on('pageerror',lambda e:r['errors'].append(str(e)))
    try:
        page.goto(a.url,wait_until='domcontentloaded');page.wait_for_function('window.__app?.S.ready&&__app.S.learnReady',timeout=90000)
        r['gpu']=page.evaluate('''()=>{const g=__app.renderer.getContext(),e=g.getExtension('WEBGL_debug_renderer_info');return {renderer:e?g.getParameter(e.UNMASKED_RENDERER_WEBGL):g.getParameter(g.RENDERER),vendor:e?g.getParameter(e.UNMASKED_VENDOR_WEBGL):g.getParameter(g.VENDOR),pixelRatio:__app.renderer.getPixelRatio()}}''')
        if a.sugar:
            page.locator('#console-input').fill('/sugar '+a.sugar);page.locator('#console-input').press('Enter')
            page.wait_for_function('(name)=>__app.life().sugar===name',arg=a.sugar)
        r['measurement']=page.evaluate('''async seconds=>{const a=__app,start=performance.now(),t0=a.life().clock,f0=window.__frames;
          const intervals=[],samples=[];let last=start,next=0;
          while(performance.now()-start<seconds*1000){const now=await Promise.race([new Promise(requestAnimationFrame),new Promise((_,reject)=>setTimeout(()=>reject(Error('No animation frame for 5s')),5000))]);intervals.push(now-last);last=now;
            if(now>=next){const l=a.life(),f=a.flight();samples.push({wall:now,time:l.clock,mode:l.mode,hunger:l.hunger,hidden:document.hidden,groom:l.groomTarget,position:f.p.slice(),velocity:f.v.slice(),yaw:f.euler()[2],goal:structuredClone(l.authority.goal),startup:structuredClone(l.authority.startup),choices:structuredClone(l.authority.choices),lessons:structuredClone(a.S.lessonStats),pending:a.S.pending,solveMs:a.S.solveMs});next=now+500;}}
          return {wallSeconds:(performance.now()-start)/1000,simulationSeconds:a.life().clock-t0,renderedFrames:window.__frames-f0,intervals,samples,console:a.console.snapshot()};}''',a.seconds)
        m=r['measurement'];v=sorted(x for x in m['intervals'] if x>0)
        r['summary']={'fps':m['renderedFrames']/m['wallSeconds'],'medianFrameMs':statistics.median(v),'p95FrameMs':v[math.ceil(.95*len(v))-1],
                      'timeRatio':m['simulationSeconds']/m['wallSeconds'],'modes':sorted({x['mode'] for x in m['samples']}),'minHunger':min(x['hunger'] for x in m['samples']),'maxHunger':max(x['hunger'] for x in m['samples'])}
        moving=[x for x in m['samples'] if math.hypot(*x['velocity'][:2])>.1]
        angles=[abs(math.atan2(math.sin(math.atan2(x['velocity'][1],x['velocity'][0])-x['yaw']),math.cos(math.atan2(x['velocity'][1],x['velocity'][0])-x['yaw'])))*180/math.pi for x in moving]
        r['summary']['movingMedianSpeed']=statistics.median(math.hypot(*x['velocity']) for x in moving) if moving else None
        r['summary']['movingMedianFacingErrorDegrees']=statistics.median(angles) if angles else None
        if a.check_startup:
            assert a.sugar is None,'Startup check requires the unchanged initial scene'
            first=next((x for x in m['samples'] if math.hypot(*x['velocity'][:2])>.05),None)
            assert first and first['time']<=1,'No visible movement during the first simulated second'
            assert first['startup']['active'] and first['choices']['executed']==0,'Initial movement must be declared body support without neural credit'
            # learn:ready reports weight statistics before the first lesson;
            # its absent updates counter denotes no lessons, as in the HUD.
            assert first['goal']['fruit'] is None and first['lessons'].get('updates',0)==0
            assert any(not x['startup']['active'] for x in m['samples']),'Launch did not end'
            r['summary']['firstMovingSimSeconds']=first['time']
        neural=next((x for x in m['samples'] if x['choices']['executed']>0),None)
        r['summary']['firstExecutedNeuralChoiceSimSeconds']=neural['time'] if neural else None
        page.screenshot(path=str(a.screenshot));r['image']={'path':str(a.screenshot.relative_to(ROOT) if a.screenshot.is_absolute() else a.screenshot),'sha256':hashlib.sha256(a.screenshot.read_bytes()).hexdigest()}
        r['served']=binding.receipt();r['sourcesUnchanged']=source_hashes(ROOT,RUNNER)==pins
        assert r['sourcesUnchanged'] and not r['errors']
        assert all(not x['hidden'] and 0<=x['hunger']<=1 for x in m['samples'])
        assert 'SwiftShader' not in r['gpu']['renderer'],'No hardware renderer available'
        s=r['summary'];assert s['fps']>=45 and s['p95FrameMs']<=50 and s['timeRatio']>=.95,'Frame/pacing target not met'
        r['passed']=True
    except Exception as e:r['error']=repr(e);r['traceback']=traceback.format_exc()
    finally:browser.close()
r['completed']=datetime.now(timezone.utc).isoformat();r['body_sha256']=hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
a.receipt.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:r.get(k) for k in ['passed','error','gpu','summary','body_sha256']},indent=2))
raise SystemExit(0 if r['passed'] else 1)
