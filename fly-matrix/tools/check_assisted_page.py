#!/usr/bin/env python3
"""Actual page/worker checks; assisted flight is never evidence of autonomous flight.

--full retains the atlas, bloom and secondary views. Receipts bind current sources.
The final near-food start is an explicit environmental test fixture, not navigation evidence.
"""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
from browser_source_binding import ServedSourceBinding, source_hashes as freeze_sources

ROOT = Path(__file__).resolve().parents[1]
READ = '''() => {const a=window.__app;return {wall:performance.now(),time:a.S.simTime,brain:a.brainInfo(),
 mode:a.S.pilot,p:a.flight().p.slice(),v:a.flight().v.slice(),authority:a.life().authority,
 lessons:a.S.lessonStats,decisions:a.S.decisions,controls:{...a.life().controls},events:a.life().events}}'''

def source_hashes():
    return freeze_sources(ROOT, 'tools/check_assisted_page.py')

def run(url, full, screenshot=None):
    frozen_sources=source_hashes()
    checks=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--use-angle=swiftshader','--enable-unsafe-swiftshader'])
        page=browser.new_page(viewport={'width':1440,'height':1000}); errors=[]
        binding=ServedSourceBinding(page.context,url,frozen_sources)
        page.on('pageerror',lambda e: errors.append(str(e)))
        page.goto(url+('' if full else '?noscan=1&nobloom=1&noviews=1'),wait_until='domcontentloaded')
        page.wait_for_function('window.__app?.S.ready && window.__app.S.learnReady',timeout=60000)
        binding.wait_for_complete(page)
        initial=page.evaluate(READ)
        assert initial['brain']['n']==initial['brain']['whole']['neurons']==150802
        assert initial['brain']['edges']==initial['brain']['whole']['edges']==1877099
        page.wait_for_function('t=>window.__app.S.simTime>=t+2 && window.__app.life().authority.observations.framesApplied>0',arg=initial['time'],timeout=90000)
        active=page.evaluate(READ)
        assert active['mode']=='assisted'
        assert sum((active['p'][k]-initial['p'][k])**2 for k in [0,1]) > .1**2
        assert active['p'][2]>.5
        assert active['authority']['observations']['accepted']>0
        assert not active['authority']['autonomous']
        checks.append('default full body travels horizontally above floor; checked actual-payload trim is used')
        assert 'Pilot + scripts' in page.locator('#authority').inner_text()
        assert 'not neural input' in page.locator('#eye .tag').inner_text()
        if screenshot: page.screenshot(path=screenshot)
        page.keyboard.press('Space')
        paused=page.evaluate('window.__app.flight().p.slice()')
        page.wait_for_timeout(1000)
        assert page.evaluate('window.__app.flight().p.slice()')==paused
        assert page.evaluate('Object.values(window.__app.life().authority.neuralTrim).every(x=>x===0)')
        checks.append('pause holds physical position and revokes neural trim')
        page.keyboard.press('Space')
        page.get_by_role('button',name='brain off',exact=True).click()
        page.wait_for_function('window.__app.S.simTime>=1',timeout=60000)
        off=page.evaluate(READ)
        assert off['mode']=='assisted-off' and off['p'][2]>.5
        assert off['authority']['observations']['requested']==0
        assert off['authority']['choices']['executed']==0
        assert all(x==0 for x in off['authority']['neuralTrim'].values())
        assert off['controls']['f']>0
        checks.append('brain-off keeps explicitly supplied flight but no neural observations, trim, decisions or updates')
        page.get_by_role('button',name='assisted + brain',exact=True).click()
        page.wait_for_function('window.__app.S.ready && window.__app.S.learnReady',timeout=60000)
        page.wait_for_function('window.__app.life().authority.observations.framesApplied>0',timeout=60000)
        page.evaluate("window.__app.worker.onmessageerror(new MessageEvent('messageerror'))")
        before=page.evaluate(READ)
        page.wait_for_function('t=>window.__app.S.simTime>t+.3',arg=before['time'],timeout=30000)
        failure=page.evaluate(READ)
        assert all(x==0 for x in failure['authority']['neuralTrim'].values())
        assert failure['controls']['f']>0
        checks.append('transport failure disables neural contribution while visible assistance continues')
        assert not errors,errors
        served_sources=binding.receipt()
        browser.close()
    assert source_hashes()==frozen_sources, "source changed during browser run"
    return {'full_views':full,'checks':checks,'initial':initial,'active':active,'brain_off':off,'worker_failure':failure,'page_errors':errors,
      'claim':'Working assisted flight and real checked motor trim; no claim of autonomous flight or learned room navigation.',
      'sha256':frozen_sources,'served_sources':served_sources}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--url',default='http://127.0.0.1:8817/');parser.add_argument('--full',action='store_true');parser.add_argument('--receipt');parser.add_argument('--screenshot')
    args=parser.parse_args();result=run(args.url,args.full,args.screenshot)
    if args.receipt:
        dest=Path(args.receipt)
        with dest.open('x') as out:json.dump(result,out,indent=2);out.write('\n')
    print(json.dumps(result,indent=2))
