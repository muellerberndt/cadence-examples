#!/usr/bin/env python3
"""Natural-scene neural goal selection and assisted body execution assay."""
import argparse
import hashlib
import json
import math
import traceback
from datetime import datetime, timezone
from pathlib import Path
from playwright.sync_api import sync_playwright
from browser_source_binding import ServedSourceBinding, source_hashes

ROOT=Path(__file__).resolve().parents[1]
RUNNER='tools/check_hybrid_restore.py'
RECORD="""(()=>{window.__hybridPackets=[];window.__hybridRequests=[];
 const send=Worker.prototype.postMessage;Worker.prototype.postMessage=function(m,...rest){
  if(m.type?.startsWith('assist:'))__hybridRequests.push({type:m.type,requestId:m.requestId,generation:m.generation,searchToken:m.searchToken,
   steps:m.steps,tolerance:m.tolerance,fruit:m.fruit,reward:m.reward,neuralCredit:m.neuralCredit,why:m.why,time:window.__app?.life().clock,
   execution:m.type==='assist:reward'?structuredClone(window.__app?.life().authority.lastChoice):undefined});return send.call(this,m,...rest);};
 const Native=Worker;window.Worker=new Proxy(Native,{construct(T,args){const w=new T(...args);w.addEventListener('message',e=>{const m=e.data;
  if(['assisted_decision','lesson'].includes(m.type))__hybridPackets.push({type:m.type,requestId:m.requestId,generation:m.generation,
   time:window.__app?.life().clock,searchToken:m.searchToken,accepted:m.accepted,decision:m.decision,residual:m.residual,tolerance:m.tolerance,
   iterations:m.iterations,reason:m.reason,targetSelection:m.targetSelection,readouts:m.type==='assisted_decision'?m.readouts:undefined,reward:m.reward,updates:m.updates,changed:m.changed,moved:m.moved,delta:m.delta});});return w;}});})();"""
OBSERVE="""async()=>{const sample=()=>{const a=__app,L=a.life(),f=a.flight();return{wall:performance.now(),time:L.clock,mode:L.mode,
 hunger:L.hunger,groom:L.groomTarget,position:f.p.slice(),velocity:f.v.slice(),touching:f.touching,onFloor:f.onFloor,fruit:L.onWhich??null,contactFruit:L.fruitAt(f.p),
 authority:structuredClone(L.authority),lesson:structuredClone(a.S.lessonStats),events:structuredClone(L.events),fps:a.S.fps};};
 const rows=[sample()],start=rows[0],deadline=performance.now()+120000;
 while(performance.now()<deadline&&rows.at(-1).time-start.time<80){await new Promise(r=>setTimeout(r,500));const s=sample();rows.push(s);
  const behavior=rows.some(x=>x.mode==='grooming')&&rows.some(x=>x.mode==='landed')&&s.mode==='flying';
  const learned=__hybridPackets.some(x=>x.type==='lesson'&&x.accepted&&x.updates>0&&x.moved>0);
  if(behavior&&learned&&s.authority.choices.executed>0)return{rows,stop:'natural_behavior_and_real_weight_update'};
 }return{rows,stop:'bounded_time_limit'};}"""


def run(args):
    pins=source_hashes(ROOT,RUNNER)
    r={'schema':'cadence.neural-goal-body-browser/1','passed':False,'started':datetime.now(timezone.utc).isoformat(),
       'sha256':pins,'errors':[],'protocol':{'url':args.url,'viewport':[1440,1000],'maxSimSeconds':80,'maxWallSeconds':120,'sugarCommand':args.sugar,'hardwareRenderer':args.hardware,
       'scope':'Default scene and seed; optional sugar command changes the reward location through the ordinary UI. No pose/readout/reward injection. Settled neural target-selection and approach/avoid choose the goal; body routes, landing and grooming are supplied execution. Learning is reported only when an actual matched lesson moves weights; no learned-navigation or learned-avoidance claim.'}}
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,channel='chromium') if args.hardware else p.chromium.launch(headless=True,args=['--use-angle=swiftshader','--enable-unsafe-swiftshader'])
        page=browser.new_page(viewport={'width':1440,'height':1000});r['browser']=browser.version
        binding=ServedSourceBinding(page.context,args.url,pins);page.on('pageerror',lambda e:r['errors'].append(str(e)));page.add_init_script(RECORD)
        try:
            page.goto(args.url,wait_until='domcontentloaded');page.wait_for_function('window.__app?.S.ready&&__app.S.learnReady',timeout=90000)
            r['gpu']=page.evaluate("()=>{const g=__app.renderer.getContext(),e=g.getExtension('WEBGL_debug_renderer_info');return e?g.getParameter(e.UNMASKED_RENDERER_WEBGL):g.getParameter(g.RENDERER)}");r['brain']=page.evaluate('__app.brainInfo()');assert r['brain']['n']==150802 and r['brain']['edges']==1877099
            r['visibleText']=page.locator('body').inner_text();assert 'brain chooses goals' in r['visibleText'].lower() and 'body executes' in r['visibleText'].lower(), 'Default must disclose brain goals and body execution'
            if args.sugar:
                page.locator('#console-input').fill('/sugar '+args.sugar);page.locator('#console-input').press('Enter')
                page.wait_for_function('(name)=>__app.life().sugar===name',arg=args.sugar)
            r['observation']=page.evaluate(OBSERVE)
            r['packets']=page.evaluate('__hybridPackets');r['requests']=page.evaluate('__hybridRequests');r['console']=page.evaluate('__app.console.snapshot()')
            r['bodyModes']=sorted({x['mode'] for x in r['observation']['rows']})
            r['acceptedNeuralDecisions']=sum(m['type']=='assisted_decision' and m.get('accepted') is True for m in r['packets'])
            r['observedLocalWeightUpdate']=any(m['type']=='lesson' and m.get('accepted') and m.get('moved',0)>0 for m in r['packets'])
            page.screenshot(path=str(args.screenshot));r['image']={'path':str(args.screenshot.resolve().relative_to(ROOT)),'sha256':hashlib.sha256(args.screenshot.read_bytes()).hexdigest()}
            page.set_viewport_size({'width':430,'height':900})
            page.wait_for_function("['brain','eye','brain-console'].every(id=>{const b=document.getElementById(id).getBoundingClientRect();return b.width>0&&b.height>0&&b.left>=0&&b.top>=0&&b.right<=innerWidth&&b.bottom<=innerHeight})",timeout=10000)
            r['mobileLayout']=page.evaluate("Object.fromEntries(['brain','eye','brain-console'].map(id=>{const b=document.getElementById(id).getBoundingClientRect();return [id,{x:b.x,y:b.y,width:b.width,height:b.height}]}))")
            page.screenshot(path=str(args.mobile));r['mobileImage']={'path':str(args.mobile.resolve().relative_to(ROOT)),'sha256':hashlib.sha256(args.mobile.read_bytes()).hexdigest()}
            r['served']=binding.receipt();assert not r['errors'] and source_hashes(ROOT,RUNNER)==pins
            rows=r['observation']['rows'];choices=[x['authority']['lastChoice'] for x in rows if x['authority'].get('lastChoice') and x['authority']['lastChoice'].get('executed')]
            assert choices,'No executed neural goal'
            selected_landings=[x for x in rows if x['mode'] in ('landed','grooming','feeding') and x['contactFruit'] and x['authority'].get('lastChoice',{}).get('executed') and x['authority']['lastChoice']['action']==0 and x['authority']['lastChoice']['fruit']==x['contactFruit']]
            r['selectedGoalLandingSamples']=len(selected_landings)
            assert selected_landings,'No natural landing on the executed brain-selected fruit observed'
            for choice in choices:
                d=next(m for m in r['packets'] if m['type']=='assisted_decision' and all(m.get(k)==choice.get(k) for k in ('requestId','generation','searchToken')))
                assert d['accepted'] and d['decision']['accepted']
                request=next(q for q in r['requests'] if q['type']=='assist:decide' and all(q.get(k)==d.get(k) for k in ('requestId','generation','searchToken')))
                assert all(v['iterations']<=request['steps'] and v['tolerance']<=request['tolerance'] for v in d['decision']['solves'].values())
                assert all(c['solve']['iterations']<=request['steps'] for c in d['targetSelection']['candidates'])
                assert d['targetSelection']['fruit']==choice['fruit'] and d['decision']['action']==choice['action']
                for phase in ('free','plus','minus'):
                    q=d['decision']['solves'][phase];assert q['converged'] and 0<=q['residual']<=q['tolerance']<=1e-6 and 0<=q['iterations']<=1024
                for candidate in d['targetSelection']['candidates']:
                    q=candidate['solve'];assert q['converged'] and 0<=q['residual']<=q['tolerance']<=1e-6 and 0<=q['iterations']<=1024
                    reads=candidate['readouts'];assert abs(candidate['score']-(reads['mbon:MBON11:right']-reads['mbon:MBON05:left']))<1e-12
                target=d['targetSelection'];assert [c['fruit'] for c in target['candidates']]==['banana','bread']
                scores=[c['score']/target['temperature'] for c in target['candidates']]
                weights=[math.exp(x-max(scores)) for x in scores];p0=weights[0]/sum(weights)
                assert abs(target['p'][0]-p0)<1e-12 and 0<=target['draw']<1
                assert target['fruit']==target['candidates'][0 if target['draw']<=p0 else 1]['fruit']
                action=d['decision'];logits=[d['readouts'][name]/target['temperature'] for name in ['mbon:MBON11:right','mbon:MBON05:left']]
                w=[math.exp(x-max(logits)) for x in logits];expected=w[0]/sum(w)
                assert abs(action['p'][0]-expected)<1e-10 and abs(sum(action['p'])-1)<1e-12
                assert 0<=action['draw']<1 and action['action']==(0 if action['draw']<=action['p'][0] else 1)
            lessons=[m for m in r['packets'] if m['type']=='lesson' and m.get('accepted') and m.get('moved',0)>0]
            r['observedLocalWeightUpdate']=bool(lessons)
            for lesson in lessons:
                same=lambda x:all(x.get(k)==lesson.get(k) for k in ('requestId','generation','searchToken'))
                decision=next(m for m in r['packets'] if m['type']=='assisted_decision' and same(m))
                assert decision['accepted'] and decision['decision']['accepted']
                for phase in ('free','plus','minus'):
                    q=decision['decision']['solves'][phase];assert q['converged'] and 0<=q['residual']<=q['tolerance']<=1e-6 and 0<=q['iterations']<=1024
                reward=next(m for m in r['requests'] if m['type']=='assist:reward' and same(m) and m['neuralCredit'] is True)
                execution=reward['execution'];assert same(execution) and execution['executed']
                assert execution['fruit']==decision['targetSelection']['fruit']
                assert decision['time']<=execution['executedAt']<=reward['time']<=lesson['time']
                choices=[x['authority']['lastChoice'] for x in r['observation']['rows'] if x['authority'].get('lastChoice')]
                assert any(same(c) and c.get('executed') and c['action']==decision['decision']['action'] for c in choices)
            if args.sugar:assert r['observedLocalWeightUpdate'],'Rewarded observation did not move any weights'
            r['passed']=True
        except Exception as e:r['error']=repr(e);r['traceback']=traceback.format_exc()
        finally:r['sourcesUnchanged']=source_hashes(ROOT,RUNNER)==pins;browser.close()
    r['completed']=datetime.now(timezone.utc).isoformat();r['body_sha256']=hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return r


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--url',default='http://127.0.0.1:8817/')
    p.add_argument('--hardware',action='store_true');p.add_argument('--sugar',choices=['banana','bread']);p.add_argument('--receipt',type=Path,required=True);p.add_argument('--screenshot',type=Path,required=True);p.add_argument('--mobile',type=Path,required=True);args=p.parse_args()
    for path in (args.receipt,args.screenshot,args.mobile):
        if path.exists():p.error(f'refusing to overwrite {path}')
    r=run(args);args.receipt.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:r.get(k) for k in ('passed','error','body_sha256')}|{'stop':r.get('observation',{}).get('stop')},indent=2))
    raise SystemExit(0 if r['passed'] else 1)
