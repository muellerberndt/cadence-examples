#!/usr/bin/env python3
"""Bounded unchanged-scene wall approach/recovery probe; preserve every outcome."""
import argparse
import hashlib
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from playwright.sync_api import sync_playwright
from browser_source_binding import ServedSourceBinding, source_hashes

ROOT=Path(__file__).resolve().parents[1]
RUNNER='tools/check_wall_recovery.py'
RECORD="""(()=>{window.__episodePackets=[];window.__episodeRequests=[];
 const send=Worker.prototype.postMessage;Worker.prototype.postMessage=function(m,...rest){
  if(m.type?.startsWith('assist:'))__episodeRequests.push({type:m.type,requestId:m.requestId,generation:m.generation,searchToken:m.searchToken,attention:m.attention,reward:m.reward,neuralCredit:m.neuralCredit,why:m.why});
  return send.call(this,m,...rest);};const Native=Worker;window.Worker=new Proxy(Native,{construct(T,args){const w=new T(...args);
  w.addEventListener('message',e=>{const m=e.data;if(!['control','assisted_decision','lesson'].includes(m.type))return;
   __episodePackets.push({type:m.type,requestId:m.requestId,generation:m.generation,searchToken:m.searchToken,attention:m.attention,
   converged:m.converged,residual:m.residual,tolerance:m.tolerance,iterations:m.iterations,accepted:m.accepted,
   decision:m.decision,targetSelection:m.targetSelection,readouts:m.type==='assisted_decision'?m.readouts:undefined,
   updates:m.updates,changed:m.changed,moved:m.moved,delta:m.delta,reason:m.reason});});return w;}});})();"""
PROBE="""async expect=>{
 const sample=()=>{const a=__app,f=a.flight(),L=a.life(),p=f.p,room=[4,3,2.6],radius=.0012;
  return {wall:performance.now(),time:L.clock,position:p.slice(),velocity:f.v.slice(),speed:Math.hypot(...f.v),
   horizontalSpeed:Math.hypot(...f.v.slice(0,2)),touching:f.touching,onFloor:f.onFloor,
   clearances:[p[0]-radius,room[0]-p[0]-radius,p[1]-radius,room[1]-p[1]-radius,p[2]-radius,room[2]-p[2]-radius],
   support:structuredClone(L.authority.motionSupport),navigation:structuredClone(L.authority.navigation),
   choices:structuredClone(L.authority.choices),lastChoice:structuredClone(L.authority.lastChoice??null),
   execution:structuredClone(L.authority.execution??L.authority.causalExecution??null),
   learning:__app.states().learningUpdates,changedConnections:__app.states().changedConnections,
   pendingReward:structuredClone(L.pendingReward??null),
   observations:structuredClone(L.authority.observations),boundary:structuredClone(L.authority.boundaryAssistance??null),
   pending:a.S.pendingId,solveMs:a.S.solveMs,controlReplies:a.S.controlReplies,paused:a.S.paused,fps:a.S.fps};};
 const rows=[sample()],start=rows[0],deadline=performance.now()+70000;let stalledSince=null,approach=null,recovery=null,executedAt=null;
 while(performance.now()<deadline&&rows.at(-1).time-start.time<40){
  await new Promise(r=>setTimeout(r,250));const s=sample();rows.push(s);
  if(expect==='attention'&&s.choices.accepted>0&&s.choices.executed>0)executedAt??=s.time;
  if(executedAt!==null&&s.time>=executedAt+2)return{rows,approach,recovery,stalledSince,executedAt,stop:'checked_choice_physically_executed'};
  const near=s.clearances.slice(0,4).map((d,i)=>({d,i})).sort((a,b)=>a.d-b.d)[0];
  if(!approach&&near.d<=.25)approach={time:s.time,index:rows.length-1,wall:near.i,clearance:near.d};
  const desired=s.support?.desiredVelocityWorld??[0,0,0],axis=Math.floor(near.i/2),outward=(near.i%2?1:-1)*desired[axis];
  if(near.d<.02&&s.horizontalSpeed<.03&&outward>.05)stalledSince??=s.time;else stalledSince=null;
  if(expect!=='attention'&&stalledSince!==null&&s.time-stalledSince>=3)return{rows,approach,recovery,stalledSince,stop:'sustained_wall_stall'};
  if(approach&&!recovery&&s.time>approach.time+.5&&s.clearances[approach.wall]>=.4&&s.horizontalSpeed>.1)
   recovery={time:s.time,index:rows.length-1,clearance:s.clearances[approach.wall],seconds:s.time-approach.time};
  if(expect==='recovery'&&recovery&&s.time>=recovery.time+2)return{rows,approach,recovery,stalledSince,stop:'recovered_and_observed_two_seconds'};
 }
 return{rows,approach,recovery,stalledSince,stop:'bounded_time_limit'};
}"""


def run(args):
    pins=source_hashes(ROOT,RUNNER)
    r={'schema':'cadence.wall-recovery-browser/1','passed':False,'started':datetime.now(timezone.utc).isoformat(),
       'sha256':pins,'expect':args.expect,'errors':[],'protocol':{'url':args.url,'viewport':[1440,1000],
       'maxWallSeconds':70,'maxSimSeconds':40,'sampleMilliseconds':250,'roomMetres':[4,3,2.6],'bodyRadiusMetres':.0012,
       'approach':'First horizontal room-plane clearance <=0.25m.',
       'stall':'Clearance<0.02m, horizontal body speed<0.03m/s, support desired velocity into nearest wall>0.05m/s for>=3simsec.',
       'recovery':'After approach, distance from that same plane>=0.4m and actual horizontal speed>0.1m/s, then observe2more simsec; recovery must occur within12simsec.',
       'attention':'At least one real accepted and physically executed choice, then2more simsec. All decision phases joined to actual worker packets; learning counters recorded without inducing reward.',
       'scope':'Natural plain-URL trajectory; no pose, readout, neural state or route injection. Supplied assistance is recorded separately. Attention/execution success is not learned wall avoidance.'}}
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--use-angle=swiftshader','--enable-unsafe-swiftshader'])
        page=browser.new_page(viewport={'width':1440,'height':1000});r['browser']=browser.version
        binding=ServedSourceBinding(page.context,args.url,pins);page.on('pageerror',lambda e:r['errors'].append(str(e)))
        page.add_init_script(RECORD)
        try:
            page.goto(args.url,wait_until='domcontentloaded')
            page.wait_for_function('window.__app?.S.ready&&__app.S.learnReady',timeout=90000)
            r['probe']=page.evaluate(PROBE,args.expect)
            page.screenshot(path=str(args.screenshot));r['image']={'path':str(args.screenshot.resolve().relative_to(ROOT)),'sha256':hashlib.sha256(args.screenshot.read_bytes()).hexdigest()}
            r['console']=page.evaluate('__app.console.snapshot()');r['served']=binding.receipt()
            r['packets']=page.evaluate('__episodePackets');r['requests']=page.evaluate('__episodeRequests')
            assert not r['errors'] and source_hashes(ROOT,RUNNER)==pins
            if args.expect=='stall':assert r['probe']['stop']=='sustained_wall_stall',r['probe']['stop']
            elif args.expect=='recovery':
                assert r['probe']['stop']=='recovered_and_observed_two_seconds',r['probe']['stop']
                assert r['probe']['recovery']['seconds']<=12,r['probe']['recovery']
            else:
                assert r['probe']['stop']=='checked_choice_physically_executed',r['probe']['stop']
                choices=[x['lastChoice'] for x in r['probe']['rows'] if x['lastChoice'] and x['lastChoice'].get('executed')]
                assert choices,'No physical execution record'
                for c in choices:
                    packet=next(m for m in r['packets'] if m['type']=='assisted_decision' and m.get('requestId')==c['requestId'] and m.get('generation')==c['generation'] and m.get('searchToken')==c['searchToken'])
                    assert packet['accepted'] and packet['decision']['accepted']
                    for phase in ('free','plus','minus'):
                        solve=packet['decision']['solves'][phase];assert solve['converged'] and 0<=solve['residual']<=solve['tolerance']<=1e-6
                    assert packet['targetSelection']['fruit']==c['fruit'] and packet['decision']['action']==c['action']
            r['passed']=True
        except Exception as e:
            r['error']=repr(e);r['traceback']=traceback.format_exc()
        finally:
            r['sourcesUnchanged']=source_hashes(ROOT,RUNNER)==pins;browser.close()
    r['completed']=datetime.now(timezone.utc).isoformat()
    r['body_sha256']=hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return r


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--url',default='http://127.0.0.1:8817/')
    p.add_argument('--expect',choices=['stall','recovery','attention'],required=True)
    p.add_argument('--receipt',type=Path,required=True);p.add_argument('--screenshot',type=Path,required=True)
    args=p.parse_args()
    for path in (args.receipt,args.screenshot):
        if path.exists():p.error(f'refusing to overwrite {path}')
    r=run(args);args.receipt.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:r.get(k) for k in ('passed','expect','error','body_sha256')}|{'stop':r.get('probe',{}).get('stop')},indent=2))
    raise SystemExit(0 if r['passed'] else 1)
