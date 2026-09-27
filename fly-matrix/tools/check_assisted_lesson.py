#!/usr/bin/env python3
"""Check one real page decision→assisted landing→sugar→local update.

An explicit near-bread initial condition makes the assay bounded. No worker choice,
phase, reward or learner state is fabricated. This does not test learned navigation.
"""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
from browser_source_binding import ServedSourceBinding, source_hashes as freeze_sources
ROOT=Path(__file__).resolve().parents[1]

def source_hashes():
    return freeze_sources(ROOT, 'tools/check_assisted_lesson.py')

def run(url):
    frozen_sources=source_hashes()
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--use-angle=swiftshader','--enable-unsafe-swiftshader'])
        page=browser.new_page(viewport={'width':1200,'height':900});errors=[]
        binding=ServedSourceBinding(page.context,url,frozen_sources)
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(url+'?noscan=1&nobloom=1&noviews=1&seed=1',wait_until='domcontentloaded')
        page.wait_for_function('window.__app?.S.ready && window.__app.S.learnReady',timeout=60000)
        binding.wait_for_complete(page)
        initial=page.evaluate('''() => {
          const a=window.__app,L=a.life(),f=a.flight(),F=a.room.fruits.bread.pos;
          window.__assayMessages=[];
          a.worker.addEventListener('message',e=>{const m=e.data;
            if (['assisted_decision','lesson'].includes(m.type)) window.__assayMessages.push({
              type:m.type,requestId:m.requestId,generation:m.generation,searchToken:m.searchToken,
              accepted:m.accepted,decision:m.decision,residual:m.residual,tolerance:m.tolerance,
              iterations:m.iterations,reason:m.reason,updates:m.updates,reward:m.reward,
              simulationTime:L.clock, readouts:m.type==='assisted_decision'?{
                approach:m.readouts?.['mbon:MBON11:right'],avoid:m.readouts?.['mbon:MBON05:left']}:undefined});
          });
          a.setSugar('bread');f.p=[F[0]-.035,F[1],F[2]+.085];f.v=[0,0,0];f.w=[0,0,0];f.q=[1,0,0,0];
          L.heading=0;L.altitude=F[2]+.085;L.bout=100;L.search=null;L.valence=null;L.senses(null,0);
          return {brain:a.brainInfo(),p:f.p.slice(),time:L.clock,bread:F.slice(),seed:a.S.seed,sugar:L.sugar};
        }''')
        assert initial['brain']['n']==initial['brain']['whole']['neurons']==150802
        assert initial['brain']['edges']==initial['brain']['whole']['edges']==1877099
        page.wait_for_function('window.__app.S.lessonStats?.updates>=1',timeout=60000)
        result=page.evaluate('''() => {const a=window.__app;return {
          time:a.S.simTime,authority:a.life().authority,p:a.flight().p,events:a.life().events,
          lesson:a.S.lessonStats,mode:a.life().mode,messages:window.__assayMessages}}''')
        choice=result['authority']['lastChoice'];lesson=result['lesson']
        decisions=[m for m in result['messages'] if m['type']=='assisted_decision' and m['accepted']]
        assert len(decisions)==1,result
        decision=decisions[0]
        for key in ['requestId','generation','searchToken']:
            assert decision[key]==choice[key]==lesson[key],key
        assert decision['decision']['action']==choice['action']==0
        assert choice['executed'] and choice['fruit']=='bread'
        assert all(s['converged'] and s['residual']<=s['tolerance']<=1e-6 and s['iterations']<=1024
                   for s in decision['decision']['solves'].values())
        sugar=[t for t,e in result['events'] if e=='lands on the bread: sugar']
        assert len(sugar)==1 and choice['executedAt']<sugar[0]
        assert lesson['accepted'] and lesson['reward']==1 and lesson['updates']==1 and lesson['changed']>0
        assert result['authority']['choices']['creditedOutcomes']==1
        assert not errors,errors
        served_sources=binding.receipt()
        browser.close()
    assert source_hashes()==frozen_sources, 'source changed during browser run'
    return {'initial_condition':initial,'result':result,'page_errors':errors,
      'claim':'A real checked MBON choice was used before assisted sugar contact; only its matching later outcome updated local synapses. Landing, navigation and feeding remain supplied assistance.',
      'limits':'One seed and one near-food initial condition; reduced rendering cost; not a general learning-performance or autonomous-navigation result.',
      'sha256':frozen_sources,'served_sources':served_sources}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--url',default='http://127.0.0.1:8817/');parser.add_argument('--receipt');args=parser.parse_args()
    result=run(args.url)
    if args.receipt:
        with Path(args.receipt).open('x') as out:json.dump(result,out,indent=2);out.write('\n')
    print(json.dumps({'claim':result['claim'],'choice':result['result']['authority']['lastChoice'],'lesson':result['result']['lesson'],'errors':result['page_errors']},indent=2))
