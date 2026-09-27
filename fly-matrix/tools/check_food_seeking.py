#!/usr/bin/env python3
"""Prospective matched initial-hunger browser assay; numerical success is not food competence."""
import argparse
import hashlib
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, urlsplit, urlunsplit, parse_qsl
from playwright.sync_api import sync_playwright
from browser_source_binding import ServedSourceBinding, source_hashes

ROOT = Path(__file__).resolve().parents[1]
RUNNER = 'tools/check_food_seeking.py'
SCRIPT = r"""(()=>{
 const setup=SETUP;
 let randomState=setup.bodySceneSeed;
 Math.random=()=>{let t=randomState+=0x6D2B79F5;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return((t^t>>>14)>>>0)/4294967296;};
 const log=window.__foodEvidence={setup,initial:null,requests:[],packets:[],learner:null};
 let app=null;
 Object.defineProperty(window,'__app',{configurable:true,get:()=>app,set:value=>{
  app=value;if(log.initial)return;
  const l=app.life();log.initial={time:l.clock,position:app.flight().p.slice(),beforeHunger:l.hunger,
   assignedHunger:setup.hunger,modelSeed:app.S.seed};l.hunger=setup.hunger;
 }});
 const W=Worker,send=W.prototype.postMessage;
 W.prototype.postMessage=function(m,...rest){
  if(m.type==='learn:init'){
   log.learner={original:structuredClone(m.config),seed:m.seed};
   if(setup.learning==='frozen')m={...m,config:{...m.config,eta:0,etaBias:0,etaCritic:0}};
   log.learner.actual=structuredClone(m.config);
  }
  if(m.type?.startsWith('assist:'))log.requests.push({type:m.type,requestId:m.requestId,generation:m.generation,
   searchToken:m.searchToken,time:app?.life().clock,hunger:app?.life().hunger,steps:m.steps,tolerance:m.tolerance,
   reward:m.reward,neuralCredit:m.neuralCredit,why:m.why,
   execution:m.type==='assist:reward'?structuredClone(app?.life().authority.lastChoice):undefined});
  return send.call(this,m,...rest);
 };
 window.Worker=new Proxy(W,{construct(T,args){const w=new T(...args);w.addEventListener('message',e=>{const m=e.data;
  if(['control','assisted_decision','lesson'].includes(m.type))log.packets.push({type:m.type,requestId:m.requestId,
   generation:m.generation,searchToken:m.searchToken,time:app?.life().clock,accepted:m.accepted,converged:m.converged,
   iterations:m.iterations,residual:m.residual,tolerance:m.tolerance,reason:m.reason,decision:m.decision,
   targetSelection:m.targetSelection,readouts:m.type==='assisted_decision'?m.readouts:undefined,
   updates:m.updates,moved:m.moved,changed:m.changed,delta:m.delta,reward:m.reward});
 });return w;}});
 window.__foodSample=()=>{const l=app.life(),f=app.flight();return{wall:performance.now(),time:l.clock,
  hunger:l.hunger,mode:l.mode,position:f.p.slice(),velocity:f.v.slice(),sugar:l.sugar,contactFruit:l.fruitAt(f.p),
  feedingContact:l.feedingContact(),perch:l.perch?.slice()??null,visits:structuredClone(l.visits),
  ethogram:{feeds:l.ethogram.feeds,grooms:l.ethogram.grooms,landings:l.ethogram.landings},
  choices:structuredClone(l.authority.choices),observations:structuredClone(l.authority.observations),
  lastChoice:structuredClone(l.authority.lastChoice),goal:structuredClone(l.authority.goal),
  lesson:structuredClone(app.S.lessonStats),workerBusy:app.S.workerBusy,workerJobId:app.S.workerJobId,fps:app.S.fps};};
})();"""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(obj):
    return tuple(obj.get(k) for k in ('generation', 'requestId', 'searchToken'))


def summarize(arm):
    rows = arm['samples']; packets = arm['evidence']['packets']; requests = arm['evidence']['requests']
    decisions = [p for p in packets if p['type'] == 'assisted_decision']
    proposals = {key(p): p for p in decisions}
    accepted = {key(q) for q in requests if q['type'] == 'assist:accept'}
    executions = {key(x['lastChoice']): x['lastChoice'] for x in rows if (x.get('lastChoice') or {}).get('executed')}
    lessons = [p for p in packets if p['type'] == 'lesson' and p.get('accepted')]
    for execution in executions.values():
        ident = key(execution); assert ident in accepted and ident in proposals
        p = proposals[ident]; assert p['accepted'] and p['decision']['accepted']
        assert p['decision']['action'] == execution['action'] and p['targetSelection']['fruit'] == execution['fruit']
        q = next(q for q in requests if q['type'] == 'assist:decide' and key(q) == ident)
        phases = [p['decision']['solves'][k] for k in ('free', 'plus', 'minus')]
        phases += [c['solve'] for c in p['targetSelection']['candidates']]
        assert all(s['converged'] and 0 <= s['residual'] <= s['tolerance'] <= q['tolerance']
                   and 0 <= s['iterations'] <= q['steps'] for s in phases)
    for lesson in lessons:
        ident = key(lesson)
        q = next(q for q in requests if q['type'] == 'assist:reward' and key(q) == ident)
        assert q['neuralCredit'] is True and q['execution']['executed'] and key(q['execution']) == ident
        assert ident in accepted and proposals[ident]['accepted']
    if arm['setup']['learning'] == 'frozen':
        assert all(arm['evidence']['learner']['actual'][name] == 0 for name in ('eta', 'etaBias', 'etaCritic'))
        assert all(p.get('moved', 0) == 0 and p.get('changed', 0) == 0 for p in lessons), 'Frozen learning moved weights'
    onset = next((x for x in rows if x['mode'] == 'feeding' and x['feedingContact']), None)
    caloric_drop = sum(max(0, a['hunger'] - b['hunger']) for a, b in zip(rows, rows[1:]))
    feeding_seconds = sum(b['time'] - a['time'] for a, b in zip(rows, rows[1:]) if a['mode'] == b['mode'] == 'feeding')
    first_eligible = next((x for x in rows if x['hunger'] > .2), None)
    first_learning = next((p for p in lessons if p.get('moved', 0) > 0), None)
    assert all(x['sugar'] == 'banana' for x in rows), 'Default reward location changed'
    assert all(0 <= x['hunger'] <= 1 for x in rows)
    controls = [p for p in packets if p['type'] == 'control']
    return {'qualifiedDecisions': sum(p.get('accepted') is True for p in decisions), 'admittedDecisions': len(accepted),
            'executedDecisions': len(executions), 'executedGoals': list(executions.values()),
            'failedDecisions': [{'requestId': p['requestId'], 'reason': p.get('reason'), 'decision': p.get('decision'),
                                 'targetSelection': p.get('targetSelection')} for p in decisions if not p.get('accepted')],
            'observationReplies': len(controls), 'qualifiedObservations': sum(p.get('converged') is True for p in controls),
            'failedObservations': [p for p in controls if not p.get('converged')],
            'admittedObservations': rows[-1]['observations']['accepted'], 'visits': rows[-1]['visits'],
            'feedingBouts': rows[-1]['ethogram']['feeds'], 'feedingObservedSeconds': feeding_seconds,
            'firstFeedingTime': onset['time'] if onset else None, 'observedHungerDecrease': caloric_drop,
            'initialHunger': arm['setup']['hunger'], 'finalHunger': rows[-1]['hunger'],
            'firstHungerAboveThresholdTime': first_eligible['time'] if first_eligible else None,
            'acceptedLessons': len(lessons), 'nonzeroLessons': sum(p.get('moved', 0) > 0 for p in lessons),
            'firstNonzeroLessonTime': first_learning['time'] if first_learning else None,
            'fedBeforeAnyNonzeroLesson': bool(onset and (not first_learning or onset['time'] < first_learning['time'])),
            'foodSeekingObserved': bool(onset and caloric_drop > .01 and rows[-1]['visits']['sweet'] > 0),
            'bodyModes': sorted({x['mode'] for x in rows}),
            'meanSampledFPS': sum(x['fps'] for x in rows) / len(rows),
            'simulatedSecondsObserved': rows[-1]['time'] - rows[0]['time']}


def run_arm(playwright, args, seed, hunger, pins):
    setup = {'modelSeed': seed, 'bodySceneSeed': seed, 'hunger': hunger, 'learning': args.learning}
    arm = {'setup': setup, 'integrityPassed': False, 'errors': [], 'samples': []}
    parts = urlsplit(args.url); query = dict(parse_qsl(parts.query)); query['seed'] = str(seed)
    url = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)); arm['url'] = url
    browser = playwright.chromium.launch(headless=True, channel='chromium')
    page = browser.new_page(viewport={'width': 1440, 'height': 1000})
    binding = ServedSourceBinding(page.context, url, pins)
    page.on('pageerror', lambda e: arm['errors'].append(str(e)))
    page.add_init_script(SCRIPT.replace('SETUP', json.dumps(setup)))
    try:
        page.goto(url, wait_until='domcontentloaded')
        page.wait_for_function('window.__app?.S.ready&&__app.S.learnReady', timeout=90000)
        initial = page.evaluate('__foodEvidence.initial'); arm['initialAssignment'] = initial
        assert initial['time'] == 0 and initial['modelSeed'] == seed and initial['assignedHunger'] == hunger
        arm['gpu'] = page.evaluate("()=>{const g=__app.renderer.getContext(),e=g.getExtension('WEBGL_debug_renderer_info');return e?g.getParameter(e.UNMASKED_RENDERER_WEBGL):g.getParameter(g.RENDERER)}")
        assert 'SwiftShader' not in arm['gpu']
        started = time.monotonic(); start = page.evaluate('__foodSample()'); arm['samples'].append(start)
        while time.monotonic() - started < args.wall_seconds and arm['samples'][-1]['time'] < args.sim_seconds:
            page.wait_for_timeout(250); arm['samples'].append(page.evaluate('__foodSample()'))
        arm['wallSeconds'] = time.monotonic() - started
        arm['evidence'] = page.evaluate('__foodEvidence')
        arm['console'] = page.evaluate('__app.console.snapshot()')
        assert arm['evidence']['learner']['seed'] == seed
        assert all(q.get('steps', 256) == 256 for q in arm['evidence']['requests'] if q['type'] in ('assist:observe', 'assist:decide'))
        arm['summary'] = summarize(arm)
        arm['served'] = binding.receipt()
        assert not arm['errors'] and source_hashes(ROOT, RUNNER) == pins
        arm['integrityPassed'] = True
    except Exception as error:
        arm['error'] = repr(error); arm['traceback'] = traceback.format_exc()
        if 'evidence' not in arm:
            try:
                arm['evidence'] = page.evaluate('__foodEvidence')
            except Exception:
                pass
    finally:
        browser.close()
    return arm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8817/')
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--seeds', default='1', help='Prospectively selected seeds, e.g.1,2,3')
    parser.add_argument('--hungers', default='1,0', help='Prospective initial levels, e.g.1 or1,0')
    parser.add_argument('--learning', choices=['frozen', 'native'], default='frozen')
    parser.add_argument('--sim-seconds', type=float, default=80)
    parser.add_argument('--wall-seconds', type=float, default=120)
    args = parser.parse_args(); seeds = [int(s) for s in args.seeds.split(',')]
    hungers = [float(h) for h in args.hungers.split(',')]
    assert seeds and len(set(seeds)) == len(seeds) and all(0 <= s < 2**32 for s in seeds)
    assert hungers and len(set(hungers)) == len(hungers) and all(h in (0, 1) for h in hungers)
    assert 0 < args.sim_seconds <= 80 and 0 < args.wall_seconds <= 120
    journal = Path(str(args.receipt) + '.journal.jsonl')
    for path in (args.receipt, journal):
        if path.exists():
            parser.error(f'refusing to overwrite {path}')
    pins = source_hashes(ROOT, RUNNER)
    protocol = {'seeds': seeds, 'initialHunger': hungers, 'learning': args.learning,
                'maxSimSecondsPerArm': args.sim_seconds, 'maxWallSecondsPerArm': args.wall_seconds,
                'sampleSeconds': .25, 'solverCap': 256, 'sugar': 'banana',
                'order': 'Seeds and initial hunger levels in listed order. Every arm retained.',
                'scope': 'Only initial hunger and seeded model/body/scene RNG are set. Frozen arms set eta/etaBias/etaCritic0 once in learn:init. No forced goals, draws, routes, poses, sensory levels, rewards or outcomes; no console interventions. Body motion is supplied assistance. Hunger0 rises naturally above.2, so it is initially fed, not persistently fed. Async scheduling and animated retinal inputs prevent exact paired-trajectory replay. Integrity success is separate from observed food competence; learning counts alone are not success.'}
    result = {'schema': 'cadence.food-seeking-browser/1', 'started': datetime.now(timezone.utc).isoformat(),
              'protocol': protocol, 'sourceSha256': pins, 'arms': [], 'completed': False}
    journal.write_text(json.dumps({'protocol': protocol, 'sourceSha256': pins}) + '\n')
    with sync_playwright() as playwright:
        for seed in seeds:
            for hunger in hungers:
                arm = run_arm(playwright, args, seed, hunger, pins); result['arms'].append(arm)
                with journal.open('a') as out:
                    out.write(json.dumps(arm, allow_nan=False) + '\n')
                print(json.dumps({'seed': seed, 'hunger': hunger, 'integrityPassed': arm['integrityPassed'],
                                  'summary': arm.get('summary'), 'error': arm.get('error')}, allow_nan=False), flush=True)
    result['sourcesUnchanged'] = source_hashes(ROOT, RUNNER) == pins
    result['completed'] = result['sourcesUnchanged'] and all(a['integrityPassed'] for a in result['arms'])
    result['journalSha256'] = sha(journal); result['finished'] = datetime.now(timezone.utc).isoformat()
    result['body_sha256'] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    args.receipt.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    raise SystemExit(0 if result['completed'] else 1)
