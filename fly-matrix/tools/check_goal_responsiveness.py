#!/usr/bin/env python3
"""Bounded normal-browser check of physical nudges and actual solver progress."""
import argparse
import hashlib
import json
import math
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from playwright.sync_api import sync_playwright
from browser_source_binding import ServedSourceBinding, source_hashes

ROOT = Path(__file__).resolve().parents[1]
RUNNER = 'tools/check_goal_responsiveness.py'
INSTRUMENT = r"""(()=>{
 const W=Worker,send=W.prototype.postMessage;
 const log=window.__responsive={requests:[],packets:[],nudges:[],maxJobs:0,jobs:new Set()};
 window.__responseSample=()=>{const a=window.__app;if(!a)return null;const l=a.life(),f=a.flight(),s=a.S;
  return{wall:performance.now(),time:l.clock,frames:window.__frames||0,mode:l.mode,paused:s.paused,
   position:f.p.slice(),velocity:f.v.slice(),spin:f.w.slice(),quaternion:f.q.slice(),perch:l.perch?.slice()??null,
   touching:f.touching,onFloor:f.onFloor,contactFruit:l.fruitAt(f.p),hunger:l.hunger,
   startup:structuredClone(l.authority.startup),waitingSupport:structuredClone(l.authority.waitingSupport),
   physicalReaction:structuredClone(l.authority.physicalReaction),goal:structuredClone(l.authority.goal),
   choices:structuredClone(l.authority.choices),lastChoice:structuredClone(l.authority.lastChoice),
   lesson:structuredClone(s.lessonStats),workerBusy:s.workerBusy,workerJobId:s.workerJobId,pending:s.pending,
   pendingId:s.pendingId,progressFrames:s.progressFrames,display:structuredClone(a.settlementProgress()),
   scanText:document.getElementById('scan-playback')?.textContent,fps:s.fps};};
 W.prototype.postMessage=function(m,...rest){
  if(m.type?.startsWith('assist:')){
   const job=['assist:observe','assist:decide'].includes(m.type),key=`${m.generation}:${m.requestId}`;
   if(job){log.jobs.add(key);log.maxJobs=Math.max(log.maxJobs,log.jobs.size);}
   log.requests.push({type:m.type,requestId:m.requestId,generation:m.generation,searchToken:m.searchToken,
    wall:performance.now(),time:window.__app?.life().clock,steps:m.steps,tolerance:m.tolerance,
    reward:m.reward,neuralCredit:m.neuralCredit,progress:m.progress,activeJobs:log.jobs.size});
  }return send.call(this,m,...rest);
 };
 window.Worker=new Proxy(W,{construct(T,args){const w=new T(...args);w.addEventListener('message',e=>{const m=e.data;
  if(['control','assisted_decision'].includes(m.type))log.jobs.delete(`${m.generation}:${m.requestId}`);
  if(['settlement_progress','control','assisted_decision','lesson'].includes(m.type)){
   const x={type:m.type,requestId:m.requestId,generation:m.generation,searchToken:m.searchToken,
    wall:performance.now(),time:window.__app?.life().clock,phase:m.phase,attention:m.attention,iteration:m.iteration,
    iterations:m.iterations,residual:m.residual,tolerance:m.tolerance,n:m.n,valid:m.valid,authoritative:m.authoritative,
    maxAbsDeltaV:m.maxAbsDeltaV,accepted:m.accepted,converged:m.converged,reason:m.reason,
    updates:m.updates,moved:m.moved,delta:m.delta,decision:m.decision,targetSelection:m.targetSelection,
    hasReadouts:!!m.readouts,hasActivity:!!m.activity,hasDelta:!!m.deltaV};
   log.packets.push(x);
  }});return w;}});
 document.addEventListener('submit',e=>{if(e.target.id!=='console-form')return;
  if(document.getElementById('console-input').value.trim()!=='/nudge')return;
  const event={before:window.__responseSample()};log.nudges.push(event);
 },true);
})();"""


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(page, text):
    page.locator('#console-input').fill(text)
    page.locator('#console-input').press('Enter')


def sample(page):
    return page.evaluate('__responseSample()')


def run(args):
    pins = source_hashes(ROOT, RUNNER)
    r = {'schema': 'cadence.goal-responsiveness-browser/1', 'passed': False,
         'started': datetime.now(timezone.utc).isoformat(), 'sourceSha256': pins, 'errors': [],
         'protocol': {'url': args.url, 'viewport': [1440, 1000], 'maxSimSeconds': 80,
                      'maxWallSeconds': 120, 'sampleSeconds': .25, 'firstNudgeSimSeconds': 8,
                      'nudgeDisplacementWindow': .1, 'minimumNudgeDisplacementMetres': .005,
                      'stationarySpeedMetresPerSecond': .03,
                      'maximumSampledAirborneStationarySeconds': 2, 'minimumMeanFPS': 45,
                      'scope': 'Unchanged scene, hardware Chromium, ordinary typed console commands. First nudge near8 simulated seconds; second only at an observed natural perch before70simsec. No injected body/neuronal state, decision or reward. Intermediate solver packets are display-only. Landing not observed within the fixed window is reported as uncovered. Quarter-second sampling cannot rule out shorter unsampled holds.'},
         'samples': [], 'nudges': []}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, channel='chromium')
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        r['browser'] = browser.version
        binding = ServedSourceBinding(page.context, args.url, pins)
        page.on('pageerror', lambda e: r['errors'].append(str(e)))
        page.add_init_script(INSTRUMENT)
        started = None
        try:
            page.goto(args.url, wait_until='domcontentloaded')
            page.wait_for_function('window.__app?.S.ready&&__app.S.learnReady', timeout=90000)
            # Register after the console's actual form handler. Browser microtask
            # checkpoints can occur between document-capture and target handlers;
            # a capture-listener microtask is therefore not an after-action witness.
            page.evaluate("document.getElementById('console-form').addEventListener('submit',()=>{const e=__responsive.nudges.at(-1);if(e&&!e.after)e.after=__responseSample();})")
            r['gpu'] = page.evaluate("()=>{const g=__app.renderer.getContext(),e=g.getExtension('WEBGL_debug_renderer_info');return e?g.getParameter(e.UNMASKED_RENDERER_WEBGL):g.getParameter(g.RENDERER)}")
            assert 'SwiftShader' not in r['gpu'], 'Hardware renderer required'
            r['brain'] = page.evaluate('__app.brainInfo()')
            assert r['brain']['n'] == 150802 and r['brain']['edges'] == 1877099
            start = sample(page); started = time.monotonic(); paused = False
            r['samples'].append(start)
            while time.monotonic() - started < 120:
                current = sample(page); r['samples'].append(current)
                elapsed = current['time'] - start['time']
                if elapsed >= 80:
                    break
                which = None
                if not r['nudges'] and elapsed >= 8:
                    which = 'initial'
                elif len(r['nudges']) == 1 and elapsed < 70 and current['mode'] != 'flying' and current['perch']:
                    which = 'natural_perch'
                if which:
                    count = len(r['nudges'])
                    command(page, '/nudge')
                    page.wait_for_function('(k)=>__responsive.nudges[k]?.after', arg=count, timeout=3000)
                    event = page.evaluate('(k)=>structuredClone(__responsive.nudges[k])', count)
                    r['nudges'].append({'kind': which, **event})
                    assert event['after']['mode'] == 'flying' and event['after']['perch'] is None, 'Nudge did not release the body'
                    assert event['after']['position'] == event['before']['position'], 'Nudge teleported the body'
                    assert event['after']['quaternion'] == event['before']['quaternion'], 'Nudge directly replaced body attitude'
                    assert abs(event['after']['velocity'][2] - event['before']['velocity'][2] - .6) < 1e-10, 'Nudge did not apply the declared physical impulse'
                    assert event['after']['choices']['executed'] == event['before']['choices']['executed'], 'Nudge minted neural execution'
                    page.wait_for_function('(t)=>__app.life().clock>=t', arg=event['after']['time'] + .1, timeout=3000)
                    later = sample(page); r['nudges'][-1]['afterWindow'] = later
                    displacement = math.dist(later['position'], event['after']['position'])
                    r['nudges'][-1]['displacementMetres'] = displacement
                    assert displacement > .005, 'Nudge has no measurable physical displacement'
                if not paused and (elapsed >= 72 or time.monotonic() - started >= 108):
                    command(page, '/pause'); page.wait_for_function('__app.S.paused', timeout=3000)
                    before = sample(page); page.wait_for_timeout(350); after = sample(page)
                    assert after['time'] == before['time'] and after['position'] == before['position'], 'Pause advanced body physics'
                    command(page, '/status')
                    r['pause'] = {'before': before, 'after': after, 'console': page.evaluate('__app.console.snapshot()')}
                    command(page, '/resume'); page.wait_for_function('!__app.S.paused', timeout=3000)
                    r['resume'] = sample(page); paused = True
                page.wait_for_timeout(250)
            r['wallSeconds'] = time.monotonic() - started
            r['packets'] = page.evaluate('__responsive.packets')
            r['requests'] = page.evaluate('__responsive.requests')
            r['maximumSimultaneousSolverJobs'] = page.evaluate('__responsive.maxJobs')
            r['console'] = page.evaluate('__app.console.snapshot()')
            r['final'] = sample(page)
            page.screenshot(path=str(args.screenshot))
            r['image'] = {'path': str(args.screenshot.resolve().relative_to(ROOT)), 'sha256': digest(args.screenshot)}
            r['served'] = binding.receipt()
            rows = r['samples']; packets = r['packets']
            progress = [x for x in packets if x['type'] == 'settlement_progress']
            assert progress and all(x.get('authoritative') is False and not x.get('hasReadouts') and not x.get('accepted') for x in progress)
            displayed = [x for x in rows if x['display']]
            assert displayed, 'No real progress reached the default scan'
            for x in displayed:
                d = x['display']
                assert any(all(q.get(k) == d.get(k) for k in ('requestId', 'generation', 'phase', 'iteration', 'residual')) for q in progress), 'Displayed progress has no matching worker packet'
                assert d['framesReceived'] <= len([q for q in progress if q['wall'] <= x['wall']]), 'Display count exceeds received packets'
            first_move = next((x for x in rows if math.hypot(*x['velocity']) > .05), None)
            first_choice = next((x for x in rows if x['choices']['executed'] > 0), None)
            longest = 0; began = None
            for x in rows:
                stationary = x['mode'] == 'flying' and not x['paused'] and math.hypot(*x['velocity']) < .03
                if stationary:
                    began = x['time'] if began is None else began
                    longest = max(longest, x['time'] - began)
                else:
                    began = None
            r['summary'] = {'firstMovingSimSeconds': first_move['time'] if first_move else None,
                            'firstExecutedChoiceSimSeconds': first_choice['time'] if first_choice else None,
                            'longestSampledAirborneStationarySeconds': longest,
                            'progressPackets': len(progress), 'displayMatchedSamples': len(displayed),
                            'phases': sorted({x['phase'] for x in progress}),
                            'naturalLandingObserved': any(x['perch'] for x in rows),
                            'perchedNudgeCovered': len(r['nudges']) == 2,
                            'bodyModes': sorted({x['mode'] for x in rows}),
                            'simulationSeconds': r['final']['time'] - start['time'],
                            'meanFPS': (r['final']['frames'] - start['frames']) / r['wallSeconds']}
            assert r['maximumSimultaneousSolverJobs'] == 1, 'Brain jobs queued while another solver was busy'
            assert longest <= 2, 'Airborne body stayed almost stationary for more than2 sampled simulated seconds'
            assert r['summary']['meanFPS'] >= 45, 'Average foreground frame rate below45FPS'
            assert first_move and first_move['time'] <= 1, 'No initial physical movement'
            assert first_choice, 'No executed neural choice within the bounded observation'
            choice = first_choice['lastChoice']
            decision = next(q for q in packets if q['type'] == 'assisted_decision' and all(q.get(k) == choice.get(k) for k in ('generation', 'requestId', 'searchToken')))
            assert decision['accepted'] and decision['decision']['accepted']
            assert decision['targetSelection']['fruit'] == choice['fruit'] and decision['decision']['action'] == choice['action']
            solves = [decision['decision']['solves'][k] for k in ('free', 'plus', 'minus')]
            solves += [c['solve'] for c in decision['targetSelection']['candidates']]
            assert all(q['converged'] and 0 <= q['residual'] <= q['tolerance'] <= 1e-6 for q in solves)
            assert r['nudges'] and paused and r['final']['time'] > r['resume']['time'], 'Nudge/pause/resume coverage incomplete'
            for event in r['nudges']:
                newer = [q for q in r['requests'] if q['type'] in ('assist:observe', 'assist:decide') and q['wall'] > event['after']['wall']]
                assert any(any(x['requestId'] == q['requestId'] and x['generation'] == q['generation'] for x in progress) for q in newer), 'No new-input solver progress after nudge'
            assert not r['errors'] and source_hashes(ROOT, RUNNER) == pins
            r['passed'] = True
        except Exception as e:
            r['error'] = repr(e); r['traceback'] = traceback.format_exc()
        finally:
            for key, expression in [('packets', '__responsive.packets'), ('requests', '__responsive.requests')]:
                if key not in r:
                    try:
                        r[key] = page.evaluate(expression)
                    except Exception:
                        pass
            r['sourcesUnchanged'] = source_hashes(ROOT, RUNNER) == pins
            browser.close()
    r['completed'] = datetime.now(timezone.utc).isoformat()
    r['body_sha256'] = hashlib.sha256(json.dumps(r, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return r


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8817/')
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--screenshot', type=Path, required=True)
    args = parser.parse_args()
    for path in (args.receipt, args.screenshot):
        if path.exists():
            parser.error(f'refusing to overwrite {path}')
    result = run(args)
    args.receipt.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result.get(k) for k in ('passed', 'error', 'summary', 'body_sha256')}, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
