// Hybrid-controller tests. Fabricated qualified replies isolate action custody;
// they do not claim that the connectome learned the supplied body routines.
import assert from 'node:assert/strict';
import { DemoLife, DEMO_DECISION_TTL, DEMO_OBSERVATION_TTL } from '../web/demo-life.js';
import { AssistedLife, ASSISTED_OUTPUTS } from '../web/assisted-life.js';
import { SETTLED_MOTOR_GROUPS } from '../web/motor.js';
import { Flight, DT } from '../web/body.js';

const room = () => ({ table: { x0: 2.45, x1: 3.55, y0: 1.95, y1: 2.65, z: .75 },
  fruits: { banana: { pos: [2.88, 2.33, .795] }, bread: { pos: [3.13, 2.28, .8068] } }, surfaceZ: () => .75 });
const create = (Type = DemoLife) => new Type(new Flight([1.2, 1, 1.2], .3), room(), 7, { generation: 2 });
const reads = () => ({ ...Object.fromEntries(SETTLED_MOTOR_GROUPS.map(name => [name, 0])), 'mn:wing:b2:left': .8 });
const solve = () => ({ converged: true, iterations: 20, residual: 1e-8, tolerance: 1e-6, reason: 'residual_tolerance' });
const observation = request => ({ ...request, type: 'control', kind: 'control', ...solve(), readouts: reads(),
  retinal: { neurons: 1831, mean: .4 } });
function open(fly, id = 1) {
  const p = fly.fruits.banana.pos;
  fly.flight.p = [p[0] - .04, p[1], p[2] + .08];
  fly.search = { since: fly.clock, askedAt: fly.clock, fruit: 'banana', asked: true, landed: null };
  fly.valence = { fruit: 'banana', action: 0, p: null, innate: true };
  return fly.openDecision(id, 2, { steps: 1024, tolerance: 1e-6, temperature: .3 });
}
function decision(request, action = 0) {
  const values = action === 0 ? [.9, .1] : [.1, .9], w = values.map(v => Math.exp(v / request.temperature));
  const p = w.map(v => v / (w[0] + w[1]));
  return { ...request, type: 'assisted_decision', kind: 'decision', accepted: true, ...solve(),
    readouts: Object.fromEntries(ASSISTED_OUTPUTS.map((n, i) => [n, values[i]])),
    decision: { accepted: true, choice: action, action, p, draw: .5, greedy: false,
      solves: { free: solve(), plus: solve(), minus: solve() } } };
}
let passed = 0;
function test(name, run) { run(); passed++; console.log(`ok ${name}`); }

test('no-neural-reply body follows the existing seeded assisted physics exactly', () => {
  const demo = create(), previous = create(AssistedLife), start = [...demo.flight.p];
  for (let i = 0; i < 4000; i++) for (const fly of [demo, previous]) if (fly.step(DT)) fly.decide();
  assert.deepEqual(demo.flight.p, previous.flight.p); assert.deepEqual(demo.flight.q, previous.flight.q);
  assert.deepEqual(demo.controls, previous.controls);
  assert.ok(Math.hypot(...demo.flight.p.map((x, i) => x - start[i])) > .1);
  assert.equal(demo.authority.autonomous, false); assert.equal(demo.authority.navigation.active, false);
  assert.equal(demo.authority.navigation.framesApplied, 0); assert.equal(demo.authority.motionSupport.enabled, false);
  assert.match(demo.authority.navigationScript, /fruit|wall/);
  assert.equal(demo.authority.choices.executed, 0);
});

test('late checked state is displayed/scheduled without extending raw motor authority', () => {
  const fly = create(), request = fly.expectObservation(1);
  assert.equal(request.deadline, DEMO_OBSERVATION_TTL); assert.equal(request.freshUntil, 1);
  fly.clock = 2; const reply = observation(request);
  assert.ok(fly.applyObservation(reply)); fly.step(DT);
  assert.deepEqual(fly.readouts, reply.readouts); assert.equal(fly.lastQualifiedObservation.requestId, 1);
  assert.equal(fly._observation, null); assert.equal(fly._routeObservation, null);
  assert.ok(Object.values(fly.authority.neuralTrim).every(v => v === 0));
  assert.equal(fly.authority.navigation.active, false);
  reply.retinal.mean = 1; assert.equal(fly.authority.retinal.mean, .4);
});

test('fresh checked motor outputs still alter actual body torques', () => {
  const fly = create(), zero = create();
  const m = observation(fly.expectObservation(1)), z = observation(zero.expectObservation(1));
  for (const key of Object.keys(z.readouts)) z.readouts[key] = 0;
  assert.ok(fly.applyObservation(m)); assert.ok(zero.applyObservation(z));
  fly.step(DT); zero.step(DT);
  assert.notDeepEqual(fly.flight.w, zero.flight.w);
  assert.equal(fly.controls.f, zero.controls.f);
  fly.clock = 1; fly.step(DT); assert.ok(Object.values(fly.authority.neuralTrim).every(v => v === 0));
});

test('invalid, expired and wrong-generation replies never restore authority', () => {
  for (const mutate of [m => { m.residual = 2; }, m => { m.converged = false; },
    m => { m.generation++; }, m => { m.attention = 'banana'; }, m => { delete m.readouts.mn9; }]) {
    const fly = create(), reply = observation(fly.expectObservation(1)); mutate(reply);
    assert.equal(fly.applyObservation(reply), false); assert.equal(fly.lastQualifiedObservation, null);
  }
  const fly = create(), reply = observation(fly.expectObservation(1));
  fly.clock = DEMO_OBSERVATION_TTL; assert.equal(fly.applyObservation(reply), false);
  const fresh = create(), pending = observation(fresh.expectObservation(1));
  fresh.revokeNeural('pause'); assert.equal(fresh.applyObservation(pending), false);
  assert.equal(fresh.lastQualifiedObservation, null);
});

test('ten-second choice allowance also prevents the legacy three-second landing fallback', () => {
  const fly = create(), request = open(fly);
  assert.equal(request.deadline, DEMO_DECISION_TTL); assert.equal(request.selectTarget, false);
  assert.equal(request.attention, 'banana');
  fly.clock = 6; fly.decide(); assert.equal(fly.landing, null);
  assert.ok(fly.applyAssistedDecision(decision(request, 0)));
  assert.equal(fly.search.neuralChoice.executed, false);
  fly.decide(); assert.equal(fly.search.neuralChoice.executed, true); assert.ok(fly.landing);
  const legacy = create(AssistedLife), old = open(legacy);
  assert.equal(old.deadline, 3); legacy.clock = 6;
  assert.equal(legacy.applyAssistedDecision(decision(old)), false);
});

test('checked MBON approach versus avoidance changes supplied route and real trajectory', () => {
  const approach = create(), avoid = create();
  assert.ok(approach.applyAssistedDecision(decision(open(approach), 0)));
  assert.ok(avoid.applyAssistedDecision(decision(open(avoid), 1)));
  approach.decide(); avoid.decide();
  assert.ok(approach.landing); assert.equal(avoid.landing, null);
  assert.equal(approach.authority.choices.executed, 1); assert.equal(avoid.authority.choices.executed, 1);
  for (let k = 0; k < 500; k++) { approach.step(DT); avoid.step(DT); }
  assert.ok(Math.hypot(...approach.flight.p.map((x, i) => x - avoid.flight.p[i])) > .001);
});

test('only an actually executed, matching, unrevoked choice grants outcome credit', () => {
  const notExecuted = create(); notExecuted.applyAssistedDecision(decision(open(notExecuted)));
  notExecuted.outcome(1, 'contact', 'banana'); assert.equal(notExecuted.pendingReward.neuralCredit, false);
  const executed = create(), request = open(executed);
  executed.applyAssistedDecision(decision(request)); executed.decide();
  executed.outcome(1, 'contact', 'banana');
  assert.equal(executed.pendingReward.neuralCredit, true);
  assert.equal(executed.pendingReward.searchToken, request.searchToken);
  const revoked = create(); revoked.applyAssistedDecision(decision(open(revoked))); revoked.decide();
  revoked.revokeNeural('pause'); revoked.outcome(1, 'contact', 'banana'); assert.equal(revoked.pendingReward.neuralCredit, false);
});

test('expired or failed nudged choices are rejected without pretending learning', () => {
  const fly = create(), request = open(fly); fly.clock = 10;
  assert.equal(fly.applyAssistedDecision(decision(request)), false);
  for (const phase of ['free', 'plus', 'minus']) {
    const other = create(), reply = decision(open(other)); reply.decision.solves[phase].converged = false;
    assert.equal(other.applyAssistedDecision(reply), false); assert.equal(other.authority.choices.executed, 0);
  }
});

test('restored grooming, feeding and takeoff are explicitly supplied body routines', () => {
  const fly = create(); fly.mode = 'landed'; fly.startGrooming();
  assert.equal(fly.pose().mode, 'grooming'); assert.ok(fly.pose().groom);
  fly.startFeeding(); assert.equal(fly.pose().mode, 'feeding');
  const before = fly.flight.v[2]; fly.takeoff('test body routine');
  assert.equal(fly.mode, 'flying'); assert.equal(fly.flight.v[2], before + .28);
  assert.equal(fly.pose().neural, undefined); assert.equal(fly.authority.choices.executed, 0);
  assert.match(fly.authority.behaviorFallback, /grooming.*feeding|feeding.*grooming/);
  fly.mode = 'landed'; assert.equal(fly.senses(null, 0).leg_touch, .6, 'contact intensity is not lowered to pass a solve');
});

console.log(`${passed} hybrid demo groups passed`);
