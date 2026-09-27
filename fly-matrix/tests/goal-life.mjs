// Brain-intent custody plus real body integration, using explicit synthetic
// qualified replies. No fixture claims natural connectome navigation skill.
import assert from 'node:assert/strict';
import { GoalLife, GOAL_BODY } from '../web/goal-life.js';
import { Flight, DT, ROOM, RADIUS } from '../web/body.js';
import { POLICY_OUTPUTS, targetProbabilities } from '../web/neural-policy.js';
import { Life } from '../web/life.js';

const room = () => ({ table: { x0: 2.45, x1: 3.55, y0: 1.95, y1: 2.65, z: .75 },
  fruits: { banana: { pos: [2.88, 2.33, .795] }, bread: { pos: [3.13, 2.28, .8068] } }, surfaceZ: () => .75 });
const create = (p = [1.2, 1, 1.2], options = {}) => new GoalLife(new Flight(p, .3), room(), 7, { generation: 2, ...options });
const solve = () => ({ converged: true, iterations: 20, residual: 1e-8, tolerance: 1e-6, reason: 'residual_tolerance' });
function reply(request, fruit = 'banana', action = 0) {
  const candidates = ['banana', 'bread'].map(name => {
    const v = name === fruit ? [.9, .1] : [.1, .9];
    return { fruit: name, score: v[0] - v[1], readouts: Object.fromEntries(POLICY_OUTPUTS.map((n, i) => [n, v[i]])), solve: solve() };
  });
  const v = action ? [.1, .9] : [.9, .1], w = v.map(x => Math.exp(x / .3)), p = w.map(x => x / (w[0] + w[1]));
  return { ...request, type: 'assisted_decision', kind: 'decision', accepted: true, ...solve(),
    readouts: Object.fromEntries(POLICY_OUTPUTS.map((n, i) => [n, v[i]])),
    targetSelection: { fruit, candidates, p: targetProbabilities(candidates, .3), draw: .5, temperature: .3 },
    decision: { accepted: true, action, choice: action, p, draw: .5, greedy: false,
      solves: { free: solve(), plus: solve(), minus: solve() } } };
}
function admit(fly, fruit = 'banana', action = 0, id = 1) {
  const request = fly.openDecision(id, 2, { steps: 1024, tolerance: 1e-6, temperature: .3 });
  assert.ok(request?.selectTarget); assert.equal(fly.search.fruit, null);
  assert.ok(fly.applyAssistedDecision(reply(request, fruit, action))); return request;
}
function advance(fly, seconds) { for (let i = 0; i < Math.round(seconds / DT); i++) if (fly.step(DT)) fly.decide(); }
let passed = 0;
function test(name, run) { run(); passed++; console.log(`ok ${name}`); }

test('startup opt-out hovers without a strongest-odor or random route', () => {
  const fly = create(undefined, { startup: false, waiting: false }), p = [...fly.flight.p];
  fly.smelled = 'banana'; fly.smellC = 1;
  fly.rng = () => { throw Error('random route forbidden'); };
  advance(fly, 2);
  assert.ok(Math.hypot(fly.flight.p[0] - p[0], fly.flight.p[1] - p[1]) < 1e-5);
  assert.equal(fly.search, null); assert.equal(fly.authority.goal.fruit, null);
  assert.equal(fly.authority.navigation.active, false); assert.equal(fly.authority.goal.framesApplied, 0);
  assert.ok(fly.odourTick());
});

test('brain target beats a nearer/stronger opposite odor, then drives real selected-goal motion', () => {
  const fly = create(); fly.smelled = 'banana'; fly.smellC = 1;
  admit(fly, 'bread', 0);
  assert.equal(fly.search.fruit, 'bread'); assert.equal(fly.authority.goal.fruit, 'bread');
  assert.equal(fly.authority.goal.executed, false); assert.equal(fly.authority.choices.executed, 0);
  const start = [...fly.flight.p]; advance(fly, 2);
  assert.ok(fly.flight.p[0] > start[0] + .2 && fly.flight.p[1] > start[1] + .1);
  assert.equal(fly.authority.choices.executed, 1); assert.equal(fly.authority.lastChoice.targetSelection.fruit, 'bread');
  assert.ok(fly.authority.goal.framesApplied > 0); assert.equal(fly.authority.navigation.framesApplied, 0);
});

test('opposite MBON intent produces approach versus bounded retreat, without choosing another fruit', () => {
  const approach = create(), avoid = create(); admit(approach); admit(avoid, 'banana', 1);
  const p = [...approach.flight.p]; advance(approach, 1); advance(avoid, 1);
  assert.ok(approach.flight.p[0] > p[0]); assert.ok(avoid.flight.p[0] < p[0]);
  assert.equal(avoid.authority.goal.fruit, 'banana'); assert.equal(avoid.authority.goal.action, 1);
  assert.match(avoid.authority.navigationScript, /clearance/);
});

test('slow qualified decisions retain a full bounded body-execution window for travel and landing', () => {
  const fly = create(undefined, { startup: false, waiting: false }), request = fly.openDecision(1, 2);
  advance(fly, 9);
  assert.ok(fly.applyAssistedDecision(reply(request, 'bread', 0)));
  assert.equal(fly.search.goalDeadline, fly.clock + GOAL_BODY.maxExecutionSeconds);
  for (let i = 0; i < 24000 && fly.mode === 'flying'; i++) {
    if (fly.step(DT)) { fly.decide(); fly.odourTick(); }
  }
  assert.ok(fly.clock > 15); assert.equal(fly.mode, 'landed');
  assert.equal(fly.visits.bread, 1); assert.ok(fly.search.neuralChoice.executed);
});

test('all walls and corners bound an avoid goal inside clearance and physically recover from contact', () => {
  const positions = [[.001,1.5,1.2],[3.999,1.5,1.2],[2,.001,1.2],[2,2.999,1.2],
    [.001,.001,1.2],[3.999,.001,1.2],[.001,2.999,1.2],[3.999,2.999,1.2]];
  for (const p of positions) {
    const fly = create(p); admit(fly, 'banana', 1);
    const target = fly.authority.goal.target;
    assert.ok(target.every((x,i) => x >= GOAL_BODY.clearance && x <= ROOM[i] - GOAL_BODY.clearance));
    assert.ok(fly.authority.goal.clearanceLimited);
    advance(fly, 3);
    assert.ok(fly.flight.p[0] > .09 && fly.flight.p[0] < ROOM[0] - .09, JSON.stringify(fly.flight.p));
    assert.ok(fly.flight.p[1] > .09 && fly.flight.p[1] < ROOM[1] - .09, JSON.stringify(fly.flight.p));
    assert.equal(fly.flight.touching, 0);
  }
});

test('candidate phases, draw, identity and all actor phases qualify before any selected route', () => {
  const mutations = [m => { m.targetSelection.fruit = 'bread'; }, m => { m.targetSelection.p[0] = .2; },
    m => { m.targetSelection.candidates[0].solve.converged = false; }, m => { m.generation++; },
    m => { m.searchToken += '-old'; }, m => { m.decision.solves.minus.converged = false; },
    m => { m.decision.solves.plus.residual = 10; }];
  for (const mutate of mutations) {
    const fly = create(), request = fly.openDecision(1), m = reply(request); mutate(m);
    assert.equal(fly.applyAssistedDecision(m), false);
    assert.equal(fly.authority.goal.fruit, null); assert.equal(fly.authority.choices.executed, 0);
  }
});

test('admission alone earns no reward credit; actual selected-goal execution does', () => {
  const none = create(); admit(none); none.outcome(1, 'test contact', 'banana'); assert.equal(none.pendingReward.neuralCredit, false);
  const yes = create(), request = admit(yes); advance(yes, .2); yes.outcome(1, 'test contact', 'banana');
  assert.equal(yes.pendingReward.neuralCredit, true); assert.equal(yes.pendingReward.searchToken, request.searchToken);
  const wrong = create(); admit(wrong); advance(wrong, .2); wrong.outcome(1, 'other fruit contact', 'bread');
  assert.equal(wrong.pendingReward.neuralCredit, false);
});

test('hard revoke withdraws goal immediately and old replies cannot restore it', () => {
  const fly = create(); admit(fly); advance(fly, .2); const stoppedAt = [...fly.flight.p];
  fly.revokeNeural('pause'); assert.equal(fly.authority.goal.phase, 'waiting');
  assert.deepEqual(fly.pilot.target, stoppedAt); assert.equal(fly.search, null); assert.equal(fly.authority.goal.fruit, null);
  const waiting = create(), request = waiting.openDecision(1); waiting.revokeNeural('reset');
  assert.equal(waiting.applyAssistedDecision(reply(request)), false);
  advance(fly, 1); assert.equal(fly.authority.goal.framesApplied > 0, true);
  assert.equal(fly.authority.navigation.active, false);
  fly.outcome(1, 'contact after revocation', 'banana'); assert.equal(fly.pendingReward.neuralCredit, false);
});

test('immediate perch without a physical selected-goal step cannot mint execution', () => {
  const fly = create([3.13, 2.28, .8068 + RADIUS]);
  admit(fly, 'bread'); fly.step(DT);
  assert.equal(fly.mode, 'landed'); assert.equal(fly.authority.choices.executed, 0);
  assert.equal(fly.authority.goal.framesApplied, 0);
  fly.outcome(-1, 'later contact punishment', 'bread'); assert.equal(fly.pendingReward.neuralCredit, false);
});

test('selected approach lands through body integration, then uses declared feeding and grooming', () => {
  const fly = create([2.65,2.33,1.0]); admit(fly, 'banana', 0);
  for (let i = 0; i < 20000 && fly.mode === 'flying'; i++) if (fly.step(DT)) fly.decide();
  assert.equal(fly.mode, 'landed'); assert.equal(fly.visits.banana, 1); assert.ok(fly.pendingReward?.neuralCredit);
  assert.ok(Math.hypot(fly.flight.p[0] - fly.fruits.banana.pos[0], fly.flight.p[1] - fly.fruits.banana.pos[1]) < .03);
  fly.senses(null, 0); advance(fly, 2.2); assert.equal(fly.mode, 'feeding');
  advance(fly, 4.1); assert.equal(fly.mode, 'grooming'); assert.equal(fly.pose().groom, 'head');
  advance(fly, 2.6); assert.equal(fly.mode, 'flying'); assert.equal(fly.authority.goal.phase, 'waiting');
  assert.match(fly.authority.bodyRoutines, /not learned/);
});

test('goal controller never invokes the legacy random flight/strongest-odor decision methods', () => {
  const methods = ['decide','odourTick','chooseLanding'], previous = methods.map(name => Life.prototype[name]);
  try {
    for (const name of methods) Life.prototype[name] = () => { throw Error(`legacy ${name}`); };
    const fly = create(); assert.ok(fly.odourTick()); admit(fly); advance(fly, .2); fly.decide();
  } finally { methods.forEach((name, i) => { Life.prototype[name] = previous[i]; }); }
});

console.log(`${passed} goal controller groups passed`);
