// Real worker/GoalLife custody after an unattended equation fails to settle.
// The tiny negative-feedback cell fails for both odors together, but each
// attended input converges at exactly the same cap/tolerance. This is a
// scheduling counterexample, not evidence of full-connectome food competence.
import assert from 'node:assert/strict';
import { GoalLife } from '../web/goal-life.js';
import { Flight, DT } from '../web/body.js';
import { ActorCriticLearner } from '../web/learner.js';
import { SETTLED_MOTOR_GROUPS } from '../web/motor.js';
import { chooseWorkerJob, qualifiedObservationId } from '../web/worker-scheduling.js';

const b64 = a => Buffer.from(a.buffer).toString('base64');
const weights = [.6, -.4, .9, -.9, 5.1, 5.1, -20];
const payload = { n: 8, edges: 7, synapses: 7, whole: { neurons: 8 },
  model: { dt: .5, slope: 1, threshold: 6, leak: 0, gain: 1, stimulus_amplitude: 12, adaptation: null },
  arrays: { row_ptr: b64(new Int32Array([0, 0, 0, 0, 0, 1, 2, 4, 7])),
    pre: b64(new Int32Array([6, 6, 2, 3, 2, 3, 7])), weight: b64(new Float64Array(weights)),
    sign: b64(new Float64Array(weights)), count: b64(new Uint16Array(7).fill(1)),
    bias: b64(new Float64Array([0, 0, 0, 0, 6, 5, 0, 0])) },
  populations: { photoreceptor: [0, 1], kc: [6], mbon: [4, 5], motor: [4, 5],
    'orn:decaying_fruit:left': [2], 'orn:decaying_fruit:right': [2],
    'orn:yeasty:left': [3], 'orn:yeasty:right': [3],
    'mbon:MBON11:right': [4], 'mbon:MBON05:left': [5],
    ...Object.fromEntries(SETTLED_MOTOR_GROUPS.map((name, i) => [name, [4 + i % 2]])) } };
const config = { outputs: ['mbon:MBON11:right', 'mbon:MBON05:left'], actions: [0, 1],
  plastic: { pre: ['kc'], post: ['mbon'] }, critic: 'kc', beta: .1, temperature: .3,
  eta: .1, etaBias: 0, etaCritic: .05, gamma: .95, lam: .9, cap: 3, dopamineCap: 1 };
const room = { table: { x0: 2.45, x1: 3.55, y0: 1.95, y1: 2.65, z: .75 },
  fruits: { banana: { pos: [2.88, 2.33, .795] }, bread: { pos: [3.13, 2.28, .8068] } }, surfaceZ: () => .75 };
const life = () => new GoalLife(new Flight([1.2, 1, 1.2], .3), room, 7,
  { generation: 2, startup: false, waiting: false });
const senses = { 'orn:decaying_fruit:left': 1, 'orn:decaying_fruit:right': 1,
  'orn:yeasty:left': 1, 'orn:yeasty:right': 1 };
const retina = { width: 2, height: 1, rgba: new Uint8Array([0, 0, 0, 255, 0, 0, 0, 255]), origin: 'bottom-left' };
const options = { steps: 256, tolerance: 1e-6, temperature: .3 };
const identity = m => Object.fromEntries(['requestId', 'generation', 'searchToken'].map(k => [k, m[k]]));
const original = { self: globalThis.self, fetch: globalThis.fetch, stats: ActorCriticLearner.prototype.stats };
const replies = []; let actor, passed = 0;
globalThis.self = { postMessage: m => replies.push(m) };
globalThis.fetch = async () => ({ json: async () => payload });
ActorCriticLearner.prototype.stats = function (...args) { actor = this; return original.stats.apply(this, args); };
const send = async (m, count = 1) => {
  const n = replies.length; await self.onmessage({ data: m });
  assert.equal(replies.length - n, count); return replies.at(-1);
};
const init = async () => {
  assert.equal((await send({ type: 'init', url: 'tiny counterexample', generation: 2 })).type, 'ready');
  assert.equal((await send({ type: 'learn:init', config, seed: 1 })).type, 'learn:ready');
};
const job = (b, independent = true) => chooseWorkerJob({ observationDue: true, wantDecision: b.odourTick(),
  qualifiedObservationId: qualifiedObservationId(b, 2), lastDecisionId: b._lastDecisionId,
  decisionCertifiesInput: independent });
const propose = async (b, id = 2) => {
  const request = b.openDecision(id, 2, options); assert.ok(request);
  return send({ ...request, type: 'assist:decide', senses, retina });
};
const qualified = s => assert.ok(s.converged && s.iterations <= options.steps
  && s.tolerance === options.tolerance && s.residual <= options.tolerance);
async function test(name, f) { await init(); await f(); passed++; console.log(`ok ${name}`); }
try {
  await import('../web/worker.js');
  await test('failed combined observation cannot starve valid independently checked attended goals', async () => {
    const b = life(), request = b.expectObservation(1, 2, options);
    const observation = await send({ ...request, type: 'assist:observe', senses, retina });
    assert.equal(observation.converged, false); assert.equal(observation.iterations, 256);
    assert.ok(observation.residual > 6); assert.equal(observation.stateRestored, true);
    assert.equal(b.applyObservation(observation), false); assert.equal(b._observation, null);
    assert.equal(b.lastQualifiedObservation, null); assert.equal(job(b, false), 'observation');
    assert.equal(job(b), 'decision');
    const decision = await propose(b);
    assert.equal(decision.accepted, true); decision.targetSelection.candidates.forEach(c => qualified(c.solve));
    Object.values(decision.decision.solves).forEach(qualified);
    assert.equal(b.applyAssistedDecision(decision), true);
    await send({ type: 'assist:accept', ...identity(decision) }, 0);
    assert.equal(b._observation, null); assert.equal(b.authority.choices.executed, 0);
    assert.equal(b.authority.observations.accepted, 0);
    b.step(DT); assert.equal(b.authority.choices.executed, 1);
    assert.ok(Object.values(b.authority.neuralTrim).every(v => v === 0));
    console.log(`  unattended residual=${observation.residual}; attended sweeps=${decision.targetSelection.candidates.map(c => c.solve.iterations).join(',')}`);
  });
  await test('unqualified target evidence still grants neither a goal nor motor trim', async () => {
    const b = life(), request = b.openDecision(1, 2, { ...options, steps: 1 }); assert.ok(request);
    const result = await send({ ...request, type: 'assist:decide', senses, retina });
    assert.equal(result.accepted, false); assert.equal(result.reason, 'target_phase_failed');
    assert.equal(b.applyAssistedDecision(result), false); assert.equal(b.search, null);
    assert.equal(b.authority.choices.executed, 0); assert.equal(actor.pending, null); assert.equal(b._observation, null);
  });
  await test('a failed real nudged solve is not rescued by independently qualified scheduling', async () => {
    const b = life(), settle = actor.brain.settleNudgedResidual;
    actor.brain.settleNudgedResidual = function (...args) { args[4] = 1; return settle.apply(this, args); };
    try {
      const result = await propose(b, 1);
      assert.ok(result.targetSelection.candidates.every(c => c.solve.converged));
      assert.equal(result.accepted, false); assert.equal(result.reason, 'plus_phase_failed');
      assert.equal(b.applyAssistedDecision(result), false); assert.equal(b.authority.choices.executed, 0);
      assert.equal(actor.pending, null); assert.equal(b.search, null);
    } finally { actor.brain.settleNudgedResidual = settle; }
  });
  await test('stale identity, replay, reset and expiry remain fail-closed after independent scheduling', async () => {
    const b = life(), decision = await propose(b, 1);
    for (const changed of [{ generation: 3 }, { requestId: 0 }, { searchToken: 'old' }]) {
      assert.equal(b.applyAssistedDecision({ ...decision, ...changed }), false);
      assert.ok(b._pendingDecision); assert.equal(b.authority.choices.executed, 0);
    }
    assert.equal(b.applyAssistedDecision(decision), true);
    assert.equal(b.applyAssistedDecision(decision), false);
    b.revokeNeural('reset'); assert.equal(b.applyAssistedDecision(decision), false);
    assert.equal(b.authority.goal.decisionRequestId, null); assert.equal(b.authority.choices.executed, 0);
    const fresh = life(), pending = await propose(fresh, 2); fresh.clock = fresh._pendingDecision.deadline;
    assert.equal(fresh.applyAssistedDecision(pending), false); assert.equal(fresh.search, null);
    const fed = life(); fed.hunger = 0; assert.equal(job(fed), 'observation'); assert.equal(fed.openDecision(1), null);
  });
} finally {
  globalThis.self = original.self; globalThis.fetch = original.fetch;
  ActorCriticLearner.prototype.stats = original.stats;
}
console.log(`${passed} independent goal scheduling worker groups passed`);
