// Synthetic admissions test lesson custody without running a solve.
import assert from 'node:assert/strict';
import { Brain } from '../web/brain.js';

function fixture({ frozen = false, accept = true, ticks = 3 } = {}) {
  const brain = Object.create(Brain.prototype);
  brain.p = { teach: 2, teach_level: 0.8, window: 4 };
  brain.readouts = ['forward', 'reverse'];
  brain.outputPatch = { forward: 2, reverse: 3 };
  brain.frozen = frozen;
  brain.state = Float64Array.of(0.5, -0.5, 0, 0);
  brain.weights = Float64Array.of(0, 0);
  brain.biases = Float64Array.of(0, 0, 0, 0);
  brain.stretch = Array.from({ length: ticks }, (_, t) =>
    [Float64Array.of(0.1 * t, 0, 0), [0.1 * t, 0.1 * t + 0.05]]);
  brain.admissions = 0; brain.lessons = 0; brain.rejected = 0; brain.refusals = 0;
  brain.calls = [];
  brain.engine = {
    nPatches: 4, nInputs: 3,
    settle(inputs, state, weights, biases, opts) {
      brain.calls.push({ inputs: Float64Array.from(inputs), clamps: new Map(opts.clamps), B: opts.B, learn: opts.learn });
      if (!accept) return { qualified: false, reason: 'line_search' };
      return { qualified: true, reason: 'qualified', energy: 0.5, sweeps: 9,
               weights: Float64Array.of(1, -0.5), biases: Float64Array.of(0.25, 0, 0, 0) };
    },
  };
  return brain;
}

{ // an accepted lesson: retained change, counters, spent stretch, built clamps
  const b = fixture();
  const r = b.learn(['food']);
  assert.equal(r.updated, true); assert.equal(r.reason, 'qualified'); assert.equal(r.rows, 3);
  assert.equal(b.lessons, 1); assert.equal(b.rejected, 0); assert.equal(b.admissions, 1);
  assert.equal(b.stretch.length, 0);
  assert.deepEqual(Array.from(r.applied), [1, -0.5]);
  assert.deepEqual(Array.from(r.appliedBiases), [0.25, 0, 0, 0]);
  assert.deepEqual(Array.from(b.weights), [1, -0.5]);
  const [call] = b.calls;
  assert.equal(call.B, 3); assert.equal(call.learn, true);
  // the early tick restates the free prediction; the last two are corrected
  assert.deepEqual([call.clamps.get(2), call.clamps.get(3)], [0, 0.05]);
  for (const row of [1, 2])
    assert.deepEqual([call.clamps.get(row * 4 + 2), call.clamps.get(row * 4 + 3)], [0.8, 0]);
  // every row starts from the same retained live state
  assert.deepEqual(Array.from(call.inputs.subarray(0, 3)), [0, 0, 0]);
}
{ // pain and joint outcomes build their targets
  const pain = fixture(); pain.learn(['pain']);
  const p = pain.calls[0].clamps;
  assert.deepEqual([p.get(2 * 4 + 2), p.get(2 * 4 + 3)], [0, 0.8]);
  const both = fixture(); both.learn(['food', 'pain']);
  const q = both.calls[0].clamps;
  assert.deepEqual([q.get(2 * 4 + 2), q.get(2 * 4 + 3)], [0.8, 0.8]);   // no rival to rest
}
{ // a refused solve never claims applied learning
  const b = fixture({ accept: false });
  const r = b.learn(['food']);
  assert.equal(r.updated, false); assert.equal(r.reason, 'line_search'); assert.equal(r.applied, undefined);
  assert.equal(b.lessons, 0); assert.equal(b.rejected, 1); assert.equal(b.admissions, 0);
  assert.deepEqual(Array.from(b.weights), [0, 0]);
}
{ // a frozen worm spends the stretch without a solve
  const b = fixture({ frozen: true });
  const r = b.learn(['pain']);
  assert.equal(r.updated, false); assert.equal(r.reason, 'frozen');
  assert.equal(b.calls.length, 0); assert.equal(b.rejected, 1); assert.equal(b.stretch.length, 0);
}
{ // an outcome without experience is rejected
  const b = fixture({ ticks: 0 });
  const r = b.learn(['food']);
  assert.equal(r.updated, false); assert.equal(r.reason, 'no_experience');
  assert.equal(b.calls.length, 0); assert.equal(b.rejected, 1);
}
console.log('worm learning accounting: 5 scenarios passed');
