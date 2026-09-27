import assert from 'node:assert/strict';
import { progressScanFrame } from '../web/settlement-progress.js';
const packet = () => ({ type: 'settlement_progress', authoritative: false, valid: true,
  requestId: 3, generation: 2, phase: 'plus', n: 3, iteration: 16, residual: .1, tolerance: 1e-6,
  activity: Float32Array.from([.2, .7, -.1]), deltaV: Float32Array.from([.01, -.001, 0]) });
const m = packet(), before = structuredClone(m), f = progressScanFrame(m, [2, 0, 1], 4);
assert.deepEqual(m, before); assert.deepEqual([...f.activity], [m.activity[1], m.activity[2], m.activity[0], 0]);
assert.ok(f.heat[2] > f.heat[0] && f.heat[0] > 0); assert.equal(f.heat[1], 0);
assert.equal('converged' in f, false); assert.equal('readouts' in f, false);
const larger = packet(); larger.deltaV[0] *= 2;
assert.equal(progressScanFrame(larger, [2, 0, 1], 4).heat[0], f.heat[0], 'other neurons cannot rescale an unchanged repair');
for (const change of [p => p.authoritative = true, p => p.valid = false, p => p.phase = 'decision',
  p => p.residual = null, p => p.iteration = -1, p => p.activity[0] = NaN,
  p => p.deltaV[1] = Infinity, p => p.activity = new Float32Array(2)]) {
  const bad = packet(); change(bad); assert.throws(() => progressScanFrame(bad, [2, 0, 1], 4));
}
assert.throws(() => progressScanFrame(packet(), [0, 0, 1], 4));
assert.throws(() => progressScanFrame(packet(), [0, 2, 4], 4));
console.log('progress view checks passed: exact mapping, fixed scale, detached diagnostic arrays and malformed packet rejection');
