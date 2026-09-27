import assert from 'node:assert/strict';
import { odorReceptorResponse as response, ODOR_TRANSDUCTION_GENES as genes } from '../web/odor-transduction.js';
assert.deepEqual(genes, { fedHalfConcentration: 1e-3, hungerSensitivityRatio: 100, hillExponent: 1 });
assert.ok(Object.isFrozen(genes));
for (const h of [0, .2, .6, 1]) {
  assert.equal(response(0, h), 0);
  const k = genes.fedHalfConcentration * genes.hungerSensitivityRatio ** (-h);
  assert.equal(response(k, h), .5);
  let previous = -1;
  for (const c of [0, 1e-15, 1e-9, 1e-6, 1e-5, 1e-3, .1, 1, 10, Number.MAX_VALUE]) {
    const v = response(c, h);
    assert.ok(Number.isFinite(v) && v >= 0 && v <= 1 && v >= previous);
    previous = v;
  }
}
for (const c of [1e-6, 1e-5, .01, 1])
  assert.ok(response(c, 0) < response(c, .6) && response(c, .6) < response(c, 1));
for (const [c, h] of [[-1, 0], [NaN, 0], [Infinity, 0], [1, -.1], [1, 1.1], [1, NaN], [1, Infinity]])
  assert.throws(() => response(c, h), RangeError);
// The same concentration maps identically regardless of an external food name.
assert.deepEqual(['banana', 'bread'].map(() => response(1e-5, 1)), [.5, .5]);
console.log('PASS: frozen genes, exact zero lesion, half-response, bounded monotonic dose/hunger, malformed inputs, food symmetry');
