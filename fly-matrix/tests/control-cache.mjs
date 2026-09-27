// Differential regression for the control solver's cached equation defect.
// Run: node tests/control-cache.mjs [--payload]
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { SettlingBrain } from "../web/brain.js";

const b64 = a => Buffer.from(a.buffer, a.byteOffset, a.byteLength).toString("base64");
const fixture = () => ({ n: 3, edges: 6,
  model: { dt: 0.2, slope: 4, threshold: 1.5, leak: 0, gain: 1, stimulus_amplitude: 1, adaptation: null },
  arrays: { row_ptr: b64(new Int32Array([0, 2, 4, 6])), pre: b64(new Int32Array([1, 2, 0, 2, 0, 1])),
    weight: b64(new Float64Array([0.3, -0.1, 0.4, 0.2, -0.15, 0.1])) }, populations: { input: [0] } });

// The original algorithm is an independent composition of the preserved public single-step
// operator and a freshly reconstructed potential-equation residual after every step.
function uncached(brain, budget, tolerance) {
  let iterations = 0, residual = brain.equationResidual();
  const initialResidual = Number.isFinite(residual) ? residual : null;
  if (Number.isFinite(residual)) brain.s.set(brain.controlActivation);
  while (Number.isFinite(residual) && residual > tolerance && iterations < budget) {
    brain.step(); iterations++; residual = brain.equationResidual();
  }
  return { converged: Number.isFinite(residual) && residual <= tolerance, iterations,
    initialResidual,
    residual: Number.isFinite(residual) ? residual : null, tolerance,
    reason: !Number.isFinite(residual) ? "nonfinite_state_or_equation" : residual <= tolerance ? "residual_tolerance" : "iteration_limit" };
}
function compare(reference, candidate, budget, tolerance, label) {
  const t0 = performance.now(), expected = uncached(reference, budget, tolerance), referenceMs = performance.now() - t0;
  const t1 = performance.now(), actual = candidate.settleControl(budget, tolerance), candidateMs = performance.now() - t1;
  assert.deepEqual(actual, expected, `${label}: termination differs`);
  assert.deepEqual(candidate.v, reference.v, `${label}: potentials differ`);
  assert.deepEqual(candidate.s, reference.s, `${label}: activations differ`);
  assert.equal(candidate.steps, reference.steps, `${label}: cumulative iteration count differs`);
  return { label, reference_ms: referenceMs, cached_ms: candidateMs, ratio: referenceMs / candidateMs, ...actual };
}

{
  const reference = new SettlingBrain(fixture()), candidate = new SettlingBrain(fixture());
  const both = apply => { apply(reference); apply(candidate); };
  both(b => b.stimulate("input", 2));
  compare(reference, candidate, 200, 1e-10, "cold signed recurrence");
  compare(reference, candidate, 1, 1e-10, "unchanged converged state");
  both(b => { b.drive[0] = 0.15; b.bias[2] = 0.5; b.w[0] = -0.25; });
  compare(reference, candidate, 1, 1e-12, "drive, bias and weight changed between requests");
  compare(reference, candidate, 200, 1e-10, "warm completion after capped request");
  both(b => { b.pre[1] = 0; b.dt = 0.1; });
  compare(reference, candidate, 200, 1e-10, "topology and timestep changed between requests");
  both(b => { b.v[0] = 20; b.s[0] = 0; });
  compare(reference, candidate, 3, 1e-10, "restored potential with stale activity");
  both(b => { b.w[0] = Infinity; });
  compare(reference, candidate, 3, 1e-10, "nonfinite recurrence rejects cached authority");
}

if (process.argv.includes("--payload")) {
  const payload = JSON.parse(readFileSync(new URL("../web/data/brain_full.json", import.meta.url), "utf8"));
  const reference = new SettlingBrain(payload), candidate = new SettlingBrain(payload);
  const senses = payload.lessons_setup.decision_senses;
  for (const [label, input] of [
    ["real payload cold", senses],
    ["real payload odour", { ...senses, "orn:decaying_fruit:left": 0.8, "orn:decaying_fruit:right": 0.7 }],
    ["real payload changed", { "ocelli:left": 0.1, "ocelli:right": 0.9, "lptc:hs:left": 0.9, leg_touch: 0.6 }],
  ]) {
    for (const brain of [reference, candidate]) { brain.clearStimuli(); for (const [name, level] of Object.entries(input)) brain.stimulate(name, level); }
    console.log(JSON.stringify(compare(reference, candidate, 256, 1e-6, label)));
  }
  // Timing is descriptive, not a machine-dependent test threshold or a real-time guarantee.
}
console.log("control cache differential checks passed: identical potentials, activities, residuals and solve decisions");
