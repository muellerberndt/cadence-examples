// Measured public neural state: honest missing/stale data and request custody.
// Run: node tests/neural-state.mjs
import assert from "node:assert/strict";
import { SettlingBrain } from "../web/brain.js";
import { BrainStateTelemetry } from "../web/neural-state.js";
const b64 = a => Buffer.from(a.buffer, a.byteOffset, a.byteLength).toString("base64");
const payload = { n: 3, edges: 2, model: { dt: .5, slope: 1, threshold: 0, leak: 1, gain: 1, stimulus_amplitude: 1 },
  arrays: { row_ptr: b64(new Int32Array([0, 0, 0, 2])), pre: b64(new Int32Array([0, 1])),
    weight: b64(new Float64Array([.7, -.2])) }, populations: {} };
const message = (requestId = 1, changes = {}) => ({ type: "control", kind: "control", generation: 1, requestId,
  initialResidual: 2, residual: 1e-8, tolerance: 1e-6, converged: true, iterations: 24, active: 2, reason: "residual_tolerance", ...changes });
const context = (requestId = 1, changes = {}) => ({ generation: 1, requestId, observationTime: .25, neuronCount: 3, ...changes });
const near = (a, b) => assert.ok(Math.abs(a - b) < 2e-14, `${a} != ${b}`);
let tests = 0;
function test(name, run) { run(); tests++; console.log(`ok ${name}`); }

test("initialResidual is independently reconstructed from the complete initial potential equation", () => {
  const b = new SettlingBrain(payload); b.v.set([.4, -.6, .1]); b.s.fill(.99);
  b.drive.set([.7, 1.3, 0]); b.bias[2] = -.02;
  const activities = Array.from(b.v, v => Math.tanh(v / 2));
  const initial = Math.max(Math.abs(.7 - .4), Math.abs(1.3 + .6),
    Math.abs(.7 * activities[0] - .2 * activities[1] - .02 - .1));
  const solve = b.settleControl(256, 1e-12);
  near(solve.initialResidual, initial); assert.equal(solve.converged, true);
  const unchanged = b.settleControl(256, 1e-12);
  assert.equal(unchanged.iterations, 0); assert.equal(unchanged.initialResidual, solve.residual);
  b.drive[0] += .25;
  const changedExpected = Math.abs(b.drive[0] - b.v[0]);
  const changed = b.settleControl(256, 1e-12); near(changed.initialResidual, changedExpected);
  assert.ok(Math.abs(changed.initialResidual - .25) <= unchanged.residual,
    "input mismatch also includes the preceding solve's remaining equation defect");
});

test("initial mismatch is independent of dt, trace availability, and final convergence", () => {
  const a = new SettlingBrain(payload), b = new SettlingBrain(payload);
  a.drive[1] = b.drive[1] = 7; b.dt = 1e-9;
  const regular = a.settleControl(1, 1e-6), tiny = b.settleControl(1, 1e-6, true);
  assert.equal(regular.initialResidual, 7); assert.equal(tiny.initialResidual, 7);
  assert.equal(tiny.trace.initialResidual, 7); assert.equal(tiny.trace.frames[0].residual, 7);
  assert.equal(tiny.converged, false); assert.ok(tiny.residual > 6.9);
  const c = new SettlingBrain(payload); c.w[0] = NaN;
  assert.equal(c.settleControl(1, 1e-6).initialResidual, null);
  assert.equal(a.settleControl(0, 1e-6).initialResidual, null);
});

test("fresh public metrics copy actual counts and residuals without mutating the reply", () => {
  const state = new BrainStateTelemetry({ generation: 1 }), reply = message();
  const before = structuredClone(reply);
  assert.equal(state.observe(reply, context()), true);
  assert.deepEqual(reply, before);
  const display = state.read({ time: .5, generation: 1 });
  assert.equal(display.state, "live"); assert.equal(display.raw, 2); near(display.level, 2 / 3);
  assert.equal(display.sample.active, 2); assert.equal(display.sample.neuronCount, 3);
  assert.equal(display.sample.iterations, 24); assert.equal(display.sample.residual, 1e-8);
  reply.initialResidual = 999; reply.active = 3;
  assert.equal(state.read({ time: .5 }).raw, 2);
  assert.throws(() => { display.sample.active = 0; }, TypeError);
});

test("diagnostic, actor, wrong-generation, mismatched and replayed requests cannot replace a current observation", () => {
  const state = new BrainStateTelemetry({ generation: 1 }); state.observe(message(), context());
  for (const changes of [{ type: "retina_diagnostic" }, { diagnosticOnly: true }, { type: "assisted_decision" },
    { kind: "diagnostic" }, { generation: 2 }, { requestId: 3 }]) assert.equal(state.observe(message(2, changes), context(2)), false);
  assert.equal(state.observe(message(), context()), false);
  assert.equal(state.observe(message(2), context(2, { observationTime: .1 })), false);
  assert.equal(state.read({ time: .5 }).sample.requestId, 1);
});

test("a disturbance invalidates even unseen in-flight packets without inventing an activity or mismatch pulse", () => {
  const state = new BrainStateTelemetry({ generation: 1 }); state.observe(message(), context());
  state.invalidate("external impulse; waiting for fresh input", 5);
  let display = state.read({ time: .5 });
  assert.equal(display.state, "waiting"); assert.equal(display.raw, null); assert.equal(display.level, null); assert.equal(display.sample, null);
  assert.equal(state.observe(message(5), context(5)), false, "pre-impulse reply was issued but never observed");
  assert.equal(state.observe(message(6, { initialResidual: 0 }), context(6, { observationTime: .6 })), true);
  display = state.read({ time: .6 }); assert.equal(display.raw, 0); assert.equal(display.level, 0);
});

test("TTL uses observation time, pause labels recorded values, and generation reset removes all history", () => {
  const state = new BrainStateTelemetry({ generation: 1 }); state.observe(message(), context());
  assert.equal(state.read({ time: .5, paused: true }).state, "paused");
  assert.equal(state.read({ time: .5, paused: true }).raw, 2);
  const late = state.read({ time: 1.25 }); assert.equal(late.state, "stale"); assert.equal(late.raw, null);
  assert.equal(state.read({ time: .5, enabled: false }).state, "off");
  assert.equal(state.read({ time: .5, enabled: false }).raw, null);
  assert.equal(state.read({ time: .5, generation: 2 }).sample, null);
  state.reset(2); assert.equal(state.read({ time: 0 }).state, "waiting");
  assert.equal(state.observe(message(9), context(9)), false);
  assert.throws(() => state.reset(1));
});

test("failed/capped observations retain measured mismatch but absent activity never becomes zero", () => {
  const state = new BrainStateTelemetry({ generation: 1 });
  const capped = message(1, { converged: false, reason: "iteration_limit", residual: .4 }); delete capped.active;
  assert.equal(state.observe(capped, context()), true);
  const display = state.read({ time: .5 }); assert.equal(display.raw, 2); assert.equal(display.sample.active, null);
  assert.equal(display.sample.converged, false); assert.equal(display.sample.residual, .4);
  for (const changes of [{ initialResidual: NaN }, { initialResidual: null }, { active: -1 }, { active: 4 },
    { iterations: .5 }, { tolerance: 0 }, { residual: Infinity }, { residual: null }, { residual: .1 }]) {
    const observer = new BrainStateTelemetry({ generation: 1 });
    assert.equal(observer.observe(message(1, changes), context()), false);
    assert.equal(observer.read({ time: .5 }).available, false);
  }
});

test("activity threshold includes exactly one half, and meter arithmetic remains finite", () => {
  const b = new SettlingBrain(payload); b.s.set([.5 - Number.EPSILON, .5, .5 + Number.EPSILON]);
  assert.equal(b.activeCount(.5), 2);
  const state = new BrainStateTelemetry({ generation: 1, scale: Number.MAX_VALUE });
  state.observe(message(1, { initialResidual: Number.MAX_VALUE }), context());
  assert.equal(state.read({ time: .5 }).level, .5);
});

console.log(`${tests} public neural state groups passed`);
