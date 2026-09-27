// Fixed-input control settlement must establish a potential-equation residual, not merely
// small activation motion. Run: node tests/settlement.mjs [--payload]
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { SettlingBrain, MAX_CONTROL_STEPS } from "../web/brain.js";

const b64 = (a) => Buffer.from(a.buffer, a.byteOffset, a.byteLength).toString("base64");
function payload({ dt = 0.5, weight = 0.4, slope = 1, threshold = 0, leak = 1 } = {}) {
  return { n: 2, edges: 1, model: { dt, slope, threshold, leak, gain: 1, stimulus_amplitude: 1, adaptation: null },
    arrays: { row_ptr: b64(new Int32Array([0, 0, 1])), pre: b64(new Int32Array([0])), weight: b64(new Float64Array([weight])) },
    populations: { input: [0], output: [1], "haltere:left": [0], "vis:LC4": [1], "vis:LPLC2": [0, 1], motor: [1],
      "power:left": [1], "mn9": [1], "mn:ttm:left": [1], "mn:ttm:right": [1] } };
}

// The triangular two-cell graph has an analytic fixed point despite its directed edge.
{
  const b = new SettlingBrain(payload()); b.stimulate("input", 1);
  const r = b.settleControl(100, 1e-10);
  assert.equal(r.converged, true); assert.ok(r.iterations > 0 && r.iterations <= 100);
  assert.ok(r.residual <= r.tolerance); assert.equal(r.residual, b.equationResidual());
  assert.ok(Math.abs(b.v[0] - 1) < 1e-10);
  assert.ok(Math.abs(b.v[1] - 0.4 * Math.tanh(0.5)) < 1e-10);
  assert.equal(b.settleControl(1, 1e-10).iterations, 0);
  // A changed fixed input invalidates the previous answer and must receive a new solve.
  b.clearStimuli(); b.stimulate("input", 0.2);
  const short = b.settleControl(1, 1e-10);
  assert.equal(short.converged, false); assert.equal(short.iterations, 1);
  assert.equal(short.reason, "iteration_limit"); assert.ok(short.residual > short.tolerance);
  const next = b.settleControl(100, 1e-10);
  assert.equal(next.converged, true); assert.ok(Math.abs(b.v[0] - 0.2) < 1e-10);
}

// Saturation hides a large potential error from the legacy activation-motion rule.
{
  const b = new SettlingBrain(payload({ dt: 0.2, weight: 0, slope: 4, threshold: 1.5, leak: 0 }));
  b.v.fill(100); b.s.fill(1);
  assert.equal(b.settleFree(20, 1e-6), 1);
  assert.ok(b.equationResidual() > 70);
  const r = b.settleControl(1, 1e-6);
  assert.equal(r.converged, false); assert.equal(r.reason, "iteration_limit");
  assert.ok(r.residual > 60);
}

// Small dt cannot manufacture convergence by shrinking a step residual.
{
  const b = new SettlingBrain(payload({ dt: 1e-9 })); b.stimulate("input", 1);
  const r = b.settleControl(1, 1e-6);
  assert.equal(r.converged, false); assert.ok(r.residual > 0.99);
}

for (const steps of [0, -1, 0.5, Infinity, NaN, MAX_CONTROL_STEPS + 1]) {
  const b = new SettlingBrain(payload()), r = b.settleControl(steps, 1e-6);
  assert.equal(r.converged, false); assert.equal(r.reason, "invalid_max_steps"); assert.equal(b.steps, 0);
}
for (const tolerance of [0, -1, Infinity, NaN]) {
  const r = new SettlingBrain(payload()).settleControl(2, tolerance);
  assert.equal(r.converged, false); assert.equal(r.reason, "invalid_tolerance");
}
for (const field of ["v", "s", "drive", "bias", "w"]) {
  const b = new SettlingBrain(payload()); b[field][0] = NaN;
  const r = b.settleControl(2, 1e-6);
  assert.equal(r.converged, false); assert.equal(r.residual, null);
  assert.equal(r.reason, "nonfinite_state_or_equation");
}
{
  const b = new SettlingBrain(payload()); b.pre[0] = b.n;
  assert.equal(b.settleControl(2, 1e-6).reason, "invalid_graph");
  const c = new SettlingBrain(payload({ dt: 0 }));
  assert.equal(c.settleControl(2, 1e-6).reason, "invalid_model");
}

// Exercise the real worker protocol with deterministic in-process transport. Control replies
// must echo ownership tokens, omit action sampling, and expose no readouts on failure.
{
  const originalSelf = globalThis.self, originalFetch = globalThis.fetch;
  const originalControl = SettlingBrain.prototype.settleControl;
  let workerBrain = null, controlCalls = 0;
  SettlingBrain.prototype.settleControl = function (...args) { workerBrain = this; controlCalls++; return originalControl.apply(this, args); };
  const replies = [];
  globalThis.self = { postMessage: (m) => replies.push(m) };
  globalThis.fetch = async () => ({ json: async () => payload() });
  try {
    await import("../web/worker.js");
    const send = async (m) => { const before = replies.length; await self.onmessage({ data: m }); assert.equal(replies.length, before + 1); return replies.at(-1); };
    const before = await send({ type: "settle_control", requestId: 1, generation: 7, senses: {}, steps: 100, tolerance: 1e-6 });
    assert.equal(before.converged, false); assert.equal(before.reason, "brain_not_initialized");
    assert.equal(before.requestId, 1); assert.equal(before.generation, 7);
    const ready = await send({ type: "init", requestId: 2, generation: 8, url: "fixture", readouts: ["output"] });
    assert.equal(ready.type, "ready"); assert.equal(ready.requestId, 2); assert.equal(ready.generation, 8);
    const good = await send({ type: "settle_control", requestId: 3, generation: 8, senses: { "haltere:left": 1 }, steps: 100, tolerance: 1e-10 });
    assert.equal(good.type, "control"); assert.equal(good.kind, "control");
    assert.equal(good.requestId, 3); assert.equal(good.generation, 8);
    assert.equal(good.converged, true); assert.equal(good.tolerance, 1e-10); assert.ok(good.residual <= good.tolerance);
    assert.ok(Number.isFinite(good.readouts.output)); assert.ok("mn:ttm:left" in good.readouts);
    assert.ok(good.s instanceof Float32Array); assert.ok(!("decision" in good));
    // Do not let arbitrary known names act as input ports, or let a declared sensory alias
    // inject into motor indices. A valid first entry cannot partially change the old drive.
    const snapshot = () => ({ v: [...workerBrain.v], s: [...workerBrain.s], drive: [...workerBrain.drive],
      bias: [...workerBrain.bias], weights: [...workerBrain.w], steps: workerBrain.steps, controlCalls });
    const unchanged = snapshot();
    for (const [senses, reason] of [
      [{ "haltere:left": 0.2, "power:left": 1 }, "undeclared_sensory_population:power:left"],
      [{ mn9: 1 }, "undeclared_sensory_population:mn9"],
      [{ input: 1 }, "undeclared_sensory_population:input"],
      [{ "vis:LC4": 1 }, "sensory_motor_overlap:vis:LC4"],
      [{ "haltere:left": 0.2, "vis:LPLC2": 1 }, "sensory_motor_overlap:vis:LPLC2"],
    ]) {
      const bad = await send({ type: "settle_control", requestId: 31, generation: 8, senses, steps: 100, tolerance: 1e-10 });
      assert.equal(bad.converged, false); assert.equal(bad.reason, reason); assert.ok(!("readouts" in bad));
      assert.deepEqual(snapshot(), unchanged, "rejected sensory request changed neural state or called the solver");
    }
    const capped = await send({ type: "settle_control", requestId: 4, generation: 8, senses: {}, steps: 1, tolerance: 1e-10 });
    assert.equal(capped.converged, false); assert.equal(capped.reason, "iteration_limit");
    assert.ok(!("readouts" in capped)); assert.ok(!("s" in capped));
    for (const senses of [{ missing: 1 }, { "haltere:left": NaN }, { "haltere:left": -1 }, { "haltere:left": 1.01 }, null, []]) {
      const bad = await send({ type: "settle_control", requestId: 5, generation: 9, senses, steps: 100, tolerance: 1e-6 });
      assert.equal(bad.type, "control"); assert.equal(bad.converged, false);
      assert.equal(bad.requestId, 5); assert.equal(bad.generation, 9); assert.ok(!("readouts" in bad));
    }
    const reset = await send({ type: "reset", requestId: 6, generation: 10 });
    assert.equal(reset.type, "reset"); assert.equal(reset.requestId, 6); assert.equal(reset.generation, 10);
    const legacy = await send({ type: "run", stimuli: { input: 1 }, steps: 1 });
    assert.equal(legacy.type, "state"); assert.ok(!("converged" in legacy));
    globalThis.fetch = async () => { throw new Error("fixture_fetch_failure"); };
    const failedBoot = await send({ type: "init", requestId: 7, generation: 11, url: "broken" });
    assert.equal(failedBoot.type, "error"); assert.equal(failedBoot.requestId, 7); assert.equal(failedBoot.generation, 11);
  } finally { globalThis.self = originalSelf; globalThis.fetch = originalFetch; SettlingBrain.prototype.settleControl = originalControl; }
}

if (process.argv.includes("--payload")) {
  const d = JSON.parse(readFileSync(new URL("../web/data/brain_full.json", import.meta.url), "utf8"));
  const b = new SettlingBrain(d);
  for (const [name, level] of Object.entries(d.lessons_setup?.decision_senses || {})) b.stimulate(name, level);
  const start = performance.now(), r = b.settleControl(256, 1e-6);
  assert.ok(r.iterations <= 256); assert.ok(Number.isFinite(r.residual));
  assert.equal(r.converged, r.residual <= r.tolerance);
  assert.equal(r.residual, b.equationResidual());
  console.log(`payload control diagnostic: ${JSON.stringify({ n: b.n, edges: b.edges, ...r, ms: performance.now() - start })}`);
  // This smoke check reports either result honestly. It is not a flight or convergence gate.
}
console.log("control settlement checks passed: equation residual, caps, saturation, nonfinite rejection, declared sensory ports, motor-overlap rejection without mutation, worker ownership and legacy isolation");
