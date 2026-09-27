// Supplied-motion mechanics and custody checks. Synthetic checked readouts
// isolate the adapter; these are not evidence of learned connectome flight.
// Run: node tests/neural-motion-support.mjs
import assert from "node:assert/strict";
import { NeuralLife, MOTION_SUPPORT_DEFAULTS } from "../web/neural-life.js";
import { smoothVelocity, motionSupportOptions } from "../web/motion-support.js";
import { Flight, HandPilot, DT, wrap_ } from "../web/body.js";
import { SETTLED_MOTOR_GROUPS, GAIN_TILT } from "../web/motor.js";
import { NAVIGATION_READOUTS, POLICY_OUTPUTS } from "../web/neural-policy.js";
const room = () => ({ table: { x0: 2.45, x1: 3.55, y0: 1.95, y1: 2.65, z: .75 },
  fruits: { banana: { pos: [2.88, 2.33, .795] }, bread: { pos: [3.13, 2.28, .8068] } }, surfaceZ: () => .75 });
const life = (options = {}, environment = room()) => new NeuralLife(new Flight([2.2, 1.5, 1.2], .3), environment, 7, { generation: 2, motionSupport: true, ...options });
const reads = (changes = {}) => ({ ...Object.fromEntries([...new Set([...SETTLED_MOTOR_GROUPS, ...NAVIGATION_READOUTS, ...POLICY_OUTPUTS])].map(n => [n, 0])), ...changes });
const backward = () => reads({ "mn:wing:b2:left": .021 / GAIN_TILT, "mn:wing:i2:right": .031 / GAIN_TILT });
const forward = () => reads({ "dn:DNp09": .8 });
const close = (a, b, tolerance = 1e-10) => assert.ok(Math.abs(a - b) <= tolerance, a + " differs from " + b);
const solve = () => ({ converged: true, iterations: 20, residual: 1e-8, tolerance: 1e-6, reason: "residual_tolerance" });
const observe = (fly, values = backward(), delay = 0) => {
  const request = fly.expectObservation(fly._lastObservationId + 1);
  assert.ok(request); advance(fly, delay);
  assert.ok(fly.applyObservation({ type: "control", kind: "control", ...request, ...solve(), readouts: values }));
  return request;
};
function advance(fly, seconds, callback = () => {}) {
  for (let k = 0; k < Math.round(seconds / DT); k++) { fly.step(DT); callback(); }
}
function decision(request) {
  const candidates = ["banana", "bread"].map((fruit, i) => {
    const values = i ? [.1, .9] : [.9, .1];
    return { fruit, score: values[0] - values[1], readouts: Object.fromEntries(POLICY_OUTPUTS.map((n, j) => [n, values[j]])), solve: solve() };
  });
  const probability = 1 / (1 + Math.exp(-.8 / .3));
  const targetProbability = 1 / (1 + Math.exp(-1.6 / .3));
  return { type: "assisted_decision", kind: "decision", ...request, ...solve(), accepted: true,
    readouts: Object.fromEntries(POLICY_OUTPUTS.map((n, i) => [n, i ? .1 : .9])),
    targetSelection: { candidates, fruit: "banana", p: [targetProbability, 1 - targetProbability], draw: .5, temperature: .3 },
    decision: { accepted: true, choice: 0, action: 0, p: [probability, 1 - probability], draw: .5, greedy: false,
      solves: { free: solve(), plus: solve(), minus: solve() } } };
}
const admit = fly => { const request = fly.openDecision(fly._lastDecisionId + 1); assert.ok(request); assert.ok(fly.applyAssistedDecision(decision(request))); return request; };
let passed = 0, failed = 0;
function test(name, run) { try { run(); passed++; console.log("ok " + name); } catch (e) { failed++; console.error("FAIL " + name + "\n" + e.stack); } }

test("strict remains opt-out and adapter parameters are declared, bounded candidates", () => {
  assert.equal(life({ motionSupport: false }).authority.motionSupport.enabled, false);
  assert.deepEqual(life().authority.motionSupport.config, MOTION_SUPPORT_DEFAULTS);
  assert.equal(MOTION_SUPPORT_DEFAULTS.cruiseSpeed, .3);
  for (const option of [null, { mystery: 1 }, { cruiseSpeed: NaN }, { holdSeconds: 99 }, { maxObservationAgeSeconds: .5 }])
    assert.throws(() => motionSupportOptions(option));
});

test("smoothing respects the full-vector acceleration bound through a direction reversal", () => {
  let velocity = [0, 0, 0];
  for (let k = 0; k < 4000; k++) {
    const desired = k < 2000 ? [.3, .1, .12] : [-.3, -.1, -.12];
    const next = smoothVelocity(velocity, desired, DT, MOTION_SUPPORT_DEFAULTS);
    assert.ok(Math.hypot(...next.map((x, i) => x - velocity[i])) <= MOTION_SUPPORT_DEFAULTS.maxAcceleration * DT + 1e-14);
    assert.ok(Math.hypot(...next.map((x, i) => x - desired[i])) < Math.hypot(...velocity.map((x, i) => x - desired[i])));
    velocity = next;
  }
});

test("a checked direction yields sustained near-0.3m/s physical travel through an input gap", () => {
  const fly = life(), start = fly.flight.p.slice(), speeds = [];
  observe(fly); advance(fly, 4.8, () => { if (fly.clock > 2) speeds.push(Math.hypot(fly.flight.v[0], fly.flight.v[1])); });
  const median = speeds.sort((a, b) => a - b)[Math.floor(speeds.length / 2)];
  const support = fly.authority.motionSupport;
  assert.ok(median > .25 && median < .35, "actual median speed " + median);
  assert.ok(Math.hypot(fly.flight.p[0] - start[0], fly.flight.p[1] - start[1]) > 1);
  assert.equal(fly.authority.navigation.active, false); assert.equal(support.phase, "held");
  assert.ok(support.heldFrames > 0 && support.delayedFrames > 0);
  assert.ok(fly.authority.navigation.framesApplied < 2001);
  assert.ok(Object.values(fly.authority.neuralTrim).every(x => x === 0));
  const headingError = Math.abs(wrap_(fly.flight.euler()[2] - Math.atan2(fly.flight.v[1], fly.flight.v[0])));
  assert.ok(headingError < .05);
  console.log("  actual median speed " + median.toFixed(6) + " m/s; facing error " + headingError.toFixed(6) + " rad");
});

test("delayed checked execution earns its own event record without masquerading as fresh motor activity", () => {
  const fly = life(); admit(fly);
  const request = observe(fly, backward(), 2);
  close(request.deadline - request.observationTime, 4);
  advance(fly, .5);
  assert.equal(fly.authority.motionSupport.phase, "delayed_checked");
  assert.ok(fly.authority.motionSupport.delayedFrames > 0);
  assert.equal(fly.authority.navigation.framesApplied, 0);
  assert.equal(fly.authority.choices.executed, 1);
  assert.equal(fly.authority.execution.current.phase, "delayed_checked");
  assert.equal(fly.authority.execution.current.controlRequestId, request.requestId);
  assert.ok(Object.values(fly.authority.neuralTrim).every(x => x === 0));
  fly.outcome(1, "contact during delayed support", "banana");
  assert.equal(fly.pendingReward.neuralCredit, true);
  assert.equal(fly.pendingReward.executionEvidence.phase, "delayed_checked");
});

test("routine attention and episode transitions retain course but detach old credit", () => {
  const fly = life(); observe(fly); advance(fly, 1.2);
  const before = fly.authority.motionSupport.velocityWorld.slice(), course = fly.heading;
  admit(fly);
  fly.step(DT);
  assert.equal(fly.authority.motionSupport.phase, "held");
  assert.equal(fly.authority.motionSupport.source.detached, true);
  assert.equal(fly.authority.choices.executed, 0); close(fly.heading, course);
  assert.ok(Math.hypot(...fly.authority.motionSupport.velocityWorld) > .29);
  assert.ok(Math.hypot(...fly.authority.motionSupport.velocityWorld.map((x, i) => x - before[i])) <= DT + 1e-12);
  fly.outcome(0, "routine episode ended"); fly.step(DT);
  assert.ok(fly.authority.motionSupport.active);
  assert.equal(fly.pendingReward.neuralCredit, false); close(fly.heading, course);
});

test("an earlier executed choice retains bounded outcome credit after its motor input expires", () => {
  const fly = life(); admit(fly); observe(fly, forward()); fly.step(DT);
  assert.equal(fly.authority.choices.executed, 1);
  advance(fly, 1.1);
  fly.outcome(1, "contact after source freshness ended", "banana");
  assert.equal(fly.pendingReward.neuralCredit, true);
  assert.equal(fly.pendingReward.executionEvidence.phase, "fresh");
  assert.equal(fly.authority.choices.creditedOutcomes, 1);
  assert.equal(fly.authority.choices.uncreditedOutcomes, 0);
  const fresh = life(); admit(fresh); observe(fresh, forward()); fresh.step(DT);
  fresh.outcome(1, "contact with fresh attended input", "banana");
  assert.equal(fresh.pendingReward.neuralCredit, true);
  assert.equal(fresh.authority.choices.creditedOutcomes, 1);
});

test("zero input supplies no direction and a new zero command smoothly stops prior cruise", () => {
  const silent = life(); observe(silent, reads()); advance(silent, 2);
  assert.deepEqual(silent.authority.motionSupport.velocityWorld, [0, 0, 0]);
  assert.equal(silent.authority.motionSupport.appliedFrames, 0);
  const fly = life(); observe(fly); advance(fly, 1.5); observe(fly, reads());
  let lastSpeed = Math.hypot(...fly.authority.motionSupport.velocityWorld);
  advance(fly, 3, () => {
    const speed = Math.hypot(...fly.authority.motionSupport.velocityWorld);
    assert.ok(speed <= lastSpeed + 1e-14); lastSpeed = speed;
  });
  assert.ok(lastSpeed < 2e-6);
  assert.ok(Math.hypot(fly.flight.v[0], fly.flight.v[1]) < 1e-5);
});

test("lease expiry stops new directional execution while hard revoke clears support immediately", () => {
  const fly = life(); observe(fly); advance(fly, 1.2); const deadline = fly._motionSource.holdUntil;
  fly.clock = deadline; fly.step(DT);
  assert.equal(fly.authority.motionSupport.source, null);
  assert.equal(fly.authority.motionSupport.phase, "stopping");
  assert.deepEqual(fly.authority.motionSupport.desiredVelocityWorld, [0, 0, 0]);
  advance(fly, 3);
  assert.ok(Math.hypot(fly.flight.v[0], fly.flight.v[1]) < 1e-5);
  const active = life(); observe(active); advance(active, 1.2);
  const request = active.expectObservation(active._lastObservationId + 1);
  active.revokeNeural("pause, reset or disturbance");
  assert.deepEqual(active.authority.motionSupport.velocityWorld, [0, 0, 0]);
  assert.equal(active.authority.motionSupport.active, false);
  assert.deepEqual(active.pilot.target, active.flight.p);
  assert.equal(active.applyObservation({ type: "control", kind: "control", ...request, ...solve(), readouts: backward() }), false);
  active.step(DT); assert.equal(active.authority.motionSupport.source, null);
});

test("disabled neural control cannot retain held motion or reinstate delayed input", () => {
  const fly = life(); observe(fly); advance(fly, 1.2);
  const request = fly.expectObservation(fly._lastObservationId + 1);
  fly.neuralEnabled = false; fly.step(DT);
  assert.equal(fly.authority.motionSupport.active, false);
  assert.deepEqual(fly.authority.motionSupport.velocityWorld, [0, 0, 0]);
  assert.ok(Object.values(fly.authority.neuralTrim).every(x => x === 0));
  assert.equal(fly.applyObservation({ type: "control", kind: "control", ...request, ...solve(), readouts: backward() }), false);
  assert.equal(fly._motionSource, null);
});

test("failed matching solves clear held motion and over-age or wrong-generation replies cannot restore it", () => {
  for (const reason of ["capped", "late", "wrong-kind"]) {
    const fly = life(); observe(fly); advance(fly, 1.1);
    const request = fly.expectObservation(fly._lastObservationId + 1);
    const message = { type: "control", kind: "control", ...request, ...solve(), readouts: backward() };
    if (reason === "capped") { message.converged = false; message.residual = .1; }
    if (reason === "late") fly.clock = request.deadline;
    if (reason === "wrong-kind") message.kind = "decision";
    assert.equal(fly.applyObservation(message), false);
    assert.equal(fly.authority.motionSupport.source, null);
    assert.deepEqual(fly.authority.motionSupport.velocityWorld, [0, 0, 0]);
    assert.ok(Object.values(fly.authority.neuralTrim).every(x => x === 0));
    message.generation = 1; assert.equal(fly.applyObservation(message), false);
  }
});

test("hidden fruit coordinates cannot change supported world velocity or actual trajectory", () => {
  const altered = room(); altered.fruits.banana.pos = [4, .2, .9]; altered.fruits.bread.pos = [.2, 2.5, .8];
  const a = life(), b = life({}, altered); observe(a); observe(b);
  advance(a, 2); advance(b, 2);
  assert.deepEqual(a.authority.motionSupport.velocityWorld, b.authority.motionSupport.velocityWorld);
  assert.deepEqual(a.flight.p, b.flight.p); assert.deepEqual(a.flight.q, b.flight.q);
  const v = a.authority.motionSupport.velocityWorld, p = a.flight.p.slice(); a.step(DT);
  for (let i = 0; i < 3; i++) close(HandPilot.K_POS * (a.pilot.target[i] - p[i]), a.authority.motionSupport.velocityWorld[i]);
  assert.ok(Math.hypot(...v) > .29);
});

console.log(passed + " neural motion-support tests passed; " + failed + " failed");
if (failed) process.exitCode = 1;
