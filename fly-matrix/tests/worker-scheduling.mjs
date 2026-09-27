// Serial-worker fairness without changing numerical, motor or learning gates.
// Run: node tests/worker-scheduling.mjs
import assert from "node:assert/strict";
import { chooseWorkerJob, qualifiedObservationId } from "../web/worker-scheduling.js";
import { NeuralLife } from "../web/neural-life.js";
import { Flight, DT } from "../web/body.js";
import { NAVIGATION_READOUTS, POLICY_OUTPUTS } from "../web/neural-policy.js";
import { SETTLED_MOTOR_GROUPS } from "../web/motor.js";

const room = { table: { x0: 2.45, x1: 3.55, y0: 1.95, y1: 2.65, z: .75 },
  fruits: { banana: { pos: [2.88, 2.33, .795] }, bread: { pos: [3.13, 2.28, .8068] } }, surfaceZ: () => .75 };
const life = (motionSupport = true) => new NeuralLife(new Flight([2, 1, 1.2], .3), room, 1,
  { generation: 2, motionSupport });
const reads = () => ({ ...Object.fromEntries([...new Set([...SETTLED_MOTOR_GROUPS, ...NAVIGATION_READOUTS, ...POLICY_OUTPUTS])].map(k => [k, 0])), "dn:DNp09": .8 });
const solve = request => ({ type: "control", kind: "control", ...request,
  converged: true, iterations: 30, residual: 1e-8, tolerance: 1e-6, readouts: reads() });
const advance = (b, seconds) => { for (let k = 0; k < Math.round(seconds / DT); k++) b.step(DT); };
const choose = b => chooseWorkerJob({ observationDue: true, wantDecision: true,
  qualifiedObservationId: qualifiedObservationId(b, 2), lastDecisionId: b._lastDecisionId });
let passed = 0;
function test(name, run) { run(); passed++; console.log(`ok ${name}`); }

test("startup and failed admission cannot schedule a choice before a checked observation", () => {
  const b = life(); assert.equal(choose(b), "observation");
  const request = b.expectObservation(1, 2);
  assert.equal(b.applyObservation({ ...solve(request), converged: false, residual: .1 }), false);
  assert.equal(qualifiedObservationId(b, 2), null); assert.equal(choose(b), "observation");
  assert.equal(chooseWorkerJob({ observationDue: false, wantDecision: true, qualifiedObservationId: null, lastDecisionId: -1 }), null);
});

test("actual controller's two-second delayed admission no longer starves actor decisions", () => {
  const b = life(), request = b.expectObservation(1, 2);
  advance(b, 2); assert.equal(b.applyObservation(solve(request)), true);
  assert.equal(b._routeObservation, null); assert.equal(b._observation, null);
  assert.equal(qualifiedObservationId(b, 2), 1); assert.equal(choose(b), "decision");
  const decision = b.openDecision(2, 2); assert.ok(decision);
  assert.equal(choose(b), "observation", "each issued decision consumes its preceding observation turn");
  assert.equal(b.authority.navigation.framesApplied, 0); assert.equal(b.authority.choices.executed, 0);
  assert.equal(b.authority.choices.creditedOutcomes, 0, "scheduling grants no execution or learning credit");
});

test("wanted decisions and observations alternate under arbitrarily repeated slow replies", () => {
  let lastDecisionId = -1, observationId = null, nextId = 0;
  const jobs = [];
  for (let k = 0; k < 100; k++) {
    const job = chooseWorkerJob({ observationDue: true, wantDecision: true,
      qualifiedObservationId: observationId, lastDecisionId });
    jobs.push(job);
    if (job === "observation") observationId = nextId++; else lastDecisionId = nextId++;
  }
  assert.deepEqual(jobs, Array.from({ length: 100 }, (_, i) => i % 2 ? "decision" : "observation"));
  assert.equal(chooseWorkerJob({ observationDue: true, wantDecision: false,
    qualifiedObservationId: observationId, lastDecisionId: -1 }), "observation");
});

test("closed episodes, old generations, expiry and hard revocation cannot reuse admission", () => {
  const b = life(), request = b.expectObservation(1, 2); advance(b, 2); assert.ok(b.applyObservation(solve(request)));
  assert.equal(qualifiedObservationId(b, 1), null);
  b._motionSource.detached = true; assert.equal(qualifiedObservationId(b, 2), null);
  b._motionSource.detached = false; b.clock = b._motionSource.holdUntil;
  assert.equal(qualifiedObservationId(b, 2), null);
  b.clock = request.observationTime + 2; b.revokeNeural("test reset or disturbance");
  assert.equal(qualifiedObservationId(b, 2), null); assert.equal(choose(b), "observation");
});

test("failed matching solve removes an earlier good scheduling source", () => {
  const b = life(); const first = b.expectObservation(1, 2); advance(b, 2); assert.ok(b.applyObservation(solve(first)));
  assert.equal(choose(b), "decision");
  const second = b.expectObservation(2, 2);
  assert.equal(b.applyObservation({ ...solve(second), converged: false, residual: .01 }), false);
  assert.equal(choose(b), "observation"); assert.equal(b._motionSource, null);
});

test("strict fresh-only controller uses its real route and still rejects expired inputs", () => {
  const b = life(false), request = b.expectObservation(1, 2); advance(b, .5);
  assert.ok(b.applyObservation(solve(request))); assert.equal(choose(b), "decision");
  advance(b, .501); assert.equal(qualifiedObservationId(b, 2), null);
  const late = b.expectObservation(2, 2); advance(b, 1.1);
  assert.equal(b.applyObservation(solve(late)), false); assert.equal(choose(b), "observation");
});

test("malformed scheduling IDs cannot grant a decision turn", () => {
  for (const id of [null, undefined, NaN, Infinity, -1, 1.5, "2"])
    assert.equal(chooseWorkerJob({ observationDue: true, wantDecision: true, qualifiedObservationId: id, lastDecisionId: -1 }), "observation");
  assert.equal(chooseWorkerJob({ observationDue: true, wantDecision: true, qualifiedObservationId: 2, lastDecisionId: 2 }), "observation");
});
test("goal-controller admission schedules choices without inventing fresh motor authority", () => {
  const b = { neuralEnabled: true, clock: 2, _lastDecisionId: -1,
    lastQualifiedObservation: { requestId: 1, generation: 2, deadline: 4 } };
  assert.equal(choose(b), "decision");
  assert.equal(qualifiedObservationId(b, 3), null);
  b.clock = 4; assert.equal(choose(b), "observation");
  b.clock = 2; b.lastQualifiedObservation = null; assert.equal(choose(b), "observation");
});
test("independently certified goal jobs do not depend on a different unattended equilibrium", () => {
  for (const qualifiedObservationId of [null, 1, 2]) assert.equal(chooseWorkerJob({ observationDue: true,
    wantDecision: true, qualifiedObservationId, lastDecisionId: 2, decisionCertifiesInput: true }), "decision");
  assert.equal(chooseWorkerJob({ observationDue: false, wantDecision: true,
    qualifiedObservationId: null, lastDecisionId: -1, decisionCertifiesInput: true }), "decision");
});
test("independent qualification is explicit and cannot create an unwanted goal", () => {
  for (const decisionCertifiesInput of [undefined, false, 1, "true", {}]) assert.equal(chooseWorkerJob({
    observationDue: true, wantDecision: true, qualifiedObservationId: null, lastDecisionId: -1,
    decisionCertifiesInput }), "observation");
  for (const wantDecision of [false, undefined, 1]) assert.equal(chooseWorkerJob({ observationDue: true,
    wantDecision, qualifiedObservationId: null, lastDecisionId: -1, decisionCertifiesInput: true }), "observation");
  assert.equal(chooseWorkerJob({ observationDue: false, wantDecision: false,
    qualifiedObservationId: null, lastDecisionId: -1, decisionCertifiesInput: true }), null);
});
console.log(`${passed} worker scheduling groups passed`);
