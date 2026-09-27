// Episode-owned causal action records. Synthetic qualified readouts isolate
// controller custody; these tests do not establish neural task competence.
import assert from "node:assert/strict";
import { NeuralLife } from "../web/neural-life.js";
import { Flight, DT, RADIUS } from "../web/body.js";
import { NAVIGATION_READOUTS, POLICY_OUTPUTS } from "../web/neural-policy.js";
import { SETTLED_MOTOR_GROUPS } from "../web/motor.js";
import { EPISODE_S, DECISION_S } from "../web/life.js";

const environment = { table: { x0: 2.45, x1: 3.55, y0: 1.95, y1: 2.65, z: .75 },
  fruits: { banana: { pos: [2.88, 2.33, .795] }, bread: { pos: [3.13, 2.28, .8068] } }, surfaceZ: () => 0 };
const make = () => new NeuralLife(new Flight([2, 1, 1.2], .3), environment, 1, { generation: 2, motionSupport: true });
const advance = (b, seconds) => { for (let k = 0; k < Math.round(seconds / DT); k++) b.step(DT); };
const solve = () => ({ converged: true, residual: 1e-8, tolerance: 1e-6, iterations: 20 });
const reads = (moving = true) => ({ ...Object.fromEntries([...new Set([...SETTLED_MOTOR_GROUPS, ...NAVIGATION_READOUTS, ...POLICY_OUTPUTS])].map(k => [k, 0])), "dn:DNp09": moving ? .8 : 0 });
function admit(b, action = 0) {
  const request = b.openDecision(1, 2); assert.ok(request);
  const candidates = ["banana", "bread"].map((fruit, i) => ({ fruit, score: i ? -.8 : .8,
    readouts: Object.fromEntries(POLICY_OUTPUTS.map((k, j) => [k, j === i ? .9 : .1])), solve: solve() }));
  const targetP = 1 / (1 + Math.exp(-1.6 / .3)), actionP = 1 / (1 + Math.exp(-.8 / .3));
  const message = { type: "assisted_decision", kind: "decision", ...request, ...solve(), accepted: true,
    readouts: { [POLICY_OUTPUTS[0]]: .9, [POLICY_OUTPUTS[1]]: .1 },
    targetSelection: { candidates, fruit: "banana", p: [targetP, 1 - targetP], draw: .5, temperature: .3 },
    decision: { accepted: true, action, choice: action, p: [actionP, 1 - actionP], draw: action ? .99 : .5,
      greedy: false, solves: { free: solve(), plus: solve(), minus: solve() } } };
  assert.ok(b.applyAssistedDecision(message)); return request;
}
function observe(b, delay = 2, moving = true) {
  const request = b.expectObservation(Math.max(b._lastObservationId, b._lastDecisionId) + 1, 2);
  advance(b, delay);
  assert.ok(b.applyObservation({ type: "control", kind: "control", ...request, ...solve(), readouts: reads(moving) }));
  return request;
}
const executed = (delay = 2) => { const b = make(); admit(b); observe(b, delay); b.step(DT); return b; };
let passed = 0;
function test(name, run) { run(); passed++; console.log(`ok ${name}`); }

test("admission alone cannot create an action record; a real delayed physical step can", () => {
  const b = make(), decision = admit(b), observation = observe(b);
  assert.equal(b._executionEvidence, null); assert.equal(b.authority.choices.executed, 0);
  const steps = b.flight.steps; b.step(DT); assert.equal(b.flight.steps, steps + 1);
  const e = b._executionEvidence;
  assert.equal(e.decisionRequestId, decision.requestId); assert.equal(e.controlRequestId, observation.requestId);
  assert.equal(e.phase, "delayed_checked"); assert.ok(e.sourceAgeSeconds >= 2 && e.sourceAgeSeconds < 4);
  assert.equal(e.searchToken, decision.searchToken); assert.equal(e.fruit, "banana"); assert.equal(e.action, 0);
  assert.equal(e.episodeDeadline, EPISODE_S); assert.equal(e.executedAt, b.clock);
  assert.equal(b.authority.navigation.framesApplied, 0); assert.equal(b.authority.choices.delayedExecuted, 1);
  assert.ok(Object.values(b.authority.neuralTrim).every(x => x === 0));
});

test("the immutable original execution survives refresh expiry and yields one later terminal credit", () => {
  const b = executed(), original = structuredClone(b._executionEvidence);
  advance(b, 3); assert.equal(b.authority.motionSupport.phase, "held");
  observe(b, .1); b.step(DT);
  assert.deepEqual(b._executionEvidence, original, "a new observation cannot extend the original episode or rewrite execution");
  b.authority.execution.current.action = 1;
  b.outcome(1, "actual matching sugar contact", "banana");
  assert.equal(b.pendingReward.neuralCredit, true); assert.deepEqual(b.pendingReward.executionEvidence, original);
  assert.equal(b._executionEvidence, null); assert.equal(b.authority.execution.current, null);
  b.outcome(1, "duplicate contact", "banana"); assert.equal(b.pendingReward.neuralCredit, false);
  assert.equal(b.authority.choices.creditedOutcomes, 1);
});

test("held-only, detached, zero and grounded commands cannot create an execution", () => {
  const held = make(); admit(held); observe(held);
  held.clock = held._motionSource.observationTime + 4; held.step(DT);
  assert.equal(held.authority.motionSupport.phase, "held"); assert.equal(held._executionEvidence, null);
  held.outcome(-1, "contact without executed actor action"); assert.equal(held.pendingReward.neuralCredit, false);
  const detached = make(); observe(detached, 0); detached.step(DT); admit(detached); detached.step(DT);
  assert.equal(detached._motionSource.detached, true); assert.equal(detached._executionEvidence, null);
  const zero = make(); admit(zero); observe(zero, 0, false); zero.step(DT); assert.equal(zero._executionEvidence, null);
  const grounded = make(); admit(grounded); observe(grounded, 0);
  grounded.flight.p = [2, 1, RADIUS]; grounded.flight.v = [0, 0, 0]; grounded.flight.touching = 1; grounded.flight.onFloor = true;
  grounded.step(DT); assert.equal(grounded.wingsOff, true); assert.equal(grounded._executionEvidence, null);
});

test("a matching capped or malformed observation and hard revoke invalidate old credit", () => {
  for (const mode of ["capped", "token", "revoke"]) {
    const b = executed();
    if (mode === "revoke") b.revokeNeural("disturbance invalidated episode");
    else {
      const request = b.expectObservation(3, 2), message = { type: "control", kind: "control", ...request, ...solve(), readouts: reads() };
      if (mode === "capped") { message.converged = false; message.residual = .1; }
      else message.searchToken = "another episode";
      assert.equal(b.applyObservation(message), false);
    }
    assert.equal(b._executionEvidence, null); b.outcome(1, "later contact", "banana");
    assert.equal(b.pendingReward.neuralCredit, false);
  }
});

test("unmatched stale packets cannot erase an actual current execution", () => {
  const b = executed(), before = b._executionEvidence;
  const request = b.expectObservation(3, 2);
  assert.equal(b.applyObservation({ type: "control", kind: "control", ...request, requestId: 2, ...solve(), readouts: reads() }), false);
  assert.equal(b._executionEvidence, before);
  b.outcome(-1, "real later event"); assert.equal(b.pendingReward.neuralCredit, true);
});

test("old actor identity, wrong target, avoidance sugar and no-choice outcomes are not credited", () => {
  for (const mode of ["generation", "token", "request", "fruit", "avoid", "no-choice"]) {
    const b = mode === "avoid" ? make() : executed();
    if (mode === "avoid") { admit(b, 1); observe(b); b.step(DT); }
    if (mode === "generation") b.generation++;
    if (mode === "token") b.search.neuralChoice.searchToken = "new";
    if (mode === "request") b.search.neuralChoice.requestId++;
    if (mode === "no-choice") b.search.neuralChoice = null;
    b.outcome(1, "sugar contact", mode === "fruit" ? "bread" : "banana");
    assert.equal(b.pendingReward.neuralCredit, false, mode);
  }
});

test("original fifteen-second episode bounds reward, with only the scheduled zero timeout allowed just after", () => {
  const timely = executed(); timely.clock = timely._executionEvidence.episodeDeadline - DT;
  timely.outcome(-1, "physical event before deadline"); assert.equal(timely.pendingReward.neuralCredit, true);
  const late = executed(); late.clock = late._executionEvidence.episodeDeadline;
  late.outcome(1, "contact at expired deadline", "banana"); assert.equal(late.pendingReward.neuralCredit, false);
  const timeout = executed(); timeout.clock = timeout._executionEvidence.episodeDeadline + DECISION_S / 2;
  timeout.odourTick(); assert.equal(timeout.pendingReward.reward, 0); assert.equal(timeout.pendingReward.neuralCredit, true);
  const tooLate = executed(); tooLate.clock = tooLate._executionEvidence.episodeDeadline + DECISION_S + .01;
  tooLate.odourTick(); assert.equal(tooLate.pendingReward.neuralCredit, false);
  const forged = executed(); forged.clock = forged._executionEvidence.episodeDeadline + .01;
  forged.outcome(1, "not a zero timeout", "banana", { episodeTimeout: true });
  assert.equal(forged.pendingReward.neuralCredit, false);
});

test("an exact four-second source and a disabled model cannot create or preserve actor execution", () => {
  const b = make(); admit(b); observe(b, 2); b.clock = b._motionSource.observationTime + 4;
  b.step(DT); assert.equal(b._executionEvidence, null);
  const disabled = executed(); disabled.neuralEnabled = false; disabled.step(DT);
  assert.equal(disabled._executionEvidence, null);
  disabled.outcome(-1, "disabled outcome"); assert.equal(disabled.pendingReward.neuralCredit, false);
});
console.log(`${passed} neural execution-credit groups passed`);
