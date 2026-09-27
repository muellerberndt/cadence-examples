// Assistance accounting and causal neural authority, including the full retained-connectome payload.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { Flight, DT, CONTROL_KEYS } from "../web/body.js";
import { AssistedLife, ASSISTED_OUTPUTS, ASSISTED_PLANE_LIMIT, ASSISTED_SHIFT_LIMIT } from "../web/assisted-life.js";
import { SETTLED_MOTOR_GROUPS } from "../web/motor.js";
import { SettlingBrain } from "../web/brain.js";

const room = () => ({ table: { x0: 2.45, x1: 3.55, y0: 1.95, y1: 2.65, z: .75 },
  fruits: { banana: { pos: [2.88, 2.33, .795] }, bread: { pos: [3.13, 2.28, .8068] } }, surfaceZ: () => .75 });
const life = () => new AssistedLife(new Flight([1.2, 1, 1.2], .3), room(), 7, { generation: 2 });
const zeroReads = () => Object.fromEntries(SETTLED_MOTOR_GROUPS.map(name => [name, 0]));
const solve = () => ({ converged: true, iterations: 20, residual: 1e-8, tolerance: 1e-6, reason: "residual_tolerance" });
const observation = (request, readouts) => ({ type: "control", kind: "control", ...request, ...solve(), readouts });
const open = (fly, id = 1) => {
  const pos = fly.fruits.banana.pos;
  fly.flight.p = [pos[0] - .04, pos[1], pos[2] + .05];
  fly.search = { since: fly.clock, askedAt: fly.clock, fruit: "banana", asked: true, landed: null };
  fly.valence = { fruit: "banana", action: 0, p: null, innate: true };
  return fly.openDecision(id, 2, { steps: 1024, tolerance: 1e-6, temperature: .3 });
};
const decision = (request, values = [.1, .9], draw = .5) => {
  const logits = values.map(v => Math.exp(v / request.temperature));
  const p = [logits[0] / (logits[0] + logits[1]), logits[1] / (logits[0] + logits[1])];
  const action = draw > p[0] ? 1 : 0;
  return { type: "assisted_decision", kind: "decision", ...request, ...solve(), accepted: true,
    readouts: Object.fromEntries(ASSISTED_OUTPUTS.map((name, i) => [name, values[i]])),
    decision: { accepted: true, choice: action, action, p, draw, greedy: false,
      solves: { free: solve(), plus: solve(), minus: solve() } } };
};
let tests = 0;
const test = (name, run) => { run(); tests++; console.log(`ok ${name}`); };

test("assistance is explicit and independently keeps the body moving", () => {
  const fly = life(), start = [...fly.flight.p];
  fly.bout = 100; // Hold the declared flight bout open while testing the pilot.
  for (let i = 0; i < 4000; i++) { if (fly.step(DT)) fly.decide(); }
  assert.ok(Math.hypot(fly.flight.p[0] - start[0], fly.flight.p[1] - start[1]) > .1);
  assert.ok(fly.flight.p[2] > .5);
  assert.equal(fly.authority.autonomous, false);
  assert.match(fly.authority.pilotWings, /HandPilot/);
  assert.match(fly.authority.behaviorFallback, /perch/);
  assert.equal(fly.authority.choices.executed, 0);
  assert.deepEqual(fly.authority.executedControls, fly.authority.pilotControls);
});

test("checked motor trim changes real forces while pilot power and frequency are untouched", () => {
  const active = life(), ablated = life(), reads = zeroReads();
  reads["mn:wing:b2:left"] = 1; reads["mn:wing:iii1:left"] = 1;
  reads["power:left"] = 1; // decoded power is explicitly not used in this assisted path
  assert.ok(active.applyObservation(observation(active.expectObservation(1), reads)));
  assert.ok(ablated.applyObservation(observation(ablated.expectObservation(1), zeroReads())));
  active.step(DT); ablated.step(DT);
  assert.ok(Math.abs(active.authority.neuralTrim.betaL - ASSISTED_PLANE_LIMIT) < 1e-14);
  assert.ok(Math.abs(active.authority.neuralTrim.sL - ASSISTED_SHIFT_LIMIT) < 1e-14);
  for (const key of ["aL", "aR", "f"]) assert.equal(active.controls[key], ablated.controls[key]);
  assert.notDeepEqual(active.flight.v, ablated.flight.v);
  assert.notDeepEqual(active.flight.w, ablated.flight.w);
  assert.equal(active.authority.observations.framesApplied, 1);
  assert.equal(ablated.authority.observations.framesApplied, 0);
  for (const key of CONTROL_KEYS) assert.equal(active.authority.pilotControls[key] + active.authority.neuralTrim[key], active.controls[key]);
});

test("invalid, stale or expired observations cannot supply trim", () => {
  for (const mutate of [m => { m.residual = 1; }, m => { m.converged = false; }, m => { delete m.readouts["mn9"]; }, m => { m.readouts["mn9"] = NaN; }, m => { m.generation++; }]) {
    const fly = life(), m = observation(fly.expectObservation(1), zeroReads()); mutate(m);
    assert.equal(fly.applyObservation(m), false); fly.step(DT);
    assert.ok(Object.values(fly.authority.neuralTrim).every(v => v === 0));
  }
  const fly = life(), request = fly.expectObservation(1);
  fly.clock = request.deadline;
  assert.equal(fly.applyObservation(observation(request, zeroReads())), false);
  const fresh = life(), reads = zeroReads(); reads["mn:wing:b2:left"] = 1;
  fresh.applyObservation(observation(fresh.expectObservation(1), reads));
  fresh.clock = 1; fresh.step(DT);
  assert.ok(Object.values(fresh.authority.neuralTrim).every(v => v === 0));
});

test("revocation immediately removes displayed/applied trim without a physics step", () => {
  const fly = life(), reads = zeroReads(); reads["mn:wing:b2:left"] = 1;
  assert.ok(fly.applyObservation(observation(fly.expectObservation(1), reads)));
  fly.step(DT);
  assert.notEqual(fly.controls.betaL, fly.authority.pilotControls.betaL);
  const pilot = { ...fly.authority.pilotControls }, state = [...fly.flight.p, ...fly.flight.v, fly.clock];
  fly.revokeNeural("paused");
  assert.deepEqual(fly.controls, pilot);
  assert.deepEqual(fly.authority.executedControls, pilot);
  assert.ok(Object.values(fly.authority.neuralTrim).every(v => v === 0));
  assert.ok(Object.values(fly.authority.requestedNeuralTrim).every(v => v === 0));
  assert.deepEqual([...fly.flight.p, ...fly.flight.v, fly.clock], state);
  fly.mode = "landed"; fly.revokeNeural("perched");
  assert.ok(Object.values(fly.controls).every(v => v === 0));
  assert.ok(Object.values(fly.authority.executedControls).every(v => v === 0));
});

test("unchecked legacy readouts and choices do not become neural decisions", () => {
  const a = life(), b = life();
  a.readouts = { gf: 1, "dn:landing": 1, "dn:DNa02:right": 1, "mn9": 1 };
  a.setBaseline({ gf: 0 });
  assert.equal(a.applyDecision({ action: 1, p: [0, 1] }), false);
  a.decide(true); b.decide(true);
  assert.equal(a.heading, b.heading); assert.equal(a.escape, b.escape);
  assert.deepEqual(a.landing, b.landing);
  assert.equal(a.authority.choices.accepted, 0);
});

test("checked MBON choice changes actual navigation and is counted only on execution", () => {
  const approach = life(), avoid = life();
  const a = open(approach), b = open(avoid);
  assert.ok(approach.applyAssistedDecision(decision(a, [.9, .1])));
  assert.ok(avoid.applyAssistedDecision(decision(b, [.1, .9])));
  assert.equal(avoid.authority.choices.accepted, 1);
  assert.equal(avoid.authority.choices.executed, 0);
  approach.decide(); avoid.decide();
  assert.equal(approach.authority.choices.executed, 1); assert.equal(avoid.authority.choices.executed, 1);
  assert.notEqual(approach.heading, avoid.heading);
  assert.ok(approach.landing); assert.equal(avoid.landing, null);
  avoid.decide(); assert.equal(avoid.authority.choices.executed, 1);
});

test("all three phase residuals and readout-derived probabilities are required", () => {
  for (const mutate of [m => { m.accepted = false; }, m => { m.decision.solves.plus.residual = .01; },
    m => { m.decision.solves.minus.converged = false; }, m => { delete m.decision.solves.free; },
    m => { m.decision.p = [.5, .5]; }, m => { m.decision.draw = 1; }, m => { m.decision.action = 0; },
    m => { m.readouts[ASSISTED_OUTPUTS[0]] = Infinity; }, m => { m.readouts[ASSISTED_OUTPUTS[0]] = 1e308; }]) {
    const fly = life(), request = open(fly), message = decision(request); mutate(message);
    assert.equal(fly.applyAssistedDecision(message), false);
    assert.equal(fly.valence.innate, true);
    assert.equal(fly.authority.choices.executed, 0);
  }
});

test("duplicates, wrong tokens, new searches, arrivals after landing and timeouts are rejected", () => {
  for (const change of [(f, m) => { m.searchToken += "old"; }, (f, m) => { m.generation++; },
    f => { f.search = { ...f.search }; }, f => { f.search.landed = "banana"; },
    f => { f.mode = "landed"; }, f => { f.clock += 3.01; }]) {
    const fly = life(), message = decision(open(fly)); change(fly, message);
    assert.equal(fly.applyAssistedDecision(message), false);
    assert.equal(fly.authority.choices.accepted, 0);
  }
  const fly = life(), message = decision(open(fly));
  assert.ok(fly.applyAssistedDecision(message));
  assert.equal(fly.applyAssistedDecision(message), false);
  assert.equal(fly.authority.choices.accepted, 1);
});

test("reward credit requires an executed matching neural choice, never a retroactive one", () => {
  const fly = life(); fly.applyAssistedDecision(decision(open(fly), [.9, .1]));
  fly.outcome(1, "before execution", "banana");
  assert.equal(fly.pendingReward.neuralCredit, false); assert.equal(fly.pendingReward.fresh, true);
  const executed = life(); executed.applyAssistedDecision(decision(open(executed), [.9, .1])); executed.decide();
  executed.outcome(1, "sugar on banana", "banana");
  assert.equal(executed.pendingReward.neuralCredit, true);
  assert.equal(executed.pendingReward.searchToken, "2:1");
  const wrong = life(); wrong.applyAssistedDecision(decision(open(wrong), [.9, .1])); wrong.decide();
  wrong.outcome(1, "sugar on other fruit", "bread");
  assert.equal(wrong.pendingReward.neuralCredit, false);
  const revoked = life(); revoked.applyAssistedDecision(decision(open(revoked), [.9, .1]));
  assert.deepEqual(revoked.revokeNeural("reset"), { requestId: 1, generation: 2, searchToken: "2:1", reason: "reset" });
  revoked.decide();
  assert.equal(revoked.authority.choices.executed, 0);
  revoked.outcome(1, "revoked", "banana");
  assert.equal(revoked.pendingReward.neuralCredit, false);
});

test("actual 150802-neuron settled motor readouts causally change the assisted body against an output lesion", () => {
  const payload = JSON.parse(readFileSync(new URL("../web/data/brain_full.json", import.meta.url), "utf8"));
  assert.equal(payload.n, 150802);
  assert.equal(payload.n, payload.whole.neurons);
  assert.equal(payload.edges, payload.whole.edges);
  const brain = new SettlingBrain(payload);
  // Fixed level-flight sensory fixture; no motor drive, reward, teacher or tuning.
  const senses = { "haltere:left": .5, "haltere:right": .5, "ocelli:left": Math.SQRT1_2, "ocelli:right": Math.SQRT1_2,
    "lptc:hs:left": .5, "lptc:vs:left": .5, "lptc:hs:right": .5, "lptc:vs:right": .5,
    "orn:decaying_fruit:left": .0883182888003951, "orn:decaying_fruit:right": .0883182888003951,
    "orn:yeasty:left": .0883182888003951, "orn:yeasty:right": .0883182888003951 };
  for (const [name, level] of Object.entries(senses)) brain.stimulate(name, level);
  const solved = brain.settleControl(1024, 1e-6); assert.equal(solved.converged, true);
  const reads = Object.fromEntries(SETTLED_MOTOR_GROUPS.map(name => [name, brain.mean(name)]));
  const measured = life(), lesion = life();
  for (const [fly, readouts] of [[measured, reads], [lesion, zeroReads()]]) {
    const request = fly.expectObservation(1);
    assert.ok(fly.applyObservation({ type: "control", kind: "control", ...request, ...solved, readouts }));
    fly.step(DT);
  }
  for (const key of ["aL", "aR", "f"]) assert.equal(measured.controls[key], lesion.controls[key]);
  assert.ok(Object.values(measured.authority.neuralTrim).some(v => Math.abs(v) > 1e-10));
  assert.notDeepEqual(measured.flight.w, lesion.flight.w);
  assert.notDeepEqual(measured.flight.v, lesion.flight.v);
  console.log(`  real payload: residual=${solved.residual}, iterations=${solved.iterations}, applied motor trim=${JSON.stringify(measured.authority.neuralTrim)}`);
});

console.log(`${tests} assisted authority/causality tests passed`);
