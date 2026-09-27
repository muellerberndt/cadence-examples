// Controller authority tests with synthetic qualified neural replies and real body
// mechanics. These isolate causal plumbing; they do not claim the connectome has
// learned these commands or that supplied stabilizing flight is autonomous.
// Run: node tests/neural-life.mjs
import assert from "node:assert/strict";
import { NeuralLife, NEURAL_TARGET_DECISION_TTL } from "../web/neural-life.js";
import { Life } from "../web/life.js";
import { Flight, HandPilot, DT, CONTROL_KEYS, GRAVITY, hoverTrim } from "../web/body.js";
import { SETTLED_MOTOR_GROUPS, settledMotor, GAIN_TILT } from "../web/motor.js";
import { NAVIGATION_READOUTS, POLICY_OUTPUTS, decodeNavigation } from "../web/neural-policy.js";

const room = () => ({ table: { x0: 2.45, x1: 3.55, y0: 1.95, y1: 2.65, z: .75 },
  fruits: { banana: { pos: [2.88, 2.33, .795] }, bread: { pos: [3.13, 2.28, .8068] } }, surfaceZ: () => .75 });
const life = (options = {}, environment = room()) => new NeuralLife(new Flight([1.2, 1, 1.2], .3), environment, 7, { generation: 2, ...options });
const reads = (changes = {}) => ({ ...Object.fromEntries([...new Set([...SETTLED_MOTOR_GROUPS, ...NAVIGATION_READOUTS, ...POLICY_OUTPUTS])].map(name => [name, 0])), ...changes });
const forward = () => reads({ "dn:DNp09": .8, "dn:DNa02:right": .35, "dn:DNa02:left": .05 });
const planeReads = (left, right = left) => {
  const out = reads();
  for (const [side, beta] of [["left", left], ["right", right]]) {
    out[`mn:wing:${beta >= 0 ? "b2" : "i2"}:${side}`] = Math.abs(beta) / GAIN_TILT;
  }
  return out;
};
const close = (a, b, tolerance = 1e-10) => assert.ok(Math.abs(a - b) <= tolerance, `${a} != ${b}`);
const solve = () => ({ converged: true, iterations: 20, residual: 1e-8, tolerance: 1e-6, reason: "residual_tolerance" });
const observation = (request, readouts) => ({ type: "control", kind: "control", ...request, ...solve(), readouts,
  retinal: { mapping: "test-pixel-fixture", receptors: 8 } });
const observe = (fly, values = forward(), id = 1) => {
  const request = fly.expectObservation(id);
  assert.ok(request);
  assert.equal(fly.applyObservation(observation(request, values)), true);
  return request;
};
const physical = fly => [...fly.flight.p, ...fly.flight.v, ...fly.flight.w, ...fly.flight.q];
const softmax2 = values => {
  const e = values.map(x => Math.exp(x / .3));
  return e.map(x => x / (e[0] + e[1]));
};
function decision(request, fruit = "banana", action = 0) {
  const candidates = ["banana", "bread"].map(name => {
    const values = name === fruit ? [.9, .1] : [.1, .9];
    return { fruit: name, score: values[0] - values[1],
      readouts: Object.fromEntries(POLICY_OUTPUTS.map((n, i) => [n, values[i]])), solve: solve() };
  });
  const values = action === 0 ? [.9, .1] : [.1, .9];
  return { type: "assisted_decision", kind: "decision", ...request, ...solve(), accepted: true,
    readouts: Object.fromEntries(POLICY_OUTPUTS.map((n, i) => [n, values[i]])),
    targetSelection: { candidates, fruit, p: softmax2(candidates.map(c => c.score)), draw: .5, temperature: .3 },
    decision: { accepted: true, choice: action, action, p: softmax2(values), draw: .5, greedy: false,
      solves: { free: solve(), plus: solve(), minus: solve() } } };
}
const admit = (fly, fruit = "banana", action = 0, id = 1) => {
  const request = fly.openDecision(id, 2, { steps: 1024, tolerance: 1e-6, temperature: .3 });
  assert.ok(request?.selectTarget);
  assert.equal(fly.search.fruit, null, "no geometric fruit selection before neural reply");
  assert.ok(fly.applyAssistedDecision(decision(request, fruit, action)));
  return request;
};
function expectedNeutral(fly) {
  const pilot = new HandPilot(fly.flight.p.slice(), fly.flight.euler()[2], 0);
  pilot.last = { ...fly.pilot.last };
  return pilot.controls(fly.flight);
}
function neutralFields(fly) {
  assert.deepEqual(fly.authority.navigation.command, { yawRate: 0, forwardSpeed: 0, verticalSpeed: 0, feed: 0 });
  assert.equal(fly.authority.navigation.active, false);
  assert.equal(fly.speed, 0); assert.equal(fly.pilot.speed, 0);
  assert.equal(fly.pilot.route, null);
  assert.deepEqual(fly.pilot.target, fly.flight.p);
  assert.equal(fly.pilot.heading, fly.flight.euler()[2]);
}
function sameControls(actual, expected) {
  for (const key of CONTROL_KEYS) assert.ok(Math.abs(actual[key] - expected[key]) < 1e-12,
    `${key}: ${actual[key]} differs from neutral ${expected[key]}`);
}
let passed = 0, failed = 0;
function test(name, run) {
  try { run(); passed++; console.log(`ok ${name}`); }
  catch (error) { failed++; console.error(`FAIL ${name}\n${error.stack}`); }
}

test("opposite descending readouts change requested yaw and real body trajectory", () => {
  const a = life(), b = life();
  observe(a, forward());
  observe(b, reads({ "dn:DNp09": .8, "dn:DNa02:left": .35, "dn:DNa02:right": .05 }));
  for (let k = 0; k < 500; k++) { a.step(DT); b.step(DT); }
  assert.ok(a.authority.navigation.command.yawRate > 0 && b.authority.navigation.command.yawRate < 0);
  assert.equal(a.authority.navigation.command.forwardSpeed, b.authority.navigation.command.forwardSpeed);
  assert.ok(Math.hypot(...a.flight.p.map((v, k) => v - b.flight.p[k])) > 1e-3);
  assert.notDeepEqual(a.flight.q, b.flight.q);
  assert.equal(a.authority.navigation.framesApplied, 500);
});

test("zero navigation readouts remove motion request despite the remaining stabilizer", () => {
  const active = life(), lesion = life();
  observe(active); observe(lesion, reads());
  for (let k = 0; k < 500; k++) { active.step(DT); lesion.step(DT); }
  assert.deepEqual(lesion.authority.navigation.command, { yawRate: 0, forwardSpeed: 0, verticalSpeed: 0, feed: 0 });
  assert.equal(lesion.authority.navigation.framesApplied, 0);
  assert.ok(Math.hypot(...active.flight.p.map((v, k) => v - lesion.flight.p[k])) > .01);
  assert.ok(lesion.controls.f > 0, "neutral route retains declared stabilizing assistance");
});

test("motor-plane sign matches the body's supplied-hover force convention", () => {
  for (const beta of [-.03, .03]) {
    const motor = settledMotor(planeReads(beta)), bare = new Flight([1, 1, 1], 0), assisted = new Flight([1, 1, 1], 0);
    close(motor.wings[2], beta); close(motor.wings[3], beta);
    assert.equal(motor.wings[0], 0); assert.equal(motor.wings[1], 0); assert.equal(motor.wings[6], 0);
    bare.step(DT, { aL: 0, aR: 0, betaL: beta, betaR: beta, sL: 0, sR: 0, f: 0 });
    assert.equal(bare.v[0], 0, "neural steering without any power must not create thrust");
    assisted.step(DT, { ...hoverTrim(), betaL: beta, betaR: beta });
    // Two hover wings supply mg in total; positive plane rotates it forward.
    close(assisted.v[0] / DT, GRAVITY * Math.sin(beta));
    assert.equal(Math.sign(decodeNavigation(planeReads(beta)).forwardSpeed), Math.sign(assisted.v[0]));
  }
});

test("zero-DNp09 opponent motor activity reverses plane, forward request and actual motion", () => {
  const positive = life(), negative = life(), silent = life(), input = planeReads(.025);
  const swapped = reads();
  for (const side of ["left", "right"]) swapped[`mn:wing:i2:${side}`] = input[`mn:wing:b2:${side}`];
  assert.equal(input["dn:DNp09"], 0); assert.equal(swapped["dn:DNp09"], 0);
  for (const k of [2, 3]) close(settledMotor(input).wings[k], -settledMotor(swapped).wings[k]);
  const start = positive.flight.p.slice(), heading = positive.flight.euler()[2];
  observe(positive, input); observe(negative, swapped); observe(silent, reads());
  for (let k = 0; k < 500; k++) { positive.step(DT); negative.step(DT); silent.step(DT); }
  const displacement = fly => (fly.flight.p[0] - start[0]) * Math.cos(heading) + (fly.flight.p[1] - start[1]) * Math.sin(heading);
  assert.ok(displacement(positive) > .01); assert.ok(displacement(negative) < -.01);
  assert.ok(Math.abs(displacement(silent)) < .005);
  close(positive.authority.navigation.command.forwardSpeed, -negative.authority.navigation.command.forwardSpeed);
  assert.equal(silent.authority.navigation.command.forwardSpeed, 0);
  assert.equal(silent.authority.navigation.framesApplied, 0);
  assert.ok(positive.controls.aL > 0 && positive.controls.f > 0);
  assert.match(positive.authority.pilotWings, /HandPilot supplies power/);
  assert.equal(positive.authority.navigation.gains.hoverAdmittanceSeconds, 1);
  assert.equal(positive.authority.autonomous, false, "projection uses explicit pilot power, not demonstrated neural power");
});

test("asymmetric motor planes predict the declared assisted request with no cruise offset", () => {
  const input = planeReads(.021, -.031), expected = GRAVITY * (Math.sin(.021) + Math.sin(-.031)) / 2;
  assert.equal(input["dn:DNp09"], 0);
  close(decodeNavigation(input).forwardSpeed, expected);
  assert.ok(expected < -.048 && expected > -.050);
  close(decodeNavigation(input, 1).forwardSpeed, -expected);
  assert.equal(decodeNavigation(reads()).forwardSpeed, 0);
  assert.ok(decodeNavigation(reads(), 1).forwardSpeed === 0);
  close(decodeNavigation(planeReads(.227)).forwardSpeed, .3);
  close(decodeNavigation(planeReads(-.227)).forwardSpeed, -.3);
});

test("avoid action reverses motor-derived forward motion at fixed neural input", () => {
  const approach = life(), avoid = life(), input = planeReads(.025);
  const start = approach.flight.p.slice(), heading = approach.flight.euler()[2];
  admit(approach, "banana", 0); admit(avoid, "banana", 1);
  observe(approach, input, 2); observe(avoid, input, 2);
  for (let k = 0; k < 500; k++) { approach.step(DT); avoid.step(DT); }
  close(approach.authority.navigation.command.forwardSpeed, -avoid.authority.navigation.command.forwardSpeed);
  const project = fly => (fly.flight.p[0] - start[0]) * Math.cos(heading) + (fly.flight.p[1] - start[1]) * Math.sin(heading);
  assert.ok(project(approach) > .01); assert.ok(project(avoid) < -.01);
  assert.equal(approach.authority.choices.executed, 1); assert.equal(avoid.authority.choices.executed, 1);
});

test("hidden fruit and room geometry cannot redirect identical neural observations", () => {
  const alternate = room();
  alternate.fruits.banana.pos = [-100, 200, 20]; alternate.fruits.bread.pos = [70, -30, -2];
  alternate.table = { x0: -100, x1: 100, y0: -100, y1: 100, z: 50 };
  alternate.surfaceZ = () => { throw Error("route consulted room height"); };
  for (const input of [forward(), planeReads(.021, -.031)]) {
    const a = life(), b = life({}, alternate);
    observe(a, input); observe(b, input);
    for (let k = 0; k < 300; k++) {
      a.step(DT); b.step(DT); a.decide(); b.decide();
      assert.deepEqual(a.authority.navigation.command, b.authority.navigation.command);
      assert.deepEqual(a.controls, b.controls);
    }
    assert.deepEqual(physical(a), physical(b));
    assert.deepEqual(a.pilot.target, b.pilot.target);
  }
});

test("neural mode never enters inherited route, landing, takeoff or behavior routines", () => {
  const fly = life(), names = ["decide", "odourTick", "chooseLanding", "landingPoint", "step", "takeoff", "hold", "startEscape", "startGrooming"];
  const saved = new Map(names.map(name => [name, Life.prototype[name]]));
  try {
    for (const name of names) Life.prototype[name] = () => { throw Error(`inherited ${name} called`); };
    observe(fly);
    for (let k = 0; k < 200; k++) { fly.step(DT); fly.decide(); fly.odourTick(); }
    assert.equal(fly.chooseLanding(), false); assert.equal(fly.landingPoint(), null);
    assert.equal(fly.takeoff(), false); assert.equal(fly.startEscape(), false);
    assert.equal(fly.startGrooming(), false); assert.equal(fly.hold(), false);
    assert.equal(fly.landing, null); assert.equal(fly.pilot.route, null);
  } finally { for (const [name, fn] of saved) Life.prototype[name] = fn; }
});

test("brain-off cannot request or admit navigation/choices and keeps neutral requests", () => {
  const fly = life({ neuralEnabled: false });
  assert.equal(fly.expectObservation(1), false); assert.equal(fly.openDecision(1), null);
  assert.equal(fly.odourTick(), false);
  assert.equal(fly.applyObservation(observation({ requestId: 1, generation: 2, attention: null, searchToken: null }, forward())), false);
  for (let k = 0; k < 100; k++) fly.step(DT);
  assert.deepEqual(fly.authority.navigation.command, { yawRate: 0, forwardSpeed: 0, verticalSpeed: 0, feed: 0 });
  assert.equal(fly.authority.navigation.active, false);
  assert.equal(fly.authority.choices.executed, 0); assert.equal(fly.authority.observations.requested, 0);
});

test("revocation immediately restores neutral stabilizer without advancing physics", () => {
  const fly = life(); observe(fly);
  for (let k = 0; k < 100; k++) fly.step(DT);
  const expected = expectedNeutral(fly), state = physical(fly), clock = fly.clock;
  fly.revokeNeural("paused or brain off");
  neutralFields(fly); sameControls(fly.controls, expected);
  assert.deepEqual(fly.authority.executedControls, fly.controls);
  assert.deepEqual(physical(fly), state); assert.equal(fly.clock, clock);
});

test("invalid matching observation immediately withdraws old route and wing steering", () => {
  for (const mutate of [m => { m.attention = "bread"; }, m => { m.searchToken = "wrong-token"; },
    m => { m.readouts["dn:DNp09"] = NaN; }, m => { m.residual = 1; }, m => { m.converged = false; }]) {
    const fly = life(); observe(fly);
    for (let k = 0; k < 100; k++) fly.step(DT);
    const expected = expectedNeutral(fly), state = physical(fly), message = observation(fly.expectObservation(2), forward());
    mutate(message); assert.equal(fly.applyObservation(message), false);
    neutralFields(fly); sameControls(fly.controls, expected);
    assert.deepEqual(physical(fly), state);
  }
});

test("deadline expiry cannot keep an earlier heading or absolute route target", () => {
  const fly = life(), request = observe(fly);
  for (let k = 0; k < 100; k++) fly.step(DT);
  const yaw = fly.flight.euler()[2], position = fly.flight.p.slice();
  fly.clock = request.deadline;
  fly.step(DT);
  assert.deepEqual(fly.authority.navigation.command, { yawRate: 0, forwardSpeed: 0, verticalSpeed: 0, feed: 0 });
  assert.equal(fly.authority.navigation.active, false);
  assert.equal(fly.pilot.heading, yaw); assert.deepEqual(fly.pilot.target, position);
});

test("old request IDs cannot replace the current qualified route", () => {
  const fly = life(), old = observe(fly);
  observe(fly, reads({ "dn:DNp09": .1, "dn:DNa02:left": .4 }), 2);
  assert.equal(fly.applyObservation(observation(old, forward())), false);
  fly.step(DT);
  assert.ok(fly.authority.navigation.command.yawRate < 0);
});

test("admitted choice alone and preadmission observations receive no execution credit", () => {
  for (const preobserve of [false, true]) {
    const fly = life();
    if (preobserve) { observe(fly, forward(), 1); fly.step(DT); }
    admit(fly, "banana", 0, 2);
    assert.equal(fly.authority.choices.accepted, 1); assert.equal(fly.authority.choices.executed, 0);
    fly.decide(); fly.step(DT);
    assert.equal(fly.authority.choices.executed, 0);
    fly.outcome(1, "outcome before attended motion", "banana");
    assert.equal(fly.pendingReward.neuralCredit, false);
    assert.equal(fly.pendingReward.fresh, true);
  }
});

test("full-graph target decisions can take four to six seconds but execution still needs fresh attended input", () => {
  for (const delay of [4, 6]) {
    const fly = life(), request = fly.openDecision(1), message = decision(request);
    const oldInput = fly.expectObservation(1), oldMessage = observation(oldInput, forward());
    assert.equal(request.deadline, NEURAL_TARGET_DECISION_TTL);
    assert.equal(fly._pendingDecision.deadline, request.deadline);
    assert.equal(oldInput.deadline, 1, "motor freshness must remain one simulated second");
    for (let k = 0; k < Math.round(delay / DT); k++) {
      if (fly.step(DT)) { fly.decide(); fly.odourTick(); }
    }
    assert.equal(fly.search.neuralToken, request.searchToken, "the pending search must survive the legacy three-second wait");
    assert.ok(fly.applyAssistedDecision(message));
    fly.step(DT); assert.equal(fly.authority.choices.executed, 0);
    assert.equal(fly.applyObservation(oldMessage), false, "preadmission input must not execute a delayed choice");
    const attended = fly.expectObservation(2);
    close(attended.deadline - fly.clock, 1);
    assert.equal(attended.attention, "banana"); assert.equal(attended.searchToken, request.searchToken);
    assert.ok(fly.applyObservation(observation(attended, forward())));
    assert.equal(fly.authority.choices.executed, 0);
    fly.step(DT); assert.equal(fly.authority.choices.executed, 1);
    fly.outcome(1, "physical sugar after delayed choice", "banana");
    assert.equal(fly.pendingReward.neuralCredit, true);
    assert.equal(fly.pendingReward.requestId, request.requestId);
    assert.equal(fly.pendingReward.searchToken, request.searchToken);
  }
});

test("target decisions still expire at ten seconds, independently of reply arrival order", () => {
  assert.equal(NEURAL_TARGET_DECISION_TTL, 10);
  for (const tickFirst of [false, true]) {
    const fly = life(), request = fly.openDecision(1), message = decision(request);
    fly.clock = request.deadline - .001; fly.odourTick();
    assert.ok(fly._pendingDecision);
    fly.clock = request.deadline;
    if (tickFirst) { fly.odourTick(); assert.equal(fly.search, null); }
    assert.equal(fly.applyAssistedDecision(message), false);
    fly.step(DT);
    assert.equal(fly.authority.choices.accepted, 0);
    assert.equal(fly.authority.choices.executed, 0);
    assert.equal(fly.authority.navigation.active, false);
  }
});

test("a longer target window cannot authorize replaced searches or old generations", () => {
  const fly = life(), oldRequest = fly.openDecision(1), oldMessage = decision(oldRequest);
  fly.clock = 4; fly.revokeNeural("old search cancelled");
  const current = fly.openDecision(2);
  assert.notEqual(current.searchToken, oldRequest.searchToken);
  assert.equal(fly.applyAssistedDecision(oldMessage), false);
  assert.equal(fly._pendingDecision.requestId, current.requestId);
  const reset = life({ generation: 3 }), newRequest = reset.openDecision(1, 3);
  assert.equal(reset.applyAssistedDecision(oldMessage), false);
  assert.equal(reset._pendingDecision.searchToken, newRequest.searchToken);
  assert.equal(reset.authority.choices.accepted, 0);
});

test("the delayed target window does not relax candidate or free/nudged residual qualification", () => {
  for (const fail of [m => { m.targetSelection.candidates[0].solve.residual = 1e-3; },
    ...["free", "plus", "minus"].map(phase => m => { m.decision.solves[phase].converged = false; })]) {
    const fly = life(), request = fly.openDecision(1), message = decision(request);
    fly.clock = 6; fail(message);
    assert.equal(fly.applyAssistedDecision(message), false);
    assert.equal(fly.authority.choices.accepted, 0);
    assert.equal(fly.authority.choices.executed, 0);
  }
});

test("execution needs a matched postadmission attended observation and nonzero applied route", () => {
  const zero = life(); admit(zero);
  const silent = zero.expectObservation(2);
  assert.equal(silent.attention, "banana"); assert.equal(silent.searchToken, "2:1");
  assert.ok(zero.applyObservation(observation(silent, reads()))); zero.step(DT);
  assert.equal(zero.authority.choices.executed, 0);
  const fly = life(), request = admit(fly, "bread");
  const attended = fly.expectObservation(2);
  assert.equal(attended.attention, "bread"); assert.equal(attended.searchToken, request.searchToken);
  assert.ok(fly.applyObservation(observation(attended, forward())));
  assert.equal(fly.authority.choices.executed, 0);
  fly.step(DT); assert.equal(fly.authority.choices.executed, 1);
  fly.step(DT); assert.equal(fly.authority.choices.executed, 1);
  fly.outcome(1, "physical sugar on bread", "bread");
  assert.equal(fly.pendingReward.neuralCredit, true);
  for (const key of ["requestId", "generation", "searchToken"]) assert.equal(fly.pendingReward[key], request[key]);
});

test("wrong attention token or wrong fruit outcome cannot authorize credit", () => {
  const wrongToken = life(); admit(wrongToken);
  const message = observation(wrongToken.expectObservation(2), forward()); message.searchToken += ":wrong";
  assert.equal(wrongToken.applyObservation(message), false); wrongToken.step(DT);
  wrongToken.outcome(1, "unexecuted", "banana"); assert.equal(wrongToken.pendingReward.neuralCredit, false);
  const wrongFruit = life(); admit(wrongFruit); observe(wrongFruit, forward(), 2); wrongFruit.step(DT);
  wrongFruit.outcome(1, "other fruit", "bread"); assert.equal(wrongFruit.pendingReward.neuralCredit, false);
});

test("an observation still in flight cannot restore authority after its episode closes", () => {
  const fly = life(); admit(fly);
  const delayed = observation(fly.expectObservation(2), { ...forward(), "mn:wing:b2:left": 1 });
  fly.outcome(0, "episode closed before observation delivery");
  assert.equal(fly.search, null); assert.equal(fly.pendingReward.neuralCredit, false);
  assert.equal(fly.applyObservation(delayed), false);
  fly.step(DT);
  assert.equal(fly.authority.navigation.active, false);
  assert.ok(Object.values(fly.authority.neuralTrim).every(value => value === 0));
  assert.equal(fly.authority.choices.executed, 0);
});

test("approach versus avoid changes neural velocity direction without a geometric bearing", () => {
  const a = life(), b = life(); admit(a, "banana", 0); admit(b, "banana", 1);
  observe(a, forward(), 2); observe(b, forward(), 2);
  a.step(DT); b.step(DT);
  assert.equal(a.authority.navigation.command.forwardSpeed, -b.authority.navigation.command.forwardSpeed);
  assert.equal(a.authority.navigation.command.yawRate, -b.authority.navigation.command.yawRate);
  assert.equal(a.authority.choices.executed, 1); assert.equal(b.authority.choices.executed, 1);
});

test("target scores, probabilities, draw and all qualified phases must agree", () => {
  for (const mutate of [m => { m.targetSelection.candidates[0].score += .2; },
    m => { m.targetSelection.p = [.5, .5]; }, m => { m.targetSelection.fruit = "bread"; },
    m => { m.targetSelection.draw = NaN; }, m => { m.targetSelection.candidates[1].solve.residual = 1; },
    m => { m.decision.solves.minus.converged = false; }]) {
    const fly = life(), request = fly.openDecision(1), message = decision(request);
    mutate(message); assert.equal(fly.applyAssistedDecision(message), false);
    fly.step(DT); assert.equal(fly.authority.choices.executed, 0);
    assert.equal(fly.authority.targetSelection.accepted, 0);
  }
});

console.log(`${passed} neural navigation authority tests passed; ${failed} failed`);
if (failed) process.exitCode = 1;
