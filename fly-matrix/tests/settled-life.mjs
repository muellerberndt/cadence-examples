// Run: node fly-matrix/tests/settled-life.mjs from cadence-examples.
// Small deterministic authority tests; no connectome payload or training is required.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { Flight, HandPilot, RADIUS, DT } from "../web/body.js";
import { Life } from "../web/life.js";
import { SETTLED_MOTOR_GROUPS, settledMotor } from "../web/motor.js";
import { SettledLife } from "../web/settled-life.js";

const room = () => ({ fruits: { banana: { pos: [2.8, 2.3, 0.75] }, bread: { pos: [3.1, 2.3, 0.75] } }, surfaceZ: () => 0 });
const reads = (extra = {}) => ({ ...Object.fromEntries(SETTLED_MOTOR_GROUPS.map(key => [key, 0])), ...extra });
const active = () => reads({ "power:left": 0.8, "power:right": 0.6, "mn:wing:b2:left": 0.2 });
const spec = (life, requestId = life.lastRequestId + 1) => ({ requestId, generation: life.generation, tolerance: 1e-6, steps: 8 });
const reply = (request, readouts = active(), extra = {}) => ({ type: "control", kind: "control", ...request, converged: true, residual: 1e-8, iterations: 4, readouts, ...extra });
const fixture = () => new SettledLife(new Flight([1, 1, 1]), room());
const accept = (life, readouts = active()) => { const request = spec(life); assert.equal(life.expectControl(request), true); const message = reply(request, readouts); assert.equal(life.applyControl(message), true); return message; };
const zero = (life) => { assert.ok(Object.values(life.controls).every(v => v === 0)); assert.equal(life.proboscis, 0); assert.equal(life.wingsOff, true); };
let passed = 0;
function test(name, run) { run(); passed++; console.log(`ok ${name}`); }

// A component regression must not accidentally reconnect the old behavior.
HandPilot.prototype.controls = () => { throw new Error("legacy pilot called"); };
for (const name of ["decide", "odourTick", "chooseLanding", "takeoff", "startEscape", "startFeeding", "startGrooming"]) Life.prototype[name] = () => { throw new Error(`legacy ${name} called`); };

test("strict runtime imports no legacy policy and starts with zero actuators", () => {
  const source = readFileSync(new URL("../web/settled-life.js", import.meta.url), "utf8");
  assert.doesNotMatch(source, /from\s+["']\.\/life\.js["']|\bHandPilot\b|Math\.random|Math\.atan2/);
  const life = fixture(); zero(life);
  assert.equal(life.source, "patchnet");
  assert.equal("pilot" in life, false);
  assert.equal("landing" in life, false);
  for (let k = 0; k < 100; k++) life.step(DT);
  assert.ok(life.flight.p[2] < 1, "gravity must keep acting without a command");
  zero(life);
  assert.equal(life.ethogram.grooms, 0);
  assert.equal(life.ethogram.saccades, 0);
  assert.equal(life.ethogram.feeds, 0);
});

test("only a matching converged reply determines direct wing controls", () => {
  const life = fixture(), readouts = active(); accept(life, readouts);
  assert.deepEqual(Object.values(life.controls), settledMotor(readouts).wings);
  const before = { ...life.controls };
  life.step(DT);
  assert.deepEqual(life.controls, before);
  assert.equal(life.commandStatus.state, "active");
  readouts["power:left"] = 0; // Caller mutation must not change the accepted snapshot.
  assert.equal(life.readouts["power:left"], 0.8);
});

test("zero readouts and unsupported leg/grooming activity supply no fallback power", () => {
  const life = fixture(); accept(life, reads()); zero(life);
  const baseline = fixture();
  accept(life, reads({ "mn:ttm:left": 1, "mn:ttm:right": 1, "dn:grooming": 1 }));
  assert.equal(life.commandStatus.unsupportedLegDrive, true);
  life.step(DT); baseline.step(DT);
  assert.deepEqual(life.flight.p, baseline.flight.p);
  assert.deepEqual(life.flight.v, baseline.flight.v);
  assert.equal(life.pose().groom, null);
  assert.equal(life.ethogram.grooms, 0);
  zero(life);
});

test("command expires from request time, not late reply time", () => {
  const life = fixture(), request = spec(life);
  life.expectControl(request);
  life.clock = 0.1;
  assert.equal(life.applyControl(reply(request)), true);
  life.clock = 0.12;
  life.step(DT);
  zero(life); assert.equal(life.commandStatus.reason, "command expired");
  const expired = fixture(), late = spec(expired);
  expired.expectControl(late); expired.clock = expired.commandTTL;
  assert.equal(expired.applyControl(reply(late)), false); zero(expired);
});

test("missing replies expire without flight or landing policy", () => {
  const life = fixture(); life.expectControl(spec(life)); life.clock = 0.13; life.step(DT);
  zero(life); assert.equal(life.pending, null);
  assert.equal(life.commandStatus.reason, "controller reply expired");
  assert.equal(life.ethogram.landings, 0);
});

test("old motor expiry preserves a newer on-time solve without extending its deadline", () => {
  const life = fixture(); accept(life);
  for (let k = 0; k < 200; k++) life.step(DT); // t = .100, old command expires at .120.
  const request = spec(life); life.expectControl(request);
  const deadline = life.pending.deadline;
  for (let k = 0; k < 50; k++) life.step(DT); // t = .125, new solve remains valid to .220.
  zero(life);
  assert.equal(life.pending.requestId, request.requestId);
  assert.equal(life.pending.deadline, deadline);
  assert.equal(life.commandStatus.state, "awaiting");
  assert.equal(life.applyControl(reply(request)), true, "the old command cannot cancel a newer observation");
  assert.equal(life.commandDeadline, deadline, "arrival must not renew observation validity");
  assert.ok(life.controls.f > 0);
  for (let k = 0; k < 191; k++) life.step(DT); // Beyond the newer command's original deadline.
  zero(life); assert.equal(life.pending, null);
});

test("a preserved pending solve still expires and cannot restore motor power", () => {
  const life = fixture(); accept(life);
  for (let k = 0; k < 200; k++) life.step(DT);
  const request = spec(life); life.expectControl(request);
  for (let k = 0; k < 250; k++) life.step(DT);
  zero(life); assert.equal(life.pending, null);
  assert.equal(life.commandStatus.reason, "controller reply expired");
  assert.equal(life.applyControl(reply(request)), false); zero(life);
});

test("nonconverged, incomplete, NaN and mismatched replies all revoke active power", () => {
  const mutations = [
    m => { m.type = "state"; },
    m => { m.kind = "state"; },
    m => { m.requestId++; },
    m => { m.generation++; },
    m => { m.converged = false; },
    m => { m.residual = 0.01; },
    m => { m.residual = NaN; },
    m => { m.residual = -1; },
    m => { m.tolerance = 1; },
    m => { m.iterations = 9; },
    m => { m.iterations = 1.5; },
    m => { delete m.readouts[SETTLED_MOTOR_GROUPS[0]]; },
    m => { m.readouts[SETTLED_MOTOR_GROUPS[0]] = NaN; },
    m => { m.readouts[SETTLED_MOTOR_GROUPS[0]] = "1"; },
    m => { m.readouts.extra = Infinity; },
  ];
  for (const mutate of mutations) {
    const life = fixture(); accept(life);
    const request = spec(life); life.expectControl(request);
    const bad = reply(request); mutate(bad);
    assert.equal(life.applyControl(bad), false); zero(life);
    assert.equal(life.pending, null);
  }
});

test("old generations, superseded requests and duplicate replies cannot restore authority", () => {
  const life = fixture(), old = spec(life); life.expectControl(old);
  life.revokeControl("reset", 1);
  const current = spec(life); life.expectControl(current);
  assert.equal(life.applyControl(reply(old)), false); zero(life);
  const first = spec(life); life.expectControl(first);
  const second = { ...first, requestId: first.requestId + 1 }; life.expectControl(second);
  assert.equal(life.applyControl(reply(first)), false); zero(life);
  const valid = accept(life);
  assert.equal(life.applyControl(valid), false); zero(life);
  assert.equal(life.expectControl(valid), false);
  assert.throws(() => life.revokeControl("backwards", 0), RangeError);
});

test("hidden fruit geometry changes sensors but never changes accepted actuators", () => {
  const a = fixture(), b = fixture();
  accept(a); accept(b);
  const before = b.senses();
  b.fruits.banana.pos.splice(0, 3, 1.02, 1, 1);
  b.fruitMoved("banana"); b.setSugar("bread");
  const after = b.senses();
  assert.notEqual(before["orn:decaying_fruit:left"], after["orn:decaying_fruit:left"]);
  assert.deepEqual(a.controls, b.controls);
  a.step(DT); b.step(DT);
  assert.deepEqual(a.flight.p, b.flight.p);
  assert.deepEqual(a.flight.q, b.flight.q);
  assert.deepEqual(a.controls, b.controls);
  assert.equal(b.pendingReward, null);
});

test("sensory observation supplies no privileged positions or motor commands", () => {
  const life = fixture();
  const s = life.senses();
  assert.ok(Object.values(s).every(v => Number.isFinite(v) && v >= 0 && v <= 1));
  assert.ok(Object.keys(s).every(k => k.includes(":") || k === "leg_touch"));
  assert.equal("fruit" in s, false);
  assert.equal("target" in s, false);
  assert.equal("heading" in s, false);
  assert.equal(s["haltere:left"], 0, "no wing oscillation means no haltere tone");
  zero(life);
});

function contactFixture({ touching = 1, supported = true, zOffset = 0 } = {}) {
  const r = room(), B = r.fruits.banana.pos;
  const f = new Flight([B[0], B[1], B[2] + RADIUS + zOffset]);
  // Declared fixed-contact fixture: isolate ingestion gating from the collision integrator.
  f.touching = touching; f.onFloor = supported;
  f.step = () => {};
  return new SettledLife(f, r);
}

test("feeding requires proboscis drive and actual support contact on sugar", () => {
  for (const options of [{ touching: 0 }, { supported: false }, { zOffset: 0.02 }]) {
    const life = contactFixture(options), initial = life.hunger;
    accept(life, reads({ mn9: 1 })); life.step(DT);
    assert.ok(life.hunger >= initial); assert.equal(life.pose().feed, 1);
    assert.equal(life.pose().feeding, false);
    assert.equal(life.pose().neural, true);
    assert.equal(life.ethogram.feeds, 0);
    assert.equal(life.senses()["grn:sugar:labellum"], 0);
  }
  const quiet = contactFixture(), before = quiet.hunger;
  accept(quiet, reads()); quiet.step(DT);
  assert.ok(quiet.hunger >= before); assert.equal(quiet.ethogram.feeds, 0);
  const fed = contactFixture(), initial = fed.hunger;
  accept(fed, reads({ mn9: 1 })); fed.step(DT);
  assert.ok(fed.hunger < initial); assert.equal(fed.mode, "feeding");
  assert.equal(fed.pose().feed, 1); assert.equal(fed.senses()["grn:sugar:labellum"], 1);
  assert.equal(fed.pendingReward, null, "no retrospective approach decision is generated");
  fed.setSugar("bread"); const hunger = fed.hunger; fed.step(DT);
  assert.ok(fed.hunger >= hunger); assert.equal(fed.pose().feed, 1);
  assert.equal(fed.pose().feeding, false);
});

test("external impulses and fruit movement do not create takeoff or approach commands", () => {
  const life = fixture();
  life.poke([0.1, 0, 0], [0, 0, 0], [0, 0, 0]);
  assert.deepEqual(life.flight.v, [0.1, 0, 0]); zero(life);
  life.fruitMoved("banana"); zero(life);
  assert.equal(life.pendingReward, null);
  assert.throws(() => life.poke([NaN, 0, 0]), RangeError);
  assert.throws(() => life.step(NaN), RangeError);
});

console.log(`${passed} settled-authority tests passed`);
