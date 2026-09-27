// Physical-facing regression tests with qualified synthetic neural readouts.
// These test the supplied stabilizer convention, not learned turning competence.
// Run: node tests/neural-facing.mjs
import assert from "node:assert/strict";
import { NeuralLife } from "../web/neural-life.js";
import { Flight, HandPilot, DT, wrap_ } from "../web/body.js";
import { SETTLED_MOTOR_GROUPS, GAIN_TILT } from "../web/motor.js";
import { NAVIGATION_READOUTS, POLICY_OUTPUTS, decodeNavigation } from "../web/neural-policy.js";

const room = () => ({ table: { x0: 2.45, x1: 3.55, y0: 1.95, y1: 2.65, z: .75 },
  fruits: { banana: { pos: [2.88, 2.33, .795] }, bread: { pos: [3.13, 2.28, .8068] } }, surfaceZ: () => .75 });
const makeLife = () => new NeuralLife(new Flight([1.2, 1, 1.2], .3), room(), 7, { generation: 2 });
const reads = (changes = {}) => ({ ...Object.fromEntries([...new Set([
  ...SETTLED_MOTOR_GROUPS, ...NAVIGATION_READOUTS, ...POLICY_OUTPUTS,
])].map(name => [name, 0])), ...changes });
const backward = () => reads({ "mn:wing:b2:left": .021 / GAIN_TILT, "mn:wing:i2:right": .031 / GAIN_TILT });
const forward = () => reads({ "dn:DNp09": .8 });
const close = (actual, expected, tolerance = 1e-10) => assert.ok(Math.abs(actual - expected) <= tolerance,
  `${actual} differs from ${expected} by more than ${tolerance}`);
const angleClose = (a, b, tolerance = 1e-10) => close(wrap_(a - b), 0, tolerance);
const observe = (fly, readouts) => {
  const request = fly.expectObservation(fly._lastObservationId + 1);
  assert.ok(request);
  assert.equal(fly.applyObservation({ type: "control", kind: "control", ...request,
    converged: true, iterations: 20, residual: 1e-8, tolerance: 1e-6, readouts }), true);
};
function advance(fly, readouts, seconds) {
  const steps = Math.round(seconds / DT);
  for (let k = 0; k < steps; k++) {
    if (k % 500 === 0) observe(fly, readouts);
    fly.step(DT);
  }
}
const travelError = fly => Math.abs(wrap_(fly.flight.euler()[2] - Math.atan2(fly.flight.v[1], fly.flight.v[0])));
const displacementAlong = (fly, start, heading) => (fly.flight.p[0] - start[0]) * Math.cos(heading)
  + (fly.flight.p[1] - start[1]) * Math.sin(heading);
let passed = 0, failed = 0;
function test(name, run) {
  try { run(); passed++; console.log(`ok ${name}`); }
  catch (error) { failed++; console.error(`FAIL ${name}\n${error.stack}`); }
}

test("negative neural velocity and yaw/climb retain their original world-frame meaning", () => {
  const fly = makeLife(), values = { ...backward(), "dn:DNa02:right": .02, "dn:DNp03": .2 };
  const before = fly.flight.p.slice(), heading = fly.heading, command = decodeNavigation(values);
  assert.ok(command.forwardSpeed < 0);
  observe(fly, values); fly.step(DT);
  assert.deepEqual(fly.authority.navigation.command, command);
  angleClose(fly.heading, heading + command.yawRate * DT);
  for (const [i, velocity] of [command.forwardSpeed * Math.cos(fly.heading),
    command.forwardSpeed * Math.sin(fly.heading), command.verticalSpeed].entries()) {
    close(HandPilot.K_POS * (fly.pilot.target[i] - before[i]), velocity);
  }
  angleClose(fly.pilot.heading, Math.atan2(command.forwardSpeed * Math.sin(fly.heading),
    command.forwardSpeed * Math.cos(fly.heading)));
  assert.match(fly.authority.facingAssistance, /supplied candidate/);
  assert.equal(fly.authority.autonomous, false);
});

test("a reverse request turns through the real wing-force integrator, with no attitude teleport", () => {
  const fly = makeLife(), replay = new Flight(fly.flight.p, fly.flight.euler()[2]);
  const initialQ = fly.flight.q.slice(), initialYaw = fly.flight.euler()[2];
  replay.q = initialQ.slice(); // Avoid an unrelated quaternion -> Euler -> quaternion round-trip.
  observe(fly, backward());
  assert.deepEqual(fly.flight.q, initialQ, "observation admission must not rotate the body");
  fly.step(DT); replay.step(DT, fly.controls);
  assert.deepEqual(fly.flight.q, replay.q, "orientation must be exactly the result of the physical actuator step");
  assert.deepEqual(fly.flight.w, replay.w);
  assert.deepEqual(fly.flight.p, replay.p);
  const turned = Math.abs(wrap_(fly.flight.euler()[2] - initialYaw));
  assert.ok(turned > 1e-6 && turned < .01, `first physical turn is ${turned} rad, not an instantaneous pi flip`);
  advance(fly, backward(), .5);
  assert.ok(travelError(fly) < .05, `nose/travel error ${travelError(fly)} rad`);
  assert.ok(Math.abs(wrap_(fly.flight.euler()[2] - initialYaw)) > 3,
    "the body must actually reverse its facing direction");
});

test("positive neural travel retains forward facing and the existing velocity magnitude", () => {
  const fly = makeLife(), start = fly.flight.p.slice(), heading = fly.heading;
  advance(fly, forward(), .5);
  assert.ok(displacementAlong(fly, start, heading) > .09);
  close(Math.hypot(fly.flight.v[0], fly.flight.v[1]), .24, .002);
  angleClose(fly.facingHeading, fly.heading);
  assert.ok(travelError(fly) < .01);
});

test("a positive-to-negative command physically reverses both travel and facing without changing the neural heading", () => {
  const fly = makeLife(), heading = fly.heading;
  advance(fly, forward(), .5);
  const priorQ = fly.flight.q.slice();
  observe(fly, backward());
  assert.deepEqual(fly.flight.q, priorQ);
  // The body must shed its old forward momentum; do not demand an instantaneous
  // velocity flip or that it undo all earlier displacement in a fixed window.
  advance(fly, backward(), .4);
  const start = fly.flight.p.slice();
  advance(fly, backward(), .1);
  assert.ok(displacementAlong(fly, start, heading) < -.003, "the actual later path must go backward in the neural reference frame");
  angleClose(fly.heading, heading);
  assert.ok(travelError(fly) < .05);
  assert.ok(Math.abs(wrap_(fly.flight.euler()[2] - heading)) > 3);
});

test("zero speed retains facing across the sign boundary and still permits a neural yaw request", () => {
  const fly = makeLife();
  advance(fly, backward(), .5);
  const before = fly.facingHeading, reference = fly.heading;
  observe(fly, reads()); fly.step(DT);
  angleClose(fly.facingHeading, before);
  assert.equal(fly.authority.navigation.command.forwardSpeed, 0);
  const yawOnly = reads({ "dn:DNa02:right": .1 }), yaw = decodeNavigation(yawOnly).yawRate;
  advance(fly, yawOnly, .25);
  angleClose(fly.heading, reference + yaw * .25);
  angleClose(fly.facingHeading, before + yaw * .25);
  assert.equal(fly.authority.navigation.command.forwardSpeed, 0);
  assert.ok(Math.abs(wrap_(fly.flight.euler()[2] - before)) > .1);
  const priorQ = fly.flight.q.slice();
  observe(fly, forward());
  assert.deepEqual(fly.flight.q, priorQ);
  advance(fly, forward(), .5);
  angleClose(fly.facingHeading, fly.heading);
  assert.ok(travelError(fly) < .05);
});

test("hard revocation clears course and facing memory without teleporting the body", () => {
  const fly = makeLife(); advance(fly, backward(), .25);
  const bodyYaw = fly.flight.euler()[2], beforeQ = fly.flight.q.slice();
  fly.revokeNeural("test withdrawal");
  assert.deepEqual(fly.flight.q, beforeQ);
  assert.equal(fly.authority.navigation.active, false);
  assert.deepEqual(fly.authority.navigation.command, { yawRate: 0, forwardSpeed: 0, verticalSpeed: 0, feed: 0 });
  angleClose(fly.heading, bodyYaw); angleClose(fly.pilot.heading, bodyYaw); angleClose(fly.facingHeading, bodyYaw);
  observe(fly, reads()); fly.step(DT);
  angleClose(fly.pilot.heading, bodyYaw);
});

test("identical negative inputs separated by expiry keep their course and produce net physical progress", () => {
  const fly = makeLife(), start = fly.flight.p.slice(), course = fly.heading;
  const command = decodeNavigation(backward()); let travelled = 0;
  for (let cycle = 0; cycle < 4; cycle++) {
    observe(fly, backward());
    for (let k = 0; k < Math.round(1.25 / DT); k++) {
      const before = fly.flight.p.slice(), actualYaw = fly.flight.euler()[2];
      fly.step(DT);
      travelled += Math.hypot(fly.flight.p[0] - before[0], fly.flight.p[1] - before[1]);
      angleClose(fly.heading, course);
      if (fly.authority.navigation.active) {
        assert.deepEqual(fly.authority.navigation.command, command);
        close(HandPilot.K_POS * (fly.pilot.target[0] - before[0]), command.forwardSpeed * Math.cos(course));
        close(HandPilot.K_POS * (fly.pilot.target[1] - before[1]), command.forwardSpeed * Math.sin(course));
      } else {
        assert.deepEqual(fly.authority.navigation.command, { yawRate: 0, forwardSpeed: 0, verticalSpeed: 0, feed: 0 });
        assert.deepEqual(fly.pilot.target, before, "an expired observation supplies no target velocity");
        angleClose(fly.pilot.heading, actualYaw, 1e-12);
        assert.ok(Object.values(fly.authority.neuralTrim).every(x => x === 0));
      }
    }
    assert.equal(fly.authority.navigation.active, false, "each cycle must include a real inactive gap");
  }
  const progress = -displacementAlong(fly, start, course);
  assert.ok(progress > .08, `repeated reverse command made only ${progress} metres of net progress`);
  assert.ok(progress / travelled > .8, `expiry must not alternate travel direction: net ${progress}, path ${travelled}`);
});

test("a matching late reply cannot rebase the course or restore old movement", () => {
  for (const unresolved of [false, true]) {
    const fly = makeLife(), course = fly.heading; advance(fly, backward(), .5);
    const request = fly.expectObservation(fly._lastObservationId + 1), actualYaw = fly.flight.euler()[2];
    fly.clock = request.deadline;
    assert.equal(fly.applyObservation({ type: "control", kind: "control", ...request,
      converged: !unresolved, iterations: unresolved ? 1024 : 20,
      residual: unresolved ? .1 : 1e-8, tolerance: 1e-6, readouts: backward() }), false);
    angleClose(fly.heading, course); angleClose(fly.pilot.heading, actualYaw);
    assert.equal(fly.authority.navigation.active, false);
    assert.ok(Object.values(fly.authority.neuralTrim).every(x => x === 0));
    fly.step(DT);
    assert.deepEqual(fly.authority.navigation.command, { yawRate: 0, forwardSpeed: 0, verticalSpeed: 0, feed: 0 });
    observe(fly, backward()); fly.step(DT);
    assert.deepEqual(fly.authority.navigation.command, decodeNavigation(backward()));
    angleClose(fly.heading, course);
  }
});

console.log(`${passed} neural physical-facing tests passed; ${failed} failed`);
if (failed) process.exitCode = 1;
