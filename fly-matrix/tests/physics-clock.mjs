import assert from "node:assert/strict";
import { Flight } from "../web/body.js";
import { SettledLife } from "../web/settled-life.js";
import { PhysicsClock } from "../web/physics-clock.js";
import { SETTLED_MOTOR_GROUPS } from "../web/motor.js";

const make = () => {
  const flight = new Flight([1, 1, 1]);
  return { flight, life: new SettledLife(flight, { fruits: {}, surfaceZ: () => 0 }), clock: new PhysicsClock() };
};
const request = { requestId: 1, generation: 0, steps: 256, tolerance: 1e-6 };
const reply = { type: "control", kind: "control", ...request, converged: true, residual: 0, iterations: 1,
  readouts: Object.fromEntries(SETTLED_MOTOR_GROUPS.map(k => [k, k.startsWith("power:") ? 1 : 0])) };

// A worker that never answers cannot freeze the body or keep its old command alive.
const waiting = make(); waiting.life.expectControl(request);
for (let i = 0; i < 20; i++) waiting.clock.advance(waiting.life, 0.01, 0.5);
assert.ok(Math.abs(waiting.life.clock - 0.1) < 1e-12);
assert.ok(waiting.flight.p[2] < 0.96 && waiting.flight.v[2] < -0.9);
assert.ok(waiting.life.pending, "the body moved while a neural request remained pending");
waiting.clock.advance(waiting.life, 0.05, 0.5);
assert.equal(waiting.life.applyControl(reply), false, "a late settled reply cannot revive motors");
assert.ok(Object.values(waiting.life.controls).every(x => x === 0));

// On-time motor output really acts through the physical body, then expires.
const active = make(); active.life.expectControl(request); assert.equal(active.life.applyControl(reply), true);
for (let i = 0; i < 10; i++) active.clock.advance(active.life, 0.01, 1);
assert.ok(active.flight.p[2] > waiting.flight.p[2] + 0.03);
active.clock.advance(active.life, 0.03, 1);
assert.ok(Object.values(active.life.controls).every(x => x === 0));

// Display frame rate does not change fixed-step mechanics for equal elapsed time.
const fast = make(), slow = make();
for (let i = 0; i < 50; i++) fast.clock.advance(fast.life, 0.004, 0.5);
for (let i = 0; i < 4; i++) slow.clock.advance(slow.life, 0.05, 0.5);
assert.deepEqual(fast.flight.p, slow.flight.p);
assert.deepEqual(fast.flight.v, slow.flight.v);
assert.equal(fast.flight.steps, 200);
const tab = make(); tab.clock.advance(tab.life, 50, 1);
assert.equal(tab.flight.steps, 500); assert.equal(tab.clock.droppedWallSeconds, 49.75);
for (const duration of [NaN, Infinity, -1]) assert.throws(() => tab.clock.advance(tab.life, duration));
console.log("physics clock: pending-worker motion, on-time motor force, expiry, stale rejection, frame independence and tab-resume cap passed");
