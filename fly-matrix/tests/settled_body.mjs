// Passive support tops: zero neural power cannot hover, navigate, level or snap.
import assert from "node:assert/strict";
import { Flight, DT, RADIUS, MASS, GRAVITY, K_CONTACT, CONTROL_KEYS } from "../web/body.js";
import { settledMotor } from "../web/motor.js";

const zero = Object.fromEntries(CONTROL_KEYS.map(k => [k, 0]));
const step = (flight, count) => { for (let i = 0; i < count; i++) flight.step(DT, zero); };
const table = 0.75;
const top = (x, y) => x > 1 && x < 3 && y > 1 && y < 2 ? table : 0;

const falling = new Flight([2, 1.5, 0.8]);
falling.surfaceHeight = top;
falling.setAttitude(0.23, -0.19, 0.6);
const originalQ = falling.q.slice();
step(falling, 1);
assert.ok(falling.v[2] < 0 && falling.p[2] < 0.8);
assert.equal(falling.onFloor, false);
step(falling, 1999);
assert.ok(falling.landed && falling.onFloor && falling.touching > 0);
assert.equal(falling.supportHeight, table);
assert.ok(Math.abs(falling.p[2] - (table + RADIUS - MASS * GRAVITY / K_CONTACT)) < 1e-9);
assert.ok(Math.abs(falling.v[2]) < 1e-9);
assert.ok(Math.max(...originalQ.map((q, i) => Math.abs(q - falling.q[i]))) < 1e-14, "passive contact must not level attitude");
assert.equal(falling.phase, 0, "zero motor frequency does not beat the wings");

// No repulsive metre-deep spring when entering a footprint from underneath.
const underneath = new Flight([0.9999, 1.5, 0.2]);
underneath.surfaceHeight = top;
underneath.v[0] = 0.4;
step(underneath, 15);
assert.ok(underneath.p[0] > 1);
assert.ok(underneath.p[2] < 0.2 && underneath.v[2] < 0);
assert.equal(underneath.supportHeight, null);
assert.equal(underneath.touching, 0);

// A moved fruit appearing overhead cannot pull or teleport a resting fly up.
let raised = false;
const resting = new Flight([2, 1.5, table + RADIUS]);
resting.surfaceHeight = () => raised ? [table, table + 0.05] : [table];
step(resting, 600);
const before = resting.p[2];
raised = true;
step(resting, 300);
assert.equal(resting.supportHeight, table);
assert.ok(Math.abs(resting.p[2] - before) < 1e-10);

// The higher eligible top takes the contact, with no duplicate floor springs.
const fruit = new Flight([2, 1.5, 0.85]);
fruit.surfaceHeight = () => [table, table + 0.05, table + 0.05];
step(fruit, 2000);
assert.equal(fruit.supportHeight, table + 0.05);
assert.equal(fruit.touching, 1);
assert.ok(fruit.landed);
// Removing a support invokes gravity, not a supplied takeoff or held position.
fruit.surfaceHeight = () => [];
const height = fruit.p[2];
step(fruit, 20);
assert.ok(fruit.p[2] < height && fruit.v[2] < 0);
assert.equal(fruit.supportHeight, null);

// Optional callback returning only the floor preserves the unextended dynamics.
const original = new Flight([2, 1.5, 0.03]), empty = new Flight([2, 1.5, 0.03]);
empty.surfaceHeight = () => [0, NaN, Infinity, -1];
step(original, 1000); step(empty, 1000);
assert.deepEqual(empty.p, original.p);
assert.deepEqual(empty.q, original.q);
assert.deepEqual(empty.v, original.v);

// Motor changes reach the mechanics only via the seven actuator values.
const off = new Flight([2, 1.5, 1]), on = new Flight([2, 1.5, 1]);
off.step(DT, zero);
on.step(DT, Object.fromEntries(CONTROL_KEYS.map((k, i) => [k, settledMotor({ "power:left": 1 }).wings[i]])));
assert.ok(on.v[2] > off.v[2] && on.w[0] > off.w[0]);
assert.ok(on.phase > off.phase);
console.log("settled body: passive top contact, gravity, attitude, below-top crossing, moved/removed support, layered tops and motor causality passed");
