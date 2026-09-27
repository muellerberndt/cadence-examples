// Pure decoder parity and causal boundaries; no network download or trained fixture.
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { settledMotor, SETTLED_MOTOR_GROUPS, wingControls } from "../web/motor.js";
import { Flight, CONTROL_KEYS, DT, GRAVITY } from "../web/body.js";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const zero = { wings: Array(7).fill(0), proboscis: 0, legs: { ttm_left: 0, ttm_right: 0 } };
assert.deepEqual(settledMotor({}), zero);
assert.deepEqual(settledMotor(Object.fromEntries(SETTLED_MOTOR_GROUPS.map(k => [k, 0]))), zero);
assert.equal(wingControls({}).frequency, 180); // historical comparator unchanged
const steeringOnly = settledMotor(Object.fromEntries(SETTLED_MOTOR_GROUPS.filter(k => k.startsWith("mn:wing:")).map(k => [k, 1])));
assert.equal(steeringOnly.wings[0], 0);
assert.equal(steeringOnly.wings[1], 0);
assert.equal(steeringOnly.wings[6], 0);
for (const bad of [NaN, Infinity, -Infinity, true, "1", null, undefined])
  assert.throws(() => settledMotor({ "power:left": bad }), RangeError);
const oneSide = settledMotor({ "power:left": 1 });
assert.equal(oneSide.wings[0], 1);
assert.equal(oneSide.wings[1], 0);
assert.equal(oneSide.wings[6], 200);
assert.deepEqual(settledMotor({ "power:left": 1, target: {}, heading: NaN, state: {} }), oneSide);

// Check forces through the independent rigid-body implementation, not just the
// decoder formula: equal activity p must produce p body weights at level rest.
for (const p of [0, .001, .25, .5, 1]) {
  const output = settledMotor({ "power:left": p, "power:right": p }).wings;
  const flight = new Flight([2, 1.5, 1.2]);
  flight.step(DT, Object.fromEntries(CONTROL_KEYS.map((key, i) => [key, output[i]])));
  assert.ok(Math.abs(1 + flight.v[2] / (DT * GRAVITY) - p) < 1e-14, `power ${p} is not linear in lift`);
  assert.equal(output[6], p > 0 ? 200 : 0);
}
const unilateral = settledMotor({ "power:left": .5, "power:right": 0 }).wings;
assert.equal(unilateral[1], 0);
const unilateralBody = new Flight([2, 1.5, 1.2]);
unilateralBody.step(DT, Object.fromEntries(CONTROL_KEYS.map((key, i) => [key, unilateral[i]])));
assert.ok(Math.abs(1 + unilateralBody.v[2] / (DT * GRAVITY) - .25) < 1e-14);
assert.ok(unilateralBody.w[0] > 0);

const cases = [{}, ...SETTLED_MOTOR_GROUPS.map(k => ({ [k]: 1 }))];
for (let i = 0; i < 12; i++)
  cases.push(Object.fromEntries(SETTLED_MOTOR_GROUPS.map((k, j) => [k, (((i + 3) * (j + 7)) % 23 - 4) / 13])));
const reference = JSON.parse(execFileSync(process.env.PYTHON || "python3", ["-c", `
import json, sys
sys.path.insert(0, sys.argv[1])
from fruitfly.motor import settled_motor
print(json.dumps([settled_motor(case) for case in json.load(sys.stdin)]))
`, root], { input: JSON.stringify(cases), encoding: "utf8" }));
let worst = 0;
for (let i = 0; i < cases.length; i++) {
  const actual = settledMotor(cases[i]), expected = reference[i];
  actual.wings.forEach((x, j) => { worst = Math.max(worst, Math.abs(x - expected.wings[j])); });
  assert.deepEqual(actual.legs, expected.legs);
  assert.equal(actual.proboscis, expected.proboscis);
}
assert.ok(worst < 1e-14, `Python/JS decoder mismatch ${worst}`);
console.log(`settled motor: ${cases.length} Python/JS parity cases, worst ${worst}; silent/steering-only/invalid/one-sided controls passed`);
