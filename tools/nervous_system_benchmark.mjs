import assert from "node:assert/strict";
import { writeFileSync, readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { DrawingArm, RETINA } from "../eye-arm/brain.js";
import { imageFixture } from "../eye-arm/fixtures.js";
import { Forager } from "../fly/brain.js";
import { WormArena } from "../worm/brain.js";
import { SynapticMemory, flowers } from "../shared/engine.js";
import { settlingTrace } from "../shared/telemetry.js";
const root = new URL("../", import.meta.url),
  arm = [];
for (const image of ["square", "flower", "two_marks"])
  for (const condition of [
    "intact",
    "disturbed",
    "feedback_off",
    "joint_motors_off",
    "pencil_motors_off",
    "eye_off",
  ]) {
    const a = new DrawingArm(
      condition === "eye_off"
        ? Array(RETINA ** 2).fill(0)
        : imageFixture(image),
    );
    if (condition === "joint_motors_off") a.brain.motor.mask.fill(0, 11, 15);
    if (condition === "pencil_motors_off") a.brain.motor.mask.fill(0, 15, 17);
    if (condition === "feedback_off") a.closed = false;
    const initial = a.q.slice();
    let raised = 0,
      contact = 0;
    for (let t = 0; t < 6000; t++) {
      if (t === 180 && ["disturbed", "feedback_off"].includes(condition))
        a.disturb();
      a.step();
      if (a.lifted) raised++;
      else contact++;
    }
    const row = {
      image,
      condition,
      coverage: a.coverage,
      ink: a.ink.length,
      raised,
      contact,
      targets: a.targets.length,
      strokes: a.strokes,
      completed: a.target < 0,
      displacement: Math.hypot(...a.q.map((v, i) => v - initial[i])),
    };
    if (["intact", "disturbed"].includes(condition)) {
      assert.ok(row.coverage > 0.95, JSON.stringify(row));
      assert.ok(row.raised > 0 && row.contact > 0);
    }
    if (condition === "joint_motors_off") assert.equal(row.displacement, 0);
    if (["pencil_motors_off", "eye_off"].includes(condition))
      assert.equal(row.ink, 0);
    arm.push(row);
  }
const data = JSON.parse(readFileSync(new URL("worm/worm.json", root))),
  worm = [];
for (const condition of [
  "intact",
  "smell_off",
  "chemical_motors_off",
  "direction_motors_off",
  "sealed_wall",
]) {
  const w = new WormArena(data);
  if (condition === "smell_off") w.smell = false;
  if (condition === "chemical_motors_off")
    w.motor.forEach((i) => (w.mask[i] = 0));
  if (condition === "direction_motors_off") w.nerves.mask.fill(0, 4);
  if (condition === "sealed_wall") {
    for (let y = 0; y < w.rows; y++) w.walls[y * w.cols + 8] = true;
    w.repairOdor();
  }
  let collision = false;
  for (let i = 0; i < 250; i++) {
    w.step();
    collision ||= w.walls[w.cell];
  }
  const row = { condition, eaten: w.eaten, moves: w.moves, collision };
  assert.ok(!collision);
  condition === "intact" ? assert.equal(w.eaten, 2) : assert.equal(w.eaten, 0);
  worm.push(row);
}
const fly = [];
for (const condition of ["intact", "motors_off"]) {
  const f = new Forager(new SynapticMemory()),
    field = flowers();
  if (condition === "motors_off") f.nerves.mask.fill(0, 2);
  for (let t = 0; t < 4000; t++) f.step(field);
  const row = {
    condition,
    encounters: f.encounters,
    nectar: f.nectar,
    displacement: Math.hypot(f.x - 0.45, f.y - 0.5),
  };
  condition === "intact"
    ? assert.ok(f.encounters > 10)
    : assert.equal(row.displacement, 0);
  fly.push(row);
}
const a = new DrawingArm(imageFixture("square"));
for (let t = 0; t < 100; t++) a.step();
const composite = a.brain.snapshot(),
  last = settlingTrace(composite).frames.at(-1),
  parity = {
    trace_error: Math.max(
      ...last.map((v, i) => Math.abs(v - composite.state[i])),
    ),
  };
assert.ok(parity.trace_error < 1e-12);
const files = [
  "eye-arm/brain.js",
  "eye-arm/paper.js",
  "eye-arm/fixtures.js",
  "fly/brain.js",
  "worm/brain.js",
  "shared/nervous_system.js",
  "shared/engine.js",
  "shared/embodied.js",
  "worm/worm_arena.js",
  "shared/telemetry.js",
  "tools/nervous_system_benchmark.mjs",
];
const sources = Object.fromEntries(
  files.map((f) => [
    f,
    createHash("sha256")
      .update(readFileSync(new URL(f, root)))
      .digest("hex"),
  ]),
);
const result = {
  schema: "cadence.task-nervous-systems/v1",
  steps: { arm: 6000, worm: 250, fly: 4000 },
  sources,
  arm,
  worm,
  fly,
  parity,
};
for (const folder of ["eye-arm", "worm", "fly"])
  writeFileSync(
    new URL(`${folder}/evidence.json`, root),
    // Evidence precision exceeds the spatial success tolerance while avoiding
    // platform-specific last-bit differences accumulated over 6,000 control steps.
    // Seven decimal places remain well below one paper pixel.
    JSON.stringify(
      result,
      (_, v) => (typeof v === "number" ? Math.round(v * 1e7) / 1e7 : v),
      2,
    ) + "\n",
  );
console.log(JSON.stringify({ arm, worm, fly, parity }));
