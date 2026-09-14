// Run the full small behavior schedule. --write records it; default verifies it.
import { readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { WormArena } from "./worm_arena.js";
const root = new URL("./", import.meta.url);
const data = JSON.parse(readFileSync(new URL("worm.json", root)));
const rows = [];
function run(condition, setup, expected, ticks = 160) {
  const w = new WormArena(data);
  setup(w);
  w.repairOdor();
  let collision = false;
  for (let i = 0; i < ticks; i++) {
    w.step();
    collision ||= w.walls[w.cell];
  }
  assert.equal(collision, false, condition);
  assert.equal(w.eaten, expected, condition);
  assert.ok(w.residual < 1e-8, condition);
  rows.push({ condition, ticks, eaten: w.eaten, moves: w.moves, collision });
  return w;
}
run("preset_maze", () => {}, 2);
run("open_field", (w) => w.reset("open"), 1);
run(
  "food_behind_sealed_wall",
  (w) => {
    w.reset("open");
    for (let y = 0; y < w.rows; y++) w.edit(y * w.cols + 12, "wall");
  },
  0,
);
run(
  "door_opened",
  (w) => {
    w.reset("open");
    for (let y = 0; y < w.rows; y++) w.edit(y * w.cols + 12, "wall");
    w.edit(3 * w.cols + 12, "erase");
  },
  1,
);
run("smell_disabled", (w) => (w.smell = false), 0);
run("motor_owners_removed", (w) => w.motor.forEach((i) => (w.mask[i] = 0)), 0);
run("no_food", (w) => w.food.clear(), 0);
for (let row = 2; row <= 18; row += 4) {
  run(
    `painted_food_row_${row}`,
    (w) => {
      w.reset("open");
      w.food.clear();
      w.edit(row * w.cols + 26, "food");
      for (let y = 3; y <= 17; y++) w.edit(y * w.cols + 15, "wall");
    },
    1,
  );
}
const protectedBody = new WormArena(data);
assert.equal(protectedBody.edit(protectedBody.cell, "wall"), false);
const replacement = run("finished_food", () => {}, 2);
replacement.edit(10 * replacement.cols + 3, "food");
replacement.repairOdor();
for (let k = 0; k < 160; k++) replacement.step();
assert.equal(replacement.eaten, 3);
const body = {
  contract:
    "Supplied diffusion, gradient heading, motor-gated body steps and contact consumption. These trials test the adapter, not biological fidelity or an ML performance advantage.",
  sources: Object.fromEntries(
    ["worm_arena.js", "engine.js", "worm.json", "habitat_benchmark.mjs"].map(
      (name) => [
        name,
        createHash("sha256")
          .update(readFileSync(new URL(name, root)))
          .digest("hex"),
      ],
    ),
  ),
  rows,
  body_paint_protected: true,
  new_food_after_completion: true,
};
const serialized = JSON.stringify(body, null, 2) + "\n";
const out = new URL("habitat_evidence.json", root);
if (process.argv.includes("--write")) writeFileSync(out, serialized);
else
  assert.equal(
    readFileSync(out, "utf8"),
    serialized,
    "Habitat evidence does not match a fresh execution.",
  );
console.log(
  JSON.stringify({
    conditions: rows.length,
    collision_free: true,
    food_and_barriers: true,
    sensory_and_motor_ablation: true,
    receipt: fileURLToPath(out),
  }),
);
