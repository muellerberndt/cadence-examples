// Additional task probes, fixed before the final public controller was installed.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { Forager } from "../fly/brain.js";
import { DrawingArm, RETINA } from "../eye-arm/brain.js";
import { SynapticMemory, flowers, keys, argmax } from "../shared/engine.js";
import { random } from "../shared/embodied.js";

const root = new URL("../", import.meta.url);
const seeds = [107, 131, 157, 181, 211, 239];
export function report() {
  const fly = [];
  for (const seed of seeds) for (const revisit of [false, true]) {
    const rng = random(seed);
    const field = flowers().map((f, i) => ({
      ...f, x: 0.12 + rng() * 0.76, y: 0.12 + rng() * 0.76, value: i % 4,
    }));
    const agent = new Forager(new SynapticMemory(), 0, 1, { revisit });
    let before;
    for (let tick = 0; tick < 8000; tick++) {
      if (tick === 4000) {
        before = { nectar: agent.nectar, encounters: agent.encounters };
        // No notification or lesson is supplied to the agent.
        field.forEach(f => { f.value = 3 - f.value; });
      }
      agent.step(field);
    }
    fly.push({
      seed, condition: revisit ? "aging_observations" : "previous_selection",
      before,
      after: {
        nectar: agent.nectar - before.nectar,
        encounters: agent.encounters - before.encounters,
        correct: field.filter(f =>
          argmax(agent.memory.predict(keys()[f.kind])) === f.value).length,
      },
    });
  }
  const arm = [];
  for (const shape of ["diagonal", "cross", "circle"]) {
    const pixels = Array.from({ length: RETINA ** 2 }, (_, i) => {
      const x = i % RETINA, y = Math.floor(i / RETINA);
      if (shape === "diagonal") return +(x === y && x >= 6 && x < 42);
      if (shape === "cross") return +((Math.abs(x - 24) < 2 && y >= 6 && y < 42)
        || (Math.abs(y - 24) < 2 && x >= 6 && x < 42));
      return +(Math.abs(Math.hypot(x - 24, y - 24) - 15) < 1);
    });
    const brain = new DrawingArm(pixels);
    let ticks = 0;
    for (; ticks < 6000 && !(brain.finishedOnce && brain.lifted); ticks++) brain.step();
    assert.ok(brain.coverage >= 0.95, shape);
    arm.push({ shape, targets: brain.targets.length, ticks,
      coverage: brain.coverage, strokes: brain.strokes });
  }
  const files = ["tools/adaptation_benchmark.mjs", "fly/brain.js",
    "eye-arm/brain.js", "eye-arm/paper.js", "shared/engine.js",
    "shared/embodied.js", "shared/nervous_system.js"];
  return {
    schema: "cadence.adaptation-probes/v1",
    schedule: {
      development_seeds: [17, 29, 41], evaluation_seeds: seeds,
      selection: "Amplitude 1.5 and time constant 15 seconds chosen on development seeds; evaluation layouts were unused during selection.",
      fly_ticks: 8000, nectar_reversal_tick: 4000, dt: 0.025,
      arm_tick_budget: 6000,
    },
    limits: "Same memory and motor circuit, different supplied exploration rule; trajectories differ. This is an adaptation tradeoff, not an MLP comparison. Arm coverage measures marks at reference samples, not exact image similarity.",
    sources: Object.fromEntries(files.map(f => [f,
      createHash("sha256").update(readFileSync(new URL(f, root))).digest("hex")])),
    fly, arm,
  };
}
if (process.argv[1] && pathToFileURL(process.argv[1]).href === import.meta.url) {
  const result = report();
  writeFileSync(new URL("evidence/adaptation_evidence.json", root),
    JSON.stringify(result, null, 2) + "\n");
  console.log(JSON.stringify({ fly: result.fly, arm: result.arm }));
}
