// Chaos: the sill under many random dot paths, with every brain, and its invariants.
//
//   node dozing-cat/tests/chaos.mjs
//
// Random mouse paths (walks, jumps, rests, the laser off and on) and random scheduled dots
// (walks, curves, jitters; the laws changed and restored) drive the world for thousands of
// decisions under each governor and each control. At every decision: the paw stays on the
// sill, a present dot stays on the sill, the flags agree with the dot (present when it is
// there; a catch or a miss is a vanishing), the hold count never exceeds the rule's three,
// and every number the brain produces is finite. The catch rule is checked directly: a still
// dot with the paw within reach is caught on exactly the third decision, whatever the brain
// does, and a caught dot returns only when the mouse has moved more than 0.03 away. The
// scheduled dots with the evolved governor patch must produce catches: a cat that never
// caught anything would be a broken twin, whatever the parity test said of its arithmetic.
import { readFileSync } from "node:fs";
import { BeliefPatch } from "../web/belief.js";
import { Life, ARMS } from "../web/life.js";
import { World, SCHEDULE, Rng, setConstants, drawEvents, CATCH_RADIUS, CATCH_HOLD } from "../web/cat.js";

const dir = new URL(".", import.meta.url).pathname;
const spec = JSON.parse(readFileSync(dir + "../web/data/brain.json", "utf8"));
setConstants(spec.world);
let problems = 0, decisions = 0;
const fail = (msg) => { problems++; if (problems <= 20) console.log("FAIL:", msg); };
const finite = (x) => Number.isFinite(x);
const inBox = (p) => p[0] >= 0 && p[0] <= 1 && p[1] >= 0 && p[1] <= 1;
const genomeFor = (arm) => spec.genomes[arm === "patch" ? "patch" : "threshold"][arm === "patch" ? "evolved" : "hand_set"];

function invariants(world, out, where) {
  if (!inBox(world.paw)) fail(`${where}: the paw left the sill at ${world.paw}`);
  if (world.dot !== null && !inBox(world.dot)) fail(`${where}: the dot left the sill at ${world.dot}`);
  if (out.flags.present !== (world.dot !== null)) fail(`${where}: the present flag disagrees with the dot`);
  if ((out.flags.catch || out.flags.miss) && !out.flags.vanish) fail(`${where}: a catch or a miss without a vanishing`);
  if (out.flags.catch && world.dot !== null && world.source === "schedule") fail(`${where}: the dot stayed after a catch`);
  if (world.hold > CATCH_HOLD) fail(`${where}: the hold count passed ${CATCH_HOLD}`);
  if (!finite(out.surprise) || !finite(out.residual) || !finite(out.baseline) || !out.action.every(finite) || !out.expected.every(finite)) fail(`${where}: a number is not finite`);
  if (!life_ok(out)) fail(`${where}: the belief is not finite`);
}
const life_ok = (out) => out.path.belief[0][0].every(finite);

// -- random mouse paths under every brain
for (const arm of ARMS) {
  for (let seed = 1; seed <= 3; seed++) {
    const rng = new Rng(100 * seed + arm.length);
    const patch = new BeliefPatch(spec.belief);
    const world = new World({ ...SCHEDULE, decisions: 1e9, change: "none" }, [], new Rng(seed), "mouse");
    const life = new Life(patch, world, genomeFor(arm), spec.scale, spec.floors, { arm, rng: new Rng(seed + 9), machinery: spec.machinery });
    let mouse = null, catches = 0;
    for (let t = 0; t < 1500; t++) {
      const u = rng.random();
      if (u < 0.01) mouse = null;                                        // the laser off
      else if (u < 0.03) mouse = [rng.random(), rng.random()];           // a jump
      else if (mouse && u < 0.6) mouse = [Math.min(1, Math.max(0, mouse[0] + rng.normal(0, 0.02))), Math.min(1, Math.max(0, mouse[1] + rng.normal(0, 0.02)))]; // a walk
      else if (!mouse && u < 0.05) mouse = [rng.random(), rng.random()]; // the laser on
      world.mouse = mouse === null ? null : mouse.slice();
      const out = life.decide();
      decisions++;
      invariants(world, out, `mouse path, ${arm}, seed ${seed}, decision ${t}`);
      if (out.flags.catch) { catches++; if (world.mouseCaught === null && mouse !== null) fail(`${arm}: a catch under the mouse did not remember where`); }
    }
    console.log(`mouse paths, ${arm}, seed ${seed}: ${catches} catches, ${life.totals.learn_calls} learn calls, awake ${(1 - life.totals.decisions.habit / 1500).toFixed(2)}`);
  }
}

// -- the catch rule, whatever the brain does: a still dot with the paw within reach is caught on the third decision, and returns only after a move
{
  const rng = new Rng(77);
  for (let trial = 0; trial < 50; trial++) {
    const patch = new BeliefPatch(spec.belief);
    const world = new World({ ...SCHEDULE, decisions: 1e9, change: "none" }, [], new Rng(1), "mouse");
    const arm = ARMS[trial % ARMS.length];
    const life = new Life(patch, world, genomeFor(arm), spec.scale, spec.floors, { arm, rng: new Rng(2), machinery: spec.machinery });
    const dot = [0.1 + 0.8 * rng.random(), 0.1 + 0.8 * rng.random()];
    world.paw = [Math.min(1, Math.max(0, dot[0] + rng.normal(0, 0.01))), Math.min(1, Math.max(0, dot[1] + rng.normal(0, 0.01)))];
    world.mouse = dot.slice();
    let caughtAt = -1;
    for (let t = 0; t < 8; t++) {
      const out = life.decide();
      decisions++;
      const near = Math.hypot(world.paw[0] - dot[0], world.paw[1] - dot[1]) <= CATCH_RADIUS;
      if (out.flags.catch) { caughtAt = t; break; }
      if (!near) break; // the brain pushed the paw out of reach: the rule does not apply
    }
    if (caughtAt >= 0 && caughtAt !== CATCH_HOLD - 1) fail(`catch rule, trial ${trial}: caught on decision ${caughtAt}, expected ${CATCH_HOLD - 1}`);
    if (caughtAt >= 0) {
      if (world.dot !== null || world.mouseCaught === null) fail(`catch rule, trial ${trial}: the dot did not vanish on the catch`);
      world.mouse = [dot[0] + 0.02, dot[1]];
      let out = life.decide(); decisions++;
      if (world.dot !== null || out.flags.appear) fail(`catch rule, trial ${trial}: the dot returned after a move of 0.02`);
      world.mouse = [dot[0] + 0.04, dot[1]];
      out = life.decide(); decisions++;
      if (world.dot === null || !out.flags.appear) fail(`catch rule, trial ${trial}: the dot did not return after a move of 0.04`);
    }
  }
  console.log("catch rule: 50 trials with the paw placed within reach of a still dot");
}

// -- scheduled dots, the laws changed and restored, under every brain
let patchCatches = 0;
for (const arm of ARMS) {
  for (let seed = 1; seed <= 2; seed++) {
    const patch = new BeliefPatch(spec.belief);
    const schedule = { ...SCHEDULE, decisions: 1e9, change: "none" };
    const world = new World(schedule, drawEvents(new Rng(seed), { ...schedule, decisions: 20000 }), new Rng(seed + 50), "schedule");
    const life = new Life(patch, world, genomeFor(arm), spec.scale, spec.floors, { arm, rng: new Rng(seed + 9), machinery: spec.machinery });
    const laws = ["gravity", "faster", "wrap", "restore"];
    let catches = 0, misses = 0;
    for (let t = 0; t < 2400; t++) {
      if (t % 600 === 300) world.fire(laws[Math.floor(t / 600) % laws.length]);
      const out = life.decide();
      decisions++;
      invariants(world, out, `schedule, ${arm}, seed ${seed}, decision ${t}`);
      if (out.flags.catch) catches++;
      if (out.flags.miss) misses++;
    }
    if (arm === "patch") patchCatches += catches;
    console.log(`scheduled dots, ${arm}, seed ${seed}: ${world.dots} dots, ${catches} catches, ${misses} misses, ${life.totals.learn_calls} learn calls (${life.totals.kept} kept), awake ${(1 - life.totals.decisions.habit / 2400).toFixed(2)}, ${life.compute().moments_per_decision.toFixed(1)} moments per decision`);
  }
}
if (patchCatches < 10) fail(`the evolved governor patch caught ${patchCatches} scheduled dots in two lives of 2400 decisions; the experiment's cat catches about 25 per 4000`);

console.log(problems ? `CHAOS FAILED (${problems} problems over ${decisions} decisions)` : `CHAOS OK (${decisions} decisions, no problem)`);
process.exit(problems ? 1 : 0);
