// Parity: the browser brain against cadence's Life on the recorded decisions.
//
//   node dozing-cat/tests/parity.mjs                                  # the committed export and cases
//   node dozing-cat/tests/parity.mjs --data /tmp/x/brain.json --cases /tmp/x/parity_cases.json
//
// Every case replays the scripted dot through web/cat.js's World and web/life.js's Life with
// the same genome and asserts, decision by decision, that the twin's beliefs, expectations,
// residuals, surprises, baselines, slow averages, readbacks, governor activations and steps,
// imagination costs, actions, modes, paws and flags are the recorded ones to a relative
// difference under TOLERANCE. The learn case forces the same learn call at the same decision
// with the recorded refit starts and checks the window's losses, the parameter update, the
// refitted habit and the life that follows.
import { readFileSync } from "node:fs";
import { BeliefPatch } from "../web/belief.js";
import { Life } from "../web/life.js";
import { World, SCHEDULE, Rng, setConstants } from "../web/cat.js";

const TOLERANCE = 1e-9;
const dir = new URL(".", import.meta.url).pathname;
const arg = (name, fallback) => { const i = process.argv.indexOf(name); return i > 0 ? process.argv[i + 1] : fallback; };
const spec = JSON.parse(readFileSync(arg("--data", dir + "../web/data/brain.json"), "utf8"));
const cases = JSON.parse(readFileSync(arg("--cases", dir + "parity_cases.json"), "utf8"));
setConstants(spec.world);
if (cases.belief_npz_sha256 !== spec.belief.npz_sha256) { console.log("PARITY FAILED: the cases were recorded with another belief than the export"); process.exit(1); }

const rel = (a, b) => { let n = 0, d = 0; for (let k = 0; k < b.length; k++) { n += (a[k] - b[k]) ** 2; d += b[k] ** 2; } return Math.sqrt(n / Math.max(d, 1e-300)); };
const relScalar = (a, b) => Math.abs(a - b) / Math.max(Math.abs(b), 1e-12);
let worst = 0, ok = true, decisions = 0;
const worstBy = {};
function check(name, value, where) {
  worstBy[name] = Math.max(worstBy[name] || 0, value);
  worst = Math.max(worst, value);
  if (!(value < TOLERANCE)) { ok = false; console.log(`  ${where}: ${name} differs by ${value.toExponential(2)}`); return false; }
  return true;
}
function exact(name, a, b, where) {
  if (a !== b) { ok = false; console.log(`  ${where}: ${name} is ${a}, recorded ${b}`); return false; }
  return true;
}

for (const c of cases.cases) {
  const patch = new BeliefPatch(spec.belief);
  const schedule = { ...SCHEDULE, ...spec.world.schedule, decisions: 1e9, change: "none" };
  const world = new World(schedule, [], new Rng(0), "mouse");
  const life = new Life(patch, world, c.genome, spec.scale, spec.floors, { arm: c.arm, rng: new Rng(3), machinery: spec.machinery });
  const t0 = performance.now();
  let failures = 0;
  for (const s of c.steps) {
    const where = `${c.name}, decision ${s.t}`;
    const forced = (c.learns || []).find((l) => l.t === s.t);
    if (forced) {
      const before = life.patch.parameters();
      const entry = life.learn(forced.starts.length ? forced.starts.map((row) => Float64Array.from(row)) : null);
      const after = life.patch.parameters();
      const rec = forced.entry;
      exact("learn.window", entry.window, rec.window, where); exact("learn.valid", entry.valid, rec.valid, where); exact("learn.kept", entry.kept, rec.kept, where);
      exact("learn.passes_kept", entry.passes_kept, rec.passes_kept, where); exact("learn.refit", entry.refit, rec.refit, where); exact("learn.moments", entry.moments, rec.moments, where);
      check("learn.loss_before", relScalar(entry.loss_before, rec.loss_before), where);
      check("learn.loss_after", relScalar(entry.loss_after, rec.loss_after), where);
      check("learn.update_size", relScalar(entry.update_size, rec.update_size), where);
      for (const k in after) { const mine = Float64Array.from(after[k], (v, i) => v - before[k][i]); const theirs = forced.delta[k]; if (theirs.every((v) => v === 0)) exact(`learn.delta.${k} zero`, mine.every((v) => v === 0), true, where); else check(`learn.delta.${k}`, rel(mine, theirs), where); }
      if (rec.refit) {
        for (const k of ["home_x", "home_y", "drift"]) check(`learn.habit.${k}`, relScalar(entry.habit_after[k], forced.habit_after[k]), where);
        check("learn.imagined_cost", relScalar(entry.imagined_cost, forced.imagined_cost), where);
        exact("learn.refit_moments", entry.refit_moments, forced.refit_moments, where);
      }
      check("learn.baseline_after", relScalar(life.baseline, forced.baseline_after), where);
      if (forced.state_after) check("learn.state_after", rel(life.patch.state[0], forced.state_after), where);
    }
    world.mouse = s.mouse === null ? null : s.mouse.slice();
    const out = life.decide({ wantCode: false });
    decisions++;
    let fine = true;
    fine &= exact("mode", out.mode, s.mode, where);
    if (s.reading) fine &= check("reading", rel(out.reading, s.reading), where);
    fine &= check("action", rel(out.action, s.action), where);
    fine &= check("belief", rel(life.patch.state[0], s.belief), where);
    fine &= check("expected", rel(out.expected, s.expected), where);
    fine &= check("residual", relScalar(out.residual, s.residual), where);
    fine &= check("surprise", relScalar(out.surprise, s.surprise), where);
    fine &= check("baseline", relScalar(out.baseline, s.baseline), where);
    fine &= check("slow", relScalar(life.slow, s.slow), where);
    fine &= check("readback", rel(out.readback, s.readback), where);
    fine &= check("target", rel(out.target, s.target), where);
    fine &= check("paw", rel(world.paw, s.paw), where);
    if (s.governor) {
      fine &= exact("governor.steps", out.governorSteps, s.governor.steps, where);
      fine &= exact("governor.synapses", life.governor.synapses, s.governor.synapses, where);
      fine &= check("governor.activation", rel(out.governorActivation, s.governor.activation), where);
    }
    if (s.costs) fine &= check("imagination", rel(out.imagined.costs, s.costs), where);
    else fine &= exact("imagined", out.imagined === null, true, where);
    fine &= exact("dot present", world.dot !== null, s.dot !== null, where);
    if (s.dot) fine &= check("dot", rel(world.dot, s.dot), where);
    for (const k of ["appear", "catch", "miss", "vanish", "present"]) fine &= exact(`flag ${k}`, out.flags[k], s.flags[k], where);
    fine &= exact("imagine_left", life.imagineLeft, s.imagine_left, where);
    fine &= exact("above", life.above, s.above, where);
    fine &= check("moments", relScalar(out.moments, s.moments), where);
    fine &= exact("learned this decision", !!out.learn, s.learn, where);
    if (!fine && ++failures >= 3) { console.log("  (stopping this case after three failing decisions)"); break; }
  }
  const ms = performance.now() - t0;
  const modes = { habit: 0, imagine: 0, learn: 0 }; for (const s of c.steps) modes[s.mode]++;
  console.log(`${c.name}: ${c.steps.length} decisions (habit ${modes.habit}, imagine ${modes.imagine}, learn ${modes.learn}) in ${ms.toFixed(0)} ms, ${(ms / c.steps.length).toFixed(2)} ms per decision${failures ? `, ${failures} failing` : ""}`);
}
const summary = Object.entries(worstBy).sort((a, b) => b[1] - a[1]).slice(0, 6).map(([k, v]) => `${k} ${v.toExponential(1)}`).join("  ");
console.log(`worst by quantity: ${summary}`);
console.log(ok ? `PARITY OK (${decisions} decisions, worst relative difference ${worst.toExponential(2)}, tolerance ${TOLERANCE})` : "PARITY FAILED");
process.exit(ok ? 0 : 1);
