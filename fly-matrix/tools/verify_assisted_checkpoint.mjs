#!/usr/bin/env node
// Portable checkpoint qualification, separate from the immutable assay producers.
// Custody remains exact. Replayed historical scalars use a declared absolute 1e-10
// tolerance; phase decisions/counts and the original 1e-6 equation gate stay exact.
// Receipts store whole-state hashes, not whole states. A differing replay hash is
// reported, never called an elementwise or bitwise historical state reproduction.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { spawnSync } from "node:child_process";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
export const SCALAR_ABSOLUTE_TOLERANCE = 1e-10;
const hashArray = a => createHash("sha256").update(Buffer.from(a.buffer, a.byteOffset, a.byteLength)).digest("hex");
const near = (actual, expected, label) => {
  assert.ok(typeof actual === "number" && Number.isFinite(actual)
    && typeof expected === "number" && Number.isFinite(expected)
    && Math.abs(actual - expected) <= SCALAR_ABSOLUTE_TOLERANCE, `${label}: ${actual} != ${expected}`);
};

export function compareProbe(actual, recorded, label) {
  assert.equal(actual.label, recorded.label, `${label} label`);
  for (const key of ["converged", "iterations", "reason", "tolerance"])
    assert.equal(actual.solve[key], recorded.solve[key], `${label} phase ${key}`);
  near(actual.solve.residual, recorded.solve.residual, `${label} solver residual replay`);
  near(actual.independent_residual, recorded.independent_residual, `${label} independent residual replay`);
  assert.equal(actual.outputs.length, recorded.outputs.length);
  actual.outputs.forEach((v, i) => near(v, recorded.outputs[i], `${label} output replay ${i}`));
  if (actual.p === null || recorded.p === null) assert.equal(actual.p, recorded.p, `${label} probability availability`);
  else {
    assert.equal(actual.p.length, recorded.p.length);
    actual.p.forEach((v, i) => near(v, recorded.p[i], `${label} probability replay ${i}`));
  }
  near(actual.kc_mean, recorded.kc_mean, `${label} KC mean replay`);
  assert.equal(actual.active_neurons, recorded.active_neurons, `${label} active neuron count`);
  assert.equal(actual.saturation, recorded.saturation, `${label} output saturation`);
}

// Independent activation/equation evaluation over ALL neurons, not solver caches.
function inspectState(brain) {
  const rest = 1 / (1 + Math.exp(brain.slope * brain.threshold));
  const activity = Float64Array.from(brain.v, v => {
    assert.ok(Number.isFinite(v), "nonfinite replay potential");
    const d = 1 / (1 + Math.exp(-brain.slope * (v - brain.threshold))) - rest;
    return d > 0 ? d / (1 - rest) : brain.leak ? d * brain.leak / rest : 0;
  });
  let residual = 0, activationError = 0;
  for (let i = 0; i < brain.n; i++) {
    assert.ok(Number.isFinite(brain.s[i]), "nonfinite replay activity");
    activationError = Math.max(activationError, Math.abs(activity[i] - brain.s[i]));
    let input = 0;
    for (let e = brain.rowPtr[i]; e < brain.rowPtr[i + 1]; e++) input += brain.w[e] * activity[brain.pre[e]];
    const defect = input + (brain.drive[i] + brain.bias[i]) - brain.v[i];
    assert.ok(Number.isFinite(defect), "nonfinite independent equation defect");
    residual = Math.max(residual, Math.abs(defect));
  }
  assert.ok(activationError <= 1e-12, "all-neuron potential/activity consistency");
  return { residual, activationError };
}

export async function verifyCheckpoint(receiptPath, sourceRoot = ROOT) {
  const audit = spawnSync(process.execPath,
    [resolve(sourceRoot, "tools/verify_assisted_evidence.mjs"), receiptPath],
    { cwd: sourceRoot, encoding: "utf8", maxBuffer: 1024 * 1024 });
  assert.equal(audit.status, 0, `strict receipt audit failed:\n${audit.stderr}`);
  const receipt = JSON.parse(readFileSync(receiptPath, "utf8")), protocol = receipt.protocol;
  // The independent audit above requires all imported solver/learner/payload bytes
  // to match the historical source pins before any of them execute here.
  const { SettlingBrain } = await import(pathToFileURL(resolve(sourceRoot, "web/brain.js")).href);
  const { ActorCriticLearner } = await import(pathToFileURL(resolve(sourceRoot, "web/learner.js")).href);
  const payload = JSON.parse(readFileSync(resolve(sourceRoot, "web/data/brain.json"), "utf8"));
  const rows = [];
  for (const name of protocol.arms) {
    const arm = receipt.arms[name], brain = new SettlingBrain(payload);
    const learner = new ActorCriticLearner(brain, protocol.learner);
    assert.deepEqual(Array.from(learner.edges), arm.checkpoint.edges, "declared plastic seam");
    arm.checkpoint.edges.forEach((e, i) => { brain.w[e] = arm.checkpoint.weights[i]; });
    assert.equal(hashArray(brain.w), arm.checkpoint.learned_weight_sha256, "exact checkpoint reconstruction");
    for (const recorded of arm.after) {
      brain.reset(); brain.clearStimuli();
      const expectedDrive = new Float64Array(brain.n);
      for (const [sense, level] of Object.entries(recorded.senses)) {
        brain.stimulate(sense, level);
        for (const i of brain.sets[sense]) expectedDrive[i] = Math.max(expectedDrive[i], brain.amplitude * level);
      }
      assert.deepEqual(brain.drive, expectedDrive, "all-neuron sensory drive");
      const solve = brain.settleControl(protocol.solve.maxSteps, protocol.solve.tolerance);
      const { residual, activationError } = inspectState(brain);
      assert.equal(solve.tolerance, protocol.solve.tolerance);
      assert.ok(Number.isSafeInteger(solve.iterations) && solve.iterations >= 0 && solve.iterations <= protocol.solve.maxSteps);
      if (solve.converged) assert.ok(residual <= protocol.solve.tolerance, "original full-equation qualification gate");
      else {
        assert.equal(solve.reason, "iteration_limit"); assert.equal(solve.iterations, protocol.solve.maxSteps);
        assert.ok(residual > protocol.solve.tolerance, "failed probe must remain unqualified");
      }
      const outputs = protocol.learner.outputs.map(n => brain.s[brain.sets[n][0]]);
      const logits = outputs.map(v => v / protocol.learner.temperature), maximum = Math.max(...logits);
      const e = logits.map(v => Math.exp(v - maximum)), sum = e.reduce((a, b) => a + b, 0);
      let active = 0, kcSum = 0;
      for (const v of brain.s) if (v >= 0.05) active++;
      for (const i of brain.sets.kc) kcSum += brain.s[i];
      const actual = { label: recorded.label, solve, independent_residual: residual, outputs,
        p: solve.converged ? e.map(v => v / sum) : null,
        kc_mean: kcSum / brain.sets.kc.length, active_neurons: active,
        saturation: outputs.filter(v => v < 0.02 || v > 0.98).length / outputs.length };
      compareProbe(actual, recorded, `${name}/${recorded.label}`);
      assert.equal(hashArray(brain.w), arm.checkpoint.learned_weight_sha256, "checkpoint replay cannot learn");
      rows.push({ arm: name, ...actual, activationError,
        state_hash_matches: hashArray(brain.s) === recorded.state_sha256,
        potential_hash_matches: hashArray(brain.v) === recorded.potential_sha256 });
    }
  }
  return { schema: "cadence.assisted-portable-checkpoint/v1", runtime: process.version,
    platform: process.platform, architecture: process.arch, scalar_absolute_tolerance: SCALAR_ABSOLUTE_TOLERANCE,
    equation_tolerance: protocol.solve.tolerance, receipt_sha256: receipt.sha256, rows,
    scope: "Exact historical source/receipt/journal/parameter custody; checkpoint-only replay with full-equation qualification and scalar numerical comparison. No training replay, whole-state elementwise equality, cross-platform bitwise identity, generalization or flight claim." };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2), at = args.indexOf("--source-root");
  const sourceRoot = at >= 0 ? args.splice(at, 2)[1] : ROOT;
  assert.ok(args.length === 1 && sourceRoot, "usage: node tools/verify_assisted_checkpoint.mjs RECEIPT [--source-root ARCHIVED_ROOT]");
  console.log(JSON.stringify(await verifyCheckpoint(resolve(args[0]), resolve(sourceRoot)), null, 2));
}
