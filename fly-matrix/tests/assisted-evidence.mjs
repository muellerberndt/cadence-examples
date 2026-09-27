// Tamper tests against the frozen actual-payload evidence. The genuine checkpoint replay must
// pass first; then separately rehashed mutations must fail for their scientific inconsistency.
// Default: portable scalar checkpoint replay plus exact custody and phase gates.
// --exact retains the immutable producer's bitwise replay (recorded on Apple/Node25).
// Run: node tests/assisted-evidence.mjs [--1024] [--exact] from fly-matrix/.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtempSync, readFileSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { legacyAssistedFixture } from "./fixtures/legacy-assisted.js";
import { compareProbe } from "../tools/verify_assisted_checkpoint.mjs";

const { root: ROOT } = legacyAssistedFixture();
const suffix = process.argv.includes("--1024") ? "_1024" : "";
const exact = process.argv.includes("--exact");
const runner = join(ROOT, `tools/assisted_evidence${suffix}.mjs`);
const portable = fileURLToPath(new URL("../tools/verify_assisted_checkpoint.mjs", import.meta.url));
const original = join(ROOT, `receipts/assisted_evidence${suffix}.json`);
const receipt = JSON.parse(readFileSync(original, "utf8"));
const journal = readFileSync(original + ".journal.jsonl", "utf8");
const check = path => spawnSync(process.execPath,
  exact ? [runner, "--verify", path] : [portable, path, "--source-root", ROOT],
  { encoding: "utf8", maxBuffer: 1024 * 1024 });
const valid = check(original);
assert.equal(valid.status, 0, "genuine receipt/checkpoint replay must pass before mutation tests: " + valid.stderr);
console.log(`ok genuine source-bound ${exact ? "exact" : "portable numerical"} checkpoint replay`);
if (!exact) {
  const result = JSON.parse(valid.stdout);
  console.log(`state hashes equal on this runtime: ${result.rows.filter(r => r.state_hash_matches).length}/${result.rows.length}; hashes are diagnostics, not portable equality gates`);
  assert.equal(result.scalar_absolute_tolerance, 1e-10);
  assert.equal(result.equation_tolerance, 1e-6);
  assert.equal(result.rows.length, 8);
}
const temp = mkdtempSync(join(tmpdir(), "cadence-assisted-evidence-"));
try {
  const variants = [
    { name: "unrehashed probability corruption", mutate: r => { r.arms.plastic.after[1].p[0] += 0.1; }, rehash: false, error: /receipt body hash/ },
    { name: "rehashed missing phase", mutate: r => { delete r.arms.plastic.trials.find(t => t.decision.accepted).decision.solves.plus; }, rehash: true, error: /deep-equal/ },
    { name: "rehashed failed residual", mutate: r => { r.arms.plastic.trials.find(t => t.decision.accepted).independent_residuals.plus = 0.1; }, rehash: true, error: exact ? /independent_residuals/ : /value <= 1e-6/ },
    { name: "rehashed inconsistent plastic weight", mutate: r => { r.arms.plastic.checkpoint.weights[0] += 0.01; }, rehash: true, error: /efficacy\/weight consistency|local efficacy coupling/ },
    { name: "rehashed invented probability", mutate: r => { r.arms.plastic.after[1].p[1] -= 0.01; }, rehash: true, error: /probability replay|probability from neural readouts/ },
    { name: "omitted retained-trial journal event", mutate: () => {}, rehash: true, dropJournalTrial: true, error: /complete journal binding|journal byte hash/ },
  ];
  for (const variant of variants) {
    const r = structuredClone(receipt), path = join(temp, "receipt.json");
    variant.mutate(r);
    if (variant.rehash) { const { sha256, ...body } = r; r.sha256 = createHash("sha256").update(JSON.stringify(body)).digest("hex"); }
    writeFileSync(path, JSON.stringify(r));
    let lines = journal;
    if (variant.dropJournalTrial) {
      const events = journal.trimEnd().split("\n");
      let index = events.findIndex(line => {
        const event = JSON.parse(line); return event.stage === "trial" && !event.row.decision.accepted;
      });
      if (index < 0) index = events.findIndex(line => JSON.parse(line).stage === "trial");
      assert.ok(index >= 0, "this assay retains its trials"); events.splice(index, 1); lines = events.join("\n") + "\n";
    }
    writeFileSync(path + ".journal.jsonl", lines);
    const result = check(path);
    assert.notEqual(result.status, 0, variant.name + " must fail verification");
    assert.match(result.stderr, variant.error, variant.name + " failed for an unexpected reason");
    console.log("ok rejects " + variant.name);
  }
} finally { rmSync(temp, { recursive: true, force: true }); }
// Independently exercise the portable numerical comparator, so custody failures
// cannot hide an accidentally omitted numeric comparison.
const reference = receipt.arms.plastic.after[1];
compareProbe(reference, structuredClone(reference), "unchanged numerical reference");
const numericMutations = [
  ["probability", r => { r.p[0] += 1e-5; r.p[1] -= 1e-5; }, /probability replay/],
  ["output", r => { r.outputs[0] += 1e-5; }, /output replay/],
  ["KC mean", r => { r.kc_mean += 1e-5; }, /KC mean replay/],
  ["independent defect", r => { r.independent_residual += 1e-7; }, /independent residual replay/],
  ["nonfinite defect", r => { r.independent_residual = NaN; }, /independent residual replay/],
  ["solver defect", r => { r.solve.residual += 1e-7; }, /solver residual replay/],
  ["phase status", r => { r.solve.converged = false; }, /phase converged/],
  ["iteration count", r => { r.solve.iterations++; }, /phase iterations/],
  ["tolerance", r => { r.solve.tolerance = 1e-5; }, /phase tolerance/],
  ["active count", r => { r.active_neurons++; }, /active neuron count/],
];
for (const [name, mutate, expected] of numericMutations) {
  const changed = structuredClone(reference); mutate(changed);
  assert.throws(() => compareProbe(changed, reference, name), expected);
}
console.log(`assisted evidence: ${exact ? "exact" : "portable"} replay, six custody mutations and ten numerical/phase mutations passed; no training or full-state numerical replay claim`);
