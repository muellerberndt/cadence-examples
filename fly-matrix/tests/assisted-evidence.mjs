// Tamper tests against the frozen actual-payload evidence. The genuine checkpoint replay must
// pass first; then separately rehashed mutations must fail for their scientific inconsistency.
// Run: node tests/assisted-evidence.mjs [--1024] from fly-matrix/.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtempSync, readFileSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import { legacyAssistedFixture } from "./fixtures/legacy-assisted.js";

const { root: ROOT } = legacyAssistedFixture();
const suffix = process.argv.includes("--1024") ? "_1024" : "";
const runner = join(ROOT, `tools/assisted_evidence${suffix}.mjs`);
const original = join(ROOT, `receipts/assisted_evidence${suffix}.json`);
const receipt = JSON.parse(readFileSync(original, "utf8"));
const journal = readFileSync(original + ".journal.jsonl", "utf8");
const check = path => spawnSync(process.execPath, [runner, "--verify", path], { encoding: "utf8", maxBuffer: 1024 * 1024 });
const valid = check(original);
assert.equal(valid.status, 0, "genuine receipt/checkpoint replay must pass before mutation tests: " + valid.stderr);
console.log("ok genuine source-bound checkpoint replay");
const temp = mkdtempSync(join(tmpdir(), "cadence-assisted-evidence-"));
try {
  const variants = [
    { name: "unrehashed probability corruption", mutate: r => { r.arms.plastic.after[1].p[0] += 0.1; }, rehash: false, error: /receipt body hash/ },
    { name: "rehashed missing phase", mutate: r => { delete r.arms.plastic.trials.find(t => t.decision.accepted).decision.solves.plus; }, rehash: true, error: /deep-equal/ },
    { name: "rehashed failed residual", mutate: r => { r.arms.plastic.trials.find(t => t.decision.accepted).independent_residuals.plus = 0.1; }, rehash: true, error: /independent_residuals/ },
    { name: "rehashed inconsistent plastic weight", mutate: r => { r.arms.plastic.checkpoint.weights[0] += 0.01; }, rehash: true, error: /efficacy\/weight consistency/ },
    { name: "rehashed invented probability", mutate: r => { r.arms.plastic.after[1].p[1] -= 0.01; }, rehash: true, error: /probability replay/ },
    { name: "omitted retained-trial journal event", mutate: () => {}, rehash: true, dropJournalTrial: true, error: /complete journal binding/ },
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
console.log("assisted evidence: genuine replay and six causal/provenance mutation tests passed");
