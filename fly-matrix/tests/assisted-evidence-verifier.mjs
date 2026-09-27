// Adversarial receipts are temporary copies. Frozen experiments and their hashes never change.
import assert from "node:assert/strict";
import { readFileSync, writeFileSync, mkdtempSync, rmSync } from "node:fs";
import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { legacyAssistedFixture } from "./fixtures/legacy-assisted.js";

const { root: ROOT } = legacyAssistedFixture();
const hash = value => createHash("sha256").update(value).digest("hex");
const original = JSON.parse(readFileSync(join(ROOT, "receipts/assisted_evidence.json"), "utf8"));
const journal = readFileSync(join(ROOT, "receipts/assisted_evidence.json.journal.jsonl"), "utf8");
const dir = mkdtempSync(join(tmpdir(), "fly-assisted-verifier-test-")), path = join(dir, "receipt.json");
const accepted = body => body.arms.plastic.trials.find(t => t.decision.accepted);
const mutations = [
  ["null accepted residual", b => { accepted(b).decision.solves.plus.residual = null; }, /unqualified accepted phase/],
  ["negative residual", b => { accepted(b).decision.solves.plus.residual = -1; }, /invalid phase residual/],
  ["excess iterations", b => { accepted(b).decision.solves.plus.iterations = 513; }, /phase iteration cap/],
  ["tolerance drift", b => { accepted(b).decision.solves.plus.tolerance = 1; }, /phase tolerance changed/],
  ["protocol-description drift", b => { b.protocol.reward = "an invented reward protocol"; }, /complete frozen protocol declaration/],
  ["stored metric falsehood", b => { b.arms.plastic.metrics.after_score = 1; }, /metrics independently derived/],
  ["invented positive claim", b => { b.claims.rewarded_discrimination_improved = true; }, /claim booleans independently derived/],
  ["journal not joined", (b, rows) => { rows.find(r => r.stage === "trial").row.senses["haltere:left"] = 0; }, /journal\/receipt semantic equality/],
  ["wrong supplied reward", b => { accepted(b).reward += 1; }, /AssertionError/],
  ["missing trial", b => { b.arms.plastic.trials.pop(); }, /AssertionError/],
  ["checkpoint corruption", b => { b.arms.plastic.checkpoint.efficacy[0] += 0.1; }, /local efficacy coupling/],
];
try {
  for (const [name, mutate, expected] of mutations) {
    const body = structuredClone(original), rows = journal.trimEnd().split("\n").map(s => JSON.parse(s));
    mutate(body, rows);
    const changedJournal = rows.map(row => JSON.stringify(row)).join("\n") + "\n";
    body.journal_sha256 = hash(changedJournal);
    const { sha256, ...unsigned } = body;
    writeFileSync(path, JSON.stringify({ ...unsigned, sha256: hash(JSON.stringify(unsigned)) }));
    writeFileSync(path + ".journal.jsonl", changedJournal);
    const result = spawnSync(process.execPath, ["tools/verify_assisted_evidence.mjs", path], { encoding: "utf8", cwd: ROOT });
    assert.notEqual(result.status, 0, `${name} incorrectly passed`);
    assert.match(result.stderr, expected, `${name} did not fail at the expected check`);
    console.log(`rejected ${name}`);
  }
} finally { rmSync(dir, { recursive: true, force: true }); }
console.log(`${mutations.length} adversarial verifier cases passed; originals untouched`);
