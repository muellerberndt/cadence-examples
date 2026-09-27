#!/usr/bin/env node
// Canonical independent verifier for the two frozen assisted-learning assays.
// node tools/verify_assisted_evidence.mjs RECEIPT [--replay]
// Default: custody, journal semantics, protocol, diagnostics, updates, checkpoints and claims.
// --replay additionally reruns the entire frozen assay, including every training/control phase,
// and compares all fields except explicit execution times and the resulting receipt/journal hashes.
// The frozen producers remain unchanged. Hashes establish consistency, not a signed provenance.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync, mkdtempSync, rmSync } from "node:fs";
import { resolve, join } from "node:path";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import { runInNewContext } from "node:vm";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const hash = bytes => createHash("sha256").update(bytes).digest("hex");
const finite = n => typeof n === "number" && Number.isFinite(n);
const same = (a, b, label) => assert.deepEqual(a, b, label);
const digest = (s, label) => assert.match(s, /^[0-9a-f]{64}$/, label);
const bytes = a => Buffer.from(a.buffer, a.byteOffset, a.byteLength);
const config = { outputs: ["mbon:MBON11:right", "mbon:MBON05:left"], actions: [0, 1], plastic: { pre: ["kc"], post: ["mbon"] }, critic: "kc", beta: 0.1, temperature: 0.3, gamma: 0.95, lam: 0.9, eta: 1, etaBias: 0, etaCritic: 0.05, cap: 3, dopamineCap: 1, tonic: {} };
const scope = "bounded in-silico causal diagnostic on the actual 60000-neuron demo payload; no body/pilot, visual perception, flight competence, generalization, biological equivalence or exact-gradient claim";
const labels = ["fruit", "yeast", "fruit_odor_lesion", "yeast_odor_lesion"];
const odors = ["decaying_fruit", "yeasty"];
const thresholds = { causal_probability_change: 1e-4, causal_weight_change: 1e-9, restoration_probability_error: 1e-10, independent_update_error: 1e-10, direction_improvement: 1e-3 };
const read = path => JSON.parse(readFileSync(path, "utf8"));
function decode(value, T) {
  const b = Buffer.from(value, "base64"), copy = new Uint8Array(b.length); copy.set(b);
  return new T(copy.buffer);
}
function softmax(outputs) {
  const z = outputs.map(x => x / config.temperature), maximum = Math.max(...z);
  const e = z.map(x => Math.exp(x - maximum)), sum = e.reduce((a, b) => a + b, 0);
  return e.map(x => x / sum);
}
const qualified = rows => rows.every(p => p.solve.converged && finite(p.independent_residual) && p.independent_residual <= 1e-6);
const score = rows => rows[0].p && rows[1].p ? (rows[0].p[0] + rows[1].p[1]) / 2 : null;
const delta = (a, b) => a[0].p && a[1].p && b[0].p && b[1].p ? Math.max(...[0, 1].map(i => Math.abs(a[i].p[0] - b[i].p[0]))) : null;
function metrics(arm) {
  const after = score(arm.after);
  return { before_score: score(arm.before), after_score: after, wrong_odor_score: after === null ? null : 1 - after,
    maximum_probability_change: delta(arm.before, arm.after), restore_error: delta(arm.before, arm.restored), reinstate_error: delta(arm.after, arm.reinstated) };
}
function claims(body) {
  const a = body.arms.plastic, f = body.arms.frozen_actor, am = metrics(a), fm = metrics(f);
  const lessons = a.trials.filter(t => t.update?.accepted).length;
  const odor = a.before[0].p && a.before[1].p ? Math.abs(a.before[0].p[0] - a.before[1].p[0]) : null;
  return {
    input_changes_settled_output: qualified(a.before) && odor > thresholds.causal_probability_change,
    odor_lesion_removes_label_information: qualified(a.after.slice(2)) && a.after[2].state_sha256 === a.after[3].state_sha256,
    all_training_phases_qualified: Object.values(body.arms).every(x => x.trials.every(t => t.decision.accepted && t.update?.accepted)),
    accepted_local_updates_independently_reproduced: Object.values(body.arms).every(x => x.trials.filter(t => t.update?.accepted).every(t => finite(t.independent_update_max_error) && t.independent_update_max_error >= 0 && t.independent_update_max_error <= thresholds.independent_update_error)),
    accepted_plastic_lessons: lessons,
    local_weight_memory_changes_output: lessons > 0 && qualified(a.after) && a.weight_change.max_absolute > thresholds.causal_weight_change && am.maximum_probability_change > thresholds.causal_probability_change,
    frozen_actor_unchanged: qualified(f.before) && qualified(f.after) && f.weight_change.initial_sha256 === f.weight_change.learned_sha256 && fm.maximum_probability_change <= thresholds.restoration_probability_error,
    initial_weight_restoration_removes_effect: qualified(a.before) && qualified(a.restored) && am.restore_error <= thresholds.restoration_probability_error,
    learned_weight_reinstatement_recovers_effect: qualified(a.after) && qualified(a.reinstated) && am.reinstate_error <= thresholds.restoration_probability_error,
    rewarded_discrimination_improved: qualified(a.before) && qualified(a.after) && am.after_score - am.before_score > thresholds.direction_improvement,
    learned_greedy_discrimination: !!(qualified(a.after) && a.after[0].p[0] > 0.5 && a.after[1].p[0] < 0.5),
    shuffled_all_free_probes_qualified: qualified(body.shuffled.probes), scope,
  };
}
function checkPhase(s, cap) {
  assert.equal(typeof s.converged, "boolean");
  assert.ok(Number.isSafeInteger(s.iterations) && s.iterations >= 0 && s.iterations <= cap, "phase iteration cap");
  assert.equal(s.tolerance, 1e-6, "phase tolerance changed");
  assert.ok(s.residual === null || (finite(s.residual) && s.residual >= 0), "invalid phase residual");
  if (s.converged) {
    assert.ok(finite(s.residual) && s.residual <= s.tolerance, "unqualified accepted phase");
    assert.equal(s.reason, "residual_tolerance");
  } else if (s.reason === "iteration_limit") {
    assert.equal(s.iterations, cap); assert.ok(finite(s.residual) && s.residual > s.tolerance);
  }
}
function verify(path) {
  const body = read(path), { sha256, ...unsigned } = body;
  assert.equal(hash(JSON.stringify(unsigned)), sha256, "receipt body hash");
  const p = body.protocol, cap = p.solve.maxSteps, followup = cap === 1024;
  assert.ok(cap === 512 || followup); same(p.solve, { maxSteps: cap, tolerance: 1e-6 });
  assert.equal(body.schema, `cadence.fly-matrix.assisted-evidence/${followup ? "budget-followup-1024-v1" : "v1"}`);
  assert.equal(p.schema, `cadence.fly-matrix.assisted-evidence-protocol/${followup ? "budget-followup-1024-v1" : "v1"}`);
  assert.equal(p.seed, 3); assert.equal(p.trials_per_arm, 24); assert.equal(p.scope, scope);
  same(p.arms, ["plastic", "frozen_actor"]); same(Object.keys(body.arms), p.arms);
  same(p.learner, config); same(p.thresholds, thresholds); same(p.probes, labels);
  same(p.observation, { strongest: 0.5, other: 0.16, context: "data/lessons.json setup.decision_senses; synthetic fixed flight-context sensor levels, not rendered vision" });
  same(p.shuffled, { seed: 1, training: false, calibration: false, interpretation: "structural lesion, not a matched learning-capacity benchmark; saturation and convergence must be reported" });
  const paths = ["tools/assisted_evidence.mjs", ...(followup ? ["tools/assisted_evidence_1024.mjs"] : []), "web/brain.js", "web/learner.js", "web/data/brain.json", "web/data/lessons.json"];
  same(body.sources, paths.map(path => ({ path, sha256: hash(readFileSync(resolve(ROOT, path))) })), "source binding");
  // Read only the trusted frozen runner's declarative protocol, not its verifier or
  // mathematical checks. The independent operational checks below remain separate.
  const runner = followup ? paths[1] : paths[0];
  const declaration = readFileSync(resolve(ROOT, runner), "utf8").match(/const PROTOCOL = (\{[\s\S]*?\n\});\nconst loadPayload/);
  assert.ok(declaration, "frozen protocol declaration not found");
  const declared = runInNewContext(`(${declaration[1]})`, { CONFIG: config }, { timeout: 1000 });
  same(p, JSON.parse(JSON.stringify(declared)), "complete frozen protocol declaration");
  if (followup) {
    assert.equal(p.budget_followup.kind, "adaptive_compute_budget_followup");
    assert.equal(p.budget_followup.original_receipt, "receipts/assisted_evidence.json");
    const original = read(resolve(ROOT, p.budget_followup.original_receipt)), { sha256: pinned, ...old } = original;
    assert.equal(hash(JSON.stringify(old)), pinned); assert.equal(pinned, p.budget_followup.original_receipt_sha256);
    assert.equal(hash(readFileSync(resolve(ROOT, paths[0]))), p.budget_followup.original_runner_file_sha256);
  }
  const journalBytes = readFileSync(path + ".journal.jsonl");
  assert.equal(hash(journalBytes), body.journal_sha256, "journal byte hash");
  const journal = journalBytes.toString("utf8").trimEnd().split("\n").map(s => JSON.parse(s));
  const expectedJournal = [{ stage: "frozen_protocol", protocol: p, sources: body.sources, started: body.started }];
  let seed = 3;
  const draws = Array.from({ length: 24 }, () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296));
  same(body.draws, draws, "declared random stream");
  const payload = read(resolve(ROOT, "web/data/brain.json")), setup = read(resolve(ROOT, "web/data/lessons.json")).setup;
  same(body.payload, { neurons: payload.n, edges: payload.edges, model: payload.model });
  for (const name of config.outputs) assert.equal(payload.populations[name].length, 1);
  const sensory = (odor, lesion = false) => {
    const senses = { ...setup.decision_senses };
    for (let k = 0; k < 2; k++) for (const side of ["left", "right"]) senses[`orn:${odors[k]}:${side}`] = lesion ? 0 : k === odor ? 0.5 : 0.16;
    return senses;
  };
  const originalW = decode(payload.arrays.weight, Float64Array), pre = decode(payload.arrays.pre, Int32Array), row = decode(payload.arrays.row_ptr, Int32Array);
  const factors = payload.arrays.gain_pre ? decode(payload.arrays.gain_pre, Float64Array) : null;
  const counts = payload.arrays.count ? decode(payload.arrays.count, Uint16Array) : null;
  const gains = payload.arrays.log_gain ? decode(payload.arrays.log_gain, Float64Array) : null;
  const kc = new Set(payload.populations.kc), mbon = new Set(payload.populations.mbon), plastic = [];
  for (let i = 0; i < payload.n; i++) if (mbon.has(i)) for (let e = row[i]; e < row[i + 1]; e++) if (kc.has(pre[e])) plastic.push(e);
  const initialHash = hash(bytes(originalW));
  function checkProbes(probes) {
    assert.equal(probes.length, 4);
    probes.forEach((r, i) => {
      assert.equal(r.label, labels[i]); same(r.senses, sensory(i % 2, i >= 2)); checkPhase(r.solve, cap);
      assert.ok(finite(r.independent_residual) && r.independent_residual >= 0);
      assert.ok(finite(r.milliseconds) && r.milliseconds >= 0);
      assert.equal(r.outputs.length, 2); assert.ok(r.outputs.every(finite));
      digest(r.state_sha256); digest(r.potential_sha256);
      if (r.solve.converged) {
        assert.ok(r.independent_residual <= 1e-6); same(r.p, softmax(r.outputs), "probability from neural readouts");
      } else assert.equal(r.p, null, "failed probes must not supply probabilities");
      assert.equal(r.saturation, r.outputs.filter(s => s < 0.02 || s > 0.98).length / 2);
      assert.ok(finite(r.kc_mean) && Number.isSafeInteger(r.active_neurons) && r.active_neurons >= 0 && r.active_neurons <= payload.n);
    });
  }
  for (const name of p.arms) {
    const arm = body.arms[name]; assert.equal(arm.name, name); assert.equal(arm.trials.length, 24);
    checkProbes(arm.before); expectedJournal.push(...arm.before.map(row => ({ stage: "probe", row, arm: name, phase: "before" })));
    let previousHash = initialHash, updates = 0;
    for (const [i, t] of arm.trials.entries()) {
      expectedJournal.push({ stage: "trial", arm: name, row: t });
      assert.equal(t.trial, i); assert.equal(t.odor, odors[i % 2]); assert.equal(t.uniform, draws[i]);
      same(t.senses, sensory(i % 2)); assert.equal(t.weight_before_sha256, previousHash, "continuous weight history");
      digest(t.weight_after_sha256); previousHash = t.weight_after_sha256;
      assert.ok(finite(t.milliseconds) && t.milliseconds >= 0);
      const d = t.decision; assert.equal(typeof d.accepted, "boolean");
      for (const solve of Object.values(d.solves)) checkPhase(solve, cap);
      if (d.accepted) {
        same(Object.keys(d.solves).sort(), ["free", "minus", "plus"]);
        assert.ok(Object.values(d.solves).every(s => s.converged));
        assert.equal(d.greedy, false); assert.equal(d.draw, draws[i]); assert.equal(d.reason, "residual_qualified");
        assert.equal(d.p.length, 2); assert.ok(d.p.every(x => finite(x) && x >= 0 && x <= 1));
        assert.ok(Math.abs(d.p[0] + d.p[1] - 1) <= 1e-12);
        assert.equal(d.choice, draws[i] > d.p[0] ? 1 : 0); assert.equal(d.action, d.choice); assert.ok(finite(d.value));
        same(Object.keys(t.independent_residuals).sort(), ["free", "minus", "plus"]);
        same(Object.keys(t.phase_potential_sha256).sort(), ["free", "minus", "plus"]);
        for (const value of Object.values(t.independent_residuals)) assert.ok(finite(value) && value >= 0 && value <= 1e-6);
        Object.values(t.phase_potential_sha256).forEach(s => digest(s));
        const reward = d.choice === 0 ? i % 2 === 0 ? 1 : -1 : 0;
        assert.equal(t.reward, reward); assert.equal(typeof t.update.accepted, "boolean");
        if (t.update.accepted) {
          assert.ok(finite(t.independent_update_max_error) && t.independent_update_max_error >= 0 && t.independent_update_max_error <= thresholds.independent_update_error);
          const lesson = t.update.lesson, td = reward - d.value;
          assert.equal(lesson.reward, reward); assert.equal(lesson.value, d.value); assert.equal(lesson.nextValue, 0);
          assert.equal(lesson.tdError, td); assert.equal(lesson.delta, Math.max(-1, Math.min(1, td)));
          assert.equal(lesson.updates, ++updates); assert.equal(lesson.dropped, 0);
        } else { assert.equal(t.independent_update_max_error, null); assert.equal(t.weight_before_sha256, t.weight_after_sha256); }
      } else {
        assert.ok(!("action" in d) && !("p" in d)); assert.equal(t.reward, null); assert.equal(t.update, null);
        same(t.independent_residuals, {}); same(t.phase_potential_sha256, {}); assert.equal(t.independent_update_max_error, null);
        assert.equal(t.weight_before_sha256, t.weight_after_sha256);
        assert.ok(Object.values(d.solves).some(s => !s.converged), "rejected assay decision has no failed phase");
      }
    }
    for (const phase of ["after", "restored", "reinstated"]) {
      checkProbes(arm[phase]); expectedJournal.push(...arm[phase].map(row => ({ stage: "probe", row, arm: name, phase })));
    }
    same(arm.metrics, metrics(arm), "metrics independently derived from probes");
    same(arm.checkpoint.edges, plastic, "independent KC-to-MBON edge selection");
    assert.equal(arm.plastic_edges, plastic.length); assert.equal(arm.checkpoint.weights.length, plastic.length); assert.equal(arm.checkpoint.efficacy.length, plastic.length);
    const restored = Float64Array.from(originalW);
    for (let k = 0; k < plastic.length; k++) {
      const e = plastic[k], efficacy = arm.checkpoint.efficacy[k], weight = arm.checkpoint.weights[k];
      assert.ok(finite(efficacy) && Math.abs(efficacy) <= config.cap && finite(weight));
      const factor = factors ? factors[e] : payload.model.gain * (counts ? counts[e] : 1) * (gains ? Math.exp(gains[pre[e]]) : 1);
      const expected = factor * efficacy;
      assert.ok(Math.abs(weight - expected) <= 1e-14 * Math.max(1, Math.abs(expected)), "local efficacy coupling"); restored[e] = weight;
    }
    let changed = 0, maximum = 0;
    for (let e = 0; e < restored.length; e++) { const d = Math.abs(restored[e] - originalW[e]); if (d > 1e-12) changed++; maximum = Math.max(maximum, d); }
    const learnedHash = hash(bytes(restored));
    same(arm.weight_change, { changed, max_absolute: maximum, initial_sha256: initialHash, learned_sha256: learnedHash });
    assert.equal(arm.checkpoint.learned_weight_sha256, learnedHash); assert.equal(previousHash, learnedHash);
    if (name === "frozen_actor") assert.equal(learnedHash, initialHash, "frozen actor really froze");
  }
  same(body.shuffled.intervention, p.shuffled); checkProbes(body.shuffled.probes);
  let shuffleSeed = 1;
  const permutation = Int32Array.from({ length: payload.n }, (_, i) => i);
  for (let i = permutation.length - 1; i > 0; i--) {
    shuffleSeed = (Math.imul(shuffleSeed, 1664525) + 1013904223) >>> 0;
    const j = Math.floor((shuffleSeed / 4294967296) * (i + 1)); [permutation[i], permutation[j]] = [permutation[j], permutation[i]];
  }
  assert.equal(body.shuffled.permutation_sha256, hash(bytes(permutation)), "declared structural lesion");
  expectedJournal.push(...body.shuffled.probes.map(row => ({ stage: "probe", row, arm: "shuffled", phase: "inference_only" })));
  same(journal, expectedJournal, "complete ordered journal/receipt semantic equality");
  same(body.claims, claims(body), "claim booleans independently derived from audited probes and updates");
  return { body, runner };
}
function semantics(value, path = []) {
  if (Array.isArray(value)) return value.map((v, i) => semantics(v, [...path, i]));
  if (value && typeof value === "object") return Object.fromEntries(Object.entries(value)
    .filter(([key]) => key !== "milliseconds" && !(path.length === 0 && ["started", "completed", "sha256", "journal_sha256"].includes(key)))
    .map(([key, v]) => [key, semantics(v, [...path, key])]));
  return value;
}

const args = process.argv.slice(2), replay = args.includes("--replay"), names = args.filter(a => a !== "--replay");
assert.ok(names.length === 1 && !names[0].startsWith("--"), "usage: node tools/verify_assisted_evidence.mjs RECEIPT [--replay]");
const path = resolve(names[0]), checked = verify(path);
console.log(`Strict receipt audit passed: ${path}; all trials, diagnostics, journal rows, metrics, claims and weight checkpoints checked.`);
if (replay) {
  const dir = mkdtempSync(join(tmpdir(), "fly-assisted-replay-")), out = join(dir, "receipt.json");
  console.log(`Full frozen training/control replay starting; temporary evidence: ${dir}`);
  try {
    const run = spawnSync(process.execPath, [checked.runner, "--run", "--out", out], { cwd: ROOT, stdio: "inherit" });
    assert.equal(run.status, 0, "frozen replay failed");
    const actual = verify(out);
    same(semantics(actual.body), semantics(checked.body), "full semantic replay, including all training and restoration/lesion/shuffle probes");
    rmSync(dir, { recursive: true, force: true });
    console.log("Full replay matched every semantic field, trial, phase diagnostic, weight checkpoint and control probe. Temporary files removed; this does not establish generalization or flight.");
  } catch (error) {
    console.error(`Replay failed; evidence preserved at ${dir}`); throw error;
  }
} else console.log("No neural trajectory was rerun. Add --replay for full training and all control-probe reproduction.");
