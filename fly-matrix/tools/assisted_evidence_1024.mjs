#!/usr/bin/env node
// Adaptive compute-budget follow-up; original512 evidence is preserved.
// Actual-payload causal learning diagnostic. No body, flight pilot, rendered vision or training
// success selection. Every trial and rejected solve is retained. Source hashes freeze before run.
// node tools/assisted_evidence_1024.mjs --run [--out receipts/assisted_evidence_1024.json]
// node tools/assisted_evidence_1024.mjs --verify receipts/assisted_evidence_1024.json
// Verification re-solves saved checkpoints; it does not replay training or establish generalization.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { existsSync, readFileSync, writeFileSync, appendFileSync, mkdirSync } from "node:fs";
import { dirname, resolve, relative } from "node:path";
import { fileURLToPath } from "node:url";
import { SettlingBrain } from "../web/brain.js";
import { ActorCriticLearner } from "../web/learner.js";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const SOURCES = ["tools/assisted_evidence.mjs", "tools/assisted_evidence_1024.mjs", "web/brain.js", "web/learner.js", "web/data/brain.json", "web/data/lessons.json"];
const hash = value => createHash("sha256").update(value).digest("hex");
const bytes = array => Buffer.from(array.buffer, array.byteOffset, array.byteLength);
const hashArray = array => hash(bytes(array));
const sourceHashes = () => SOURCES.map(path => ({ path, sha256: hash(readFileSync(resolve(ROOT, path))) }));
const OUTPUTS = ["mbon:MBON11:right", "mbon:MBON05:left"];
const ODORS = ["decaying_fruit", "yeasty"];
const CONFIG = { outputs: OUTPUTS, actions: [0, 1], plastic: { pre: ["kc"], post: ["mbon"] }, critic: "kc", beta: 0.1, temperature: 0.3, gamma: 0.95, lam: 0.9, eta: 1, etaBias: 0, etaCritic: 0.05, cap: 3, dopamineCap: 1, tonic: {} };
const PROTOCOL = {
  schema: "cadence.fly-matrix.assisted-evidence-protocol/budget-followup-1024-v1",
  scope: "bounded in-silico causal diagnostic on the actual 60000-neuron demo payload; no body/pilot, visual perception, flight competence, generalization, biological equivalence or exact-gradient claim",
  budget_followup: {"kind": "adaptive_compute_budget_followup", "original_receipt": "receipts/assisted_evidence.json", "original_receipt_sha256": "e1684c14223f4b09453d248bb34f02c1bc90f3ea75b7b2f735e68a4e61608b1e", "original_runner_file_sha256": "c9f16eb16c3c32acd2e5dad346979cd416052a4acfe912914a16ffb128d86688", "changed": "maxSteps increases from512 to1024; same graph, dt=.2, input amplitudes, initial weights/biases, rewards, RNG draws, trial count and learning parameters", "reason": "Original fruit free phases failed the512 cap. A read-only residual trace of unchanged equations subsequently reached tolerance at1020 iterations. This resource follow-up is not an independent confirmatory experiment."},
  seed: 3, trials_per_arm: 24, arms: ["plastic", "frozen_actor"],
  learner: CONFIG, solve: { maxSteps: 1024, tolerance: 1e-6 },
  trials: "alternate fruit then yeast; reset neural state before every trial; identical exogenous LCG uniforms in both arms; no outcome-based stopping or retuning",
  reward: "fruit approach +1; yeast approach -1; avoidance 0; one terminal outcome per accepted neural choice",
  frozen_actor: "eta=0 and etaBias=0; critic still learns, same qualified phases and observations; actor weights cannot move",
  observation: { strongest: 0.5, other: 0.16, context: "data/lessons.json setup.decision_senses; synthetic fixed flight-context sensor levels, not rendered vision" },
  probes: ["fruit", "yeast", "fruit_odor_lesion", "yeast_odor_lesion"],
  controls: ["frozen actor", "remove both odor inputs while retaining nominal label", "swap odor identity at inference", "restore exact initial weights", "reinstate learned weights", "permute presynaptic neuron identities at inference with frozen numeric weights and biases"],
  shuffled: { seed: 1, training: false, calibration: false, interpretation: "structural lesion, not a matched learning-capacity benchmark; saturation and convergence must be reported" },
  thresholds: { causal_probability_change: 1e-4, causal_weight_change: 1e-9, restoration_probability_error: 1e-10, independent_update_error: 1e-10, direction_improvement: 1e-3 },
  limits: "single seed and two repeated synthetic odor inputs; assay rewards are supplied externally; selected gains, naive plastic seam, calibrated MBON biases and softmax decoder are disclosed model choices; directed graph contrast is not covered by a reciprocal equilibrium-gradient theorem",
};
const loadPayload = () => JSON.parse(readFileSync(resolve(ROOT, "web/data/brain.json"), "utf8"));
const setup = () => JSON.parse(readFileSync(resolve(ROOT, "web/data/lessons.json"), "utf8")).setup;
const uniform = seed => () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296);
const probs = (state, outputs) => {
  const logits = outputs.map(i => state[i] / CONFIG.temperature), m = Math.max(...logits);
  const e = logits.map(v => Math.exp(v - m)), sum = e.reduce((a, b) => a + b, 0);
  return e.map(v => v / sum);
};

// A separate equation evaluation, not brain.equationResidual or a movement stopping proxy.
function residual(brain, v, beta = 0, choice = 0) {
  const rest = 1 / (1 + Math.exp(brain.slope * brain.threshold));
  const state = Float64Array.from(v, x => {
    const r = 1 / (1 + Math.exp(-brain.slope * (x - brain.threshold))) - rest;
    return r > 0 ? r / (1 - rest) : brain.leak ? r * brain.leak / rest : 0;
  });
  const outputs = OUTPUTS.map(name => brain.sets[name][0]), p = probs(state, outputs);
  let maximum = 0;
  for (let i = 0; i < brain.n; i++) {
    let sum = 0;
    for (let e = brain.rowPtr[i]; e < brain.rowPtr[i + 1]; e++) sum += brain.w[e] * state[brain.pre[e]];
    const j = outputs.indexOf(i);
    if (j >= 0) sum += beta * ((j === choice ? 1 : 0) - p[j]);
    maximum = Math.max(maximum, Math.abs(sum + (brain.drive[i] + brain.bias[i]) - v[i]));
  }
  return maximum;
}
function observe(brain, odor, lesion = false) {
  brain.reset(); brain.clearStimuli();
  const senses = { ...setup().decision_senses };
  for (let k = 0; k < 2; k++) for (const side of ["left", "right"]) senses[`orn:${ODORS[k]}:${side}`] = lesion ? 0 : k === odor ? PROTOCOL.observation.strongest : PROTOCOL.observation.other;
  // Every input must be a declared sensory population, distinct from all annotated motors.
  const motor = new Set();
  for (const [name, indices] of Object.entries(brain.sets)) if (name === "motor" || name === "mn9" || name.startsWith("mn:") || /^(power|steering|tension|neck_mn|leg_mn)(:|$)/.test(name)) for (const i of indices) motor.add(i);
  for (const [name, level] of Object.entries(senses)) {
    assert.match(name, /^(haltere:|ocelli:|lptc:|orn:)/);
    assert.ok(brain.sets[name]?.length && level >= 0 && level <= 1);
    assert.ok(brain.sets[name].every(i => !motor.has(i)), `sensory/motor overlap ${name}`);
    brain.stimulate(name, level);
  }
  return senses;
}
function probe(brain, label, odor, lesion = false) {
  const senses = observe(brain, odor, lesion), t0 = performance.now();
  const solve = brain.settleControl(PROTOCOL.solve.maxSteps, PROTOCOL.solve.tolerance);
  const independentResidual = residual(brain, brain.v);
  const outputs = OUTPUTS.map(name => brain.sets[name][0]);
  return { label, senses, solve, independent_residual: independentResidual, milliseconds: performance.now() - t0,
    state_sha256: hashArray(brain.s), potential_sha256: hashArray(brain.v),
    outputs: outputs.map(i => brain.s[i]), p: solve.converged ? probs(brain.s, outputs) : null,
    saturation: outputs.filter(i => brain.s[i] < 0.02 || brain.s[i] > 0.98).length / outputs.length,
    kc_mean: brain.mean("kc"), active_neurons: brain.activeCount(0.05) };
}
function probes(brain, emit) {
  return PROTOCOL.probes.map((label, index) => {
    const row = probe(brain, label, index % 2, index >= 2); emit?.({ stage: "probe", row }); return row;
  });
}
function permutePresynapticIdentities(brain) {
  const rnd = uniform(PROTOCOL.shuffled.seed), permutation = Int32Array.from({ length: brain.n }, (_, i) => i);
  for (let i = permutation.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [permutation[i], permutation[j]] = [permutation[j], permutation[i]]; }
  for (let e = 0; e < brain.edges; e++) brain.pre[e] = permutation[brain.pre[e]];
  return hashArray(permutation);
}
const deltaP = (a, b) => a[0].p && a[1].p && b[0].p && b[1].p ? Math.max(...[0, 1].map(i => Math.abs(a[i].p[0] - b[i].p[0]))) : null;
const score = rows => rows[0].p && rows[1].p ? (rows[0].p[0] + rows[1].p[1]) / 2 : null;
const allQualified = rows => rows.every(r => r.solve.converged && Number.isFinite(r.independent_residual) && r.independent_residual <= PROTOCOL.solve.tolerance);

async function trainArm(name, payload, draws, emit) {
  const brain = new SettlingBrain(payload), learner = new ActorCriticLearner(brain, { ...CONFIG, eta: name === "frozen_actor" ? 0 : CONFIG.eta });
  for (const output of OUTPUTS) assert.equal(brain.sets[output]?.length, 1, "one physical MBON per action readout");
  const originalWeights = Float64Array.from(brain.w), originalBias = Float64Array.from(brain.bias), initialEfficacy = Float64Array.from(learner.efficacy);
  const recordProbe = phase => row => emit({ ...row, arm: name, phase });
  const before = probes(brain, recordProbe("before")), trials = [];
  for (let trial = 0; trial < PROTOCOL.trials_per_arm; trial++) {
    const odor = trial % 2, senses = observe(brain, odor), weightBefore = hashArray(brain.w), t0 = performance.now();
    const decision = learner.actSettled({ ...PROTOCOL.solve, u: draws[trial] });
    const independent = {}, phaseHashes = {};
    if (decision.accepted) {
      independent.free = residual(brain, brain.v);
      independent.plus = residual(brain, learner.pending.plusV, CONFIG.beta, decision.choice);
      independent.minus = residual(brain, learner.pending.minusV, -CONFIG.beta, decision.choice);
      for (const [phase, v] of [["free", brain.v], ["plus", learner.pending.plusV], ["minus", learner.pending.minusV]]) phaseHashes[phase] = hashArray(v);
      if (!Object.values(independent).every(r => Number.isFinite(r) && r <= PROTOCOL.solve.tolerance)) {
        emit({ stage: "independent_residual_failure", arm: name, trial, decision, independent });
        throw new Error("independent phase residual failed; no reward update applied");
      }
    }
    const reward = decision.accepted ? (decision.choice === 0 ? odor === 0 ? 1 : -1 : 0) : null;
    // Every trial is terminal, so prior eligibility is zero. Independently reconstruct the
    // stipulated endpoint-product update in efficacy units, including reward and clipping.
    const expected = decision.accepted ? Float64Array.from(learner.edges, (edge, k) => {
      const pre = brain.pre[edge], post = learner.post[k], pending = learner.pending;
      const contrast = (pending.plus[pre] * pending.plus[post] - pending.minus[pre] * pending.minus[post]) / (2 * CONFIG.beta);
      const delta = Math.max(-CONFIG.dopamineCap, Math.min(CONFIG.dopamineCap, reward - decision.value));
      return Math.max(-CONFIG.cap, Math.min(CONFIG.cap, learner.efficacy[k] + learner.cfg.eta * delta * contrast));
    }) : null;
    const update = decision.accepted ? learner.learnSettled(reward, true, PROTOCOL.solve) : null;
    let updateError = null;
    if (update?.accepted) {
      updateError = 0;
      for (let k = 0; k < expected.length; k++) updateError = Math.max(updateError, Math.abs(expected[k] - learner.efficacy[k]));
    }
    const row = { trial, odor: ODORS[odor], senses, uniform: draws[trial], decision, independent_residuals: independent,
      phase_potential_sha256: phaseHashes, reward, update, independent_update_max_error: updateError,
      weight_before_sha256: weightBefore, weight_after_sha256: hashArray(brain.w), milliseconds: performance.now() - t0 };
    if (!decision.accepted) assert.equal(row.weight_before_sha256, row.weight_after_sha256, "rejected phase changed weights");
    trials.push(row); emit({ stage: "trial", arm: name, row });
    if (update?.accepted) assert.ok(updateError <= PROTOCOL.thresholds.independent_update_error, "independent local update reconstruction failed");
    if ((trial + 1) % 6 === 0) console.log(`${name}: ${trial + 1}/${PROTOCOL.trials_per_arm}, accepted ${trials.filter(r => r.update?.accepted).length}`);
  }
  const learnedWeights = Float64Array.from(brain.w), trainedEfficacy = Float64Array.from(learner.efficacy);
  assert.deepEqual(brain.bias, originalBias, "no readout bias calibration during training");
  const after = probes(brain, recordProbe("after"));
  brain.w.set(originalWeights); learner.efficacy.set(initialEfficacy);
  const restored = probes(brain, recordProbe("restored"));
  brain.w.set(learnedWeights); learner.efficacy.set(trainedEfficacy);
  const reinstated = probes(brain, recordProbe("reinstated"));
  let changed = 0, maxChange = 0;
  for (let e = 0; e < brain.edges; e++) { const d = Math.abs(learnedWeights[e] - originalWeights[e]); if (d > 1e-12) changed++; maxChange = Math.max(maxChange, d); }
  return { name, plastic_edges: learner.edges.length, before, trials, after, restored, reinstated,
    weight_change: { changed, max_absolute: maxChange, initial_sha256: hashArray(originalWeights), learned_sha256: hashArray(learnedWeights) },
    checkpoint: { edges: Array.from(learner.edges), efficacy: Array.from(trainedEfficacy), weights: Array.from(learner.edges, e => learnedWeights[e]), learned_weight_sha256: hashArray(learnedWeights) },
    metrics: { before_score: score(before), after_score: score(after), wrong_odor_score: score(after) === null ? null : 1 - score(after),
      maximum_probability_change: deltaP(before, after), restore_error: deltaP(before, restored), reinstate_error: deltaP(after, reinstated) } };
}
function evaluateClaims(arms, shuffled) {
  const a = arms.plastic, f = arms.frozen_actor, t = PROTOCOL.thresholds;
  const odor = a.before[0].p && a.before[1].p ? Math.abs(a.before[0].p[0] - a.before[1].p[0]) : null;
  const lessons = a.trials.filter(r => r.update?.accepted).length;
  return {
    input_changes_settled_output: allQualified(a.before) && odor > t.causal_probability_change,
    odor_lesion_removes_label_information: a.after[2].solve.converged && a.after[3].solve.converged && a.after[2].state_sha256 === a.after[3].state_sha256,
    all_training_phases_qualified: Object.values(arms).every(a => a.trials.every(r => r.decision.accepted && r.update?.accepted)),
    accepted_local_updates_independently_reproduced: Object.values(arms).every(a => a.trials.filter(r => r.update?.accepted).every(r => Number.isFinite(r.independent_update_max_error) && r.independent_update_max_error <= t.independent_update_error)),
    accepted_plastic_lessons: lessons,
    local_weight_memory_changes_output: lessons > 0 && allQualified(a.after) && a.weight_change.max_absolute > t.causal_weight_change && a.metrics.maximum_probability_change > t.causal_probability_change,
    frozen_actor_unchanged: allQualified(f.before) && allQualified(f.after) && f.weight_change.initial_sha256 === f.weight_change.learned_sha256 && f.metrics.maximum_probability_change <= t.restoration_probability_error,
    initial_weight_restoration_removes_effect: allQualified(a.before) && allQualified(a.restored) && a.metrics.restore_error <= t.restoration_probability_error,
    learned_weight_reinstatement_recovers_effect: allQualified(a.after) && allQualified(a.reinstated) && a.metrics.reinstate_error <= t.restoration_probability_error,
    rewarded_discrimination_improved: allQualified(a.before) && allQualified(a.after) && a.metrics.after_score - a.metrics.before_score > t.direction_improvement,
    learned_greedy_discrimination: !!(allQualified(a.after) && a.after[0].p[0] > 0.5 && a.after[1].p[0] < 0.5),
    shuffled_all_free_probes_qualified: allQualified(shuffled.probes),
    scope: PROTOCOL.scope,
  };
}
function writeExclusive(path, obj) { mkdirSync(dirname(path), { recursive: true }); writeFileSync(path, JSON.stringify(obj, null, 2) + "\n", { flag: "wx" }); }
async function run(out) {
  if (existsSync(out) || existsSync(out + ".journal.jsonl")) throw new Error("refusing to overwrite existing experiment evidence");
  const sources = sourceHashes(), started = new Date().toISOString(), payload = loadPayload();
  const emit = row => appendFileSync(out + ".journal.jsonl", JSON.stringify(row) + "\n");
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out + ".journal.jsonl", JSON.stringify({ stage: "frozen_protocol", protocol: PROTOCOL, sources, started }) + "\n", { flag: "wx" });
  const random = uniform(PROTOCOL.seed), draws = Array.from({ length: PROTOCOL.trials_per_arm }, random), arms = {};
  try {
    for (const name of PROTOCOL.arms) arms[name] = await trainArm(name, payload, draws, emit);
    const shuffledBrain = new SettlingBrain(payload), permutation = permutePresynapticIdentities(shuffledBrain);
    const shuffled = { intervention: PROTOCOL.shuffled, permutation_sha256: permutation, probes: probes(shuffledBrain, row => emit({ ...row, arm: "shuffled", phase: "inference_only" })) };
    assert.deepEqual(sourceHashes(), sources, "source or payload changed during the frozen run");
    const body = { schema: "cadence.fly-matrix.assisted-evidence/budget-followup-1024-v1", started, completed: new Date().toISOString(), protocol: PROTOCOL, sources,
      runtime: { node: process.version, platform: process.platform, architecture: process.arch }, payload: { neurons: payload.n, edges: payload.edges, model: payload.model },
      draws, arms, shuffled, claims: evaluateClaims(arms, shuffled), journal_sha256: hash(readFileSync(out + ".journal.jsonl")) };
    writeExclusive(out, { ...body, sha256: hash(JSON.stringify(body)) });
    console.log(JSON.stringify({ receipt: relative(ROOT, out), claims: body.claims, scores: Object.fromEntries(Object.entries(arms).map(([k, v]) => [k, v.metrics])) }, null, 2));
  } catch (error) {
    emit({ stage: "fatal_error", error: String(error), completed: new Date().toISOString() });
    throw error; // The frozen protocol and all completed trials remain in the journal.
  }
}
async function verify(path) {
  const receipt = JSON.parse(readFileSync(path, "utf8")), { sha256, ...body } = receipt;
  assert.equal(hash(JSON.stringify(body)), sha256, "receipt body hash");
  assert.deepEqual(body.protocol, PROTOCOL, "frozen protocol"); assert.deepEqual(body.sources, sourceHashes(), "source binding");
  assert.equal(hash(readFileSync(path + ".journal.jsonl")), body.journal_sha256, "complete journal binding");
  assert.deepEqual(body.claims, evaluateClaims(body.arms, body.shuffled), "derived claims");
  const payload = loadPayload();
  for (const name of PROTOCOL.arms) {
    const arm = body.arms[name], brain = new SettlingBrain(payload), learner = new ActorCriticLearner(brain, CONFIG);
    assert.equal(arm.trials.length, PROTOCOL.trials_per_arm);
    for (const row of arm.trials) {
      if (row.decision.accepted) {
        assert.deepEqual(Object.keys(row.decision.solves).sort(), ["free", "minus", "plus"]);
        assert.ok(Object.values(row.decision.solves).every(s => s.converged && s.residual <= PROTOCOL.solve.tolerance));
        assert.deepEqual(Object.keys(row.independent_residuals).sort(), ["free", "minus", "plus"]);
        assert.ok(Object.values(row.independent_residuals).every(r => Number.isFinite(r) && r >= 0 && r <= PROTOCOL.solve.tolerance));
        if (row.update?.accepted) assert.ok(Number.isFinite(row.independent_update_max_error) && row.independent_update_max_error <= PROTOCOL.thresholds.independent_update_error);
      } else assert.equal(row.weight_before_sha256, row.weight_after_sha256);
    }
    assert.deepEqual(Array.from(learner.edges), arm.checkpoint.edges, "declared plastic seam");
    assert.equal(arm.checkpoint.weights.length, learner.edges.length);
    assert.equal(arm.checkpoint.efficacy.length, learner.edges.length);
    for (let k = 0; k < learner.edges.length; k++) {
      const edge = learner.edges[k], expected = brain.gainPre[edge] * arm.checkpoint.efficacy[k];
      assert.ok(Math.abs(expected - arm.checkpoint.weights[k]) <= 1e-14 * Math.max(1, Math.abs(expected)), "efficacy/weight consistency");
      brain.w[edge] = arm.checkpoint.weights[k];
    }
    assert.equal(hashArray(brain.w), arm.checkpoint.learned_weight_sha256, "saved actor checkpoint reconstruction");
    const current = probes(brain);
    for (let i = 0; i < current.length; i++) {
      assert.deepEqual(current[i].p, arm.after[i].p, `${name} probability replay`);
      assert.equal(current[i].state_sha256, arm.after[i].state_sha256, `${name} state replay`);
      assert.equal(current[i].independent_residual, arm.after[i].independent_residual, `${name} independent residual replay`);
    }
  }
  console.log("assisted evidence: protocol/source/journal integrity, all recorded accepted-phase gates, derived claims and both actor checkpoint probes verified; training was not replayed");
}
const args = process.argv.slice(2);
if (args.includes("--help") || args.length === 0) console.log("Usage: node tools/assisted_evidence_1024.mjs --run [--out receipts/assisted_evidence_1024.json] | --verify RECEIPT\nFrozen bounded diagnostic; output and journal must be new paths.");
else if (args[0] === "--verify" && args[1]) await verify(resolve(ROOT, args[1]));
else if (args[0] === "--run") {
  const index = args.indexOf("--out"), out = index < 0 ? "receipts/assisted_evidence_1024.json" : args[index + 1];
  if (!out) throw new Error("--out requires a path"); await run(resolve(ROOT, out));
} else throw new Error("invalid arguments; use --help");
