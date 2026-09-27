// Actual worker transport: pixels -> retained sensory cells -> recurrent state,
// candidate neural values -> sampled target, then qualified local learning.
// node tests/neural-worker.mjs [--payload] [--receipt NEW.json]
// The optional full-payload protocol is fixed here, including capped failures;
// it is a causal wiring assay, not learned navigation or biological vision.
import assert from "node:assert/strict";
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { createHash } from "node:crypto";
import { SettlingBrain } from "../web/brain.js";
import { ActorCriticLearner } from "../web/learner.js";
import { SETTLED_MOTOR_GROUPS } from "../web/motor.js";
import { createRetina } from "../web/retina.js";

const root = new URL("../", import.meta.url), paths = ["tests/neural-worker.mjs", "web/worker.js", "web/brain.js",
  "web/learner.js", "web/retina.js", "web/neural-policy.js", "web/motor.js", "web/body.js",
  "web/assisted-life.js", "web/life.js", "web/senses.js", "web/data/brain_full.json"];
const hash = bytes => createHash("sha256").update(bytes).digest("hex");
const sourceHashes = () => Object.fromEntries(paths.map(path => [path, hash(readFileSync(new URL(path, root)))]));
const frozen = sourceHashes(), cases = [], started = new Date().toISOString();
const receiptArg = process.argv.indexOf("--receipt"), receiptPath = receiptArg >= 0 ? process.argv[receiptArg + 1] : null;
if (receiptArg >= 0) assert.ok(receiptPath && !receiptPath.startsWith("--") && !existsSync(receiptPath), "receipt must be a new path");
const b64 = array => Buffer.from(array.buffer, array.byteOffset, array.byteLength).toString("base64");
const fixture = () => ({ n: 7, edges: 8, synapses: 8, whole: { neurons: 7 },
  model: { dt: .5, slope: 1, threshold: 0, leak: 1, gain: 1, stimulus_amplitude: 1, adaptation: null },
  arrays: { row_ptr: b64(new Int32Array([0, 0, 0, 0, 0, 2, 4, 8])),
    pre: b64(new Int32Array([5, 6, 4, 6, 0, 1, 2, 3])),
    weight: b64(new Float64Array([.05, .6, .05, -.4, .3, -.2, .9, -.9])),
    sign: b64(new Float64Array([.05, .6, .05, -.4, .3, -.2, .9, -.9])), count: b64(new Uint16Array(8).fill(1)) },
  populations: { photoreceptor: [0, 1], kc: [6], mbon: [4, 5], motor: [4, 5],
    "haltere:left": [2], "orn:decaying_fruit:left": [2], "orn:decaying_fruit:right": [2],
    "orn:yeasty:left": [3], "orn:yeasty:right": [3],
    "mbon:MBON11:right": [4], "mbon:MBON05:left": [5],
    ...Object.fromEntries(SETTLED_MOTOR_GROUPS.map((name, i) => [name, [4 + i % 2]])) } });
const config = { outputs: ["mbon:MBON11:right", "mbon:MBON05:left"], actions: [0, 1],
  plastic: { pre: ["kc"], post: ["mbon"] }, critic: "kc", beta: .1, temperature: .3,
  eta: .1, etaBias: 0, etaCritic: .05, gamma: .95, lam: .9, cap: 3, dopamineCap: 1 };
const original = { self: globalThis.self, fetch: globalThis.fetch,
  stats: ActorCriticLearner.prototype.stats, control: SettlingBrain.prototype.settleControl };
let source = fixture(), liveBrain = null, liveLearner = null, id = 0;
const replies = [];
globalThis.self = { postMessage: message => replies.push(message) };
globalThis.fetch = async () => ({ json: async () => source });
ActorCriticLearner.prototype.stats = function (...args) { liveLearner = this; liveBrain = this.brain; return original.stats.apply(this, args); };
SettlingBrain.prototype.settleControl = function (...args) { liveBrain = this; return original.control.apply(this, args); };
const send = async (message, count = 1) => {
  const before = replies.length;
  await self.onmessage({ data: message });
  assert.equal(replies.length - before, count, `${message.type}: reply count`);
  return replies.at(-1);
};
const initialize = async (payload = fixture(), seed = 1) => {
  source = payload;
  const ready = await send({ type: "init", url: "frozen test payload", generation: 7 });
  assert.equal(ready.type, "ready", JSON.stringify(ready));
  const learned = await send({ type: "learn:init", config, seed });
  assert.equal(learned.type, "learn:ready");
};
function frame(pattern = "black") {
  const width = 64, height = 32, rgba = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const k = 4 * (y * width + x);
    const light = pattern === "right" ? x >= width / 2 : pattern === "left" ? x < width / 2 : false;
    rgba[k] = rgba[k + 1] = rgba[k + 2] = light ? 96 : 0; rgba[k + 3] = 255;
  }
  return { width, height, rgba, origin: "bottom-left" };
}
const bothOdors = { "orn:decaying_fruit:left": .9, "orn:decaying_fruit:right": .9,
  "orn:yeasty:left": .9, "orn:yeasty:right": .9 };
const observe = extra => send({ type: "assist:observe", requestId: ++id, generation: 7,
  senses: {}, retina: frame(), steps: 1024, tolerance: 1e-6, ...extra });
const propose = extra => send({ type: "assist:decide", requestId: ++id, generation: 7, searchToken: `7:${id}`,
  senses: bothOdors, retina: frame(), steps: 1024, tolerance: 1e-6, selectTarget: true, ...extra });
const identity = m => Object.fromEntries(["requestId", "generation", "searchToken"].map(key => [key, m[key]]));
const arraysHash = arrays => hash(Buffer.concat(arrays.map(a => Buffer.from(a.buffer, a.byteOffset, a.byteLength))));
const snapshot = () => ({ brain: arraysHash([liveBrain.v, liveBrain.s, liveBrain.drive, liveBrain.w, liveBrain.bias]),
  learner: arraysHash([liveLearner.efficacy, liveLearner.trace, liveLearner.traceBias, liveLearner.traceCritic, liveLearner.wCritic]),
  steps: liveBrain.steps, updates: liveLearner.updates, bCritic: liveLearner.bCritic });
const qualified = solve => {
  assert.equal(solve.converged, true, JSON.stringify(solve));
  assert.ok(Number.isFinite(solve.residual) && solve.residual >= 0 && solve.residual <= 1e-6);
  assert.equal(solve.tolerance, 1e-6);
  assert.ok(Number.isSafeInteger(solve.iterations) && solve.iterations >= 0 && solve.iterations <= 1024);
};
// This check reconstructs activation and the sparse equation without calling the
// production activation/equationResidual or any production residual helper.
function independentResidual(b, potential = b.v, choice = null, beta = 0) {
  const s = new Float64Array(b.n), rest = 1 / (1 + Math.exp(b.slope * b.threshold));
  for (let i = 0; i < b.n; i++) {
    const raw = 1 / (1 + Math.exp(-b.slope * (potential[i] - b.threshold))) - rest;
    s[i] = raw > 0 ? raw / (1 - rest) : b.leak * raw / rest;
  }
  const pushes = new Map();
  if (choice !== null) {
    const outputs = liveLearner.outputs, logits = outputs.map(i => s[i] / liveLearner.cfg.temperature);
    const peak = Math.max(...logits), weights = logits.map(x => Math.exp(x - peak)), total = weights.reduce((a, x) => a + x, 0);
    outputs.forEach((i, k) => pushes.set(i, beta * ((k === choice ? 1 : 0) - weights[k] / total)));
  }
  let residual = 0;
  for (let i = 0; i < b.n; i++) {
    let input = 0;
    for (let e = b.rowPtr[i]; e < b.rowPtr[i + 1]; e++) input += b.w[e] * s[b.pre[e]];
    residual = Math.max(residual, Math.abs(input + b.drive[i] + b.bias[i] + (pushes.get(i) ?? 0) - potential[i]));
  }
  return residual;
}
function phases(proposal) {
  assert.equal(proposal.accepted, true, JSON.stringify(proposal));
  assert.deepEqual(Object.keys(proposal.decision.solves).sort(), ["free", "minus", "plus"]);
  Object.values(proposal.decision.solves).forEach(qualified);
  const pending = liveLearner.pending;
  const checked = { free: independentResidual(liveBrain),
    plus: independentResidual(liveBrain, pending.plusV, pending.choice, liveLearner.cfg.beta),
    minus: independentResidual(liveBrain, pending.minusV, pending.choice, -liveLearner.cfg.beta) };
  for (const [phase, residual] of Object.entries(checked)) {
    assert.ok(residual <= 1e-6 + 1e-12, `${phase}: independent residual ${residual}`);
    assert.ok(Math.abs(residual - proposal.decision.solves[phase].residual) <= 1e-10);
  }
  return checked;
}
function selection(proposal) {
  const target = proposal.targetSelection;
  assert.equal(target.accepted, true);
  assert.equal(proposal.decision.solves.free.iterations, 0, 'selected target equilibrium is rechecked without repeating its repairs');
  assert.ok(!('state' in target) && !('states' in target), 'internal candidate arrays never enter the reply');
  assert.deepEqual(target.candidates.map(c => c.fruit), ["banana", "bread"]);
  for (const c of target.candidates) {
    qualified(c.solve);
    assert.equal(c.score, c.readouts[config.outputs[0]] - c.readouts[config.outputs[1]]);
  }
  const chosen = target.candidates.find(c => c.fruit === target.fruit);
  assert.equal(proposal.decision.solves.free.residual, chosen.solve.residual);
  for (const name of config.outputs) assert.equal(proposal.readouts[name], chosen.readouts[name]);
  const peak = Math.max(...target.candidates.map(c => c.score / target.temperature));
  const exps = target.candidates.map(c => Math.exp(c.score / target.temperature - peak));
  const expected = exps.map(x => x / (exps[0] + exps[1]));
  expected.forEach((p, k) => assert.ok(Math.abs(p - target.p[k]) < 1e-12));
  assert.ok(target.draw >= 0 && target.draw < 1);
  assert.equal(target.fruit, target.draw <= target.p[0] ? "banana" : "bread");
  return target;
}
const test = async (name, fn) => {
  const result = { name, passed: false };
  cases.push(result);
  try { result.evidence = await fn(); result.passed = true; console.log(`ok ${name}`); }
  catch (error) { result.error = String(error); throw error; }
};
let failure = null;
try {
  await import("../web/worker.js");
  await initialize();
  await test("retinal pixels drive only worker-owned photoreceptors", async () => {
    const packet = frame("right"), result = await observe({ retina: packet }); qualified(result);
    const encoded = createRetina({ n: 7, photoreceptors: [0, 1] }).encode(packet);
    for (let i = 0; i < 7; i++) assert.equal(liveBrain.drive[i], i < 2 ? encoded.levels[i] : 0);
    assert.equal(result.retinal.neurons, 2); assert.equal(result.retinal.width, 64);
    assert.ok(independentResidual(liveBrain) <= 1e-6);
    return { retinal: result.retinal, residual: result.residual };
  });
  await test("malformed retinal packets cannot mutate state, weights or traces", async () => {
    for (const type of ["assist:observe", "assist:decide"]) for (const mutate of [
      f => { f.rgba = f.rgba.slice(1); }, f => { f.rgba = new Float32Array(f.rgba); },
      f => { f.width = 0; }, f => { f.height = Infinity; }, f => { f.origin = "unknown"; },
      f => { f.indices = [4]; f.levels = [1]; },
    ]) {
      const retinal = frame(); mutate(retinal); const before = snapshot();
      const bad = await send({ type, requestId: ++id, generation: 7, searchToken: `7:${id}`,
        senses: bothOdors, retina: retinal, selectTarget: true, steps: 1024, tolerance: 1e-6 });
      assert.equal(bad.converged, false); assert.equal(bad.accepted, false);
      assert.ok(!bad.readouts && !bad.s && !bad.decision);
      assert.deepEqual(snapshot(), before);
    }
  });
  await test("retinal membership overlapping a motor cell fails closed", async () => {
    const corrupt = fixture(); corrupt.populations.photoreceptor = [0, 4]; source = corrupt;
    const ready = await send({ type: "init", url: "overlap fixture" });
    if (ready.type === "ready") {
      await send({ type: "learn:init", config, seed: 1 });
      const before = snapshot(), rejected = await observe();
      assert.equal(rejected.converged, false); assert.deepEqual(snapshot(), before);
    } else assert.match(ready.reason, /retin|motor|overlap/);
    await initialize();
  });
  await test("attention masks only the other odor and preserves retinal drive", async () => {
    for (const fruit of ["banana", "bread"]) {
      const observed = await observe({ senses: bothOdors, attention: fruit, searchToken: "attention-check", retina: frame("right") });
      qualified(observed); assert.equal(observed.attention, fruit); assert.equal(observed.searchToken, "attention-check");
      assert.equal(liveBrain.drive[fruit === "banana" ? 2 : 3], .9);
      assert.equal(liveBrain.drive[fruit === "banana" ? 3 : 2], 0);
      assert.ok(liveBrain.drive[1] > 0);
    }
  });
  await test("candidate values and sampled fruit change under a neural-weight intervention", async () => {
    await initialize(fixture(), 1); const first = await propose(); phases(first); const a = selection(first);
    const swapped = fixture(), weights = new Float64Array([.05, .6, .05, -.4, .3, -.2, -.9, .9]);
    swapped.arrays.weight = b64(weights); swapped.arrays.sign = b64(weights);
    await initialize(swapped, 1); const second = await propose(); phases(second); const b = selection(second);
    assert.equal(a.draw, b.draw, "same seed must use the same target draw");
    assert.ok(a.p[0] > .5 && b.p[0] < .5);
    assert.ok(Math.abs(a.p[0] - b.p[0]) > .05);
    assert.notEqual(a.fruit, b.fruit, "fixed seed must expose the actual target intervention");
    return { original: a, altered_odor_path_weights: b };
  });
  await test("candidate probes share one starting state and cannot supply unqualified choices", async () => {
    await initialize(); const before = snapshot();
    const denied = await propose({ steps: 1 });
    assert.equal(denied.accepted, false); assert.equal(denied.reason, "target_phase_failed");
    assert.equal(liveLearner.pending, null); assert.ok(!denied.readouts && !denied.s);
    assert.equal(snapshot().learner, before.learner);
    assert.equal(liveBrain.steps, before.steps); assert.ok(liveBrain.v.every(x => x === 0));
  });
  await test("a failed action nudge restores the state before candidate reuse and discards eligibility", async () => {
    await initialize();
    const beforeV = liveBrain.v.slice(), beforeS = liveBrain.s.slice(), beforeSteps = liveBrain.steps;
    const priorLearner = snapshot().learner;
    const nudge = liveBrain.settleNudgedResidual;
    liveBrain.settleNudgedResidual = () => ({ converged: false, iterations: 1, residual: .1, tolerance: 1e-6, reason: 'test_failure' });
    try {
      const denied = await propose();
      assert.equal(denied.targetSelection.accepted, true);
      assert.equal(denied.accepted, false); assert.equal(denied.reason, 'plus_phase_failed');
      assert.equal(denied.decision.solves.free.iterations, 0);
      assert.deepEqual(liveBrain.v, beforeV); assert.deepEqual(liveBrain.s, beforeS); assert.equal(liveBrain.steps, beforeSteps);
      assert.equal(snapshot().learner, priorLearner); assert.equal(liveLearner.pending, null);
      assert.ok(!denied.s && !denied.readouts);
    } finally { liveBrain.settleNudgedResidual = nudge; }
  });
  if (process.argv.includes("--payload")) {
    const payload = JSON.parse(readFileSync(new URL("web/data/brain_full.json", root), "utf8"));
    await initialize(payload);
    await test("full retained graph carries pixel effects beyond receptors; afferent lesion removes and restoration recovers them", async () => {
      assert.equal(liveBrain.n, 150802); assert.equal(liveBrain.edges, 1877099);
      const recipients = new Uint8Array(liveBrain.n);
      for (const i of liveBrain.sets.photoreceptor) recipients[i] = 1;
      const savedWeights = liveBrain.w.slice(), summaries = [];
      const solveFrame = async (name, pixels) => {
        await send({ type: "reset", generation: 7 });
        const reply = await observe({ retina: pixels });
        const independent = independentResidual(liveBrain);
        summaries.push({ name, converged: reply.converged, iterations: reply.iterations,
          residual: reply.residual, independent, retinal: reply.retinal,
          navigation_readouts: Object.fromEntries(Object.entries(reply.readouts ?? {}).filter(([name]) =>
            ["dn:DNa02:left", "dn:DNa02:right", "dn:DNp09", "dn:DNp03", "dn:landing", "mn9"].includes(name))) });
        qualified(reply); assert.ok(independent <= 1e-6 + 1e-12);
        assert.equal(reply.retinal.neurons, 1831);
        for (let i = 0; i < liveBrain.n; i++) if (!recipients[i]) assert.equal(liveBrain.drive[i], 0);
        return liveBrain.s.slice();
      };
      const difference = (a, b) => {
        let max = 0, changed = 0;
        for (let i = 0; i < a.length; i++) if (!recipients[i]) {
          const d = Math.abs(a[i] - b[i]); max = Math.max(max, d); if (d > 1e-8) changed++;
        }
        return { maximum: max, changed_over_1e_minus_8: changed };
      };
      try {
        const dark = await solveFrame("original_black", frame()), light = await solveFrame("original_right_gray96", frame("right"));
        const intact = difference(dark, light); assert.ok(intact.maximum > 1e-5 && intact.changed_over_1e_minus_8 > 0);
        let lesioned = 0;
        for (let e = 0; e < liveBrain.edges; e++) if (recipients[liveBrain.pre[e]]) { liveBrain.w[e] = 0; lesioned++; }
        const cutDark = await solveFrame("afferents_cut_black", frame()), cutLight = await solveFrame("afferents_cut_right_gray96", frame("right"));
        const lesion = difference(cutDark, cutLight);
        assert.ok(lesion.maximum <= 2e-6, `lesioned downstream difference ${lesion.maximum}`);
        assert.ok(lesion.maximum < intact.maximum / 100);
        liveBrain.w.set(savedWeights);
        const restored = await solveFrame("restored_right_gray96", frame("right"));
        assert.deepEqual(restored, light);
        return { nodes: liveBrain.n, edges: liveBrain.edges, recipient_count: 1831, lesioned_edge_classes: lesioned,
          intact, lesion, restored_exact: true, solves: summaries };
      } finally { liveBrain.w.set(savedWeights); cases.at(-1).solves = summaries; }
    });
    await test("full graph chooses from checked odor candidates and learns only its admitted outcome", async () => {
      await send({ type: "reset", generation: 7 });
      const senses = { ...payload.lessons_setup.decision_senses, ...bothOdors };
      const choice = await propose({ senses, retina: frame() });
      // Record the actual capped result before asserting; a failed phase is not
      // permission to fabricate a decision or silently raise this protocol's cap.
      cases.at(-1).proposal = { accepted: choice.accepted, reason: choice.reason,
        targetSelection: choice.targetSelection, decision: choice.decision, readouts: choice.readouts };
      const residuals = phases(choice), target = selection(choice);
      const before = hash(Buffer.from(liveBrain.w.buffer)), updates = liveLearner.updates;
      const unaccepted = await send({ type: "assist:reward", ...identity(choice), reward: 1, done: true, neuralCredit: true });
      assert.equal(unaccepted.accepted, false); assert.equal(hash(Buffer.from(liveBrain.w.buffer)), before);
      await send({ type: "assist:accept", ...identity(choice) }, 0);
      const result = await send({ type: "assist:reward", ...identity(choice), reward: 1, done: true, neuralCredit: true });
      assert.equal(result.accepted, true); assert.ok(result.moved > 0); assert.equal(liveLearner.updates, updates + 1);
      assert.notEqual(hash(Buffer.from(liveBrain.w.buffer)), before);
      return { targetSelection: target, phase_residuals: residuals, changed_edge_classes: result.moved,
        scope: "one synthetic terminal reward through actual worker; no flight, discrimination or general-learning claim" };
    });
  }
} catch (error) { failure = error; console.error(error.stack); }
finally {
  globalThis.self = original.self; globalThis.fetch = original.fetch;
  ActorCriticLearner.prototype.stats = original.stats; SettlingBrain.prototype.settleControl = original.control;
  try { assert.deepEqual(sourceHashes(), frozen, "source changed during test"); }
  catch (error) { failure ??= error; cases.push({ name: "source custody", passed: false, error: String(error) }); }
  const receipt = { schema: "cadence.neural-worker-test/2", started, completed: new Date().toISOString(),
    passed: !failure, full_payload: process.argv.includes("--payload"), sources: frozen, node: process.version,
    protocol: { max_steps: 1024, tolerance: 1e-6, synthetic_image: "64x32 bottom-left RGBA; black versus right-half RGB96, alpha255",
      target_seed: 1, full_learning_reward: 1, learner_config: config,
      revision: "After the supplied hover-thrust projection was added to the navigation decoder; original experiment and producer retained separately. Neural solve and learning parameters unchanged.",
      limitations: "Causal input/decoder/learning wiring, not biological vision or autonomous navigation. eta=.1 in this worker unit assay; the page uses eta=1." }, cases };
  receipt.sha256 = hash(JSON.stringify(receipt));
  if (receiptPath) writeFileSync(receiptPath, JSON.stringify(receipt, null, 2) + "\n", { flag: "wx" });
  console.log(`${cases.filter(c => c.passed).length}/${cases.length} neural worker tests passed`);
}
if (failure) process.exitCode = 1;
