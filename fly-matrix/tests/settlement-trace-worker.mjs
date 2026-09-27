// Independent arithmetic and custody checks for actual settlement telemetry.
// node tests/settlement-trace-worker.mjs [--payload] [--receipt NEW.json]
// Full-state witnesses remain bounded in RAM; receipts contain scalar checks,
// hashes and bounded trace metadata, never a full state history.
import assert from "node:assert/strict";
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { createHash } from "node:crypto";
import { SettlingBrain } from "../web/brain.js";
import { ActorCriticLearner } from "../web/learner.js";
import { SETTLED_MOTOR_GROUPS } from "../web/motor.js";

const root = new URL("../", import.meta.url), paths = ["tests/settlement-trace-worker.mjs", "web/worker.js",
  "web/brain.js", "web/settlement-trace.js", "web/learner.js", "web/retina.js", "web/neural-policy.js",
  "web/motor.js", "web/body.js", "web/assisted-life.js", "web/life.js", "web/senses.js", "web/data/brain_full.json"];
const hash = bytes => createHash("sha256").update(bytes).digest("hex");
const sourceHashes = () => Object.fromEntries(paths.map(path => [path, hash(readFileSync(new URL(path, root)))]));
const frozen = sourceHashes(), cases = [], started = new Date().toISOString();
const at = process.argv.indexOf("--receipt"), receiptPath = at < 0 ? null : process.argv[at + 1];
if (at >= 0) assert.ok(receiptPath && !receiptPath.startsWith("--") && !existsSync(receiptPath), "receipt must be a new path");
const b64 = a => Buffer.from(a.buffer, a.byteOffset, a.byteLength).toString("base64");
const fixture = () => ({ n: 7, edges: 8, synapses: 8, whole: { neurons: 7 },
  model: { dt: .5, slope: 1, threshold: 0, leak: 1, gain: 1, stimulus_amplitude: 1, adaptation: null },
  arrays: { row_ptr: b64(new Int32Array([0, 0, 0, 0, 0, 2, 4, 8])),
    pre: b64(new Int32Array([5, 6, 4, 6, 0, 1, 2, 3])),
    weight: b64(new Float64Array([.05, .6, .05, -.4, .3, -.2, .9, -.9])),
    sign: b64(new Float64Array([.05, .6, .05, -.4, .3, -.2, .9, -.9])), count: b64(new Uint16Array(8).fill(1)) },
  populations: { photoreceptor: [0, 1], kc: [6], mbon: [4, 5], motor: [4, 5],
    "orn:decaying_fruit:left": [2], "orn:decaying_fruit:right": [2], "orn:yeasty:left": [3], "orn:yeasty:right": [3],
    "mbon:MBON11:right": [4], "mbon:MBON05:left": [5],
    ...Object.fromEntries(SETTLED_MOTOR_GROUPS.map((name, k) => [name, [4 + k % 2]])) } });
const config = { outputs: ["mbon:MBON11:right", "mbon:MBON05:left"], actions: [0, 1],
  plastic: { pre: ["kc"], post: ["mbon"] }, critic: "kc", beta: .1, temperature: .3,
  eta: .1, etaBias: 0, etaCritic: .05, gamma: .95, lam: .9, cap: 3, dopamineCap: 1 };
const bothOdors = { "orn:decaying_fruit:left": .9, "orn:decaying_fruit:right": .9,
  "orn:yeasty:left": .9, "orn:yeasty:right": .9 };
const frame = () => {
  const width = 64, height = 32, rgba = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const k = 4 * (y * width + x); rgba.fill(x >= width / 2 ? 96 : 0, k, k + 3); rgba[k + 3] = 255;
  }
  return { width, height, rgba, origin: "bottom-left" };
};
const original = { self: globalThis.self, fetch: globalThis.fetch, stats: ActorCriticLearner.prototype.stats,
  control: SettlingBrain.prototype.settleControl, residual: SettlingBrain.prototype._equationResidualFromActivation };
let payload = fixture(), brain = null, learner = null, witness = null, diagnosticRuns = null;
const replies = [];
globalThis.self = { postMessage: m => replies.push(m) };
globalThis.fetch = async () => ({ json: async () => payload });
ActorCriticLearner.prototype.stats = function (...args) { learner = this; brain = this.brain; return original.stats.apply(this, args); };
SettlingBrain.prototype.settleControl = function (...args) {
  brain = this;
  const before = diagnosticRuns ? { v: arrayHash(this.v), s: arrayHash(this.s), w: arrayHash(this.w),
    bias: arrayHash(this.bias), drive: this.drive.slice() } : null;
  const result = original.control.apply(this, args);
  if (diagnosticRuns) diagnosticRuns.push({ before, activity: this.s.slice(), result: { ...result, trace: undefined } });
  return result;
};
SettlingBrain.prototype._equationResidualFromActivation = function (...args) {
  const residual = original.residual.apply(this, args);
  if (witness) {
    const record = { iteration: witness.iteration++, v: this.v.slice(), previous: witness.previous ?? null };
    // Mirror only bounded retention, not any numerical calculation. Every kept
    // potential and preceding state comes directly from the live solve; all
    // residual/update arithmetic is reconstructed independently after the run.
    if (record.iteration === 0 || record.iteration % witness.stride === 0) {
      if (witness.saved.size === 8) {
        witness.saved = new Map([...witness.saved].filter((_, k) => k % 2 === 0)); witness.stride *= 2;
      }
      witness.saved.set(record.iteration, record);
    }
    witness.last = record; witness.previous = record.v;
  }
  return residual;
};
async function send(message, count = 1) {
  const before = replies.length; await self.onmessage({ data: message });
  assert.equal(replies.length - before, count, `${message.type}: reply count`); return replies.at(-1);
}
async function initialize(source) {
  payload = source;
  assert.equal((await send({ type: "init", url: "frozen test payload", generation: 11 })).type, "ready");
  assert.equal((await send({ type: "learn:init", config, seed: 1 })).type, "learn:ready");
}
const arrayHash = a => a == null ? null : hash(Buffer.from(a.buffer, a.byteOffset, a.byteLength));
function normalized(value) {
  if (ArrayBuffer.isView(value)) return { type: value.constructor.name, length: value.length, sha256: arrayHash(value) };
  if (Array.isArray(value)) return value.map(normalized);
  if (value && typeof value === "object") return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, normalized(v)]));
  return value;
}
function snapshot() {
  return { brain: Object.fromEntries(["v", "s", "drive", "w", "bias", "total", "controlActivation", "controlDefect"].map(k => [k, normalized(brain[k])])),
    scratchExists: Object.fromEntries(["total", "controlActivation", "controlDefect"].map(k => [k, Object.hasOwn(brain, k)])),
    steps: brain.steps, learner: Object.fromEntries(["efficacy", "trace", "traceBias", "traceCritic", "wCritic", "bCritic", "updates", "dropped", "pending"].map(k => [k, normalized(learner[k])])) };
}
function independentState(b, potential, tolerance) {
  const activity = new Float64Array(b.n), defect = new Float64Array(b.n), input = new Float64Array(b.n);
  const rest = 1 / (1 + Math.exp(b.slope * b.threshold));
  for (let i = 0; i < b.n; i++) {
    const raw = 1 / (1 + Math.exp(-b.slope * (potential[i] - b.threshold))) - rest;
    activity[i] = raw > 0 ? raw / (1 - rest) : b.leak * raw / rest;
  }
  let residual = 0, settledCount = 0;
  for (let i = 0; i < b.n; i++) {
    for (let e = b.rowPtr[i]; e < b.rowPtr[i + 1]; e++) input[i] += b.w[e] * activity[b.pre[e]];
    defect[i] = input[i] + (b.drive[i] + b.bias[i]) - potential[i];
    residual = Math.max(residual, Math.abs(defect[i]));
    if (Math.abs(defect[i]) <= tolerance) settledCount++;
  }
  return { activity, defect, input, residual, settledCount };
}
function near(a, b, tolerance = 1e-11) { assert.ok(Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= tolerance, `${a} != ${b}`); }
function verifyTrace(trace, evidence) {
  assert.equal(trace.schema, "cadence-settlement-trace/v1"); assert.equal(trace.n, brain.n); assert.equal(trace.edgeCount, brain.edges);
  assert.ok(trace.frames.length >= 2 && trace.frames.length <= 8);
  const patches = trace.patches, checks = [];
  for (const edge of trace.edges) {
    assert.equal(edge.pre, brain.pre[edge.id]); assert.equal(edge.weight, brain.w[edge.id]);
    assert.ok(brain.rowPtr[edge.post] <= edge.id && edge.id < brain.rowPtr[edge.post + 1]);
  }
  const shared = new Map();
  for (const edge of trace.edges) { if (!shared.has(edge.pre)) shared.set(edge.pre, new Set()); shared.get(edge.pre).add(edge.post); }
  assert.deepEqual(trace.sharedInputs, [...shared].filter(([, posts]) => posts.size > 1).map(([id, posts]) => ({ id, posts: [...posts] })));
  for (const patch of patches) {
    assert.equal(patch.incomingCount, brain.rowPtr[patch.id + 1] - brain.rowPtr[patch.id]);
    assert.equal(patch.sampledIncomingCount, trace.edges.filter(e => e.post === patch.id).length);
    for (const label of patch.labels) assert.ok(brain.sets[label].includes(patch.id));
  }
  for (const f of trace.frames) {
    const state = evidence.saved.get(f.iteration) ?? (evidence.last.iteration === f.iteration ? evidence.last : null);
    assert.ok(state, `missing independent state at iteration ${f.iteration}`);
    const actual = independentState(brain, state.v, trace.tolerance);
    near(f.residual, actual.residual); assert.equal(f.settledCount, actual.settledCount);
    const prev = state.previous ? independentState(brain, state.previous, trace.tolerance) : null;
    let worstUpdateError = 0;
    if (prev) for (let i = 0; i < brain.n; i++) {
      const expected = state.previous[i] + brain.dt * prev.defect[i];
      worstUpdateError = Math.max(worstUpdateError, Math.abs(expected - state.v[i]));
    }
    assert.ok(worstUpdateError <= 1e-11, `full-state Euler mismatch ${worstUpdateError}`);
    patches.forEach(({ id }, k) => {
      near(f.potential[k], state.v[id]); near(f.activity[k], actual.activity[id]);
      near(f.drive[k], brain.drive[id]); near(f.bias[k], brain.bias[id]);
      near(f.synapticInput[k], actual.input[id]); near(f.defect[k], actual.defect[id]);
      if (prev) {
        near(f.previousPotential[k], state.previous[id]); near(f.previousDefect[k], prev.defect[id]);
        near(f.deltaV[k], state.v[id] - state.previous[id]);
      } else { assert.equal(f.previousPotential, null); assert.equal(f.previousDefect, null); assert.equal(f.deltaV[k], 0); }
    });
    near(f.localMaxDefect, Math.max(...patches.map(({ id }) => Math.abs(actual.defect[id]))));
    trace.edges.forEach((edge, k) => { near(f.edgeActivity[k], actual.activity[edge.pre]); near(f.contribution[k], edge.weight * actual.activity[edge.pre]); });
    checks.push({ iteration: f.iteration, residual: actual.residual, settledCount: actual.settledCount,
      localMaxDefect: f.localMaxDefect, checkedNeuronUpdates: prev ? brain.n : 0, worstUpdateError });
  }
  return { neurons: brain.n, edges: brain.edges, selectedPatches: patches.length, sampledEdges: trace.edges.length, frames: checks };
}
function independentRetina(input, members) {
  const ids = [...members].sort((a, b) => a - b), rows = Math.max(1, Math.round(Math.sqrt(ids.length / 2)));
  const short = Math.floor(ids.length / rows), extra = ids.length % rows, levels = new Float32Array(ids.length);
  const { width, height, rgba, origin } = input;
  const lum = (x, y) => { const k = 4 * (y * width + x); return (.2126 * rgba[k] + .7152 * rgba[k + 1] + .0722 * rgba[k + 2]) / 255; };
  for (let row = 0, k = 0; row < rows; row++) {
    const cols = short + (row < extra ? 1 : 0);
    for (let col = 0; col < cols; col++, k++) {
      const x = (col + .5) / cols * (width - 1), v = (row + .5) / rows;
      const y = (origin === "bottom-left" ? 1 - v : v) * (height - 1), x0 = Math.floor(x), y0 = Math.floor(y);
      const x1 = Math.min(x0 + 1, width - 1), y1 = Math.min(y0 + 1, height - 1), dx = x - x0, dy = y - y0;
      levels[k] = (1 - dy) * ((1 - dx) * lum(x0, y0) + dx * lum(x1, y0)) + dy * ((1 - dx) * lum(x0, y1) + dx * lum(x1, y1));
    }
  }
  return { indices: ids, levels };
}
function verifyPixels(trace, packet) {
  const recorded = trace.retinal, expected = independentRetina(packet, brain.sets.photoreceptor);
  assert.equal(recorded.frame.width, packet.width); assert.equal(recorded.frame.height, packet.height); assert.equal(recorded.frame.origin, packet.origin);
  assert.deepEqual(Array.from(recorded.frame.rgba), Array.from(packet.rgba));
  assert.deepEqual(Array.from(recorded.indices), expected.indices); assert.deepEqual(Array.from(recorded.levels), Array.from(expected.levels));
  for (let k = 0; k < expected.indices.length; k++) near(brain.drive[expected.indices[k]], brain.amplitude * expected.levels[k]);
  return { receptors: expected.indices.length, rgba_sha256: arrayHash(packet.rgba), levels_sha256: arrayHash(expected.levels) };
}
function cleanReply(m) {
  const { ms, trace, retinal, ...rest } = m;
  const { capture, ...retinalSummary } = retinal || {};
  return normalized({ ...rest, retinal: retinal ? retinalSummary : retinal });
}
async function runArm(source, traceEnabled) {
  await initialize(source);
  const packet = frame(), options = { maxPatches: 32, maxEdges: 96, maxFrames: 8, ...(source.n === 7 ? { focus: [4, 5] } : {}) };
  if (traceEnabled) witness = { iteration: 0, stride: 1, saved: new Map(), previous: null, last: null };
  const observation = await send({ type: "assist:observe", requestId: 1, generation: 11, senses: {},
    retina: packet, steps: 256, tolerance: 1e-6, ...(traceEnabled ? { trace: options } : {}) });
  const evidence = witness; witness = null;
  assert.equal(observation.converged, true, JSON.stringify(cleanReply(observation)));
  const observedState = snapshot();
  let arithmetic = null, pixels = null;
  if (traceEnabled) {
    assert.deepEqual(observation.trace.identity, { generation: 11, requestId: 1, source: "control" });
    arithmetic = verifyTrace(observation.trace, evidence); pixels = verifyPixels(observation.trace, packet);
    if (source.n === 7) assert.ok(observation.trace.frames[0].residual > observation.trace.frames[0].localMaxDefect,
      "hidden receptor must expose selected-only residual bug");
  } else assert.ok(!observation.trace);
  const proposal = await send({ type: "assist:decide", requestId: 2, generation: 11, searchToken: "11:2",
    senses: bothOdors, retina: packet, steps: 1024, tolerance: 1e-6, selectTarget: true });
  assert.equal(proposal.accepted, true, JSON.stringify(cleanReply(proposal)));
  return { observation: cleanReply(observation), observedState, proposal: cleanReply(proposal), finalState: snapshot(), arithmetic, pixels };
}
async function test(name, fn) {
  const item = { name, passed: false }; cases.push(item);
  try { item.evidence = await fn(); item.passed = true; console.log(`ok ${name}`); }
  catch (error) { item.error = String(error); throw error; }
}
let failure = null;
try {
  await import("../web/worker.js");
  await test("fresh and rejected diagnostics preserve cache existence, identities and learner state", async () => {
    await initialize(fixture());
    for (const change of [{}, { steps: 1 }, { retina: { ...frame(), rgba: new Uint8Array(3) } }, { trace: { maxFrames: 999 } }]) {
      const before = snapshot(), refs = Object.fromEntries(["v", "s", "drive", "total", "controlActivation", "controlDefect"].map(k => [k, brain[k]]));
      const m = await send({ type: "inspect:retina", requestId: 90, generation: 11, senses: {}, retina: frame(), steps: 256, tolerance: 1e-6,
        trace: { maxFrames: 8 }, ...change });
      assert.equal(m.type, "retina_diagnostic"); assert.equal(m.diagnosticOnly, true);
      assert.deepEqual(snapshot(), before);
      for (const [key, ref] of Object.entries(refs)) assert.equal(brain[key], ref, `scratch reference changed: ${key}`);
      assert.ok(!m.s && !m.decision && !m.readouts && m.kind !== "control");
      if (Object.keys(change).length) assert.ok(!m.comparable);
    }
    return { cases: 4, original_scratch_cache_absence_preserved: true };
  });
  for (const [label, source] of [["tiny", fixture()], ...(process.argv.includes("--payload")
    ? [["actual full retained graph", JSON.parse(readFileSync(new URL("web/data/brain_full.json", root), "utf8"))]] : [])]) {
    await test(`${label}: trace on/off exact state and decision parity, independent full-state arithmetic`, async () => {
      const off = await runArm(source, false), on = await runArm(source, true);
      for (const key of ["observation", "observedState", "proposal", "finalState"]) assert.deepEqual(on[key], off[key], key);
      return { arithmetic: on.arithmetic, pixels: on.pixels, state_sha256: hash(JSON.stringify(on.finalState)),
        targetDraw: on.proposal.targetSelection.draw, actionDraw: on.proposal.decision.draw };
    });
    await test(`${label}: isolated retinal comparison restores live state and outstanding proposal`, async () => {
      const before = snapshot();
      diagnosticRuns = [];
      const m = await send({ type: "inspect:retina", requestId: 91, generation: 11, senses: {}, retina: frame(), steps: 256, tolerance: 1e-6,
        trace: { maxPatches: 32, maxEdges: 96, maxFrames: 8 } });
      const captured = diagnosticRuns; diagnosticRuns = null;
      assert.equal(m.type, "retina_diagnostic"); assert.equal(m.requestId, 91); assert.equal(m.generation, 11); assert.equal(m.diagnosticOnly, true);
      assert.ok(!m.s && !m.decision && !m.targetSelection && !m.readouts && m.kind !== "control");
      assert.deepEqual(snapshot(), before, "diagnostic modified live neural/learner state or cached solve arrays");
      for (const branch of [m.actual, m.black]) assert.equal(branch.solve.converged, true);
      assert.equal(m.comparable, true); assert.equal(captured.length, 2);
      const [actual, black] = captured;
      for (const key of ["v", "s", "w", "bias"]) assert.equal(actual.before[key], black.before[key], `counterfactual changed ${key}`);
      const receptors = new Set(brain.sets.photoreceptor), contrast = { count: 0, changedCount: 0, maxAbsDelta: 0, maxIndex: null };
      for (let i = 0; i < brain.n; i++) if (!receptors.has(i)) {
        assert.equal(actual.before.drive[i], black.before.drive[i]); contrast.count++;
        const delta = Math.abs(actual.activity[i] - black.activity[i]);
        if (delta !== 0) contrast.changedCount++;
        if (delta > contrast.maxAbsDelta) { contrast.maxAbsDelta = delta; contrast.maxIndex = i; }
      }
      assert.deepEqual(m.nonreceptor, contrast);
      assert.ok(contrast.maxAbsDelta > 1e-8 && contrast.changedCount > 0);
      for (const name of Object.keys(m.deltaReadouts)) {
        const mean = values => brain.sets[name].reduce((sum, i) => sum + values[i], 0) / brain.sets[name].length;
        near(m.actual.readouts[name], mean(actual.activity)); near(m.black.readouts[name], mean(black.activity));
        near(m.deltaReadouts[name], mean(actual.activity) - mean(black.activity));
      }
      // Diagnostic cannot replace, accept or cancel the earlier proposal.
      await send({ type: "assist:accept", requestId: 2, generation: 11, searchToken: "11:2" }, 0);
      const lesson = await send({ type: "assist:reward", requestId: 2, generation: 11, searchToken: "11:2", neuralCredit: true,
        reward: 1, done: true, steps: 256, tolerance: 1e-6 });
      assert.equal(lesson.accepted, true);
      return { actual: m.actual.solve, black: m.black.solve, nonreceptor: m.nonreceptor, deltaReadouts: m.deltaReadouts,
        restored_state_sha256: hash(JSON.stringify(before)), proposal_survived: true };
    });
  }
  assert.deepEqual(sourceHashes(), frozen, "source changed during run");
} catch (error) { failure = String(error.stack || error); console.error(failure); }
finally {
  globalThis.self = original.self; globalThis.fetch = original.fetch;
  ActorCriticLearner.prototype.stats = original.stats; SettlingBrain.prototype.settleControl = original.control;
  SettlingBrain.prototype._equationResidualFromActivation = original.residual;
}
const receipt = { schema: "cadence.settlement-trace-worker-check/1", started, completed: new Date().toISOString(),
  passed: failure === null, error: failure, sources_unchanged: JSON.stringify(sourceHashes()) === JSON.stringify(frozen), sha256: frozen,
  protocol: { pattern: "64x32 right-half RGB96, alpha255, bottom-left", observation_cap: 256, tolerance: 1e-6,
    trace_frames: 8, seed: 1, decision_cap: 1024, diagnostic_cap: 256,
    scope: "One fixed-input wiring/telemetry assay, no navigation or equilibrium uniqueness claim; independent full-state witnesses retained only in RAM." }, cases };
receipt.body_sha256 = hash(JSON.stringify(receipt));
if (receiptPath) writeFileSync(receiptPath, JSON.stringify(receipt, null, 2) + "\n", { flag: "wx" });
console.log(JSON.stringify({ passed: receipt.passed, checks: cases.length, sha256: receipt.body_sha256 }));
if (failure) process.exitCode = 1;
