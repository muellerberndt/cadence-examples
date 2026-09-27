// Independent arithmetic and observer isolation on a directed, signed graph.
// These are numerical telemetry tests, not a biological or behavioral claim.
// Run: node tests/settlement-trace.mjs
import assert from "node:assert/strict";
import { SettlingBrain } from "../web/brain.js";
import { TRACE_MAX_PATCHES, TRACE_MAX_EDGES, TRACE_MAX_FRAMES } from "../web/settlement-trace.js";

const b64 = a => Buffer.from(a.buffer, a.byteOffset, a.byteLength).toString("base64");
const fixture = (dt = .37) => ({ n: 6, edges: 8,
  model: { dt, slope: 1, threshold: 0, leak: 1, gain: 1, stimulus_amplitude: 1 },
  arrays: { row_ptr: b64(new Int32Array([0, 0, 0, 2, 5, 8, 8])),
    pre: b64(new Int32Array([0, 1, 0, 2, 4, 0, 2, 3])),
    weight: b64(new Float64Array([.7, -.2, .1, .4, -.15, -.3, .5, .05])) },
  populations: { photoreceptor: [0, 1], input: [0, 1], output: [3, 4], motor: [3, 4] } });
const brain = (dt = .37) => { const b = new SettlingBrain(fixture(dt)); b.drive.set([.8, .2, 0, 0, 0, 5]); b.bias[2] = .01; return b; };
const near = (a, b, tolerance = 3e-14) => assert.ok(Math.abs(a - b) <= tolerance, `${a} != ${b}`);
const bytes = a => Buffer.from(a.buffer, a.byteOffset, a.byteLength);
const stateKeys = ["v", "s", "w", "bias", "drive", "total", "controlActivation", "controlDefect"];
function sameState(a, b) {
  for (const key of stateKeys) assert.deepEqual(bytes(a[key]), bytes(b[key]), `bit-identical ${key}`);
  assert.equal(a.steps, b.steps);
}
let tests = 0;
function test(name, run) { run(); tests++; console.log(`ok ${name}`); }

test("trace on/off is bit-identical, including restored-potential activation and scratch state", () => {
  const plain = brain(), observed = brain();
  // Public control semantics repair stale activity from authoritative potentials.
  for (const b of [plain, observed]) { b.v.set([.01, -.2, .3, -.1, .8, .1]); b.s.fill(.77); }
  const expected = plain.settleControl(256, 1e-10);
  const { trace, ...actual } = observed.settleControl(256, 1e-10, true);
  assert.deepEqual(actual, expected); sameState(plain, observed);
  assert.equal(trace.converged, true); assert.ok(trace.frames.length >= 2);
  const before = observed.v.slice(); trace.frames[0].potential[0] = 999;
  assert.deepEqual(observed.v, before, "display copies cannot alter the solved state");
});

test("each recorded frame reconstructs all-neuron residuals, full local inputs and actual signed CSR contributions", () => {
  const b = brain(), independent = [];
  const original = b._equationResidualFromActivation;
  b._equationResidualFromActivation = function (a) {
    const activity = Array.from(this.v, v => Math.tanh(v / 2));
    const input = Array(this.n).fill(0);
    for (let i = 0; i < this.n; i++) for (let e = this.rowPtr[i]; e < this.rowPtr[i + 1]; e++) input[i] += activity[this.pre[e]] * this.w[e];
    const defect = input.map((sum, i) => sum + (this.drive[i] + this.bias[i]) - this.v[i]);
    independent.push({ potential: [...this.v], activity, input, defect, residual: Math.max(...defect.map(Math.abs)) });
    return original.call(this, a);
  };
  const trace = b.settleControl(256, 1e-10, { focus: [0, 2, 3, 4], maxFrames: 32, maxEdges: 3 }).trace;
  assert.equal(trace.frames[0].residual, 5, "the worst cell is deliberately outside the displayed patches");
  assert.ok(trace.frames[0].localMaxDefect < trace.frames[0].residual);
  assert.ok(trace.patches.some(p => p.sampledIncomingCount < p.incomingCount));
  for (const frame of trace.frames) {
    const ref = independent[frame.iteration];
    near(frame.residual, ref.residual);
    assert.equal(frame.settledCount, ref.defect.filter(x => Math.abs(x) <= trace.tolerance).length);
    trace.patches.forEach((patch, k) => {
      near(frame.potential[k], ref.potential[patch.id]); near(frame.activity[k], ref.activity[patch.id]);
      near(frame.synapticInput[k], ref.input[patch.id]); near(frame.defect[k], ref.defect[patch.id]);
      if (frame.iteration) {
        const previous = independent[frame.iteration - 1];
        assert.equal(frame.previousPotential[k], previous.potential[patch.id]);
        near(frame.previousDefect[k], previous.defect[patch.id]);
        assert.equal(frame.deltaV[k], frame.potential[k] - frame.previousPotential[k]);
        assert.equal(frame.potential[k], frame.previousPotential[k] + trace.dt * frame.previousDefect[k], "actual rounded Euler update from the preceding STEP, not preceding display frame");
      } else { assert.equal(frame.deltaV[k], 0); assert.equal(frame.previousDefect, null); }
    });
    trace.edges.forEach((edge, k) => {
      assert.equal(b.pre[edge.id], edge.pre); assert.equal(b.w[edge.id], edge.weight);
      assert.ok(edge.id >= b.rowPtr[edge.post] && edge.id < b.rowPtr[edge.post + 1]);
      near(frame.edgeActivity[k], ref.activity[edge.pre]);
      assert.equal(frame.contribution[k], frame.edgeActivity[k] * edge.weight);
    });
  }
  const last = trace.frames.at(-1); assert.equal(last.iteration, trace.iterations); assert.equal(last.residual, trace.residual);
});

test("overlaps are exactly shared presynaptic state among actual selected incoming edges", () => {
  const trace = brain().settleControl(50, 1e-6, { focus: [2, 3, 4] }).trace;
  assert.ok(trace.edges.some(e => e.weight < 0));
  assert.deepEqual(trace.sharedInputs.find(p => p.id === 0).posts, [2, 3, 4]);
  const expected = new Map();
  for (const e of trace.edges) { if (!expected.has(e.pre)) expected.set(e.pre, new Set()); expected.get(e.pre).add(e.post); }
  assert.deepEqual(trace.sharedInputs, [...expected].filter(([, posts]) => posts.size > 1).map(([id, posts]) => ({ id, posts: [...posts] })));
});

test("initial/final states and genuine previous-step updates remain bounded at every frame budget", () => {
  for (const maxFrames of [2, 3, 7, TRACE_MAX_FRAMES]) {
    const b = brain(1e-6), trace = b.settleControl(1024, 1e-15, { maxFrames }).trace;
    assert.equal(trace.converged, false); assert.equal(trace.reason, "iteration_limit");
    assert.ok(trace.frames.length <= maxFrames); assert.equal(trace.frames[0].iteration, 0);
    assert.equal(trace.frames.at(-1).iteration, 1024);
    assert.ok(trace.frames.at(-1).deltaV[5] < 1e-5, "a sparse-frame gap must not inflate the last update");
  }
  const b = brain(); b.settleControl(256, 1e-10);
  const trace = b.settleControl(256, 1e-10, true).trace;
  assert.equal(trace.iterations, 0); assert.equal(trace.frames.length, 1);
});

test("a short solve exposes its actual early propagation despite a large requested iteration cap", () => {
  const trace = brain().settleControl(1024, .1, true).trace;
  assert.equal(trace.converged, true); assert.ok(trace.iterations > 3 && trace.iterations < TRACE_MAX_FRAMES);
  assert.deepEqual(trace.frames.map(f => f.iteration), Array.from({ length: trace.iterations + 1 }, (_, k) => k));
});

test("bounded default focus retains sensory, named MBON, descending and motor endpoints with specific labels", () => {
  const names = ["mbon:MBON11:right", "mbon:MBON05:left", "dn:DNa02:left", "dn:DNa02:right", "dn:DNp09",
    "dn:DNp03", "dn:landing", "power:left", "power:right", "steering:left", "steering:right"];
  const n = 80, row = [0], pre = [];
  for (let i = 0; i < n; i++) { if (i >= 8 && i < 32) pre.push(i % 6); row.push(pre.length); }
  const payload = { ...fixture(), n, edges: pre.length,
    arrays: { row_ptr: b64(Int32Array.from(row)), pre: b64(Int32Array.from(pre)), weight: b64(new Float64Array(pre.length).fill(.1)) },
    populations: { ...Object.fromEntries(Array.from({ length: 12 }, (_, k) => [`generic${k}`, [62]])),
      photoreceptor: [0, 1, 2, 3, 4, 5, 6, 7], ...Object.fromEntries(names.map((name, k) => [name, [60 + k]])) } };
  const b = new SettlingBrain(payload); b.drive[0] = .4;
  const trace = b.settleControl(100, 1e-6, { maxPatches: 48, maxEdges: 10 }).trace;
  const selected = new Set(trace.patches.map(p => p.id));
  assert.ok(trace.patches.length <= 48); assert.ok(trace.edges.length <= 10);
  assert.ok(selected.has(0)); assert.ok(trace.patches.some(p => p.id >= 8 && p.id < 32));
  for (let k = 0; k < names.length; k++) assert.ok(selected.has(60 + k), names[k]);
  assert.ok(trace.patches.find(p => p.id === 62).labels.includes("dn:DNa02:left"), "generic labels cannot crowd out the actual named output");
});

test("invalid trace limits and requested recipients reject before changing state", () => {
  for (const options of [{ maxFrames: TRACE_MAX_FRAMES + 1 }, { maxFrames: 1 }, { maxPatches: TRACE_MAX_PATCHES + 1 },
    { maxEdges: TRACE_MAX_EDGES + 1 }, { maxEdges: 0 }, { focus: [0, 0] }, { focus: [6] }, { focus: [NaN] },
    { focus: { length: 1e9 } }, { focus: "0" }, { focus: [] }]) {
    const b = brain(), v = b.v.slice(), drive = b.drive.slice();
    assert.throws(() => b.settleControl(256, 1e-6, options));
    assert.deepEqual(b.v, v); assert.deepEqual(b.drive, drive); assert.equal(b.steps, 0);
    assert.equal(Object.hasOwn(b, "controlDefect"), false);
  }
});

test("nonfinite equations remain failures with null rather than invented residuals", () => {
  const b = brain(); b.w[0] = NaN;
  const result = b.settleControl(10, 1e-6, true);
  assert.equal(result.converged, false); assert.equal(result.trace.residual, null);
  assert.equal(result.trace.frames[0].settledCount, null);
  assert.equal(result.trace.frames[0].localMaxDefect, null);
});

console.log(`${tests} settlement trace groups passed`);
