// Numerical capture, anatomical mapping and real transferable-buffer ownership.
// Run: node tests/neural-replay.mjs [--payload]
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { SettlingBrain } from "../web/brain.js";
import { REPLAY_MAX_BYTES, replayScanFrame, replayTransferables, validateReplayOptions } from "../web/neural-replay.js";

const b64 = a => Buffer.from(a.buffer, a.byteOffset, a.byteLength).toString("base64");
const fixture = (dt = .37) => ({ n: 6, edges: 8,
  model: { dt, slope: 1, threshold: 0, leak: 1, gain: 1, stimulus_amplitude: 1 },
  arrays: { row_ptr: b64(new Int32Array([0, 0, 0, 2, 5, 8, 8])),
    pre: b64(new Int32Array([0, 1, 0, 2, 4, 0, 2, 3])),
    weight: b64(new Float64Array([.7, -.2, .1, .4, -.15, -.3, .5, .05])) },
  populations: { photoreceptor: [0, 1], "orn:decaying_fruit:left": [0], "orn:yeasty:left": [1], output: [3, 4] } });
const make = (dt = .37) => { const b = new SettlingBrain(fixture(dt)); b.drive.set([.8, .2, 0, 0, 0, 5]); b.bias[2] = .01; return b; };
const keys = ["v", "s", "w", "bias", "drive", "total", "controlActivation", "controlDefect"];
const bytes = a => a == null ? null : Buffer.from(a.buffer, a.byteOffset, a.byteLength);
const snapshot = b => Object.fromEntries([...keys.map(k => [k, bytes(b[k])?.toString("hex") ?? null]), ["steps", b.steps]]);
let tests = 0;
async function test(name, run) { await run(); tests++; console.log(`ok ${name}`); }

await test("replay on/off and simultaneous local trace preserve every Float64 state byte", () => {
  const plain = make(), captured = make();
  for (const b of [plain, captured]) { b.v.set([.01, -.2, .3, -.1, .8, .1]); b.s.fill(.77); }
  const expected = plain.settleControl(256, 1e-10, true);
  const { replay, ...actual } = captured.settleControl(256, 1e-10, true, true);
  assert.deepEqual(actual, expected); assert.deepEqual(snapshot(captured), snapshot(plain));
  assert.deepEqual(replay.frames.map(f => f.iteration), [0, 1, 2, 4, 8, 16, replay.iterations]);
  const before = snapshot(captured); replay.frames[0].activity[0] = 999; replay.frames[0].deltaV[0] = 999;
  assert.deepEqual(snapshot(captured), before);
});

await test("all sampled activities and signed updates equal their real iteration, never a decimated difference", () => {
  const b = make(), witnesses = [];
  const original = b._equationResidualFromActivation;
  b._equationResidualFromActivation = function (a) {
    const residual = original.call(this, a);
    witnesses.push({ activity: Float32Array.from(a), potential: this.v.slice(), residual });
    return residual;
  };
  const { replay } = b.settleControl(256, 1e-10, null, true);
  let observedNegative = false, max = 0;
  for (const f of replay.frames) {
    const witness = witnesses[f.iteration];
    assert.deepEqual(f.activity, witness.activity); assert.equal(f.residual, witness.residual);
    for (let i = 0; i < b.n; i++) {
      const expected = f.iteration ? Math.fround(witness.potential[i] - witnesses[f.iteration - 1].potential[i]) : 0;
      assert.equal(f.deltaV[i], expected); observedNegative ||= expected < 0;
      max = Math.max(max, Math.abs(expected));
    }
    assert.equal(f.maxAbsDeltaV, Math.max(...f.deltaV.map(Math.abs)));
  }
  assert.ok(observedNegative); assert.equal(replay.maxAbsDeltaV, max);
  assert.equal(replay.arrayBytes, replay.frames.length * b.n * 8);
});

await test("bounded initial/final capture handles caps, short solves and zero iterations", () => {
  for (const maxFrames of [2, 3, 7]) {
    const b = make(1e-6), { replay } = b.settleControl(1024, 1e-15, null, { maxFrames });
    assert.equal(replay.converged, false); assert.equal(replay.reason, "iteration_limit");
    assert.ok(replay.frames.length <= maxFrames); assert.equal(replay.frames[0].iteration, 0);
    assert.equal(replay.frames.at(-1).iteration, 1024);
    assert.ok(replay.frames.at(-1).deltaV[5] < 1e-5);
  }
  const b = make(); b.settleControl(256, 1e-10);
  const replay = b.settleControl(256, 1e-10, null, true).replay;
  assert.equal(replay.frames.length, 1); assert.equal(replay.iterations, 0);
  assert.equal(replay.maxAbsDeltaV, 0); assert.ok(replayScanFrame(replay, 0).heat.every(v => v === 0));
  const short = make().settleControl(2, 1e-15, null, true).replay;
  assert.deepEqual(short.frames.map(f => f.iteration), [0, 1, 2]);
});

await test("atlas mapping preserves signs/activity and uses one heat scale for the entire replay", () => {
  const replay = make().settleControl(256, 1e-10, null, true).replay;
  replay.identity = { requestId: 19, generation: 4, source: "control" };
  const members = Int32Array.from([7, 2, 5, 8, 0, 3]);
  for (let k = 0; k < replay.frames.length; k++) {
    const shown = replayScanFrame(replay, k, members, 9), frame = replay.frames[k];
    assert.deepEqual(shown.identity, replay.identity); assert.equal(shown.mappedCount, 6);
    assert.match(shown.label, /Recorded solve #19/);
    for (let i = 0; i < 6; i++) {
      assert.equal(shown.activity[members[i]], frame.activity[i]);
      assert.equal(shown.heat[members[i]], Math.fround(Math.log1p(99 * Math.abs(frame.deltaV[i]) / replay.maxAbsDeltaV) / Math.log(100)));
    }
    for (const omitted of [1, 4, 6]) assert.equal(shown.heat[omitted], 0);
  }
  assert.ok(Math.max(...replayScanFrame(replay, replay.frames.length - 1).heat) < 1e-6,
    "a final tiny update cannot be renormalized into a bright replay peak");
  assert.throws(() => replayScanFrame(replay, 0, [0, 0, 1, 2, 3, 4]));
  assert.throws(() => replayScanFrame(replay, 0, [0, 1, 2, 3, 4, 6]));
});

await test("invalid capture budget rejects atomically and nonfinite data cannot be shown as settled", () => {
  for (const options of [0, [], "yes", { maxFrames: 1 }, { maxFrames: 8 }, { maxFrames: 2.2 }]) {
    const b = make(), before = snapshot(b);
    assert.throws(() => b.settleControl(20, 1e-6, null, options)); assert.deepEqual(snapshot(b), before);
  }
  assert.throws(() => validateReplayOptions(true, 200000));
  assert.ok(validateReplayOptions(true, 150802));
  const b = make(); b.w[0] = NaN;
  const replay = b.settleControl(20, 1e-6, null, true).replay;
  assert.equal(replay.converged, false); assert.equal(replay.residual, null);
  assert.equal(replay.frames[0].valid, false); assert.throws(() => replayScanFrame(replay, 0));
});

await test("real structuredClone transfer detaches only replay copies, preserving live solver ownership", () => {
  const b = make(), replay = b.settleControl(256, 1e-10, null, true).replay, before = snapshot(b);
  const transfer = replayTransferables(replay);
  assert.equal(transfer.length, replay.frames.length * 2);
  const received = structuredClone(replay, { transfer });
  assert.ok(transfer.every(buffer => buffer.byteLength === 0));
  assert.deepEqual(snapshot(b), before); assert.equal(received.frames.at(-1).activity.length, b.n);
  assert.ok(received.arrayBytes <= REPLAY_MAX_BYTES); replayScanFrame(received, received.frames.length - 1);
});

await test("worker sends bound replay on successful and capped solves, validates before drive mutation", async () => {
  const prior = { self: globalThis.self, fetch: globalThis.fetch, solve: SettlingBrain.prototype.settleControl };
  let brain, message, transferred = 0;
  globalThis.fetch = async () => ({ json: async () => fixture() });
  globalThis.self = { postMessage(m, transfer = []) {
    transferred = transfer.length; message = structuredClone(m, { transfer });
    assert.ok(transfer.every(buffer => buffer.byteLength === 0));
  } };
  SettlingBrain.prototype.settleControl = function (...args) { brain = this; return prior.solve.apply(this, args); };
  try {
    await import("../web/worker.js");
    const send = async data => { await self.onmessage({ data }); return message; };
    await send({ type: "init", url: "fixture", generation: 7 });
    for (const type of ["settle_control", "assist:observe"]) {
      const reply = await send({ type, requestId: 23, generation: 7, senses: { "orn:decaying_fruit:left": .8 }, steps: 256, tolerance: 1e-10, replay: true });
      assert.equal(reply.converged, true); assert.deepEqual(reply.replay.identity, { requestId: 23, generation: 7, source: "control" });
      assert.equal(transferred, 1 + 2 * reply.replay.frames.length);
      assert.deepEqual(reply.s, Float32Array.from(brain.s)); assert.equal(brain.v.length, 6);
      const before = snapshot(brain);
      const invalid = await send({ type, requestId: 24, generation: 7, senses: { "orn:decaying_fruit:left": .1 }, replay: { maxFrames: 8 } });
      assert.equal(invalid.converged, false); assert.deepEqual(snapshot(brain), before);
    }
    const capped = await send({ type: "assist:observe", requestId: 25, generation: 7,
      senses: { "orn:yeasty:left": 1 }, steps: 1, tolerance: 1e-15, replay: true });
    assert.equal(capped.converged, false); assert.equal(capped.reason, "iteration_limit");
    assert.equal(capped.s, undefined); assert.equal(capped.readouts, undefined);
    assert.equal(transferred, 2 * capped.replay.frames.length);
    assert.equal(brain.s.length, 6); assert.equal(capped.replay.frames.at(-1).iteration, 1);
  } finally { globalThis.self = prior.self; globalThis.fetch = prior.fetch; SettlingBrain.prototype.settleControl = prior.solve; }
});

if (process.argv.includes("--payload")) await test("full retained graph replay is bounded and bit-identical without extra residual evaluations", () => {
  const payload = JSON.parse(readFileSync(new URL("../web/data/brain_full.json", import.meta.url)));
  const a = new SettlingBrain(payload), b = new SettlingBrain(payload);
  for (const brain of [a, b]) brain.stimulate("orn:yeasty:left", .4);
  let calls = 0;
  const original = b._equationResidualFromActivation;
  b._equationResidualFromActivation = function (activity) { calls++; return original.call(this, activity); };
  const t0 = performance.now(), plain = a.settleControl(256, 1e-6), plainMs = performance.now() - t0;
  const t1 = performance.now(), { replay, ...captured } = b.settleControl(256, 1e-6, null, true), replayMs = performance.now() - t1;
  assert.deepEqual(captured, plain); assert.deepEqual(snapshot(a), snapshot(b));
  assert.equal(calls, captured.iterations + 1); assert.equal(replay.n, 150802);
  assert.ok(replay.arrayBytes <= REPLAY_MAX_BYTES);
  const shown = replayScanFrame(replay, replay.frames.length - 1, b.members, 150802);
  assert.equal(shown.mappedCount, 150802);
  console.log(JSON.stringify({ n: b.n, edges: b.edges, iterations: captured.iterations,
    converged: captured.converged, residual: captured.residual, frames: replay.frames.map(f => f.iteration),
    arrayBytes: replay.arrayBytes, plainMs, replayMs }));
});
console.log(`${tests} neural replay groups passed`);
