// Real worker messages with deterministic in-process self/fetch transport.
// node tests/assisted-worker.mjs [--payload]
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { SettlingBrain } from "../web/brain.js";
import { ActorCriticLearner } from "../web/learner.js";
import { SETTLED_MOTOR_GROUPS } from "../web/motor.js";

const b64 = array => Buffer.from(array.buffer, array.byteOffset, array.byteLength).toString("base64");
const fixture = () => ({ n: 3, edges: 4, synapses: 4, whole: { neurons: 3 },
  model: { dt: .5, slope: 1, threshold: 0, leak: 1, gain: 1, stimulus_amplitude: 1, adaptation: null },
  arrays: { row_ptr: b64(new Int32Array([0, 0, 2, 4])), pre: b64(new Int32Array([0, 2, 0, 1])),
    weight: b64(new Float64Array([.8, .1, -.4, .1])), sign: b64(new Float64Array([.8, .1, -.4, .1])), count: b64(new Uint16Array([1, 1, 1, 1])) },
  populations: { "haltere:left": [0], "vis:LC4": [1], "dan:pam": [0], kc: [0], mbon: [1, 2], motor: [1, 2],
    "mbon:MBON11:right": [1], "mbon:MBON05:left": [2],
    ...Object.fromEntries(SETTLED_MOTOR_GROUPS.map((name, i) => [name, [i % 2 + 1]])) } });
const config = { outputs: ["mbon:MBON11:right", "mbon:MBON05:left"], actions: [0, 1], plastic: "all", critic: "kc",
  beta: .1, temperature: .3, eta: .1, etaBias: 0, etaCritic: .05, gamma: .95, lam: .9, cap: 3, dopamineCap: 1 };
const original = { self: globalThis.self, fetch: globalThis.fetch, stats: ActorCriticLearner.prototype.stats, control: SettlingBrain.prototype.settleControl };
let liveBrain = null, liveLearner = null, source = fixture();
const replies = [];
globalThis.self = { postMessage: message => replies.push(message) };
globalThis.fetch = async () => ({ json: async () => source });
ActorCriticLearner.prototype.stats = function (...args) { liveLearner = this; return original.stats.apply(this, args); };
SettlingBrain.prototype.settleControl = function (...args) { liveBrain = this; return original.control.apply(this, args); };
let assertions = 0, requestId = 0;
const send = async (message, expected = 1) => {
  const start = replies.length;
  await self.onmessage({ data: message });
  assert.equal(replies.length - start, expected, `${message.type}: unexpected reply count`);
  return replies.at(-1);
};
const identity = proposal => ({ requestId: proposal.requestId, generation: proposal.generation, searchToken: proposal.searchToken });
const arrays = names => Object.fromEntries(names.map(name => [name, [...liveLearner[name]]]));
const trace = () => arrays(["trace", "traceBias", "traceCritic"]);
const parameters = () => ({ ...arrays(["efficacy", "wCritic"]), weights: [...liveBrain.w], bias: [...liveBrain.bias], bCritic: liveLearner.bCritic, updates: liveLearner.updates });
const phaseCheck = proposal => {
  assert.equal(proposal.type, "assisted_decision"); assert.equal(proposal.kind, "decision");
  assert.equal(proposal.accepted, true); assert.equal(proposal.converged, true);
  for (const name of ["free", "plus", "minus"]) {
    const phase = proposal.decision.solves[name];
    assert.equal(phase.converged, true); assert.ok(phase.residual <= phase.tolerance);
  }
  assert.ok(proposal.readouts && proposal.s instanceof Float32Array);
  assert.ok(Number.isFinite(proposal.decision.draw));
};
const propose = async (extra = {}) => {
  const proposal = await send({ type: "assist:decide", requestId: ++requestId, generation: 3, searchToken: `3:${requestId}`,
    senses: { "haltere:left": 1 }, steps: 512, tolerance: 1e-8, ...extra });
  return proposal;
};
const test = async (name, fn) => { await fn(); assertions++; console.log(`ok ${name}`); };

try {
  await import("../web/worker.js");
  await test("uninitialized errors preserve ownership and have no authority", async () => {
    const r = await propose();
    assert.equal(r.accepted, false); assert.equal(r.converged, false);
    assert.equal(r.reason, "brain_not_initialized"); assert.equal(r.generation, 3);
    assert.ok(!r.readouts && !r.decision);
  });
  await send({ type: "init", url: "fixture", requestId: 0, generation: 3 });
  await send({ type: "learn:init", config, seed: 7 });
  await test("three checked phases propose traces but do not change parameters", async () => {
    const before = { efficacy: [...liveLearner.efficacy], wCritic: [...liveLearner.wCritic] };
    const proposal = await propose(); phaseCheck(proposal);
    assert.deepEqual([...liveLearner.efficacy], before.efficacy);
    assert.deepEqual([...liveLearner.wCritic], before.wCritic);
    assert.ok([...liveLearner.trace].some(value => value !== 0));
    await send({ type: "assist:cancel", ...identity(proposal) }, 0);
    assert.ok([...liveLearner.trace].every(value => value === 0));
    assert.equal(liveLearner.pending, null);
  });
  await test("wrong, unaccepted and replayed outcomes cannot learn; an accepted outcome changes weights once", async () => {
    const proposal = await propose(); phaseCheck(proposal);
    const before = parameters(), pendingTrace = trace();
    for (const changes of [{}, { neuralCredit: false }, { requestId: proposal.requestId + 1 }, { searchToken: "other" }, { generation: 2 }]) {
      const r = await send({ type: "assist:reward", ...identity(proposal), reward: 1, done: true, neuralCredit: true, ...changes });
      assert.equal(r.accepted, false); assert.deepEqual(parameters(), before); assert.deepEqual(trace(), pendingTrace);
    }
    await send({ type: "assist:accept", ...identity(proposal), searchToken: "wrong" }, 0);
    assert.equal((await send({ type: "assist:reward", ...identity(proposal), reward: 1, done: true, neuralCredit: true })).accepted, false);
    await send({ type: "assist:accept", ...identity(proposal) }, 0);
    const learned = await send({ type: "assist:reward", ...identity(proposal), reward: 1, done: true, neuralCredit: true });
    assert.equal(learned.accepted, true); assert.ok(learned.moved > 0);
    assert.notDeepEqual([...liveBrain.w], before.weights); assert.equal(liveLearner.updates, before.updates + 1);
    const after = parameters();
    assert.equal((await send({ type: "assist:reward", ...identity(proposal), reward: 1, done: true, neuralCredit: true })).accepted, false);
    assert.deepEqual(parameters(), after);
  });
  await test("cancel rolls all eligibility back to its preproposal snapshot, even after accept", async () => {
    liveLearner.trace.fill(.125); liveLearner.traceCritic.fill(.25);
    const prior = trace(), proposal = await propose(); phaseCheck(proposal);
    const changed = trace(); assert.notDeepEqual(changed, prior);
    await send({ type: "assist:cancel", ...identity(proposal), requestId: proposal.requestId + 1 }, 0);
    assert.deepEqual(trace(), changed);
    await send({ type: "assist:accept", ...identity(proposal) }, 0);
    await send({ type: "assist:cancel", ...identity(proposal) }, 0);
    assert.deepEqual(trace(), prior); assert.equal(liveLearner.pending, null);
  });
  await test("motor injection is rejected atomically through both assisted input APIs", async () => {
    for (const type of ["assist:observe", "assist:decide"]) {
      for (const senses of [{ "haltere:left": .2, "power:left": 1 }, { "vis:LC4": 1 }, { "haltere:left": NaN }, { "haltere:left": 1.01 }]) {
        const before = { v: [...liveBrain.v], s: [...liveBrain.s], drive: [...liveBrain.drive], parameters: parameters() };
        const bad = await send({ type, requestId: ++requestId, generation: 3, searchToken: "bad", senses, steps: 512, tolerance: 1e-8 });
        assert.equal(bad.converged, false); assert.equal(bad.accepted, false);
        assert.ok(!bad.readouts && !bad.s && !bad.decision);
        assert.deepEqual({ v: [...liveBrain.v], s: [...liveBrain.s], drive: [...liveBrain.drive], parameters: parameters() }, before);
      }
    }
  });
  await test("a failed free phase supplies no choice, readouts or eligibility", async () => {
    await send({ type: "reset", generation: 3 });
    const prior = trace(), before = parameters();
    const r = await propose({ steps: 1 });
    assert.equal(r.accepted, false); assert.ok(!r.readouts && !r.s);
    assert.deepEqual(trace(), prior); assert.deepEqual(parameters(), before);
    assert.equal(liveLearner.pending, null);
  });
  await test("failed observations retain their failure but cannot poison the next neural state or learning", async () => {
    const good = await send({ type: "assist:observe", requestId: ++requestId, generation: 3,
      senses: { "haltere:left": 1 }, steps: 512, tolerance: 1e-8 });
    assert.equal(good.converged, true);
    const v = [...liveBrain.v], s = [...liveBrain.s], before = parameters(), eligibility = trace();
    const failed = await send({ type: "assist:observe", requestId: ++requestId, generation: 3,
      senses: { "haltere:left": 0 }, steps: 1, tolerance: 1e-8 });
    assert.equal(failed.converged, false); assert.equal(failed.stateRestored, true);
    assert.ok(failed.residual > failed.tolerance); assert.ok(!failed.readouts && !failed.s);
    assert.deepEqual([...liveBrain.v], v); assert.deepEqual([...liveBrain.s], s);
    assert.deepEqual(parameters(), before); assert.deepEqual(trace(), eligibility);
    const recovered = await send({ type: "assist:observe", requestId: ++requestId, generation: 3,
      senses: { "haltere:left": 1 }, steps: 512, tolerance: 1e-8 });
    assert.equal(recovered.converged, true); assert.equal(recovered.iterations, 0);
    assert.equal(recovered.stateRestored, false);
  });
  await test("a failed nudged phase is rejected even with a qualified free equilibrium", async () => {
    await send({ type: "assist:observe", requestId: ++requestId, generation: 3, senses: { "haltere:left": 1 }, steps: 512, tolerance: 1e-8 });
    const before = parameters(), prior = trace();
    const rejected = await propose({ steps: 1 });
    assert.equal(rejected.decision.solves.free.converged, true);
    assert.equal(rejected.accepted, false);
    assert.ok(!rejected.readouts && !rejected.s);
    assert.deepEqual(parameters(), before); assert.deepEqual(trace(), prior);
    assert.equal(liveLearner.pending, null);
  });
  await test("malformed reward and invalid next sensory input cancel credit without a parameter update", async () => {
    for (const changes of [{ reward: NaN }, { done: "yes" }, { done: false, senses: { "power:left": 1 } }]) {
      const beforeTrace = trace(), proposal = await propose(); phaseCheck(proposal);
      await send({ type: "assist:accept", ...identity(proposal) }, 0);
      const before = parameters();
      const r = await send({ type: "assist:reward", ...identity(proposal), reward: 1, done: true, neuralCredit: true, ...changes });
      assert.equal(r.accepted, false); assert.deepEqual(parameters(), before);
      assert.deepEqual(trace(), beforeTrace); assert.equal(liveLearner.pending, null);
      assert.equal((await send({ type: "assist:reward", ...identity(proposal), reward: 1, done: true, neuralCredit: true })).accepted, false);
    }
  });
  await test("reset, learner reset and replacement cancel all outstanding decision authority", async () => {
    for (const mutation of [{ type: "reset", generation: 4 }, { type: "learn:reset" }, { type: "learn:init", config, seed: 7 }]) {
      const proposal = await propose(); phaseCheck(proposal);
      await send({ type: "assist:accept", ...identity(proposal) }, 0);
      await send(mutation);
      assert.equal(liveLearner.pending, null);
      const before = parameters();
      const r = await send({ type: "assist:reward", ...identity(proposal), neuralCredit: true, reward: 1, done: true });
      assert.equal(r.accepted, false); assert.deepEqual(parameters(), before);
    }
  });
  await test("a new proposal supersedes the old identity without accumulating its canceled traces", async () => {
    const prior = trace(), first = await propose(); phaseCheck(first);
    const second = await propose(); phaseCheck(second);
    await send({ type: "assist:accept", ...identity(first) }, 0);
    assert.equal((await send({ type: "assist:reward", ...identity(first), neuralCredit: true, reward: 1, done: true })).accepted, false);
    await send({ type: "assist:cancel", ...identity(second) }, 0);
    assert.deepEqual(trace(), prior);
  });
  await test("rewiring and learned-weight replacement remove old proposal authority", async () => {
    for (const mutation of [{ type: "shuffle", on: true, seed: 11 }, { type: "shuffle", on: false }, { type: "learned", on: false }]) {
      const proposal = await propose(); phaseCheck(proposal);
      await send({ type: "assist:accept", ...identity(proposal) }, 0);
      await send(mutation);
      assert.equal(liveLearner.pending, null);
      const before = { efficacy: [...liveLearner.efficacy], bias: [...liveLearner.brain.bias], weights: [...liveLearner.brain.w] };
      const r = await send({ type: "assist:reward", ...identity(proposal), reward: 1, done: true, neuralCredit: true });
      assert.equal(r.accepted, false);
      assert.deepEqual({ efficacy: [...liveLearner.efficacy], bias: [...liveLearner.brain.bias], weights: [...liveLearner.brain.w] }, before);
    }
  });
  await test("reinitializing the brain cannot reuse a learner attached to the previous graph", async () => {
    await send({ type: "init", url: "replacement fixture", generation: 4 });
    const invalid = await propose({ generation: 4, searchToken: "4:reinit" });
    assert.equal(invalid.accepted, false);
    assert.equal(invalid.reason, "learner_not_initialized");
    await send({ type: "learn:init", config, seed: 7 });
    phaseCheck(await propose({ generation: 4, searchToken: "4:fresh" }));
  });

  await test("shuffle keeps edge metadata and learned efficacy together through repeated rewires and restoration", async () => {
    const keys = ["pre", "count", "sign", "gainPre", "efficacy0"];
    const learned = [.63, -.27, 1.41, -.82];
    for (const plastic of ["all", [0, 3], { pre: ["kc"], post: ["mbon:MBON11:right"] }]) {
      source = fixture();
      const factors = [.3, .2, .1, .05], initial = [.8, -.5, 1.2, -.4];
      Object.assign(source.arrays, { count: b64(new Uint16Array([2, 3, 5, 7])),
        sign: b64(new Int8Array([1, -1, 1, -1])), sign_dtype: "int8",
        gain_pre: b64(new Float64Array(factors)), efficacy: b64(new Float64Array(initial)),
        weight: b64(Float64Array.from(initial, (value, e) => factors[e] * value)) });
      await send({ type: "init", url: "heterogeneous edge fixture" });
      await send({ type: "learn:init", config: { ...config, plastic }, seed: 7 });
      liveBrain = liveLearner.brain;
      const baseline = Object.fromEntries([...keys, "rowPtr", "w"].map(key => [key, [...liveBrain[key]]]));
      liveLearner.load([0, 1, 2, 3], learned);
      const originalWeights = [...liveBrain.w];
      const assertShuffled = () => {
        const identity = [...liveBrain.edgeIdentity];
        assert.deepEqual([...identity].sort((a, b) => a - b), [0, 1, 2, 3]);
        assert.ok(identity.some((e, k) => e !== k), "fixture must actually reorder edges");
        assert.deepEqual([...liveBrain.rowPtr], baseline.rowPtr, "postsynaptic degree counts are conserved");
        for (const key of keys) assert.deepEqual([...liveBrain[key]], identity.map(e => baseline[key][e]), `${key} stays with its edge`);
        assert.equal(liveBrain.sign.constructor, Int8Array);
        assert.deepEqual([...liveBrain.w], identity.map(e => originalWeights[e]));
        assert.deepEqual([...liveBrain.original.w], baseline.w, "original snapshot cannot alias learned weights");
        if (Array.isArray(plastic)) assert.deepEqual([...liveLearner.edges].map(e => identity[e]), plastic);
        for (let k = 0; k < liveLearner.edges.length; k++) {
          const e = liveLearner.edges[k];
          assert.ok(Math.abs(liveLearner.efficacy[k] - learned[identity[e]]) < 1e-14, "efficacy follows edge identity, including changed plastic subsets");
        }
        const before = [...liveBrain.w]; liveLearner.applyWeights();
        for (let e = 0; e < before.length; e++) assert.ok(Math.abs(liveBrain.w[e] - before[e]) < 1e-14, "reconstructing weights must preserve the shuffled graph");
      };
      await send({ type: "shuffle", on: true, seed: 11 }); assertShuffled();
      for (let k = 0; k < liveLearner.efficacy.length; k++) liveLearner.efficacy[k] += .111;
      liveLearner.applyWeights(); // learning in the control must not overwrite original learning
      await send({ type: "shuffle", on: true, seed: 21 }); assertShuffled();
      await send({ type: "learned", on: true, edges: [1], efficacy: [.71] });
      assert.equal(liveBrain.w[liveBrain.edgePosition[1]], factors[1] * .71, "checkpoint edge IDs remain payload IDs");
      await send({ type: "learned", on: false });
      assert.deepEqual([...liveBrain.w], [...liveBrain.edgeIdentity].map(e => baseline.w[e]));
      await send({ type: "shuffle", on: false });
      for (const key of [...keys, "rowPtr"]) assert.deepEqual([...liveBrain[key]], baseline[key]);
      assert.deepEqual([...liveBrain.w], originalWeights);
      assert.equal(liveBrain.edgeIdentity, null); assert.equal(liveBrain.edgePosition, null);
      for (let k = 0; k < liveLearner.edges.length; k++) assert.ok(Math.abs(liveLearner.efficacy[k] - learned[liveLearner.edges[k]]) < 1e-14);
      liveLearner.load([0], [.92]);
      const newerWeights = [...liveBrain.w];
      await send({ type: "shuffle", on: false });
      assert.deepEqual([...liveBrain.w], newerWeights, "redundant unshuffle cannot restore obsolete learning");
    }
  });

  await test("shuffle without a learner preserves optional metadata, checkpoint identity and original weights", async () => {
    for (const withGains of [false, true]) {
      source = fixture();
      if (withGains) {
        source.arrays.count = b64(new Uint16Array([2, 3, 5, 7]));
        source.arrays.log_gain = b64(new Float64Array([.1, -.3, .2]));
      } else { delete source.arrays.count; delete source.arrays.sign; }
      await send({ type: "init", url: "no learner fixture" });
      await send({ type: "assist:observe", senses: {}, steps: 1, tolerance: 1e-6 });
      const baseline = Object.fromEntries(["pre", "count", "sign", "gainPre", "efficacy0", "w"].map(key => [key, liveBrain[key]?.slice() ?? null]));
      await send({ type: "shuffle", on: true, seed: 11 }, 0);
      for (const [key, values] of Object.entries(baseline)) assert.deepEqual(liveBrain[key], values ? values.constructor.from(liveBrain.edgeIdentity, e => values[e]) : null);
      const edge = liveBrain.edgePosition[2], factor = liveBrain.gainPre ? liveBrain.gainPre[edge] : liveBrain.gain;
      await send({ type: "learned", on: true, edges: [2], efficacy: [.314] }, 0);
      assert.equal(liveBrain.w[edge], factor * .314);
      await send({ type: "shuffle", on: false }, 0);
      for (const [key, values] of Object.entries(baseline)) assert.deepEqual(liveBrain[key], values);
    }
  });

  await test("invalid checkpoint and plastic edge IDs are rejected before changing parameters in either wiring", async () => {
    source = fixture();
    await send({ type: "init", url: "edge validation fixture" });
    await send({ type: "learn:init", config, seed: 7 });
    liveBrain = liveLearner.brain;
    liveLearner.load([0], [.93]); // a rejected load must not reset existing learning
    for (const on of [false, true]) {
      await send({ type: "shuffle", on, seed: 11 });
      for (const edge of [-1, 4, .5, NaN, Infinity, Number.MAX_SAFE_INTEGER + 1, "0", undefined]) {
        for (const mutation of [{ type: "learned", on: true, edges: [edge], efficacy: [.27] },
          { type: "learn:init", config: { ...config, plastic: [edge] }, seed: 99 }]) {
          const before = parameters(), oldLearner = liveLearner;
          const rejected = await send(mutation);
          assert.equal(rejected.type, "error"); assert.match(rejected.reason, /^invalid_edge_id:/);
          assert.equal(liveLearner, oldLearner); assert.deepEqual(parameters(), before);
        }
      }
    }
    await send({ type: "shuffle", on: false }); // invalid configs did not replace the working learner configuration
    assert.deepEqual([...liveLearner.edges], [0, 1, 2, 3]);
    assert.equal(liveBrain.w[0], .93);
  });

  if (process.argv.includes("--payload")) {
    await test("actual payload observation and checked learning proposal use the real worker path", async () => {
      source = JSON.parse(readFileSync(new URL("../web/data/brain_full.json", import.meta.url), "utf8"));
      await send({ type: "init", url: "actual payload", generation: 5 });
      await send({ type: "learn:init", config: { ...config, plastic: { pre: ["kc"], post: ["mbon"] } }, seed: 7 });
      const senses = { ...source.lessons_setup.decision_senses, "orn:decaying_fruit:left": .9, "orn:decaying_fruit:right": .9,
        "orn:yeasty:left": .16, "orn:yeasty:right": .16 };
      const observed = await send({ type: "assist:observe", requestId: ++requestId, generation: 5, senses, steps: 1024, tolerance: 1e-6 });
      assert.equal(observed.converged, true); assert.ok(observed.readouts && observed.s);
      const p = await propose({ generation: 5, searchToken: "5:actual", senses, steps: 1024, tolerance: 1e-6 });
      phaseCheck(p);
      const before = parameters();
      await send({ type: "assist:accept", ...identity(p) }, 0);
      const r = await send({ type: "assist:reward", ...identity(p), reward: 1, done: true, neuralCredit: true });
      assert.equal(r.accepted, true); assert.ok(r.moved > 0); assert.notDeepEqual([...liveBrain.w], before.weights);
      console.log(`  payload ${source.n}: phases=${JSON.stringify(p.decision.solves)}, changed plastic classes=${r.moved}`);
    });
  }
} finally {
  globalThis.self = original.self; globalThis.fetch = original.fetch;
  ActorCriticLearner.prototype.stats = original.stats; SettlingBrain.prototype.settleControl = original.control;
}
console.log(`${assertions} assisted worker integration tests passed`);
