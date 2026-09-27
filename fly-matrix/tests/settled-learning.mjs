// Opt-in local learning must qualify every equilibrium and commit atomically.
// Small directed fixtures exercise the actual solver, never a mocked positive solve.
import assert from "node:assert/strict";
import { SettlingBrain } from "../web/brain.js";
import { ActorCriticLearner } from "../web/learner.js";

const encode = a => Buffer.from(a.buffer, a.byteOffset, a.byteLength).toString("base64");
function payload({ dt = 0.4, slope = 1, threshold = 0, leak = 1 } = {}) {
  return { n: 3, edges: 2, model: { dt, slope, threshold, leak, gain: 1, stimulus_amplitude: 1 },
    arrays: { row_ptr: encode(new Int32Array([0, 0, 1, 2])), pre: encode(new Int32Array([0, 0])),
      weight: encode(new Float64Array([0.4, -0.2])), efficacy: encode(new Float64Array([0.4, -0.2])),
      bias: encode(new Float64Array([0, 0.2, 0.4])) },
    populations: { input: [0], output: [1, 2], critic: [0] } };
}
function fixture(config = {}, model = {}) {
  const brain = new SettlingBrain(payload(model)); brain.stimulate("input", 1);
  const learner = new ActorCriticLearner(brain, { outputs: "output", critic: "critic", plastic: "all",
    plasticNeurons: [1], beta: 0.1, temperature: 0.3, eta: 0.5, etaBias: 0.1, ...config });
  return { brain, learner };
}
const snapshot = (b, l) => ({ v: [...b.v], s: [...b.s], steps: b.steps, drive: [...b.drive], bias: [...b.bias], w: [...b.w],
  efficacy: [...l.efficacy], trace: [...l.trace], traceBias: [...l.traceBias], traceCritic: [...l.traceCritic],
  critic: [...l.wCritic], bCritic: l.bCritic, updates: l.updates });
const close = (a, b, tolerance = 2e-15) => assert.ok(Math.abs(a - b) <= tolerance, `${a} differs from ${b}`);
// Independent dense evaluation; tanh(v/2) is this fixture's smooth activation.
function residual(v, b, beta = 0, target = [1, 0], temperature = 0.3) {
  const s = [...v].map(x => Math.tanh(x / 2));
  const matrix = [[0, 0, 0], [b.w[0], 0, 0], [b.w[1], 0, 0]];
  const z = [s[1] / temperature, s[2] / temperature], e = z.map(x => Math.exp(x - Math.max(...z)));
  const p = e.map(x => x / (e[0] + e[1]));
  return Math.max(...v.map((x, i) => Math.abs(matrix[i].reduce((a, w, j) => a + w * s[j], 0)
    + (i ? beta * (target[i - 1] - p[i - 1]) : 0) + b.drive[i] + b.bias[i] - x)));
}

{
  const { brain: b, learner: l } = fixture();
  assert.equal(b.settleControl(128, 1e-10).converged, true);
  const before = snapshot(b, l);
  for (const beta of [0.1, -0.1]) {
    const r = b.settleNudgedResidual([1, 2], [1, 0], beta, 0.3, 128, 1e-10);
    assert.equal(r.converged, true); assert.ok(r.residual <= 1e-10);
    close(r.residual, residual(r.v, b, beta));
    assert.ok(residual(r.v, b, 0) > 0.01, "a free-only residual would misclassify this phase");
    r.s.forEach((s, i) => close(s, Math.tanh(r.v[i] / 2)));
    const old = b.settleNudged([1, 2], [1, 0], beta, 0.3, r.iterations, null);
    old.s.forEach((s, i) => close(s, r.s[i]));
    assert.deepEqual(snapshot(b, l), before, "copied nudged solve mutated live state");
  }
}

// Legacy saturation and a tiny Euler step cannot qualify a potential equation.
{
  const { brain: b } = fixture({}, { dt: 0.2, slope: 4, threshold: 1.5, leak: 0 });
  b.v.fill(100); b.s.fill(1);
  assert.equal(b.settleNudged([1, 2], [1, 0], 0.1, 0.3, 20, 1e-6).taken, 1);
  const r = b.settleNudgedResidual([1, 2], [1, 0], 0.1, 0.3, 1, 1e-6);
  assert.equal(r.converged, false); assert.ok(r.residual > 70);
  const tiny = fixture({}, { dt: 1e-12 }).brain.settleNudgedResidual([1, 2], [1, 0], 0.1, 0.3, 1, 1e-6);
  assert.equal(tiny.converged, false); assert.ok(tiny.residual > 0.9);
}

for (const args of [
  [[1, 1], [1, 0], 0.1, 0.3], [[1, 3], [1, 0], 0.1, 0.3], [[1, 2], [1, 1], 0.1, 0.3],
  [[1, 2], [NaN, 0], 0.1, 0.3], [[1, 2], [1, 0], NaN, 0.3], [[1, 2], [1, 0], 0.1, 0],
]) {
  const { brain: b, learner: l } = fixture(), before = snapshot(b, l);
  const r = b.settleNudgedResidual(...args);
  assert.equal(r.converged, false); assert.equal(r.reason, "invalid_nudge"); assert.equal(r.iterations, 0);
  assert.deepEqual(snapshot(b, l), before);
}
for (const field of ["v", "s", "bias", "drive", "w"]) {
  const { brain: b } = fixture(); b[field][0] = NaN;
  const r = b.settleNudgedResidual([1, 2], [1, 0], 0.1, 0.3);
  assert.equal(r.converged, false); assert.equal(r.residual, null);
}

// Real free, plus, and minus caps all preserve eligibility and parameters, without crediting
// a previous action. Minus is slower than plus in this fixture; 40 steps separate them.
for (const phase of ["free", "plus", "minus"]) {
  const { brain: b, learner: l } = fixture();
  if (phase !== "free") assert.equal(b.settleControl(128, 1e-10).converged, true);
  l.trace.set([0.25, -0.5]); l.traceBias[0] = 0.2; l.traceCritic.set([0.3, 0.4]);
  l.pending = { value: 2, settled: true };
  const before = snapshot(b, l), maxSteps = phase === "minus" ? 40 : 1;
  const r = l.actSettled({ u: 0, maxSteps, tolerance: 1e-10 });
  assert.equal(r.accepted, false); assert.equal(r.reason, `${phase}_phase_failed`);
  if (phase === "minus") assert.equal(r.solves.plus.converged, true);
  assert.equal(r.solves[phase].converged, false);
  assert.ok(!("action" in r)); assert.ok(!("p" in r));
  assert.deepEqual(snapshot(b, l), before); assert.equal(l.pending, null);
  assert.equal(l.learnSettled(1, true).reason, "no_qualified_decision");
}

// Accepted phases drive the local contrast and the declared efficacy update exactly.
{
  const { brain: b, learner: l } = fixture();
  const r = l.actSettled({ u: 0, maxSteps: 128, tolerance: 1e-10 });
  assert.equal(r.accepted, true); assert.equal(r.draw, 0); assert.equal(r.action, 0);
  assert.ok(JSON.stringify(r).length < 1500, "public decision leaked per-neuron arrays");
  for (const p of Object.values(r.solves)) assert.ok(p.converged && p.residual <= 1e-10);
  close(r.solves.free.residual, residual(b.v, b));
  close(r.solves.plus.residual, residual(l.pending.plusV, b, 0.1));
  close(r.solves.minus.residual, residual(l.pending.minusV, b, -0.1));
  const initial = [...l.efficacy], { plus, minus } = l.pending;
  for (let k = 0; k < 2; k++) close(l.trace[k], (plus[0] * plus[k + 1] - minus[0] * minus[k + 1]) / 0.2);
  const expected = initial.map((x, k) => x + 0.5 * l.trace[k]);
  const learned = l.learnSettled(1, true);
  assert.equal(learned.accepted, true); assert.equal(learned.lesson.moved, 2);
  expected.forEach((x, k) => { close(l.efficacy[k], x); close(b.w[k], x); });
  assert.ok(l.trace.every(x => x === 0)); assert.equal(l.updates, 1); assert.equal(l.pending, null);
}

// A frozen arm receives the same valid phases and reward but cannot move parameters.
{
  const { brain: b, learner: l } = fixture({ eta: 0, etaBias: 0, etaCritic: 0 });
  assert.equal(l.actSettled({ u: 0, maxSteps: 128, tolerance: 1e-10 }).accepted, true);
  const before = snapshot(b, l); assert.equal(l.learnSettled(1, true).accepted, true);
  assert.deepEqual([...b.w], before.w); assert.deepEqual([...b.bias], before.bias);
  assert.deepEqual([...l.wCritic], before.critic); assert.equal(l.bCritic, before.bCritic);
}

// Nonterminal bootstrapping needs a qualified next state; terminal and nonfinite rewards
// are distinct cases. Overflow must be caught before the first weight is changed.
for (const failure of ["next_phase", "reward", "overflow"]) {
  const { brain: b, learner: l } = fixture();
  assert.equal(l.actSettled({ u: 0, maxSteps: 128, tolerance: 1e-10 }).accepted, true);
  if (failure === "next_phase") b.stimulate("input", 3);
  if (failure === "overflow") l.traceCritic[0] = 1e200;
  const before = snapshot(b, l);
  const r = l.learnSettled(failure === "reward" ? NaN : 1, failure !== "next_phase", { maxSteps: 1, tolerance: 1e-10 });
  assert.equal(r.accepted, false);
  assert.equal(r.reason, { next_phase: "next_phase_failed", reward: "invalid_reward", overflow: "nonfinite_update" }[failure]);
  assert.deepEqual(snapshot(b, l), before); assert.equal(l.pending, null);
}

{
  const { brain: b, learner: l } = fixture();
  assert.equal(l.actSettled({ greedy: true, maxSteps: 128, tolerance: 1e-10 }).accepted, true);
  assert.equal(l.pending, null); assert.ok(l.trace.every(x => x === 0));
  // Legacy eligibility cannot be mislabeled as checked.
  l.act(false, 0);
  assert.equal(l.learnSettled(1, true).reason, "no_qualified_decision");
  const before = snapshot(b, l);
  assert.equal(l.actSettled({ u: 1 }).accepted, false);
  assert.deepEqual(snapshot(b, l), before);
}
console.log("settled learning checks passed: independent free/nudged residuals, copied-state isolation, all-phase atomic rejection, local contrast/update, frozen control, checked next state and finite updates");
