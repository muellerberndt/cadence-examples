// One continuing life, in the browser: the governor names the mode from the brain's own
// signals, the paw acts by the habit or by imagination, the belief assimilates the moment, the
// surprise is measured; learning observes the executed window when the governor says so. The
// same decision as cat.py's Life.decide, step for step: the readback port (the last surprise
// over its baseline, log-compressed; a running average of the same; the repair residual over
// its routine median; the last mode; a constant), the governor patch or the threshold rule or
// the two controls, ten candidate pushes held for the horizon in the belief's private
// imagination, the executed window kept for learning, the baseline's EMA, the dots' ledger.

import { BeliefPatch } from "./belief.js";
import { GovernorPatch, READBACK, MOTOR } from "./governor.js";
import { ACTIONS, CLIP, HABIT_KEYS, MASS, MODES, OUTPUTS, PREDICTED, READINGS, CONST, clip, habitAct, readingOf, targetOf, taskCost } from "./cat.js";

export const ARMS = ["patch", "threshold", "always_awake", "never_wakes"];

/** Python's round: half to even. The experiment rounds its integer genes with it. */
export function pyRound(x) {
  const f = Math.floor(x), d = x - f;
  if (d < 0.5) return f;
  if (d > 0.5) return f + 1;
  return f % 2 === 0 ? f : f + 1;
}
function median(values) {
  const v = Float64Array.from(values).sort();
  const n = v.length, h = n >> 1;
  return n % 2 ? v[h] : (v[h - 1] + v[h]) / 2;
}
const norm2 = (x, y) => Math.sqrt(x * x + y * y);

/** The multiply-accumulates of one moment of the patch: the port, the transition and gate,
 *  the store's code and read per iteration, the repair map, the readout. */
export function momentMacs(patch) {
  const za = patch.belief + patch.actions;
  const fi = patch.belief + patch.encoded + patch.belief + patch.recordWidth + 1;
  const store = (patch.encoded + patch.belief) * patch.records.cells + patch.records.active * patch.recordWidth;
  return patch.inputs * patch.encoded + 2 * patch.belief * za + patch.iterations * (store + patch.belief * fi) + store + patch.outputs * patch.belief;
}

/** Readings from random paws, a dot present in half of them: the starts of a habit refit. */
export function randomStarts(rng, n) {
  const out = [];
  for (let i = 0; i < n; i++) {
    const dot = rng.random() < 0.5 ? [rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95)] : null;
    const flow = [rng.normal(0.0, 0.015), rng.normal(0.0, 0.015)];
    const paw = [rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95)];
    out.push(readingOf(dot, flow, paw));
  }
  return out;
}

function imaginedHabitCost(patch, h, startsO, startsZ, steps, scale) {
  const n = startsO.length;
  const r = startsO.map((row) => Float64Array.from(row));
  let z = startsZ;
  let total = 0.0;
  for (let s = 0; s < steps; s++) {
    const a = r.map((row) => habitAct(h, row));
    const path = patch.imagine(a.map((row) => [row]), { state: z });
    for (let i = 0; i < n; i++) { const y = path.output[i][0]; for (let k = 0; k < OUTPUTS; k++) r[i][PREDICTED[k]] += y[k] * scale[k]; }
    z = path.finalState;
    let mean = 0.0; for (let i = 0; i < n; i++) mean += taskCost(r[i], a[i]); mean /= n;
    total += mean;
  }
  return [total / steps, n * steps];
}

/** A pattern search over the habit's three genes, every trial evaluated in imagination only,
 *  from a scratch copy of the patch. `starts` are the readings the search starts from. */
export function fitHabit(patch, starts, scale, start, { steps = 12 } = {}) {
  const scratch = new BeliefPatch.Scratch(patch);
  const n = starts.length;
  const z0 = scratch.assimilate(starts.map((o) => [o]), starts.map(() => [new Float64Array(ACTIONS)]), { state: starts.map(() => new Float64Array(patch.belief)) }).finalState;
  let h = {}; for (const k of HABIT_KEYS) h[k] = start[k];
  let [best, moments] = imaginedHabitCost(scratch, h, starts, z0, steps, scale);
  let step = 0.25;
  while (step > 4e-3) {
    let improved = false;
    for (const k of HABIT_KEYS) {
      for (const sign of [1.0, -1.0]) {
        const trial = { ...h };
        trial[k] = clip(h[k] + sign * step, 0.0, k !== "drift" ? 1.0 : 2.0);
        const [cost, m] = imaginedHabitCost(scratch, trial, starts, z0, steps, scale);
        moments += m;
        if (cost < best) { h = trial; best = cost; improved = true; }
      }
    }
    step = improved ? step : step / 2;
  }
  return { habit: h, cost: best, moments, n };
}

// A scratch copy of a patch: the same parameters and store, its own state (reset), as
// BeliefPatch.restore(patch.snapshot()) followed by reset() makes in the library.
BeliefPatch.Scratch = function (patch) {
  const copy = Object.create(BeliefPatch.prototype);
  Object.assign(copy, patch);
  copy.P = patch.parameters();
  copy.state = null;
  return copy;
};

export const MACHINERY = { window: 96, learn_rate: 0.1, passes: 6, horizon: 6, spread: 1.0, hold: true, rollback: true, habituate: 2.0, refit: true, min_cooldown: 32, write: false };

export class Life {
  /** `patch` a BeliefPatch, `world` a World, `genome` the governor's genes (the habit's and the
   *  baseline's among them), `scale` the motion units of the five predicted changes, `floors`
   *  the pretrained belief's routine surprise and residual; `arm` the governor; `rng` draws the
   *  starts of a habit refit (the library's Life draws them from numpy). */
  constructor(patch, world, genome, scale, floors, { arm = "patch", rng = null, machinery = null } = {}) {
    if (!ARMS.includes(arm)) throw new Error(`arm must be one of ${ARMS}`);
    this.patch = patch; this.world = world; this.g = { ...genome }; this.arm = arm;
    this.m = { ...MACHINERY, ...(machinery || {}) };
    this.scale = Float64Array.from(scale);
    this.rng = rng;
    this.habit = {}; for (const k of HABIT_KEYS) this.habit[k] = this.g[k];
    this.baseline0 = floors.median;
    this.residual0 = Math.max(floors.residual_median === undefined ? 1.0 : floors.residual_median, 1e-6);
    this.floor = this.g.floor * this.baseline0;
    this.baseline = Math.max(this.baseline0, this.floor);
    this.governor = arm === "patch" ? new GovernorPatch(this.g) : null;
    this.macs = momentMacs(patch);
    this.t = 0;
    this.mode = "habit";
    this.imagineLeft = 0;
    this.above = 0;
    this.recent = [];
    this.cooldownLeft = 0;
    this.sinceLearn = 1e9;
    this.slow = 0.0;
    this.surpriseLast = 0.0;
    this.residualLast = 0.0;
    this.readbackLast = new Float64Array(READBACK.length);
    this.o = []; this.a = []; this.y = []; this.states = [];
    this.learns = [];
    this.dots = []; // per dot: appearance, catch or miss, wake latency
    this.totals = { assimilate: 0, imagine: 0, observe: 0, refit: 0, governor: 0.0, imagine_calls: 0, learn_calls: 0, kept: 0, undone: 0, decisions: { habit: 0, imagine: 0, learn: 0 } };
    this.patch.reset();
  }

  /** The readback port: what both governors read. */
  readback() {
    const out = new Float64Array(READBACK.length);
    out[0] = Math.log1p(this.surpriseLast / this.baseline);
    out[1] = this.slow;
    out[2] = this.residualLast / this.residual0;
    out[3 + MODES[this.mode]] = 1.0;
    out[6] = 1.0;
    return out;
  }

  /** Ten candidate pushes (rest, the eight directions at `spread`, and the habit's own), each
   *  held for the horizon in the belief's private imagination; the imagined cost is the task's;
   *  the best first push is returned with the costs and the moments spent. */
  imagine(r, z) {
    const h = this.m.horizon, spread = this.m.spread;
    const candidates = CONST.directions.map((d) => [clip(spread * d[0], -1.0, 1.0), clip(spread * d[1], -1.0, 1.0)]);
    const own = habitAct(this.habit, r);
    candidates.push([clip(own[0], -1.0, 1.0), clip(own[1], -1.0, 1.0)]);
    const n = candidates.length;
    let zz = candidates.map(() => (z === null ? new Float64Array(this.patch.belief) : Float64Array.from(z)));
    const rr = candidates.map(() => Float64Array.from(r));
    let a = candidates.map((c) => c.slice());
    const cost = new Float64Array(n);
    for (let step = 0; step < h; step++) {
      const path = this.patch.imagine(a.map((row) => [row]), { state: zz });
      for (let i = 0; i < n; i++) { const y = path.output[i][0]; for (let k = 0; k < OUTPUTS; k++) rr[i][PREDICTED[k]] += y[k] * this.scale[k]; }
      zz = path.finalState;
      for (let i = 0; i < n; i++) cost[i] += taskCost(rr[i], a[i]);
      if (!this.m.hold) a = rr.map((row) => habitAct(this.habit, row));
    }
    let best = 0; for (let i = 1; i < n; i++) if (cost[i] < cost[best]) best = i;
    return { action: candidates[best], moments: n * h, costs: cost, candidates };
  }

  /** `observe` on the executed window from its boundary, for `passes`; the validity is whether
   *  the window's loss fell; an invalid update is undone. `starts` (optional) are the readings a
   *  refit starts from; without them they are drawn from this life's generator. */
  learn(starts = null) {
    const m = this.m;
    const w = Math.min(pyRound(m.window), this.a.length);
    const i0 = this.a.length - w, i1 = this.a.length;
    const o = [this.o.slice(i0, i1)], a = [this.a.slice(i0, i1)], y = [this.y.slice(i0, i1)];
    const snapshot = this.patch.snapshot();
    const before = this.patch.parameters();
    const boundary = this.states[i0] === null ? null : [this.states[i0]];
    let loss0 = null, lastLoss = null, moments = 0, passesKept = 0;
    for (let pass = 0; pass < pyRound(m.passes); pass++) {
      const keptSnapshot = this.patch.snapshot();
      const result = this.patch.observe(o, a, y, { rate: m.learn_rate, state: boundary });
      moments += 3 * w;
      if (loss0 === null) loss0 = result.initialLoss;
      if (!result.updated) break;
      if (lastLoss !== null && result.initialLoss !== null && result.initialLoss > lastLoss) { this.patch.restore(keptSnapshot); break; }
      lastLoss = result.initialLoss;
      passesKept += 1;
    }
    const check = this.patch.observe(o, a, y, { rate: 0.0, state: boundary });
    moments += w;
    const loss1 = check.initialLoss;
    const after = this.patch.parameters();
    let size = 0.0; for (const k in after) for (let i = 0; i < after[k].length; i++) size += (after[k][i] - before[k][i]) ** 2;
    size = Math.sqrt(size);
    const valid = loss0 !== null && loss1 !== null && loss1 < 0.9 * loss0;
    const entry = { t: this.t, window: w, loss_before: loss0, loss_after: loss1, update_size: size, valid, kept: valid || !m.rollback, passes_kept: passesKept, refit: false, habit_before: { ...this.habit }, moments };
    if (!valid && m.rollback) {
      this.patch.restore(snapshot);
      this.totals.undone += 1;
      this.baseline = Math.min(this.baseline * m.habituate, this.recent.length ? median(this.recent) : this.baseline);
    } else {
      this.totals.kept += 1;
      const errs = new Float64Array(w);
      for (let k = 0; k < w; k++) { const out = check.path.output[0][k], tg = y[0][k]; let s = 0.0; for (let c = 0; c < OUTPUTS; c++) s += (out[c] - tg[c]) ** 2; errs[k] = s / OUTPUTS; }
      this.baseline = Math.max(this.floor, median(errs));
      if (m.refit) {
        const st = starts || randomStarts(this.rng, 8);
        const fit = fitHabit(this.patch, st, this.scale, this.habit);
        this.habit = fit.habit;
        Object.assign(entry, { refit: true, habit_after: { ...this.habit }, imagined_cost: fit.cost, refit_moments: fit.moments });
        this.totals.refit += fit.moments;
      }
    }
    this.totals.observe += moments;
    this.totals.learn_calls += 1;
    this.above = 0;
    this.recent = [];
    this.slow = 0.0;
    this.cooldownLeft = pyRound(this.g.cooldown === undefined ? 0 : this.g.cooldown);
    this.sinceLearn = 0;
    this.learns.push(entry);
    return entry;
  }

  /** One decision. `starts` are passed through to a refit (the parity test supplies them). */
  decide({ wantCode = false, starts = null } = {}) {
    const g = this.g, m = this.m, r = this.world.reading();
    const z = this.patch.state === null ? null : Float64Array.from(this.patch.state[0]);
    const started = performance.now();
    const rb = this.readback();
    this.readbackLast = rb;
    let governorSteps = 0, governorActivation = null;
    const canLearn = this.sinceLearn >= pyRound(m.min_cooldown) && this.a.length >= 32 && this.cooldownLeft === 0;
    let learnNow = false, imagineNow = false, want = null;
    if (this.arm === "patch") {
      const settled = this.governor.settle(rb);
      want = settled.mode; governorSteps = settled.steps; governorActivation = settled.activation;
      learnNow = want === "learn" && canLearn;
      imagineNow = want === "imagine" || want === "learn";
    } else if (this.arm === "threshold" || this.arm === "always_awake") {
      const persist = pyRound(g.persist);
      let share = 0.0;
      if (this.recent.length) { let c = 0; for (const s of this.recent) if (s > g.k_learn * this.baseline) c++; share = c / this.recent.length; }
      learnNow = this.recent.length >= persist && share >= g.persist_share && canLearn;
      imagineNow = this.imagineLeft > 0 || this.arm === "always_awake";
    }
    const entry = learnNow ? this.learn(starts) : null;
    if (entry !== null) imagineNow = true;
    let moments = 0, action, mode, imagined = null;
    if (imagineNow) {
      imagined = this.imagine(r, z);
      action = imagined.action; moments = imagined.moments;
      this.imagineLeft = Math.max(0, this.imagineLeft - 1);
      mode = "imagine";
    } else { action = habitAct(this.habit, r); mode = "habit"; }
    if (entry !== null) mode = "learn";
    const path = this.patch.assimilate([[r]], [[action]], { wantCode });
    const expected = path.output[0][0], residual = path.residual[0][0];
    const [r2, flags] = this.world.step(action);
    const y = targetOf(r, r2);
    for (let k = 0; k < OUTPUTS; k++) y[k] = clip(y[k] / this.scale[k], -CLIP, CLIP);
    let surprise = 0.0; for (let k = 0; k < OUTPUTS; k++) surprise += (expected[k] - y[k]) ** 2; surprise /= OUTPUTS;
    const ms = performance.now() - started;
    // the executed window
    this.o.push(r); this.a.push(Float64Array.from(action)); this.y.push(y); this.states.push(z);
    if (this.a.length > 512) { this.o.shift(); this.a.shift(); this.y.shift(); this.states.shift(); }
    // the threshold rule's bookkeeping (the always-awake control learns by it too)
    let threshold;
    if (this.arm === "threshold" || this.arm === "always_awake") {
      const spike = surprise > g.k_imagine * this.baseline && (this.imagineLeft === 0 || !!g.renew);
      if (spike) this.imagineLeft = Math.max(this.imagineLeft, pyRound(g.imagine_budget));
      this.recent.push(surprise);
      const keep = pyRound(g.persist);
      if (this.recent.length > keep) this.recent.splice(0, this.recent.length - keep);
      threshold = g.k_learn * this.baseline;
    } else {
      this.recent.push(surprise);
      if (this.recent.length > 96) this.recent.splice(0, this.recent.length - 96);
      threshold = 3.0 * this.baseline;
    }
    if (surprise > threshold) this.above += 1;
    else { this.above = 0; this.baseline = Math.max(this.floor, this.baseline + g.baseline_rate * (surprise - this.baseline)); }
    const rate = g.slow_rate === undefined ? 0.02 : g.slow_rate;
    this.slow = this.slow + rate * (Math.log1p(surprise / this.baseline) - this.slow);
    if (this.cooldownLeft > 0) this.cooldownLeft -= 1;
    this.sinceLearn += 1;
    this.surpriseLast = surprise; this.residualLast = residual; this.mode = mode;
    const govMoments = governorSteps * (this.governor ? this.governor.synapses : 0) / this.macs;
    this.totals.assimilate += 1;
    this.totals.imagine += moments;
    this.totals.imagine_calls += moments > 0 ? 1 : 0;
    this.totals.governor += govMoments;
    this.totals.decisions[mode] += 1;
    // the dots' ledger
    if (flags.appear) this.dots.push({ appear: this.t, wake: null, end: null, outcome: null });
    if (this.dots.length && this.dots[this.dots.length - 1].end === null) {
      const d = this.dots[this.dots.length - 1];
      if (mode !== "habit" && d.wake === null) d.wake = this.t;
      if (flags.catch) { d.end = this.t; d.outcome = "catch"; }
      else if (flags.miss) { d.end = this.t; d.outcome = "miss"; }
      else if (flags.vanish) { d.end = this.t; d.outcome = "vanished"; }
    }
    this.t += 1;
    return { mode, want, action, residual, surprise, baseline: this.baseline, expected, reading: r, next: r2, flags, learn: entry, ms, governorSteps, governorActivation, readback: rb, moments: 1 + moments + govMoments, path, imagined, z, target: y };
  }

  /** The totals as moments: the whole life's cost and its cost per decision. */
  compute() {
    const t = this.totals;
    const total = t.assimilate + t.imagine + t.observe + t.refit + t.governor;
    const n = Math.max(1, t.decisions.habit + t.decisions.imagine + t.decisions.learn);
    return { total_moments: total, moments_per_decision: total / n, decisions_by_mode: { ...t.decisions }, moment_macs: this.macs };
  }
}

export { READINGS, MASS, READBACK, MOTOR };
