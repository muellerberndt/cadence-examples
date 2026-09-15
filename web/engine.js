// The experience agent in JavaScript: a port of agent/brain.py for one
// stream, built from the snapshot that agent/web.py exports
// (format "cadence-experience-web/1"). The same event transaction as Agent.step_batch:
// validate, free phase, score, credit, world repair (with the replay ring), context, decide,
// predict, commit. The numerics follow the cadence CPU backend that the Python agent runs
// on: the fused settling kernel (block transport in the layout's pair order, the potential
// update, the nudge, the residual check of `equilibrate`), the centred contrast of
// `Learner.contrast`, `Learner.apply` (plastic masks, synapse rate, reciprocal pair
// averaging, the efficacy cap), the momentum with bias correction of `Agent._adaptive`,
// the eligibility credit of `Agent._credit` and the Trace of cadence.stream. Float64
// everywhere. Where numpy reduces a small array (a softmax, the critic norm) the pairwise
// summation of numpy is reproduced; BLAS products are summed in index order, so the
// remaining differences are rounding-level (see parity.mjs).
//
// A snapshot carries the pending decision and the ring's warm potentials (`ring.v`, `ring.a`),
// so the page continues the exact life the Python agent exported.
//
// A snapshot whose config declares `records` carries the records world head of brain.py
// (`RecordsHead`): the expansion is regenerated from the seed with the same Mulberry32
// draws and Box-Muller pairs, the habituation mean and every records table are loaded from
// the snapshot (float32 tables widen exactly; `options.records` supplies the float64 sidecar
// the fixture writes), and the transaction takes the records path: the executed reading is
// coded in `_decide` (the mean adapts there and only there), the outcome is written into
// the pending code's records in `_repair` (no ring, no settle), and imagination is a record
// read (no settle). Every reading yields two codes, plain (the consequence records) and
// valued (each pathway divided by its running norm; the reward and terminal records).
// `options.onRecords` reports every code, imagined batch and write.
//
// No dependencies. ES module: import { ExperienceAgent, Mulberry32 } from "./engine.js".

export const ENGINE_VERSION = "cadence-experience-engine/1";
export const SNAPSHOT_FORMAT = "cadence-experience-web/1";
export const RECORDS_FORMAT = "cadence-experience-web-records/1"; // the float64 sidecar of the records tables (web.records_sidecar)
export const CONTROLLERS = ["actor", "planner", "exploration", "demonstration"];
export const SCALE_CAP = 8.0; // cadence.learning.SCALE_CAP: the magnitude a synapse may not exceed

const DTYPES = { f4: Float32Array, f8: Float64Array, u4: Uint32Array, i4: Int32Array, u2: Uint16Array, i2: Int16Array, u1: Uint8Array, i1: Int8Array };

/** Whether a snapshot carries what a continued life needs: the pending decision (or its absence, declared) and the ring's warm potentials. */
export function snapshotCarriesLife(snapshot) {
  const ring = snapshot.ring;
  const ringComplete = !ring || !ring.drive || (!!ring.v && !!ring.a);
  return "pending" in snapshot && ringComplete;
}

/** Decode a `{dtype, shape, b64}` array (little-endian bytes) into a typed array; the same as brain_scan.js decodeArray. */
export function unpackArray(spec) {
  const T = DTYPES[spec.dtype];
  if (!T) throw Error(`Unsupported array dtype ${spec.dtype}`);
  const bytes = Uint8Array.from(atob(spec.b64), (c) => c.charCodeAt(0));
  const copy = new Uint8Array(bytes.length);
  copy.set(bytes);
  return new T(copy.buffer, 0, copy.byteLength / T.BYTES_PER_ELEMENT);
}

/** A `{dtype, shape, b64}` array as Float64Array (a float32 export widens exactly). */
export function unpackF64(spec) {
  const a = unpackArray(spec);
  return a instanceof Float64Array ? a : Float64Array.from(a);
}

/** The 32-bit generator of brain_scan.js (`mulberry`) and of brain.py (`Mulberry32`), bit-identical. */
export class Mulberry32 {
  constructor(seed) { this.state = Number(seed) >>> 0; }
  random() {
    const a = (this.state + 0x6d2b79f5) >>> 0;
    this.state = a;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  choice(probabilities) {
    const u = this.random();
    let total = 0.0;
    for (let k = 0; k < probabilities.length; k++) {
      total += Number(probabilities[k]);
      if (u < total) return k;
    }
    return probabilities.length - 1;
  }
  /** `n` draws at once (brain.py `batch`): the state advances by the constant, so the draw at index j is the hash of (state + j * 0x6D2B79F5) mod 2^32, bit-identical to `n` calls of `random`. */
  batch(n) {
    const out = new Float64Array(n);
    let a = this.state;
    for (let j = 0; j < n; j++) {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      out[j] = ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }
    this.state = a;
    return out;
  }
  /** Standard normals by Box-Muller from `batch` draws (brain.py `normals`): pairs = ceil(n / 2), u1 the first half (floored at 1e-12), u2 the second, r cos then r sin. */
  normals(n) {
    const pairs = Math.floor((n + 1) / 2), u = this.batch(2 * pairs), out = new Float64Array(2 * pairs);
    for (let k = 0; k < pairs; k++) {
      const r = Math.sqrt(-2.0 * Math.log(Math.max(u[k], 1e-12))), angle = 2.0 * Math.PI * u[pairs + k];
      out[k] = r * Math.cos(angle);
      out[pairs + k] = r * Math.sin(angle);
    }
    return out.length === n ? out : out.slice(0, n);
  }
}

/** numpy's pairwise summation of a contiguous float64 reduction (what `.sum()` and `.mean()` do). */
export function npSum(values, start = 0, count = values.length - start) {
  if (count < 8) {
    let res = 0.0;
    for (let i = 0; i < count; i++) res += values[start + i];
    return res;
  }
  if (count <= 128) {
    const r = [values[start], values[start + 1], values[start + 2], values[start + 3], values[start + 4], values[start + 5], values[start + 6], values[start + 7]];
    let i = 8;
    for (; i < count - (count % 8); i += 8) {
      r[0] += values[start + i]; r[1] += values[start + i + 1]; r[2] += values[start + i + 2]; r[3] += values[start + i + 3];
      r[4] += values[start + i + 4]; r[5] += values[start + i + 5]; r[6] += values[start + i + 6]; r[7] += values[start + i + 7];
    }
    let res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < count; i++) res += values[start + i];
    return res;
  }
  let n2 = Math.floor(count / 2);
  n2 -= n2 % 8;
  return npSum(values, start, n2) + npSum(values, start + n2, count - n2);
}

export function npMean(values) { return values.length ? npSum(values) / values.length : NaN; }

const clip = (x, lo, hi) => (x < lo ? lo : x > hi ? hi : x);
const isContiguous = (index) => index.length > 0 && index.every((v, k) => v === index[0] + k);
const zeros = (size) => new Float64Array(size);

function fieldFrom(d) {
  const f = { name: d.name, kind: d.kind, width: d.width | 0, lo: d.lo ?? 0.0, hi: d.hi ?? 1.0, source: d.source ?? "" };
  if (f.kind !== "categorical" && f.kind !== "continuous") throw Error("kind must be categorical or continuous");
  if (!(f.width >= 1)) throw Error("width must be a positive integer");
  if (!Number.isFinite(f.lo) || !Number.isFinite(f.hi) || f.hi <= f.lo) throw Error("continuous bounds must be finite with hi above lo");
  return f;
}

/** The agent configuration with the Python defaults filled in. */
export function normalizeConfig(c) {
  const g = c.graph;
  const graph = {
    observation: g.observation.map(fieldFrom), prediction: g.prediction.map(fieldFrom), action_fields: g.action_fields.map((k) => k | 0),
    goal: g.goal ?? 0, perceptual: g.perceptual ?? 0, workspace: g.workspace ?? 64, dynamics: g.dynamics ?? 32, episodic: g.episodic ?? 0, intention: g.intention ?? 0,
    missing_flags: g.missing_flags ?? true, flags_per_field: g.flags_per_field ?? false, density: g.density ?? 1.0, scale: g.scale ?? 1.0, source_scale: g.source_scale ?? 1.0,
    sensory_to_dynamics: g.sensory_to_dynamics ?? true, bias: g.bias ?? 0.25, input_gain: g.input_gain ?? 2.0, readout_init: g.readout_init ?? 0.0, dt: g.dt ?? 0.5,
  };
  const w = c.world || {};
  const world = {
    beta: w.beta ?? 0.05, eta: w.eta ?? 0.02, eta_bias: w.eta_bias ?? 0.002, momentum: w.momentum ?? 0.0, normalize: w.normalize ?? 0.0, temperature: w.temperature ?? 0.2,
    label_smoothing: w.label_smoothing ?? 0.05, reject_unconverged: w.reject_unconverged ?? true, max_phase_gap: w.max_phase_gap ?? 0.3, step_clip: w.step_clip ?? 0.1,
    fan_in_rate: w.fan_in_rate ?? true, free_steps: w.free_steps ?? 400, nudged_steps: w.nudged_steps ?? 400, chunk: w.chunk ?? 16, tolerance: w.tolerance ?? 1e-7,
    imagine_tolerance: w.imagine_tolerance ?? null, residual_tolerance: w.residual_tolerance ?? 1e-6,
  };
  const a = c.actor || {};
  const actor = {
    beta: a.beta ?? 0.05, eta: a.eta ?? 0.5, eta_bias: a.eta_bias ?? 0.05, eta_critic: a.eta_critic ?? 0.05, temperature: a.temperature ?? 0.2, gamma: a.gamma ?? 0.99,
    lam: a.lam ?? 0.9, delta_cap: a.delta_cap ?? 5.0, step_clip: a.step_clip ?? 0.1, epsilon: a.epsilon ?? 0.0, critic_normalize: a.critic_normalize ?? true,
  };
  const r = c.replay, rc = c.records;
  return {
    graph, world, actor,
    episodic: c.episodic == null ? null : { key: c.episodic.key ?? "observation", decay: c.episodic.decay ?? 1.0, rate: c.episodic.rate ?? 1.0, amplitude: c.episodic.amplitude ?? 1.0, reset_on_episode: c.episodic.reset_on_episode ?? true },
    replay: r == null ? null : { capacity: r.capacity ?? 4096, per_transition: r.per_transition ?? 1 },
    records: rc == null ? null : { granules: rc.granules ?? 8000, active: rc.active ?? 40, rate: rc.rate ?? 0.2, reward_rate: rc.reward_rate ?? 1.0, habituation: rc.habituation ?? 1e-5, bias_scale: rc.bias_scale ?? 0.3, context: rc.context ?? false, normalize_blocks: rc.normalize_blocks ?? true, block_rate: rc.block_rate ?? 0.002, task_sets: rc.task_sets ?? false },
    context_decay: c.context_decay ?? 0.8, context_amplitude: c.context_amplitude ?? 1.0, controller: c.controller ?? "actor",
    learning: c.learning ?? true, world_learning: c.world_learning ?? true, motor_learning: c.motor_learning ?? true,
  };
}

/** A moment as the JSON of stream.jsonl (lists), made into typed arrays with the Moment invariants. */
export function normalizeMoment(m) {
  if (m && m.__moment) return m;
  const observation = {}, observed = {};
  for (const [name, value] of Object.entries(m.observation || {})) {
    const arr = Float64Array.from(value, Number);
    if (!arr.every(Number.isFinite)) throw Error("records must be finite");
    const flag = m.observed && m.observed[name] != null ? Uint8Array.from(m.observed[name], (x) => (x ? 1 : 0)) : new Uint8Array(arr.length).fill(1);
    if (flag.length !== arr.length) throw Error(`observed flags of '${name}' must match the observation's shape`);
    observation[name] = arr;
    observed[name] = flag;
  }
  for (const name of Object.keys(m.observed || {})) if (!(name in observation)) throw Error("observed flags name unknown observation fields");
  const feedback_for = m.feedback_for ?? null, executed = m.executed ?? null;
  if ((feedback_for === null) !== (executed === null)) throw Error("feedback names both the decision and the executed action, or neither");
  const reward_known = !!m.reward_known;
  if (feedback_for === null && reward_known) throw Error("a known reward belongs to an executed decision");
  if (m.terminated && m.truncated) throw Error("an event is terminated or truncated, not both");
  const mask = Uint8Array.from(m.action_mask, (x) => (x ? 1 : 0));
  if (!mask.length) throw Error("action_mask must be a nonempty boolean vector");
  let final_observation = null;
  if (m.final_observation != null) { final_observation = {}; for (const [k, v] of Object.entries(m.final_observation)) final_observation[k] = Float64Array.from(v, Number); }
  const out = {
    __moment: true, life_id: String(m.life_id), episode_id: m.episode_id | 0, event_id: m.event_id | 0, tick: m.tick | 0, dt: m.dt ?? 1.0,
    observation, observed, action_mask: mask, feedback_for, executed, reward: Number(m.reward ?? 0.0), reward_known, terminated: !!m.terminated, truncated: !!m.truncated,
    final_observation, goal: m.goal == null ? null : Float64Array.from(m.goal, Number), replay_of: m.replay_of ?? null,
  };
  if (!Number.isFinite(out.reward)) throw Error("reward must be finite");
  out.has_feedback = out.feedback_for !== null;
  out.any_legal = mask.some((x) => x === 1);
  return out;
}

/** Decision.to_dict(): the JSON form a decision takes in stream.jsonl. */
export function decisionToDict(d) {
  const prediction = {};
  for (const [k, v] of Object.entries(d.prediction)) prediction[k] = Array.from(v, Number);
  return {
    decision_id: d.decision_id, event_id: d.event_id, stream: d.stream, parameter_version: d.parameter_version, context_version: d.context_version,
    action: d.action, controller: d.controller, log_probability: d.log_probability, probabilities: Array.from(d.probabilities, Number), prediction,
    uncertainty: { ...d.uncertainty }, budget: Object.fromEntries(Object.entries(d.budget).map(([k, v]) => [k, v | 0])), intention: d.intention ?? null,
  };
}

/** A settled state of a batch: potentials, activations and adaptation as flat (batch, n) arrays. */
export function brainState(v, s, a, batch, steps) { return { v, s, a, batch, steps }; }

function stateRows(state, n, sel) {
  const v = zeros(sel.length * n), s = state.s ? zeros(sel.length * n) : null, a = zeros(sel.length * n);
  sel.forEach((r, k) => { v.set(state.v.subarray(r * n, (r + 1) * n), k * n); a.set(state.a.subarray(r * n, (r + 1) * n), k * n); if (s) s.set(state.s.subarray(r * n, (r + 1) * n), k * n); });
  return brainState(v, s, a, sel.length, state.steps);
}

function driveRows(drive, n, sel) {
  const out = zeros(sel.length * n);
  sel.forEach((r, k) => out.set(drive.subarray(r * n, (r + 1) * n), k * n));
  return out;
}

/** cadence.stream.Trace at focus 0: the carried context of every stream, entering the next settle as a stimulus. */
class Trace {
  constructor(n, source, target, decay, amplitude) {
    if (!(decay >= 0 && decay < 1)) throw Error("decay lies in [0, 1)");
    this.n = n; this.hidden = source; this.glow = target; this.decay = decay; this.amplitude = amplitude;
    if (this.glow.length !== this.hidden.length) throw Error("one context neuron per workspace neuron");
    this.width = source.length;
    this.reset(0);
  }
  get streams() { return this.trace.length / Math.max(1, this.width); }
  reset(batch, rows = null) {
    if (rows === null || this.streams !== batch) {
      this.trace = zeros(batch * this.width); this.last = zeros(batch * this.width); this.cold = new Uint8Array(batch).fill(1);
    } else for (const r of rows) { this.trace.fill(0, r * this.width, (r + 1) * this.width); this.last.fill(0, r * this.width, (r + 1) * this.width); this.cold[r] = 1; }
  }
  /** Read the trace into a copy of the drive; another batch size reads zero context. */
  stimulate(drive, batch) {
    const out = Float64Array.from(drive), n = this.n, same = this.streams === batch;
    for (let b = 0; b < batch; b++) for (let j = 0; j < this.width; j++) out[b * n + this.glow[j]] = same ? this.amplitude * this.trace[b * this.width + j] : 0.0;
    return out;
  }
  /** After the free phase: the trace decays toward the source neurons' activation. */
  update(state) {
    const batch = state.batch, n = this.n;
    if (this.streams !== batch) this.reset(batch);
    const h = zeros(batch * this.width);
    for (let b = 0; b < batch; b++) for (let j = 0; j < this.width; j++) h[b * this.width + j] = state.s[b * n + this.hidden[j]];
    for (let k = 0; k < h.length; k++) this.trace[k] = this.decay * this.trace[k] + (1.0 - this.decay) * h[k];
    this.last = h;
    this.cold.fill(0);
  }
}

/** The ledger of every phase: counts, steps, worst residual and unconverged phases. */
export class Ledger {
  constructor() { this.phases = {}; this.steps = {}; this.max_residual = {}; this.unconverged = {}; this.rejected_updates = 0; this.clipped_deltas = 0; this.deltas = 0; this.real_transitions = 0; this.replay_writes = 0; this.episodic_writes = 0; this.record_writes = 0; this.imagined = 0; }
  record(phase, steps, residual, tolerance) {
    this.phases[phase] = (this.phases[phase] || 0) + 1;
    this.steps[phase] = (this.steps[phase] || 0) + (steps | 0);
    this.max_residual[phase] = Math.max(this.max_residual[phase] || 0.0, residual);
    if (!(residual <= tolerance)) this.unconverged[phase] = (this.unconverged[phase] || 0) + 1;
  }
  toDict() {
    const mean_steps = {};
    for (const k of Object.keys(this.phases)) mean_steps[k] = this.steps[k] / Math.max(1, this.phases[k]);
    return { phases: { ...this.phases }, mean_steps, max_residual: { ...this.max_residual }, unconverged_phases: { ...this.unconverged }, rejected_updates: this.rejected_updates, clipped_deltas: this.clipped_deltas, deltas: this.deltas, real_transitions: this.real_transitions, replay_writes: this.replay_writes, episodic_writes: this.episodic_writes, record_writes: this.record_writes, imagined: this.imagined };
  }
  static fromDict(d) {
    const out = new Ledger();
    if (!d) return out;
    for (const k of Object.keys(d.phases || {})) { out.phases[k] = d.phases[k] | 0; out.steps[k] = Math.round((d.mean_steps?.[k] ?? 0) * out.phases[k]); }
    out.max_residual = { ...(d.max_residual || {}) };
    for (const k of Object.keys(d.unconverged_phases || {})) out.unconverged[k] = d.unconverged_phases[k] | 0;
    out.rejected_updates = d.rejected_updates | 0; out.clipped_deltas = d.clipped_deltas | 0; out.deltas = d.deltas | 0; out.real_transitions = d.real_transitions | 0;
    out.replay_writes = d.replay_writes | 0; out.episodic_writes = d.episodic_writes | 0; out.record_writes = d.record_writes | 0; out.imagined = d.imagined | 0;
    return out;
  }
}

/** The indices of the `k` largest entries of `values`, ascending (numpy argpartition keeps the k largest; among exact ties the earlier index is kept here). A min-heap of size k over one pass. */
export function topK(values, k) {
  const n = values.length;
  if (k >= n) return Int32Array.from({ length: n }, (_, i) => i);
  const heap = new Int32Array(k); // indices; heap[0] holds the smallest of the kept values
  let size = 0;
  const less = (i, j) => values[i] < values[j];
  for (let g = 0; g < n; g++) {
    if (size < k) {
      let c = size++;
      heap[c] = g;
      while (c > 0) { const parent = (c - 1) >> 1; if (less(heap[c], heap[parent])) { const t = heap[c]; heap[c] = heap[parent]; heap[parent] = t; c = parent; } else break; }
    } else if (values[g] > values[heap[0]]) {
      heap[0] = g;
      let c = 0;
      for (;;) {
        const l = 2 * c + 1, r = l + 1;
        let m = c;
        if (l < k && less(heap[l], heap[m])) m = l;
        if (r < k && less(heap[r], heap[m])) m = r;
        if (m === c) break;
        const t = heap[c]; heap[c] = heap[m]; heap[m] = t; c = m;
      }
    }
  }
  return Int32Array.from(heap.subarray(0, size)).sort((a, b) => a - b);
}

/** A sparse code {index, values} from a dense code vector. */
export function sparseCode(dense) {
  const index = [];
  for (let g = 0; g < dense.length; g++) if (dense[g] !== 0) index.push(g);
  return { index: Int32Array.from(index), values: Float64Array.from(index, (g) => dense[g]), granules: dense.length };
}

/** The dense code vector of a sparse code. */
export function denseCode(code) { const out = new Float64Array(code.granules); for (let k = 0; k < code.index.length; k++) out[code.index[k]] = code.values[k]; return out; }

/** The code pair {plain, valued, granules} of a dense (2, granules) array as the snapshot carries the pending code (a (granules,) array stands for both). */
export function codePairOf(dense, granules) {
  if (dense.length === 2 * granules) return { plain: sparseCode(dense.subarray(0, granules)), valued: sparseCode(dense.subarray(granules, 2 * granules)), granules };
  if (dense.length === granules) { const one = sparseCode(dense); return { plain: one, valued: one, granules }; }
  throw Error("the pending code does not match the records head");
}

export const VALUED_SOURCES = ["reward", "terminal"]; // the fields read through the valued code

/**
 * The records world head (brain.py RecordsHead): a reading of the source neurons minus each
 * unit's running mean (habituation), a fixed random expansion regenerated from the seed, the
 * `active` largest cells kept (the rest inhibited), unit length; delta-rule records per
 * prediction field. Two codes come from one reading: the plain code, read by the consequence
 * records, and the valued code, in which every input pathway (observation, goal, action,
 * recall, context: index arrays into the reading) is divided by its running norm (an equal
 * say), read by the reward and terminal records (VALUED_SOURCES). `code` returns code pairs
 * {plain, valued} of sparse codes {index, values} (the dense vector is zero off the active
 * cells, so a read or a write touches only those rows). The expansion is the same Mulberry32
 * stream as Python's; `h = x @ W + b` sums over the inputs in index order, then adds `b`; the
 * pathway norms and the row norm reproduce numpy's pairwise sums; a read sums the active rows
 * in index order (Python's BLAS order differs at rounding level).
 */
export class RecordsHead {
  constructor(inputs, fields, config, seed, blocks = [], tasks = []) {
    this.config = config; this.fields = fields; this.inputs = inputs | 0; this.granules = config.granules | 0;
    const I = this.inputs, G = this.granules;
    this.blocks = blocks.map((b) => Int32Array.from(b)).filter((b) => b.length > 0); // the input pathways, empty ones dropped
    this.blockNorm = new Float64Array(this.blocks.length).fill(1.0);
    this.lastInput = new Float64Array(I); // the reading the expansion last saw (after habituation and normalisation), for a page
    const rng = new Mulberry32((seed * 7919 + 13) >>> 0);
    const z = rng.normals(I * G), scale = Math.sqrt(I);
    this.W = new Float64Array(I * G);
    for (let k = 0; k < this.W.length; k++) this.W[k] = z[k] / scale;
    const zb = rng.normals(G);
    this.b = new Float64Array(G);
    for (let g = 0; g < G; g++) this.b[g] = zb[g] * config.bias_scale;
    // task sets (records.py `tasks`): the cells divided into one group per task unit by a Fisher-Yates
    // shuffle from G - 1 draws of the same generator, after the offsets; the valued code of a reading
    // draws its winners from the group of the task unit with the largest value
    this.tasks = Int32Array.from(tasks);
    this.taskOfCell = new Int32Array(G);
    if (this.tasks.length) {
      if (Math.floor(G / this.tasks.length) < config.active) throw Error("every task group needs at least active cells");
      const order = new Int32Array(G); for (let g = 0; g < G; g++) order[g] = g;
      const draws = rng.batch(Math.max(G - 1, 0));
      for (let i = G - 1; i > 0; i--) { const j = Math.floor(draws[G - 1 - i] * (i + 1)); const t = order[i]; order[i] = order[j]; order[j] = t; }
      // numpy array_split: the first G % T groups have one cell more
      const T = this.tasks.length, base = Math.floor(G / T), extra = G % T;
      let at = 0;
      for (let k = 0; k < T; k++) { const size = base + (k < extra ? 1 : 0); for (let j = 0; j < size; j++) this.taskOfCell[order[at + j]] = k; at += size; }
    }
    this.mean = new Float64Array(I);
    this.seen = 0; // readings the mean has followed: its rate is max(habituation, 1 / seen)
    this.tables = {};
    for (const f of fields) this.tables[f.name] = new Float64Array(G * f.width);
    this.writes = 0;
    this._acc = new Float64Array(G); this._h = new Float64Array(G); this._sq = new Float64Array(G);
  }

  /** The shared expansion (brain.py `_winners`): the sparse winner codes of `batch` rows of a flat (batch, inputs) array. */
  /** The task each row's task units select (-1: every cell allowed), from the raw reading (records.py `_allowed`). */
  _taskOf(raw, batch) {
    const I = this.inputs, out = new Int32Array(batch).fill(-1);
    if (!this.tasks.length) return out;
    for (let b = 0; b < batch; b++) {
      let best = -1, value = 0.0;
      for (let j = 0; j < this.tasks.length; j++) { const u = raw[b * I + this.tasks[j]]; if (best < 0 || u > value) { best = j; value = u; } }
      out[b] = value > 0.0 ? best : -1;
    }
    return out;
  }

  _winners(x, batch, taskOf = null) {
    const I = this.inputs, G = this.granules;
    const k = Math.min(this.config.active, G), acc = this._acc, h = this._h, sq = this._sq, W = this.W, out = [];
    for (let b = 0; b < batch; b++) {
      acc.fill(0);
      for (let i = 0; i < I; i++) {
        const xi = x[b * I + i];
        if (xi === 0) continue;
        const row = i * G;
        for (let g = 0; g < G; g++) acc[g] += xi * W[row + g];
      }
      for (let g = 0; g < G; g++) h[g] = acc[g] + this.b[g];
      if (taskOf !== null && taskOf[b] >= 0) { const task = taskOf[b]; for (let g = 0; g < G; g++) if (this.taskOfCell[g] !== task) h[g] = -Infinity; }
      const index = topK(h, k), values = new Float64Array(index.length);
      for (let j = 0; j < index.length; j++) { const v = h[index[j]] > 0.0 ? h[index[j]] : 0.0; values[j] = v; sq[index[j]] = v * v; }
      const norm = Math.sqrt(npSum(sq, 0, G));
      for (let j = 0; j < index.length; j++) sq[index[j]] = 0;
      if (norm > 0) { const d = Math.max(norm, 1e-12); for (let j = 0; j < values.length; j++) values[j] = values[j] / d; }
      out.push({ index, values, granules: G });
    }
    return out;
  }

  /** The code pairs of `batch` readings (a flat (batch, inputs) array): plain then valued per row. `adapt`: the running mean and the pathway norms follow each row first (the executed reading only); the norms shape the valued code alone. */
  code(readings, batch, adapt) {
    const I = this.inputs, hab = this.config.habituation;
    if (readings.length !== batch * I) throw Error("readings must have one entry per reading neuron and row");
    const x = Float64Array.from(readings);
    const taskOf = this.tasks.length ? this._taskOf(x, batch) : null; // from the raw reading, before habituation
    if (hab > 0) {
      if (adapt) {
        for (let b = 0; b < batch; b++) {
          this.seen += 1;
          const rate = Math.max(hab, 1.0 / this.seen);
          for (let i = 0; i < I; i++) this.mean[i] += rate * (x[b * I + i] - this.mean[i]);
        }
      }
      for (let b = 0; b < batch; b++) for (let i = 0; i < I; i++) x[b * I + i] = x[b * I + i] - this.mean[i];
    }
    this.lastInput = x.subarray(0, I);
    const plain = this._winners(x, batch);
    let valued = plain;
    if (this.config.normalize_blocks && this.blocks.length) {
      const v = Float64Array.from(x), rate = this.config.block_rate;
      for (let k = 0; k < this.blocks.length; k++) {
        const block = this.blocks[k], bsq = new Float64Array(block.length), norms = new Float64Array(batch);
        for (let b = 0; b < batch; b++) { for (let j = 0; j < block.length; j++) { const value = v[b * I + block[j]]; bsq[j] = value * value; } norms[b] = Math.sqrt(npSum(bsq)); }
        if (adapt) for (let b = 0; b < batch; b++) this.blockNorm[k] += rate * (norms[b] - this.blockNorm[k]);
        const d = this.blockNorm[k] + 1e-3;
        for (let b = 0; b < batch; b++) for (let j = 0; j < block.length; j++) v[b * I + block[j]] = v[b * I + block[j]] / d;
      }
      valued = this._winners(v, batch, taskOf);
    } else if (taskOf !== null) valued = this._winners(x, batch, taskOf);
    return plain.map((code, b) => ({ plain: code, valued: valued[b], granules: this.granules }));
  }

  /** The code a field reads: the valued code for reward and terminal records, the plain code otherwise. */
  codeOf(pair, f) { return VALUED_SOURCES.includes(f.source) ? pair.valued : pair.plain; }

  /** `code @ records[field]` over the active rows in index order. */
  read(code, f) {
    const R = this.tables[f.name], W = f.width, y = new Float64Array(W);
    for (let k = 0; k < code.index.length; k++) { const row = code.index[k] * W, c = code.values[k]; for (let j = 0; j < W; j++) y[j] += c * R[row + j]; }
    return y;
  }

  /** Each field's record read through its code: class probabilities (the positive part, normalised; uniform when nothing is read) or values in sensor units. */
  predict(pair) {
    const out = {};
    for (const f of this.fields) {
      const y = this.read(this.codeOf(pair, f), f);
      if (f.kind === "categorical") {
        const p = Float64Array.from(y, (v) => (v > 0.0 ? v : 0.0));
        const total = npSum(p);
        out[f.name] = total > 0 ? Float64Array.from(p, (v) => v / total) : new Float64Array(f.width).fill(1.0 / f.width);
      } else out[f.name] = Float64Array.from(y, (v) => f.lo + (clip(v, 0.05, 0.8) - 0.05) / 0.75 * (f.hi - f.lo));
    }
    return out;
  }

  /** The delta rule on the active rows of the field's code: R[g] += rate * (code[g] * err) per field present in `targets`; returns the fields written and reports them for a page. */
  learn(pair, targets) {
    let written = 0;
    const report = [];
    for (const f of this.fields) {
      if (!(f.name in targets)) continue;
      const { target, known } = targets[f.name];
      const reward = VALUED_SOURCES.includes(f.source);
      const rate = reward ? this.config.reward_rate : this.config.rate;
      const code = this.codeOf(pair, f);
      const read = this.read(code, f), W = f.width, err = new Float64Array(W);
      if (f.kind === "categorical") { const y = new Float64Array(W); y[target[0] | 0] = 1.0; for (let j = 0; j < W; j++) err[j] = y[j] - read[j]; }
      else for (let j = 0; j < W; j++) err[j] = (target[j] - read[j]) * (known[j] ? 1.0 : 0.0);
      const R = this.tables[f.name];
      for (let k = 0; k < code.index.length; k++) { const row = code.index[k] * W, c = code.values[k]; for (let j = 0; j < W; j++) R[row + j] += rate * (c * err[j]); }
      let largest = 0.0; for (let j = 0; j < W; j++) if (Math.abs(err[j]) > largest) largest = Math.abs(err[j]);
      report.push({ name: f.name, rate, reward, valued: reward, error: largest });
      written += 1;
    }
    this.writes += written;
    return { written, report };
  }

  parameters() { let out = 0; for (const f of this.fields) out += this.tables[f.name].length; return out; }

  /** Load the mean, the pathway norms, the tables (any float precision widens to float64) and the write count from a snapshot block or the float64 sidecar. */
  load(block) {
    if (block.mean) { const mean = unpackF64(block.mean); if (mean.length !== this.inputs) throw Error("the records mean does not match the reading"); this.mean = mean; }
    if (block.block_norm) { const norms = unpackF64(block.block_norm); if (norms.length !== this.blocks.length) throw Error(`the records block norms (${norms.length}) do not match the pathways (${this.blocks.length})`); this.blockNorm = norms; }
    for (const f of this.fields) {
      const spec = block.tables ? block.tables[f.name] : null;
      if (!spec) throw Error(`the records block lacks the table of ${f.name}`);
      const table = unpackF64(spec);
      if (table.length !== this.granules * f.width) throw Error(`the records table of ${f.name} does not match the head`);
      this.tables[f.name] = table;
    }
    if (block.writes !== undefined) this.writes = block.writes | 0;
    if (block.seen !== undefined) this.seen = block.seen | 0;
    if (block.task_of_cell && block.task_of_cell.length) {
      if (block.task_of_cell.length !== this.granules) throw Error("the records task groups do not match the cells");
      for (let g = 0; g < this.granules; g++) if ((block.task_of_cell[g] | 0) !== this.taskOfCell[g]) throw Error(`the regenerated task groups differ from the snapshot at cell ${g}`);
    }
  }

  snapshot() { const tables = {}; for (const f of this.fields) tables[f.name] = Float64Array.from(this.tables[f.name]); return { seen: this.seen, tasks: Array.from(this.tasks), task_of_cell: Array.from(this.taskOfCell), mean: Float64Array.from(this.mean), block_norm: Float64Array.from(this.blockNorm), blocks: this.blocks.map((b) => Array.from(b)), tables, writes: this.writes }; }
}

/** One learning head (cadence.learning.Learner): owns a plastic subset of the shared efficacy and bias. */
class Learner {
  constructor(agent, { beta, eta, etaBias, momentum = 0.0, normalize = 0.0, normalizeFloor = 1e-3, plasticSynapses, plasticNeurons, synapseRate, slots }) {
    this.agent = agent;
    this.beta = beta; this.eta = eta; this.etaBias = etaBias; this.momentum = momentum; this.normalize = normalize; this.normalizeFloor = normalizeFloor;
    this.plasticSynapses = plasticSynapses; this.plasticNeurons = plasticNeurons; this.synapseRate = synapseRate;
    this.allTrainable = plasticSynapses.every((x) => x === 1);
    this.slotSizes = slots;
    this.velocity = zeros(agent.E); this.velocityBias = zeros(agent.n); this.secondMoment = zeros(agent.E); this.secondMomentBias = zeros(agent.n);
    this.updates = 0; this.contrastUpdates = 0;
    const reverse = agent.reverse;
    const paired = [];
    for (let e = 0; e < agent.E; e++) if (reverse[e] >= 0) paired.push(e);
    this.paired = Int32Array.from(paired);
    this.pairedReverse = Int32Array.from(paired, (e) => reverse[e]);
    for (let k = 0; k < this.paired.length; k++) if (reverse[this.pairedReverse[k]] !== this.paired[k]) throw Error("reciprocal learning needs unique pairs");
  }
  /** The centred contrast (nudged − opposite) / (2 β), the batch mean per synapse and per neuron: the block Gram path of Learner.contrast. */
  contrast(free, plus, minus) {
    const a = this.agent, n = a.n, batch = plus.batch, span = 2.0 * this.beta, sp = plus.s, sm = minus.s;
    const g = zeros(a.layout.size);
    for (let k = 0; k < a.layout.pairs; k++) {
      const [a0, a1, b0, b1] = a.layout.bounds(k), nb = b1 - b0, off = a.layout.offset[k];
      let same = true;
      for (let b = 0; b < batch && same; b++) for (let i = a0; i < a1; i++) if (sp[b * n + i] !== sm[b * n + i]) { same = false; break; }
      for (let i = a0; i < a1; i++) for (let j = b0; j < b1; j++) {
        let block = 0.0;
        for (let b = 0; b < batch; b++) block += sp[b * n + i] * (sp[b * n + j] - sm[b * n + j]);
        if (!same) {
          let second = 0.0;
          for (let b = 0; b < batch; b++) second += (sp[b * n + i] - sm[b * n + i]) * sm[b * n + j];
          block = block + second;
        }
        g[off + (i - a0) * nb + (j - b0)] = block;
      }
    }
    const edges = zeros(a.E), neurons = zeros(n), norm = batch * span;
    for (let e = 0; e < a.E; e++) edges[e] = g[a.layout.edgeIndex[e]] / norm;
    for (let i = 0; i < n; i++) { let acc = 0.0; for (let b = 0; b < batch; b++) acc += sp[b * n + i] - sm[b * n + i]; neurons[i] = (acc / batch) / span; }
    return [edges, neurons];
  }
  /** The same contrast per row (batch, edges) and (batch, n): what a per-row eligibility trace reads. */
  contrastRows(free, plus, minus) {
    const a = this.agent, n = a.n, batch = plus.batch, span = 2.0 * this.beta, sp = plus.s, sm = minus.s;
    const edges = zeros(batch * a.E), neurons = zeros(batch * n);
    for (let b = 0; b < batch; b++) {
      for (let e = 0; e < a.E; e++) {
        const i = a.pre[e], j = a.post[e];
        const ap = sp[b * n + i], am = sm[b * n + i], bp = sp[b * n + j], bm = sm[b * n + j];
        edges[b * a.E + e] = (ap * (bp - bm) + (ap - am) * bm) / span;
      }
      for (let i = 0; i < n; i++) neurons[b * n + i] = (sp[b * n + i] - sm[b * n + i]) / span;
    }
    return [edges, neurons];
  }
  /** Learner.apply: masks, the synapse rate, reciprocal averaging, the cap; then the shared parameters move. */
  apply(deltaScale, deltaBias) {
    const a = this.agent, E = a.E, n = a.n;
    const delta = Float64Array.from(deltaScale);
    if (delta.length !== E || deltaBias.length !== n) throw Error("steps must be one per synapse and one per neuron");
    for (let e = 0; e < E; e++) if (!Number.isFinite(delta[e])) throw Error("steps must be finite");
    for (let i = 0; i < n; i++) if (!Number.isFinite(deltaBias[i])) throw Error("steps must be finite");
    if (!this.allTrainable) for (let e = 0; e < E; e++) if (!this.plasticSynapses[e]) delta[e] = 0.0;
    if (this.synapseRate) for (let e = 0; e < E; e++) delta[e] *= this.synapseRate[e];
    if (this.paired.length) {
      const mean = zeros(this.paired.length);
      for (let k = 0; k < this.paired.length; k++) mean[k] = 0.5 * (delta[this.paired[k]] + delta[this.pairedReverse[k]]);
      for (let k = 0; k < this.paired.length; k++) delta[this.paired[k]] = mean[k];
    }
    if (!this.allTrainable) for (let e = 0; e < E; e++) if (!this.plasticSynapses[e]) delta[e] = 0.0;
    const changed = [];
    const efficacy = a.efficacy, bias = a.bias, db = zeros(n);
    for (let e = 0; e < E; e++) {
      let scale = efficacy[e] + delta[e];
      if (this.plasticSynapses[e]) scale = clip(scale, -SCALE_CAP, SCALE_CAP);
      if (scale !== efficacy[e]) changed.push(e);
      efficacy[e] = scale;
    }
    for (let i = 0; i < n; i++) { db[i] = this.plasticNeurons[i] ? deltaBias[i] : 0.0; bias[i] = bias[i] + db[i]; }
    this.updates += 1;
    const absDelta = zeros(E); for (let e = 0; e < E; e++) absDelta[e] = Math.abs(delta[e]);
    const absBias = zeros(n); for (let i = 0; i < n; i++) absBias[i] = Math.abs(db[i]);
    return { scale_step: E ? npMean(absDelta) : 0.0, bias_step: npMean(absBias), changed: Int32Array.from(changed) };
  }
  parameters() {
    const a = this.agent, keys = new Int32Array(a.E);
    for (let e = 0; e < a.E; e++) keys[e] = a.reverse[e] >= 0 ? Math.min(e, a.reverse[e]) : e;
    const counted = new Uint8Array(a.E);
    for (let e = 0; e < a.E; e++) if (this.plasticSynapses[e]) counted[keys[e]] = 1;
    let out = 0; for (let e = 0; e < a.E; e++) out += counted[e];
    for (let i = 0; i < a.n; i++) out += this.plasticNeurons[i];
    return out;
  }
}

/** One life in the browser: fixed graph, one brain, one parameter set, all memories, one stream. */
export class ExperienceAgent {
  /**
   * @param snapshot the JSON that web.export wrote
   * @param options.planner the planner callable (agent, row, moment, drive, free) => {action, prediction, budget}
   */
  constructor(snapshot, options = {}) {
    if (snapshot.format !== SNAPSHOT_FORMAT) throw Error(`not a ${SNAPSHOT_FORMAT} snapshot`);
    this.config = normalizeConfig(snapshot.config);
    this.streams = 1;
    this.seed = snapshot.seed | 0;
    this.planner = options.planner || null;
    this.recallDrive = options.recallDrive || null; // (moment, row) => vector over the recall neurons: a stage's own stores read into the recall port (Agent.recall_drive)
    this.onSettleStep = options.onSettleStep || null; // (phase, activation, potential, {batch, step, moved}) after every settling step
    this.onLearn = options.onLearn || null; // ({phase, changed, dopamine, rejected}) after every credit and repair
    this.onPhase = options.onPhase || null; // ({phase, steps, residual, converged, batch}) after every checked phase
    this.onRecords = options.onRecords || null; // ({kind: "code" | "imagine" | "write", ...}) at every records-head event
    this.lastPhase = null;
    if (this.config.controller !== "actor" && this.config.controller !== "planner") throw Error("controller must be actor or planner");
    if (this.config.episodic !== null) throw Error("this engine carries no episodic store");
    const N = snapshot.neuron;
    this.neuron = { dt: N.dt, slope: N.slope, threshold: N.threshold, leak: N.leak, rest: N.rest, amplitude: N.amplitude, gain: N.gain, scaleUp: 1.0 / (1.0 - N.rest), scaleDown: N.rest > 0.0 ? N.leak / N.rest : 0.0 };
    const c = snapshot.connectome;
    this.n = c.n | 0; this.E = c.synapses | 0;
    this.pre = Int32Array.from(unpackArray(c.pre)); this.post = Int32Array.from(unpackArray(c.post)); this.count = unpackF64(c.count);
    if (this.pre.length !== this.E || this.post.length !== this.E) throw Error("connectome arrays do not match the synapse count");
    this.efficacy = unpackF64(snapshot.efficacy);
    this.bias = unpackF64(snapshot.bias);
    this.logGain = snapshot.log_gain ? unpackF64(snapshot.log_gain) : zeros(this.n);
    this.gainPre = zeros(this.E);
    for (let e = 0; e < this.E; e++) this.gainPre[e] = this.neuron.gain * this.count[e] * Math.exp(this.logGain[this.pre[e]]);
    const P = snapshot.ports, port = (name) => Int32Array.from(P[name] || []);
    this.ports = {
      observation: port("observation"), goal: port("goal"), action: port("action"), perception: port("perception"), workspace: port("workspace"), context: port("context"),
      recall: port("recall"), dynamics: port("dynamics"), prediction: port("prediction"), motor: port("motor"), intention: port("intention"),
      observation_fields: Object.fromEntries(Object.entries(P.observation_fields).map(([k, v]) => [k, Int32Array.from(v)])),
      observation_flags: Object.fromEntries(Object.entries(P.observation_flags || {}).map(([k, v]) => [k, Int32Array.from(v)])),
      prediction_fields: Object.fromEntries(Object.entries(P.prediction_fields).map(([k, v]) => [k, Int32Array.from(v)])),
      action_slots: P.action_slots.map((s) => Int32Array.from(s)), motor_slots: P.motor_slots.map((s) => Int32Array.from(s)),
      sources: Int32Array.from(P.sources || [...P.observation, ...P.goal, ...P.action, ...P.context, ...P.recall]),
    };
    this.lastRecall = zeros(this.streams * this.ports.recall.length); // Agent._last_recall: what the hook read at the last real moment, reused by imagination
    // the records world head (Agent.reading, Agent.records): the reading port from the snapshot, the expansion from the seed
    this.records = null;
    this.reading = this.ports.sources;
    if (this.config.records !== null) {
      const rc = snapshot.records || null;
      if (rc && rc.reading) this.reading = Int32Array.from(rc.reading);
      else if (!this.config.records.context) this.reading = Int32Array.from([...this.ports.observation, ...this.ports.goal, ...this.ports.action, ...this.ports.recall]);
      // the input pathways: the positions of the observation, goal, action, recall and context ports that lie in the reading (Agent.__init__), from the snapshot when it lists them
      let blocks;
      if (rc && rc.blocks) blocks = rc.blocks.map((b) => Int32Array.from(b));
      else { const position = new Map(Array.from(this.reading, (neuron, k) => [neuron, k])); blocks = [this.ports.observation, this.ports.goal, this.ports.action, this.ports.recall, this.ports.context].map((port) => Int32Array.from(Array.from(port).filter((n) => position.has(n)).map((n) => position.get(n)))); }
      this.records = new RecordsHead(this.reading.length, this.config.graph.prediction, this.config.records, rc && rc.seed !== undefined ? rc.seed | 0 : this.seed, blocks, this.config.records.task_sets ? (rc && rc.tasks ? rc.tasks : blocks[1] || []) : []);
      this.recordsSource = "regenerated (no tables in the snapshot)";
      if (rc && rc.tables) { this.records.load(rc); this.recordsSource = `snapshot (${rc.precision || "float32"} tables)`; }
      if (options.records) { this.records.load(options.records); this.recordsSource = "sidecar (float64 tables)"; }
    }
    this.layout = this._layout();
    this.blocks = this.layout.blocks.map((size) => zeros(size));
    this._rebuildBlocks();
    // plasticity
    const pl = snapshot.plasticity;
    const maskOf = (indices, size) => { const m = new Uint8Array(size); for (const k of indices) m[k] = 1; return m; };
    this.worldEdges = maskOf(pl.world_edges, this.E); this.motorEdges = maskOf(pl.motor_edges, this.E);
    this.worldNeurons = maskOf(pl.world_neurons, this.n); this.motorNeurons = maskOf(pl.motor_neurons, this.n);
    this.reverse = Int32Array.from(unpackArray(pl.reverse));
    this.synapseRate = pl.synapse_rate ? unpackF64(pl.synapse_rate) : null;
    const w = this.config.world, ac = this.config.actor;
    this.world = null;
    if (this.config.graph.prediction.length) {
      this.world = new Learner(this, { beta: w.beta, eta: w.eta, etaBias: w.eta_bias, momentum: w.momentum, normalize: w.normalize, plasticSynapses: this.worldEdges, plasticNeurons: this.worldNeurons, synapseRate: this.synapseRate, slots: [this.ports.prediction.length] });
      const ws = snapshot.world_state;
      if (ws) { this.world.velocity = unpackF64(ws.velocity); this.world.velocityBias = unpackF64(ws.velocity_bias); this.world.secondMoment = unpackF64(ws.second_moment); this.world.secondMomentBias = unpackF64(ws.second_moment_bias); this.world.contrastUpdates = ws.contrast_updates | 0; this.world.updates = ws.updates | 0; }
    }
    this.motor = new Learner(this, { beta: ac.beta, eta: ac.eta, etaBias: ac.eta_bias, plasticSynapses: this.motorEdges, plasticNeurons: this.motorNeurons, synapseRate: this.synapseRate, slots: this.config.graph.action_fields.slice() });
    const ms = snapshot.motor_state;
    if (ms) { this.motor.velocity = unpackF64(ms.velocity); this.motor.velocityBias = unpackF64(ms.velocity_bias); this.motor.contrastUpdates = ms.contrast_updates | 0; this.motor.updates = ms.updates | 0; }
    // context, critic, eligibility, generators, cursors
    this.context = new Trace(this.n, this.ports.workspace, this.ports.context, this.config.context_decay, this.config.context_amplitude);
    this.context.reset(1);
    const ctx = snapshot.context || {};
    if (ctx.trace && ctx.trace.length === this.context.width) { this.context.trace = Float64Array.from(ctx.trace); this.context.last = Float64Array.from(ctx.last); this.context.cold[0] = ctx.cold ? 1 : 0; }
    this.wCritic = Float64Array.from(snapshot.critic.w); this.bCritic = Number(snapshot.critic.b);
    const el = snapshot.eligibility;
    this.elig = el ? unpackF64(el.edges) : zeros(this.E); this.eligBias = el ? unpackF64(el.bias) : zeros(this.n); this.traceCritic = el ? Float64Array.from(el.critic) : zeros(this.ports.workspace.length + 1);
    this.rng = [new Mulberry32(snapshot.rng.action)];
    this.replayRng = new Mulberry32(snapshot.rng.replay);
    const cur = snapshot.cursors;
    this.lastEvent = [cur.last_event]; this.episode = [cur.episode]; this.nextDecisionId = cur.next_decision_id; this.parameterVersion = cur.parameter_version; this.contextVersion = cur.context_version;
    this.warm = snapshot.warm ? brainState(unpackF64(snapshot.warm.v), null, unpackF64(snapshot.warm.a), 1, 0) : null;
    this.pending = [null];
    this.ledger = Ledger.fromDict(snapshot.ledger);
    this.lastReport = {};
    this.lastConverged = true;
    this.halted = false;
    this.ring = []; this.ringAt = 0;
    this.ringSource = "none"; this.pendingSource = "none";
    this._loadRing(snapshot.ring, "snapshot");
    this._loadPending(snapshot.pending, "snapshot");
    this.stateVersion = 0; // bumped on every parameter change, for pages that cache weights
  }

  // -- construction helpers

  _layout() {
    const n = this.n, cuts = new Set([0, n]);
    for (const name of ["observation", "goal", "action", "perception", "workspace", "context", "recall", "dynamics", "prediction", "motor", "intention"]) {
      const idx = this.ports[name];
      if (idx.length && isContiguous(idx)) { cuts.add(idx[0]); cuts.add(idx[idx.length - 1] + 1); }
    }
    const starts = Int32Array.from([...cuts].sort((x, y) => x - y));
    const ranges = starts.length - 1;
    const rangeOf = new Int32Array(n);
    for (let r = 0; r < ranges; r++) for (let i = starts[r]; i < starts[r + 1]; i++) rangeOf[i] = r;
    const keys = new Set();
    for (let e = 0; e < this.E; e++) keys.add(rangeOf[this.pre[e]] * ranges + rangeOf[this.post[e]]);
    const uniq = [...keys].sort((x, y) => x - y);
    const pairPre = Int32Array.from(uniq, (k) => Math.floor(k / ranges)), pairPost = Int32Array.from(uniq, (k) => k % ranges);
    const width = (r) => starts[r + 1] - starts[r];
    const offset = new Int32Array(uniq.length + 1), blocks = [];
    for (let k = 0; k < uniq.length; k++) { const size = width(pairPre[k]) * width(pairPost[k]); blocks.push(size); offset[k + 1] = offset[k] + size; }
    const pairOf = new Map(uniq.map((k, idx) => [k, idx]));
    const edgeIndex = new Int32Array(this.E), edgePair = new Int32Array(this.E);
    for (let e = 0; e < this.E; e++) {
      const k = pairOf.get(rangeOf[this.pre[e]] * ranges + rangeOf[this.post[e]]);
      const a0 = starts[pairPre[k]], b0 = starts[pairPost[k]], nb = width(pairPost[k]);
      edgePair[e] = k; edgeIndex[e] = offset[k] + (this.pre[e] - a0) * nb + (this.post[e] - b0);
    }
    const hasInput = new Uint8Array(ranges);
    for (let k = 0; k < uniq.length; k++) hasInput[pairPost[k]] = 1;
    const seen = new Set(edgeIndex);
    if (seen.size !== this.E) throw Error("parallel synapses share a matrix entry; merge them first");
    return { starts, ranges, rangeOf, pairPre, pairPost, pairs: uniq.length, offset, edgeIndex, edgePair, blocks, hasInput, size: offset[uniq.length], bounds: (k) => [starts[pairPre[k]], starts[pairPre[k] + 1], starts[pairPost[k]], starts[pairPost[k] + 1]] };
  }

  _rebuildBlocks() {
    for (const b of this.blocks) b.fill(0);
    const lay = this.layout;
    for (let e = 0; e < this.E; e++) this.blocks[lay.edgePair[e]][lay.edgeIndex[e] - lay.offset[lay.edgePair[e]]] = this.gainPre[e] * this.efficacy[e];
  }

  /** The replay ring from an export: drives, and the warm potentials when the export carries them (rest otherwise). */
  _loadRing(ring, source) {
    this.ring = []; this.ringAt = ring ? ring.at | 0 : 0;
    if (!ring || !ring.drive) { this.ringSource = `${source} (no stored transitions)`; return; }
    const n = this.n, stored = ring.drive.shape[0];
    const drives = unpackF64(ring.drive), vs = ring.v ? unpackF64(ring.v) : null, as = ring.a ? unpackF64(ring.a) : null;
    if (drives.length !== stored * n || (vs && vs.length !== stored * n) || (as && as.length !== stored * n)) throw Error("the ring's arrays do not match the neuron count");
    if (!ring.targets || ring.targets.length !== stored) throw Error("the ring needs one targets entry per stored transition");
    for (let k = 0; k < stored; k++) {
      const v = vs ? vs.slice(k * n, (k + 1) * n) : zeros(n), a = as ? as.slice(k * n, (k + 1) * n) : zeros(n);
      this.ring.push([drives.slice(k * n, (k + 1) * n), v, a, this._ringTargets(ring.targets[k])]);
    }
    const precision = (spec) => (spec.dtype === "f8" ? "float64" : "float32");
    const sample = ring.stored !== undefined && ring.stored !== stored ? `${stored} of ${ring.stored}` : `${stored}`;
    this.ringSource = `${source} (${sample} transitions, ${precision(ring.drive)} drives, ${vs && as ? `${precision(ring.v)} warm potentials` : "warm states at rest"})`;
    if (this.ringAt >= this.ring.length) this.ringAt = 0;
  }

  /** The decision awaiting its outcome, as web.export writes it (`Decision.to_dict()` carries no mask: the legal entries are those with probability). */
  _loadPending(p, source) {
    if (!p) { this.pending[0] = null; this.pendingSource = `${source} (none)`; return; }
    const n = this.n, E = this.E, d = p.decision, prediction = {};
    for (const [k, v] of Object.entries(d.prediction || {})) prediction[k] = Float64Array.from(v);
    const probabilities = Float64Array.from(d.probabilities);
    const mask = p.action_mask ? Uint8Array.from(p.action_mask, (x) => (x ? 1 : 0)) : Uint8Array.from(probabilities, (x) => (x > 0 ? 1 : 0));
    if (!CONTROLLERS.includes(d.controller) || !mask[d.action]) throw Error("the pending decision is not a legal decision");
    const decision = { decision_id: d.decision_id | 0, event_id: d.event_id | 0, stream: d.stream | 0, parameter_version: d.parameter_version | 0, context_version: d.context_version | 0, action: d.action | 0, action_mask: mask, probabilities, log_probability: Number(d.log_probability), controller: d.controller, prediction, uncertainty: { ...(d.uncertainty || {}) }, budget: { ...(d.budget || {}) }, intention: d.intention ?? null };
    const drive = unpackF64(p.drive), stateV = unpackF64(p.state_v), stateA = unpackF64(p.state_a), workspace = unpackF64(p.workspace);
    const scoreEdges = p.score_edges ? unpackF64(p.score_edges) : null, scoreBias = p.score_bias ? unpackF64(p.score_bias) : null;
    if (drive.length !== n || stateV.length !== n || stateA.length !== n || workspace.length !== this.ports.workspace.length || (scoreEdges && scoreEdges.length !== E) || (scoreBias && scoreBias.length !== n)) throw Error("the pending record's arrays do not match the connectome");
    let code = null;
    if (p.code) { const dense = unpackF64(p.code); if (this.records === null) throw Error("the pending decision carries a code but the snapshot has no records head"); code = codePairOf(dense, this.records.granules); }
    this.pending[0] = { decision, drive, state_v: stateV, state_a: stateA, value: Number(p.value), score_edges: scoreEdges, score_bias: scoreBias, workspace, code };
    this.pendingSource = `${source} (decision ${decision.decision_id}, ${decision.controller}${code ? ", with its code" : ""})`;
  }

  _ringTargets(entry) {
    const targets = {};
    for (const f of this.config.graph.prediction) if (entry && entry[f.name]) targets[f.name] = { target: Float64Array.from(entry[f.name].target), known: Uint8Array.from(entry[f.name].known, (x) => (x ? 1 : 0)) };
    return targets;
  }

  // -- shared parameter ownership

  get heads() { return [this.world, this.motor].filter((h) => h !== null); }

  _rebind(owner) {
    this._rebuildBlocks();
    this.parameterVersion += 1;
    this.stateVersion += 1;
  }

  setPlanner(fn) { this.planner = fn; }

  /** Install a stage's recall hook: (moment, row) => finite vector over the recall neurons, added to the sensory drive of every real moment and of every imagined prediction. */
  setRecallDrive(fn, last = null) {
    this.recallDrive = fn || null;
    if (last !== null) { if (last.length !== this.lastRecall.length) throw Error("last recall must have one entry per recall neuron"); this.lastRecall = Float64Array.from(last); }
  }

  /** The effective weight of every synapse (gain × count × efficacy × exp(log_gain[pre])), for a page's scan. */
  effectiveWeights() { const out = new Float32Array(this.E); for (let e = 0; e < this.E; e++) out[e] = this.gainPre[e] * this.efficacy[e]; return out; }

  /** The learned parameters, generators and cursors, for a parity check or a page's ledger. */
  snapshotState() {
    return {
      efficacy: Float64Array.from(this.efficacy), bias: Float64Array.from(this.bias), parameter_version: this.parameterVersion, context_version: this.contextVersion,
      rng: { action: this.rng[0].state, replay: this.replayRng.state }, critic_w: Float64Array.from(this.wCritic), critic_b: this.bCritic,
      context_trace: Float64Array.from(this.context.trace), world_velocity: this.world ? Float64Array.from(this.world.velocity) : null,
      ring_stored: this.ring.length, ring_at: this.ringAt, next_decision_id: this.nextDecisionId, last_event: this.lastEvent[0], episode: this.episode[0],
      pending: this.pending[0] ? this.pending[0].decision.decision_id : null, ledger: this.ledger.toDict(), ring_source: this.ringSource, pending_source: this.pendingSource,
      last_recall: Float64Array.from(this.lastRecall),
      records: this.records === null ? null : this.records.snapshot(), records_source: this.records === null ? null : this.recordsSource,
    };
  }

  // -- encoding

  /** The sensory drive of one moment: observed values (unobserved read zero) and the missing flags. */
  encodeObservation(observation, observed) {
    const spec = this.config.graph, out = zeros(this.n);
    for (const f of spec.observation) {
      if (!(f.name in observation)) continue; // an absent field is entirely unobserved
      const value = observation[f.name];
      if (value.length !== f.width) throw Error(`observation field '${f.name}' must have shape (${f.width},)`);
      const flag = observed === null || observed === undefined || !(f.name in observed) ? new Uint8Array(f.width).fill(1) : observed[f.name];
      const neurons = this.ports.observation_fields[f.name];
      for (let k = 0; k < f.width; k++) {
        let x = value[k];
        if (f.kind === "continuous") x = clip((x - f.lo) / (f.hi - f.lo), 0.0, 1.0);
        out[neurons[k]] = spec.input_gain * x * (flag[k] ? 1.0 : 0.0);
      }
      if (spec.missing_flags) {
        const flags = this.ports.observation_flags[f.name];
        if (spec.flags_per_field) { let all = true; for (let k = 0; k < f.width; k++) if (!flag[k]) all = false; out[flags[0]] = spec.input_gain * (all ? 0.0 : 1.0); }
        else for (let k = 0; k < f.width; k++) out[flags[k]] = spec.input_gain * (flag[k] ? 0.0 : 1.0);
      }
    }
    return out;
  }

  _drive(moments) {
    const n = this.n, drive = zeros(this.streams * n);
    moments.forEach((m, i) => {
      drive.set(this.encodeObservation(m.observation, m.observed), i * n);
      if (this.ports.goal.length) {
        if (m.goal === null || m.goal.length !== this.ports.goal.length) throw Error("this brain needs a goal of the declared width on every moment");
        for (let k = 0; k < this.ports.goal.length; k++) drive[i * n + this.ports.goal[k]] = this.config.graph.input_gain * m.goal[k];
      }
    });
    const out = this.context.stimulate(drive, this.streams);
    const recall = this.ports.recall, width = recall.length;
    if (this.recallDrive !== null && width) { // Agent._drive: the stage's stores read into the recall port, remembered for imagination
      moments.forEach((m, i) => {
        const read = this.recallDrive(m, i);
        if (!read || read.length !== width) throw Error("recall_drive must return a finite vector over the recall neurons");
        for (let k = 0; k < width; k++) {
          const x = Number(read[k]);
          if (!Number.isFinite(x)) throw Error("recall_drive must return a finite vector over the recall neurons");
          this.lastRecall[i * width + k] = x;
          out[i * n + recall[k]] += x;
        }
      });
    }
    return out;
  }

  _withAction(drive, batch, actions) {
    const n = this.n, out = Float64Array.from(drive), gain = this.config.graph.input_gain;
    for (let i = 0; i < batch; i++) {
      for (const j of this.ports.action) out[i * n + j] = 0.0;
      const parts = this._splitAction(actions[i] | 0);
      this.ports.action_slots.forEach((slot, k) => { out[i * n + slot[parts[k]]] = gain; });
    }
    return out;
  }

  /** A joint action index into one index per field (row-major over the fields). */
  _splitAction(action) {
    const sizes = this.config.graph.action_fields, total = sizes.reduce((p, k) => p * k, 1);
    if (!(action >= 0 && action < total)) throw Error("action index outside the joint action space");
    const out = [];
    for (let k = sizes.length - 1; k >= 0; k--) { out.push(action % sizes[k]); action = Math.floor(action / sizes[k]); }
    return out.reverse();
  }

  _jointAction(parts) { let action = 0; this.config.graph.action_fields.forEach((k, i) => { action = action * k + (parts[i] | 0); }); return action; }

  /** `drive[:, reading]`: the records head's reading of every row of a (batch, n) drive. */
  _readings(drive, batch) {
    const n = this.n, R = this.reading, I = R.length, out = new Float64Array(batch * I);
    for (let b = 0; b < batch; b++) for (let i = 0; i < I; i++) out[b * I + i] = drive[b * n + R[i]];
    return out;
  }

  // -- the neuron model and the settling kernel (cadence.fused on the CPU backend)

  _act(x) {
    const N = this.neuron;
    const r = 1.0 / (1.0 + Math.exp(-N.slope * (x - N.threshold))) - N.rest;
    if (r > 0.0) return r * N.scaleUp;
    if (N.leak === 0.0) return 0.0;
    return r * N.scaleDown;
  }

  _nudgeArrays(nudge, batch) {
    const n = this.n;
    if (!nudge) return null;
    const target = nudge.target.length === n ? (() => { const t = zeros(batch * n); for (let b = 0; b < batch; b++) t.set(nudge.target, b * n); return t; })() : Float64Array.from(nudge.target);
    if (target.length !== batch * n) throw Error("nudge target must match the drive's neurons and batch size");
    let nmask = Float64Array.from(nudge.mask);
    if (nmask.length !== n) throw Error("nudge mask must have one entry per neuron");
    const softmaxT = nudge.softmaxT == null ? 0.0 : Number(nudge.softmaxT);
    if (nudge.softmaxT != null && !(softmaxT > 0 && Number.isFinite(softmaxT))) throw Error("softmax_temperature must be finite and positive");
    const weight = nudge.weight ? Float64Array.from(nudge.weight) : new Float64Array(batch).fill(1.0);
    const beta = Number(nudge.beta);
    if (!Number.isFinite(beta)) throw Error("nudge beta must be finite");
    let gid, ngroups;
    if (!nudge.groups) { gid = Int32Array.from(nmask, (m) => (m > 0 ? 0 : -1)); ngroups = 1; }
    else {
      const ids = [...new Set(Array.from(nudge.groups).filter((g) => g >= 0))].sort((x, y) => x - y);
      const compact = new Map(ids.map((g, k) => [g, k]));
      gid = Int32Array.from(nudge.groups, (g) => (g >= 0 ? compact.get(g) : -1));
      ngroups = ids.length;
    }
    if (softmaxT > 0) nmask = Float64Array.from(nmask, (m, i) => (gid[i] >= 0 ? m : 0.0));
    const group = [];
    for (let i = 0; i < n; i++) if (nmask[i] > 0.0) group.push(i);
    const position = new Int32Array(n).fill(-1);
    group.forEach((i, k) => { position[i] = k; });
    return { target, nmask, gid, ngroups, beta, softmaxT, weight, group: Int32Array.from(group), position };
  }

  _softmaxRow(nd, s, b, p, zmax, total) {
    const n = this.n;
    for (let g = 0; g < nd.ngroups; g++) { zmax[g] = -1e300; total[g] = 0.0; }
    for (let k = 0; k < nd.group.length; k++) { const g = nd.gid[nd.group[k]], z = s[b * n + nd.group[k]] / nd.softmaxT; p[k] = z; if (z > zmax[g]) zmax[g] = z; }
    for (let k = 0; k < nd.group.length; k++) { const g = nd.gid[nd.group[k]]; p[k] = Math.exp(p[k] - zmax[g]); total[g] += p[k]; }
    for (let k = 0; k < nd.group.length; k++) p[k] /= total[nd.gid[nd.group[k]]];
  }

  /** The block transport: the synaptic input of every row, summed block by block in the layout's pair order. */
  _transport(s, batch, syn, cache, movedRange, first) {
    const n = this.n, lay = this.layout;
    syn.fill(0);
    for (let k = 0; k < lay.pairs; k++) {
      const a0 = lay.starts[lay.pairPre[k]], a1 = lay.starts[lay.pairPre[k] + 1], b0 = lay.starts[lay.pairPost[k]], b1 = lay.starts[lay.pairPost[k] + 1], nb = b1 - b0;
      const c = cache[k], blk = this.blocks[k];
      if (first || movedRange[lay.pairPre[k]]) {
        c.fill(0);
        for (let b = 0; b < batch; b++) {
          const cb = b * nb, sb = b * n;
          for (let i = a0; i < a1; i++) {
            const si = s[sb + i];
            if (si === 0) continue;
            const row = (i - a0) * nb;
            for (let j = 0; j < nb; j++) c[cb + j] += si * blk[row + j];
          }
        }
      }
      for (let b = 0; b < batch; b++) { const cb = b * nb, ob = b * n + b0; for (let j = 0; j < nb; j++) syn[ob + j] += c[cb + j]; }
    }
  }

  /** cadence.fused.fused_settle: `steps` steps in place on v and a from the published activations of v; returns the activations and the steps taken. */
  _run(phase, v, a, drive, steps, nudge, tolerance) {
    const n = this.n, batch = v.length / n, lay = this.layout, dt = this.neuron.dt;
    const standing = zeros(batch * n);
    for (let b = 0; b < batch; b++) for (let i = 0; i < n; i++) standing[b * n + i] = drive[b * n + i] + this.bias[i];
    const s = zeros(batch * n);
    let anyV = false;
    for (let k = 0; k < v.length; k++) if (v[k] !== 0) { anyV = true; break; }
    if (!anyV) { const rest = this._act(0.0); s.fill(rest); } else for (let k = 0; k < v.length; k++) s[k] = this._act(v[k]);
    const nd = this._nudgeArrays(nudge, batch);
    const freezable = new Uint8Array(lay.ranges);
    for (let r = 0; r < lay.ranges; r++) freezable[r] = lay.hasInput[r] ? 0 : 1;
    if (nd) for (let i = 0; i < n; i++) if (nd.nmask[i] > 0.0) freezable[lay.rangeOf[i]] = 0;
    const cache = [];
    for (let k = 0; k < lay.pairs; k++) cache.push(zeros(batch * (lay.starts[lay.pairPost[k] + 1] - lay.starts[lay.pairPost[k]])));
    const movedRange = new Uint8Array(lay.ranges).fill(1), movedPotential = new Uint8Array(lay.ranges), frozen = new Uint8Array(lay.ranges);
    const syn = zeros(batch * n), activityChange = zeros(batch);
    const p = nd ? zeros(nd.group.length) : null, zmax = nd ? zeros(Math.max(nd.ngroups, 1)) : null, total = nd ? zeros(Math.max(nd.ngroups, 1)) : null;
    const softmax = nd !== null && nd.softmaxT > 0.0, quadratic = nd !== null && !softmax;
    const useTolerance = tolerance !== null && tolerance !== undefined;
    let taken = 0;
    for (let t = 0; t < steps; t++) {
      this._transport(s, batch, syn, cache, movedRange, t === 0);
      movedRange.fill(0); movedPotential.fill(0);
      let moved = 0.0;
      for (let b = 0; b < batch; b++) {
        if (softmax) this._softmaxRow(nd, s, b, p, zmax, total);
        const wb = nd ? nd.weight[b] : 0.0;
        for (let r = 0; r < lay.ranges; r++) {
          if (frozen[r]) continue;
          for (let i = lay.starts[r]; i < lay.starts[r + 1]; i++) {
            const idx = b * n + i;
            let tot = syn[idx] + standing[idx];
            if (quadratic && nd.nmask[i] > 0.0) tot += nd.beta * wb * nd.nmask[i] * (nd.target[idx] - s[idx]);
            tot -= v[idx];
            let vn = v[idx] + dt * tot;
            if (softmax && nd.nmask[i] > 0.0) vn += dt * nd.beta * wb * (nd.target[idx] - p[nd.position[i]]);
            if (vn !== v[idx]) { // a neuron whose potential did not move publishes what it did
              movedPotential[r] = 1;
              v[idx] = vn;
              const sn = this._act(vn);
              const d = Math.abs(sn - s[idx]);
              activityChange[b] += d;
              if (d > 0.0) { movedRange[r] = 1; if (d > moved) moved = d; }
              s[idx] = sn;
            }
          }
        }
      }
      for (let r = 0; r < lay.ranges; r++) if (freezable[r] && !movedPotential[r] && t > 0) frozen[r] = 1; // a still range hearing nothing stays still
      taken = t + 1;
      if (this.onSettleStep) this.onSettleStep(phase, s.subarray(0, n), v.subarray(0, n), { batch, step: taken, moved });
      if (useTolerance && moved < tolerance) break;
    }
    return { s, taken, activityChange };
  }

  /** Brain.settle_batch on the CPU backend: from rest or from `state`, at most `steps` steps, stopping early under the movement tolerance. */
  settleBatch(drive, { steps = 60, state = null, nudge = null, tolerance = null, phase = "settle" } = {}) {
    const n = this.n;
    if (drive.length % n !== 0 || drive.length === 0) throw Error("drive must have shape (batch, neurons) with at least one row");
    const batch = drive.length / n;
    if (!(steps >= 0)) throw Error("steps must be a nonnegative integer");
    if (tolerance === 0) tolerance = null;
    for (let k = 0; k < drive.length; k++) if (!Number.isFinite(drive[k])) throw Error("drive must be finite");
    let v, a;
    if (state === null) { v = zeros(batch * n); a = zeros(batch * n); }
    else {
      if (state.v.length !== batch * n || state.a.length !== batch * n) throw Error("state batch does not match the drive batch");
      v = Float64Array.from(state.v); a = Float64Array.from(state.a);
      for (let k = 0; k < v.length; k++) if (!Number.isFinite(v[k]) || !Number.isFinite(a[k])) throw Error("warm potentials and adaptation must be finite");
    }
    const { s, taken, activityChange } = this._run(phase, v, a, drive, steps | 0, nudge, tolerance);
    const out = brainState(v, s, a, batch, taken);
    out.activityChange = activityChange;
    return out;
  }

  /** Brain.residual (cadence.fused.fused_residual): the largest fixed-point equation error per row. */
  residual(drive, state, nudge = null) {
    const n = this.n, batch = state.batch, lay = this.layout, dt = this.neuron.dt;
    const s = zeros(batch * n);
    for (let k = 0; k < s.length; k++) s[k] = this._act(state.v[k]);
    const standing = zeros(batch * n);
    for (let b = 0; b < batch; b++) for (let i = 0; i < n; i++) standing[b * n + i] = drive[b * n + i] + this.bias[i];
    const cache = [];
    for (let k = 0; k < lay.pairs; k++) cache.push(zeros(batch * (lay.starts[lay.pairPost[k] + 1] - lay.starts[lay.pairPost[k]])));
    const syn = zeros(batch * n);
    this._transport(s, batch, syn, cache, null, true);
    const nd = this._nudgeArrays(nudge, batch);
    const p = nd ? zeros(nd.group.length) : null, zmax = nd ? zeros(Math.max(nd.ngroups, 1)) : null, total = nd ? zeros(Math.max(nd.ngroups, 1)) : null;
    const out = zeros(batch);
    for (let b = 0; b < batch; b++) {
      if (nd && nd.softmaxT > 0.0) this._softmaxRow(nd, s, b, p, zmax, total);
      let worst = 0.0, finite = true;
      for (let i = 0; i < n; i++) {
        const idx = b * n + i;
        let tot = syn[idx] + standing[idx] - state.v[idx];
        if (nd && nd.nmask[i] > 0.0) {
          if (nd.softmaxT > 0.0) tot += nd.beta * nd.weight[b] * (nd.target[idx] - p[nd.position[i]]);
          else tot += nd.beta * nd.weight[b] * nd.nmask[i] * (nd.target[idx] - s[idx]);
        }
        const err = Math.abs(1.0 * tot + 0.0 * state.v[idx] / dt);
        if (!(err <= 1e300)) finite = false;
        else if (err > worst) worst = err;
      }
      out[b] = finite ? worst : Infinity;
    }
    return out;
  }

  /** Brain.equilibrate: chunks of steps until the neuron equations hold to the tolerance or the budget is spent. */
  equilibrate(drive, { budget = 512, chunk = 32, tolerance = 1e-5, state = null, nudge = null, phase = "settle" } = {}) {
    if (!(budget >= 0) || !(chunk >= 1)) throw Error("budget must be an integer >= 0 and chunk >= 1");
    if (!(tolerance >= 0) || !Number.isFinite(tolerance)) throw Error("tolerance must be finite and nonnegative");
    let current = this.settleBatch(drive, { steps: 0, state, nudge, phase });
    let error = this.residual(drive, current, nudge);
    let used = 0;
    const all = (err) => err.every((x) => x <= tolerance);
    while (used < budget && !all(error)) {
      current = this.settleBatch(drive, { steps: Math.min(chunk, budget - used), state: current, nudge, phase });
      used += current.steps;
      error = this.residual(drive, current, nudge);
    }
    return { state: current, residual: error, steps: used, tolerance };
  }

  /** Agent._settle: checked settling with the ledger; every phase records its steps and its residual. */
  _settle(phase, drive, { warm = null, nudge = null, budget = null } = {}) {
    const w = this.config.world;
    const steps = budget === null ? (nudge !== null ? w.nudged_steps : w.free_steps) : budget;
    const eq = this.equilibrate(drive, { budget: steps, chunk: w.chunk, tolerance: w.residual_tolerance, state: warm, nudge, phase });
    let residual = -Infinity;
    for (const r of eq.residual) if (r > residual || Number.isNaN(r)) residual = r;
    for (const x of eq.state.s) if (!Number.isFinite(x)) { this.halted = true; throw Error("nonfinite neural state; the life is halted"); }
    this.ledger.record(phase, eq.steps, residual, w.residual_tolerance);
    this.lastConverged = residual <= w.residual_tolerance;
    this.lastPhase = { phase, steps: eq.steps, residual, converged: this.lastConverged, batch: eq.state.batch };
    if (this.onPhase) this.onPhase(this.lastPhase);
    return brainState(eq.state.v, eq.state.s, eq.state.a, eq.state.batch, eq.steps);
  }

  _value(state) {
    const n = this.n, ws = this.ports.workspace, out = zeros(state.batch);
    for (let b = 0; b < state.batch; b++) { let acc = 0.0; for (let j = 0; j < ws.length; j++) acc += state.s[b * n + ws[j]] * this.wCritic[j]; out[b] = acc + this.bCritic; }
    return out;
  }

  // -- decoding predictions

  /** The prediction fields of a settled row: class probabilities or values in sensor units. */
  decode(state, row = 0) {
    const out = {}, n = this.n, s = state.s;
    for (const f of this.config.graph.prediction) {
      const neurons = this.ports.prediction_fields[f.name];
      if (f.kind === "categorical") {
        const z = zeros(neurons.length);
        let top = -Infinity;
        for (let k = 0; k < neurons.length; k++) { z[k] = s[row * n + neurons[k]] / this.config.world.temperature; if (z[k] > top) top = z[k]; }
        for (let k = 0; k < z.length; k++) z[k] = Math.exp(z[k] - top);
        const sum = npSum(z);
        out[f.name] = Float64Array.from(z, (x) => x / sum);
      } else out[f.name] = Float64Array.from(neurons, (i) => f.lo + (clip(s[row * n + i], 0.05, 0.8) - 0.05) / 0.75 * (f.hi - f.lo));
    }
    return out;
  }

  _continuousTarget(f, value) { return Float64Array.from(value, (x) => 0.05 + 0.75 * clip((x - f.lo) / (f.hi - f.lo), 0.0, 1.0)); }

  /** What a moment exposes for each prediction field: (target, observed mask) in the field's neuron coordinates. */
  _outcomeTargets(m) {
    const out = {};
    for (const f of this.config.graph.prediction) {
      if (f.source === "reward") { if (!m.reward_known) continue; out[f.name] = { target: this._continuousTarget(f, [m.reward]), known: Uint8Array.of(1) }; continue; }
      if (f.source === "terminal") { out[f.name] = { target: Float64Array.of(m.terminated ? 1 : 0), known: Uint8Array.of(1) }; continue; }
      if (!(f.source in m.observation)) continue;
      const value = m.observation[f.source], known = m.observed[f.source];
      if (f.kind === "categorical") {
        if (!known.every((x) => x === 1)) continue;
        let sum = 0.0, top = -Infinity, arg = 0;
        for (let k = 0; k < value.length; k++) { sum += value[k]; if (value[k] > top) { top = value[k]; arg = k; } }
        if (value.length !== f.width || Math.abs(sum - 1.0) > 1e-8 + 1e-5 || top < 0.99) throw Error(`categorical outcome '${f.source}' must be one-hot over ${f.width} classes`);
        out[f.name] = { target: Float64Array.of(arg), known: Uint8Array.of(1) };
      } else {
        if (value.length !== f.width) throw Error(`continuous outcome '${f.source}' must have ${f.width} values`);
        if (!known.some((x) => x === 1)) continue;
        out[f.name] = { target: this._continuousTarget(f, value), known };
      }
    }
    return out;
  }

  /** Score a stored prediction against what the moment actually exposes. */
  score(prediction, m) {
    const report = {}, targets = this._outcomeTargets(m);
    for (const f of this.config.graph.prediction) {
      if (!(f.name in prediction) || !(f.name in targets)) continue;
      const { target, known } = targets[f.name], p = prediction[f.name];
      if (f.kind === "categorical") {
        const k = target[0] | 0;
        let arg = 0; for (let j = 1; j < p.length; j++) if (p[j] > p[arg]) arg = j;
        report[`${f.name}/nll`] = -Math.log(Math.max(p[k], 1e-12)); report[`${f.name}/correct`] = arg === k ? 1.0 : 0.0;
      } else {
        const predicted = this._continuousTarget(f, p), err = [];
        for (let k = 0; k < predicted.length; k++) if (known[k]) err.push((predicted[k] - target[k]) ** 2);
        report[`${f.name}/mse`] = err.length ? npMean(Float64Array.from(err)) : NaN;
      }
    }
    return report;
  }

  // -- the event transaction

  /** One real event of the one stream: the moment as in stream.jsonl; returns Decision.to_dict() or null. */
  step(moment) {
    const d = this.stepBatch([normalizeMoment(moment)])[0];
    return d === null ? null : decisionToDict(d);
  }

  stepBatch(moments) {
    if (this.halted) throw Error("the life is halted after a numerical failure");
    if (moments.length !== this.streams) throw Error(`one moment per stream: expected ${this.streams}, got ${moments.length}`);
    moments = moments.map(normalizeMoment);
    const n = this.n;
    // 1. validate every event before any state changes
    moments.forEach((m, i) => this._validate(i, m));
    moments.forEach((m, i) => { if (m.episode_id !== this.episode[i]) { this._resetStream(i); this.episode[i] = m.episode_id; } });
    const drive = this._drive(moments);
    // 2. the free phase under the current parameters: the bootstrap value and the scoring state
    const freePre = this._settle("free", drive, { warm: this.warm });
    const valueNext = this._value(freePre);
    const report = { scores: {}, learning: {} };
    const versionBefore = this.parameterVersion;
    moments.forEach((m, i) => { if (m.has_feedback) report.scores[i] = this.score(this.pending[i].decision.prediction, m); });
    // 3.-4. reward credit of the executed decisions, through their saved eligibility
    Object.assign(report.learning, this._credit(moments, valueNext));
    // 5. repair the world model from (saved drive, executed action, observed consequence)
    Object.assign(report.learning, this._repair(moments));
    // 6. witnessed associations: no episodic store here
    // 7. context advances once per real moment; a fresh free phase under the current parameters
    const freePost = this.parameterVersion !== versionBefore ? this._settle("free_post", drive, { warm: freePre }) : freePre;
    this.context.update(freePost);
    this.contextVersion += 1;
    const closing = moments.map((m) => m.terminated || m.truncated);
    // 8.-9. decide and predict for the streams that continue
    const decisions = new Array(this.streams).fill(null);
    const v = Float64Array.from(freePost.v), a = Float64Array.from(freePost.a);
    moments.forEach((m, i) => {
      this.pending[i] = null;
      if (closing[i]) { this._resetStream(i); v.fill(0, i * n, (i + 1) * n); a.fill(0, i * n, (i + 1) * n); return; }
      if (!m.any_legal) throw Error("a continuing moment offers no legal action");
      decisions[i] = this._decide(i, m, drive, freePost);
    });
    this.warm = brainState(v, Float64Array.from(freePost.s), a, this.streams, freePost.steps);
    // 10. commit
    moments.forEach((m, i) => { this.lastEvent[i] = m.event_id; if (m.has_feedback) this.ledger.real_transitions += 1; });
    this.lastReport = report;
    return decisions;
  }

  _validate(i, m) {
    if (m.event_id <= this.lastEvent[i]) throw Error(`out-of-order or duplicate event ${m.event_id} on stream ${i}`);
    if (m.replay_of !== null) throw Error("a replayed observed event cannot enter step as a new real observation");
    const pending = this.pending[i];
    if (m.has_feedback) {
      if (pending === null) throw Error(`feedback for decision ${m.feedback_for} but no decision is pending on stream ${i}`);
      if (m.feedback_for !== pending.decision.decision_id) throw Error(`feedback names decision ${m.feedback_for}; pending is ${pending.decision.decision_id}`);
      if (m.executed !== pending.decision.action) throw Error(`executed action ${m.executed} differs from the committed action ${pending.decision.action}`);
      if (m.episode_id !== this.episode[i]) throw Error("feedback cannot cross an episode boundary");
    } else if (pending !== null) throw Error(`decision ${pending.decision.decision_id} on stream ${i} awaits feedback; a new observation cannot replace it`);
    if (m.episode_id !== this.episode[i] && m.episode_id < this.episode[i]) throw Error("episodes must not go backwards");
    const joint = this.config.graph.action_fields.reduce((p, k) => p * k, 1);
    if (m.action_mask.length !== joint) throw Error("action_mask must have one entry per joint action");
    for (const name of Object.keys(m.observation)) if (!(name in this.ports.observation_fields)) throw Error(`unknown observation field '${name}'`);
  }

  /** An episode boundary: designated transient state only; learned parameters stay. */
  _resetStream(i) {
    const n = this.n;
    this.context.reset(this.streams, [i]);
    this.elig.fill(0); this.eligBias.fill(0); this.traceCritic.fill(0);
    this.pending[i] = null;
    if (this.warm !== null) { const v = Float64Array.from(this.warm.v), a = Float64Array.from(this.warm.a); v.fill(0, i * n, (i + 1) * n); a.fill(0, i * n, (i + 1) * n); this.warm = brainState(v, this.warm.s ? Float64Array.from(this.warm.s) : null, a, this.warm.batch, this.warm.steps); }
  }

  // -- credit

  _credit(moments, valueNext) {
    const cfg = this.config.actor, decay = cfg.gamma * cfg.lam, E = this.E, n = this.n, width = this.traceCritic.length;
    const credited = [], deltas = zeros(this.streams), td = zeros(this.streams);
    moments.forEach((m, i) => {
      if (!m.has_feedback) return;
      const p = this.pending[i];
      for (let e = 0; e < E; e++) this.elig[e] *= decay;
      for (let k = 0; k < n; k++) this.eligBias[k] *= decay;
      if (p.score_edges !== null) { for (let e = 0; e < E; e++) this.elig[e] += p.score_edges[e]; for (let k = 0; k < n; k++) this.eligBias[k] += p.score_bias[k]; }
      for (let k = 0; k < width; k++) this.traceCritic[k] *= decay;
      for (let k = 0; k < width - 1; k++) this.traceCritic[k] += p.workspace[k];
      this.traceCritic[width - 1] += 1.0;
      if (!m.reward_known) return; // an outcome without a reward event teaches the world model only
      let bootstrap = m.terminated ? 0.0 : valueNext[i];
      if (m.truncated) { // a time limit bootstraps from the final observation, never from a reset one
        const final = this._hypotheticalDrive(m.final_observation !== null ? m.final_observation : m.observation, null, m.goal, i);
        bootstrap = this._value(this._settle("bootstrap", final))[0];
      }
      const error = m.reward + cfg.gamma * bootstrap - p.value;
      td[i] = error;
      const delta = clip(error, -cfg.delta_cap, cfg.delta_cap);
      this.ledger.deltas += 1;
      if (delta !== error) this.ledger.clipped_deltas += 1;
      deltas[i] = delta;
      credited.push(i);
    });
    const out = {};
    if (credited.length) {
      const rows = credited, batch = rows.length;
      out.dopamine = npMean(Float64Array.from(rows, (i) => deltas[i]));
      out.td_error = npMean(Float64Array.from(rows, (i) => Math.abs(td[i])));
      let changed = new Int32Array(0);
      if (this.config.learning && this.config.motor_learning) {
        const stepEdges = zeros(E), stepBias = zeros(n);
        for (const i of rows) { for (let e = 0; e < E; e++) stepEdges[e] += deltas[i] * this.elig[e]; for (let k = 0; k < n; k++) stepBias[k] += deltas[i] * this.eligBias[k]; }
        let any = false;
        for (let e = 0; e < E; e++) { stepEdges[e] /= batch; if (stepEdges[e] !== 0) any = true; }
        for (let k = 0; k < n; k++) { stepBias[k] /= batch; if (stepBias[k] !== 0) any = true; }
        if (any) {
          const report = this.motor.apply(Float64Array.from(stepEdges, (x) => clip(cfg.eta * x, -cfg.step_clip, cfg.step_clip)), Float64Array.from(stepBias, (x) => cfg.eta_bias * x));
          changed = report.changed;
          this._rebind(this.motor);
        }
        const trace = Float64Array.from(this.traceCritic);
        if (cfg.critic_normalize) { const sq = Float64Array.from(trace, (x) => x ** 2); const norm = 1.0 + npSum(sq); for (let k = 0; k < width; k++) trace[k] = trace[k] / norm; }
        const step = Float64Array.from(trace, (x) => cfg.eta_critic * (td[rows[0]] * x));
        for (let k = 0; k < width - 1; k++) this.wCritic[k] += step[k];
        this.bCritic += step[width - 1];
      }
      if (this.onLearn) this.onLearn({ phase: "credit", changed, dopamine: out.dopamine, td_error: out.td_error, rejected: false });
    }
    moments.forEach((m) => { if (m.has_feedback && (m.terminated || m.truncated)) { this.elig.fill(0); this.eligBias.fill(0); this.traceCritic.fill(0); } });
    return out;
  }

  // -- world repair

  /** The records head's repair (Agent._repair_records): the witnessed outcome written into the records the executed reading touched. No settle, no ring, nothing else moves. */
  _repairRecords(moments) {
    let written = 0;
    moments.forEach((m, i) => {
      if (!m.has_feedback) return;
      const p = this.pending[i];
      if (p.code === null || p.code === undefined) return;
      const targets = this._outcomeTargets(m);
      if (!Object.keys(targets).length) return;
      const out = this.records.learn(p.code, targets);
      written += out.written;
      if (this.onRecords) this.onRecords({ kind: "write", index: p.code.plain.index, values: p.code.plain.values, valuedIndex: p.code.valued.index, valuedValues: p.code.valued.values, fields: out.report, written: out.written, prediction: this.records.predict(p.code) });
    });
    this.ledger.record_writes += written;
    return written ? { records_written: written } : {};
  }

  _repair(moments) {
    if (!(this.config.learning && this.config.world_learning) || this.world === null) return {};
    if (this.records !== null) return this._repairRecords(moments);
    const rows = [];
    moments.forEach((m, i) => { if (m.has_feedback) rows.push(i); });
    if (!rows.length) return {};
    const n = this.n, E = this.E, cfg = this.config.world;
    const hasTargets = (t) => Object.keys(t).length > 0;
    const transitions = rows.map((i) => { const p = this.pending[i]; return [p.drive, p.state_v, p.state_a, this._outcomeTargets(moments[i])]; });
    let replayed = 0;
    if (this.config.replay !== null) {
      const cap = this.config.replay.capacity;
      for (const t of transitions) if (hasTargets(t[3])) { if (this.ring.length < cap) this.ring.push(t); else { this.ring[this.ringAt] = t; this.ringAt = (this.ringAt + 1) % cap; } }
      const stored = this.ring.length - transitions.filter((t) => hasTargets(t[3])).length;
      for (let k = 0; k < this.config.replay.per_transition * rows.length; k++) {
        if (stored <= 0) break;
        transitions.push(this.ring[Math.floor(this.replayRng.random() * stored)]);
        replayed += 1;
      }
      this.ledger.replay_writes += replayed;
    }
    // group rows by their observed target mask, per nudge kind
    const drives = [], warmV = [], warmA = [];
    const catTarget = [], catMask = [], catGroups = [], catFields = [], conTarget = [], conMask = [], conFields = [];
    for (const [driveRow, vRow, aRow, targets] of transitions) {
      const tCat = zeros(n), mCat = zeros(n), gCat = new Int32Array(n).fill(-1), tCon = zeros(n), mCon = zeros(n);
      let kCat = 0, kCon = 0;
      this.config.graph.prediction.forEach((f, k) => {
        if (!(f.name in targets)) return;
        const neurons = this.ports.prediction_fields[f.name], { target, known } = targets[f.name];
        if (f.kind === "categorical") { for (const i of neurons) { mCat[i] = 1.0; gCat[i] = k; } tCat[neurons[target[0] | 0]] = 1.0; kCat += 1; }
        else { neurons.forEach((i, j) => { if (known[j]) mCon[i] = 1.0; tCon[i] = target[j]; }); kCon += 1; }
      });
      if (!mCat.some((x) => x !== 0) && !mCon.some((x) => x !== 0)) continue; // all missing: no update from this row
      drives.push(driveRow); warmV.push(vRow); warmA.push(aRow);
      catTarget.push(tCat); catMask.push(mCat); catGroups.push(gCat); catFields.push(kCat);
      conTarget.push(tCon); conMask.push(mCon); conFields.push(kCon);
    }
    if (!drives.length) return {};
    const batch = drives.length, drive = zeros(batch * n), wv = zeros(batch * n), wa = zeros(batch * n);
    drives.forEach((d, b) => { drive.set(d, b * n); wv.set(warmV[b], b * n); wa.set(warmA[b], b * n); });
    const free = this._settle("repair_free", drive, { warm: brainState(wv, null, wa, batch, 0) });
    let converged = this.lastConverged;
    const edgesTotal = zeros(E), neuronsTotal = zeros(n);
    let contributing = 0;
    const smoothing = cfg.label_smoothing;
    for (const [kind, targetsK, masksK, groupsK, fieldsK] of [["categorical", catTarget, catMask, catGroups, catFields], ["continuous", conTarget, conMask, conTarget.map(() => null), conFields]]) {
      const keys = new Map();
      masksK.forEach((mask, r) => {
        if (!mask.some((x) => x !== 0)) return;
        const key = Array.from(mask).join(",") + "|" + (groupsK[r] !== null ? Array.from(groupsK[r]).join(",") : "");
        if (!keys.has(key)) keys.set(key, []);
        keys.get(key).push(r);
      });
      for (const groupRows of keys.values()) {
        const sel = groupRows, mask = masksK[sel[0]], groups = groupsK[sel[0]];
        const beta = cfg.beta / Math.max(1, fieldsK[sel[0]]);
        const target = zeros(sel.length * n);
        sel.forEach((r, k) => target.set(targetsK[r], k * n));
        const temperature = kind === "categorical" ? cfg.temperature : null;
        if (kind === "categorical" && smoothing > 0) {
          const ids = [...new Set(Array.from(groups).filter((g) => g >= 0))].sort((x, y) => x - y);
          for (const g of ids) {
            const members = []; for (let i = 0; i < n; i++) if (groups[i] === g) members.push(i);
            for (let k = 0; k < sel.length; k++) for (const i of members) target[k * n + i] = (1.0 - smoothing) * target[k * n + i] + smoothing / members.length;
          }
        }
        const rowsDrive = driveRows(drive, n, sel), warm = stateRows(free, n, sel);
        const plus = this._settle("repair_plus", rowsDrive, { warm, nudge: { target, mask, beta, softmaxT: temperature, groups } });
        converged = converged && this.lastConverged;
        const minus = this._settle("repair_minus", rowsDrive, { warm, nudge: { target, mask, beta: -beta, softmaxT: temperature, groups } });
        converged = converged && this.lastConverged;
        let gap = 0.0;
        for (let k = 0; k < plus.s.length; k++) { const d = Math.abs(plus.s[k] - minus.s[k]); if (d > gap) gap = d; }
        if (gap > cfg.max_phase_gap) converged = false; // the nudged phases sit in different attractors: no gradient here
        const [e, b] = this.world.contrast(warm, plus, minus);
        for (let k = 0; k < E; k++) edgesTotal[k] += e[k] * sel.length;
        for (let k = 0; k < n; k++) neuronsTotal[k] += b[k] * sel.length;
        contributing += sel.length;
      }
    }
    if (!contributing) return {};
    if (cfg.reject_unconverged && !converged) {
      this.ledger.rejected_updates += 1;
      if (this.onLearn) this.onLearn({ phase: "world", changed: new Int32Array(0), dopamine: null, rejected: true });
      return { world_rejected: 1.0 };
    }
    let edges = Float64Array.from(edgesTotal, (x) => x / batch), neurons = Float64Array.from(neuronsTotal, (x) => x / batch);
    [edges, neurons] = this._adaptive(this.world, edges, neurons, cfg.momentum, cfg.normalize);
    edges = Float64Array.from(edges, (x) => clip(cfg.eta * x, -cfg.step_clip, cfg.step_clip));
    const report = this.world.apply(edges, Float64Array.from(neurons, (x) => cfg.eta_bias * x));
    this.world.contrastUpdates += 1;
    this._rebind(this.world);
    if (this.onLearn) this.onLearn({ phase: "world", changed: report.changed, dopamine: null, rejected: false, scale_step: report.scale_step });
    return { world_scale_step: report.scale_step, world_bias_step: report.bias_step, replayed };
  }

  _adaptive(head, edges, neurons, momentum, normalize) {
    const rawE = edges, rawN = neurons, count = head.contrastUpdates + 1;
    if (momentum > 0) {
      for (let k = 0; k < edges.length; k++) head.velocity[k] = momentum * head.velocity[k] + (1 - momentum) * edges[k];
      for (let k = 0; k < neurons.length; k++) head.velocityBias[k] = momentum * head.velocityBias[k] + (1 - momentum) * neurons[k];
      const correction = 1.0 - momentum ** count;
      edges = Float64Array.from(head.velocity, (x) => x / correction); neurons = Float64Array.from(head.velocityBias, (x) => x / correction);
    }
    if (normalize > 0) {
      for (let k = 0; k < rawE.length; k++) head.secondMoment[k] = normalize * head.secondMoment[k] + (1 - normalize) * rawE[k] ** 2;
      for (let k = 0; k < rawN.length; k++) head.secondMomentBias[k] = normalize * head.secondMomentBias[k] + (1 - normalize) * rawN[k] ** 2;
      const correction = 1.0 - normalize ** count;
      edges = Float64Array.from(edges, (x, k) => x / (Math.sqrt(head.secondMoment[k] / correction) + head.normalizeFloor));
      neurons = Float64Array.from(neurons, (x, k) => x / (Math.sqrt(head.secondMomentBias[k] / correction) + head.normalizeFloor));
    }
    return [edges, neurons];
  }

  // -- decisions

  /** Joint-action probabilities over the legal entries, and per-field marginals, from the motor activations. */
  _policy(state, row, mask) {
    const cfg = this.config.actor, n = this.n, s = state.s, sizes = this.config.graph.action_fields;
    const legalJoint = []; for (let j = 0; j < mask.length; j++) if (mask[j]) legalJoint.push(j);
    const perField = [];
    this.ports.motor_slots.forEach((slot, k) => {
      const legal = new Uint8Array(sizes[k]);
      for (const joint of legalJoint) legal[this._splitAction(joint)[k]] = 1;
      const z = zeros(sizes[k]);
      let top = -Infinity;
      for (let c = 0; c < sizes[k]; c++) { z[c] = legal[c] ? s[row * n + slot[c]] / cfg.temperature : -Infinity; if (legal[c] && z[c] > top) top = z[c]; }
      for (let c = 0; c < sizes[k]; c++) z[c] = Math.exp(z[c] - top);
      const sum = npSum(z);
      perField.push(Float64Array.from(z, (x) => x / sum));
    });
    const joint = zeros(mask.length);
    for (const j of legalJoint) { const parts = this._splitAction(j); let prod = 1.0; for (let k = 0; k < sizes.length; k++) prod = prod * perField[k][parts[k]]; joint[j] = prod; }
    const total = npSum(joint);
    return [Float64Array.from(joint, (x) => x / total), perField];
  }

  _decide(i, m, drive, free) {
    const cfg = this.config.actor, n = this.n, mask = m.action_mask;
    const legal = []; for (let j = 0; j < mask.length; j++) if (mask[j]) legal.push(j);
    let controller = this.config.controller;
    let budget = {}, prediction = {}, scoreEdges = null, scoreBias = null, action;
    const [probabilities] = this._policy(free, i, mask);
    if (controller === "planner") {
      if (this.planner === null) throw Error("a planner controller needs a planner callable");
      if (cfg.epsilon > 0 && this.rng[i].random() < cfg.epsilon) { action = legal[Math.floor(this.rng[i].random() * legal.length)]; controller = "exploration"; }
      else {
        const out = this.planner(this, i, m, drive, free);
        action = out.action | 0; prediction = out.prediction || {}; budget = out.budget || {};
        if (!mask[action]) throw Error("the planner chose an illegal action");
      }
    } else {
      if (cfg.epsilon > 0 && this.rng[i].random() < cfg.epsilon) { action = legal[Math.floor(this.rng[i].random() * legal.length)]; controller = "exploration"; }
      else {
        action = this.rng[i].choice(probabilities);
        if (!mask[action]) throw Error("sampled an illegal action");
        if (legal.length > 1 && this.config.learning && this.config.motor_learning) [scoreEdges, scoreBias] = this._score(i, m, action, drive, free);
        else if (legal.length > 1) { /* frozen: no score */ }
        else { scoreEdges = zeros(this.E); scoreBias = zeros(n); }
      }
    }
    const rowDrive = drive.subarray(i * n, (i + 1) * n);
    const driveA = this._withAction(rowDrive, 1, [action]);
    let stateV, stateA, code = null;
    if (this.records !== null) {
      // the executed reading's code: the habituation mean adapts here and only here
      const readings = this._readings(driveA, 1);
      code = this.records.code(readings, 1, this.config.learning && this.config.world_learning)[0];
      if (!Object.keys(prediction).length && this.config.graph.prediction.length) prediction = this.records.predict(code);
      stateV = free.v.slice(i * n, (i + 1) * n); stateA = free.a.slice(i * n, (i + 1) * n);
      if (this.onRecords) this.onRecords({ kind: "code", index: code.plain.index, values: code.plain.values, valuedIndex: code.valued.index, valuedValues: code.valued.values, reading: Float64Array.from(this.records.lastInput), blockNorm: Float64Array.from(this.records.blockNorm), prediction: this.records.predict(code), controller, action });
    } else if (!Object.keys(prediction).length && this.config.graph.prediction.length) {
      const predicted = this._settle("predict", driveA, { warm: stateRows(free, n, [i]) });
      prediction = this.decode(predicted, 0);
      stateV = predicted.v.slice(0, n); stateA = predicted.a.slice(0, n);
    } else { stateV = free.v.slice(i * n, (i + 1) * n); stateA = free.a.slice(i * n, (i + 1) * n); }
    const logProbability = Math.log(Math.max(probabilities[action], 1e-300));
    const decision = {
      decision_id: this.nextDecisionId, event_id: m.event_id, stream: i, parameter_version: this.parameterVersion, context_version: this.contextVersion,
      action, action_mask: Uint8Array.from(mask), probabilities, log_probability: logProbability, controller, prediction, uncertainty: {}, budget, intention: null,
    };
    this.nextDecisionId += 1;
    const workspace = Float64Array.from(this.ports.workspace, (j) => free.s[i * n + j]);
    this.pending[i] = { decision, drive: driveA, state_v: stateV, state_a: stateA, value: this._value(free)[i], score_edges: controller === "actor" ? scoreEdges : null, score_bias: controller === "actor" ? scoreBias : null, workspace, code };
    return decision;
  }

  /** The actor's per-synapse score of the sampled action: masked centred contrast (softmax-grouped nudges). */
  _score(i, m, action, drive, free) {
    const cfg = this.config.actor, n = this.n, E = this.E;
    const mask = zeros(n), groups = new Int32Array(n).fill(-1), target = zeros(n), parts = this._splitAction(action), sizes = this.config.graph.action_fields;
    let fields = 0;
    const legalJoint = []; for (let j = 0; j < m.action_mask.length; j++) if (m.action_mask[j]) legalJoint.push(j);
    this.ports.motor_slots.forEach((slot, k) => {
      const legal = new Uint8Array(sizes[k]);
      for (const joint of legalJoint) legal[this._splitAction(joint)[k]] = 1;
      let count = 0; for (const x of legal) count += x;
      if (count < 2) return; // a forced field has no policy-score derivative
      for (let c = 0; c < sizes[k]; c++) if (legal[c]) { mask[slot[c]] = 1.0; groups[slot[c]] = k; }
      target[slot[parts[k]]] = 1.0;
      fields += 1;
    });
    if (fields === 0) return [zeros(E), zeros(n)];
    const beta = cfg.beta / fields;
    const rowDrive = driveRows(drive, n, [i]), warm = stateRows(free, n, [i]);
    const plus = this._settle("score_plus", rowDrive, { warm, nudge: { target, mask, beta, softmaxT: cfg.temperature, groups } });
    let converged = this.lastConverged;
    const minus = this._settle("score_minus", rowDrive, { warm, nudge: { target, mask, beta: -beta, softmaxT: cfg.temperature, groups } });
    converged = converged && this.lastConverged;
    let gap = 0.0;
    for (let k = 0; k < plus.s.length; k++) { const d = Math.abs(plus.s[k] - minus.s[k]); if (d > gap) gap = d; }
    if (gap > this.config.world.max_phase_gap) converged = false;
    if (this.config.world.reject_unconverged && !converged) { this.ledger.rejected_updates += 1; return [zeros(E), zeros(n)]; }
    const [edges, neurons] = this.motor.contrastRows(warm, plus, minus);
    return [edges.slice(0, E), neurons.slice(0, n)];
  }

  // -- read-only imagination

  /** A one-row drive for a hypothetical observation in the stream's current context. */
  _hypotheticalDrive(observation, observed, goal, row, context = true) {
    const n = this.n, drive = this.encodeObservation(observation, observed);
    if (this.ports.goal.length) {
      if (goal === null || goal === undefined || goal.length !== this.ports.goal.length) throw Error("this brain needs a goal of the declared width to imagine");
      for (let k = 0; k < this.ports.goal.length; k++) drive[this.ports.goal[k]] = this.config.graph.input_gain * goal[k];
    }
    if (context && this.context.streams === this.streams) {
      for (let j = 0; j < this.context.width; j++) drive[this.ports.context[j]] = this.context.amplitude * this.context.trace[row * this.context.width + j];
      const recall = this.ports.recall, width = recall.length;
      if (this.recallDrive !== null && width) for (let k = 0; k < width; k++) drive[recall[k]] += this.lastRecall[row * width + k]; // Agent._hypothetical_drive: the last real read
    }
    return drive;
  }

  /** The consequence the model predicts for a hypothetical observation and action; writes nothing (a record read with a records head, a settle otherwise). */
  predict(observation, action, { observed = null, goal = null, row = 0, context = true } = {}) {
    const w = this.config.world;
    const drive = this._withAction(this._hypotheticalDrive(observation, observed, goal, row, context), 1, [action]);
    if (this.records !== null) {
      this.ledger.imagined += 1;
      const code = this.records.code(this._readings(drive, 1), 1, false)[0];
      if (this.onRecords) this.onRecords({ kind: "imagine", batch: 1, index: code.plain.index, values: code.plain.values });
      return this.records.predict(code);
    }
    const state = this.settleBatch(drive, { steps: w.free_steps, tolerance: w.imagine_tolerance === null ? w.tolerance : w.imagine_tolerance, phase: "imagine" });
    this.ledger.imagined += 1;
    return this.decode(state, 0);
  }

  /** `predict` for several hypothetical (observation, action) pairs in one settle (one batched code with a records head). `observed`: one flag map per observation, or null entries (every field observed); an imagined reading carries the same missing flags a real moment would. */
  predictBatch(observations, actions, goal = null, { observed = null, row = 0, context = true } = {}) {
    if (observations.length !== actions.length || !observations.length) throw Error("one action per hypothetical observation");
    if (observed !== null && observed !== undefined && observed.length !== observations.length) throw Error("one observed map per hypothetical observation");
    const n = this.n, w = this.config.world, batch = observations.length;
    const drive = zeros(batch * n);
    observations.forEach((obs, b) => drive.set(this._hypotheticalDrive(obs, observed ? observed[b] ?? null : null, goal, row, context), b * n));
    const withAction = this._withAction(drive, batch, actions.map((a) => a | 0));
    this.ledger.imagined += batch;
    if (this.records !== null) {
      const codes = this.records.code(this._readings(withAction, batch), batch, false);
      if (this.onRecords) this.onRecords({ kind: "imagine", batch, index: codes[0].plain.index, values: codes[0].plain.values });
      return codes.map((code) => this.records.predict(code));
    }
    const state = this.settleBatch(withAction, { steps: w.free_steps, tolerance: w.imagine_tolerance === null ? w.tolerance : w.imagine_tolerance, phase: "imagine" });
    return actions.map((_, k) => this.decode(state, k));
  }

  // -- inspection and streams

  parameters() {
    return { neurons: this.n, directed_synapses: this.E, world_parameters: this.world ? this.world.parameters() : 0, record_parameters: this.records ? this.records.parameters() : 0, motor_parameters: this.motor.parameters(), critic_parameters: this.wCritic.length + 1 };
  }

  /** Attach the stream to a new environment: fresh cursors and transient state; refused while a decision awaits feedback. */
  newStream(row = 0) {
    if (this.pending[row] !== null) throw Error("a pending decision must receive its outcome before the stream changes");
    this._resetStream(row);
    this.lastEvent[row] = -1; this.episode[row] = -1;
  }

  /** Drop a decision awaiting its outcome without feedback (evaluation copies only). */
  abandon(row = 0) { this.pending[row] = null; }
}
