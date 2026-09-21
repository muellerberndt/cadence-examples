// The composer's brain in the browser: the forward pass of one Cadence record patch and its playing rule.
// The same arithmetic as the Python library: a gated linear context, a k-winner record code over a fixed random
// projection regenerated from the seed, a record read added to a linear readout. No learning happens here.

export function mulberry(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6D2B79F5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), state | 1) >>> 0;
    t = (t ^ ((t + (Math.imul(t ^ (t >>> 7), t | 61) >>> 0)) >>> 0)) >>> 0;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// n standard normals by Box-Muller over pairs of draws: all radii first, then all angles, cosines then sines.
export function normals(random, n) {
  const pairs = (n + 1) >> 1, u = new Float64Array(2 * pairs), out = new Float64Array(2 * pairs);
  for (let i = 0; i < 2 * pairs; i++) u[i] = random();
  for (let i = 0; i < pairs; i++) {
    const radius = Math.sqrt(-2 * Math.log(Math.max(u[i], 1e-12))), angle = 2 * Math.PI * u[pairs + i];
    out[i] = radius * Math.cos(angle); out[pairs + i] = radius * Math.sin(angle);
  }
  return out.subarray(0, n);
}

export class Brain {
  constructor(model, params, recordsY, recordsMean) {
    this.model = model; this.n = model.inputs; this.hidden = model.hidden; this.outputs = model.outputs;
    const p = new Float64Array(params); let at = 0; this.p = {};
    for (const [name, shape] of model.params_order) { const size = shape.reduce((a, b) => a * b, 1); this.p[name] = p.subarray(at, at + size); at += size; }
    this.y = new Float32Array(recordsY); this.mean = new Float64Array(recordsMean);
    const r = model.records; this.cells = r.cells; this.active = r.active; this.reading = r.reading;
    const random = mulberry(r.seed), root = Math.sqrt(this.reading);
    this.projection = normals(random, this.reading * this.cells).slice();
    for (let i = 0; i < this.projection.length; i++) this.projection[i] /= root;
    this.offset = normals(random, this.cells).slice();
    for (let i = 0; i < this.cells; i++) this.offset[i] *= r.bias;
    this.scale = new Float64Array(this.hidden); this.timescales = [];
    for (let j = 0; j < this.hidden; j++) {
      const tau = Math.exp(Math.log(2) + (Math.log(model.slowest) - Math.log(2)) * j / (this.hidden - 1)), retention = 1 - 1 / tau;
      this.scale[j] = Math.sqrt((1 + retention) / (1 - retention)); this.timescales.push(Math.round(tau * 100) / 100);
    }
    this.inputScale = Math.sqrt(this.n) / model.input_norm;
    this.h = new Float64Array(this.hidden); this.x = new Float64Array(this.reading); this.drive = new Float64Array(this.cells);
  }

  reset() { this.h.fill(0); }

  // One half-beat: the input ports u (length n) in, everything the page draws out.
  step(u) {
    const {G, g, B, b, C, c} = this.p, n = this.n, H = this.hidden, gate = new Array(H);
    for (let j = 0; j < H; j++) {
      let a = g[j], z = b[j]; const row = j * n;
      for (let i = 0; i < n; i++) { const v = u[i]; if (v !== 0) { a += G[row + i] * v; z += B[row + i] * v; } }
      const l = 1 / (1 + Math.exp(-a)); gate[j] = l; this.h[j] = l * this.h[j] + (1 - l) * Math.tanh(z);
    }
    const x = this.x, cells = this.cells;
    for (let i = 0; i < n; i++) x[i] = u[i] * this.inputScale - this.mean[i];
    for (let j = 0; j < H; j++) x[n + j] = this.h[j] * this.scale[j] - this.mean[n + j];
    const drive = this.drive; drive.set(this.offset);
    for (let i = 0; i < this.reading; i++) { const v = x[i]; if (v === 0) continue; const row = i * cells; const P = this.projection; for (let k = 0; k < cells; k++) drive[k] += v * P[row + k]; }
    // the k largest drives, by a min-heap of size k
    const K = this.active, heap = new Int32Array(K); let size = 0;
    const less = (a, b) => drive[a] < drive[b];
    for (let k = 0; k < cells; k++) {
      if (size < K) { let i = size++; heap[i] = k; while (i > 0) { const parent = (i - 1) >> 1; if (less(heap[i], heap[parent])) { const t = heap[i]; heap[i] = heap[parent]; heap[parent] = t; i = parent; } else break; } }
      else if (drive[k] > drive[heap[0]]) { heap[0] = k; let i = 0; for (;;) { const l = 2 * i + 1, r = l + 1; let m = i; if (l < K && less(heap[l], heap[m])) m = l; if (r < K && less(heap[r], heap[m])) m = r; if (m === i) break; const t = heap[i]; heap[i] = heap[m]; heap[m] = t; i = m; } }
    }
    const winners = Array.from(heap).sort((a, b) => a - b); let norm = 0;
    const values = winners.map(k => { const v = Math.max(drive[k], 0); norm += v * v; return v; }); norm = Math.sqrt(norm);
    const code = values.map(v => norm > 0 ? v / norm : v), O = this.outputs, read = new Array(O).fill(0), push = new Array(K);
    winners.forEach((cell, i) => { const row = cell * O; let sq = 0; for (let k = 0; k < O; k++) { const w = code[i] * this.y[row + k]; read[k] += w; sq += w * w; } push[i] = Math.sqrt(sq); });
    const out = new Array(O);
    for (let k = 0; k < O; k++) { let s = c[k]; const row = k * H; for (let j = 0; j < H; j++) s += C[row + j] * this.h[j]; out[k] = s + read[k]; }
    return {gate, h: Array.from(this.h), read, out, cells: winners, cell_weights: code, cell_push: push};
  }
}

const argmax = (a, s, e) => { let m = s; for (let i = s + 1; i < e; i++) if (a[i] > a[m]) m = i; return m; };
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

// The event actually played from an output row. 'argmax' plays the highest-scoring slice; 'sample' draws a change
// point with the predicted probability times `energy` and there makes a roll or a retrigger from the slice heard before.
export function executed(out, L, retriggers, mode, random, previous, energy = 1) {
  const e = new Array(L.event_ports).fill(0);
  const change = mode === 'sample' ? random() < clamp(out[L.change] * energy, 0, 1) : out[L.change] > 0.5;
  e[L.change] = change ? 1 : 0;
  for (let k = 0; k < L.texture_ports; k++) e[L.texture_start + k] = clamp(out[L.texture_start + k], 0, 1);
  if (out[L.drum_on] > 0.5) {
    const heard = previous && previous[L.drum_on] > 0.5; let crop;
    if (mode === 'sample' && change && heard && random() < 0.5) crop = argmax(previous, 0, L.crops);
    else if (mode === 'sample' && change) {
      const w = retriggers.map(k => Math.pow(Math.max(out[k], 0), 2) + 1e-9), total = w.reduce((a, b) => a + b, 0); let draw = random() * total, pick = retriggers.length - 1;
      for (let i = 0; i < w.length; i++) { draw -= w[i]; if (draw <= 0) { pick = i; break; } }
      crop = retriggers[pick];
    } else crop = argmax(out, 0, L.crops);
    e[crop] = 1; e[L.drum_on] = 1; e[L.drum_gain] = clamp(out[L.drum_gain], 0, 1);
  }
  if (out[L.bass_on] > 0.5) { e[argmax(out, L.note_start, L.note_start + L.notes)] = 1; e[L.bass_on] = 1; e[L.bass_hold] = out[L.bass_hold] > 0.5 ? 1 : 0; }
  return e;
}

// Sixteen bars (or more) from silence. A generator: each next() computes one half-beat and yields its trace step.
export function* compose(brain, {bars = 16, mode = 'sample', energy = 1, seed = 1} = {}) {
  const model = brain.model, L = model.layout, random = mulberry(seed), horizon = 8 * bars;
  brain.reset(); let previous = null;
  for (let t = 0; t < horizon; t++) {
    const u = new Array(model.inputs).fill(0), heard = t === 0 ? model.count_in : previous;
    if (t === 0) u[0] = 1;
    u[L.clock_start + (t % 8)] = 1;
    for (let k = 0; k < L.event_ports; k++) u[L.heard_start + k] = heard[k];
    const step = brain.step(u);
    previous = executed(step.out, L, model.retriggers, mode, random, mode === 'sample' ? heard : null, energy);
    yield Object.assign(step, {t, phase: 'generated', heard: Array.from(heard), clock: t % 8, played: previous});
  }
}
