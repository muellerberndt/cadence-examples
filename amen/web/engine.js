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
    this.pristine = {y: this.y.slice(), mean: this.mean.slice(), seen: model.records.seen || 0, inputNorm: model.input_norm}; this.seen = this.pristine.seen;
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
    this.inputNorm = model.input_norm; this.inputScale = Math.sqrt(this.n) / this.inputNorm;
    this.h = new Float64Array(this.hidden); this.x = new Float64Array(this.reading); this.drive = new Float64Array(this.cells);
  }

  reset() { this.h.fill(0); }

  // Forget what a dub wrote: the record tables, their running mean and count return to the trained ones.
  forget() { this.y.set(this.pristine.y); this.mean.set(this.pristine.mean); this.seen = this.pristine.seen; this.inputNorm = this.pristine.inputNorm; this.inputScale = Math.sqrt(this.n) / this.inputNorm; }

  // The reading of a moment: the input and the context, both with unit variance per unit.
  readingOf(u, h) { const r = new Float64Array(this.reading); for (let i = 0; i < this.n; i++) r[i] = u[i] * this.inputScale; for (let j = 0; j < this.hidden; j++) r[this.n + j] = h[j] * this.scale[j]; return r; }

  // The k-winner code of a reading under the current mean: the winning cells and their unit-length positive drives.
  codeOf(reading) {
    const cells = this.cells, drive = this.drive; drive.set(this.offset);
    for (let i = 0; i < this.reading; i++) { const v = reading[i] - this.mean[i]; if (v === 0) continue; const row = i * cells, P = this.projection; for (let k = 0; k < cells; k++) drive[k] += v * P[row + k]; }
    const K = this.active, heap = new Int32Array(K); let size = 0; const less = (a, b) => drive[a] < drive[b];
    for (let k = 0; k < cells; k++) {
      if (size < K) { let i = size++; heap[i] = k; while (i > 0) { const parent = (i - 1) >> 1; if (less(heap[i], heap[parent])) { const s = heap[i]; heap[i] = heap[parent]; heap[parent] = s; i = parent; } else break; } }
      else if (drive[k] > drive[heap[0]]) { heap[0] = k; let i = 0; for (;;) { const l = 2 * i + 1, r = l + 1; let m = i; if (l < K && less(heap[l], heap[m])) m = l; if (r < K && less(heap[r], heap[m])) m = r; if (m === i) break; const s = heap[i]; heap[i] = heap[m]; heap[m] = s; i = m; } }
    }
    const winners = Array.from(heap).sort((a, b) => a - b); let norm = 0;
    const values = winners.map(k => { const v = Math.max(drive[k], 0); norm += v * v; return v; }); norm = Math.sqrt(norm);
    return {winners, code: values.map(v => norm > 0 ? v / norm : v)};
  }

  // Write what was witnessed, as the library does after an observed path: the readings first move the running mean,
  // then each moment's code takes the residual the slow readout left, by the delta rule at the record rate, in order.
  memorize(moments, rate = this.model.records.rate) {
    const r = this.model.records, {C, c} = this.p, H = this.hidden, O = this.outputs;
    // the witnessed inputs first move the running input norm, and the readings are taken under the new one
    for (const m of moments) { let sq = 0; for (let i = 0; i < this.n; i++) sq += m.u[i] * m.u[i]; this.inputNorm += 0.01 * (Math.max(Math.sqrt(sq), 1e-6) - this.inputNorm); }
    this.inputScale = Math.sqrt(this.n) / this.inputNorm;
    for (const m of moments) m.reading = this.readingOf(m.u, m.h);
    for (const m of moments) { this.seen += 1; const rate = Math.max(r.habituation, 1 / this.seen); for (let i = 0; i < this.reading; i++) this.mean[i] += rate * (m.reading[i] - this.mean[i]); }
    for (const m of moments) {
      const {winners, code} = this.codeOf(m.reading);
      for (let k = 0; k < O; k++) {
        let slow = c[k]; const row = k * H; for (let j = 0; j < H; j++) slow += C[row + j] * m.h[j];
        let read = 0; for (let i = 0; i < winners.length; i++) read += code[i] * this.y[winners[i] * O + k];
        const error = m.target[k] - slow - read; for (let i = 0; i < winners.length; i++) this.y[winners[i] * O + k] += rate * code[i] * error;
      }
    }
  }

  // One half-beat: the input ports u (length n) in, everything the page draws out.
  step(u) {
    const {G, g, B, b, C, c} = this.p, n = this.n, H = this.hidden, gate = new Array(H);
    for (let j = 0; j < H; j++) {
      let a = g[j], z = b[j]; const row = j * n;
      for (let i = 0; i < n; i++) { const v = u[i]; if (v !== 0) { a += G[row + i] * v; z += B[row + i] * v; } }
      const l = 1 / (1 + Math.exp(-a)); gate[j] = l; this.h[j] = l * this.h[j] + (1 - l) * Math.tanh(z);
    }
    const reading = this.readingOf(u, this.h), {winners, code} = this.codeOf(reading);
    const O = this.outputs, K = this.active, read = new Array(O).fill(0), push = new Array(K);
    winners.forEach((cell, i) => { const row = cell * O; let sq = 0; for (let k = 0; k < O; k++) { const w = code[i] * this.y[row + k]; read[k] += w; sq += w * w; } push[i] = Math.sqrt(sq); });
    const out = new Array(O);
    for (let k = 0; k < O; k++) { let s = c[k]; const row = k * H; for (let j = 0; j < H; j++) s += C[row + j] * this.h[j]; out[k] = s + read[k]; }
    return {gate, h: Array.from(this.h), read, out, cells: winners, cell_weights: code, cell_push: push};
  }
}

const argmax = (a, s, e) => { let m = s; for (let i = s + 1; i < e; i++) if (a[i] > a[m]) m = i; return m; };
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

// The event actually played from an output row. 'argmax' plays the highest-scoring slice; 'sample' draws a change
// point with the predicted probability times `energy` and there makes one of three moves from the slice heard before,
// with equal probability: a roll (repeat it), a retrigger (restart the break at a bar start) or a bar jump (the same
// position in another bar of the break); where a move has a choice, the brain's squared scores decide.
// With `bassPower` set, a new bass note is drawn with probability proportional to its score to that power, the
// brain's own spread over plausible notes; otherwise the highest-scoring note plays.
export function executed(out, L, retriggers, mode, random, previous, energy = 1, bassPower = 0, signature = null) {
  const e = new Array(L.event_ports).fill(0);
  const change = mode === 'sample' ? random() < clamp(out[L.change] * energy, 0, 1) : out[L.change] > 0.5;
  e[L.change] = change ? 1 : 0;
  for (let k = 0; k < L.texture_ports; k++) e[L.texture_start + k] = clamp(out[L.texture_start + k], 0, 1);
  if (out[L.drum_on] > 0.5) {
    const heard = previous && previous[L.drum_on] > 0.5; let crop;
    const pick = (options, prefer) => { const w = options.map((k, i) => (Math.pow(Math.max(out[k], 0), 2) + 1e-9) * (prefer ? prefer[i] : 1)), total = w.reduce((a, b) => a + b, 0); let draw = random() * total;
      for (let i = 0; i < w.length; i++) { draw -= w[i]; if (draw <= 0) return options[i]; } return options[options.length - 1]; };
    const move = mode === 'sample' && change ? random() : -1, shares = signature ? signature.moves : [1 / 3, 1 / 3, 1 / 3];
    if (move >= 0 && heard && move < shares[0]) crop = argmax(previous, 0, L.crops);
    else if (move >= 0 && heard && move >= shares[0] + shares[1]) { const next = (argmax(previous, 0, L.crops) + 1) % L.crops; crop = pick([8, 16, 24].map(d => (next + d) % L.crops), signature && signature.jumps); }
    else if (move >= 0) crop = pick(retriggers, signature && signature.retriggers);
    else crop = argmax(out, 0, L.crops);
    e[crop] = 1; e[L.drum_on] = 1; e[L.drum_gain] = clamp(out[L.drum_gain], 0, 1);
  }
  if (out[L.bass_on] > 0.5) {
    const hold = out[L.bass_hold] > 0.5; let note = argmax(out, L.note_start, L.note_start + L.notes);
    if (bassPower > 0 && !hold) {
      let total = 0; const w = []; for (let k = 0; k < L.notes; k++) { const v = Math.pow(Math.max(out[L.note_start + k], 0), bassPower); w.push(v); total += v; }
      if (total > 0) { let draw = random() * total; for (let k = 0; k < L.notes; k++) { draw -= w[k]; if (draw <= 0) { note = L.note_start + k; break; } } }
    }
    e[note] = 1; e[L.bass_on] = 1; e[L.bass_hold] = hold ? 1 : 0;
  }
  return e;
}

// Sixteen bars (or more) from silence. A generator: each next() computes one half-beat and yields its trace step.
// `variation` in [0, 1] lets each dub differ: over the first `riffBars` the bass is drawn from the brain's note
// scores (power 8 at 0+, down to 2 at 1), which sets the key and the riff it then hears and continues; afterwards a
// note is drawn only where a change point fires. At 0 every choice is the highest score.
// Each dub also draws a signature from its seed: which bar of the break it enters on, its shares of the three departure
// moves (roll, retrigger, bar jump) and its preferred targets, so a dub's edits recur within it and differ between dubs.
// The dub's drum pattern is its own opening: over the first `signature.loopBars` bars departures are drawn at a raised
// probability, and afterwards the slice played at the cycle start and at each of the opening's departures returns at the same
// place in every cycle, unless the brain draws a fresh departure there. Between those places the brain plays on from what it
// hears, so a fill stays local and the groove comes back.
// With `memory` set, the brain is its own primer: after `memoryBars` it writes what it played into its records, as the
// library does for a heard track, and plays on from that memory; the dub's own pattern then returns every four bars.
export function* compose(brain, {bars = 16, mode = 'sample', energy = 1, seed = 1, variation = 0, riffBars = 2, memory = false, memoryBars = 4, memoryRate = undefined, openingEnergy = undefined} = {}) {
  const model = brain.model, L = model.layout, random = mulberry(seed), horizon = 8 * bars, moments = [];
  const dice = mulberry((seed ^ 0x2c1b3c6d) >>> 0), v = mode === 'sample' ? Math.min(1, Math.max(0, variation)) : 0;
  const spread = n => { const d = Array.from({length: n}, () => -Math.log(1 - dice())), total = d.reduce((a, b) => a + b, 0); return d.map(x => (1 - v) / n + v * x / total); };
  const signature = v > 0 ? {entry: Math.floor(dice() * 4), moves: spread(3), retriggers: spread(model.retriggers.length), jumps: spread(3), loopBars: [1, 2, 2, 2, 4, 4][Math.floor(dice() * 6)]} : null;
  const cycle = signature ? 8 * signature.loopBars : 0, pattern = [];
  const countIn = Array.from(model.count_in);
  if (signature) { const was = argmax(countIn, 0, L.crops); countIn[was] = 0; countIn[(8 * signature.entry + L.crops - 1) % L.crops] = 1; }
  brain.forget(); brain.reset(); let previous = null;
  for (let t = 0; t < horizon; t++) {
    const u = new Array(model.inputs).fill(0), heard = t === 0 ? countIn : previous;
    if (t === 0) u[0] = 1;
    u[L.clock_start + (t % 8)] = 1;
    for (let k = 0; k < L.event_ports; k++) u[L.heard_start + k] = heard[k];
    const step = brain.step(u);
    const power = variation > 0 && mode === 'sample' ? 8 - 6 * Math.min(1, variation) : 0, riff = t < 8 * riffBars;
    const opening = memory && t < 8 * memoryBars, setting = cycle > 0 && t < cycle;
    const heat = setting && energy > 0 ? Math.max(energy, 1) * (1 + 4 * v) : opening && openingEnergy !== undefined ? openingEnergy : energy;
    let event = executed(step.out, L, model.retriggers, mode, random, mode === 'sample' ? heard : null, heat, (riff || opening) ? power : 0, signature);
    if (!riff && power > 0 && event[L.change] > 0.5) { const drawn = executed(step.out, L, model.retriggers, 'argmax', random, null, energy, power); for (let k = L.note_start; k < L.note_start + L.notes; k++) event[k] = drawn[k]; }
    if (cycle > 0 && event[L.drum_on] > 0.5) {
      const crop = argmax(event, 0, L.crops), fresh = event[L.change] > 0.5;
      if (setting) pattern[t] = {crop, edit: fresh};
      else { const want = pattern[t % cycle]; if (want && !fresh && (t % cycle === 0 || want.edit) && crop !== want.crop) { event[crop] = 0; event[want.crop] = 1; event[L.change] = 1; event.returned = true; } }
    }
    previous = event;
    if (memory && t < 8 * memoryBars) { moments.push({u, h: step.h, target: event}); if (t === 8 * memoryBars - 1) brain.memorize(moments, memoryRate); }
    yield Object.assign(step, {t, phase: 'generated', heard: Array.from(heard), clock: t % 8, played: previous, signature});
  }
}
