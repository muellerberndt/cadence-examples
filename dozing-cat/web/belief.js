// The belief patch in the browser: cadence's BeliefPatch (src/cadence/belief.py) with the
// arithmetic of the library in the library's order, so a moment computed here is the moment
// Python computes.
//
//   p    = g * z[t-1] + (1 - g) * tanh(T [z; a] + t_b),  g = sigmoid(G [z; a] + g_b)   the expectation
//   e    = tanh(E o + e_b)                                                            the encoded evidence
//   m(k) = read(code([e sqrt(encoded) / norm, z(k)]))                                the store, a coded residual
//   z(k+1) = (1 - alpha) z(k) + alpha tanh(F [z(k); e; p; m(k); 1] + f_b)             the repair, `iterations` times
//   y    = C z + c + decode(m)                                                        the readout
//
// An imagined moment is the transition alone under a declared action with nothing observed
// (no repair iterations, the store read at the expectation); imagination writes nothing.
// Learning is the library's backward scan over a window (`observe`): the adjoint through every
// iteration and the transition, the record read treated as given, one step of every slow
// parameter. The store here holds what the export holds: the pretrained brain's store is
// empty, so its read is exactly zero at every moment (a table of zeros times any code), and
// the page skips the address of a read it does not draw; the code path stays for the viewer's
// record cells and for the parity test, and a store with records in it is read in full.

import { mulberry32, normals } from "./rng.js";

const sigmoid = (x) => 1.0 / (1.0 + Math.exp(-x));

/** y = M x + b for a row-major (rows x cols) matrix. */
function affine(M, rows, cols, x, b, out) {
  for (let i = 0; i < rows; i++) {
    let s = b ? b[i] : 0.0;
    const row = i * cols;
    for (let j = 0; j < cols; j++) s += M[row + j] * x[j];
    out[i] = s;
  }
  return out;
}

export class BeliefPatch {
  /** `spec` is the `belief` block of web/data/brain.json: the sizes, the slow parameters as
   *  flat arrays, the output code and precision, the input norm and the record store. */
  constructor(spec) {
    this.inputs = spec.inputs; this.encoded = spec.encoded; this.actions = spec.actions;
    this.belief = spec.belief; this.outputs = spec.outputs;
    this.iterations = spec.iterations; this.damping = spec.damping; this.recordWidth = spec.record_width;
    this.za = this.belief + this.actions;
    this.fi = this.belief + this.encoded + this.belief + this.recordWidth + 1;
    this.P = {};
    for (const k of ["E", "e_b", "T", "t_b", "G", "g_b", "F", "f_b", "C", "c"]) this.P[k] = Float64Array.from(spec[k]);
    this.outputPrecision = Float64Array.from(spec.output_precision);
    this.outputCode = Float64Array.from(spec.output_code); // (outputs x record_width)
    this.inputNorm = spec.input_norm;
    this.updates = spec.updates || 0;
    this.state = null; // the live belief, one row per stream (Float64Array[])
    const R = spec.records;
    this.records = new Records(R, this.encoded + this.belief);
    this._scratch = { x: new Float64Array(this.za), g: new Float64Array(this.belief), cand: new Float64Array(this.belief), u: new Float64Array(this.fi), reading: new Float64Array(this.encoded + this.belief) };
  }

  /** The slow parameters, copied. */
  parameters() { const out = {}; for (const k in this.P) out[k] = Float64Array.from(this.P[k]); return out; }
  setParameters(P) { for (const k in this.P) this.P[k].set(P[k]); }

  /** The whole patch as data: parameters, state, records; `restore` brings it back. */
  snapshot() {
    return { P: this.parameters(), state: this.state === null ? null : this.state.map((z) => Float64Array.from(z)), records: this.records.snapshot(), inputNorm: this.inputNorm, updates: this.updates };
  }
  restore(snap) {
    this.setParameters(snap.P);
    this.state = snap.state === null ? null : snap.state.map((z) => Float64Array.from(z));
    this.records.restore(snap.records);
    this.inputNorm = snap.inputNorm; this.updates = snap.updates;
  }
  reset() { this.state = null; }

  // ------------------------------------------------------------------ the moment
  _encode(o) {
    const pre = affine(this.P.E, this.encoded, this.inputs, o, this.P.e_b, new Float64Array(this.encoded));
    const e = new Float64Array(this.encoded);
    for (let i = 0; i < this.encoded; i++) e[i] = Math.tanh(pre[i]);
    return { e, pre };
  }
  _expect(z, a) {
    const x = new Float64Array(this.za); x.set(z); x.set(a, this.belief);
    const g = affine(this.P.G, this.belief, this.za, x, this.P.g_b, new Float64Array(this.belief));
    const cand = affine(this.P.T, this.belief, this.za, x, this.P.t_b, new Float64Array(this.belief));
    const p = new Float64Array(this.belief);
    for (let i = 0; i < this.belief; i++) { g[i] = sigmoid(g[i]); cand[i] = Math.tanh(cand[i]); p[i] = g[i] * z[i] + (1.0 - g[i]) * cand[i]; }
    return { x, g, cand, p };
  }
  _reading(e, z, out = new Float64Array(this.encoded + this.belief)) {
    const gain = Math.sqrt(this.encoded) / this.inputNorm;
    for (let i = 0; i < this.encoded; i++) out[i] = e[i] * gain;
    out.set(z, this.encoded);
    return out;
  }
  /** The store's read at a reading: `m` (record_width) and, when asked or when the store holds
   *  records, the code. An empty store reads exactly zero without computing its address. */
  _read(e, z, wantCode) {
    if (!wantCode && this.records.empty) return { m: new Float64Array(this.recordWidth), code: null };
    const code = this.records.code(this._reading(e, z));
    return { m: this.records.read(code), code };
  }
  /** The iterations of one moment; with nothing observed the belief is the expectation. */
  _repair(e, p, observed, wantCode) {
    const B = this.belief;
    const zs = [p], hs = [], us = [], ms = [];
    let z = p;
    if (observed) {
      for (let k = 0; k < this.iterations; k++) {
        const { m } = this._read(e, z, false);
        const u = new Float64Array(this.fi);
        u.set(z, 0); u.set(e, B); u.set(p, B + this.encoded); u.set(m, B + this.encoded + B); u[this.fi - 1] = 1.0;
        const h = affine(this.P.F, B, this.fi, u, this.P.f_b, new Float64Array(B));
        const next = new Float64Array(B);
        for (let i = 0; i < B; i++) { h[i] = Math.tanh(h[i]); next[i] = (1.0 - this.damping) * z[i] + this.damping * h[i]; }
        z = next; zs.push(z); hs.push(h); us.push(u); ms.push(m);
      }
    }
    const { m, code } = this._read(e, z, wantCode);
    const step = new Float64Array(B);
    if (zs.length > 1) { const a = zs[zs.length - 1], b = zs[zs.length - 2]; for (let i = 0; i < B; i++) step[i] = a[i] - b[i]; }
    let residual = 0.0; for (let i = 0; i < B; i++) residual += step[i] * step[i];
    residual = Math.sqrt(residual);
    return { z, zs, hs, us, ms, read: m, code, residual, step };
  }
  /** One row through `t` moments from `boundary`; `observations[k]` is null for an imagined
   *  moment. Returns the record every later pass reads. */
  _forward(observations, actions, boundary, observed, wantCode) {
    const t = actions.length;
    const record = { expect: [], e: [], ePre: [], repair: [], y: [], read: [], code: [] };
    let z = boundary;
    for (let k = 0; k < t; k++) {
      const ex = this._expect(z, actions[k]);
      let e, ePre;
      if (observations !== null && observed[k]) ({ e, pre: ePre } = this._encode(observations[k]));
      else { e = new Float64Array(this.encoded); ePre = new Float64Array(this.encoded); }
      const rep = this._repair(e, ex.p, !!observed[k], wantCode);
      z = rep.z;
      const read = new Float64Array(this.outputs);
      for (let o = 0; o < this.outputs; o++) { let s = 0.0; for (let w = 0; w < this.recordWidth; w++) s += rep.read[w] * this.outputCode[o * this.recordWidth + w]; read[o] = s; }
      const y = affine(this.P.C, this.outputs, this.belief, z, this.P.c, new Float64Array(this.outputs));
      for (let o = 0; o < this.outputs; o++) y[o] += read[o];
      record.expect.push(ex); record.e.push(e); record.ePre.push(ePre); record.repair.push(rep); record.y.push(y); record.read.push(read); record.code.push(rep.code);
    }
    return record;
  }
  _loss(records, targets) {
    // 0.5 * mean over every (row, moment, output) of precision * (slow - target)^2
    let sum = 0.0, count = 0;
    for (let r = 0; r < records.length; r++) for (let k = 0; k < records[r].y.length; k++) {
      const y = records[r].y[k], read = records[r].read[k], tg = targets[r][k];
      for (let o = 0; o < this.outputs; o++) { const d = y[o] - read[o] - tg[o]; sum += this.outputPrecision[o] * d * d; count++; }
    }
    const value = 0.5 * (sum / count);
    return Number.isFinite(value) ? value : null;
  }
  _boundary(n, state) {
    const source = state === undefined || state === null ? this.state : state;
    if (source === null) return Array.from({ length: n }, () => new Float64Array(this.belief));
    if (source.length !== n) throw new Error("state must match (batch, belief); reset when streams change");
    return source.map((z) => Float64Array.from(z));
  }

  // ------------------------------------------------------------------ interface
  /** Advance the belief through observed moments: `observations[row][k]`, `actions[row][k]`.
   *  Nothing is learned or written; the final belief becomes the live state. Returns the
   *  path: beliefs, expectations, residuals, steps, outputs, reads and (when asked) codes. */
  assimilate(observations, actions, { state, wantCode = false } = {}) {
    const n = actions.length, boundary = this._boundary(n, state);
    const records = actions.map((a, r) => this._forward(observations[r], a, boundary[r], a.map(() => true), wantCode));
    this.state = records.map((rec) => Float64Array.from(rec.repair[rec.repair.length - 1].z));
    return this._path(records);
  }
  /** A private continuation under declared actions from the live belief (or `state`): the
   *  transition alone, the store read at the expectation, no observation, no change. */
  imagine(actions, { state, wantCode = false } = {}) {
    const n = actions.length, boundary = this._boundary(n, state);
    const records = actions.map((a, r) => this._forward(null, a, boundary[r], a.map(() => false), wantCode));
    return this._path(records);
  }
  _path(records) {
    return {
      belief: records.map((rec) => rec.repair.map((rep) => rep.z)),
      expectation: records.map((rec) => rec.expect.map((ex) => ex.p)),
      residual: records.map((rec) => rec.repair.map((rep) => rep.residual)),
      step: records.map((rec) => rec.repair.map((rep) => rep.step)),
      output: records.map((rec) => rec.y),
      read: records.map((rec) => rec.read),
      code: records.map((rec) => rec.code),
      e: records.map((rec) => rec.e),
      finalState: records.map((rec) => Float64Array.from(rec.repair[rec.repair.length - 1].z)),
    };
  }
  /** Learn one window of witnessed moments (`write` stays off here, as the experiment's
   *  machinery has it): the forward, the loss before the step, the adjoint, one step of every
   *  slow parameter at `rate`. The final belief under the window becomes the live state. */
  observe(observations, actions, targets, { rate = 1.0, state } = {}) {
    const n = actions.length, boundary = this._boundary(n, state);
    const records = actions.map((a, r) => this._forward(observations[r], a, boundary[r], a.map(() => true), false));
    const path = this._path(records);
    const loss = this._loss(records, targets);
    this.state = path.finalState;
    if (loss === null) return { updated: false, reason: "nonfinite_prediction", path, delta: null, initialLoss: null };
    const delta = this._adjoint(observations, actions, boundary, records, targets);
    if (rate > 0) {
      const proposed = {};
      for (const k in this.P) {
        const p = this.P[k], d = delta[k], out = new Float64Array(p.length);
        for (let i = 0; i < p.length; i++) { out[i] = p[i] - rate * d[i]; if (!Number.isFinite(out[i])) return { updated: false, reason: "nonfinite_update", path, delta, initialLoss: loss }; }
        proposed[k] = out;
      }
      for (const k in proposed) this.P[k] = proposed[k];
      this.updates += 1;
      return { updated: true, reason: "updated", path, delta, initialLoss: loss };
    }
    return { updated: false, reason: "no_step", path, delta, initialLoss: loss };
  }

  // ------------------------------------------------------------------ learning
  _adjoint(o, a, boundary, records, targets) {
    const n = a.length, t = a[0].length, B = this.belief, En = this.encoded, O = this.outputs, W = this.recordWidth, za = this.za, fi = this.fi, alpha = this.damping;
    const { E, T, G, F, C } = this.P;
    const delta = {}; for (const k in this.P) delta[k] = new Float64Array(this.P[k].length);
    const dzNext = Array.from({ length: n }, () => new Float64Array(B));
    for (let k = t - 1; k >= 0; k--) {
      for (let r = 0; r < n; r++) {
        const rec = records[r], rep = rec.repair[k], ex = rec.expect[k];
        const zPrev = k === 0 ? boundary[r] : rec.repair[k - 1].z;
        const dy = new Float64Array(O);
        for (let oo = 0; oo < O; oo++) dy[oo] = this.outputPrecision[oo] * (rec.y[k][oo] - rec.read[k][oo] - targets[r][k][oo]) / (n * t * O);
        for (let oo = 0; oo < O; oo++) { for (let j = 0; j < B; j++) delta.C[oo * B + j] += dy[oo] * rep.z[j]; delta.c[oo] += dy[oo]; }
        let dz = new Float64Array(B);
        for (let j = 0; j < B; j++) { let s = 0.0; for (let oo = 0; oo < O; oo++) s += dy[oo] * C[oo * B + j]; dz[j] = s + dzNext[r][j]; }
        const e = rec.e[k];
        const de = new Float64Array(En), dp = new Float64Array(B);
        for (let j = rep.hs.length - 1; j >= 0; j--) {
          const h = rep.hs[j], u = rep.us[j];
          const dpre = new Float64Array(B);
          for (let i = 0; i < B; i++) { const dh = alpha * dz[i]; dz[i] = (1.0 - alpha) * dz[i]; dpre[i] = dh * (1.0 - h[i] * h[i]); }
          for (let i = 0; i < B; i++) { const row = i * fi; for (let q = 0; q < fi; q++) delta.F[row + q] += dpre[i] * u[q]; delta.f_b[i] += dpre[i]; }
          const du = new Float64Array(fi);
          for (let i = 0; i < B; i++) { const row = i * fi, d = dpre[i]; for (let q = 0; q < fi; q++) du[q] += d * F[row + q]; }
          for (let i = 0; i < B; i++) dz[i] += du[i];
          for (let i = 0; i < En; i++) de[i] += du[B + i];
          for (let i = 0; i < B; i++) dp[i] += du[B + En + i];
        }
        for (let i = 0; i < B; i++) dp[i] += dz[i];
        let anyDe = false; for (let i = 0; i < En; i++) if (de[i] !== 0.0) { anyDe = true; break; }
        if (rec.ePre[k] !== null && anyDe) {
          const obs = o[r][k];
          for (let i = 0; i < En; i++) {
            const dpreE = de[i] * (1.0 - e[i] * e[i]);
            delta.e_b[i] += dpreE;
            const row = i * this.inputs;
            for (let q = 0; q < this.inputs; q++) delta.E[row + q] += dpreE * obs[q];
          }
        }
        const { g, cand, x } = ex;
        const dpreC = new Float64Array(B), dpreG = new Float64Array(B);
        for (let i = 0; i < B; i++) {
          const dzPrev = g[i] * dp[i], dg = dp[i] * (zPrev[i] - cand[i]), dcand = dp[i] * (1.0 - g[i]);
          dpreC[i] = dcand * (1.0 - cand[i] * cand[i]);
          dpreG[i] = dg * g[i] * (1.0 - g[i]);
          dzNext[r][i] = dzPrev;
        }
        for (let i = 0; i < B; i++) { const row = i * za; for (let q = 0; q < za; q++) { delta.T[row + q] += dpreC[i] * x[q]; delta.G[row + q] += dpreG[i] * x[q]; } delta.t_b[i] += dpreC[i]; delta.g_b[i] += dpreG[i]; }
        const dxT = new Float64Array(za), dxG = new Float64Array(za);
        for (let i = 0; i < B; i++) { const row = i * za; for (let q = 0; q < za; q++) { dxT[q] += dpreC[i] * T[row + q]; dxG[q] += dpreG[i] * G[row + q]; } }
        for (let i = 0; i < B; i++) dzNext[r][i] += dxT[i] + dxG[i];
      }
    }
    return delta;
  }
}

/** The record store: the plain code of a reading (the `active` largest of
 *  (reading - mean) . projection + offset + boost, rectified, unit length) and the read of the
 *  one field `y`. The projection and the offsets are drawn from the seed by the library's
 *  generator, as records.py does, so only the learned state travels. */
export class Records {
  constructor(R, inputs) {
    this.inputs = inputs; this.cells = R.cells; this.active = R.active; this.seed = R.seed; this.bias = R.bias;
    this.habituation = R.habituation; this.rate = R.rate;
    const random = mulberry32(this.seed);
    const draws = normals(random, this.inputs * this.cells);
    const root = Math.sqrt(this.inputs);
    this.projection = new Float64Array(this.inputs * this.cells); // [unit][cell]
    for (let i = 0; i < this.projection.length; i++) this.projection[i] = draws[i] / root;
    const offsets = normals(random, this.cells);
    this.offset = new Float64Array(this.cells);
    for (let c = 0; c < this.cells; c++) this.offset[c] = offsets[c] * this.bias;
    this.width = R.width;
    this.mean = Float64Array.from(R.mean);
    this.boost = Float64Array.from(R.boost);
    this.table = Float64Array.from(R.table_y); // (cells x width)
    this.seen = R.seen; this.writes = R.writes;
    this._drive = new Float64Array(this.cells);
    this._empty = null;
  }
  get empty() {
    if (this._empty === null) { this._empty = true; for (let i = 0; i < this.table.length; i++) if (this.table[i] !== 0.0) { this._empty = false; break; } }
    return this._empty;
  }
  snapshot() { return { mean: Float64Array.from(this.mean), boost: Float64Array.from(this.boost), table: Float64Array.from(this.table), seen: this.seen, writes: this.writes }; }
  restore(s) { this.mean.set(s.mean); this.boost.set(s.boost); this.table.set(s.table); this.seen = s.seen; this.writes = s.writes; this._empty = null; }
  /** The plain code of one reading: the active cells and their activities. */
  code(reading) {
    const cells = this.cells, P = this.projection, drive = this._drive;
    for (let c = 0; c < cells; c++) drive[c] = this.offset[c] + this.boost[c];
    for (let i = 0; i < this.inputs; i++) {
      const v = this.habituation > 0 ? reading[i] - this.mean[i] : reading[i];
      if (v === 0.0) continue;
      const row = i * cells;
      for (let c = 0; c < cells; c++) drive[c] += v * P[row + c];
    }
    const k = this.active, top = new Int32Array(k), val = new Float64Array(k);
    let filled = 0, least = 0;
    for (let c = 0; c < cells; c++) {
      const d = drive[c];
      if (filled < k) {
        top[filled] = c; val[filled] = d; filled++;
        if (filled === k) { least = 0; for (let i = 1; i < k; i++) if (val[i] < val[least]) least = i; }
      } else if (d > val[least]) {
        top[least] = c; val[least] = d;
        least = 0; for (let i = 1; i < k; i++) if (val[i] < val[least]) least = i;
      }
    }
    let norm = 0.0;
    for (let i = 0; i < k; i++) { const v = val[i] > 0.0 ? val[i] : 0.0; val[i] = v; norm += v * v; }
    norm = Math.sqrt(norm);
    if (norm > 0) for (let i = 0; i < k; i++) val[i] = val[i] / Math.max(norm, 1e-12);
    return { cells: top, values: val };
  }
  /** The read of field y at a code: the sum of the active cells' records, weighted. */
  read(code) {
    const out = new Float64Array(this.width);
    for (let i = 0; i < code.cells.length; i++) { const row = code.cells[i] * this.width, v = code.values[i]; for (let w = 0; w < this.width; w++) out[w] += v * this.table[row + w]; }
    return out;
  }
  /** The code as a dense vector over the cells, for the viewer. */
  dense(code, out = new Float32Array(this.cells)) { out.fill(0); if (code) for (let i = 0; i < code.cells.length; i++) out[code.cells[i]] = code.values[i]; return out; }
}
