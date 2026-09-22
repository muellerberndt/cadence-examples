/* RecordPatch: the JavaScript twin of cadence.RecordPatchNet, with online windows and action planning.

   For inputs u[t], context h[t] and outputs y[t]:
     l[t] = sigmoid(g + G u[t])                 per-channel retention
     z[t] = tanh(B u[t] + b)                    the input port
     h[t] = l[t] * h[t-1] + (1 - l[t]) * z[t]
     m[t] = read(code([k * u[t], r * h[t]]))    the record read, a port value; k scales the input block to unit variance per unit
     y[t] = C h[t] + c + m[t]
   The slow parameters learn by the adjoint scan of the precision-weighted half mean squared
   error of the slow readout C h + c (the record read is outside the gradient); a record holds
   what the slow readout got wrong at its reading. `observe` matches the library call step for
   step (sim/parity.js checks it against ref/fixture.json). `step`/`teach`/`flush` are the same
   operation for a continuing actor: one tick at a time, learning over a window of ticks.
   `plan` repairs declared input ports by projected gradient with causal-replay acceptance, the
   library's planning rule applied to this class, with optional feedback of predicted outputs
   into the next moment's inputs. No DOM; used by sim/core.js, sim/gaitlab.js and web/index.html. */
(function (root, factory) { if (typeof module === 'object' && module.exports) module.exports = factory(); else root.CP = factory(); })(typeof self !== 'undefined' ? self : this, function () {
'use strict';

// ---------- Mulberry32: the generator of cadence.records and of the earlier worlds ----------
function Mulberry(seed) { this.s = seed >>> 0; }
Mulberry.prototype.random = function () {
  let a = (this.s = (this.s + 0x6D2B79F5) >>> 0);
  let t = Math.imul(a ^ (a >>> 15), a | 1) >>> 0;
  t = (t ^ (t + Math.imul(t ^ (t >>> 7), t | 61))) >>> 0;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};
Mulberry.prototype.normals = function (n) { // Box-Muller over pairs, as cadence.records.Mulberry32.normals
  const pairs = (n + 1) >> 1, u = new Float64Array(2 * pairs);
  for (let i = 0; i < u.length; i++) u[i] = this.random();
  const out = new Float64Array(n);
  for (let i = 0; i < pairs; i++) {
    const r = Math.sqrt(-2 * Math.log(Math.max(u[i], 1e-12))), a = 2 * Math.PI * u[pairs + i];
    out[i] = r * Math.cos(a); if (pairs + i < n) out[pairs + i] = r * Math.sin(a);
  }
  return out;
};
function gauss(rng) { return rng.normals(1)[0]; }
const sigmoid = x => 0.5 * (1 + Math.tanh(0.5 * x));
const EPS = 2.220446049250313e-16, TINY = 2.2250738585072014e-308;
const PARAMS = ['G', 'g', 'B', 'b', 'C', 'c'];

class RecordPatch {
  constructor(o) {
    const I = this.I = o.inputs | 0, H = this.H = o.hidden | 0, O = this.O = o.outputs | 0;
    if (I < 1 || H < 1 || O < 1) throw new Error('inputs, hidden and outputs are positive');
    this.D = o.cells === undefined ? 4096 : o.cells | 0; this.k = Math.min(o.active === undefined ? 32 : o.active | 0, this.D);
    this.rate = o.recordRate === undefined ? 0.5 : o.recordRate; this.hab = o.habituation === undefined ? 1e-5 : o.habituation;
    this.bias = o.recordBias === undefined ? 0.3 : o.recordBias; this.slowest = o.slowest === undefined ? 128 : o.slowest;
    this.fanin = o.fanin | 0; this.seed = (o.seed | 0) >>> 0; this.W = Math.max(1, o.window | 0 || 1);
    this.G = new Float64Array(H * I); this.g = new Float64Array(H); this.B = new Float64Array(H * I); this.b = new Float64Array(H);
    this.C = new Float64Array(O * H); this.c = new Float64Array(O); this.prec = new Float64Array(O).fill(1);
    this.scale = new Float64Array(H);
    for (let i = 0; i < H; i++) { // retention timescales log-spaced from two moments to `slowest`, or one per channel when given
      const ts = o.timescales ? o.timescales[i] : Math.exp(Math.log(2) + (H > 1 ? (Math.log(this.slowest) - Math.log(2)) * i / (H - 1) : 0)), ret = 1 - 1 / ts;
      this.g[i] = Math.log(ret / (1 - ret)); this.scale[i] = Math.sqrt((1 + ret) / (1 - ret));
    }
    this.mask = null; // an optional 0/1 mask over B and G: a channel reads only the units its cortex names
    const rng = new Mulberry(this.seed);
    if (!o.blank) {
      const nb = rng.normals(H * I); for (let i = 0; i < nb.length; i++) this.B[i] = nb[i] / Math.sqrt(I);
      const nc = rng.normals(O * H); for (let i = 0; i < nc.length; i++) this.C[i] = nc[i] / Math.sqrt(H);
    }
    // the records: a projection of the reading [u, r h] onto D cells, k winners, one table of O columns
    const R = this.R = I + H, D = this.D;
    if (D > 0) {
      const rec = new Mulberry(this.seed);
      if (this.fanin > 0) { // each cell reads fanin units of the reading; the cheap code of the worlds
        const f = this.fanin, w = rec.normals(D * f); this.pidx = new Int32Array(D * f); this.pw = new Float64Array(D * f);
        for (let i = 0; i < D * f; i++) { this.pw[i] = w[i] / Math.sqrt(f); this.pidx[i] = Math.min(R - 1, (rec.random() * R) | 0); }
      } else { // the dense projection of cadence.records: projection[i, c] = draws[i * D + c] / sqrt(R)
        const draws = rec.normals(R * D); this.proj = new Float64Array(D * R);
        for (let i = 0; i < R; i++) for (let c = 0; c < D; c++) this.proj[c * R + i] = draws[i * D + c] / Math.sqrt(R);
      }
      const off = rec.normals(D); this.offset = new Float64Array(D); for (let c = 0; c < D; c++) this.offset[c] = off[c] * this.bias;
      this.table = new Float64Array(D * O); this.mean = new Float64Array(R); this.seen = 0;
      this.inputNorm = 1; this.sqrtI = Math.sqrt(I); // the input block is read at unit variance per unit: scaled by sqrt(inputs) over its witnessed norm
      this.drive = new Float64Array(D); this.xr = new Float64Array(R);
    }
    this.h = new Float64Array(H);
    const W = this.W;
    this.wu = new Float64Array(W * I); this.wgate = new Float64Array(W * H); this.wport = new Float64Array(W * H); this.wprev = new Float64Array(W * H);
    this.whid = new Float64Array(W * H); this.wslow = new Float64Array(W * O); this.wtarget = new Float64Array(W * O); this.wknown = new Float64Array(W * O);
    this.w = 0; this.pending = false; this.updates = 0; this.writes = 0; this.revision = 0; this.replays = 0; this.rejected = 0;
    this.ci = new Int32Array(this.k); this.cv = new Float64Array(this.k); // the code of the last reading
    this.wci = new Int32Array(W * this.k); this.wcv = new Float64Array(W * this.k); this.wcn = new Int32Array(W); // the codes read at each step of the window
    this.lastRead = new Float64Array(O); this.lastSlow = new Float64Array(O);
    this.grad = { G: new Float64Array(H * I), g: new Float64Array(H), B: new Float64Array(H * I), b: new Float64Array(H), C: new Float64Array(O * H), c: new Float64Array(O) };
    this._gh = new Float64Array(W * H);
  }

  // ---------- parameters ----------
  parameters() { const p = {}; for (const k of PARAMS) p[k] = Float64Array.from(this[k]); return p; }
  setParameters(p) { for (const k of PARAMS) { if (!p[k] || p[k].length !== this[k].length) throw new Error('parameter ' + k + ' has the wrong shape'); this[k].set(p[k]); } this.revision++; }
  setPrecision(w) { if (w.length !== this.O) throw new Error('precision has one entry per output'); this.prec.set(w); }
  setInputMask(m) { if (m.length !== this.H * this.I) throw new Error('the mask covers hidden x inputs'); this.mask = m; this._applyMask(); }
  _applyMask() { const m = this.mask; if (!m) return; for (let i = 0; i < m.length; i++) if (!m[i]) { this.B[i] = 0; this.G[i] = 0; } }
  reset() { this.h.fill(0); this.w = 0; this.pending = false; }
  get state() { return Float64Array.from(this.h); }
  setState(h) { this.h.set(h); }

  // ---------- one moment of the slow path ----------
  _gatePort(u, uo, gate, go, port, po, G, g, B, b) {
    const I = this.I, H = this.H;
    for (let i = 0; i < H; i++) {
      let s = g[i], z = b[i]; const r = i * I;
      for (let j = 0; j < I; j++) { const x = u[uo + j]; s += G[r + j] * x; z += B[r + j] * x; }
      gate[go + i] = sigmoid(s); port[po + i] = Math.tanh(z);
    }
  }
  _slow(h, ho, out, oo, C, c) { const H = this.H, O = this.O; for (let j = 0; j < O; j++) { let s = c[j]; const r = j * H; for (let i = 0; i < H; i++) s += C[r + i] * h[ho + i]; out[oo + j] = s; } }

  // ---------- the records ----------
  _reading(u, uo, h, ho) { const xr = this.xr, I = this.I, H = this.H, mean = this.mean, sc = this.scale, k = this.sqrtI / this.inputNorm; for (let i = 0; i < I; i++) xr[i] = u[uo + i] * k - mean[i]; for (let i = 0; i < H; i++) xr[I + i] = h[ho + i] * sc[i] - mean[I + i]; }
  _witnessNorm(u, uo) { let n2 = 0; for (let i = 0; i < this.I; i++) n2 += u[uo + i] * u[uo + i]; this.inputNorm += 0.01 * (Math.max(Math.sqrt(n2), 1e-6) - this.inputNorm); }
  _witness(u, uo, h, ho) { // the running mean moves by a witnessed reading, as cadence.records.witness
    if (this.hab <= 0) return; const I = this.I, H = this.H, mean = this.mean, sc = this.scale, k = this.sqrtI / this.inputNorm; this.seen++; const a = Math.max(this.hab, 1 / this.seen);
    for (let i = 0; i < I; i++) mean[i] += a * (u[uo + i] * k - mean[i]); for (let i = 0; i < H; i++) mean[I + i] += a * (h[ho + i] * sc[i] - mean[I + i]);
  }
  _code(ci, cv) { // the k-winner code of the reading in xr: top-k drives, rectified, unit length; returns the active count
    const D = this.D, k = this.k, R = this.R, xr = this.xr, drive = this.drive, off = this.offset;
    if (this.proj) { const P = this.proj; for (let c = 0; c < D; c++) { let s = off[c]; const b = c * R; for (let i = 0; i < R; i++) s += P[b + i] * xr[i]; drive[c] = s; } }
    else { const f = this.fanin, idx = this.pidx, w = this.pw; for (let c = 0; c < D; c++) { let s = off[c]; const b = c * f; for (let j = 0; j < f; j++) s += w[b + j] * xr[idx[b + j]]; drive[c] = s; } }
    let n = 0, minAt = 0;
    for (let c = 0; c < D; c++) {
      const d = drive[c];
      if (n < k) { ci[n] = c; cv[n] = d; if (n === 0 || d < cv[minAt]) minAt = n; n++; }
      else if (d > cv[minAt]) { ci[minAt] = c; cv[minAt] = d; minAt = 0; for (let a = 1; a < k; a++) if (cv[a] < cv[minAt]) minAt = a; }
    }
    let norm = 0; for (let a = 0; a < n; a++) { cv[a] = Math.max(cv[a], 0); norm += cv[a] * cv[a]; }
    norm = Math.sqrt(norm); if (norm > 0) { const inv = 1 / Math.max(norm, 1e-12); for (let a = 0; a < n; a++) cv[a] *= inv; }
    return n;
  }
  _read(ci, cv, n, out, oo) { const O = this.O, T = this.table; for (let j = 0; j < O; j++) out[oo + j] = 0; for (let a = 0; a < n; a++) { const v = cv[a], b = ci[a] * O; for (let j = 0; j < O; j++) out[oo + j] += v * T[b + j]; } }
  _write(ci, cv, n, target, to, known, ko) { // the delta rule into the active cells: error = target - read, masked by known
    const O = this.O, T = this.table, r = this.rate;
    for (let j = 0; j < O; j++) {
      if (known && !known[ko + j]) continue;
      let read = 0; for (let a = 0; a < n; a++) read += cv[a] * T[ci[a] * O + j];
      const e = r * (target[to + j] - read); for (let a = 0; a < n; a++) T[ci[a] * O + j] += cv[a] * e;
    }
    this.writes++;
  }

  // ---------- a free path from a boundary: parameters as given, records as they stand ----------
  // U: T*I inputs; returns {hidden, gate, port, read, output, slow} as T-major arrays. `fb` = [[outIdx, inIdx], ...] feeds
  // moment t's prediction into moment t+1's inputs (U is written); with `feedbackAdd` the prediction is a change added to
  // moment t's input instead; `noRead` skips the records (slow path only).
  forward(U, T, boundary, opts) {
    const I = this.I, H = this.H, O = this.O, P = (opts && opts.params) || this, fb = opts && opts.feedback, add = !!(opts && opts.feedbackAdd), noRead = opts && opts.noRead, fixed = opts && opts.fixedRead;
    const hidden = new Float64Array(T * H), gate = new Float64Array(T * H), port = new Float64Array(T * H), read = new Float64Array(T * O), output = new Float64Array(T * O), slow = new Float64Array(T * O);
    let prev = boundary, po = 0;
    for (let t = 0; t < T; t++) {
      const ho = t * H, uo = t * I;
      this._gatePort(U, uo, gate, ho, port, ho, P.G, P.g, P.B, P.b);
      for (let i = 0; i < H; i++) { const l = gate[ho + i]; hidden[ho + i] = l * prev[po + i] + (1 - l) * port[ho + i]; }
      this._slow(hidden, ho, slow, t * O, P.C, P.c);
      if (fixed) { for (let j = 0; j < O; j++) read[t * O + j] = fixed[t * O + j]; } else if (this.D > 0 && !noRead) { this._reading(U, uo, hidden, ho); const n = this._code(this.ci, this.cv); this._read(this.ci, this.cv, n, read, t * O); }
      for (let j = 0; j < O; j++) output[t * O + j] = slow[t * O + j] + read[t * O + j];
      if (fb && t + 1 < T) for (const [oi, ii] of fb) U[(t + 1) * I + ii] = (add ? U[t * I + ii] : 0) + output[t * O + oi];
      prev = hidden; po = ho;
    }
    return { hidden, gate, port, read, output, slow };
  }
  imagine(U, T, state) { return this.forward(U, T, state || this.h); } // private: nothing changes
  loss(out, target, T, known) { const O = this.O; let s = 0; for (let t = 0; t < T; t++) for (let j = 0; j < O; j++) { if (known && !known[t * O + j]) continue; const e = out[t * O + j] - target[t * O + j]; s += this.prec[j] * e * e; } return 0.5 * s / (T * O); }

  // ---------- the adjoint scan of the slow readout's loss over a window ----------
  _gradient(U, T, boundary, hidden, gate, port, slow, target, known, out) {
    const I = this.I, H = this.H, O = this.O, C = this.C, gh = this._gh, G = out;
    for (const k of PARAMS) G[k].fill(0);
    const inv = 1 / (T * O), d = new Float64Array(T * O);
    for (let t = 0; t < T; t++) for (let j = 0; j < O; j++) { const i = t * O + j; d[i] = known && !known[i] ? 0 : this.prec[j] * (slow[i] - target[i]) * inv; }
    for (let t = 0; t < T; t++) for (let i = 0; i < H; i++) { let s = 0; for (let j = 0; j < O; j++) s += C[j * H + i] * d[t * O + j]; gh[t * H + i] = s; }
    for (let t = T - 1; t >= 0; t--) {
      const ho = t * H, uo = t * I;
      if (t < T - 1) for (let i = 0; i < H; i++) gh[ho + i] += gate[ho + H + i] * gh[ho + H + i];
      for (let i = 0; i < H; i++) {
        const l = gate[ho + i], z = port[ho + i], prev = t > 0 ? hidden[ho - H + i] : boundary[i];
        const gp = (1 - l) * gh[ho + i] * (1 - z * z), gs = gh[ho + i] * (prev - z) * l * (1 - l), r = i * I;
        for (let j = 0; j < I; j++) { const x = U[uo + j]; G.B[r + j] += gp * x; G.G[r + j] += gs * x; }
        G.b[i] += gp; G.g[i] += gs;
      }
      for (let j = 0; j < O; j++) { const dj = d[t * O + j], r = j * H; for (let i = 0; i < H; i++) G.C[r + i] += dj * hidden[ho + i]; G.c[j] += dj; }
    }
    return G;
  }

  // ---------- observe: the library call, one path from the live state ----------
  observe(U, T, target, opts) {
    const rate = opts && opts.rate !== undefined ? opts.rate : 1, backtrack = !!(opts && opts.backtrack), write = !(opts && opts.write === false), known = opts && opts.known;
    const boundary = Float64Array.from(this.h), f = this.forward(U, T, boundary);
    this.h.set(f.hidden.subarray((T - 1) * this.H, T * this.H));
    const loss = this.loss(f.output, target, T, known), slowLoss = this.loss(f.slow, target, T, known);
    this._gradient(U, T, boundary, f.hidden, f.gate, f.port, f.slow, target, known, this.grad);
    let writes = 0; if (write && this.D > 0) writes = this._writePath(U, T, f.hidden, f.slow, target, known);
    const a = this._admit(U, T, boundary, target, known, slowLoss, rate, backtrack);
    return { updated: a.updated, reason: a.reason, output: f.output, loss, slowLoss, finalLoss: a.finalLoss, acceptedRate: a.acceptedRate, replayLosses: a.replayLosses, replays: a.replays, writes };
  }
  _writePath(U, T, hidden, slow, target, known) { // witness first, then code with the moved mean and write in order, as the library
    const I = this.I, H = this.H, O = this.O; let n = 0;
    for (let t = 0; t < T; t++) this._witnessNorm(U, t * I); // the input norm moves first, by every witnessed moment, as the library
    for (let t = 0; t < T; t++) this._witness(U, t * I, hidden, t * H);
    const resid = new Float64Array(O);
    for (let t = 0; t < T; t++) {
      this._reading(U, t * I, hidden, t * H); const k = this._code(this.ci, this.cv);
      for (let j = 0; j < O; j++) resid[j] = target[t * O + j] - slow[t * O + j];
      this._write(this.ci, this.cv, k, resid, 0, known, t * O); n++;
    }
    return n;
  }
  _admit(U, T, boundary, target, known, initial, rate, backtrack) {
    const g = this.grad;
    if (!backtrack) {
      for (const k of PARAMS) { const p = this[k], d = g[k]; for (let i = 0; i < p.length; i++) p[i] -= rate * d[i]; }
      this._applyMask(); this.updates++; this.revision++; return { updated: true, reason: 'updated', finalLoss: null, acceptedRate: rate, replayLosses: [], replays: 0 };
    }
    let norm2 = 0; for (const k of PARAMS) { const d = g[k]; for (let i = 0; i < d.length; i++) norm2 += d[i] * d[i]; }
    const losses = []; let replays = 0;
    if (initial !== null && isFinite(norm2) && norm2 > 0 && rate > 0) {
      const trial = {}; for (const k of PARAMS) trial[k] = new Float64Array(this[k].length);
      for (let index = 0; index < 16; index++) {
        const step = rate * Math.pow(0.5, index);
        for (const k of PARAMS) { const p = this[k], d = g[k], t = trial[k]; for (let i = 0; i < p.length; i++) t[i] = p[i] - step * d[i]; }
        const f = this.forward(U, T, boundary, { params: trial, noRead: true }), loss = this.loss(f.slow, target, T, known); replays++; losses.push(loss);
        const floor = 64 * EPS * Math.max(Math.abs(initial), Math.abs(loss), TINY);
        if (isFinite(loss) && loss < initial - floor && loss <= initial - 1e-4 * step * norm2) {
          for (const k of PARAMS) this[k].set(trial[k]); this._applyMask(); this.updates++; this.revision++; this.replays += replays;
          return { updated: true, reason: 'updated', finalLoss: loss, acceptedRate: step, replayLosses: losses, replays };
        }
      }
    }
    this.replays += replays; this.rejected++;
    return { updated: false, reason: 'no_decreasing_parameter_step', finalLoss: initial, acceptedRate: 0, replayLosses: losses, replays };
  }

  // ---------- sleep: dream a cue from rest, teach the fixed dream to the slow weights, re-reference the store at dawn ----------
  dream(U, T, out) { // the store's completion of a cue from rest, as the patch would answer awake; nothing changes
    const f = this.forward(U, T, new Float64Array(this.H)); out.set(f.output.subarray(0, T * this.O)); return f; }
  consolidate(U, T, dreamOut, rate, opts) { // the slow weights learn a fixed dream: the adjoint step from rest, no write
    const saved = Float64Array.from(this.h); this.h.fill(0);
    const boundary = new Float64Array(this.H), f = this.forward(U, T, boundary), known = opts && opts.known;
    const g = this._gradient(U, T, boundary, f.hidden, f.gate, f.port, f.slow, dreamOut, known, this.grad);
    let scale = rate; const clip = opts && opts.clip;
    if (clip) { let n2 = 0; for (const k of PARAMS) { const d = g[k]; for (let i = 0; i < d.length; i++) n2 += d[i] * d[i]; } const n = Math.sqrt(n2); if (n > clip) scale = rate * clip / n; }
    let updated = false;
    if (rate > 0) { if (opts && opts.backtrack) updated = this._admit(U, T, boundary, dreamOut, known, this.loss(f.slow, dreamOut, T, known), rate, true).updated;
      else { for (const k of PARAMS) { const p = this[k], d = g[k]; for (let i = 0; i < p.length; i++) p[i] -= scale * d[i]; } this._applyMask(); this.updates++; this.revision++; updated = true; } }
    this.h.set(saved); return updated; }
  reference(U, T, dreamOut, known) { // dawn: write the dream back against the moved weights, so the store holds what they did not take
    if (this.D === 0) return 0; const saved = Float64Array.from(this.h);
    const f = this.forward(U, T, new Float64Array(this.H)), n = this._writePath(U, T, f.hidden, f.slow, dreamOut, known || null);
    this.h.set(saved); return n; }

  // ---------- the continuing actor: one tick at a time, learning over a window ----------
  step(u, out) { // advance the live context by one moment; the prediction (slow readout plus record read) goes to `out`
    const I = this.I, H = this.H, O = this.O, w = this.w, ho = w * H;
    if (this.pending) throw new Error('teach or skip the previous step first');
    this.wu.set(u, w * I); this.wprev.set(this.h, ho);
    this._gatePort(this.wu, w * I, this.wgate, ho, this.wport, ho, this.G, this.g, this.B, this.b);
    for (let i = 0; i < H; i++) { const l = this.wgate[ho + i]; this.h[i] = l * this.h[i] + (1 - l) * this.wport[ho + i]; }
    this.whid.set(this.h, ho);
    this._slow(this.h, 0, this.wslow, w * O, this.C, this.c); this.lastSlow.set(this.wslow.subarray(w * O, w * O + O));
    if (this.D > 0) { this._reading(u, 0, this.h, 0); const n = this._code(this.ci, this.cv); this._read(this.ci, this.cv, n, this.lastRead, 0); this.wci.set(this.ci, w * this.k); this.wcv.set(this.cv, w * this.k); this.wcn[w] = n; } else this.lastRead.fill(0);
    if (out) for (let j = 0; j < O; j++) out[j] = this.lastSlow[j] + this.lastRead[j];
    this.pending = true;
  }
  teach(target, known) { // the witnessed outcome of the last step; `known` masks the entries observed (all when omitted)
    if (!this.pending) throw new Error('nothing to teach'); const O = this.O, w = this.w;
    this.wtarget.set(target, w * O); if (known) this.wknown.set(known, w * O); else this.wknown.fill(1, w * O, w * O + O);
    this.w++; this.pending = false; return this.w >= this.W;
  }
  skip() { if (!this.pending) throw new Error('nothing to skip'); const O = this.O, w = this.w; this.wtarget.fill(0, w * O, w * O + O); this.wknown.fill(0, w * O, w * O + O); this.w++; this.pending = false; return this.w >= this.W; }
  flush(rate, opts) { // learn the window: the adjoint step on the slow parameters and the writes into the records
    const T = this.w; if (!T) return null; const H = this.H, I = this.I, O = this.O, write = !(opts && opts.write === false), clip = opts && opts.clip;
    const boundary = this.wprev.subarray(0, H), U = this.wu.subarray(0, T * I), slow = this.wslow, target = this.wtarget, known = this.wknown;
    const g = this._gradient(U, T, boundary, this.whid, this.wgate, this.wport, slow, target, known, this.grad);
    let writes = 0;
    if (write && this.D > 0) {
      if (opts && opts.recode === false) { // write into the codes that were read, without coding the window again: the world's cheaper path
        const resid = new Float64Array(O), k = this.k; for (let t = 0; t < T; t++) this._witnessNorm(U, t * I); for (let t = 0; t < T; t++) this._witness(U, t * I, this.whid, t * H);
        for (let t = 0; t < T; t++) { let any = 0; for (let j = 0; j < O; j++) { resid[j] = target[t * O + j] - slow[t * O + j]; any += known[t * O + j]; } if (!any) continue; this._write(this.wci.subarray(t * k, t * k + k), this.wcv.subarray(t * k, t * k + k), this.wcn[t], resid, 0, known, t * O); writes++; }
      } else writes = this._writePath(U, T, this.whid, slow, target, known);
    }
    let updated = false, scale = rate;
    if (rate > 0) {
      if (clip) { let n2 = 0; for (const k of PARAMS) { const d = g[k]; for (let i = 0; i < d.length; i++) n2 += d[i] * d[i]; } const n = Math.sqrt(n2); if (n > clip) scale = rate * clip / n; }
      if (opts && opts.backtrack) { const initial = this.loss(slow, target, T, known); updated = this._admit(U, T, Float64Array.from(boundary), target, known, initial, rate, true).updated; }
      else { for (const k of PARAMS) { const p = this[k], d = g[k]; for (let i = 0; i < p.length; i++) p[i] -= scale * d[i]; } this._applyMask(); this.updates++; this.revision++; updated = true; }
    }
    this.w = 0; return { updated, writes, steps: T };
  }

  // ---------- planning: repair declared input ports of a private path under the acquired model ----------
  // U: T*I, the path to repair in place at the control ports; goal: T*O; weights: O (zero excludes an output);
  // controls: Int32Array of input indices; lo/hi bounds, one number each or one per input port; fb feeds predicted
  // outputs forward; returns statistics.
  plan(U, T, goal, weights, controls, lo, hi, opts) {
    const loA = typeof lo === 'number' ? null : lo, hiA = typeof hi === 'number' ? null : hi, loOf = j => loA ? loA[j] : lo, hiOf = j => hiA ? hiA[j] : hi, hold = !!(opts && opts.fixedRead);
    const I = this.I, H = this.H, O = this.O, fb = (opts && opts.feedback) || null, add = !!(opts && opts.feedbackAdd), fwd = { feedback: fb, feedbackAdd: add }, rate = opts && opts.rate !== undefined ? opts.rate : 1, maxSteps = opts && opts.maxSteps !== undefined ? opts.maxSteps : 8, maxBack = opts && opts.maxBacktracks !== undefined ? opts.maxBacktracks : 8, tol = opts && opts.tolerance !== undefined ? opts.tolerance : 1e-9;
    const boundary = Float64Array.from(opts && opts.state ? opts.state : this.h);
    const cost = f => { let s = 0; for (let t = 0; t < T; t++) for (let j = 0; j < O; j++) { const e = f.output[t * O + j] - goal[t * O + j]; s += weights[j] * e * e; } return 0.5 * s / (T * O); };
    let current = Float64Array.from(U), f = this.forward(current, T, boundary, fwd), replays = 1;
    if (hold) fwd.fixedRead = f.read; // the records patch the prediction where the being stands; a proposal is judged by the slow model from there
    const initialOutput = Float64Array.from(f.output.subarray(0, O)); // the first moment's prediction for the unrepaired path, for the caller's readback
    const costs = [cost(f)], steps = []; let reason = 'step_cap', converged = false, residual = null;
    if (!isFinite(costs[0])) return { inputs: current, losses: costs, steps, replays, reason: 'nonfinite_initial', converged: false };
    const du = new Float64Array(T * I), d = new Float64Array(O), gh = new Float64Array(H), carried = new Float64Array(H), proposal = new Float64Array(T * I);
    const isControl = new Uint8Array(I); for (const c of controls) isControl[c] = 1;
    const inv = 1 / (T * O);
    for (let iteration = 0; iteration <= maxSteps; iteration++) {
      // the input gradient: error at the full prediction, sensitivity through the slow path, feedback through the chain
      du.fill(0); carried.fill(0);
      for (let t = T - 1; t >= 0; t--) {
        const ho = t * H, uo = t * I;
        for (let j = 0; j < O; j++) d[j] = weights[j] * (f.output[t * O + j] - goal[t * O + j]) * inv;
        if (fb && t + 1 < T) for (const [oi, ii] of fb) { d[oi] += du[(t + 1) * I + ii]; if (add) du[uo + ii] += du[(t + 1) * I + ii]; }
        for (let i = 0; i < H; i++) { let s = carried[i]; for (let j = 0; j < O; j++) s += this.C[j * H + i] * d[j]; gh[i] = s; }
        for (let i = 0; i < H; i++) {
          const l = f.gate[ho + i], z = f.port[ho + i], prev = t > 0 ? f.hidden[ho - H + i] : boundary[i];
          const gp = (1 - l) * gh[i] * (1 - z * z), gs = gh[i] * (prev - z) * l * (1 - l), r = i * I;
          for (let j = 0; j < I; j++) du[uo + j] += this.B[r + j] * gp + this.G[r + j] * gs;
          carried[i] = l * gh[i];
        }
      }
      let finite = true, res = 0;
      for (let t = 0; t < T; t++) for (let j = 0; j < I; j++) { const i = t * I + j; if (!isControl[j]) { du[i] = 0; continue; } if (!isFinite(du[i])) finite = false; const p = Math.min(hiOf(j), Math.max(loOf(j), current[i] - du[i])); res = Math.max(res, Math.abs(current[i] - p)); }
      if (!finite) { reason = 'nonfinite_input_gradient'; break; }
      residual = res; if (res <= tol) { converged = true; reason = 'projected_stationary'; break; }
      if (iteration === maxSteps) break;
      let accepted = false, step = rate;
      for (let back = 0; back < maxBack; back++) {
        let slope = 0;
        for (let i = 0; i < T * I; i++) { const j = i % I; proposal[i] = isControl[j] ? Math.min(hiOf(j), Math.max(loOf(j), current[i] - step * du[i])) : current[i]; slope += du[i] * (proposal[i] - current[i]); }
        if (slope < 0) {
          const replay = this.forward(proposal, T, boundary, fwd), c = cost(replay); replays++;
          if (isFinite(c) && c < costs[costs.length - 1] && c <= costs[costs.length - 1] + 1e-4 * slope) { current.set(proposal); f = replay; costs.push(c); steps.push(step); accepted = true; break; }
        }
        step *= 0.5;
      }
      if (!accepted) { reason = 'no_decreasing_causal_step'; break; }
    }
    return { inputs: current, prediction: f, initialOutput, losses: costs, steps, replays, reason, converged, residual };
  }

  // ---------- counts for the price and the page ----------
  get cells() { return this.D; }
  get slowCount() { return this.H * this.I * 2 + this.H * 2 + this.O * this.H + this.O; }
  snapshot() { const s = { params: this.parameters(), h: Float64Array.from(this.h) }; if (this.D > 0) { s.table = Float64Array.from(this.table); s.mean = Float64Array.from(this.mean); s.seen = this.seen; s.inputNorm = this.inputNorm; } return s; }
  restore(s) { this.setParameters(s.params); this.h.set(s.h); if (this.D > 0 && s.table) { this.table.set(s.table); this.mean.set(s.mean); this.seen = s.seen; if (s.inputNorm) this.inputNorm = s.inputNorm; } this.w = 0; this.pending = false; }
}

return { Mulberry, gauss, sigmoid, RecordPatch, PARAMS };
});
