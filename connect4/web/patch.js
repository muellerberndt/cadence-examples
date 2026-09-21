// The value patch in the browser: cadence's RecordPatchNet for readings of one moment that
// start from rest, with the arithmetic of the library (src/cadence/record_patch.py and
// records.py) in the library's order, so a page computes what Python computes.
//
//   s = g + G u            l = sigmoid(s)          z = tanh(B u + b)
//   h = (1 - l) z          (the context before the reading is rest)
//   reading = [u sqrt(n) / norm, h scale] - mean
//   code    = the `active` largest of reading . projection + offset, rectified, unit length
//   value   = C h + c + code . table
//
// The projection and the offsets are drawn from the seed by the library's generator, so the
// page rebuilds the same cells and only the learned state travels.

export function mulberry32(seed) {
  let state = seed >>> 0;
  return function () {
    state = (state + 0x6d2b79f5) >>> 0;
    const a = state;
    let t = Math.imul(a ^ (a >>> 15), a | 1) >>> 0;
    t = (t ^ ((t + (Math.imul(t ^ (t >>> 7), t | 61) >>> 0)) >>> 0)) >>> 0;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296.0;
  };
}

// Box-Muller as cadence.Mulberry32.normals lays it out: every uniform is drawn first, the
// radii come from the first half and the angles from the second, and the result is all the
// cosine terms followed by all the sine terms.
export function normals(random, n) {
  const pairs = (n + 1) >> 1;
  const u = new Float64Array(2 * pairs);
  for (let i = 0; i < 2 * pairs; i++) u[i] = random();
  const out = new Float64Array(n);
  for (let i = 0; i < pairs; i++) {
    const radius = Math.sqrt(-2.0 * Math.log(Math.max(u[i], 1e-12)));
    const angle = 2.0 * Math.PI * u[pairs + i];
    out[i] = radius * Math.cos(angle);
    if (pairs + i < n) out[pairs + i] = radius * Math.sin(angle);
  }
  return out;
}

const sigmoid = (x) => (x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x)));

export class ValuePatch {
  // `state` is what connect4/web/export.py writes: the sizes and the record settings,
  // and the learned arrays as Float64Array (G, g, B, b, C, c, mean, table) with the scalars
  // seen and input_norm.
  constructor(state) {
    Object.assign(this, state.sizes); // inputs, hidden, cells, active, seed, bias, rate, habituation, slowest
    for (const k of ["G", "g", "B", "b", "C", "c", "mean", "table"]) this[k] = Float64Array.from(state[k]);
    this.seen = state.seen;
    this.inputNorm = state.input_norm;
    const n = this.inputs + this.hidden;
    const random = mulberry32(this.seed);
    const draws = normals(random, n * this.cells);
    const root = Math.sqrt(n);
    // projection[unit][cell], kept cell-major per unit so a sparse reading adds whole rows
    this.projection = new Float64Array(n * this.cells);
    for (let i = 0; i < n * this.cells; i++) this.projection[i] = draws[i] / root;
    const offsets = normals(random, this.cells);
    this.offset = new Float64Array(this.cells);
    for (let c = 0; c < this.cells; c++) this.offset[c] = offsets[c] * this.bias;
    // every channel's retention timescale, log-spaced from two moments to `slowest`
    this.scale = new Float64Array(this.hidden);
    for (let k = 0; k < this.hidden; k++) {
      const t = this.hidden === 1 ? 0 : k / (this.hidden - 1);
      const timescale = Math.exp(Math.log(2.0) + t * (Math.log(this.slowest) - Math.log(2.0)));
      const retention = 1.0 - 1.0 / timescale;
      this.scale[k] = Math.sqrt((1.0 + retention) / (1.0 - retention));
    }
    this.meanDrive = new Float64Array(this.cells);
    this._settle();
    this.reads = 0;
    this.written = 0;                                  // record cells that hold something
    for (let c = 0; c < this.cells; c++) if (this.table[c] !== 0) this.written++;
  }

  // mean . projection, recomputed when the running mean moves (after a witnessed reading)
  _settle() {
    const n = this.inputs + this.hidden, cells = this.cells, P = this.projection, m = this.meanDrive;
    m.fill(0);
    for (let i = 0; i < n; i++) {
      const v = this.mean[i];
      if (v === 0) continue;
      const row = i * cells;
      for (let c = 0; c < cells; c++) m[c] += v * P[row + c];
    }
  }

  // The context and the slow readout of one reading `u` (a 0/1 array of `inputs` units).
  _context(u, h, gate, port) {
    const H = this.hidden, n = this.inputs;
    let slow = this.c[0];
    for (let k = 0; k < H; k++) {
      let s = this.g[k], z = this.b[k];
      const row = k * n;
      for (let j = 0; j < n; j++) if (u[j]) { s += this.G[row + j]; z += this.B[row + j]; }
      const l = sigmoid(s), p = Math.tanh(z);
      h[k] = (1 - l) * p;
      if (gate) { gate[k] = l; port[k] = p; }
      slow += this.C[k] * h[k];
    }
    return slow;
  }

  // The active cells of a reading and their activities; returns the record read.
  _code(u, h, cellsOut, activityOut) {
    const n = this.inputs, H = this.hidden, cells = this.cells, P = this.projection;
    const drive = this._drive || (this._drive = new Float64Array(cells));
    for (let c = 0; c < cells; c++) drive[c] = this.offset[c] - this.meanDrive[c];
    const gain = Math.sqrt(n) / this.inputNorm;
    for (let j = 0; j < n; j++) {
      if (!u[j]) continue;
      const row = j * cells;
      for (let c = 0; c < cells; c++) drive[c] += gain * P[row + c];
    }
    for (let k = 0; k < H; k++) {
      const v = h[k] * this.scale[k];
      const row = (n + k) * cells;
      for (let c = 0; c < cells; c++) drive[c] += v * P[row + c];
    }
    // the `active` largest drives: a threshold by partial selection over a copy
    const k = this.active;
    const top = this._top || (this._top = new Int32Array(k));
    const val = this._val || (this._val = new Float64Array(k));
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
    let norm = 0;
    for (let i = 0; i < k; i++) { const v = val[i] > 0 ? val[i] : 0; val[i] = v; norm += v * v; }
    norm = Math.sqrt(norm);
    let read = 0;
    for (let i = 0; i < k; i++) {
      const a = norm > 0 ? val[i] / Math.max(norm, 1e-12) : val[i];
      if (cellsOut) { cellsOut[i] = top[i]; activityOut[i] = a; }
      read += a * this.table[top[i]];
    }
    return read;
  }

  // The value of one reading, read privately. `trace`, when given, receives what the page
  // shows: the context, the active cells with their activities, the slow readout and the read.
  value(u, trace) {
    const h = this._h || (this._h = new Float64Array(this.hidden));
    const slow = this._context(u, h, null, null);
    this.reads++;
    // An empty store reads exactly zero at every reading, as the library's code times a table
    // of zeros does; the address (hidden x cells operations) is then not worth computing.
    if (!trace && this.written === 0) return slow + 0;
    if (!trace) return slow + this._code(u, h, null, null);
    trace.cells = new Int32Array(this.active);
    trace.activity = new Float64Array(this.active);
    const read = this._code(u, h, trace.cells, trace.activity);
    trace.context = Float64Array.from(h);
    trace.slow = slow;
    trace.read = read;
    return slow + read;
  }
}
