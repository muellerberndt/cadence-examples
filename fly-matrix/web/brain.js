// The fly's brain in the browser: the cadence rate model on the flight sub-net's sparse
// connectivity, with the arithmetic of the library in the library's order, so a parity test
// (tests/parity.mjs) holds this engine to cadence.Brain to a relative difference near machine
// precision.
//
//   synaptic input_i = sum over synapses e into i of  w_e * s_pre(e)      (w = gain * count * sign)
//   v_i <- v_i + dt * ( -v_i + input_i + stimulus_i + bias_i )
//   s_i = rectified sigmoid(v_i), exactly zero at rest
//
// Rows are receiving neurons; synapses are stored by (post, pre) as the library sorts them.

export function decodeArray(b64, T) {
  const bin = atob(b64); const buf = new ArrayBuffer(bin.length); const u = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
  return new T(buf);
}

export class FlyBrain {
  constructor(payload) {
    const m = payload.model;
    this.n = payload.n; this.edges = payload.edges;
    this.dt = m.dt; this.slope = m.slope; this.threshold = m.threshold; this.gain = m.gain; this.amplitude = m.stimulus_amplitude;
    this.rest = 1.0 / (1.0 + Math.exp(this.slope * this.threshold));
    this.restScale = 1.0 / (1.0 - this.rest);
    this.rowPtr = decodeArray(payload.arrays.row_ptr, Int32Array);
    this.pre = decodeArray(payload.arrays.pre, Int32Array);
    this.w = decodeArray(payload.arrays.weight, Float64Array);
    this.count = payload.arrays.count ? decodeArray(payload.arrays.count, Uint16Array) : null;  // synapses per class, so learned efficacies can replace the sign
    this.sign = payload.arrays.sign ? decodeArray(payload.arrays.sign, Int8Array) : null;
    this.bias = payload.arrays.bias ? decodeArray(payload.arrays.bias, Float64Array) : new Float64Array(this.n);
    this.members = payload.arrays.members ? decodeArray(payload.arrays.members, Int32Array) : null;
    this.sets = payload.populations;
    this.v = new Float64Array(this.n); this.s = new Float64Array(this.n);
    this.drive = new Float64Array(this.n); this.total = new Float64Array(this.n);
    this.steps = 0;
  }

  activation(v) {
    let r = Math.exp((-this.slope) * (v - this.threshold));
    r += 1.0; r = 1.0 / r; r -= this.rest;
    if (r < 0.0) r = 0.0;
    return r * this.restScale;
  }

  reset() { this.v.fill(0); this.s.fill(0); this.drive.fill(0); this.steps = 0; }

  /** Set the stimulus of a named population to a level in [0, 1] (scaled by the amplitude); levels add up by max. */
  stimulate(name, level) {
    const idx = this.sets[name]; if (!idx) return;
    const d = this.amplitude * level;
    for (const i of idx) if (d > this.drive[i]) this.drive[i] = d;
  }
  clearStimuli() { this.drive.fill(0); }

  /** One step of the neuron model under the current drive. */
  step() {
    const n = this.n, rowPtr = this.rowPtr, pre = this.pre, w = this.w, s = this.s, v = this.v, total = this.total;
    for (let i = 0; i < n; i++) {
      let sum = 0.0;
      for (let e = rowPtr[i], end = rowPtr[i + 1]; e < end; e++) sum += s[pre[e]] * w[e];
      total[i] = sum;
    }
    const dt = this.dt, drive = this.drive, bias = this.bias;
    for (let i = 0; i < n; i++) {
      let t = total[i];
      t += drive[i] + bias[i];   // standing = drive + bias, added as one number as the library does
      t -= v[i];
      t *= dt;
      v[i] += t;
      s[i] = this.activation(v[i]);
    }
    this.steps++;
  }

  /** Settle copies of the state for up to `steps` under the current drive with a cross-entropy nudge
   *  on `outputs` toward the one-hot `target` (strength beta, softmax temperature T), stopping when no
   *  activation moves by `tolerance` or more: the library's nudged phase, in its order of operations.
   *  Returns { s, taken } without touching the live state. */
  settleNudged(outputs, target, beta, T, steps, tolerance) {
    const n = this.n, rowPtr = this.rowPtr, pre = this.pre, w = this.w, dt = this.dt, drive = this.drive, bias = this.bias;
    const v = Float64Array.from(this.v), s = Float64Array.from(this.s), total = new Float64Array(n), prev = new Float64Array(n);
    const m = outputs.length, p = new Float64Array(m);
    let taken = 0;
    for (let k = 0; k < steps; k++) {
      let zmax = -Infinity;
      for (let j = 0; j < m; j++) { const z = s[outputs[j]] / T; if (z > zmax) zmax = z; }
      let sum = 0.0;
      for (let j = 0; j < m; j++) { p[j] = Math.exp(s[outputs[j]] / T - zmax); sum += p[j]; }
      for (let j = 0; j < m; j++) p[j] /= sum;
      for (let i = 0; i < n; i++) {
        let acc = 0.0;
        for (let e = rowPtr[i], end = rowPtr[i + 1]; e < end; e++) acc += s[pre[e]] * w[e];
        total[i] = acc;
      }
      for (let j = 0; j < m; j++) total[outputs[j]] += beta * (target[j] - p[j]);
      prev.set(s);
      let movement = 0.0;
      for (let i = 0; i < n; i++) {
        let t = total[i];
        t += drive[i] + bias[i];
        t -= v[i];
        t *= dt;
        v[i] += t;
        s[i] = this.activation(v[i]);
        const d = Math.abs(s[i] - prev[i]); if (d > movement) movement = d;
      }
      taken = k + 1;
      if (tolerance !== undefined && tolerance !== null && movement < tolerance) break;
    }
    return { s, taken };
  }

  mean(name) { const idx = this.sets[name]; if (!idx || !idx.length) return 0; let t = 0; for (const i of idx) t += this.s[i]; return t / idx.length; }
  peak(name) { const idx = this.sets[name]; let m = 0; if (!idx) return 0; for (const i of idx) if (this.s[i] > m) m = this.s[i]; return m; }
  activeCount(level = 0.5) { let c = 0; for (let i = 0; i < this.n; i++) if (this.s[i] >= level) c++; return c; }
}
