// A settling brain in the browser: the cadence rate model on a sparse connectome, with the
// arithmetic of the library in the library's order, so a parity test (parity.mjs) holds this
// engine to cadence.Brain to a relative difference near machine precision.
//
//   synaptic input_i = sum over synapses e into i of  w_e * s_pre(e)      (w = gain * count * exp(log_gain[pre]) * efficacy)
//   v_i <- v_i + dt * ( -v_i + input_i + stimulus_i + bias_i )
//   s_i = rectified sigmoid(v_i), exactly zero at rest (or, with a leak, negative below rest)
//
// The payload (export.py) stores the synapses by receiving neuron (CSR by post, senders in the
// library's order) and every population by name. The nudged settle is the learner's phase:
// a cross-entropy push on the output neurons toward a one-hot target, as cadence.Nudge.

export function decodeArray(b64, T) {
  const bin = atob(b64); const buf = new ArrayBuffer(bin.length); const u = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
  return new T(buf);
}

export class SettlingBrain {
  constructor(payload) {
    const m = payload.model;
    this.n = payload.n; this.edges = payload.edges;
    this.dt = m.dt; this.slope = m.slope; this.threshold = m.threshold; this.gain = m.gain; this.amplitude = m.stimulus_amplitude;
    this.leak = m.leak || 0.0;  // >0: below rest the activation is negative, scaled by leak / rest (the library's leaky variant)
    if (m.adaptation) throw new Error("the browser engine settles without adaptation; export a model with adaptation=None");
    this.rest = 1.0 / (1.0 + Math.exp(this.slope * this.threshold));
    this.restScale = 1.0 / (1.0 - this.rest);
    this.leakScale = this.leak / this.rest;
    this.rowPtr = decodeArray(payload.arrays.row_ptr, Int32Array);
    this.pre = decodeArray(payload.arrays.pre, Int32Array);
    this.w = decodeArray(payload.arrays.weight, Float64Array);
    this.count = payload.arrays.count ? decodeArray(payload.arrays.count, Uint16Array) : null;  // synapses per class
    this.gainPre = payload.arrays.gain_pre ? decodeArray(payload.arrays.gain_pre, Float64Array) : null;  // weight = gainPre * efficacy
    this.efficacy0 = payload.arrays.efficacy ? decodeArray(payload.arrays.efficacy, Float64Array) : null;  // the efficacies the brain was exported with (signed)
    this.sign = payload.arrays.sign ? decodeArray(payload.arrays.sign, payload.arrays.sign_dtype === "int8" ? Int8Array : Float64Array) : null;
    this.bias = payload.arrays.bias ? decodeArray(payload.arrays.bias, Float64Array) : new Float64Array(this.n);
    this.members = payload.arrays.members ? decodeArray(payload.arrays.members, Int32Array) : null;  // indices into a larger brain, when this is a sub-net
    this.sets = payload.populations || {};
    this.v = new Float64Array(this.n); this.s = new Float64Array(this.n);
    this.drive = new Float64Array(this.n); this.total = new Float64Array(this.n);
    this.steps = 0;
  }

  activation(v) {
    let r = Math.exp((-this.slope) * (v - this.threshold));
    r += 1.0; r = 1.0 / r; r -= this.rest;
    if (this.leak === 0.0) { if (r < 0.0) r = 0.0; return r * this.restScale; }
    return r > 0.0 ? r * this.restScale : r * this.leakScale;
  }

  reset() { this.v.fill(0); this.s.fill(0); this.steps = 0; }

  /** Set the stimulus of a named population to a level in [0, 1] (scaled by the amplitude); levels combine by max. */
  stimulate(name, level) {
    const idx = this.sets[name]; if (!idx) return;
    const d = this.amplitude * level;
    for (const i of idx) if (d > this.drive[i]) this.drive[i] = d;
  }
  /** Set one neuron's stimulus level directly (the library's `stimulus_levels`: level times amplitude). */
  setDrive(index, level) { this.drive[index] = this.amplitude * level; }
  clearStimuli() { this.drive.fill(0); }
  /** The weight of one synapse from an efficacy: gain * count * exp(log_gain[pre]) * efficacy, as the library composes it. */
  setEfficacy(e, efficacy) { this.w[e] = (this.gainPre ? this.gainPre[e] : this.gain * (this.count ? this.count[e] : 1)) * efficacy; }

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

  /** The free phase: up to `steps` steps, stopping once no activation moves by `tolerance` or more
   *  (cadence.Brain.settle_batch with a tolerance). Returns the steps taken. */
  settleFree(steps, tolerance) {
    const prev = new Float64Array(this.n);
    for (let k = 0; k < steps; k++) {
      prev.set(this.s);
      this.step();
      if (tolerance === undefined || tolerance === null) continue;
      let movement = 0.0;
      for (let i = 0; i < this.n; i++) { const d = Math.abs(this.s[i] - prev[i]); if (d > movement) movement = d; }
      if (movement < tolerance) return k + 1;
    }
    return steps;
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
