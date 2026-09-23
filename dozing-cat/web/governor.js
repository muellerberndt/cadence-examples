// The governor patch in the browser: a settling patch of the same kind as every other cadence
// brain (cadence.Brain with the graded rate neuron of neuron.py), a readback port of seven
// units, a cortex of four and a motor of three. Every synapse and every bias is a gene; the
// settled motor state, read as the most active unit, is the mode. The settle is the library's
// `equilibrate`: from rest (or from the last state when the genome says warm), chunks of ten
// steps of
//
//     v <- v + dt * (synaptic input + drive + bias - v),   s = activation(v)
//     activation(v) = max(0, sigmoid(slope (v - threshold)) - rest) / (1 - rest),  rest = sigmoid(-slope threshold)
//
// until the largest equation error |synaptic input + drive + bias - v| is at most the tolerance
// or the budget of 200 steps is spent; the steps taken are the patch's cost. Nothing in it
// learns within a life.

export const READBACK = ["fast", "slow", "residual", "mode_habit", "mode_imagine", "mode_learn", "one"];
export const CORTEX = 4;
export const MOTOR = ["habit", "imagine", "learn"];
export const NEURON = { dt: 0.5, slope: 2.0, threshold: 0.5, gain: 1.0, stimulus_amplitude: 1.0, leak: 0.0 };
export const SETTLE = { budget: 200, chunk: 10, tolerance: 1e-3 };

export class GovernorPatch {
  constructor(genome, neuron = NEURON, settle = SETTLE) {
    this.g = { ...genome };
    this.neuron = neuron; this.settleRule = settle;
    const nr = READBACK.length, nc = CORTEX, nm = MOTOR.length;
    this.readback = Array.from({ length: nr }, (_, i) => i);
    this.cortex = Array.from({ length: nc }, (_, j) => nr + j);
    this.motor = Array.from({ length: nm }, (_, k) => nr + nc + k);
    this.n = nr + nc + nm;
    const g = this.g;
    // the synapse table in the order cat.py declares it (the viewer draws it in this order)
    const table = [];
    for (let i = 0; i < nr; i++) {
      for (let j = 0; j < nc; j++) if (g[`rc_${i}_${j}`] !== 0.0) table.push([this.readback[i], this.cortex[j], g[`rc_${i}_${j}`]]);
      for (let k = 0; k < nm; k++) if (g[`rm_${i}_${k}`] !== 0.0) table.push([this.readback[i], this.motor[k], g[`rm_${i}_${k}`]]);
    }
    for (let j = 0; j < nc; j++) {
      for (let k = 0; k < nm; k++) if (g[`cm_${j}_${k}`] !== 0.0) table.push([this.cortex[j], this.motor[k], g[`cm_${j}_${k}`]]);
      for (let j2 = 0; j2 < nc; j2++) if (j !== j2 && g.lateral_cortex !== 0.0) table.push([this.cortex[j], this.cortex[j2], g.lateral_cortex]);
    }
    for (let k = 0; k < nm; k++) for (let k2 = 0; k2 < nm; k2++) if (k !== k2 && g.lateral_motor !== 0.0) table.push([this.motor[k], this.motor[k2], g.lateral_motor]);
    this.table = table;
    this.synapses = table.length;
    // the dense synapse matrix W[pre][post]: the drive of synapse e per unit presynaptic
    // activation is gain * count * sign, with count 1 and the gene as the sign
    this.W = new Float64Array(this.n * this.n);
    for (const [p, q, w] of table) this.W[p * this.n + q] += neuron.gain * 1.0 * w;
    this.bias = new Float64Array(this.n);
    for (let j = 0; j < nc; j++) this.bias[this.cortex[j]] = g[`cb_${j}`];
    for (let k = 0; k < nm; k++) this.bias[this.motor[k]] = g[`mb_${k}`];
    this.rest = 1.0 / (1.0 + Math.exp(neuron.slope * neuron.threshold));
    this.state = null; // {v, a} of the last settle, for a warm start
    this.activation = new Float64Array(this.n);
    this.v = new Float64Array(this.n);
  }
  _activation(v, out) {
    const { slope, threshold, leak } = this.neuron, rest = this.rest;
    for (let i = 0; i < v.length; i++) {
      let r = 1.0 / (Math.exp(-slope * (v[i] - threshold)) + 1.0);
      r -= rest;
      if (leak === 0.0) out[i] = r > 0.0 ? r * (1.0 / (1.0 - rest)) : 0.0;
      else out[i] = r > 0.0 ? r * (1.0 / (1.0 - rest)) : r * (leak / rest);
    }
    return out;
  }
  _synaptic(s, out) {
    const n = this.n, W = this.W;
    out.fill(0);
    for (let p = 0; p < n; p++) { const sp = s[p]; if (sp === 0.0) continue; const row = p * n; for (let q = 0; q < n; q++) out[q] += sp * W[row + q]; }
    return out;
  }
  /** Brain.residual: the largest |synaptic input + drive + bias - v| over the neurons. */
  _residual(drive, v, s) {
    const syn = this._synaptic(s, new Float64Array(this.n));
    let worst = 0.0;
    for (let i = 0; i < this.n; i++) { const err = Math.abs(syn[i] + drive[i] + this.bias[i] - v[i]); if (!(err <= worst)) worst = Number.isFinite(err) ? err : Infinity; }
    return worst;
  }
  /** The mode from the settled state under the readback as the drive on the port; the steps
   *  taken are the patch's cost. */
  settle(readback) {
    const n = this.n, { dt } = this.neuron, { budget, chunk, tolerance } = this.settleRule;
    const drive = new Float64Array(n);
    for (let i = 0; i < this.readback.length; i++) drive[this.readback[i]] = readback[i]; // equilibrate takes the dense drive as given
    const v = new Float64Array(n);
    if (this.g.warm && this.state !== null) v.set(this.state.v);
    const s = this._activation(v, new Float64Array(n)), syn = new Float64Array(n);
    let error = this._residual(drive, v, s), used = 0;
    while (used < budget && !(error <= tolerance)) {
      const steps = Math.min(chunk, budget - used);
      for (let t = 0; t < steps; t++) {
        this._synaptic(s, syn);
        for (let i = 0; i < n; i++) { let total = syn[i] + (drive[i] + this.bias[i]); total -= v[i]; total *= dt; v[i] += total; }
        this._activation(v, s);
      }
      used += steps;
      error = this._residual(drive, v, s);
    }
    this.state = { v: Float64Array.from(v) };
    this.activation.set(s); this.v.set(v);
    let best = 0, top = -Infinity;
    for (let k = 0; k < this.motor.length; k++) { const m = s[this.motor[k]]; if (m > top) { top = m; best = k; } }
    const mode = top > 1e-6 ? MOTOR[best] : "habit";
    return { mode, steps: used, activation: Float64Array.from(s), residual: error };
  }
}
