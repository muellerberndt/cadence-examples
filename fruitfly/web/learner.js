// The library's actor-critic on the fly's sub-net, in the browser: the same rule as
// cadence.plasticity.ActorCritic with one stream (tools/learn_odour.py runs it in Python).
//
//   act:    the live state is the free phase; two nudged phases (toward and away from the sampled
//           action's one-hot at strength beta) give each plastic synapse its contrast
//              (a+ (b+ - b-) + (a+ - a-) b-) / (2 beta),  a = pre, b = post
//           and the trace decays by gamma lambda and adds the contrast; the critic's trace likewise.
//   learn:  dopamine = reward + gamma V(next) - V(this), clipped at the cap; every plastic synapse
//           moves by eta dopamine trace and is clipped at the efficacy cap; the critic by
//           eta_critic dopamine on its normalised trace; a finished search forgets its traces.
//
// The plastic synapses are those from Kenyon cells onto mushroom body output neurons, the site of
// the animal's olfactory memory; the actions are read from two output neurons of declared valence.
export const DEFAULTS = { beta: 0.1, temperature: 0.05, nudgedSteps: 20, tolerance: 1e-3, gamma: 0.95, lam: 0.9, eta: 10.0, etaCritic: 0.5, cap: 3.0, dopamineCap: 1.0 };

export class FlyLearner {
  /** brain: a FlyBrain; config: { outputs: [population names, one neuron each], actions: [action id per output], plastic: "kc>mbon"|"mb", critic: population name, tonic: {name: input}, ...DEFAULTS } */
  constructor(brain, config) {
    this.brain = brain; this.cfg = { ...DEFAULTS, ...config };
    const sets = brain.sets;
    this.outputs = this.cfg.outputs.map((name) => { const idx = sets[name]; if (!idx || idx.length !== 1) throw new Error(`${name} must name one neuron`); return idx[0]; });
    this.actions = this.cfg.actions || this.outputs.map((_, j) => j);
    this.criticIndex = Int32Array.from(sets[this.cfg.critic || "kc"] || []);
    const inSet = (name) => { const m = new Uint8Array(brain.n); for (const i of (sets[name] || [])) m[i] = 1; return m; };
    const kc = inSet("kc"), mb = inSet("mbon");
    const edges = [];
    for (let i = 0; i < brain.n; i++) for (let e = brain.rowPtr[i]; e < brain.rowPtr[i + 1]; e++) {
      if (this.cfg.plastic === "mb" ? mb[i] : (kc[brain.pre[e]] && mb[i])) edges.push(e);
    }
    this.edges = Int32Array.from(edges);
    this.post = new Int32Array(this.edges.length);
    for (let i = 0, k = 0; i < brain.n; i++) for (let e = brain.rowPtr[i]; e < brain.rowPtr[i + 1]; e++) if (k < this.edges.length && this.edges[k] === e) this.post[k++] = i;
    this.sign = Float64Array.from(this.edges, (e) => (brain.sign ? brain.sign[e] : Math.sign(brain.w[e])));
    this.efficacy = Float64Array.from(this.sign);
    this.trace = new Float64Array(this.edges.length);
    this.wCritic = new Float64Array(this.criticIndex.length); this.bCritic = 0.0;
    this.traceCritic = new Float64Array(this.criticIndex.length + 1);
    this.pending = null; this.updates = 0; this.rng = Math.random;
    this.tonic = this.cfg.tonic || {};
    for (const [name, level] of Object.entries(this.tonic)) { const idx = sets[name]; if (idx) for (const i of idx) brain.bias[i] += level; }
  }

  /** The weight of every plastic synapse from its efficacy: gain * count * efficacy. */
  applyWeights() { const b = this.brain; for (let k = 0; k < this.edges.length; k++) { const e = this.edges[k]; b.w[e] = b.gain * (b.count ? b.count[e] : 1) * this.efficacy[k]; } }

  probabilities(s = this.brain.s) {
    const T = this.cfg.temperature, m = this.outputs.length, p = new Float64Array(m);
    let zmax = -Infinity; for (let j = 0; j < m; j++) zmax = Math.max(zmax, s[this.outputs[j]] / T);
    let sum = 0; for (let j = 0; j < m; j++) { p[j] = Math.exp(s[this.outputs[j]] / T - zmax); sum += p[j]; }
    for (let j = 0; j < m; j++) p[j] /= sum;
    return p;
  }

  value(s = this.brain.s) { let v = this.bCritic; for (let k = 0; k < this.criticIndex.length; k++) v += this.wCritic[k] * s[this.criticIndex[k]]; return v; }

  /** Sample an action on the live (free) state, keep its eligibility. Returns { choice, action, p, value, greedy }. */
  act(greedy = false) {
    const c = this.cfg, b = this.brain, p = this.probabilities();
    let choice = 0;
    if (greedy) { for (let j = 1; j < p.length; j++) if (p[j] > p[choice]) choice = j; }
    else { const u = this.rng(); let acc = 0; choice = p.length - 1; for (let j = 0; j < p.length; j++) { acc += p[j]; if (u < acc) { choice = j; break; } } }
    const value = this.value();
    this.pending = null;
    if (!greedy) {
      const target = new Float64Array(p.length); target[choice] = 1.0;
      const plus = b.settleNudged(this.outputs, target, c.beta, c.temperature, c.nudgedSteps, c.tolerance).s;
      const minus = b.settleNudged(this.outputs, target, -c.beta, c.temperature, c.nudgedSteps, c.tolerance).s;
      const span = 2.0 * c.beta, decay = c.gamma * c.lam, pre = b.pre;
      for (let k = 0; k < this.edges.length; k++) {
        const e = this.edges[k], j = pre[e], i = this.post[k];
        const contrast = (plus[j] * (plus[i] - minus[i]) + (plus[j] - minus[j]) * minus[i]) / span;
        this.trace[k] = this.trace[k] * decay + contrast;
      }
      for (let k = 0; k < this.criticIndex.length; k++) this.traceCritic[k] = this.traceCritic[k] * decay + b.s[this.criticIndex[k]];
      this.traceCritic[this.criticIndex.length] = this.traceCritic[this.criticIndex.length] * decay + 1.0;
      this.pending = { value, choice };
    }
    return { choice, action: this.actions[choice], p: Array.from(p), value, greedy };
  }

  /** Dopamine from the reward and the live state's value (the next state), then the three-factor step. */
  learn(reward, done) {
    if (!this.pending) return null;
    const c = this.cfg, b = this.brain;
    const nextValue = done ? 0.0 : this.value();
    const tdError = reward + c.gamma * nextValue - this.pending.value;
    let delta = tdError;
    if (c.dopamineCap > 0) delta = Math.max(-c.dopamineCap, Math.min(c.dopamineCap, delta));
    let moved = 0, sumAbs = 0;
    for (let k = 0; k < this.edges.length; k++) {
      const step = c.eta * delta * this.trace[k];
      if (step === 0) continue;
      let eff = this.efficacy[k] + step;
      if (eff > c.cap) eff = c.cap; else if (eff < -c.cap) eff = -c.cap;
      const d = eff - this.efficacy[k];
      if (d !== 0) { moved++; sumAbs += Math.abs(d); this.efficacy[k] = eff; const e = this.edges[k]; b.w[e] = b.gain * (b.count ? b.count[e] : 1) * eff; }
    }
    let energy = 0; for (let k = 0; k < this.traceCritic.length; k++) energy += this.traceCritic[k] * this.traceCritic[k];
    const norm = 1.0 / (1.0 + energy);
    for (let k = 0; k < this.criticIndex.length; k++) this.wCritic[k] += c.etaCritic * delta * this.traceCritic[k] * norm;
    this.bCritic += c.etaCritic * delta * this.traceCritic[this.criticIndex.length] * norm;
    if (done) { this.trace.fill(0); this.traceCritic.fill(0); }
    this.updates++;
    const out = { delta, tdError, reward, value: this.pending.value, nextValue, moved, meanAbsStep: moved ? sumAbs / moved : 0, updates: this.updates, ...this.stats() };
    this.pending = null;
    return out;
  }

  stats() {
    let changed = 0, sumAbs = 0, maxAbs = 0;
    for (let k = 0; k < this.edges.length; k++) { const d = Math.abs(this.efficacy[k] - this.sign[k]); if (d > 1e-9) { changed++; sumAbs += d; if (d > maxAbs) maxAbs = d; } }
    return { plastic: this.edges.length, changed, meanAbsChange: changed ? sumAbs / changed : 0, maxAbsChange: maxAbs };
  }

  /** Efficacies back to the measured signs, traces and critic to nothing. */
  reset() { this.efficacy.set(this.sign); this.trace.fill(0); this.traceCritic.fill(0); this.wCritic.fill(0); this.bCritic = 0; this.pending = null; this.updates = 0; this.applyWeights(); }

  /** Efficacies from a checkpoint (payload edge ids -> efficacy), for the plastic edges; other edges are set on the brain directly. */
  load(edges, values) {
    const where = new Map(); for (let k = 0; k < this.edges.length; k++) where.set(this.edges[k], k);
    const b = this.brain; let inside = 0;
    for (let q = 0; q < edges.length; q++) { const e = edges[q], k = where.get(e); if (k !== undefined) { this.efficacy[k] = values[q]; inside++; } b.w[e] = b.gain * (b.count ? b.count[e] : 1) * values[q]; }
    return inside;
  }
}
