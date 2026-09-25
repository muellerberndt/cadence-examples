// The library's actor-critic in the browser, one stream: the same rule as
// cadence.plasticity.ActorCritic (parity.mjs replays a recorded run of the library through it).
//
//   act:    the live state is the free phase; two nudged phases (toward and away from the sampled
//           action's one-hot at strength beta) give each plastic synapse its contrast
//              (a+ (b+ - b-) + (a+ - a-) b-) / (2 beta),  a = pre, b = post
//           and the trace decays by gamma lambda and adds the contrast; the critic's trace likewise.
//   learn:  dopamine = reward + gamma V(next) - V(this), clipped at the cap; every plastic synapse
//           moves by eta dopamine trace and is clipped at the efficacy cap; the critic by
//           eta_critic dopamine on its normalised trace; a finished search forgets its traces.
//
// Between act and learn the caller puts the next observation on the brain and settles it (the
// live state is then V(next)); a finished stream resets the brain first, as the library does.
// After learn the caller settles once more, since the weights changed under the state (the
// library's act does this refresh itself); a page that keeps stepping the brain does it by stepping.
export const DEFAULTS = { beta: 0.1, temperature: 0.2, nudgedSteps: 50, tolerance: 1e-4, gamma: 0.99, lam: 0.9, eta: 0.5, etaBias: 0.0, etaCritic: 0.05, cap: 8.0, dopamineCap: 1.0 };

function indices(brain, spec, what) {
  if (Array.isArray(spec) && spec.every((x) => typeof x === "number")) return Int32Array.from(spec);
  const names = Array.isArray(spec) ? spec : [spec];
  const out = [];
  for (const name of names) { const idx = brain.sets[name]; if (!idx) throw new Error(`${what}: no population named ${name}`); out.push(...idx); }
  return Int32Array.from(out);
}

export class ActorCriticLearner {
  /** brain: a SettlingBrain; config: {
   *    outputs: population names (one neuron each) or neuron indices, in action order,
   *    actions: the action each output stands for (default its index),
   *    plastic: "all" | an array of synapse indices (payload order) | { pre: [names], post: [names] } (synapses from pre sets onto post sets; a side left out means any),
   *    critic: a population name or neuron indices,
   *    plasticNeurons: "none" (default) | "all" | names | indices: whose bias moves by etaBias,
   *    tonic: { name: level } constant inputs added to the bias (declared),
   *    ...DEFAULTS }
   *  Each synapse has its own efficacy (the library's reciprocal=False); tied groups, per-synapse rates,
   *  momentum, normalisation, decay and dopamine centring are outside this engine. */
  constructor(brain, config) {
    this.brain = brain; this.cfg = { ...DEFAULTS, ...config };
    const c = this.cfg;
    this.outputs = Array.from(indices(brain, c.outputs, "outputs"));
    this.actions = c.actions || this.outputs.map((_, j) => j);
    this.criticIndex = indices(brain, c.critic, "critic");
    const E = brain.edges;
    let edges;
    if (c.plastic === "all" || c.plastic === undefined) edges = Array.from({ length: E }, (_, e) => e);
    else if (Array.isArray(c.plastic)) edges = Array.from(c.plastic);
    else {
      const inSet = (names) => { const m = new Uint8Array(brain.n); for (const name of names) for (const i of (brain.sets[name] || [])) m[i] = 1; return m; };
      const anyPre = !c.plastic.pre, anyPost = !c.plastic.post;
      const pre = inSet(c.plastic.pre || []), post = inSet(c.plastic.post || []);
      edges = [];
      for (let i = 0; i < brain.n; i++) for (let e = brain.rowPtr[i]; e < brain.rowPtr[i + 1]; e++) if ((anyPost || post[i]) && (anyPre || pre[brain.pre[e]])) edges.push(e);
    }
    this.edges = Int32Array.from(edges);
    const postOf = new Int32Array(E);
    for (let i = 0; i < brain.n; i++) for (let e = brain.rowPtr[i]; e < brain.rowPtr[i + 1]; e++) postOf[e] = i;
    this.post = Int32Array.from(this.edges, (e) => postOf[e]);
    // the efficacies start where the exported brain's are (the connectome's signs until a lesson moved them)
    this.sign = Float64Array.from(this.edges, (e) => (brain.efficacy0 ? brain.efficacy0[e] : brain.sign ? brain.sign[e] : Math.sign(brain.w[e])));
    this.efficacy = Float64Array.from(this.sign);
    this.trace = new Float64Array(this.edges.length);
    const pn = c.plasticNeurons === undefined || c.plasticNeurons === "none" ? [] : c.plasticNeurons === "all" ? Array.from({ length: brain.n }, (_, i) => i) : Array.from(indices(brain, c.plasticNeurons, "plasticNeurons"));
    this.neurons = Int32Array.from(pn);
    this.traceBias = new Float64Array(this.neurons.length);
    this.wCritic = new Float64Array(this.criticIndex.length); this.bCritic = 0.0;
    this.traceCritic = new Float64Array(this.criticIndex.length + 1);
    this.pending = null; this.updates = 0; this.dropped = 0; this.rng = Math.random;  // dropped: outcomes that found no decision to credit
    this.tonic = c.tonic || {};
    for (const [name, level] of Object.entries(this.tonic)) { const idx = brain.sets[name]; if (idx) for (const i of idx) brain.bias[i] += level; }
  }

  /** The weight of every plastic synapse from its efficacy. */
  applyWeights() { for (let k = 0; k < this.edges.length; k++) this.brain.setEfficacy(this.edges[k], this.efficacy[k]); }

  probabilities(s = this.brain.s) {
    const T = this.cfg.temperature, m = this.outputs.length, p = new Float64Array(m);
    let zmax = -Infinity; for (let j = 0; j < m; j++) zmax = Math.max(zmax, s[this.outputs[j]] / T);
    let sum = 0; for (let j = 0; j < m; j++) { p[j] = Math.exp(s[this.outputs[j]] / T - zmax); sum += p[j]; }
    for (let j = 0; j < m; j++) p[j] /= sum;
    return p;
  }

  value(s = this.brain.s) { let v = this.bCritic; for (let k = 0; k < this.criticIndex.length; k++) v += this.wCritic[k] * s[this.criticIndex[k]]; return v; }

  /** The fraction of outputs within 0.02 of 0 or 1, where a nudge has no slope (the library's report["saturation"]). */
  saturation() { let c = 0; for (const i of this.outputs) { const s = this.brain.s[i]; if (s < 0.02 || s > 0.98) c++; } return c / this.outputs.length; }

  /** Sample an action on the live (free) state and keep its eligibility. `u` replays a recorded uniform draw.
   *  Returns { choice, action, p, value, greedy }. */
  act(greedy = false, u = undefined) {
    const c = this.cfg, b = this.brain, p = this.probabilities();
    let choice = 0;
    if (greedy) { for (let j = 1; j < p.length; j++) if (p[j] > p[choice]) choice = j; }
    else {  // the library's draw: the count of cumulative sums below the uniform, capped
      const draw = u === undefined ? this.rng() : u;
      let acc = 0; choice = 0;
      for (let j = 0; j < p.length; j++) { acc += p[j]; if (acc < draw) choice++; }
      if (choice > p.length - 1) choice = p.length - 1;
    }
    const value = this.value(), saturation = this.saturation();
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
      for (let k = 0; k < this.neurons.length; k++) { const i = this.neurons[k]; this.traceBias[k] = this.traceBias[k] * decay + (plus[i] - minus[i]) / span; }
      for (let k = 0; k < this.criticIndex.length; k++) this.traceCritic[k] = this.traceCritic[k] * decay + b.s[this.criticIndex[k]];
      this.traceCritic[this.criticIndex.length] = this.traceCritic[this.criticIndex.length] * decay + 1.0;
      this.pending = { value, choice, saturation, plus, minus };
    }
    return { choice, action: this.actions[choice], p: Array.from(p), value, greedy };
  }

  /** Dopamine from the reward and the live state's value (the next state), then the three-factor step.
   *  An outcome with no decision pending teaches nothing and is counted in `dropped`: a page that hands
   *  outcomes over must see that count (the fly's blows were lost this way for a day). */
  learn(reward, done) {
    if (!this.pending) { this.dropped++; return null; }
    const c = this.cfg;
    const nextValue = done ? 0.0 : this.value();
    const tdError = reward + c.gamma * nextValue - this.pending.value;
    let delta = tdError;
    if (c.dopamineCap > 0) delta = Math.max(-c.dopamineCap, Math.min(c.dopamineCap, delta));
    let moved = 0, sumAbs = 0, traceAbs = 0;
    for (let k = 0; k < this.edges.length; k++) {
      traceAbs += Math.abs(this.trace[k]);
      const step = c.eta * delta * this.trace[k];
      if (step === 0) continue;
      let eff = this.efficacy[k] + step;
      if (eff > c.cap) eff = c.cap; else if (eff < -c.cap) eff = -c.cap;
      const d = eff - this.efficacy[k];
      if (d !== 0) { moved++; sumAbs += Math.abs(d); this.efficacy[k] = eff; this.brain.setEfficacy(this.edges[k], eff); }
    }
    for (let k = 0; k < this.neurons.length; k++) this.brain.bias[this.neurons[k]] += c.etaBias * delta * this.traceBias[k];
    let energy = 0; for (let k = 0; k < this.traceCritic.length; k++) energy += this.traceCritic[k] * this.traceCritic[k];
    const norm = 1.0 / (1.0 + energy);
    for (let k = 0; k < this.criticIndex.length; k++) this.wCritic[k] += c.etaCritic * delta * this.traceCritic[k] * norm;
    this.bCritic += c.etaCritic * delta * this.traceCritic[this.criticIndex.length] * norm;
    if (done) { this.trace.fill(0); this.traceBias.fill(0); this.traceCritic.fill(0); }
    this.updates++;
    const out = { delta, tdError, reward, value: this.pending.value, nextValue, moved, meanAbsStep: moved ? sumAbs / moved : 0, trace: this.edges.length ? traceAbs / this.edges.length : 0, saturation: this.pending.saturation, updates: this.updates, ...this.stats() };
    this.pending = null;
    return out;
  }

  stats() {
    let changed = 0, sumAbs = 0, maxAbs = 0;
    for (let k = 0; k < this.edges.length; k++) { const d = Math.abs(this.efficacy[k] - this.sign[k]); if (d > 1e-9) { changed++; sumAbs += d; if (d > maxAbs) maxAbs = d; } }
    return { plastic: this.edges.length, changed, meanAbsChange: changed ? sumAbs / changed : 0, maxAbsChange: maxAbs, dropped: this.dropped };
  }

  /** Efficacies back to the measured signs, traces and critic to nothing. */
  reset() { this.efficacy.set(this.sign); this.trace.fill(0); this.traceBias.fill(0); this.traceCritic.fill(0); this.wCritic.fill(0); this.bCritic = 0; this.pending = null; this.updates = 0; this.dropped = 0; this.applyWeights(); }

  /** Efficacies from a checkpoint (payload synapse ids -> efficacy); ids outside the plastic set are set on the brain directly. */
  load(edges, values) {
    const where = new Map(); for (let k = 0; k < this.edges.length; k++) where.set(this.edges[k], k);
    let inside = 0;
    for (let q = 0; q < edges.length; q++) { const e = edges[q], k = where.get(e); if (k !== undefined) { this.efficacy[k] = values[q]; inside++; } this.brain.setEfficacy(e, values[q]); }
    return inside;
  }
}
