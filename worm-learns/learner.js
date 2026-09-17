// The worm's nervous system and its learning rule, written out for the browser.
//
// This mirrors cadence 0.8 for one stream (batch 1): GradedRule activation, the owner-local
// settlement v <- v + dt (inbox + drive + bias + nudge - v), the cross-entropy nudge on the
// output owners, the centred contrast, and ActorCritic's eligibility traces, dopamine and
// linear critic. tests/parity.mjs checks it against a recorded Python run.

export function activation(rule, v) {
  const rest = 1 / (1 + Math.exp(rule.slope * rule.threshold));
  const r = 1 / (1 + Math.exp(-rule.slope * (v - rule.threshold))) - rest;
  return r > 0 ? r / (1 - rest) : r * (rule.leak / rest);
}

export class Brain {
  // net: { n, pre, post, count, sign, rule, outputs, critic, plastic, config, ac }
  constructor(net) {
    this.n = net.n;
    this.pre = Int32Array.from(net.pre);
    this.post = Int32Array.from(net.post);
    this.count = Float64Array.from(net.count);
    this.rule = net.rule;
    this.config = net.config; // LearnerConfig fields used here
    this.ac = net.ac; // ActorCriticConfig fields used here
    this.outputs = Int32Array.from(net.outputs);
    this.critic = Int32Array.from(net.critic);
    this.plastic = Uint8Array.from(net.plastic);
    this.scaleCap = net.scale_cap ?? 0; // bound on plastic seams after each step (0: only cadence's 8)
    this.scale = Float64Array.from(net.scale ?? net.sign);
    this.bias = Float64Array.from(net.bias ?? new Array(net.n).fill(0));
    this.wCritic = Float64Array.from(net.w_critic ?? new Array(net.critic.length).fill(0));
    this.bCritic = net.b_critic ?? 0;
    this.edges = this.pre.length;
    this.weights = new Float64Array(this.edges);
    this.refreshWeights();
    this.trace = new Float64Array(this.edges);
    this.traceBias = new Float64Array(this.n);
    this.traceCritic = new Float64Array(this.critic.length + 1);
    this.velocity = new Float64Array(this.edges);
    this.velocityBias = new Float64Array(this.n);
    this.moment = new Float64Array(this.edges);
    this.momentBias = new Float64Array(this.n);
    this.updates = 0;
    this.free = null; // cached settled state {v, s, steps}
    this.pending = null;
    this.mask = new Float64Array(this.n).fill(1);
    this.isOutput = new Uint8Array(this.n);
    for (const o of this.outputs) this.isOutput[o] = 1;
  }

  refreshWeights() {
    const g = this.rule.gain;
    for (let e = 0; e < this.edges; e++) this.weights[e] = g * this.count[e] * this.scale[e];
  }

  // One settlement from `start` (or rest). `nudge`: {target: Float64Array(n) one-hot on outputs, beta}.
  settle(drive, start, steps, nudge = null, trajectory = null) {
    const n = this.n, rule = this.rule, dt = rule.dt, tol = this.config.tolerance;
    const v = start ? Float64Array.from(start.v) : new Float64Array(n);
    let s = new Float64Array(n);
    for (let i = 0; i < n; i++) s[i] = activation(rule, v[i]) * this.mask[i];
    const inbox = new Float64Array(n);
    let taken = 0;
    for (let t = 0; t < steps; t++) {
      inbox.fill(0);
      for (let e = 0; e < this.edges; e++) inbox[this.post[e]] += s[this.pre[e]] * this.weights[e];
      if (nudge) this.addNudge(inbox, s, nudge);
      const next = new Float64Array(n);
      let movement = 0;
      for (let i = 0; i < n; i++) {
        v[i] = (v[i] + dt * (inbox[i] + drive[i] + this.bias[i] - v[i])) * this.mask[i];
        next[i] = activation(rule, v[i]) * this.mask[i];
        const m = Math.abs(next[i] - s[i]);
        if (m > movement) movement = m;
      }
      s = next;
      taken = t + 1;
      if (trajectory) trajectory.push(s);
      if (tol && movement < tol) break;
    }
    return { v, s, steps: taken };
  }

  addNudge(total, s, nudge) {
    const T = this.config.temperature;
    let max = -Infinity;
    for (const o of this.outputs) max = Math.max(max, s[o] / T);
    let z = 0;
    const p = new Float64Array(this.outputs.length);
    this.outputs.forEach((o, k) => { p[k] = Math.exp(s[o] / T - max); z += p[k]; });
    this.outputs.forEach((o, k) => { total[o] += nudge.beta * (nudge.target[o] - p[k] / z); });
  }

  probabilities(state) {
    const T = this.config.temperature;
    let max = -Infinity;
    for (const o of this.outputs) max = Math.max(max, state.s[o] / T);
    const p = Array.from(this.outputs, (o) => Math.exp(state.s[o] / T - max));
    const z = p.reduce((a, b) => a + b, 0);
    return p.map((x) => x / z);
  }

  value(state) {
    let x = this.bCritic;
    for (let k = 0; k < this.critic.length; k++) x += this.wCritic[k] * state.s[this.critic[k]];
    return x;
  }

  freeState(drive) {
    this.free = this.settle(drive, this.free, this.config.free_steps);
    return this.free;
  }

  // Sample an output owner (or take the most probable one), keep both nudged phases for learning.
  act(drive, u, { greedy = false, learn = true } = {}) {
    const free = this.free ?? this.freeState(drive);
    const p = this.probabilities(free);
    let choice = 0;
    if (greedy) {
      choice = p.indexOf(Math.max(...p));
    } else {
      let c = 0;
      for (const x of p) { c += x; if (c < u) choice += 1; }
      choice = Math.min(choice, p.length - 1);
    }
    if (learn && !greedy) {
      const target = new Float64Array(this.n);
      target[this.outputs[choice]] = 1;
      const beta = this.config.beta;
      const plus = this.settle(drive, free, this.config.nudged_steps, { target, beta });
      const minus = this.settle(drive, free, this.config.nudged_steps, { target, beta: -beta });
      this.pending = { plus, minus, value: this.value(free), free };
    }
    return { choice, p, free };
  }

  // Dopamine from the reward and the next state's value; every plastic seam moves on its trace.
  learn(reward, done, nextDrive) {
    const ac = this.ac, cfg = this.config, pend = this.pending;
    if (!pend) throw new Error("learn needs an act first");
    const span = 2 * cfg.beta, decay = ac.gamma * ac.lam;
    const sp = pend.plus.s, sm = pend.minus.s;
    for (let e = 0; e < this.edges; e++) {
      const a = this.pre[e], b = this.post[e];
      this.trace[e] = decay * this.trace[e] + (sp[a] * sp[b] - sm[a] * sm[b]) / span;
    }
    for (let i = 0; i < this.n; i++) this.traceBias[i] = decay * this.traceBias[i] + (sp[i] - sm[i]) / span;
    const free = pend.free;
    for (let k = 0; k < this.critic.length; k++) this.traceCritic[k] = decay * this.traceCritic[k] + free.s[this.critic[k]];
    const last = this.critic.length;
    this.traceCritic[last] = decay * this.traceCritic[last] + 1;

    let next = this.settle(nextDrive, free, cfg.free_steps);
    if (done) next = this.settle(nextDrive, { v: new Float64Array(this.n) }, cfg.free_steps);
    const nextValue = done ? 0 : this.value(next);
    let delta = reward + ac.gamma * nextValue - pend.value;
    if (ac.dopamine_cap > 0) delta = Math.max(-ac.dopamine_cap, Math.min(ac.dopamine_cap, delta));

    this.updates += 1;
    const stepScale = new Float64Array(this.edges);
    const stepBias = new Float64Array(this.n);
    for (let e = 0; e < this.edges; e++) stepScale[e] = delta * this.trace[e];
    for (let i = 0; i < this.n; i++) stepBias[i] = delta * this.traceBias[i];
    const raw = Float64Array.from(stepScale), rawBias = Float64Array.from(stepBias);
    if (ac.momentum > 0) {
      const m = ac.momentum, corr = 1 - m ** this.updates;
      for (let e = 0; e < this.edges; e++) { this.velocity[e] = m * this.velocity[e] + (1 - m) * stepScale[e]; stepScale[e] = this.velocity[e] / corr; }
      for (let i = 0; i < this.n; i++) { this.velocityBias[i] = m * this.velocityBias[i] + (1 - m) * stepBias[i]; stepBias[i] = this.velocityBias[i] / corr; }
    }
    if (ac.normalize > 0) {
      const rho = ac.normalize, corr = 1 - rho ** this.updates;
      for (let e = 0; e < this.edges; e++) { this.moment[e] = rho * this.moment[e] + (1 - rho) * raw[e] ** 2; stepScale[e] /= Math.sqrt(this.moment[e] / corr) + 1e-3; }
      for (let i = 0; i < this.n; i++) { this.momentBias[i] = rho * this.momentBias[i] + (1 - rho) * rawBias[i] ** 2; stepBias[i] /= Math.sqrt(this.momentBias[i] / corr) + 1e-3; }
    }
    let moved = 0;
    for (let e = 0; e < this.edges; e++) {
      if (!this.plastic[e]) continue;
      const step = ac.eta * stepScale[e];
      let x = Math.max(-8, Math.min(8, this.scale[e] + step));
      if (this.scaleCap) x = Math.max(-this.scaleCap, Math.min(this.scaleCap, x));
      this.scale[e] = x;
      moved += Math.abs(step);
    }
    for (let i = 0; i < this.n; i++) this.bias[i] += ac.eta_bias * stepBias[i];
    this.refreshWeights();

    let energy = 0;
    for (const x of this.traceCritic) energy += x * x;
    const norm = ac.critic_normalize ? 1 + energy : 1;
    for (let k = 0; k < this.critic.length; k++) this.wCritic[k] += (ac.eta_critic * delta * this.traceCritic[k]) / norm;
    this.bCritic += (ac.eta_critic * delta * this.traceCritic[last]) / norm;

    if (done) { this.trace.fill(0); this.traceBias.fill(0); this.traceCritic.fill(0); }
    this.free = next;
    this.pending = null;
    return { delta, value: pend.value, moved, freeSteps: next.steps };
  }
}
