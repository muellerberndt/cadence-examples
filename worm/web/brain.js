// The worm's brain in the browser: the cadence 0.50.0 reference engine
// (_repair.py), ported operation for operation for graphs without residual
// edges, and the same worm loop as worm/brain.py on top of it. Verified
// against the Python library by tests/parity.mjs.
//
// Every patch has live state x, retained incoming weights w and bias b:
//   p_i = tanh(b_i + sum(w_ij * signal_j)),  e_i = x_i - p_i
// The energy is sum(e^2)/2 + state_prior*sum(x^2)/2, plus a fixed proximal
// parameter prior during a learning solve. One projected-gradient repair with
// a scalar secant step and Armijo backtracking updates the eligible
// coordinates; a solve qualifies only when the full projected stationarity
// residual is at most the configured tolerance. Sums that the library forms
// with math.fsum are formed here with the same exact-summation algorithm.

const tanh = Math.tanh;

// ---- exact summation, Python's math.fsum ---------------------------------------
export function fsum(values) {
  const partials = [];
  for (let x of values) {
    let i = 0;
    for (let j = 0; j < partials.length; j++) {
      let y = partials[j];
      if (Math.abs(x) < Math.abs(y)) { const t = x; x = y; y = t; }
      const hi = x + y;
      const lo = y - (hi - x);
      if (lo !== 0) partials[i++] = lo;
      x = hi;
    }
    partials.length = i + 1;
    partials[i] = x;
  }
  let n = partials.length, hi = 0, lo = 0;
  if (n > 0) {
    hi = partials[--n];
    while (n > 0) {
      const x = hi, y = partials[--n];
      hi = x + y;
      const yr = hi - x;
      lo = y - yr;
      if (lo !== 0) break;
    }
    if (n > 0 && ((lo < 0 && partials[n - 1] < 0) || (lo > 0 && partials[n - 1] > 0))) {
      const y = lo * 2, x = hi + y;
      if (y === x - hi) hi = x;
    }
  }
  return hi;
}

// math.ulp: the spacing of floats at |x|.
const ulpView = new DataView(new ArrayBuffer(8));
export function ulp(x) {
  x = Math.abs(x);
  if (!isFinite(x)) return NaN;
  ulpView.setFloat64(0, x);
  ulpView.setBigUint64(0, ulpView.getBigUint64(0) + 1n);
  return ulpView.getFloat64(0) - x;
}

const clip = (v, bound) => Math.min(bound, Math.max(-bound, v));
const projected = (value, gradient, bound) =>
  gradient >= 0 ? Math.min(gradient, value + bound) : Math.max(gradient, value - bound);

// ---- the engine: one jointly repaired population graph -------------------------
export class Engine {
  constructor(spec) {
    this.nInputs = spec.n_inputs;
    this.nPatches = spec.biases.length;
    this.config = spec.config;
    const n = spec.edges.length;
    this.kind = new Int8Array(n);      // 0 input, 1 state
    this.source = new Int32Array(n);
    this.target = new Int32Array(n);
    this.incoming = Array.from({ length: this.nPatches }, () => []);
    spec.edges.forEach(([kind, source, target], k) => {
      if (kind === "residual") throw new Error("this port carries no residual edges");
      this.kind[k] = kind === "input" ? 0 : 1;
      this.source[k] = source;
      this.target[k] = target;
      this.incoming[target].push(k);
    });
  }

  // _evaluate: energy, exact errors and analytic derivatives for one example.
  evaluate(inputs, state, weights, biases, anchors, parameterGradients) {
    const P = this.nPatches, E = this.kind.length, c = this.config;
    const predictions = new Float64Array(P), errors = new Float64Array(P);
    const signals = new Float64Array(E), terms = [];
    for (let target = 0; target < P; target++) {
      terms.length = 0;
      terms.push(biases[target]);
      for (const k of this.incoming[target]) {
        signals[k] = this.kind[k] === 0 ? inputs[this.source[k]] : state[this.source[k]];
        terms.push(weights[k] * signals[k]);
      }
      const activation = fsum(terms);
      if (!isFinite(activation)) throw new Error("nonfinite");
      predictions[target] = tanh(activation);
      errors[target] = state[target] - predictions[target];
    }
    let energy = 0.5 * fsum(errors.map((e) => e * e));
    energy += 0.5 * c.state_prior * fsum(state.map((x) => x * x));
    const gradState = Float64Array.from(state, (x) => c.state_prior * x);
    const gradWeights = parameterGradients ? new Float64Array(E) : null;
    const gradBiases = parameterGradients ? new Float64Array(P) : null;
    for (let target = P - 1; target >= 0; target--) {
      const adj = errors[target];
      gradState[target] += adj;
      const adjPrediction = -adj * (1 - predictions[target] ** 2);
      if (parameterGradients) gradBiases[target] += adjPrediction;
      for (const k of this.incoming[target]) {
        if (parameterGradients) gradWeights[k] += adjPrediction * signals[k];
        if (this.kind[k] === 1) gradState[this.source[k]] += adjPrediction * weights[k];
      }
    }
    let finite = isFinite(energy);
    if (anchors) {
      const dw = weights.map((w, k) => w - anchors.weights[k]);
      const db = biases.map((b, k) => b - anchors.biases[k]);
      energy += 0.5 * c.parameter_prior * fsum([...dw.map((d) => d * d), ...db.map((d) => d * d)]);
      for (let k = 0; k < E; k++) gradWeights[k] += c.parameter_prior * dw[k];
      for (let k = 0; k < P; k++) gradBiases[k] += c.parameter_prior * db[k];
      finite = isFinite(energy);
    }
    for (const a of [predictions, errors, gradState, gradWeights ?? [], gradBiases ?? []])
      for (const v of a) if (!isFinite(v)) finite = false;
    if (!finite) throw new Error("nonfinite");
    return { energy, predictions, errors, gradState, gradWeights, gradBiases };
  }

  // _evaluate_batch: mean row energy, private states, one shared anchor.
  evaluateBatch(inputs, state, weights, biases, anchors, parameterGradients, B) {
    const P = this.nPatches, I = this.nInputs, E = this.kind.length;
    const rows = [];
    for (let row = 0; row < B; row++)
      rows.push(this.evaluate(inputs.subarray(row * I, (row + 1) * I),
                              state.subarray(row * P, (row + 1) * P),
                              weights, biases, null, parameterGradients));
    let energy = fsum(rows.map((r) => r.energy)) / B;
    const unscaled = new Float64Array(B * P), gradState = new Float64Array(B * P);
    rows.forEach((r, row) => unscaled.set(r.gradState, row * P));
    for (let k = 0; k < B * P; k++) gradState[k] = unscaled[k] / B;
    let gradWeights = null, gradBiases = null;
    if (parameterGradients) {
      gradWeights = new Float64Array(E);
      gradBiases = new Float64Array(P);
      const column = new Float64Array(B);
      for (let k = 0; k < E; k++) {
        for (let row = 0; row < B; row++) column[row] = rows[row].gradWeights[k];
        gradWeights[k] = fsum(column);       // _mean: fsum over rows, then divide
        gradWeights[k] /= B;
      }
      for (let k = 0; k < P; k++) {
        for (let row = 0; row < B; row++) column[row] = rows[row].gradBiases[k];
        gradBiases[k] = fsum(column);
        gradBiases[k] /= B;
      }
      if (anchors) {
        const dw = weights.map((w, k) => w - anchors.weights[k]);
        const db = biases.map((b, k) => b - anchors.biases[k]);
        energy += 0.5 * this.config.parameter_prior *
          fsum([...dw.map((d) => d * d), ...db.map((d) => d * d)]);
        for (let k = 0; k < E; k++) gradWeights[k] += this.config.parameter_prior * dw[k];
        for (let k = 0; k < P; k++) gradBiases[k] += this.config.parameter_prior * db[k];
      }
    }
    let finite = isFinite(energy);
    for (const a of [gradState, gradWeights ?? [], gradBiases ?? []])
      for (const v of a) if (!isFinite(v)) finite = false;
    if (!finite) throw new Error("nonfinite");
    const errors = new Float64Array(B * P), predictions = new Float64Array(B * P);
    rows.forEach((r, row) => { errors.set(r.errors, row * P); predictions.set(r.predictions, row * P); });
    return { energy, predictions, errors, gradState, gradStateUnscaled: unscaled, gradWeights, gradBiases };
  }

  stationarity(state, weights, biases, evaluated, clamps, learn, B) {
    const c = this.config;
    const gradient = evaluated.gradStateUnscaled ?? evaluated.gradState;
    let residual = 0;
    for (let i = 0; i < state.length; i++) {
      if (clamps.has(i)) continue;
      const g = evaluated.gradStateUnscaled ? gradient[i] : B * gradient[i];
      residual = Math.max(residual, Math.abs(projected(state[i], g, c.state_bound)));
    }
    if (learn) {
      for (let k = 0; k < weights.length; k++)
        residual = Math.max(residual, Math.abs(projected(weights[k], evaluated.gradWeights[k], c.parameter_bound)));
      for (let k = 0; k < biases.length; k++)
        residual = Math.max(residual, Math.abs(projected(biases[k], evaluated.gradBiases[k], c.parameter_bound)));
    }
    return residual;
  }

  // settle: one repair with a scalar secant step and Armijo backtracking.
  // clamps is a Map of flat state index -> fixed value. learn repairs the
  // parameters too, anchored at their pre-call values. B rows share them.
  settle(inputs, state, weights, biases, { clamps = new Map(), learn = false, budget = null, B = 1 } = {}) {
    const c = this.config;
    budget = budget ?? c.settle_budget;
    const anchors = learn ? { weights: Float64Array.from(weights), biases: Float64Array.from(biases) } : null;
    state = Float64Array.from(state, (x, i) => clamps.has(i) ? clamps.get(i) : x);
    weights = Float64Array.from(weights);
    biases = Float64Array.from(biases);
    let evaluations = 0;
    const compute = (x, w, b) => {
      evaluations++;
      return B === 1
        ? this.evaluate(inputs, x, w, b, anchors, learn)
        : this.evaluateBatch(inputs, x, w, b, anchors, learn, B);
    };
    let current = compute(state, weights, biases);
    let sweeps = 0, reason = "budget", nextStep = c.step, accepted = false;
    for (let iteration = 0; iteration < budget; iteration++) {
      if (this.stationarity(state, weights, biases, current, clamps, learn, B) <= c.tolerance) {
        reason = "qualified";
        break;
      }
      let trialStep = nextStep;
      accepted = false;
      for (let attempt = 0; attempt < c.backtracks; attempt++) {
        const gradient = current.gradStateUnscaled ?? current.gradState;
        const nextState = Float64Array.from(state, (x, i) =>
          clamps.has(i) ? clamps.get(i) : clip(x - trialStep * gradient[i], c.state_bound));
        let nextWeights = weights, nextBiases = biases;
        if (learn) {
          nextWeights = Float64Array.from(weights, (x, k) => clip(x - trialStep * current.gradWeights[k], c.parameter_bound));
          nextBiases = Float64Array.from(biases, (x, k) => clip(x - trialStep * current.gradBiases[k], c.parameter_bound));
        }
        const groups = [[state, nextState, "gradState", B]];
        if (learn) groups.push([weights, nextWeights, "gradWeights", 1], [biases, nextBiases, "gradBiases", 1]);
        let proposed = null, ok = false;
        try {
          const slopeTerms = [];
          let moved = false;
          for (const [old, next, key] of groups)
            for (let i = 0; i < old.length; i++) {
              slopeTerms.push(current[key][i] * (next[i] - old[i]));
              if (next[i] !== old[i]) moved = true;
            }
          const slope = fsum(slopeTerms);
          proposed = compute(nextState, nextWeights, nextBiases);
          ok = moved && isFinite(slope) && slope < 0 &&
            (proposed.energy <= current.energy + 1e-4 * slope ||
             (Math.abs(proposed.energy - current.energy) <= 8 * ulp(current.energy) &&
              this.stationarity(nextState, nextWeights, nextBiases, proposed, clamps, learn, B) <= c.tolerance));
          if (ok) {
            // the scalar secant estimate from accepted displacement and gradient change
            const changes = [];
            for (const [old, next, key, scale] of groups)
              for (let i = 0; i < old.length; i++)
                if (next[i] !== old[i])
                  changes.push([next[i] - old[i], proposed[key][i] - current[key][i], scale]);
            const distance = fsum(changes.map(([s, , scale]) => s * s / scale));
            const curvature = fsum(changes.map(([s, y]) => s * y));
            nextStep = c.step;
            if (curvature > 0) {
              const estimate = distance / curvature;
              const ceiling = c.step * 2 ** Math.min(c.backtracks - 1, 1023);
              if (isFinite(estimate) && estimate > 0) nextStep = Math.min(estimate, ceiling);
            }
            state = nextState; weights = nextWeights; biases = nextBiases;
            current = proposed;
            sweeps++;
          }
        } catch { ok = false; }
        if (ok) { accepted = true; break; }
        trialStep *= 0.5;
      }
      if (!accepted) { reason = "line_search"; break; }
    }
    const final = compute(state, weights, biases);
    const residual = this.stationarity(state, weights, biases, final, clamps, learn, B);
    const qualified = isFinite(residual) && residual <= c.tolerance;
    return {
      state, weights, biases, qualified,
      reason: qualified ? "qualified" : reason,
      energy: final.energy, stationarity: residual, sweeps, evaluations,
      errors: final.errors, predictions: final.predictions,
    };
  }
}

// ---- the worm on top of the engine, mirroring worm/brain.py --------------------
export class Brain {
  constructor(spec) {
    this.spec = spec;
    this.H = spec.H;
    this.names = spec.names;
    this.senses = spec.senses;
    this.readouts = spec.readouts;
    this.pos = spec.pos;
    this.engine = new Engine(spec);
    this.weights = Float64Array.from(spec.weights);
    this.biases = Float64Array.from(spec.biases);
    this.state = Float64Array.from(spec.state);
    this.outputPatch = Object.fromEntries(spec.outputs.map((o) => [o.name, o.patch]));
    this.p = null;                     // set by the life: params.json
    // the page's live view of the neuron-to-neuron weights, in spec.A entry order
    this.aEdge = Int32Array.from(spec.A.edge);
    this.aVals = Float64Array.from(this.aEdge, (k) => this.weights[k]);
    this.stretch = [];                 // [inputs, free outputs] per committed tick
    this.readout = new Float64Array(spec.readouts.length);
    this.frozen = false;
    this.admissions = spec.admissions;
    this.lessons = 0; this.rejected = 0; this.refusals = 0;
  }

  inputsFor(u, state = this.state) {
    const v = new Float64Array(this.spec.n_inputs);
    for (let i = 0; i < this.H; i++) v[i] = state[this.pos[i]];
    for (let k = 0; k < this.senses.length; k++) v[this.H + k] = u[k];
    return v;
  }

  // each neuron's settled state, in connectome order
  activity() {
    return Float64Array.from(this.pos, (p) => this.state[p]);
  }

  outputsOf(state) {
    return this.readouts.map((name) => state[this.outputPatch[name]]);
  }

  // one committed tick: the previous settled states come back as inputs
  sense(u) {
    const inputs = this.inputsFor(u);
    const r = this.engine.settle(inputs, this.state, this.weights, this.biases, {});
    if (r.qualified) {
      this.state = r.state;
      const free = this.outputsOf(r.state);
      this.stretch.push([inputs, free]);
      if (this.stretch.length > this.p.window) this.stretch.shift();
      this.readout = Float64Array.from(free);
    } else {
      this.refusals++;
    }
    return this.readout;
  }

  targets(kinds, free) {
    const drive = { food: 0, pain: 1 };       // forward, reverse
    const t = free.slice();
    for (const kind of kinds) t[drive[kind]] = this.p.teach_level;
    if (kinds.length === 1) t[1 - drive[kinds[0]]] = 0;
    return t;
  }

  // one outcome, one batch: correct the last `teach` ticks, restate the rest
  learn(kinds) {
    const k = Math.min(this.p.teach, this.stretch.length);
    const rows = this.stretch.map(([inputs, free], i) =>
      [inputs, i < this.stretch.length - k ? free.slice() : this.targets(kinds, free)]);
    const length = this.stretch.length;
    this.stretch = [];
    if (this.frozen || !rows.length)
      { this.rejected++; return { kinds, updated: false, reason: this.frozen ? "frozen" : "no_experience", rows: rows.length, length }; }
    const B = rows.length, P = this.engine.nPatches, I = this.engine.nInputs;
    const inputs = new Float64Array(B * I), state = new Float64Array(B * P);
    const clamps = new Map();
    rows.forEach(([u, t], row) => {
      inputs.set(u, row * I);
      state.set(this.state, row * P);
      this.readouts.forEach((name, j) => clamps.set(row * P + this.outputPatch[name], t[j]));
    });
    const before = this.weights;
    const r = this.engine.settle(inputs, state, this.weights, this.biases,
                                 { clamps, learn: true, B });
    if (!r.qualified) { this.rejected++; return { kinds, updated: false, reason: r.reason, rows: B, length }; }
    const applied = Float64Array.from(r.weights, (w, i) => w - before[i]);
    const appliedBiases = Float64Array.from(r.biases, (b, i) => b - this.biases[i]);
    this.weights = r.weights;
    this.biases = r.biases;
    for (let i = 0; i < this.aEdge.length; i++) this.aVals[i] = this.weights[this.aEdge[i]];
    this.admissions++;
    this.lessons++;
    return { kinds, updated: true, reason: r.reason, rows: B, length,
             energy: r.energy, sweeps: r.sweeps, applied, appliedBiases };
  }

  // what one sense alone does to the drives now, from rest: a pure rollout of
  // settles that feeds its own returned states back in and never touches the
  // live state (each settle still starts its search from the live state, as
  // the library's settle does)
  probe(sense) {
    const u = new Float64Array(this.senses.length);
    u[this.senses.indexOf(sense)] = 1;
    let prev = new Float64Array(this.H);
    let outputs = this.readouts.map(() => 0);
    for (let t = 0; t < this.p.probe_ticks; t++) {
      const r = this.engine.settle(this.inputsFor(u, prev), this.state, this.weights, this.biases, {});
      if (!r.qualified) break;
      prev = r.state.subarray(0, this.H);
      outputs = this.outputsOf(r.state);
    }
    return outputs;
  }
}
