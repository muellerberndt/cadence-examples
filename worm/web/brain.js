// The worm's brain in the browser: cadence 0.11.0 TemporalPatchNet mathematics
// on the connectome's sparse connectivity. Verified against the Python library
// by tests/parity.mjs.
//
//   h[t] = A tanh(h[t-1]) + B u[t]          free recurrence (exact forward pass)
//   y[t] = C tanh(h[t])                     command-neuron readouts
//
// A lesson solves two detuned equilibria of the path energy (outputs nudged by
// +beta and -beta toward the observed target), takes the centred contrast of
// each synapse's local derivative, and keeps a step only if a target-free
// replay from the same starting state predicts the observation better.

const tanh = Math.tanh;

export class Brain {
  constructor(spec) {
    this.H = spec.H; this.I = spec.I; this.O = spec.O;
    this.tolerance = spec.tolerance; this.maxIterations = spec.max_iterations;
    this.maxBacktracks = spec.max_backtracks; this.maxDamping = spec.max_damping_trials;
    const H = this.H;
    // A in CSR by postsynaptic row; entries are exactly the connectome mask.
    const nnz = spec.A.rows.length;
    this.rowPtr = new Int32Array(H + 1);
    for (const r of spec.A.rows) this.rowPtr[r + 1]++;
    for (let i = 0; i < H; i++) this.rowPtr[i + 1] += this.rowPtr[i];
    this.col = new Int32Array(nnz); this.val = new Float64Array(nnz); this.row = new Int32Array(nnz);
    const fill = this.rowPtr.slice();
    spec.A.rows.forEach((r, k) => {
      const p = fill[r]++; this.col[p] = spec.A.cols[k]; this.val[p] = spec.A.vals[k]; this.row[p] = r;
    });
    this.Bmask = []; this.B = new Float64Array(H * this.I);
    for (const [n, k, v] of spec.B) { this.B[n * this.I + k] = v; this.Bmask.push(n * this.I + k); }
    this.Cmask = []; this.C = new Float64Array(this.O * H);
    for (const [o, n, v] of spec.C) { this.C[o * H + n] = v; this.Cmask.push(o * H + n); }
    this.state = null;
    this.updates = 0;
  }

  // ---- linear algebra ----------------------------------------------------------
  Ax(x, out) {                 // out = A x
    const { rowPtr, col, val } = this;
    for (let i = 0; i < this.H; i++) {
      let s = 0; for (let k = rowPtr[i]; k < rowPtr[i + 1]; k++) s += val[k] * x[col[k]];
      out[i] = s;
    }
  }
  ATx(x, out) {                // out = A^T x
    out.fill(0);
    const { rowPtr, col, val } = this;
    for (let i = 0; i < this.H; i++) {
      const xi = x[i]; if (xi === 0) continue;
      for (let k = rowPtr[i]; k < rowPtr[i + 1]; k++) out[col[k]] += val[k] * xi;
    }
  }
  Bu(u, out) {                 // out += B u
    const I = this.I;
    for (const m of this.Bmask) { const n = (m / I) | 0, k = m - n * I; out[n] += this.B[m] * u[k]; }
  }
  Cx(x, out) {                 // out = C x   (x already tanh'd)
    out.fill(0); const H = this.H;
    for (const m of this.Cmask) { const o = (m / H) | 0, n = m - o * H; out[o] += this.C[m] * x[n]; }
  }
  CTy(y, out) {                // out = C^T y
    out.fill(0); const H = this.H;
    for (const m of this.Cmask) { const o = (m / H) | 0, n = m - o * H; out[n] += this.C[m] * y[o]; }
  }

  // ---- paths --------------------------------------------------------------------
  boundary() { return this.state ? this.state.slice() : new Float64Array(this.H); }

  causal(U, boundary) {        // U: array of Float64Array(I)
    const H = this.H, T = U.length, hidden = new Float64Array(T * H);
    let prev = boundary.map(tanh); const tmp = new Float64Array(H);
    for (let t = 0; t < T; t++) {
      this.Ax(prev, tmp); this.Bu(U[t], tmp);
      hidden.set(tmp, t * H);
      prev = tmp.map(tanh);
    }
    return hidden;
  }

  outputs(hidden, T) {
    const H = this.H, O = this.O, y = new Float64Array(T * O), s = new Float64Array(H), o = new Float64Array(O);
    for (let t = 0; t < T; t++) {
      for (let i = 0; i < H; i++) s[i] = tanh(hidden[t * H + i]);
      this.Cx(s, o); y.set(o, t * O);
    }
    return y;
  }

  imagine(U, state = null) {
    const b = state ?? this.boundary();
    const hidden = this.causal(U, b);
    return { hidden, output: this.outputs(hidden, U.length), T: U.length };
  }

  advance(U) {
    const p = this.imagine(U);
    this.state = p.hidden.slice((p.T - 1) * this.H);
    return p;
  }

  // transition defects e[t] = h[t] - A tanh(h[t-1]) - B u[t]
  defects(U, hidden, boundary) {
    const H = this.H, T = U.length, e = new Float64Array(T * H), prev = new Float64Array(H), tmp = new Float64Array(H);
    for (let t = 0; t < T; t++) {
      if (t === 0) for (let i = 0; i < H; i++) prev[i] = tanh(boundary[i]);
      else for (let i = 0; i < H; i++) prev[i] = tanh(hidden[(t - 1) * H + i]);
      this.Ax(prev, tmp); this.Bu(U[t], tmp);
      for (let i = 0; i < H; i++) e[t * H + i] = hidden[t * H + i] - tmp[i];
    }
    return e;
  }

  // ---- one detuned equilibrium ----------------------------------------------------
  // Reduced energy in h after eliminating the outputs analytically:
  //   E(h) = 1/2 sum |e_t|^2 + lam/2 sum |C tanh h_t - target_t|^2,  lam = b/(1+b)
  solve(U, boundary, target, beta) {
    const H = this.H, O = this.O, T = U.length, N = T * H;
    const b = beta / (T * O), lam = b / (1 + b);
    let hidden = this.causal(U, boundary);
    const sig = new Float64Array(N), d1 = new Float64Array(N), d2 = new Float64Array(N);
    const tmpH = new Float64Array(H), tmpH2 = new Float64Array(H), tmpO = new Float64Array(O);

    const energy = (h) => {
      const e = this.defects(U, h, boundary); let s = 0;
      for (let k = 0; k < N; k++) s += e[k] * e[k];
      const x = new Float64Array(H);
      for (let t = 0; t < T; t++) {
        for (let i = 0; i < H; i++) x[i] = tanh(h[t * H + i]);
        this.Cx(x, tmpO);
        for (let o = 0; o < O; o++) { const q = tmpO[o] - target[t * O + o]; s += lam * q * q; }
      }
      return 0.5 * s;
    };

    let e, qC, aTe;   // cached pieces at the current point
    const prepare = (h) => {
      for (let k = 0; k < N; k++) { const s = tanh(h[k]); sig[k] = s; d1[k] = 1 - s * s; d2[k] = -2 * s * (1 - s * s); }
      e = this.defects(U, h, boundary);
      qC = new Float64Array(N); aTe = new Float64Array(N);
      const x = new Float64Array(H), q = new Float64Array(O);
      for (let t = 0; t < T; t++) {
        x.set(sig.subarray(t * H, (t + 1) * H)); this.Cx(x, tmpO);
        for (let o = 0; o < O; o++) q[o] = tmpO[o] - target[t * O + o];
        this.CTy(q, tmpH); qC.set(tmpH, t * H);
        if (t < T - 1) { this.ATx(e.subarray((t + 1) * H, (t + 2) * H), tmpH); aTe.set(tmpH, t * H); }
      }
    };
    const gradient = () => {
      const g = new Float64Array(N);
      for (let k = 0; k < N; k++) g[k] = e[k] + lam * d1[k] * qC[k] - d1[k] * aTe[k];
      return g;
    };
    const hvp = (v, damping) => {   // (Hessian + damping I) v
      const out = new Float64Array(N), de = new Float64Array(N), x = new Float64Array(H);
      for (let t = 0; t < T; t++) {
        for (let i = 0; i < H; i++) de[t * H + i] = v[t * H + i];
        if (t > 0) {
          for (let i = 0; i < H; i++) x[i] = d1[(t - 1) * H + i] * v[(t - 1) * H + i];
          this.Ax(x, tmpH); for (let i = 0; i < H; i++) de[t * H + i] -= tmpH[i];
        }
      }
      const q = new Float64Array(O);
      for (let t = 0; t < T; t++) {
        for (let i = 0; i < H; i++) x[i] = d1[t * H + i] * v[t * H + i];
        this.Cx(x, q); this.CTy(q, tmpH);
        if (t < T - 1) this.ATx(de.subarray((t + 1) * H, (t + 2) * H), tmpH2); else tmpH2.fill(0);
        for (let i = 0; i < H; i++) {
          const k = t * H + i;
          out[k] = de[k] + lam * d2[k] * v[k] * qC[k] + lam * d1[k] * tmpH[i]
                 - d2[k] * v[k] * aTe[k] - d1[k] * tmpH2[i] + damping * v[k];
        }
      }
      return out;
    };
    const cg = (rhs, damping) => {    // conjugate gradients; null on nonpositive curvature
      const x = new Float64Array(N), r = rhs.slice(), p = rhs.slice();
      let rr = 0; for (let k = 0; k < N; k++) rr += r[k] * r[k];
      const stop = Math.max(1e-30, 1e-24 * rr);
      for (let it = 0; it < 600 && rr > stop; it++) {
        const Ap = hvp(p, damping); let pAp = 0;
        for (let k = 0; k < N; k++) pAp += p[k] * Ap[k];
        if (!(pAp > 0)) return null;
        const a = rr / pAp; let rr2 = 0;
        for (let k = 0; k < N; k++) { x[k] += a * p[k]; r[k] -= a * Ap[k]; rr2 += r[k] * r[k]; }
        const beta2 = rr2 / rr; rr = rr2;
        for (let k = 0; k < N; k++) p[k] = r[k] + beta2 * p[k];
      }
      return x;
    };

    let E = energy(hidden), converged = false, reason = "iteration_cap", iterations = 0;
    for (let it = 0; it <= this.maxIterations; it++) {
      prepare(hidden);
      const g = gradient();
      let res = 0; for (let k = 0; k < N; k++) res = Math.max(res, Math.abs(g[k]));
      if (!isFinite(res)) { reason = "nonfinite_residual"; break; }
      if (res <= this.tolerance) {
        // the stationary point must be a minimum: the undamped system is positive definite
        converged = cg(g, 0) !== null; reason = converged ? "converged" : "nonminimum_stationary_point";
        break;
      }
      if (it === this.maxIterations) break;
      let damping = 0, accepted = false;
      for (let trial = 0; trial < this.maxDamping && !accepted; trial++) {
        const neg = g.map((x) => -x);
        const dir = cg(neg, damping);
        if (dir) {
          let slope = 0; for (let k = 0; k < N; k++) slope += g[k] * dir[k];
          if (isFinite(slope) && slope <= 0) {
            let step = 1;
            for (let bt = 0; bt < this.maxBacktracks; bt++) {
              const prop = new Float64Array(N);
              for (let k = 0; k < N; k++) prop[k] = hidden[k] + step * dir[k];
              const E2 = energy(prop);
              if (isFinite(E2) && E2 <= E + 1e-4 * step * slope + 1e-15 * Math.max(1, Math.abs(E))) {
                hidden = prop; E = E2; accepted = true; iterations++; break;
              }
              step *= 0.5;
            }
          }
        }
        damping = damping === 0 ? 1e-8 : damping * 10;
      }
      if (!accepted) { reason = "no_decreasing_spd_step"; break; }
    }
    // outputs are eliminated analytically: y = (C tanh h + b target) / (1 + b)
    const free = this.outputs(hidden, T), output = new Float64Array(T * O);
    for (let k = 0; k < T * O; k++) output[k] = (free[k] + b * target[k]) / (1 + b);
    return { hidden, output, converged, reason, iterations, T };
  }

  // local parameter derivatives at one phase (masked; batch of one)
  derivatives(U, boundary, phase) {
    const H = this.H, O = this.O, T = phase.T;
    const e = this.defects(U, phase.hidden, boundary);
    const sig = phase.hidden.map(tanh);
    const dA = new Float64Array(this.val.length), dB = new Float64Array(this.Bmask.length), dC = new Float64Array(this.Cmask.length);
    for (let t = 0; t < T; t++) {
      for (let k = 0; k < this.val.length; k++) {
        const pre = t === 0 ? tanh(boundary[this.col[k]]) : sig[(t - 1) * H + this.col[k]];
        dA[k] -= e[t * H + this.row[k]] * pre;
      }
      this.Bmask.forEach((m, j) => { const n = (m / this.I) | 0, i = m - n * this.I; dB[j] -= e[t * H + n] * U[t][i]; });
      const cs = new Float64Array(O), x = sig.subarray(t * H, (t + 1) * H);
      this.Cx(x, cs);
      this.Cmask.forEach((m, j) => { const o = (m / H) | 0, n = m - o * H; dC[j] -= (phase.output[t * O + o] - cs[o]) * x[n]; });
    }
    return { dA, dB, dC };
  }

  parameters() { return { A: this.val.slice(), B: this.Bmask.map((m) => this.B[m]), C: this.Cmask.map((m) => this.C[m]) }; }
  setParameters(p) {
    this.val.set(p.A);
    this.Bmask.forEach((m, j) => { this.B[m] = p.B[j]; });
    this.Cmask.forEach((m, j) => { this.C[m] = p.C[j]; });
  }

  // one finite observed path, exactly cadence's observe(..., backtrack=True)
  observe(U, target, beta, rate) {
    const T = U.length, O = this.O;
    if (!(beta > 0 && beta < T * O)) throw new Error("need 0 < beta < time*outputs");
    const boundary = this.boundary();
    const free = this.imagine(U, boundary);
    this.state = free.hidden.slice((T - 1) * this.H);          // the free path is carried either way
    const plus = this.solve(U, boundary, target, beta);
    const minus = this.solve(U, boundary, target, -beta);
    if (!plus.converged || !minus.converged) return { updated: false, reason: "phase_failed", free, plus, minus };
    const gp = this.derivatives(U, boundary, plus), gm = this.derivatives(U, boundary, minus);
    const delta = {
      A: gp.dA.map((v, k) => (v - gm.dA[k]) / (2 * beta)),
      B: gp.dB.map((v, k) => (v - gm.dB[k]) / (2 * beta)),
      C: gp.dC.map((v, k) => (v - gm.dC[k]) / (2 * beta)),
    };
    const loss = (y) => { let s = 0; for (let k = 0; k < y.length; k++) s += (y[k] - target[k]) ** 2; return 0.5 * s / y.length; };
    const initial = loss(free.output);
    let norm2 = 0; for (const key of ["A", "B", "C"]) for (const v of delta[key]) norm2 += v * v;
    if (!(isFinite(initial) && isFinite(norm2) && norm2 > 0 && rate > 0))
      return { updated: false, reason: "no_decreasing_parameter_step", free, plus, minus, delta };
    const before = this.parameters();
    for (let index = 0; index < 16; index++) {
      const step = rate * 0.5 ** index;
      const prop = { A: before.A.map((v, k) => v - step * delta.A[k]),
                     B: before.B.map((v, k) => v - step * delta.B[k]),
                     C: before.C.map((v, k) => v - step * delta.C[k]) };
      this.setParameters(prop);
      const current = loss(this.imagine(U, boundary).output);
      const floor = 64 * Number.EPSILON * Math.max(Math.abs(initial), Math.abs(current || 0), Number.MIN_VALUE);
      if (isFinite(current) && current < initial - floor && current <= initial - 1e-4 * step * norm2) {
        this.updates++;
        return { updated: true, reason: "updated", free, plus, minus, delta, step, before, initial, current };
      }
    }
    this.setParameters(before);
    return { updated: false, reason: "no_decreasing_parameter_step", free, plus, minus, delta, initial };
  }

  // recurrent growth rate, identical to WormBrain.growth in Python
  growth(A = null) {
    const H = this.H, saved = this.val;
    if (A) this.val = A;
    let x = new Float64Array(H), y = new Float64Array(H), n0 = 0;
    for (let i = 0; i < H; i++) { x[i] = H === 1 ? 1 : 1 + i / (H - 1); n0 += x[i] * x[i]; }
    n0 = Math.sqrt(n0); for (let i = 0; i < H; i++) x[i] /= n0;
    let log = 0;
    for (let k = 0; k < 64; k++) {
      this.Ax(x, y); let n = 0; for (let i = 0; i < H; i++) n += y[i] * y[i]; n = Math.sqrt(n);
      if (n === 0) { this.val = saved; return 0; }
      log += Math.log(n); for (let i = 0; i < H; i++) x[i] = y[i] / n;
    }
    this.val = saved;
    return Math.exp(log / 64);
  }
}
