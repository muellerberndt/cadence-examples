// Portable numerical kernels. Python/Cadence parity is checked before publication.
export const argmax = (a) =>
  a.reduce((best, v, i) => (v > a[best] ? i : best), 0);
export const zeros = (n) => Array(n).fill(0);
export const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
export function keys(c = 0) {
  const a = Array.from({ length: 8 }, () => zeros(8));
  for (let i = 0; i < 8; i++)
    for (let j = 0; j <= i; j++) {
      let s = i === j ? 1 : c;
      for (let k = 0; k < j; k++) s -= a[i][k] * a[j][k];
      a[i][j] = i === j ? Math.sqrt(s) : s / a[j][j];
    }
  return a;
}
export class FastMemory {
  constructor() {
    this.w = Array.from({ length: 8 }, () => zeros(4));
  }
  predict(x) {
    return Array.from({ length: 4 }, (_, v) =>
      x.reduce((s, k, i) => s + k * this.w[i][v], 0),
    );
  }
  observe(x, y) {
    const norm = Math.sqrt(dot(x, x));
    if (!norm) return;
    x = x.map((v) => v / norm);
    const pred = this.predict(x);
    for (let k = 0; k < 8; k++)
      for (let v = 0; v < 4; v++) this.w[k][v] += x[k] * (y[v] - pred[v]);
  }
}
// Single-stream numerical translation of cadence.SynapticMemory. The port matrix
// is fixed size: total efficacy W = persistent C + transient residual F.
export class SynapticMemory extends FastMemory {
  constructor({ decay = 0.9, consolidation = 0.05 } = {}) {
    super();
    if (![decay, consolidation].every(v => Number.isFinite(v) && v >= 0 && v <= 1))
      throw Error("Memory rates must be in [0, 1]");
    Object.assign(this, { decay, consolidation });
    this.consolidated = this.w.map(row => row.slice());
  }
  // The third argument is the comparator's SGD budget, not a salience signal.
  observe(x, y, _steps = 1, { salience = 0, valueMask = [true, true, true, true] } = {}) {
    if (x.length !== 8 || y.length !== 4 || ![...x, ...y, salience].every(Number.isFinite)
        || salience < 0 || valueMask.length !== 4 || !valueMask.every(v => typeof v === "boolean"))
      throw Error("Expected finite cue/value, nonnegative salience and a boolean value mask");
    const norm = Math.sqrt(dot(x, x));
    if (!norm || !valueMask.some(Boolean)) return;
    x = x.map(v => v / norm);
    const slow = Array.from({ length: 4 }, (_, j) =>
      x.reduce((sum, v, i) => sum + v * this.consolidated[i][j], 0));
    const alpha = Math.min(1, this.consolidation * (1 + salience));
    const nextC = this.consolidated.map((row, i) => row.map((v, j) =>
      v + (valueMask[j] ? alpha * x[i] * (y[j] - slow[j]) : 0)));
    const nextW = this.w.map((row, i) => row.map((v, j) =>
      nextC[i][j] + this.decay * (v - this.consolidated[i][j])));
    const prediction = Array.from({ length: 4 }, (_, j) =>
      x.reduce((sum, v, i) => sum + v * nextW[i][j], 0));
    nextW.forEach((row, i) => row.forEach((v, j) => {
      if (valueMask[j]) row[j] += x[i] * (y[j] - prediction[j]);
    }));
    if (![...nextC.flat(), ...nextW.flat()].every(Number.isFinite))
      throw Error("Memory update overflow");
    this.consolidated = nextC;
    this.w = nextW;
  }
  reset() {
    this.w = this.consolidated.map(row => row.slice());
  }
  toJSON() {
    return { version: 1, decay: this.decay, consolidation: this.consolidation,
      w: this.w, consolidated: this.consolidated };
  }
  static restore(data) {
    const m = new SynapticMemory(data);
    const valid = a => Array.isArray(a) && a.length === 8
      && a.every(row => Array.isArray(row) && row.length === 4 && row.every(Number.isFinite));
    if (data.version !== 1 || !valid(data.w) || !valid(data.consolidated))
      throw Error("Invalid synaptic memory checkpoint");
    m.w = data.w.map(row => row.slice());
    m.consolidated = data.consolidated.map(row => row.slice());
    return m;
  }
  clear() {
    this.consolidated = this.w.map(row => row.map(() => 0));
    this.reset();
  }
}
export class MLP {
  constructor(data) {
    Object.assign(this, JSON.parse(JSON.stringify(data)));
  }
  hidden(x) {
    return this.b1.map((b, j) =>
      Math.tanh(b + x.reduce((s, v, i) => s + v * this.w1[i][j], 0)),
    );
  }
  predict(x) {
    const h = this.hidden(x);
    return this.b2.map(
      (b, j) => b + h.reduce((s, v, i) => s + v * this.w2[i][j], 0),
    );
  }
  observe(x, y, steps = 1, rate = 0.15) {
    for (let step = 0; step < steps; step++) {
      const h = this.hidden(x),
        error = this.b2.map(
          (b, j) =>
            (b + h.reduce((s, v, i) => s + v * this.w2[i][j], 0) - y[j]) /
            y.length,
        );
      const delta = h.map((v, i) => dot(this.w2[i], error) * (1 - v * v));
      for (let i = 0; i < h.length; i++)
        for (let j = 0; j < y.length; j++)
          this.w2[i][j] -= rate * h[i] * error[j];
      this.b2 = this.b2.map((v, j) => v - rate * error[j]);
      for (let i = 0; i < x.length; i++)
        for (let j = 0; j < h.length; j++)
          this.w1[i][j] -= rate * x[i] * delta[j];
      this.b1 = this.b1.map((v, j) => v - rate * delta[j]);
    }
  }
}
export class Worm {
  constructor(data) {
    this.data = data;
    this.n = data.names.length;
  }
  message(s) {
    const r = zeros(this.n);
    for (const [a, b, w] of this.data.edges) r[b] += w * s[a];
    return r;
  }
  next(s, drive, mask) {
    return this.message(s).map((v, i) => mask[i] * Math.tanh(v + drive[i]));
  }
  residual(s, d, m) {
    return Math.max(...this.next(s, d, m).map((v, i) => Math.abs(v - s[i])));
  }
  settle(d, m, steps = 200, tol = 1e-10, warm = null) {
    let s = warm ? warm.slice() : zeros(this.n),
      used = 0;
    for (; used < steps; used++) {
      const next = this.next(s, d, m);
      const change = Math.max(...next.map((v, i) => Math.abs(v - s[i])));
      s = next;
      if (change < tol) {
        used++;
        break;
      }
    }
    return { state: s, steps: used, residual: this.residual(s, d, m) };
  }
  surrogate(net, d, m) {
    const first = m.map((v, i) => v * Math.tanh(d[i])),
      msg = this.message(first);
    const raw = net.predict([...d, ...m, ...msg]);
    return raw.map((v, i) => m[i] * Math.tanh(d[i] + msg[i] + v));
  }
}
export function flowers() {
  return Array.from({ length: 8 }, (_, i) => ({
    x: 0.5 + 0.34 * Math.cos((i * Math.PI) / 4),
    y: 0.5 + 0.34 * Math.sin((i * Math.PI) / 4),
    kind: i,
    value: [3, 0, 2, 1, 0, 3, 1, 2][i],
  }));
}
export class Forager {
  constructor(memory, offset = 0, updates = 1) {
    this.memory = memory;
    this.updates = updates;
    this.x = 0.45 + offset;
    this.y = 0.5;
    this.angle = 0;
    this.target = -1;
    this.visits = zeros(8);
    this.cooldown = zeros(8);
    this.nectar = 0;
    this.encounters = 0;
    this.path = [];
    this.last = null;
    this.time = 0;
  }
  choose(field) {
    let best = -Infinity,
      id = -1;
    field.forEach((f, i) => {
      if (this.cooldown[i] > this.time) return;
      const prediction = this.memory.predict(keys()[f.kind]);
      const score =
        (this.visits[i] ? argmax(prediction) / 3 : 1.6) -
        0.25 * Math.hypot(f.x - this.x, f.y - this.y) +
        (this.encounters % 5 === 4 ? 1 / (1 + this.visits[i]) : 0);
      if (score > best) {
        best = score;
        id = i;
      }
    });
    this.target = id;
  }
  step(field, dt = 0.025) {
    this.time += dt;
    if (this.target < 0) this.choose(field);
    if (this.target < 0) return;
    const f = field[this.target],
      dx = f.x - this.x,
      dy = f.y - this.y,
      desired = Math.atan2(dy, dx);
    const turn = Math.atan2(
      Math.sin(desired - this.angle),
      Math.cos(desired - this.angle),
    );
    this.angle += Math.max(-3.5 * dt, Math.min(3.5 * dt, turn));
    const speed = 0.13 * Math.max(0.2, 1 - Math.abs(turn) / Math.PI);
    this.x = Math.max(
      0.03,
      Math.min(0.97, this.x + Math.cos(this.angle) * speed * dt),
    );
    this.y = Math.max(
      0.03,
      Math.min(0.97, this.y + Math.sin(this.angle) * speed * dt),
    );
    this.path.push([this.x, this.y]);
    if (this.path.length > 350) this.path.shift();
    if (Math.hypot(dx, dy) < 0.035) {
      const target = zeros(4);
      target[f.value] = 1;
      const predicted = argmax(this.memory.predict(keys()[f.kind]));
      // Nectar is revealed only by this physical encounter, never during choose().
      this.memory.observe(keys()[f.kind], target, this.updates);
      this.visits[this.target]++;
      this.encounters++;
      this.nectar += f.value;
      this.cooldown[this.target] = this.time + 4;
      this.last = { kind: f.kind, value: f.value, predicted, time: this.time };
      this.target = -1;
    }
  }
}
