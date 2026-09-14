import { zeros } from "./engine.js";
export function random(seed = 13) {
  let x = seed >>> 0;
  return () => {
    x = (Math.imul(1664525, x) + 1013904223) >>> 0;
    return x / 4294967296;
  };
}
export function forward(q) {
  const [a, b] = q;
  return [
    0.5 + 0.43 * Math.cos(a) + 0.37 * Math.cos(a + b),
    0.94 + 0.43 * Math.sin(a) + 0.37 * Math.sin(a + b),
  ];
}
export function jacobian(q) {
  const [a, b] = q;
  return [
    [-0.43 * Math.sin(a) - 0.37 * Math.sin(a + b), -0.37 * Math.sin(a + b)],
    [0.43 * Math.cos(a) + 0.37 * Math.cos(a + b), 0.37 * Math.cos(a + b)],
  ];
}
export function motorSettling(q, target) {
  const tip = forward(q),
    j = jacobian(q),
    drive = [target[0] - tip[0], target[1] - tip[1], 0, 0],
    edges = [];
  // Visual residual and motor correction settle TOGETHER with reciprocal feedback.
  for (let v = 0; v < 2; v++)
    for (let m = 0; m < 2; m++) {
      edges.push([m + 2, v, -j[v][m]]);
      edges.push([v, m + 2, 0.9 * j[v][m]]);
    }
  let potential = zeros(4),
    s = zeros(4);
  for (let step = 0; step < 80; step++) {
    const synapticInput = zeros(4);
    for (const [a, b, w] of edges) synapticInput[b] += w * s[a];
    potential = potential.map((v, i) => v + 0.25 * (synapticInput[i] + drive[i] - v));
    s = potential.map(Math.tanh);
  }
  return { state: s, drive, edges };
}
export function drawingTargets(kind = "flower") {
  const result = [];
  const count = kind === "spiral" ? 170 : 150;
  for (let i = 0; i < count; i++) {
    const t = (i / count) * Math.PI * 2;
    let x, y;
    if (kind === "spiral") {
      const r = 0.02 + (0.19 * i) / count;
      x = 0.5 + r * Math.cos(t * 3);
      y = 0.45 + r * Math.sin(t * 3);
    } else if (kind === "leaf") {
      x = 0.5 + 0.17 * Math.sin(t);
      y = 0.45 + 0.23 * Math.cos(t);
      x += 0.03 * Math.sin(t * 2);
    } else {
      const r = 0.15 + 0.055 * Math.cos(5 * t);
      x = 0.5 + r * Math.cos(t);
      y = 0.45 + r * Math.sin(t);
    }
    result.push([x, y]);
  }
  return result;
}
export class Arm {
  constructor(targets = drawingTargets()) {
    this.targets = targets;
    this.q = [-2.1, 1.2];
    this.estimate = this.q.slice();
    this.closed = true;
    this.ink = [];
    this.attempted = new Set();
    this.covered = new Set();
    this.target = 0;
    this.lifted = true;
    this.settling = null;
    this.ticks = 0;
  }
  disturb() {
    this.q[0] += 0.32;
    this.q[1] -= 0.22;
  }
  select() {
    const tip = this.closed ? forward(this.q) : forward(this.estimate);
    let best = -1,
      d = Infinity;
    this.targets.forEach((p, i) => {
      if (this.attempted.has(i)) return;
      const n = Math.hypot(p[0] - tip[0], p[1] - tip[1]);
      if (n < d) {
        d = n;
        best = i;
      }
    });
    this.target = best;
  }
  step() {
    this.ticks++;
    if (this.target < 0) return;
    const goal = this.targets[this.target],
      observed = this.closed ? this.q : this.estimate;
    this.settling = motorSettling(observed, goal);
    const change = this.settling.state
      .slice(2)
      .map((v) => Math.max(-0.09, Math.min(0.09, v * 0.9)));
    this.q = this.q.map((v, i) => v + change[i]);
    this.estimate = this.estimate.map((v, i) => v + change[i]);
    if (this.closed) this.estimate = this.q.slice();
    const tip = forward(this.q),
      readback = this.closed ? tip : forward(this.estimate),
      distance = Math.hypot(readback[0] - goal[0], readback[1] - goal[1]);
    this.lifted = distance > 0.022;
    if (!this.lifted) {
      this.ink.push(tip);
      this.targets.forEach((p, i) => {
        if (Math.hypot(p[0] - tip[0], p[1] - tip[1]) < 0.022)
          this.covered.add(i);
      });
    }
    if (distance < 0.009) {
      this.attempted.add(this.target);
      this.select();
    }
  }
  get coverage() {
    return this.covered.size / this.targets.length;
  }
}

