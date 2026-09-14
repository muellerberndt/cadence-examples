import { Worm, zeros, SynapticMemory, keys, argmax } from "./engine.js";
export function random(seed = 13) {
  let x = seed >>> 0;
  return () => {
    x = (Math.imul(1664525, x) + 1013904223) >>> 0;
    return x / 4294967296;
  };
}
export function maze(seed = 13, cols = 19, rows = 13) {
  const rng = random(seed),
    grid = Array(rows * cols).fill(1),
    stack = [[1, 1]];
  grid[cols + 1] = 0;
  while (stack.length) {
    const [x, y] = stack[stack.length - 1],
      options = [
        [2, 0],
        [-2, 0],
        [0, 2],
        [0, -2],
      ].filter(
        ([dx, dy]) =>
          x + dx > 0 &&
          x + dx < cols - 1 &&
          y + dy > 0 &&
          y + dy < rows - 1 &&
          grid[(y + dy) * cols + x + dx],
      );
    if (!options.length) {
      stack.pop();
      continue;
    }
    const [dx, dy] = options[Math.floor(rng() * options.length)];
    grid[(y + dy / 2) * cols + x + dx / 2] = 0;
    grid[(y + dy) * cols + x + dx] = 0;
    stack.push([x + dx, y + dy]);
  }
  // Extra corridors make rerouting possible; no hidden planner creates the path.
  for (let y = 1; y < rows - 1; y++)
    for (let x = 1; x < cols - 1; x++)
      if (
        grid[y * cols + x] &&
        rng() < 0.12 &&
        ((!grid[y * cols + x - 1] && !grid[y * cols + x + 1]) ||
          (!grid[(y - 1) * cols + x] && !grid[(y + 1) * cols + x]))
      )
        grid[y * cols + x] = 0;
  return { grid, cols, rows, goal: (rows - 2) * cols + cols - 2 };
}
export function neighbors(world, i) {
  const { cols, rows, grid } = world,
    x = i % cols,
    y = Math.floor(i / cols);
  return [
    [x + 1, y],
    [x - 1, y],
    [x, y + 1],
    [x, y - 1],
  ]
    .filter(
      ([a, b]) =>
        a >= 0 && b >= 0 && a < cols && b < rows && !grid[b * cols + a],
    )
    .map(([a, b]) => b * cols + a);
}
export function mazeCircuit(world, settle = true) {
  const n = world.grid.length,
    edges = [];
  for (let i = 0; i < n; i++) {
    if (world.grid[i]) continue;
    const ns = neighbors(world, i);
    for (const j of ns) edges.push([j, i, 0.985 / ns.length]);
  }
  const brain = new Worm({ names: zeros(n), edges }),
    drive = zeros(n),
    mask = world.grid.map((v) => 1 - v);
  drive[world.goal] = 0.03;
  const result = settle ? brain.settle(drive, mask, 2400, 1e-13)
    : { state: zeros(n), residual: Infinity, steps: 0 };
  return { brain, drive, mask, ...result };
}
export class Mouse {
  constructor(seed = 13) {
    this.seed = seed;
    this.world = maze(seed);
    this.cell = this.world.cols + 1;
    this.x = 1.5;
    this.y = 1.5;
    this.heading = 0;
    this.path = [this.cell];
    this.moves = 0;
    this.arrivals = 0;
    this.target = -1;
    this.repair();
  }
  repair() {
    this.circuit = mazeCircuit(this.world);
    this.target = -1;
  }
  next() {
    if (this.cell === this.world.goal) return -1;
    const options = neighbors(this.world, this.cell);
    let best = -1,
      value = this.circuit.state[this.cell];
    for (const i of options)
      if (this.circuit.state[i] > value + 1e-16) {
        value = this.circuit.state[i];
        best = i;
      }
    return best;
  }
  step(dt = 0.025) {
    if (this.cell === this.world.goal) return;
    if (this.target < 0) this.target = this.next();
    if (this.target < 0) return;
    const tx = (this.target % this.world.cols) + 0.5,
      ty = Math.floor(this.target / this.world.cols) + 0.5,
      dx = tx - this.x,
      dy = ty - this.y,
      d = Math.hypot(dx, dy);
    this.heading = Math.atan2(dy, dx);
    const step = dt * 3.2;
    if (d <= step) {
      this.x = tx;
      this.y = ty;
      this.cell = this.target;
      this.moves++;
      this.path.push(this.cell);
      this.target = -1;
      if (this.cell === this.world.goal) this.arrivals++;
    } else {
      this.x += (dx / d) * step;
      this.y += (dy / d) * step;
    }
  }
  toggle(i) {
    if (i === this.cell || i === this.world.goal || i === this.target)
      return false;
    const x = i % this.world.cols,
      y = Math.floor(i / this.world.cols);
    if (
      x === 0 ||
      y === 0 ||
      x === this.world.cols - 1 ||
      y === this.world.rows - 1
    )
      return false;
    this.world.grid[i] = 1 - this.world.grid[i];
    this.repair();
    return true;
  }
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

// Explicit task cues are learned associations, not natural-language understanding.
export class TaskLessons {
  constructor(
    records = [
      [0, 0],
      [1, 1],
      [2, 2],
    ],
  ) {
    this.memory = new SynapticMemory();
    this.records = [];
    this.known = new Set();
    const saved = Array.isArray(records) ? null : records;
    for (const [cue, dest] of saved?.records ?? records) this.teach(cue, dest);
    if (saved) {
      if (saved.version !== 2) throw Error("Invalid task checkpoint");
      this.memory = SynapticMemory.restore(saved.memory);
    }
  }
  toJSON() {
    return { version: 2, records: this.records, memory: this.memory.toJSON() };
  }
  teach(cue, destination) {
    if (
      !Number.isInteger(cue) ||
      cue < 0 ||
      cue > 7 ||
      !Number.isInteger(destination) ||
      destination < 0 ||
      destination > 3
    )
      throw Error("Invalid lesson");
    const value = zeros(4);
    value[destination] = 1;
    this.memory.observe(keys()[cue], value);
    this.known.add(cue);
    this.records = this.records.filter((r) => r[0] !== cue);
    this.records.push([cue, destination]);
  }
  recall(cue) {
    return this.known.has(cue)
      ? argmax(this.memory.predict(keys()[cue]))
      : null;
  }
}
export function stations(world) {
  const desired = [
    [17, 11],
    [1, 1],
    [1, 11],
    [17, 1],
  ];
  return desired.map(([x, y]) => {
    let best = -1,
      d = Infinity;
    world.grid.forEach((wall, i) => {
      if (wall) return;
      const distance = Math.hypot((i % 19) - x, Math.floor(i / 19) - y);
      if (distance < d) {
        d = distance;
        best = i;
      }
    });
    return best;
  });
}
