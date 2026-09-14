// Small explicit graded circuits. Motor outputs are consumed by body physics.
export const zeros = (n) => Array(n).fill(0);
export class Circuit {
  constructor(names, groups, edges, dt = 0.25) {
    Object.assign(this, { names, groups, edges, dt });
    this.state = zeros(names.length);
    this.potential = zeros(names.length);
    this.mask = this.state.map(() => 1);
  }
  settle(drive, steps = 24, tolerance = 0) {
    // Project ablated activations before their first outgoing message, as Cadence does.
    this.state = this.potential.map((v, i) => this.mask[i] * Math.tanh(v));
    const initialState = this.state.slice(),
      initialPotential = this.potential.slice();
    let used = 0;
    for (; used < steps; used++) {
      const inbox = zeros(this.state.length);
      for (const [a, b, w] of this.edges) inbox[b] += w * this.state[a];
      const error = this.potential.map((v, i) =>
        this.mask[i] ? inbox[i] + drive[i] - v : -v,
      );
      if (tolerance > 0 && Math.max(...error.map(Math.abs)) <= tolerance) break;
      this.potential = this.potential.map((v, i) =>
        this.mask[i] ? v + this.dt * (inbox[i] + drive[i] - v) : 0,
      );
      this.state = this.potential.map((v, i) => this.mask[i] * Math.tanh(v));
    }
    this.last = {
      state: this.state.slice(),
      drive: drive.slice(),
      input: drive.slice(),
      initialState,
      initialPotential,
      names: this.names,
      groups: this.groups,
      edges: this.edges.map((e) => e.slice()),
      mask: this.mask.slice(),
      recurrent: true,
      potential: this.potential.slice(),
      steps: used,
      dt: this.dt,
    };
    const inbox = zeros(this.state.length);
    for (const [a, b, w] of this.edges) inbox[b] += w * this.state[a];
    const errors = this.potential.map((v, i) =>
      Math.abs(this.mask[i] ? inbox[i] + drive[i] - v : v),
    );
    const regional = {};
    errors.forEach(
      (v, i) =>
        (regional[this.groups[i]] = Math.max(regional[this.groups[i]] ?? 0, v)),
    );
    this.last.equilibrium = {
      residual: Math.max(...errors),
      tolerance,
      converged: tolerance > 0 && Math.max(...errors) <= tolerance,
      regions: regional,
      links: this.edges.filter(
        ([a, b, w]) => w !== 0 && this.groups[a] !== this.groups[b],
      ).length,
    };
    return this.last;
  }
}

// Combine actual owners/edges before updating. All regions read the same previous
// whole-brain state; this is not concatenation of separately settled snapshots.
export function settleTogether(
  parts,
  drives,
  bridges = [],
  { steps = 200, tolerance = 1e-10, dt = 1, metadata = {} } = {},
) {
  const offsets = [];
  let n = 0;
  for (const part of parts) {
    offsets.push(n);
    n += part.state.length;
  }
  const graph = { names: [], groups: [], mask: [], edges: [], learned: [] };
  parts.forEach((p, k) => {
    const edgeOffset = graph.edges.length,
      offset = offsets[k];
    graph.names.push(...p.names);
    graph.groups.push(...p.groups);
    graph.mask.push(...(p.mask ?? p.state.map(() => 1)));
    graph.edges.push(
      ...p.edges.map(([a, b, w]) => [offset + a, offset + b, w]),
    );
    graph.learned.push(
      ...(p.learned ?? []).map(([i, id, w]) => [edgeOffset + i, id, w]),
    );
  });
  const edges = [
    ...graph.edges,
    ...bridges.map(([a, i, b, j, w]) => [offsets[a] + i, offsets[b] + j, w]),
  ];
  const net = new Circuit(graph.names, graph.groups, edges, dt);
  net.state = parts.flatMap((p) => p.state);
  net.potential = parts.flatMap(
    (p) =>
      p.potential ??
      p.state.map((v) =>
        Math.atanh(Math.max(-0.999999999, Math.min(0.999999999, v))),
      ),
  );
  net.mask = graph.mask;
  const result = net.settle(drives.flat(), steps, tolerance);
  parts.forEach((p, k) => {
    p.state = net.state.slice(offsets[k], offsets[k] + p.state.length);
    p.potential = net.potential.slice(offsets[k], offsets[k] + p.state.length);
  });
  return Object.assign(result, { learned: graph.learned }, metadata);
}

// FastSeams retains the observations. These owners publish the current recall
// through the same graded rule as the rest of a task brain. Argmax is unchanged.
export class MemoryReadout extends Circuit {
  constructor() {
    super(
      [
        ...Array.from({ length: 8 }, (_, i) => `Cue ${i + 1}`),
        ...Array.from({ length: 4 }, (_, i) => `Recall ${i}`),
      ],
      [...Array(8).fill("key"), ...Array(4).fill("record")],
      [],
    );
  }
  configure(memory, key) {
    if (!memory.w) {
      this.edges = [];
      this.learned = [];
      return [...key.map((v) => Math.atanh(0.8 * v)), ...memory.predict(key)];
    }
    this.edges = memory.w.flatMap((row, i) =>
      row.map((w, j) => [i, 8 + j, w / 0.8]),
    );
    this.learned = memory.w.flatMap((row, i) =>
      row.map((w, j) => [i * 4 + j, `key-${i}-value-${j}`, w]),
    );
    return [...key.map((v) => Math.atanh(0.8 * v)), ...zeros(4)];
  }
}
export const motorFeedback = (part, axes) =>
  Array.from({ length: axes }, (_, i) => [
    [part, axes + 2 * i, part, i, -0.12],
    [part, axes + 2 * i + 1, part, i, 0.12],
  ]).flat();
// Directional sensory error -> antagonistic motor population; retained potential is short-lived state.
export class MotorSystem extends Circuit {
  constructor(axes = ["X", "Y"]) {
    const n = axes.length;
    super(
      [
        ...axes.map((a) => `${a} sensory error`),
        ...axes.flatMap((a) => [`${a} positive motor`, `${a} negative motor`]),
      ],
      [
        ...axes.map(() => "sensor"),
        ...axes.flatMap(() => ["actuator", "actuator"]),
      ],
      axes.flatMap((_, i) => [
        [i, n + i * 2, 2],
        [i, n + i * 2 + 1, -2],
      ]),
    );
    this.axes = axes;
    this.settle(zeros(n * 3));
  }
  command(error) {
    this.settle([...error, ...zeros(error.length * 2)]);
    return error.map(
      (_, i) =>
        (Math.max(0, this.state[error.length + i * 2]) -
          Math.max(0, this.state[error.length + i * 2 + 1])) /
        2,
    );
  }
}
