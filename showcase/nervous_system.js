// Small explicit graded circuits. Motor outputs are consumed by body physics.
export const zeros = (n) => Array(n).fill(0);
export class Circuit {
  constructor(names, groups, edges, dt = 0.25) {
    Object.assign(this, { names, groups, edges, dt });
    this.state = zeros(names.length);
    this.potential = zeros(names.length);
    this.mask = this.state.map(() => 1);
  }
  settle(drive, steps = 24) {
    const initialState = this.state.slice(),
      initialPotential = this.potential.slice();
    for (let t = 0; t < steps; t++) {
      const inbox = zeros(this.state.length);
      for (const [a, b, w] of this.edges) inbox[b] += w * this.state[a];
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
      steps,
      dt: this.dt,
    };
    return this.last;
  }
}
export function compose(parts, metadata = {}) {
  let offset = 0;
  const result = {
    state: [],
    drive: [],
    input: [],
    mask: [],
    names: [],
    groups: [],
    edges: [],
    learned: [],
    blocks: parts,
    recurrent: parts.some((p) => p.recurrent),
    steps: Math.max(...parts.map((p) => p.steps ?? 0)),
  };
  for (const part of parts) {
    const edgeOffset = result.edges.length;
    for (const key of ["state", "drive", "input", "mask", "names", "groups"]) {
      const fallback =
        key === "mask"
          ? part.state.map(() => 1)
          : key === "input"
            ? (part.drive ?? zeros(part.state.length))
            : zeros(part.state.length);
      result[key].push(...(part[key] ?? fallback));
    }
    result.edges.push(
      ...part.edges.map(([a, b, w]) => [a + offset, b + offset, w]),
    );
    result.learned.push(
      ...(part.learned ?? []).map(([i, id, v]) => [i + edgeOffset, id, v]),
    );
    offset += part.state.length;
  }
  result.recurrentMask = parts.flatMap((p) =>
    p.state.map(
      (_, i) => !!p.recurrent && i < (p.recurrentCount ?? p.state.length),
    ),
  );
  return Object.assign(result, metadata);
}
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
