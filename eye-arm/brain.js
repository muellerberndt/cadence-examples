import { Circuit, settleTogether, zeros } from "../shared/nervous_system.js";
import { forward, jacobian } from "../shared/embodied.js";
import { DrawingPaper, pixelPoint, pixelAt } from "./paper.js";
export { RETINA } from "./paper.js";
import { RETINA } from "./paper.js";
const ARRIVAL = 0.0015;
export class DrawingBrain {
  constructor() {
    this.motor = new Circuit(
      [
        "Target X",
        "Target Y",
        "Pen X",
        "Pen Y",
        "Requested height",
        "Pen height",
        "Visual error X",
        "Visual error Y",
        "Height error",
        "Shoulder correction",
        "Elbow correction",
        "Shoulder flexor",
        "Shoulder extensor",
        "Elbow flexor",
        "Elbow extensor",
        "Pencil raise",
        "Pencil lower",
      ],
      [
        ...Array(6).fill("readback"),
        ...Array(3).fill("error"),
        ...Array(2).fill("premotor"),
        ...Array(6).fill("actuator"),
      ],
      [],
    );
    this.see(zeros(RETINA ** 2));
    this.motor.settle(zeros(17));
  }
  see(darkness) {
    if (
      darkness.length !== RETINA ** 2 ||
      darkness.some((v) => !Number.isFinite(v) || v < 0 || v > 1)
    )
      throw Error("Invalid retina");
    this.pixels = darkness.slice();
    // Sparse visual patches: every dark sample has a reference neuron, an ink
    // readback neuron and a missing-ink neuron. White background has no target.
    this.indices = darkness.flatMap((v, i) => (v > 0.117 ? [i] : []));
    this.lookup = new Map(this.indices.map((pixel, i) => [pixel, i]));
    this.targets = this.indices.map(pixelPoint);
    const names = this.indices.map(
      (i) => `${i % RETINA},${Math.floor(i / RETINA)}`,
    );
    const region = (label, group) =>
      new Circuit(
        names.map((p) => `${label} ${p}`),
        names.map(() => group),
        [],
      );
    this.eye = region("Reference sample", "retina");
    this.inkEye = region("Observed ink", "ink");
    this.missing = region("Missing mark", "missing");
    this.inkReadback = names.map(() => 0);
  }
  readPaper(paper) {
    this.inkReadback = this.targets.map((p) => paper.at(p));
  }
  command(q, z, target, requestedHeight) {
    const tip = forward(q),
      j = jacobian(q);
    const edges = [
      [0, 6, 1],
      [2, 6, -1],
      [1, 7, 1],
      [3, 7, -1],
      [4, 8, 1],
      [5, 8, -1],
      [9, 11, 1],
      [9, 12, -1],
      [10, 13, 1],
      [10, 14, -1],
      [8, 15, 2],
      [8, 16, -2],
    ];
    for (let v = 0; v < 2; v++)
      for (let m = 0; m < 2; m++)
        edges.push([9 + m, 6 + v, -j[v][m]], [6 + v, 9 + m, 0.9 * j[v][m]]);
    this.motor.edges = edges;
    // Encode observed values as graded sensory drives, so readback units carry the coordinates themselves.
    const observed = [
      target[0] - 0.5,
      target[1] - 0.5,
      tip[0] - 0.5,
      tip[1] - 0.5,
      requestedHeight * 0.6,
      z * 0.6,
    ];
    const drive = [
      ...observed.map((v) => Math.atanh(Math.max(-0.99, Math.min(0.99, v)))),
      ...zeros(11),
    ];
    // The attended pixel publishes the coordinate drive into the motor region.
    // Geometry/attention selects these ports; their signal still travels as a synapse.
    const pixel = this.lookup.get(pixelAt(target));
    const synapses = [];
    if (pixel !== undefined)
      for (let axis = 0; axis < 2; axis++) {
        synapses.push([0, pixel, 1, axis, drive[axis] / 0.8]);
        drive[axis] = 0;
      }
    this.indices.forEach((_, i) =>
      synapses.push([0, i, 3, i, 1], [2, i, 3, i, -1]),
    );
    for (const [positive, negative, receiver] of [
      [11, 12, 9],
      [13, 14, 10],
      [15, 16, 8],
    ])
      synapses.push(
        [1, positive, 1, receiver, -0.1],
        [1, negative, 1, receiver, 0.1],
      );
    this.joint = settleTogether(
      [this.eye, this.motor, this.inkEye, this.missing],
      [
        this.indices.map(() => Math.atanh(0.8)),
        drive,
        this.inkReadback.map((v) => Math.atanh(0.8 * v)),
        zeros(this.indices.length),
      ],
      synapses,
    );
    const s = this.motor.state;
    return [
      [11, 12],
      [13, 14],
      [15, 16],
    ].map(([a, b]) => Math.max(0, s[a]) - Math.max(0, s[b]));
  }
  snapshot() {
    return Object.assign(this.joint, {
      visualSamples: Object.fromEntries(
        ["retina", "ink", "missing"].map((g) => [
          g,
          this.indices.map((i) => [
            ((i % RETINA) + 0.5) / RETINA,
            (Math.floor(i / RETINA) + 0.5) / RETINA,
          ]),
        ]),
      ),
      regionLabels: {
        retina: "Retina · reference marks",
        ink: "Retina · actual ink",
        missing: "Missing ink",
        readback: "Target & proprioception",
        error: "Visual / height error",
        premotor: "Joint coordination",
        actuator: "Motor neurons",
      },
      adapters:
        "Reference + ink → missing marks → attention; target ↔ position error ↔ joints ↔ motors · one shared equilibrium",
      memory:
        "Retained graded potentials carry transient state between control ticks. Missing-ink neurons compare reference and actual paper readback. Raster connectivity guides attention; weights and geometry are supplied, not learned.",
    });
  }
}
export class DrawingArm {
  constructor(pixels = zeros(RETINA ** 2)) {
    this.brain = new DrawingBrain();
    this.brain.see(pixels);
    this.targets = this.brain.targets;
    this.paper = new DrawingPaper();
    this.q = [-2.1, 1.2];
    this.estimate = this.q.slice();
    this.velocity = [0, 0];
    this.z = 1;
    this.ink = [];
    this.target = -1;
    this.route = [];
    this.phase = "done";
    this.ticks = 0;
    this.closed = true;
    this.penWasDown = false;
    this.finishedOnce = false;
    this.strokes = 0;
    this.brain.command(this.q, this.z, forward(this.q), 1);
    this.select();
  }
  neighbors(i) {
    const pixel = this.brain.indices[i],
      x = pixel % RETINA,
      y = Math.floor(pixel / RETINA),
      result = [];
    for (let dy = -1; dy <= 1; dy++)
      for (let dx = -1; dx <= 1; dx++) {
        if (
          (!dx && !dy) ||
          x + dx < 0 ||
          x + dx >= RETINA ||
          y + dy < 0 ||
          y + dy >= RETINA
        )
          continue;
        const j = this.brain.lookup.get((y + dy) * RETINA + x + dx);
        if (j !== undefined) result.push(j);
      }
    return result;
  }
  select() {
    const missing = this.brain.missing.state.map((v) => v > 0.1);
    // Follow the raster's connected strokes, including already inked branches.
    // This supplied attention rule sees pixels, never pointer/stroke metadata.
    if (this.target >= 0 && this.phase === "draw") {
      const queue = [this.target],
        parent = new Map([[this.target, -1]]);
      for (let k = 0; k < queue.length; k++) {
        const i = queue[k];
        if (i !== this.target && missing[i]) {
          const path = [];
          for (let j = i; j !== this.target; j = parent.get(j)) path.push(j);
          this.route = path.reverse();
          this.target = this.route.shift();
          return;
        }
        for (const j of this.neighbors(i))
          if (!parent.has(j)) {
            parent.set(j, i);
            queue.push(j);
          }
      }
    }
    const tip = forward(this.closed ? this.q : this.estimate);
    let distance = Infinity;
    this.target = -1;
    this.route = [];
    this.targets.forEach((p, i) => {
      const d = Math.hypot(p[0] - tip[0], p[1] - tip[1]);
      if (missing[i] && d < distance) {
        distance = d;
        this.target = i;
      }
    });
    this.phase = this.target < 0 ? "done" : "raise";
    if (this.target < 0 && this.targets.length) this.finishedOnce = true;
  }
  disturb() {
    this.q[0] += 0.32;
    this.q[1] -= 0.22;
    if (this.target >= 0) this.phase = "raise";
  }
  erasePatch() {
    const marked = this.targets.filter((p) => this.paper.at(p));
    if (marked.length)
      this.paper.erase(marked[Math.floor(marked.length / 2)], 0.035);
  }
  step(defer = false) {
    this.ticks++;
    const observed = this.closed ? this.q : this.estimate,
      tip = forward(observed);
    this.brain.readPaper(this.paper);
    // Read the new ink through the joint circuit before attention consumes its
    // missing-mark states. Motor action is committed only after final routing.
    const goalForPhase = () =>
      this.phase === "raise" || this.target < 0
        ? tip
        : this.targets[this.target];
    const heightForPhase = () =>
      ["draw", "lower"].includes(this.phase) ? 0 : 1;
    let command = this.brain.command(
      observed,
      this.z,
      goalForPhase(),
      heightForPhase(),
    );
    const before = `${this.target}:${this.phase}`;
    if (this.target < 0) this.select();
    else {
      const goal = this.targets[this.target],
        distance = Math.hypot(goal[0] - tip[0], goal[1] - tip[1]);
      if (this.phase === "raise" && this.z > 0.3) this.phase = "travel";
      else if (this.phase === "travel" && distance < ARRIVAL)
        this.phase = "lower";
      else if (this.phase === "lower" && !this.lifted) this.phase = "draw";
      else if (
        this.phase === "draw" &&
        distance < ARRIVAL &&
        this.brain.missing.state[this.target] < 0.1
      ) {
        if (this.route.length) this.target = this.route.shift();
        else this.select();
      }
    }
    if (before !== `${this.target}:${this.phase}`)
      command = this.brain.command(
        observed,
        this.z,
        goalForPhase(),
        heightForPhase(),
      );
    const commit = () => {
      this.velocity = this.velocity.map(
        (v, i) =>
          0.25 * v + 0.75 * Math.max(-0.075, Math.min(0.075, command[i] * 3)),
      );
      // Only motor outputs move joints/height. Ink is deposited by real contact.
      this.q = this.q.map((v, i) => v + this.velocity[i]);
      this.estimate = this.estimate.map((v, i) => v + this.velocity[i]);
      if (this.closed) this.estimate = this.q.slice();
      this.z = Math.max(0, Math.min(1, this.z + command[2] * 0.35));
      const actual = forward(this.q),
        down = !this.lifted;
      if (down) {
        const previous = this.penWasDown ? this.ink.at(-1) : actual;
        this.paper.stroke(previous, actual);
        this.ink.push([...actual, this.penWasDown]);
        if (!this.penWasDown) this.strokes++;
      }
      this.penWasDown = down;
    };
    if (defer) return commit;
    commit();
  }
  get lifted() {
    return this.z >= 0.12;
  }
  get coverage() {
    return this.targets.length
      ? this.targets.reduce((n, p) => n + this.paper.at(p), 0) /
          this.targets.length
      : 0;
  }
}
