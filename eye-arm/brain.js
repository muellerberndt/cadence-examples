import { Circuit, settleTogether, zeros } from "../showcase/nervous_system.js";
import { forward, jacobian } from "../showcase/embodied.js";
export const RETINA = 24;
export class DrawingBrain {
  constructor() {
    this.eye = new Circuit(
      Array.from(
        { length: RETINA ** 2 },
        (_, i) => `Retinal sample ${i % RETINA},${Math.floor(i / RETINA)}`,
      ),
      Array(RETINA ** 2).fill("retina"),
      [],
    );
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
    this.pixels = zeros(RETINA ** 2);
    this.targets = [];
    this.eye.settle(this.pixels);
    this.motor.settle(zeros(17));
  }
  see(darkness) {
    if (
      darkness.length !== RETINA ** 2 ||
      darkness.some((v) => !Number.isFinite(v) || v < 0 || v > 1)
    )
      throw Error("Invalid retina");
    this.pixels = darkness.slice();
    // Pixel intensities drive actual sensory states; the readout thresholds those states.
    this.eye.settle(
      darkness.map((v) => v * 2),
      40,
    );
    this.targets = this.eye.state.flatMap((v, i) =>
      v > 0.23
        ? [
            [
              0.26 + (((i % RETINA) + 0.5) / RETINA) * 0.48,
              0.2 + ((Math.floor(i / RETINA) + 0.5) / RETINA) * 0.48,
            ],
          ]
        : [],
    );
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
    // Geometry/attention selects these ports; their signal still travels as a seam.
    const col = Math.round(((target[0] - 0.26) / 0.48) * RETINA - 0.5),
      row = Math.round(((target[1] - 0.2) / 0.48) * RETINA - 0.5),
      pixel = row * RETINA + col,
      visible =
        col >= 0 &&
        col < RETINA &&
        row >= 0 &&
        row < RETINA &&
        this.pixels[pixel] > 0.117;
    const bridges = [];
    if (visible)
      for (let axis = 0; axis < 2; axis++) {
        bridges.push([
          0,
          pixel,
          1,
          axis,
          drive[axis] / Math.tanh(2 * this.pixels[pixel]),
        ]);
        drive[axis] = 0;
      }
    for (const [positive, negative, receiver] of [
      [11, 12, 9],
      [13, 14, 10],
      [15, 16, 8],
    ])
      bridges.push(
        [1, positive, 1, receiver, -0.1],
        [1, negative, 1, receiver, 0.1],
      );
    this.joint = settleTogether(
      [this.eye, this.motor],
      [this.pixels.map((v) => v * 2), drive],
      bridges,
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
      regionLabels: {
        retina: "Retina · pixel intensity",
        readback: "Target & proprioception",
        error: "Visual / height error",
        premotor: "Joint coordination",
        actuator: "Motor neurons",
      },
      adapters:
        "Retina → target ↔ visual error ↔ joint coordination ↔ motors · one shared settlement",
      memory:
        "Retained graded potentials carry transient state between control ticks. Visited targets are explicit attention records; weights and geometry are supplied, not learned.",
    });
  }
}
export class DrawingArm {
  constructor(pixels = zeros(RETINA ** 2)) {
    this.brain = new DrawingBrain();
    this.brain.see(pixels);
    this.targets = this.brain.targets;
    this.q = [-2.1, 1.2];
    this.estimate = this.q.slice();
    this.velocity = [0, 0];
    this.z = 1;
    this.ink = [];
    this.covered = new Set();
    this.attempted = new Set();
    this.target = -1;
    this.ticks = 0;
    this.closed = true;
    this.penWasDown = false;
    this.select();
    this.brain.command(
      this.q,
      this.z,
      this.targets[this.target] ?? forward(this.q),
      1,
    );
  }
  select() {
    const tip = forward(this.closed ? this.q : this.estimate);
    let distance = Infinity;
    this.target = -1;
    this.targets.forEach((p, i) => {
      const d = Math.hypot(p[0] - tip[0], p[1] - tip[1]);
      if (!this.attempted.has(i) && d < distance) {
        distance = d;
        this.target = i;
      }
    });
  }
  disturb() {
    this.q[0] += 0.32;
    this.q[1] -= 0.22;
  }
  step() {
    this.ticks++;
    const observed = this.closed ? this.q : this.estimate,
      tip = forward(observed),
      goal = this.targets[this.target] ?? tip;
    const distance = Math.hypot(goal[0] - tip[0], goal[1] - tip[1]);
    const requestedHeight = this.target < 0 || distance > 0.026 ? 1 : 0;
    const command = this.brain.command(observed, this.z, goal, requestedHeight);
    this.velocity = this.velocity.map(
      (v, i) =>
        0.25 * v + 0.75 * Math.max(-0.075, Math.min(0.075, command[i] * 1.0)),
    );
    // Only motor population outputs move the joints. Contact alone creates ink.
    this.q = this.q.map((v, i) => v + this.velocity[i]);
    this.estimate = this.estimate.map((v, i) => v + this.velocity[i]);
    if (this.closed) this.estimate = this.q.slice();
    this.z = Math.max(0, Math.min(1, this.z + command[2] * 0.35));
    const actual = forward(this.q),
      down = this.z < 0.12;
    if (down) {
      this.ink.push([...actual, this.penWasDown]);
      this.targets.forEach((p, i) => {
        if (Math.hypot(p[0] - actual[0], p[1] - actual[1]) < 0.024)
          this.covered.add(i);
      });
    }
    this.penWasDown = down;
    if (this.target >= 0 && distance < 0.01 && down) {
      this.attempted.add(this.target);
      this.select();
    }
  }
  get lifted() {
    return this.z >= 0.12;
  }
  get coverage() {
    return this.targets.length ? this.covered.size / this.targets.length : 0;
  }
}
