import {
  Forager as MemoryForager,
  keys,
  zeros,
  argmax,
} from "../shared/engine.js";
import {
  MotorSystem,
  MemoryReadout,
  settleTogether,
  motorFeedback,
} from "../shared/nervous_system.js";
export class Forager extends MemoryForager {
  constructor(memory, offset = 0, updates = 1, { revisit = true } = {}) {
    super(memory, offset, updates);
    this.nerves = new MotorSystem(["Turn", "Forward"]);
    this.recall = new MemoryReadout();
    this.revisit = revisit;
    this.lastVisit = Array(8).fill(-Infinity);
  }
  choose(field) {
    if (!this.revisit) return super.choose(field);
    let best = -Infinity;
    this.target = -1;
    field.forEach((f, i) => {
      if (this.cooldown[i] > this.time) return;
      // A supplied exploration rule: old observations become worth checking again.
      // Only elapsed time, position and learned expectations enter selection.
      // The field's current nectar value is revealed exclusively on contact.
      const age = this.time - this.lastVisit[i];
      const uncertainty = 1 - Math.exp(-age / 15);
      const value = this.visits[i]
        ? argmax(this.memory.predict(keys()[f.kind])) / 3
        : 0;
      const score = value + 1.5 * uncertainty
        - 0.25 * Math.hypot(f.x - this.x, f.y - this.y);
      if (score > best) {
        best = score;
        this.target = i;
      }
    });
  }
  step(field, dt = 0.025, defer = false) {
    this.time += dt;
    if (this.target < 0) this.choose(field);
    if (this.target < 0) {
      this.joint = settleTogether(
        [this.recall, this.nerves],
        [this.recall.configure(this.memory, zeros(8)), zeros(6)],
        motorFeedback(1, 2),
      );
      return;
    }
    const f = field[this.target],
      dx = f.x - this.x,
      dy = f.y - this.y,
      desired = Math.atan2(dy, dx),
      turn = Math.atan2(
        Math.sin(desired - this.angle),
        Math.cos(desired - this.angle),
      );
    const recallDrive = this.recall.configure(this.memory, keys()[f.kind]);
    const synapses = [
      ...motorFeedback(1, 2),
      ...Array.from({ length: 4 }, (_, i) => [0, 8 + i, 1, 1, 0.04 * i]),
    ];
    this.joint = settleTogether(
      [this.recall, this.nerves],
      [
        recallDrive,
        [turn, 0.3 * Math.max(0.2, 1 - Math.abs(turn) / Math.PI), 0, 0, 0, 0],
      ],
      synapses,
      { dt: 0.65 },
    );
    const motor = [0, 1].map(
      (i) =>
        (Math.max(0, this.nerves.state[2 + 2 * i]) -
          Math.max(0, this.nerves.state[3 + 2 * i])) /
        2,
    );
    const commit = () => {
      this.angle += Math.max(-3.5 * dt, Math.min(3.5 * dt, motor[0] * 12 * dt));
      const speed = Math.max(0, motor[1]) * 0.5;
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
        this.memory.observe(keys()[f.kind], target, this.updates, {
          salience: Math.abs(f.value - predicted),
        });
        this.visits[this.target]++;
        this.lastVisit[this.target] = this.time;
        this.encounters++;
        this.nectar += f.value;
        this.cooldown[this.target] = this.time + 4;
        this.last = {
          kind: f.kind,
          value: f.value,
          predicted,
          time: this.time,
        };
        this.target = -1;
      }
    };
    if (defer) return commit;
    commit();
  }
  brain(field) {
    if (!this.joint)
      this.joint = settleTogether(
        [this.recall, this.nerves],
        [this.recall.configure(this.memory, zeros(8)), zeros(6)],
        motorFeedback(1, 2),
      );
    return Object.assign(this.joint, {
      regionLabels: {
        key: "Flower cue",
        record: "Nectar memory",
        sensor: "Visual bearing / approach",
        actuator: "Turn / propulsion motors",
      },
      adapters:
        "Cue → nectar recall → approach ↔ motor feedback · one shared equilibrium",
      memory:
        "Nectar contact writes associative weights. Motor potentials persist between ticks and decay. A supplied exploration rule revisits aging observations using eight contact timestamps; collision bounds and target selection are supplied.",
    });
  }
}
