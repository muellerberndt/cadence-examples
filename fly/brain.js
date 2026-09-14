import {
  Forager as MemoryForager,
  keys,
  zeros,
  argmax,
} from "../showcase/engine.js";
import { MotorSystem, compose } from "../showcase/nervous_system.js";
import { memoryCircuit } from "../showcase/telemetry.js";
export class Forager extends MemoryForager {
  constructor(...args) {
    super(...args);
    this.nerves = new MotorSystem(["Turn", "Forward"]);
  }
  step(field, dt = 0.025) {
    this.time += dt;
    if (this.target < 0) this.choose(field);
    if (this.target < 0) {
      this.nerves.command([0, 0]);
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
    const motor = this.nerves.command([
      turn,
      0.3 * Math.max(0.2, 1 - Math.abs(turn) / Math.PI),
    ]);
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
      this.memory.observe(keys()[f.kind], target, this.updates);
      this.visits[this.target]++;
      this.encounters++;
      this.nectar += f.value;
      this.cooldown[this.target] = this.time + 4;
      this.last = { kind: f.kind, value: f.value, predicted, time: this.time };
      this.target = -1;
    }
  }
  brain(field) {
    const key = this.target >= 0 ? keys()[field[this.target].kind] : zeros(8);
    return compose([memoryCircuit(this.memory, key), this.nerves.last], {
      regionLabels: {
        key: "Flower cue",
        record: "Nectar memory",
        sensor: "Visual bearing / approach",
        actuator: "Turn / propulsion motors",
      },
      adapters:
        "Flower sensor → memory + target selection → directional motor circuit → body",
      memory:
        "Nectar contact writes associative weights. Motor potentials persist between ticks and decay; target selection and collision bounds are supplied.",
    });
  }
}
