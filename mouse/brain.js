import { Mouse as SpatialMouse } from "../showcase/embodied.js";
import { FastMemory, keys, zeros } from "../showcase/engine.js";
import {
  MotorSystem,
  MemoryReadout,
  settleTogether,
  motorFeedback,
} from "../showcase/nervous_system.js";
export class Mouse extends SpatialMouse {
  constructor(seed = 13) {
    super(seed);
    this.nerves = new MotorSystem(["Horizontal", "Vertical"]);
    this.recall = new MemoryReadout();
    this.taskMemory = new FastMemory();
    this.taskMemory.observe(keys()[0], [1, 0, 0, 0]);
    this.cue = 0;
    this.destination = 0;
    this.repair();
  }
  bindTask(memory, cue, destination) {
    this.taskMemory = memory;
    this.cue = cue;
    this.destination = destination;
    this.repair();
  }
  repair() {
    super.repair();
    if (!this.nerves) return;
    const c = this.circuit;
    this.place = {
      state: c.state.slice(),
      mask: c.mask,
      edges: c.engine.data.edges,
      names: c.state.map((_, i) => `Place ${i}`),
      groups: c.state.map(() => "place"),
    };
    this.updateBrain([0, 0]);
  }
  updateBrain(error) {
    const n = this.place.state.length,
      bridges = [
        [1, 8 + this.destination, 0, this.world.goal, 0.03 / Math.tanh(1)],
        ...error.map((v, i) => [0, this.world.goal, 2, i, v]),
        ...motorFeedback(2, 2),
      ];
    this.joint = settleTogether(
      [this.place, this.recall, this.nerves],
      [
        zeros(n),
        this.recall.configure(this.taskMemory, keys()[this.cue]),
        zeros(6),
      ],
      bridges,
      {
        steps: 3000,
        tolerance: 1e-13,
        metadata: {
          regionLabels: {
            place: "Spatial planning",
            key: "Task cue",
            record: "Task memory",
            sensor: "Position error",
            actuator: "Directional motor neurons",
          },
          adapters:
            "Task recall → spatial goal field → position error ↔ motors · one shared settlement",
          memory:
            "Lessons change FastSeams between settlements. The fixed map, active task cue and body pose define the current boundary; task, field and motor owners settle together before motion. Neighbor selection is an explicit readout.",
        },
      },
    );
    this.circuit.state = this.place.state;
    this.circuit.residual = this.joint.equilibrium.residual;
    this.circuit.steps = this.joint.steps;
    return [0, 1].map(
      (i) =>
        (Math.max(0, this.nerves.state[2 + 2 * i]) -
          Math.max(0, this.nerves.state[3 + 2 * i])) /
        2,
    );
  }
  step(dt = 0.025) {
    if (this.target < 0) this.target = this.next();
    const tx = this.target < 0 ? this.x : (this.target % this.world.cols) + 0.5,
      ty =
        this.target < 0
          ? this.y
          : Math.floor(this.target / this.world.cols) + 0.5;
    const dx = tx - this.x,
      dy = ty - this.y,
      d = Math.hypot(dx, dy),
      command = this.updateBrain([dx, dy]),
      speed = Math.hypot(...command);
    if (this.target < 0 || speed < 1e-10) return;
    const step = Math.min(dt * 3.2, d);
    const nx = this.x + (command[0] / speed) * step,
      ny = this.y + (command[1] / speed) * step;
    if (this.world.grid[Math.floor(ny) * this.world.cols + Math.floor(nx)])
      return;
    this.heading = Math.atan2(command[1], command[0]);
    this.x = nx;
    this.y = ny;
    if (Math.hypot(tx - this.x, ty - this.y) < 0.04) {
      this.x = tx;
      this.y = ty;
      this.cell = this.target;
      this.moves++;
      this.path.push(this.cell);
      this.target = -1;
      if (this.cell === this.world.goal) this.arrivals++;
    }
  }
}
