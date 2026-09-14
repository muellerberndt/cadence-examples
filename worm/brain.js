import { WormArena as ChemicalHabitat } from "../showcase/worm_arena.js";
import {
  Circuit,
  MotorSystem,
  settleTogether,
  motorFeedback,
} from "../showcase/nervous_system.js";
export class WormArena extends ChemicalHabitat {
  constructor(data, layout = "maze") {
    super(data, layout);
    this.nerves = new MotorSystem(["East", "South", "West", "North"]);
    this.chemical = new Circuit(data.names, data.groups, data.edges, 1);
    this.chemical.mask = this.mask;
    this.joint = settleTogether(
      [this.chemical, this.nerves],
      [this.drive, Array(12).fill(0)],
    );
  }
  step(defer = false) {
    if (this.food.has(this.cell)) {
      this.food.delete(this.cell);
      this.eaten++;
      this.status = "Food consumed · seeking the next patch";
      this.repairOdor();
      this.joint = settleTogether(
        [this.chemical, this.nerves],
        [this.drive, Array(12).fill(0)],
        motorFeedback(1, 4),
        { steps: 400, tolerance: 1e-14 },
      );
      this.state = this.chemical.state;
      this.residual = this.joint.equilibrium.residual;
      return;
    }
    const x = this.cell % this.cols,
      y = Math.floor(this.cell / this.cols);
    const adjacent = [
      [x + 1, y],
      [x, y + 1],
      [x - 1, y],
      [x, y - 1],
    ].map(([a, b]) =>
      a < 0 || a >= this.cols || b < 0 || b >= this.rows
        ? -1
        : b * this.cols + a,
    );
    const sensed = adjacent.map((i) =>
        this.smell && i >= 0 && !this.walls[i] ? this.odor[i] : 0,
      ),
      peak = Math.max(...sensed);
    this.drive.fill(0);
    for (const i of this.data.stimuli.odor) this.drive[i] = peak * 2;
    const errors = sensed.map((v, i) =>
      adjacent[i] >= 0 && !this.walls[adjacent[i]] && peak > 0
        ? Math.max(0, (v - this.odor[this.cell]) / peak)
        : 0,
    );
    this.chemical.mask = this.mask;
    const bridges = [
      ...motorFeedback(1, 4),
      ...errors.flatMap((error, direction) =>
        this.motor.map((i) => [0, i, 1, direction, error / this.motor.length]),
      ),
    ];
    this.joint = settleTogether(
      [this.chemical, this.nerves],
      [this.drive, Array(12).fill(0)],
      bridges,
      { steps: 400, tolerance: 1e-14 },
    );
    this.state = this.chemical.state;
    this.residual = this.joint.equilibrium.residual;
    this.activity =
      this.motor.reduce((sum, i) => sum + this.state[i], 0) / this.motor.length;
    const command = errors.map(
      (_, i) =>
        (Math.max(0, this.nerves.state[4 + 2 * i]) -
          Math.max(0, this.nerves.state[5 + 2 * i])) /
        2,
    );
    let selected = -1;
    command.forEach((v, i) => {
      if (
        this.activity > 1e-12 &&
        errors[i] > 0 &&
        v > 0 &&
        (selected < 0 || v > command[selected])
      )
        selected = i;
    });
    if (selected < 0) {
      this.status = !this.smell
        ? "Smell off · waiting for a food cue"
        : this.food.size
          ? "No usable local food cue · open a path"
          : "All food eaten · place another patch";
      return;
    }
    const commit = () => {
      this.cell = adjacent[selected];
      this.path.push(this.cell);
      if (this.path.length > 200) this.path.shift();
      this.moves++;
      this.status = "Following local food cues";
    };
    if (defer) return commit;
    commit();
  }
  brain() {
    return Object.assign(this.joint, {
      regionLabels: {
        sensory: "Chemical sensory input",
        interneuron: "Interneurons",
        motor: "Chemical motor output",
        sensor: "Directional odor readback",
        actuator: "Body direction motors",
      },
      adapters:
        "Odor → chemical circuit → directional readback ↔ motors · one shared settlement",
      memory:
        "Public chemical graph plus 12 engineered sensor/motor owners. Directional potentials persist between body steps. No trained synapses, biological gait or digestion model.",
    });
  }
}
