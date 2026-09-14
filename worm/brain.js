import { WormArena as ChemicalHabitat } from "../showcase/worm_arena.js";
import { MotorSystem, compose } from "../showcase/nervous_system.js";
export class WormArena extends ChemicalHabitat {
  constructor(data, layout = "maze") {
    super(data, layout);
    this.nerves = new MotorSystem(["East", "South", "West", "North"]);
  }
  step() {
    if (this.food.has(this.cell)) {
      this.food.delete(this.cell);
      this.eaten++;
      this.status = "Food consumed · seeking the next patch";
      this.repairOdor();
      this.nerves.command([0, 0, 0, 0]);
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
    const result = this.circuit.settle(this.drive, this.mask, 200, 1e-10);
    this.state = result.state;
    this.residual = result.residual;
    this.activity =
      this.motor.reduce((s, i) => s + this.state[i], 0) / this.motor.length;
    const errors = sensed.map((v, i) =>
      this.activity > 1e-12 &&
      adjacent[i] >= 0 &&
      !this.walls[adjacent[i]] &&
      peak > 0
        ? Math.max(0, (v - this.odor[this.cell]) / peak)
        : 0,
    );
    const command = this.nerves.command(errors);
    let selected = -1;
    command.forEach((v, i) => {
      if (errors[i] > 0 && v > 1e-10 && (selected < 0 || v > command[selected]))
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
    this.cell = adjacent[selected];
    this.path.push(this.cell);
    if (this.path.length > 200) this.path.shift();
    this.moves++;
    this.status = "Following local food cues";
  }
  brain() {
    return compose(
      [
        {
          state: this.state,
          drive: this.drive,
          mask: this.mask,
          edges: this.data.edges,
          names: this.data.names,
          groups: this.data.groups,
          recurrent: true,
          steps: 200,
        },
        this.nerves.last,
      ],
      {
        regionLabels: {
          sensory: "Chemical sensory input",
          interneuron: "Interneurons",
          motor: "Chemical motor output",
          sensor: "Directional odor readback",
          actuator: "Body direction motors",
        },
        adapters:
          "Local odor → chemical circuit + directional readback → motor population → body",
        memory:
          "Public chemical graph plus 12 engineered sensor/motor owners. Directional potentials persist between body steps. No trained synapses, biological gait or digestion model.",
      },
    );
  }
}
