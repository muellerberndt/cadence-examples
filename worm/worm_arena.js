// A supplied habitat and sensory/motor adapter around the public worm circuit.
// The animal reads only adjacent odor samples. No route or food coordinates enter its controller.
import { Worm, zeros } from "../shared/engine.js";

export class WormArena {
  constructor(data, layout = "maze") {
    this.data = data;
    this.circuit = new Worm(data);
    this.cols = 31;
    this.rows = 21;
    this.motor = data.groups.flatMap((g, i) => (g === "motor" ? [i] : []));
    this.mask = Array(data.names.length).fill(1);
    this.smell = true;
    this.reset(layout);
  }
  neighbors(i) {
    const x = i % this.cols,
      y = Math.floor(i / this.cols);
    return [
      [x + 1, y],
      [x, y + 1],
      [x - 1, y],
      [x, y - 1],
    ]
      .filter(([a, b]) => a >= 0 && a < this.cols && b >= 0 && b < this.rows)
      .map(([a, b]) => b * this.cols + a);
  }
  reset(layout = "maze") {
    this.walls = Array(this.cols * this.rows).fill(false);
    this.food = new Set();
    this.cell = 10 * this.cols + 4;
    this.path = [this.cell];
    this.eaten = 0;
    this.moves = 0;
    this.state = zeros(this.circuit.n);
    this.drive = zeros(this.circuit.n);
    this.activity = 0;
    this.residual = 0;
    this.status = "Following local food cues";
    if (layout === "maze") {
      for (let y = 4; y < 17; y++) this.walls[y * this.cols + 12] = true;
      for (let y = 0; y < 12; y++) this.walls[y * this.cols + 22] = true;
      this.food.add(10 * this.cols + 27);
      this.food.add(4 * this.cols + 18);
    } else {
      this.food.add(10 * this.cols + 25);
    }
    this.repairOdor();
  }
  repairOdor() {
    // Leaky diffusion to equilibrium. Walls have no-flux boundaries; food cells are clamped.
    let field = zeros(this.walls.length);
    this.food.forEach((i) => (field[i] = 1));
    const adjacent = field.map((_, i) =>
      this.neighbors(i).filter((j) => !this.walls[j]),
    );
    for (let k = 0; k < 4000; k++) {
      let change = 0;
      const next = field.map((v, i) => {
        if (this.walls[i]) return 0;
        if (this.food.has(i)) return 1;
        const ns = adjacent[i];
        const value = ns.length
          ? (0.985 * ns.reduce((s, j) => s + field[j], 0)) / ns.length
          : 0;
        change = Math.max(change, Math.abs(value - v));
        return value;
      });
      field = next;
      if (change < 1e-10) break;
    }
    this.odor = field;
  }
  edit(i, tool) {
    if (!Number.isInteger(i) || i < 0 || i >= this.walls.length) return false;
    // Protect the moving body, including its displayed tail, from painting through it.
    if (tool === "wall" && this.path.slice(-7).includes(i)) return false;
    if (tool === "wall") {
      this.walls[i] = true;
      this.food.delete(i);
    } else if (tool === "food") {
      this.walls[i] = false;
      this.food.add(i);
    } else if (tool === "erase") {
      this.walls[i] = false;
      this.food.delete(i);
    } else return false;
    return true;
  }
  step() {
    if (this.food.has(this.cell)) {
      this.food.delete(this.cell);
      this.eaten++;
      this.status = "Food consumed · seeking the next patch";
      this.repairOdor();
      return;
    }
    const near = this.neighbors(this.cell).filter((i) => !this.walls[i]);
    const next = near.reduce(
      (best, i) => (best < 0 || this.odor[i] > this.odor[best] ? i : best),
      -1,
    );
    const sensed = this.smell && next >= 0 ? this.odor[next] : 0;
    this.drive.fill(0);
    for (const i of this.data.stimuli.odor) this.drive[i] = sensed * 2;
    const result = this.circuit.settle(this.drive, this.mask, 200, 1e-10);
    this.state = result.state;
    this.residual = result.residual;
    this.activity =
      this.motor.reduce((s, i) => s + this.state[i], 0) / this.motor.length;
    if (
      this.activity < 1e-12 ||
      next < 0 ||
      this.odor[next] <= this.odor[this.cell]
    ) {
      this.status = !this.smell
        ? "Smell off · waiting for a food cue"
        : this.food.size
          ? "No usable local food cue · open a path"
          : "All food eaten · place another patch";
      return;
    }
    // Heading follows the sampled gradient; circuit motor activity gates the supplied body step.
    this.cell = next;
    this.path.push(next);
    if (this.path.length > 200) this.path.shift();
    this.moves++;
    this.status = "Following local food cues";
  }
  snapshot() {
    return {
      cell: this.cell,
      cols: this.cols,
      rows: this.rows,
      walls: this.walls.slice(),
      food: [...this.food],
      eaten: this.eaten,
      moves: this.moves,
      smell: this.smell,
      activity: this.activity,
      residual: this.residual,
      status: this.status,
    };
  }
}
