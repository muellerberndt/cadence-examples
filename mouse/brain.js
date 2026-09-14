import { Mouse as SpatialMouse } from "../showcase/embodied.js";
import { MotorSystem } from "../showcase/nervous_system.js";
export class Mouse extends SpatialMouse {
  constructor(seed = 13) {
    super(seed);
    this.nerves = new MotorSystem(["Horizontal", "Vertical"]);
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
      command = this.nerves.command([dx, dy]),
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
