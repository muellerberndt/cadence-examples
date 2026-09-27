// Physics has its own fixed-step clock. Neural computation never stops gravity,
// momentum or contact; SettledLife independently expires stale motor authority.
import { DT } from "./body.js";

export class PhysicsClock {
  constructor({ maxWallStep = 0.25 } = {}) {
    if (!Number.isFinite(maxWallStep) || maxWallStep <= 0) throw new RangeError("invalid frame limit");
    this.maxWallStep = maxWallStep;
    this.reset();
  }

  reset() { this.remainder = 0; this.droppedWallSeconds = 0; }

  advance(life, wallSeconds, speed = 1) {
    if (!Number.isFinite(wallSeconds) || wallSeconds < 0 || !Number.isFinite(speed) || speed <= 0 || speed > 1)
      throw new RangeError("invalid physical clock input");
    const elapsed = Math.min(wallSeconds, this.maxWallStep);
    this.droppedWallSeconds += wallSeconds - elapsed;
    this.remainder += elapsed * speed;
    const steps = Math.floor((this.remainder + 1e-12) / DT);
    for (let i = 0; i < steps; i++) life.step(DT);
    this.remainder = Math.max(0, this.remainder - steps * DT);
    return steps * DT;
  }
}
