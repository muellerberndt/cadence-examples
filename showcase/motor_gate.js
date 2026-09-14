// A presentation gate for an already computed settlement. Never changes its mathematics.
// The actuator closure is held until all captured iterations have been inspected.
export class MotorGate {
  constructor() {
    this.enabled = false;
    this.cancel();
  }
  get busy() {
    return this.pending !== null;
  }
  cancel() {
    this.pending = null;
    this.remaining = 0;
  }
  prepare(make, steps) {
    if (this.busy) return false;
    const commit = make();
    if (typeof commit !== "function") return false;
    if (!this.enabled) {
      commit();
      return false;
    }
    this.pending = commit;
    this.remaining = steps();
    return true;
  }
  advance(iterations) {
    if (!this.busy) return false;
    this.remaining -= iterations;
    if (this.remaining > 0) return false;
    const commit = this.pending;
    this.cancel();
    commit();
    return true;
  }
}
