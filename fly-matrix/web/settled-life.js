// Strict actuator authority: environmental sensors -> settled worker readouts -> muscles.
// This runtime has no navigation, target tracking, behavioral timers or fallback pilot.
import { RADIUS, A_MAX, BETA_MAX, S_MAX, F_MAX } from "./body.js";
import { haltereTone, ocelliLR, opticFlowDrive, antennaDrive } from "./senses.js";
import { settledMotor, SETTLED_MOTOR_GROUPS } from "./motor.js";

export const CONTROL_PERIOD = 0.1;
export const COMMAND_TTL = 0.12;
export const RESIDUAL_TOLERANCE = 1e-6;
const WING_KEYS = ["aL", "aR", "betaL", "betaR", "sL", "sR", "f"];
const WING_LIMITS = [[0, A_MAX], [0, A_MAX], [-BETA_MAX, BETA_MAX], [-BETA_MAX, BETA_MAX], [-S_MAX, S_MAX], [-S_MAX, S_MAX], [0, F_MAX]];
const zeroWings = () => Object.fromEntries(WING_KEYS.map((key) => [key, 0]));
const finite = (value) => typeof value === "number" && Number.isFinite(value);
const integer = (value) => Number.isSafeInteger(value) && value >= 0;
const bounded = (value) => finite(value) && value >= 0 && value <= 1;

export class SettledLife {
  constructor(flight, room, seedIgnored = 1, options = {}) {
    // The legacy constructor's seed position is accepted, but no random policy runs here.
    if (seedIgnored && typeof seedIgnored === "object") options = seedIgnored;
    this.flight = flight;
    this.room = room;
    this.generation = options.generation ?? 0;
    this.commandTTL = options.commandTTL ?? COMMAND_TTL;
    this.tolerance = options.tolerance ?? RESIDUAL_TOLERANCE;
    if (!integer(this.generation) || !finite(this.commandTTL) || this.commandTTL <= 0 || !finite(this.tolerance) || this.tolerance <= 0) throw new RangeError("invalid control contract");
    this.clock = 0;
    this.since = 0;
    this.controls = zeroWings();
    this.proboscis = 0;
    this.wingsOff = true;
    this.source = "patchnet";
    this.readouts = {};
    this.events = [];
    this.pending = null;
    this.lastRequestId = -1;
    this.commandDeadline = -Infinity;
    this.commandStatus = { state: "unavailable", reason: "no settled command", generation: this.generation };
    this.unsupported = { walking: true, grooming: true, legForces: true };
    this.hunger = 0.6;
    this.fruits = {};
    for (const [name, fruit] of Object.entries(room.fruits || {})) this.fruits[name] = { pos: fruit.pos, odour: name === "banana" ? "decaying_fruit" : name === "bread" ? "yeasty" : fruit.odour };
    this.sugar = this.fruits.banana ? "banana" : null;
    this.smelled = null;
    this.smellC = 0;
    this.smellDrive = 0;
    this.lastP = { banana: null, bread: null };
    this.visits = { banana: 0, bread: 0, sweet: 0, punished: 0 };
    this.ethogram = { saccades: 0, microSaccades: 0, bouts: [], sits: [], grooms: 0, feeds: 0, landings: 0, flying_s: 0, sitting_s: 0, decisions: 0 };
    this.pendingReward = null; // No action is invented retrospectively to receive a reward.
    this.lastHandAngle = null;
    this.loom = 0;
    this.handDist = Infinity;
    this._lastContactFruit = null;
    this._feeding = false;
    this._updateContact();
    this.mode = this.contact ? "landed" : "flying"; // Observed body state, not an action selector.
  }

  log(text) { this.events.push([this.clock, text]); if (this.events.length > 40) this.events.shift(); }

  /** Register the actual outgoing solve contract before sending it to the worker. */
  expectControl(request, generation = this.generation, options = {}) {
    const spec = typeof request === "object" && request !== null ? request : { requestId: request, generation, ...options };
    const id = spec.requestId, gen = spec.generation ?? this.generation;
    const tolerance = spec.tolerance ?? this.tolerance, steps = spec.steps ?? 1024;
    if (!integer(id) || gen !== this.generation || id <= this.lastRequestId || !finite(tolerance) || tolerance <= 0 || tolerance > this.tolerance || !integer(steps) || steps > 1024) {
      this.revokeControl("invalid request contract");
      return false;
    }
    this.lastRequestId = id;
    this.pending = { requestId: id, generation: gen, tolerance, steps, deadline: this.clock + this.commandTTL };
    this.commandStatus = { state: "awaiting", reason: "waiting for settled command", requestId: id, generation: gen };
    return true;
  }

  /** Clear authority on errors, mode changes or resets. A new generation rejects old replies. */
  revokeControl(reason = "control revoked", generation = this.generation) {
    if (!integer(generation) || generation < this.generation) throw new RangeError("control generation cannot move backwards");
    if (generation !== this.generation) { this.generation = generation; this.lastRequestId = -1; }
    this.pending = null;
    this.commandDeadline = -Infinity;
    this.controls = zeroWings();
    this.proboscis = 0;
    this.wingsOff = true;
    this.readouts = {};
    this.commandStatus = { state: "unavailable", reason, generation: this.generation };
  }

  applyControl(message) {
    const pending = this.pending;
    const reject = (reason) => { this.revokeControl(reason); return false; };
    if (!message || message.type !== "control" || message.kind !== "control") return reject("invalid command envelope");
    if (!pending || message.generation !== this.generation || message.generation !== pending.generation || message.requestId !== pending.requestId) return reject("stale or unrequested command");
    if (this.clock >= pending.deadline) return reject("command arrived after its deadline");
    if (message.converged !== true) return reject("controller did not converge");
    if (!finite(message.residual) || message.residual < 0 || !finite(message.tolerance) || message.tolerance <= 0 || message.tolerance > pending.tolerance || message.residual > message.tolerance || !integer(message.iterations) || message.iterations > pending.steps) return reject("invalid convergence evidence");
    const readouts = message.readouts;
    if (!readouts || typeof readouts !== "object" || Array.isArray(readouts) || !Object.values(readouts).every(finite) || !SETTLED_MOTOR_GROUPS.every((key) => Object.hasOwn(readouts, key) && finite(readouts[key]))) return reject("incomplete or nonfinite motor readouts");
    let motors;
    try { motors = settledMotor(readouts); } catch { return reject("motor decoding failed"); }
    if (!motors || !Array.isArray(motors.wings) || motors.wings.length !== 7 || !motors.wings.every((value, i) => finite(value) && value >= WING_LIMITS[i][0] && value <= WING_LIMITS[i][1]) || !bounded(motors.proboscis) || !motors.legs || !bounded(motors.legs.ttm_left) || !bounded(motors.legs.ttm_right)) return reject("invalid actuator shape or range");
    this.controls = Object.fromEntries(WING_KEYS.map((key, i) => [key, motors.wings[i]]));
    this.proboscis = motors.proboscis;
    this.wingsOff = this.controls.f <= 0 || this.controls.aL + this.controls.aR <= 0;
    this.readouts = { ...readouts };
    this.commandDeadline = pending.deadline; // A late reply does not buy a new validity interval.
    this.pending = null;
    this.commandStatus = { state: "active", reason: "settled motor command", generation: message.generation, requestId: message.requestId, residual: message.residual, iterations: message.iterations, unsupportedLegDrive: motors.legs.ttm_left > 0 || motors.legs.ttm_right > 0 };
    this.ethogram.decisions++;
    return true;
  }

  _updateContact() {
    const f = this.flight;
    this.contact = finite(f.touching) && f.touching > 0;
    this.surfaceContact = this.contact && f.onFloor === true;
    this.onWhich = null;
    if (this.surfaceContact) {
      for (const [name, fruit] of Object.entries(this.fruits)) {
        const [x, y, z] = fruit.pos;
        // Taste requires an actual contact and a foot at the fruit surface, not proximity in air.
        if (Math.hypot(f.p[0] - x, f.p[1] - y) < 0.06 && Math.abs(f.p[2] - RADIUS - z) < 0.003) { this.onWhich = name; break; }
      }
    }
    this.onFruit = this.onWhich !== null;
    this.onSugar = this.onFruit && this.onWhich === this.sugar;
  }

  setSugar(name) {
    if (name !== null && !Object.hasOwn(this.fruits, name)) throw new RangeError("unknown sugar surface");
    this.sugar = name;
    this._updateContact();
    this.log(name ? `sugar on the ${name}` : "no sugar anywhere");
  }

  fruitMoved(name) {
    if (!Object.hasOwn(this.fruits, name)) return false;
    this._updateContact();
    this.log(`the ${name} is moved`);
    // Surface contact and gravity, not a scripted takeoff, determine the body's response.
    return false;
  }

  poke(dv = [0, 0, 0], gust = [0, 0, 0], spin = [0, 0, 0]) {
    if (![dv, gust, spin].every((v) => Array.isArray(v) && v.length === 3 && v.every(finite))) throw new RangeError("invalid environmental impulse");
    this.flight.swat(dv, gust, spin);
    this.visits.punished++;
    this.log("external impulse");
  }

  /** Environmental concentration at the two receptors; this never writes motor commands. */
  odour(source) {
    const f = this.flight, [x, y, z] = f.p, R = f.rotation();
    const at = (px, py, pz) => { const d2 = (px - source[0]) ** 2 + (py - source[1]) ** 2 + (pz - source[2]) ** 2; return 0.85 * Math.exp(-d2 / 0.02) + 0.15 * Math.exp(-d2 / 0.5); };
    const c = at(x, y, z), left = at(x + 0.05 * R[1], y + 0.05 * R[4], z + 0.05 * R[7]), right = at(x - 0.05 * R[1], y - 0.05 * R[4], z - 0.05 * R[7]);
    const ratio = Math.log(left + 1e-9) - Math.log(right + 1e-9), gain = 0.5 + this.hunger;
    return { left: Math.min(1, gain * c * (1 + 1.5 * Math.max(0, ratio))), right: Math.min(1, gain * c * (1 + 1.5 * Math.max(0, -ratio))), c };
  }

  senses(handPos = null, dt = 0) {
    this._updateContact();
    const f = this.flight, [hl, hr] = haltereTone(!this.wingsOff), [ol, or_] = ocelliLR(f.rotation());
    if (handPos !== null) {
      if (!Array.isArray(handPos) || handPos.length !== 3 || !handPos.every(finite) || !finite(dt) || dt < 0) throw new RangeError("invalid sensory observation");
      this.handDist = Math.hypot(...handPos.map((v, i) => v - f.p[i]));
      const angle = 2 * Math.atan(0.025 / Math.max(this.handDist, 0.005));
      this.loom = this.lastHandAngle !== null && dt > 0 ? Math.min(1, Math.max(0, (angle - this.lastHandAngle) / (4 * dt))) : 0;
      this.lastHandAngle = angle;
    } else { this.lastHandAngle = null; this.loom = 0; this.handDist = Infinity; }
    const senses = {
      "haltere:left": hl, "haltere:right": hr, "ocelli:left": ol, "ocelli:right": or_,
      "vis:LC4": this.loom, "vis:LPLC2": this.loom,
      "grn:sugar:labellum": this.onSugar ? 1 : 0, "grn:sugar:front_leg": this.onSugar ? 1 : 0,
      "leg_touch": this.contact ? 0.6 : 0,
    };
    for (const side of ["left", "right"]) {
      const flow = opticFlowDrive(f.w[2], f.w[0], f.w[1], side);
      senses[`lptc:hs:${side}`] = flow; senses[`lptc:vs:${side}`] = flow;
      senses[`jo:C:${side}`] = antennaDrive(f.speed()); senses[`jo:E:${side}`] = antennaDrive(f.speed());
    }
    this.smelled = null; this.smellC = 0; this.smellDrive = 0;
    for (const [name, fruit] of Object.entries(this.fruits)) {
      if (!fruit.odour) continue;
      const d = this.odour(fruit.pos);
      for (const side of ["left", "right"]) senses[`orn:${fruit.odour}:${side}`] = Math.max(senses[`orn:${fruit.odour}:${side}`] || 0, d[side]);
      if (d.c > this.smellC) { this.smellC = d.c; this.smelled = d.c >= 0.04 ? name : null; this.smellDrive = Math.max(d.left, d.right); }
    }
    return senses;
  }

  step(dt) {
    if (!finite(dt) || dt <= 0) throw new RangeError("invalid body timestep");
    this.clock += dt;
    this.since += dt;
    if (finite(this.commandDeadline) && this.clock >= this.commandDeadline) {
      // The old motors expire independently of a newer solve. Preserve that solve's
      // original deadline; receiving its reply must never extend its observation's life.
      const pending = this.pending;
      this.revokeControl("command expired");
      if (pending && this.clock < pending.deadline) {
        this.pending = pending;
        this.commandStatus = { state: "awaiting", reason: "previous command expired; waiting for settled command", requestId: pending.requestId, generation: pending.generation };
      }
    }
    if (this.pending && this.clock >= this.pending.deadline) this.revokeControl("controller reply expired");
    this.flight.step(dt, this.controls);
    this._updateContact();
    const feeding = this.onSugar && this.proboscis > 0 && this.clock < this.commandDeadline;
    if (this.onWhich && this.onWhich !== this._lastContactFruit) this.visits[this.onWhich] = (this.visits[this.onWhich] || 0) + 1;
    if (feeding && !this._feeding) { this.ethogram.feeds++; this.visits.sweet++; }
    if (this.contact && this.mode === "flying") this.ethogram.landings++;
    this.mode = feeding ? "feeding" : this.contact ? "landed" : "flying";
    this.hunger = Math.max(0, Math.min(1, this.hunger + dt * (feeding ? -0.12 * this.proboscis : 1 / 60)));
    if (this.contact) this.ethogram.sitting_s += dt; else this.ethogram.flying_s += dt;
    this._lastContactFruit = this.onWhich;
    this._feeding = feeding;
    if (this.since >= CONTROL_PERIOD) { this.since %= CONTROL_PERIOD; return true; }
    return false;
  }

  pose() { return { neural: true, mode: this.mode, t: this.clock, groom: null, feed: this.proboscis, feeding: this._feeding, hunger: this.hunger }; }
}
