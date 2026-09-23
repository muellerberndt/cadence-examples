// The sill, the readings, the task cost and the habit, in the browser: the same rules as the
// experiment's cat.py, in the same order of operations, so a decision made here is the
// decision the Python brain makes. The unit box; a paw moved by a two-dimensional push per
// decision; a laser dot that is the pointer (source "mouse") or a drawn event (source
// "schedule": a walk, a curve or a jitter, reflecting at the walls, with the changed laws for
// the driven check). The brain reads a six-by-six retina of Gaussian bumps over the dot, the
// fovea's readout of it (centroid, optic flow, mass) and its own paw; it never reads the dot's
// law. Nothing here is random in mouse mode; the schedule mode draws from the page's own
// generator (rng.js), so its dots differ from the experiment's, which come from numpy's.

import { Rng } from "./rng.js";
export { Rng };

export const GRID = 6;
export const RETINA = GRID * GRID;
export const FOVEA = RETINA;
export const FLOW = RETINA + 2;
export const MASS = RETINA + 4;
export const PAW = RETINA + 5;
export const READINGS = RETINA + 7;
export const PREDICTED = [FOVEA, FOVEA + 1, MASS, PAW, PAW + 1];
export const OUTPUTS = PREDICTED.length;
export const ACTIONS = 2;
export const PAW_SPEED = 0.05;
export const CATCH_RADIUS = 0.08;
export const CATCH_HOLD = 3;
export const EFFORT = 0.02;
export const FASTER = 2.67;
export const GRAVITY = 0.008;
export const VMAX = 0.045;
export const CLIP = 4.0;
export const MODES = { habit: 0, imagine: 1, learn: 2 };
export const LAWS = ["gravity", "faster", "wrap"];

export const SCHEDULE = {
  decisions: 4000, dot_gap: [40, 140], dot_timeout: 150, dot_speed: 0.015, turn: 0.35,
  kinds: { walk: 0.5, curve: 0.3, jitter: 0.2 }, jitter_length: [8, 20], change_at: 2400, change: "gravity",
};

// The constants the experiment computes in numpy are taken from the export so that the
// floats are the same ones (the eight directions at pi/4, the retina's grid and width).
export const CONST = {
  grid: Array.from({ length: GRID }, (_, i) => (i + 0.5) / GRID),
  sigma: 0.7 / GRID,
  directions: [[0, 0], [1, 0], [Math.SQRT1_2, Math.SQRT1_2], [0, 1], [-Math.SQRT1_2, Math.SQRT1_2], [-1, 0], [-Math.SQRT1_2, -Math.SQRT1_2], [0, -1], [Math.SQRT1_2, -Math.SQRT1_2]],
};
export function setConstants(world) {
  if (world.retina_grid) CONST.grid = world.retina_grid.slice();
  if (world.retina_sigma !== undefined) CONST.sigma = world.retina_sigma;
  if (world.directions) CONST.directions = world.directions.map((d) => d.slice());
}

export const clip = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
const norm2 = (x, y) => Math.sqrt(x * x + y * y);

/** Gaussian bumps on the grid, peak one, for one position: row-major (y row, x column). */
export function retina(position, out = new Float64Array(RETINA)) {
  const g = CONST.grid, s = CONST.sigma;
  for (let j = 0; j < GRID; j++) {
    const dy = position[1] - g[j], by = Math.exp(-0.5 * (dy / s) ** 2);
    for (let i = 0; i < GRID; i++) {
      const dx = position[0] - g[i];
      out[j * GRID + i] = by * Math.exp(-0.5 * (dx / s) ** 2);
    }
  }
  return out;
}

/** The readings: the retina over the dot, the fovea readout (the dot's centroid centred on
 *  the sill, its flow in units of the routine speed, its mass) and the paw's position, centred. */
export function readingOf(dot, flow, paw) {
  const out = new Float64Array(READINGS);
  if (dot) {
    retina(dot, out.subarray(0, RETINA));
    out[FOVEA] = 2.0 * dot[0] - 1.0;
    out[FOVEA + 1] = 2.0 * dot[1] - 1.0;
    if (flow) {
      out[FLOW] = clip(flow[0] / SCHEDULE.dot_speed, -4.0, 4.0);
      out[FLOW + 1] = clip(flow[1] / SCHEDULE.dot_speed, -4.0, 4.0);
    }
    out[MASS] = 1.0;
  }
  out[PAW] = 2.0 * paw[0] - 1.0;
  out[PAW + 1] = 2.0 * paw[1] - 1.0;
  return out;
}

export const pawOf = (r) => [clip((r[PAW] + 1.0) / 2.0, 0.0, 1.0), clip((r[PAW + 1] + 1.0) / 2.0, 0.0, 1.0)];
export const dotOf = (r) => [clip((r[FOVEA] + 1.0) / 2.0, 0.0, 1.0), clip((r[FOVEA + 1] + 1.0) / 2.0, 0.0, 1.0)];

/** What the sill is for: the paw on the dot when there is one (the mass says how much of one),
 *  with a small price on the push. */
export function taskCost(r, a) {
  const mass = clip(r[MASS], 0.0, 1.0);
  const p = pawOf(r), d = dotOf(r);
  return mass * ((p[0] - d[0]) ** 2 + (p[1] - d[1]) ** 2) + EFFORT * (a[0] * a[0] + a[1] * a[1]);
}

/** The habit: the paw drifts toward home; its rest position and drift gain are genes. */
export function habitAct(h, r) {
  const paw = pawOf(r);
  return [clip(h.drift * (h.home_x - paw[0]) / PAW_SPEED * 0.1, -1.0, 1.0), clip(h.drift * (h.home_y - paw[1]) / PAW_SPEED * 0.1, -1.0, 1.0)];
}
export const HABIT_KEYS = ["home_x", "home_y", "drift"];

/** The raw change of the predicted readings (centroid, mass, paw) between two moments. */
export function targetOf(r, r2, out = new Float64Array(OUTPUTS)) {
  for (let k = 0; k < OUTPUTS; k++) out[k] = r2[PREDICTED[k]] - r[PREDICTED[k]];
  return out;
}

// ------------------------------------------------------------------ the schedule
/** The dots of a life, drawn once per seed: for each, the gap after the previous dot's
 *  vanishing, its kind, its start, its heading, its curvature and (for a jitter) its length. */
export function drawEvents(rng, schedule) {
  const names = Object.keys(schedule.kinds);
  const total = names.reduce((s, k) => s + schedule.kinds[k], 0);
  const dots = [];
  for (let n = 0; n < Math.floor(schedule.decisions / 20) + 4; n++) {
    let u = rng.random() * total, kind = names[names.length - 1];
    for (const k of names) { if (u < schedule.kinds[k]) { kind = k; break; } u -= schedule.kinds[k]; }
    dots.push({
      gap: rng.integers(schedule.dot_gap[0], schedule.dot_gap[1]),
      kind,
      start: [rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)],
      heading: rng.uniform(0.0, 2 * Math.PI),
      curve: (rng.random() < 0.5 ? -1.0 : 1.0) * rng.uniform(0.05, 0.15),
      length: rng.integers(schedule.jitter_length[0], schedule.jitter_length[1]),
    });
  }
  return dots;
}

/** The sill. `step(a)` executes one push of the paw and returns the next reading and what
 *  happened at that decision. `source` is "schedule" (the drawn dots) or "mouse" (the page's
 *  pointer is the dot). */
export class World {
  constructor(schedule, events, rng, source = "schedule") {
    this.schedule = schedule; this.events = events.slice(); this.rng = rng; this.source = source;
    this.paw = [0.5, 0.5];
    this.dot = null;
    this.previous = null; // the dot a decision ago, for the flow
    this.vel = [0.0, 0.0];
    this.kind = "walk"; this.curve = 0.0; this.length = 0;
    this.t = 0; this.hold = 0; this.age = 0;
    this.wait = this.events.length ? this.events[0].gap : 1e9;
    this.index = 0;
    this.law = "routine"; this.changed = false;
    this.mouse = null; this.mouseCaught = null;
    this.catches = 0; this.misses = 0; this.dots = 0;
    this.pending = [];
    this.scriptLeft = 0; // the demo: scheduled dots run for this many decisions in mouse mode
  }
  get present() { return this.dot !== null; }
  get speed() { return this.schedule.dot_speed * (this.law === "faster" ? FASTER : 1.0); }
  get flow() { return this.dot === null || this.previous === null ? null : [this.dot[0] - this.previous[0], this.dot[1] - this.previous[1]]; }
  reading() { return readingOf(this.dot, this.flow, this.paw); }
  fire(event) { this.pending.push(event); }
  _spawn(event) {
    this.dot = [event.start[0], event.start[1]];
    this.previous = null;
    const h = event.heading, s = this.speed;
    this.vel = [s * Math.cos(h), s * Math.sin(h)];
    this.kind = event.kind; this.curve = event.curve; this.length = event.length;
    this.age = 0; this.hold = 0;
    this.dots += 1;
  }
  _vanish() {
    this.dot = null; this.previous = null; this.hold = 0;
    if (this.source === "schedule" || this.scriptLeft > 0) {
      this.index += 1;
      this.wait = this.index < this.events.length ? this.events[this.index].gap : 1e9;
    }
  }
  /** The dot's law: a walk (heading noise), a curve (constant turning), a jitter (a walk with
   *  jumps); reflection at the walls; the changed laws gravity, faster and wrap. */
  _advance() {
    const s = this.schedule;
    const rot = this.kind === "curve" ? this.curve : this.rng.normal(0.0, s.turn);
    const c = Math.cos(rot), sn = Math.sin(rot);
    let v = [c * this.vel[0] - sn * this.vel[1], sn * this.vel[0] + c * this.vel[1]];
    if (this.law === "gravity") {
      v = [v[0], v[1] - GRAVITY];
      const speed = norm2(v[0], v[1]);
      if (speed > VMAX) v = [v[0] / speed * VMAX, v[1] / speed * VMAX];
    } else {
      const n = Math.max(norm2(v[0], v[1]), 1e-9), sp = this.speed;
      v = [v[0] / n * sp, v[1] / n * sp];
    }
    const step = [v[0], v[1]];
    if (this.kind === "jitter") { step[0] += this.rng.normal(0.0, 0.03); step[1] += this.rng.normal(0.0, 0.03); }
    const p = [this.dot[0] + step[0], this.dot[1] + step[1]];
    for (let axis = 0; axis < 2; axis++) {
      if (p[axis] < 0.0 || p[axis] > 1.0) {
        if (this.law === "wrap") p[axis] = ((p[axis] % 1.0) + 1.0) % 1.0;
        else { p[axis] = p[axis] < 0.0 ? -p[axis] : 2.0 - p[axis]; v[axis] = -v[axis]; }
      }
    }
    this.vel = v;
    this.previous = this.dot;
    this.dot = [clip(p[0], 0.0, 1.0), clip(p[1], 0.0, 1.0)];
  }
  step(a) {
    const s = this.schedule;
    const flags = { appear: false, catch: false, miss: false, change: false, vanish: false, present: false };
    if (s.change !== "none" && this.t === s.change_at && this.source === "schedule") this.pending.push(s.change);
    const pending = this.pending; this.pending = [];
    for (const event of pending) {
      if (LAWS.includes(event)) { this.law = event; this.changed = true; flags.change = true; }
      else if (event === "restore") { this.law = "routine"; this.changed = false; }
      else if (event === "script") { this.scriptLeft = 600; if (this.dot === null) this.wait = Math.min(this.wait, 5); }
    }
    // the paw
    const ax = clip(a[0], -1.0, 1.0), ay = clip(a[1], -1.0, 1.0);
    this.paw = [clip(this.paw[0] + PAW_SPEED * ax, 0.0, 1.0), clip(this.paw[1] + PAW_SPEED * ay, 0.0, 1.0)];
    // the dot
    if (this.source === "mouse" && this.scriptLeft <= 0) {
      if (this.mouse === null) { this.dot = null; this.previous = null; }
      else {
        if (this.mouseCaught !== null && norm2(this.mouse[0] - this.mouseCaught[0], this.mouse[1] - this.mouseCaught[1]) > 0.03) this.mouseCaught = null;
        if (this.mouseCaught === null) {
          if (this.dot === null) { flags.appear = true; this.dots += 1; this.age = 0; this.hold = 0; this.previous = null; }
          else this.previous = this.dot;
          this.dot = [this.mouse[0], this.mouse[1]];
        } else { this.dot = null; this.previous = null; }
      }
    } else {
      if (this.scriptLeft > 0) this.scriptLeft -= 1;
      if (this.dot !== null) {
        this._advance();
        this.age += 1;
        if (this.kind === "jitter" && this.age >= this.length) { this._vanish(); flags.vanish = true; }
        else if (this.age >= s.dot_timeout) { this._vanish(); this.misses += 1; flags.miss = flags.vanish = true; }
      } else if (this.source === "schedule" || this.scriptLeft > 0) {
        this.wait -= 1;
        if (this.wait <= 0 && this.index < this.events.length) { this._spawn(this.events[this.index]); flags.appear = true; }
      }
    }
    // the catch
    if (this.dot !== null) {
      if (norm2(this.paw[0] - this.dot[0], this.paw[1] - this.dot[1]) <= CATCH_RADIUS) this.hold += 1;
      else this.hold = 0;
      if (this.hold >= CATCH_HOLD) {
        this.catches += 1;
        flags.catch = flags.vanish = true;
        if (this.source === "mouse" && this.scriptLeft <= 0) {
          this.mouseCaught = this.mouse === null ? null : [this.mouse[0], this.mouse[1]];
          this.dot = null; this.hold = 0;
        } else this._vanish();
      }
    }
    flags.present = this.dot !== null;
    this.t += 1;
    return [this.reading(), flags];
  }
}
