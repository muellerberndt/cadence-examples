// The two-link planar arm in JavaScript: a port of arm/env.py.
// Links of 0.5 each, angles wrapped to [-pi, pi], velocities clipped to +-max_velocity, a
// 50 Hz body clock with semi-implicit Euler, one torque pair held for `substeps` body
// steps. Sensors are proprioception only; the goal is the visible target marker. The random
// numbers come from a Mulberry32 stream (Python's env draws from numpy; the page runs its
// own life, so the two need not agree). `requestTruncation` ends the episode at the next
// decision as a time limit would, so a page can move to a new target without abandoning a
// decision that awaits its outcome.

import { Mulberry32 } from "../../web/engine.js";

export const TORQUES = [];
for (const a of [-1.0, 0.0, 1.0]) for (const b of [-1.0, 0.0, 1.0]) TORQUES.push([a, b]);

export const FIELDS = {
  angles: [4, -1.0, 1.0], velocity: [2, -1.0, 1.0], hand: [2, -1.0, 1.0], d_hand: [2, -0.12, 0.12], dd_hand: [2, -0.03, 0.03],
  d_velocity: [2, -0.15, 0.15], d_angles: [2, -0.25, 0.25], flags: [2, 0.0, 1.0],
};

const TWO_PI = 2 * Math.PI;

/** Python's `(angle + pi) % (2 pi) - pi` (a nonnegative modulus). */
export function wrap(angle) {
  const m = (((angle + Math.PI) % TWO_PI) + TWO_PI) % TWO_PI;
  return m - Math.PI;
}

/** The little random interface the arm and the moving paths draw from. */
export class ArmRandom {
  constructor(seed) { this.rng = new Mulberry32(seed); }
  random() { return this.rng.random(); }
  uniform(lo, hi) { return lo + (hi - lo) * this.rng.random(); }
  choice(values) { return values[Math.floor(this.rng.random() * values.length)]; }
}

export const DEFAULT_ARM = { lengths: [0.5, 0.5], dt: 0.02, substeps: 5, max_velocity: 2.0, friction: 0.1, gain: 1.0, success_radius: 0.05, success_hold: 10, horizon: 200, target_radius: [0.2, 0.95], torque_cost: 0.001 };

export class Arm {
  constructor(config = {}, { seed = 0, life_id = "arm", event = 0, episode = -1, movingTarget = null } = {}) {
    this.config = { ...DEFAULT_ARM, ...config, lengths: [...(config.lengths || DEFAULT_ARM.lengths)], target_radius: [...(config.target_radius || DEFAULT_ARM.target_radius)] };
    this.rng = new ArmRandom(seed);
    this.life_id = life_id;
    this.movingTarget = movingTarget; // a callable tick -> [x, y], or null for a static target
    this._episode = episode; this._event = event; this._tick = 0;
    this.theta = new Float64Array(2); this.omega = new Float64Array(2); this._target = new Float64Array(2);
    this._previous = null; this._flags = new Float64Array(2); this._hold = 0; this._pending = null; this._decisionAction = null;
    this._truncateNext = false;
  }

  // -- kinematics (supplied for sensors and rendering)

  handOf(theta) {
    const [l1, l2] = this.config.lengths;
    return Float64Array.of(l1 * Math.cos(theta[0]) + l2 * Math.cos(theta[0] + theta[1]), l1 * Math.sin(theta[0]) + l2 * Math.sin(theta[0] + theta[1]));
  }
  elbowOf(theta) { const l1 = this.config.lengths[0]; return Float64Array.of(l1 * Math.cos(theta[0]), l1 * Math.sin(theta[0])); }
  get hand() { return this.handOf(this.theta); }
  get target() { return Float64Array.from(this._target); }
  get tick() { return this._tick; }
  get episode() { return this._episode; }
  get event() { return this._event; }
  get hold() { return this._hold; }
  /** Evaluation and rendering only: never handed to the agent. */
  get state() { return { theta: Float64Array.from(this.theta), omega: Float64Array.from(this.omega), target: this.target, hand: this.hand, elbow: this.elbowOf(this.theta) }; }
  distance() { const h = this.hand; return Math.sqrt((h[0] - this._target[0]) ** 2 + (h[1] - this._target[1]) ** 2); }

  // -- episodes

  sampleTarget() {
    const r = this.rng.uniform(this.config.target_radius[0], this.config.target_radius[1]);
    const a = this.rng.uniform(-Math.PI, Math.PI);
    return Float64Array.of(r * Math.cos(a), r * Math.sin(a));
  }

  reset({ theta = null, target = null } = {}) {
    this._episode += 1; this._tick = 0; this._hold = 0; this._pending = null; this._truncateNext = false;
    this.theta = theta === null ? Float64Array.of(this.rng.uniform(-Math.PI, Math.PI), this.rng.uniform(-Math.PI, Math.PI)) : Float64Array.from(theta);
    this.omega = new Float64Array(2);
    this._flags = new Float64Array(2);
    this._target = target === null ? this.sampleTarget() : Float64Array.from(target);
    if (this.movingTarget !== null) this._target = Float64Array.from(this.movingTarget(0));
    this._previous = { hand: this.hand, omega: Float64Array.from(this.omega), theta: Float64Array.from(this.theta), d_hand: new Float64Array(2) };
    return this._moment({ feedback: false, reward: 0.0, terminated: false, truncated: false });
  }

  observation() {
    const hand = this.hand, p = this._previous;
    const dHand = Float64Array.of(hand[0] - p.hand[0], hand[1] - p.hand[1]);
    return {
      angles: Float64Array.of(Math.sin(this.theta[0]), Math.cos(this.theta[0]), Math.sin(this.theta[1]), Math.cos(this.theta[1])),
      velocity: Float64Array.from(this.omega, (w) => w / this.config.max_velocity),
      hand,
      d_hand: dHand,
      dd_hand: Float64Array.of(dHand[0] - p.d_hand[0], dHand[1] - p.d_hand[1]),
      d_velocity: Float64Array.of(this.omega[0] - p.omega[0], this.omega[1] - p.omega[1]),
      d_angles: Float64Array.of(wrap(this.theta[0] - p.theta[0]), wrap(this.theta[1] - p.theta[1])),
      flags: Float64Array.from(this._flags),
    };
  }

  goal() { return Float64Array.of((this._target[0] + 1.0) / 2.0, (this._target[1] + 1.0) / 2.0); } // the marker, in [0, 1] for the goal port

  _moment({ feedback, reward, terminated, truncated }) {
    const observation = this.observation();
    const m = {
      life_id: this.life_id, episode_id: this._episode, event_id: this._event, tick: this._tick, dt: this.config.dt * this.config.substeps,
      observation, observed: {}, action_mask: new Array(TORQUES.length).fill(1),
      feedback_for: feedback ? this._pending : null, executed: feedback ? this._decisionAction : null,
      reward, reward_known: feedback, terminated, truncated, final_observation: truncated ? this.observation() : null, goal: this.goal(),
    };
    this._event += 1;
    return m;
  }

  /** Continue from a recorded moment: the body state its sensors imply (angles from their sines and
   *  cosines, velocities, the previous step from the measured deltas, the target from the goal
   *  marker) and its cursors, so a life resumes where a snapshot left it. The success hold is
   *  not observable and starts at zero. */
  restore(m) {
    const o = m.observation, c = this.config;
    this._episode = m.episode_id | 0; this._event = (m.event_id | 0) + 1; this._tick = m.tick | 0; this._hold = 0; this._pending = null; this._truncateNext = false;
    this.theta = Float64Array.of(Math.atan2(o.angles[0], o.angles[1]), Math.atan2(o.angles[2], o.angles[3]));
    this.omega = Float64Array.of(o.velocity[0] * c.max_velocity, o.velocity[1] * c.max_velocity);
    this._flags = Float64Array.from(o.flags);
    this._target = m.goal ? Float64Array.of(m.goal[0] * 2.0 - 1.0, m.goal[1] * 2.0 - 1.0) : this.sampleTarget();
    if (this.movingTarget !== null) this._target = Float64Array.from(this.movingTarget(this._tick));
    this._previous = {
      hand: Float64Array.of(o.hand[0] - o.d_hand[0], o.hand[1] - o.d_hand[1]),
      omega: Float64Array.of(this.omega[0] - o.d_velocity[0], this.omega[1] - o.d_velocity[1]),
      theta: Float64Array.of(wrap(this.theta[0] - o.d_angles[0]), wrap(this.theta[1] - o.d_angles[1])),
      d_hand: Float64Array.of(o.d_hand[0] - o.dd_hand[0], o.d_hand[1] - o.dd_hand[1]),
    };
    return this;
  }

  /** Hold one torque pair for `substeps` body steps; record velocity-limit flags. */
  integrate(action) {
    const c = this.config, torque = TORQUES[action | 0].map((t) => t * c.gain), flags = new Float64Array(2);
    for (let k = 0; k < c.substeps; k++) {
      for (let j = 0; j < 2; j++) {
        this.omega[j] = this.omega[j] + c.dt * (torque[j] - c.friction * this.omega[j]);
        if (Math.abs(this.omega[j]) >= c.max_velocity) flags[j] = Math.max(flags[j], 1.0);
        this.omega[j] = Math.min(c.max_velocity, Math.max(-c.max_velocity, this.omega[j]));
      }
      for (let j = 0; j < 2; j++) this.theta[j] = wrap(this.theta[j] + c.dt * this.omega[j]);
    }
    this._flags = flags;
  }

  /** End the episode at the next decision, as a time limit would (a page moving to a new target). */
  requestTruncation() { this._truncateNext = true; }

  act(decisionId, action) {
    if (this._pending !== null) throw Error("the previous decision has not been fed back");
    action = action | 0;
    if (!(action >= 0 && action < TORQUES.length)) throw Error("action outside the nine torque pairs");
    this._pending = decisionId; this._decisionAction = action;
    const before = this.distance();
    const hand = this.hand;
    this._previous = { hand, omega: Float64Array.from(this.omega), theta: Float64Array.from(this.theta), d_hand: Float64Array.of(hand[0] - this._previous.hand[0], hand[1] - this._previous.hand[1]) };
    this.integrate(action);
    this._tick += 1;
    if (this.movingTarget !== null) this._target = Float64Array.from(this.movingTarget(this._tick));
    const after = this.distance();
    const torque = TORQUES[action];
    let reward = (before - after) - this.config.torque_cost * (torque[0] * torque[0] + torque[1] * torque[1]);
    this._hold = after <= this.config.success_radius ? this._hold + 1 : 0;
    const terminated = this.movingTarget === null && this._hold >= this.config.success_hold;
    if (terminated) reward += 1.0;
    const truncated = !terminated && (this._tick >= this.config.horizon || this._truncateNext);
    const m = this._moment({ feedback: true, reward, terminated, truncated });
    this._pending = null;
    if (truncated || terminated) this._truncateNext = false;
    return m;
  }
}

/** A fresh line, circle or smooth combination; the path is hidden, the agent sees positions. */
export function movingPath(kind, rng, decisionsPerSecond = 10.0) {
  const inside = (radius) => { const r = Math.sqrt(rng.uniform(0.04, radius * radius)), a = rng.uniform(-Math.PI, Math.PI); return [r * Math.cos(a), r * Math.sin(a)]; };
  if (kind === "line") {
    const a = inside(0.85), b = inside(0.85), speed = rng.uniform(0.02, 0.05); // arm lengths per second
    const length = Math.max(Math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2), 1e-6);
    return (tick) => {
      let s = ((speed * tick) / decisionsPerSecond) % (2 * length);
      s = s <= length ? s : 2 * length - s;
      return [a[0] + (b[0] - a[0]) * (s / length), a[1] + (b[1] - a[1]) * (s / length)];
    };
  }
  if (kind === "circle") {
    const centre = inside(0.3), radius = rng.uniform(0.15, 0.3), omega = rng.uniform(0.05, 0.15) * rng.choice([-1, 1]), phase = rng.uniform(0, 2 * Math.PI); // radians per second
    return (tick) => { const t = tick / decisionsPerSecond; return [centre[0] + radius * Math.cos(omega * t + phase), centre[1] + radius * Math.sin(omega * t + phase)]; };
  }
  const c1 = movingPath("circle", rng, decisionsPerSecond), c2 = movingPath("line", rng, decisionsPerSecond);
  return (tick) => { const p = c1(tick), q = c2(tick); return [0.5 * (p[0] + q[0]), 0.5 * (p[1] + q[1])]; };
}
