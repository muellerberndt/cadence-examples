// One life on the plate, mirrored from worm/world.py and worm/life.py.
import { Brain } from "./brain.js";

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const EAT_RATE = 0.05;
const TAU = 2 * Math.PI;
const wrap = (a) => { a = (a + Math.PI) % TAU; if (a < 0) a += TAU; return a - Math.PI; };
const clamp = (v, m) => Math.max(-m, Math.min(m, v));

export class Life {
  constructor(spec, p, seed = 0) {
    this.p = p; this.brain = new Brain(spec);
    this.random = mulberry32(seed * 7919 + 17);
    this.uniform = (lo, hi) => lo + (hi - lo) * this.random();
    [this.w, this.h] = p.plate;
    this.t = 0; this.tick_count = 0; this.items = []; this.events = [];
    this.prev = { food: 0, pain: 0 }; this.value_prev = 0; this.bias = 0;
    this.pending = { food: 0, pain: 0 };
    this.stretch = []; this.path = null; this.readout = [0, 0]; this.input = [0, 0, 0, 0];
    this.activity = new Float64Array(spec.H); this.previousActivity = new Float64Array(spec.H);
    this.lessons = 0; this.rejected = 0;
    // The body is its centreline: points from the head (first) to the tail (last),
    // always exactly one body length. Whichever end leads lays the track, the rest
    // follows it, and the leading end can only change direction at a bounded curvature.
    const heading = this.uniform(-Math.PI, Math.PI), L = p.body_length, crawl = 1.25 * L;
    const x0 = this.w / 2 - 0.926 * crawl * Math.cos(heading), y0 = this.h / 2 - 0.926 * crawl * Math.sin(heading);
    this.body = { x: x0, y: y0, heading, theta: heading, phase: 0, tail_heading: 0, tail_theta: 0,
                  trail: [], mode: "forward", timer: 0, turn: 0, turn_after: 0, walled: false };
    for (let k = 0; k <= 100; k++) this.body.trail.push([x0 - L * k / 100 * Math.cos(heading), y0 - L * k / 100 * Math.sin(heading)]);
    // born crawling: a body length and a quarter of travel gives it its wave, and brings it to the middle of the plate
    for (let k = 0, n = Math.round(crawl / (p.crawl_speed * p.physics_step)); k < n; k++) this.physics(p.physics_step, false);
    this.t = 0;
  }

  // ---- the plate -----------------------------------------------------------------
  freeSpot(clearance = 1.6) {
    let x = 0, y = 0;
    for (let k = 0; k < 200; k++) {
      x = this.uniform(1, this.w - 1); y = this.uniform(1, this.h - 1);
      if (this.items.every((it) => Math.hypot(x - it.x, y - it.y) > 2 * clearance) &&
          Math.hypot(x - this.body.x, y - this.body.y) > clearance) break;
    }
    return [x, y];
  }
  place(kind, x, y, odour, respawn = false) {
    const it = { kind, x, y, odour, amount: 1, respawn, born: this.t };
    this.items.push(it); return it;
  }
  remove(it) { this.items = this.items.filter((o) => o !== it); }

  odour(x, y) {
    const s2 = 2 * this.p.odour_sigma ** 2; let a = 0, b = 0;
    for (const it of this.items) {
      if (!it.odour || it.amount <= 0) continue;
      const c = it.amount ** 0.5 * Math.exp(-((x - it.x) ** 2 + (y - it.y) ** 2) / s2);
      if (it.odour === "A") a += c; else b += c;
    }
    return [Math.min(a, 1), Math.min(b, 1)];
  }
  contact(kind) {
    const r = kind === "food" ? this.p.food_radius : this.p.noxious_radius;
    return this.items.find((it) => it.kind === kind && it.amount > 0 &&
      Math.hypot(this.body.x - it.x, this.body.y - it.y) <= r) ?? null;
  }

  // ---- the body --------------------------------------------------------------------
  // direction of the centreline at one end, pointing out of the body; over `span` mm it is the body's axis there
  direction(end, span = 0) {
    const tr = this.body.trail, n = tr.length, step = end === "head" ? 1 : -1, first = end === "head" ? 0 : n - 1;
    let k = first, acc = 0;
    while (k + step >= 0 && k + step < n) {
      acc += Math.hypot(tr[k + step][0] - tr[k][0], tr[k + step][1] - tr[k][1]); k += step;
      if (acc > Math.max(span, 1e-9)) break;
    }
    return Math.atan2(tr[first][1] - tr[k][1], tr[first][0] - tr[k][0]);
  }
  // keep the centreline exactly one body length, giving up the excess at the trailing end
  trim(end) {
    const tr = this.body.trail;
    let extra = -this.p.body_length;
    for (let k = 1; k < tr.length; k++) extra += Math.hypot(tr[k][0] - tr[k - 1][0], tr[k][1] - tr[k - 1][1]);
    while (extra > 0 && tr.length > 2) {
      const a = end === "tail" ? tr.length - 1 : 0, b = end === "tail" ? tr.length - 2 : 1;
      const seg = Math.hypot(tr[b][0] - tr[a][0], tr[b][1] - tr[a][1]);
      if (seg <= extra) { end === "tail" ? tr.pop() : tr.shift(); extra -= seg; }
      else { const f = extra / seg; tr[a] = [tr[a][0] + f * (tr[b][0] - tr[a][0]), tr[a][1] + f * (tr[b][1] - tr[a][1])]; extra = 0; }
    }
  }
  // a reversal: the tail leads for `seconds`, then the head curls through `turn` radians (an omega turn)
  reverse(seconds, turn) {
    const b = this.body;
    Object.assign(b, { mode: "reverse", timer: seconds, turn_after: turn, turn: 0,
                       tail_theta: this.direction("tail"), tail_heading: this.direction("tail", this.p.swing_wavelength) });
  }
  resume() {
    const b = this.body;
    Object.assign(b, { mode: "forward", theta: this.direction("head"), heading: this.direction("head", this.p.swing_wavelength),
                       turn: b.turn_after, walled: false });
  }

  physics(dt, onFood) {
    const b = this.body, p = this.p, tr = b.trail;
    const speed = b.mode === "reverse" ? p.reverse_speed : onFood ? p.dwell_speed : p.crawl_speed;
    const ds = speed * dt, bend = p.max_curvature * ds;
    b.phase = (b.phase + TAU * ds / p.swing_wavelength) % TAU;
    if (b.mode === "reverse") {
      b.tail_theta += clamp(wrap(b.tail_heading + p.swing_amplitude * Math.sin(b.phase) - b.tail_theta), bend);
      const [tx, ty] = tr[tr.length - 1], nx = tx + ds * Math.cos(b.tail_theta), ny = ty + ds * Math.sin(b.tail_theta), m = 0.08;
      if (nx < m || nx > this.w - m || ny < m || ny > this.h - m) b.timer = 0;      // the tail has met the edge of the plate
      else { tr.push([nx, ny]); this.trim("head"); [b.x, b.y] = tr[0]; }
      b.timer -= dt;
      if (b.timer <= 0) this.resume();
    } else {
      // the edge of the plate: a turn toward the mirrored heading, begun a margin away
      let rx = Math.cos(b.heading), ry = Math.sin(b.heading), hit = false;
      if ((b.x < p.wall_margin && rx < 0) || (b.x > this.w - p.wall_margin && rx > 0)) { rx = -rx; hit = true; }
      if ((b.y < p.wall_margin && ry < 0) || (b.y > this.h - p.wall_margin && ry > 0)) { ry = -ry; hit = true; }
      if (hit && (!b.walled || b.turn === 0)) b.turn = wrap(Math.atan2(ry, rx) - b.heading);
      b.walled = hit;
      let swing = p.swing_amplitude, d = 0;
      if (b.turn !== 0) {           // a turn under way curls the head round at the turn's curvature
        d = clamp(b.turn, p.turn_curvature * ds);
        b.heading = wrap(b.heading + d); b.turn -= d; swing *= p.turn_swing;
      }
      // the direction of travel follows the heading and the head swing with what is left of the bend the body allows
      b.theta = wrap(b.theta + d + clamp(wrap(b.heading + swing * Math.sin(b.phase) - b.theta - d), bend - Math.abs(d)));
      const e = 0.06;
      b.x = Math.min(Math.max(b.x + ds * Math.cos(b.theta), e), this.w - e);
      b.y = Math.min(Math.max(b.y + ds * Math.sin(b.theta), e), this.h - e);
      tr.unshift([b.x, b.y]); this.trim("tail");
    }
    this.t += dt;
  }
  swingDirection() { return Math.cos(this.body.phase) >= 0 ? 1 : -1; }

  // ---- the brain, by cadence's documented calls ------------------------------------------
  sense(u) {
    const T = this.p.window;
    this.stretch.push(Float64Array.from(u));
    if (this.stretch.length > 2 * T) { this.brain.advance(this.stretch.slice(0, T)); this.stretch = this.stretch.slice(T); }
    this.path = this.brain.imagine(this.stretch);
    const H = this.brain.H, last = this.path.T - 1;
    this.previousActivity = this.activity;
    this.activity = this.path.hidden.slice(last * H, (last + 1) * H).map(Math.tanh);
    return [this.path.output[last * 2], this.path.output[last * 2 + 1]];
  }

  learn(kinds) {
    const O = 2, T = this.path.T, y = this.path.output.slice(), k = Math.min(this.p.teach, T);
    for (const kind of kinds) {
      const want = kind === "food" ? 0 : 1;
      for (let t = T - k; t < T; t++) { y[t * O + want] = this.p.teach_level; if (kinds.length === 1) y[t * O + 1 - want] = 0; }
    }
    const r = this.brain.observe(this.stretch, y, this.p.beta, this.p.rate);
    const stretch = this.stretch; this.stretch = [];
    let halvings = 0;
    if (r.updated) {
      let after = this.brain.parameters();
      while (this.brain.growth(after.A) > this.p.stability && halvings < 12) {
        after = { A: r.before.A.map((v, i) => v + 0.5 * (after.A[i] - v)),
                  B: r.before.B.map((v, i) => v + 0.5 * (after.B[i] - v)),
                  C: r.before.C.map((v, i) => v + 0.5 * (after.C[i] - v)) };
        halvings++;
      }
      if (this.brain.growth(after.A) > this.p.stability) after = r.before;   // no stable share of it: the lesson is not kept
      if (halvings) this.brain.setParameters(after);
      this.lessons++;
    } else this.rejected++;
    // what the visual layer needs: each synapse's applied change, each neuron's credit
    const H = this.brain.H, applied = r.updated ? this.brain.parameters().A.map((v, i) => v - r.before.A[i]) : null;
    const credit = new Float64Array(H);
    if (r.plus && r.minus) for (let t = 0; t < r.plus.T; t++) for (let i = 0; i < H; i++)
      credit[i] += Math.abs(r.plus.hidden[t * H + i] - r.minus.hidden[t * H + i]) / r.plus.T;
    return { kinds, updated: r.updated, reason: r.reason, applied, credit, halvings, length: stretch.length, at: this.t };
  }

  // ---- one tick: ten physics steps, then the brain thinks once -----------------------------
  tick() {
    const p = this.p, steps = Math.round(p.tick / p.physics_step);
    for (let s = 0; s < steps; s++) this.step();
    return this.think();
  }
  step() { this.physics(this.p.physics_step, this.contact("food") !== null); }
  think() {
    const p = this.p;
    const [a, b] = this.odour(this.body.x, this.body.y);
    const eating = this.contact("food"), hurting = this.contact("noxious");
    const food = eating || this.pending.food > 0 ? 1 : 0, pain = hurting || this.pending.pain > 0 ? 1 : 0;
    for (const k in this.pending) this.pending[k] = Math.max(0, this.pending[k] - 1);
    const u = [a, b, food, pain];
    const y = this.sense(u); this.readout = y; this.input = u;
    const n = this.tick_count;
    const foodOn = food > this.prev.food, painOn = pain > this.prev.pain;
    this.prev = { food, pain };
    if (eating) {
      eating.amount -= EAT_RATE * p.tick;
      if (eating.amount <= 0) {
        if (eating.respawn) { [eating.x, eating.y] = this.freeSpot(); eating.amount = 1; eating.born = this.t; }
        else this.remove(eating);
      }
    }
    const arrived = [["food", foodOn], ["pain", painOn]].filter(([, on]) => on).map(([k]) => k);
    for (const kind of arrived) this.events.push({ tick: n, t: this.t, event: kind });
    const lesson = arrived.length ? this.learn(arrived) : null;

    const body = this.body, forward = y[0], reverse = y[1], value = forward - reverse, dv = value - this.value_prev;
    this.value_prev = value;
    if (body.mode === "forward") {
      this.bias = p.taxis_memory * this.bias + (1 - p.taxis_memory) * dv * this.swingDirection();
      body.heading += p.taxis * this.bias;
      if (!eating) {
        const turn = p.base_turn + p.reverse_gain * Math.max(0, reverse - forward - p.reverse_threshold) + p.kinesis * Math.max(0, -dv);
        if (this.random() < turn) {
          const side = this.random() < p.ventral_bias ? -1 : 1;      // omega turns curl ventrally, mostly
          this.reverse(p.pirouette_reverse * (1 + 2 * Math.min(1, Math.max(0, reverse))), side * this.uniform(1.5, 3.0));
        }
      }
    }
    this.tick_count++;
    return { tick: n, u, y, lesson, arrived };
  }

  treat() { this.pending.food = 2; this.events.push({ tick: this.tick_count, t: this.t, event: "treat" }); }
  poke() { this.pending.pain = 2; this.events.push({ tick: this.tick_count, t: this.t, event: "poke" }); }

  probe(sense) {       // what one smell alone now does to the command neurons, from rest
    const T = this.p.window, k = { odour_A: 0, odour_B: 1 }[sense];
    const U = Array.from({ length: T }, () => { const u = new Float64Array(4); u[k] = 1; return u; });
    const out = this.brain.imagine(U, new Float64Array(this.brain.H)).output;
    return [out[(T - 1) * 2], out[(T - 1) * 2 + 1]];
  }
}
