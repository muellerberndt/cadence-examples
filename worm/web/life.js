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

export class Life {
  constructor(spec, p, seed = 0) {
    this.p = p; this.brain = new Brain(spec);
    this.random = mulberry32(seed * 7919 + 17);
    this.uniform = (lo, hi) => lo + (hi - lo) * this.random();
    [this.w, this.h] = p.plate;
    this.t = 0; this.tick_count = 0; this.items = []; this.events = [];
    this.prev = { food: 0, pain: 0 }; this.value_prev = 0; this.bias = 0;
    this.pending = { food: 0, pain: 0 };
    this.stretch = []; this.path = null; this.readout = [0, 0];
    this.activity = new Float64Array(spec.H); this.previousActivity = new Float64Array(spec.H);
    this.lessons = 0; this.rejected = 0;
    const heading = this.uniform(-Math.PI, Math.PI);
    this.body = { x: this.w / 2, y: this.h / 2, heading, trail: [], mode: "forward", timer: 0, turn_after: 0 };
    const L = p.body_length;
    for (let k = 0; k < 60; k++) {
      const d = 3 * L * k / 60;
      this.body.trail.push([this.body.x - d * Math.cos(heading), this.body.y - d * Math.sin(heading)]);
    }
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
  reverse(seconds, turn) { Object.assign(this.body, { mode: "reverse", timer: seconds, turn_after: turn }); }

  physics(dt, onFood) {
    const b = this.body, p = this.p;
    if (b.mode === "reverse") {
      let dist = p.reverse_speed * dt;
      while (dist > 0 && b.trail.length > 2) {
        const [hx, hy] = b.trail[0], [nx, ny] = b.trail[1];
        const seg = Math.hypot(nx - hx, ny - hy);
        if (seg <= dist) { b.trail.shift(); dist -= seg; }
        else { const f = dist / seg; b.trail[0] = [hx + f * (nx - hx), hy + f * (ny - hy)]; dist = 0; }
      }
      [b.x, b.y] = b.trail[0];
      b.timer -= dt;
      if (b.timer <= 0) {
        b.mode = "forward"; b.heading += b.turn_after;
        if (b.trail.length > 3) { const [tx, ty] = b.trail[3]; b.heading = Math.atan2(b.y - ty, b.x - tx) + b.turn_after; }
      }
    } else {
      const speed = onFood ? p.dwell_speed : p.crawl_speed;
      const phase = b.heading + p.swing_amplitude * Math.sin(2 * Math.PI * this.t / p.swing_period);
      b.x += speed * dt * Math.cos(phase); b.y += speed * dt * Math.sin(phase);
      const m = 0.25;
      if (b.x < m || b.x > this.w - m) { b.heading = Math.PI - b.heading; b.x = Math.min(Math.max(b.x, m), this.w - m); }
      if (b.y < m || b.y > this.h - m) { b.heading = -b.heading; b.y = Math.min(Math.max(b.y, m), this.h - m); }
      b.trail.unshift([b.x, b.y]);
    }
    let total = 0, keep = 1;
    for (let k = 1; k < b.trail.length; k++) {
      total += Math.hypot(b.trail[k][0] - b.trail[k - 1][0], b.trail[k][1] - b.trail[k - 1][1]);
      keep = k + 1; if (total > 3 * p.body_length) break;
    }
    b.trail.length = keep;
    this.t += dt;
  }
  swingDirection() { return Math.cos(2 * Math.PI * this.t / this.p.swing_period) >= 0 ? 1 : -1; }

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
    const y = this.sense(u); this.readout = y;
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
          const side = this.random() < 0.5 ? 1 : -1;
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
