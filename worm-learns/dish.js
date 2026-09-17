// The dish for the browser: two odour spots, food under one of them. Mirrors Dish in worm.py
// (same geometry, senses, step and outcomes) for one worm; its random numbers are the page's own.

export const DISH = {
  radius: 1.0,
  sourceDistance: 0.6,
  contact: 0.12,
  sigma: 0.6,
  head: 0.05,
  contrast: 8.0,
  step: 0.04,
  turn: 0.45,
  wobble: 0.15,
  limit: 150,
};

export const ODOURS = ["diacetyl", "butanone"];

// Small seeded generator (mulberry32) so a page reload replays the same dish.
export function generator(seed) {
  let a = seed >>> 0;
  const next = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const normal = () => {
    const u = Math.max(next(), 1e-12), v = next();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  return { next, normal, uniform: (lo, hi) => lo + (hi - lo) * next() };
}

export class Dish {
  constructor({ seed = 1, meaning = 0, config = DISH } = {}) {
    this.c = config;
    this.rng = generator(seed);
    this.meaning = meaning; // which odour sits on food for new searches
    this.reset();
  }

  reset() {
    const c = this.c, r = this.rng;
    this.pos = [0, 0];
    this.heading = r.uniform(-Math.PI, Math.PI);
    const theta = r.uniform(-Math.PI, Math.PI);
    const other = theta + Math.PI + r.uniform(-Math.PI / 3, Math.PI / 3);
    const food = [Math.cos(theta) * c.sourceDistance, Math.sin(theta) * c.sourceDistance];
    const empty = [Math.cos(other) * c.sourceDistance, Math.sin(other) * c.sourceDistance];
    this.food = this.meaning;
    this.spots = this.food === 0 ? [food, empty] : [empty, food];
    this.t = 0;
    this.touching = false;
    this.trail = [[0, 0]];
  }

  // Concentration of odour k at a point, for drawing the clouds.
  concentration(k, x, y) {
    const [sx, sy] = this.spots[k];
    return Math.exp(-((x - sx) ** 2 + (y - sy) ** 2) / (2 * this.c.sigma ** 2));
  }

  // [diacetyl left, diacetyl right, butanone left, butanone right, touch, touch]
  observe() {
    const c = this.c;
    const f = [Math.cos(this.heading), Math.sin(this.heading)];
    const left = [-f[1], f[0]];
    const tip = [this.pos[0] + f[0] * c.head, this.pos[1] + f[1] * c.head];
    const obs = [0, 0, 0, 0, 0, 0];
    for (let k = 0; k < 2; k++) {
      const [sx, sy] = this.spots[k];
      const dl = (tip[0] + left[0] * c.head - sx) ** 2 + (tip[1] + left[1] * c.head - sy) ** 2;
      const dr = (tip[0] - left[0] * c.head - sx) ** 2 + (tip[1] - left[1] * c.head - sy) ** 2;
      const diff = Math.tanh((c.contrast * (dr - dl)) / (2 * c.sigma ** 2));
      obs[2 * k] = Math.max(diff, 0);
      obs[2 * k + 1] = Math.max(-diff, 0);
    }
    obs[4] = obs[5] = this.touching ? 1 : 0;
    return obs;
  }

  // action: 0 turn left, 1 turn right, 2 forward. Returns {reward, finished, outcome, approach}.
  step(action) {
    const c = this.c;
    const foodSpot = this.spots[this.food];
    const before = Math.hypot(foodSpot[0] - this.pos[0], foodSpot[1] - this.pos[1]);
    if (action === 0) this.heading += c.turn;
    if (action === 1) this.heading -= c.turn;
    this.heading += c.wobble * this.rng.normal();
    this.heading = ((((this.heading + Math.PI) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)) - Math.PI;
    const nx = this.pos[0] + Math.cos(this.heading) * c.step;
    const ny = this.pos[1] + Math.sin(this.heading) * c.step;
    this.touching = Math.hypot(nx, ny) > c.radius;
    if (!this.touching) this.pos = [nx, ny];
    this.t += 1;
    this.trail.push([...this.pos]);
    const dFood = Math.hypot(foodSpot[0] - this.pos[0], foodSpot[1] - this.pos[1]);
    const emptySpot = this.spots[1 - this.food];
    const dEmpty = Math.hypot(emptySpot[0] - this.pos[0], emptySpot[1] - this.pos[1]);
    let outcome = null;
    if (dFood < c.contact) outcome = "food";
    else if (dEmpty < c.contact) outcome = "empty";
    else if (this.t >= c.limit) outcome = "timeout";
    const approach = (before - dFood) / c.step;
    return { reward: outcome === "food" ? 1 : 0, finished: outcome !== null, outcome, approach, steps: this.t };
  }
}
