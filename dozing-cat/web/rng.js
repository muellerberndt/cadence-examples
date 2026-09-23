// The page's generator: Mulberry32, the generator of brain_scan.js and of cadence.Mulberry32.
// `normals` lays out Box-Muller as cadence.Mulberry32.normals does (every uniform drawn first,
// the radii from the first half and the angles from the second, all the cosine terms then all
// the sine terms), so the record store's projection rebuilt here is the library's projection.

export function mulberry32(seed) {
  let state = seed >>> 0;
  return function () {
    state = (state + 0x6d2b79f5) >>> 0;
    const a = state;
    let t = Math.imul(a ^ (a >>> 15), a | 1) >>> 0;
    t = (t ^ ((t + (Math.imul(t ^ (t >>> 7), t | 61) >>> 0)) >>> 0)) >>> 0;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296.0;
  };
}

export function normals(random, n) {
  const pairs = (n + 1) >> 1;
  const u = new Float64Array(2 * pairs);
  for (let i = 0; i < 2 * pairs; i++) u[i] = random();
  const out = new Float64Array(n);
  for (let i = 0; i < pairs; i++) {
    const radius = Math.sqrt(-2.0 * Math.log(Math.max(u[i], 1e-12)));
    const angle = 2.0 * Math.PI * u[pairs + i];
    out[i] = radius * Math.cos(angle);
    if (pairs + i < n) out[pairs + i] = radius * Math.sin(angle);
  }
  return out;
}

/** The draws the sill's schedule and the habit refit need, from one Mulberry32 stream. */
export class Rng {
  constructor(seed) { this.next = mulberry32(seed); this.spare = null; }
  random() { return this.next(); }
  uniform(lo, hi) { return lo + (hi - lo) * this.random(); }
  integers(lo, hi) { return lo + Math.floor(this.random() * (hi - lo + 1)); } // inclusive of hi
  normal(mean = 0.0, sd = 1.0) {
    if (this.spare !== null) { const v = this.spare; this.spare = null; return mean + sd * v; }
    const u = Math.max(1e-12, this.random()), v = this.random();
    const radius = Math.sqrt(-2.0 * Math.log(u)), angle = 2.0 * Math.PI * v;
    this.spare = radius * Math.sin(angle);
    return mean + sd * radius * Math.cos(angle);
  }
}
