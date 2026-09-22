/* cadence-bodies core: the substrate, soft bodies, the two-patch brain, the genome and the population. No DOM.
   Used by web/index.html (inlined after patch.js), sim/probe.js and sim/gaitlab.js (node). */
(function (root, factory) { if (typeof module === 'object' && module.exports) module.exports = factory(require('./patch.js')); else root.CB = factory(root.CP); })(typeof self !== 'undefined' ? self : this, function (CP) {
'use strict';
const { Mulberry, RecordPatch } = CP;

// ---------- the rules: each a flag, each one physical rule ----------
const RULES = { rock: true, mud: true, bite: true, injury: true, growth: true, devnoise: true, night: false, split: true,
  records: true, slow: true, plan: true, noise: true, sleep: false }; // sleep: whether the genome may express a sleeping night // the last four gate what the genome may express: the controls
const CFG = { w: 96, h: 96, soil: 6, grow: 0.05, decay: 0.01, diffuse: 0.05, day: 900, season: 9000, contrastMin: 0.25, foodCap: 8, rockFraction: 0.10, mudFraction: 0.22, mudKeep: 0.45, warmup: 400, nightBelow: 0.35,
  scentPasses: 4, substeps: 3, iters: 2, muscleAmp: 0.35, muscleLag: 0.5, contact: 0.6, contactPush: 0.4, sense: 2.5, aheadAt: 1.0, growAt: 6, birthNodes: 3, biteLands: 0.5, biteTake: 1, window: 8, clip: 5, backtrack: false, recode: false, fixedRead: true, policyCells: false, maxNodes: 10, maxMuscles: 8 };
const POP = { initial: 220, birth: 20, max: 500, base: 0.035, node: 0.006, muscle: 0.02, bite: 0.1, read: 0.0001, channel: 0.0003, cell: 0.01 / 256, write: 0.0005, replay: 0.001, slow: 0.000004, cue: 0.01 / 256 };
const SLEEP = { windows: 32 }; // a sleeper keeps this many of the day's windows to dream from
const N = CFG.w * CFG.h;
const DIRS = [[0, -1], [1, 0], [0, 1], [-1, 0]];
const MAXN = CFG.maxNodes, MAXM = CFG.maxMuscles, MAXS = 2 * MAXN;
// the reading: senses, then one strain per muscle port; the clock; the motor (muscles then the mouth)
const SENSES = ['food L-R', 'food ahead', 'food here', 'rock ahead', 'other ahead', 'energy', 'pain', 'light', 'grip', 'speed', 'turn', 'size'];
const NS = SENSES.length, SN = NS + MAXM, NM = MAXM + 1, NI = SN + 2 + NM, NOUT = SN + 1, HP = 16, HM = 16, ACTIVE = 16, FANIN = 8;
const PORT = { policy: { inputs: NI, hidden: HP, outputs: NM }, model: { inputs: NI, hidden: HM, outputs: NOUT } };
const CLOCK = SN, MOTOR = SN + 2; // offsets in the reading
const S = {}; SENSES.forEach((n, i) => S[n] = i);
// the reading in groups: a cortex is a block of channels that reads the groups its mask names
const RG = { chemo: [S['food L-R'], S['food ahead'], S['food here']], obstacle: [S['rock ahead'], S['other ahead']], self: [S.energy, S.pain, S.light, S.grip], motion: [S.speed, S.turn, S.size],
  strain: Array.from({ length: MAXM }, (_, m) => NS + m), clock: [CLOCK, CLOCK + 1], motor: Array.from({ length: NM }, (_, m) => MOTOR + m) };
const RG_NAMES = Object.keys(RG), gbit = name => 1 << RG_NAMES.indexOf(name), ALL_MASK = (1 << RG_NAMES.length) - 1, MAXH = 64, SLOWEST = [8, 32, 128];
function maskUnits(mask) { const u = new Uint8Array(NI); RG_NAMES.forEach((n, k) => { if (mask & (1 << k)) for (const j of RG[n]) u[j] = 1; }); return u; }
function maskCount(mask) { let c = 0; RG_NAMES.forEach((n, k) => { if (mask & (1 << k)) c += RG[n].length; }); return c; }
function maskNames(mask) { return RG_NAMES.filter((n, k) => mask & (1 << k)); }
function brainH(cortices) { return cortices.reduce((s, c) => s + c.channels, 0); }
function inputMask(cortices) { const H = brainH(cortices), m = new Uint8Array(H * NI); let row = 0; for (const c of cortices) { const u = maskUnits(c.mask); for (let i = 0; i < c.channels; i++, row++) m.set(u, row * NI); } return m; }
function blockTimescales(n, slowest) { const ts = new Float64Array(n); for (let i = 0; i < n; i++) ts[i] = Math.exp(Math.log(2) + (n > 1 ? (Math.log(SLOWEST[slowest]) - Math.log(2)) * i / (n - 1) : 0)); return ts; }
function timescalesOf(cortices) { const ts = new Float64Array(brainH(cortices)); let row = 0; for (const c of cortices) { ts.set(blockTimescales(c.channels, c.slowest), row); row += c.channels; } return ts; }
function cortexStart(cortices, k) { let s = 0; for (let i = 0; i < k; i++) s += cortices[i].channels; return s; }
function cortexOfRow(cortices, row) { let s = 0; for (let k = 0; k < cortices.length; k++) { s += cortices[k].channels; if (row < s) return k; } return cortices.length - 1; }
function defaultBrain() { // the founders: a motor cortex of eight fast channels on the scent and the clock, a context cortex on everything; one model cortex on everything
  return { policy: [{ mask: gbit('chemo') | gbit('clock'), channels: 8, slowest: 0 }, { mask: ALL_MASK, channels: 8, slowest: 2 }], model: [{ mask: ALL_MASK, channels: 16, slowest: 2 }] }; }
function cloneBrain(br) { return { policy: br.policy.map(c => Object.assign({}, c)), model: br.model.map(c => Object.assign({}, c)) }; }
function applyMask(th, cortices) { const m = inputMask(cortices); for (let i = 0; i < m.length; i++) if (!m[i]) { th.B[i] = 0; th.G[i] = 0; } }
const wrap = (v, m) => (v >= 0 && v < m) ? v : ((v % m) + m) % m, ddx = (a, b, m) => { let d = b - a; if (d > m / 2) d -= m; else if (d < -m / 2) d += m; return d; };
function gauss(rng) { return rng.normals(1)[0]; }

// blobs on the torus for rock and mud
function blobs(rng, fraction, count, maxLen) {
  const out = new Uint8Array(N); let filled = 0; const target = fraction * N;
  for (let b = 0; b < count && filled < target; b++) {
    let x = (rng.random() * CFG.w) | 0, y = (rng.random() * CFG.h) | 0; const len = 6 + (rng.random() * (maxLen || 40)) | 0;
    for (let s = 0; s < len; s++) {
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) { const i = ((y + dy + CFG.h) % CFG.h) * CFG.w + (x + dx + CFG.w) % CFG.w; if (!out[i]) { out[i] = 1; filled++; } }
      const [dx, dy] = DIRS[(rng.random() * 4) | 0]; x = (x + dx + CFG.w) % CFG.w; y = (y + dy + CFG.h) % CFG.h;
    }
  }
  return out;
}

// ---------- the substrate: mass in soil and food, one sun, rock, mud ----------
class Substrate {
  constructor(seed, rules, cfg) {
    this.rules = Object.assign({}, RULES, rules || {}); this.cfg = Object.assign({}, CFG, cfg || {}); this.rng = new Mulberry(seed); this.tick = 0;
    this.rock = this.rules.rock ? blobs(this.rng, this.cfg.rockFraction, 200) : new Uint8Array(N);
    this.mud = this.rules.mud ? blobs(this.rng, this.cfg.mudFraction, 120, 60) : new Uint8Array(N);
    this.keep = new Float64Array(N); for (let i = 0; i < N; i++) this.keep[i] = this.mud[i] ? this.cfg.mudKeep : 1;
    this.soil = new Int32Array(N); for (let i = 0; i < N; i++) this.soil[i] = this.rock[i] ? 0 : this.cfg.soil;
    this.food = new Int32Array(N); this.lightRow = new Float64Array(CFG.w); this.contrast = 0; this.peak = 0;
    this.scent = new Float64Array(N); this._tmp = new Float64Array(N);
    for (let t = 0; t < this.cfg.warmup; t++) this.step(); this.tick = 0;
  }
  smell() { // the scent of food: the food field blurred over the torus, what a head can smell from a few cells away
    const a = this.scent, b = this._tmp, w = CFG.w, h = CFG.h, r = 2, k = 1 / (2 * r + 1);
    for (let i = 0; i < N; i++) a[i] = this.food[i];
    for (let it = 0; it < this.cfg.scentPasses; it++) {
      for (let y = 0; y < h; y++) { const o = y * w; let s = 0; for (let d = -r; d <= r; d++) s += a[o + ((d + w) % w)]; for (let x = 0; x < w; x++) { b[o + x] = s * k; s += a[o + ((x + r + 1) % w)] - a[o + ((x - r + w) % w)]; } }
      for (let x = 0; x < w; x++) { let s = 0; for (let d = -r; d <= r; d++) s += b[((d + h) % h) * w + x]; for (let y = 0; y < h; y++) { a[y * w + x] = s * k; s += b[((y + r + 1) % h) * w + x] - b[((y - r + h) % h) * w + x]; } }
    }
  }
  scentAt(x, y) { return this.scent[this.cell(x, y)]; }
  cell(x, y) { const xi = x | 0, yi = y | 0; return ((yi >= 0 && yi < CFG.h) ? yi : (((yi % CFG.h) + CFG.h) % CFG.h)) * CFG.w + ((xi >= 0 && xi < CFG.w) ? xi : (((xi % CFG.w) + CFG.w) % CFG.w)); }
  light() {
    const t = this.tick, c = CFG.contrastMin + (1 - CFG.contrastMin) * 0.5 * (1 - Math.cos(2 * Math.PI * t / CFG.season));
    this.contrast = c; this.peak = ((t / CFG.day) % 1) * CFG.w;
    for (let x = 0; x < CFG.w; x++) this.lightRow[x] = 0.5 + 0.5 * c * Math.cos(2 * Math.PI * (x / CFG.w - t / CFG.day));
  }
  step() {
    this.light();
    const { soil, food, rng, rock } = this, cfg = this.cfg;
    for (let i = 0; i < N; i++) { // growth: fertility scales with the soil up to four units; one draw per cell
      const u = rng.random(); if (rock[i] || soil[i] <= 0 || food[i] >= cfg.foodCap) continue;
      if (u < cfg.grow * this.lightRow[i % CFG.w] * Math.min(soil[i], 4) / 4) { soil[i]--; food[i]++; }
    }
    for (let i = 0; i < N; i++) { const u = rng.random(); if (u < cfg.decay && food[i] > 0) { food[i]--; soil[i]++; } }
    for (let i = 0; i < N; i++) { // diffusion of soil to a random neighbour
      const u = rng.random(); if (u >= cfg.diffuse || soil[i] <= 0) continue; const d = Math.min((rng.random() * 4) | 0, 3);
      const x = i % CFG.w, y = (i / CFG.w) | 0; const j = ((y + DIRS[d][1] + CFG.h) % CFG.h) * CFG.w + (x + DIRS[d][0] + CFG.w) % CFG.w; if (rock[j]) continue; soil[i]--; soil[j]++;
    }
    this.smell(); this.tick++;
  }
  dark(i) { return this.rules.night && this.lightRow[i % CFG.w] < this.cfg.nightBelow; }
  get mass() { let m = 0; for (let i = 0; i < N; i++) m += this.soil[i] + this.food[i]; return m; }
}

// ---------- the genome: a body graph, a clock, life genes, and the slow parameters of two patches ----------
const GENE = { freq: [0.12, 0.18, 0.25, 0.35, 0.5], splitAt: [24, 32, 44, 60, 80], noise: [0, 0.03, 0.08, 0.15], horizon: [0, 1, 2, 4], cells: [0, 128, 256, 512, 1024],
  slowRate: [0, 0.3, 1, 3], recordRate: [0.2, 0.5, 1], planRadius: [0.25, 0.5, 1], dFood: [0, 0.25, 0.5, 1], dEnergy: [0, 0.25, 0.5, 1], dPain: [0, 0.25, 0.5, 1], dSpeed: [0, 0.25, 0.5, 1], sleep: [0, 1] };
const G = (g, k) => GENE[k][g[k]];
const P = { seed: 0 };
function randomTheta(port, cortices, seed, cScale) { // the library's initialization at the cortices' width, masked to what each cortex reads, the readout scaled down
  const p = new RecordPatch(Object.assign({}, port, { hidden: brainH(cortices), seed, cells: 0, timescales: timescalesOf(cortices) })).parameters(); for (let i = 0; i < p.C.length; i++) p.C[i] *= cScale; applyMask(p, cortices); return p; }
// theta surgery: channels inserted or deleted at a row, keeping every other weight where it was
function rowsInsert(th, port, at, count, rng, units, ts) {
  const I = port.inputs, O = port.outputs, H0 = th.b.length, H1 = H0 + count;
  const B = new Float64Array(H1 * I), Gm = new Float64Array(H1 * I), b = new Float64Array(H1), g = new Float64Array(H1), C = new Float64Array(O * H1);
  for (let i = 0; i < H0; i++) { const ni = i < at ? i : i + count; B.set(th.B.subarray(i * I, (i + 1) * I), ni * I); Gm.set(th.G.subarray(i * I, (i + 1) * I), ni * I); b[ni] = th.b[i]; g[ni] = th.g[i]; for (let o = 0; o < O; o++) C[o * H1 + ni] = th.C[o * H0 + i]; }
  for (let k = 0; k < count; k++) { const ni = at + k; for (let j = 0; j < I; j++) B[ni * I + j] = units[j] ? 0.3 * gauss(rng) / Math.sqrt(I) : 0; const ret = 1 - 1 / ts[k]; g[ni] = Math.log(ret / (1 - ret)); }
  th.B = B; th.G = Gm; th.b = b; th.g = g; th.C = C;
}
function rowsDelete(th, port, at, count) {
  const I = port.inputs, O = port.outputs, H0 = th.b.length, H1 = H0 - count;
  const B = new Float64Array(H1 * I), Gm = new Float64Array(H1 * I), b = new Float64Array(H1), g = new Float64Array(H1), C = new Float64Array(O * H1);
  for (let i = 0, ni = 0; i < H0; i++) { if (i >= at && i < at + count) continue; B.set(th.B.subarray(i * I, (i + 1) * I), ni * I); Gm.set(th.G.subarray(i * I, (i + 1) * I), ni * I); b[ni] = th.b[i]; g[ni] = th.g[i]; for (let o = 0; o < O; o++) C[o * H1 + ni] = th.C[o * H0 + i]; ni++; }
  th.B = B; th.G = Gm; th.b = b; th.g = g; th.C = C;
}
function mutateBrain(c, rng, r) { // structure: a mask bit flipped, a cortex added, removed, resized or split; the readout of new channels starts at zero
  for (const name of ['policy', 'model']) {
    const cx = c.brain[name], th = c.theta[name], port = PORT[name];
    if (rng.random() < r.mask) { const k = (rng.random() * cx.length) | 0, bit = 1 << ((rng.random() * RG_NAMES.length) | 0), m = cx[k].mask ^ bit; if (m) { const was = maskUnits(cx[k].mask), now = maskUnits(m), s0 = cortexStart(cx, k); cx[k].mask = m;
      for (let i = s0; i < s0 + cx[k].channels; i++) for (let j = 0; j < NI; j++) { if (now[j] && !was[j]) th.B[i * NI + j] = 0.3 * gauss(rng) / Math.sqrt(NI); else if (!now[j]) { th.B[i * NI + j] = 0; th.G[i * NI + j] = 0; } } } }
    if (rng.random() < r.cortexAdd && brainH(cx) + 4 <= MAXH) { const src = cx[(rng.random() * cx.length) | 0], bit = 1 << ((rng.random() * RG_NAMES.length) | 0), mask = (src.mask ^ bit) || src.mask, channels = rng.random() < 0.5 ? 4 : 8, slowest = (rng.random() * SLOWEST.length) | 0;
      rowsInsert(th, port, brainH(cx), channels, rng, maskUnits(mask), blockTimescales(channels, slowest)); cx.push({ mask, channels, slowest }); }
    if (rng.random() < r.cortexRemove && cx.length > 1) { const k = (rng.random() * cx.length) | 0; rowsDelete(th, port, cortexStart(cx, k), cx[k].channels); cx.splice(k, 1); }
    if (rng.random() < r.cortexResize) { const k = (rng.random() * cx.length) | 0, c0 = cx[k], grow = rng.random() < 0.5; if (grow && brainH(cx) + 2 <= MAXH) { rowsInsert(th, port, cortexStart(cx, k) + c0.channels, 2, rng, maskUnits(c0.mask), blockTimescales(2, c0.slowest)); c0.channels += 2; } else if (!grow && c0.channels > 2) { rowsDelete(th, port, cortexStart(cx, k) + c0.channels - 2, 2); c0.channels -= 2; } }
    if (rng.random() < r.cortexSplit) { const k = (rng.random() * cx.length) | 0, c0 = cx[k], bits = RG_NAMES.map((n, i) => 1 << i).filter(b => c0.mask & b); if (c0.channels >= 4 && bits.length >= 2) {
      let ma = 0, mb = 0; for (const b of bits) { if (rng.random() < 0.5) ma |= b; else mb |= b; } if (!ma) { ma = bits[0]; mb &= ~ma; } if (!mb) { mb = bits[bits.length - 1]; ma &= ~mb; }
      const half = c0.channels >> 1, s0 = cortexStart(cx, k); cx.splice(k, 1, { mask: ma, channels: half, slowest: c0.slowest }, { mask: mb, channels: c0.channels - half, slowest: c0.slowest });
      const ua = maskUnits(ma), ub = maskUnits(mb); for (let i = s0; i < s0 + c0.channels; i++) { const u = i < s0 + half ? ua : ub; for (let j = 0; j < NI; j++) if (!u[j]) { th.B[i * NI + j] = 0; th.G[i * NI + j] = 0; } } } }
  }
}
function duplicateChannel(c, srcPort, dstPort, rng, mirror) { // a new muscle's drive: the channel that drives the source port is duplicated with a shifted clock phase (or mirrored), and the new port reads the copy
  const th = c.theta.policy, cx = c.brain.policy, H = th.b.length, I = NI; if (brainH(cx) + 1 > MAXH) return;
  let best = 0, bv = -1; for (let i = 0; i < H; i++) { const v = Math.abs(th.C[srcPort * H + i]); if (v > bv) { bv = v; best = i; } }
  const k = cortexOfRow(cx, best), at = best + 1; rowsInsert(th, PORT.policy, at, 1, rng, maskUnits(cx[k].mask), blockTimescales(1, cx[k].slowest)); cx[k].channels++;
  const H1 = H + 1; for (let j = 0; j < I; j++) { th.B[at * I + j] = th.B[best * I + j]; th.G[at * I + j] = th.G[best * I + j]; } th.b[at] = th.b[best]; th.g[at] = th.g[best];
  if (mirror) th.B[at * I + S['food L-R']] = -th.B[best * I + S['food L-R']]; // a bilateral pair steers by the sign of the scent's side
  else { const ang = rng.random() * 2 * Math.PI, cs = Math.cos(ang), sn = Math.sin(ang), s0 = th.B[best * I + CLOCK], c0 = th.B[best * I + CLOCK + 1]; th.B[at * I + CLOCK] = s0 * cs - c0 * sn; th.B[at * I + CLOCK + 1] = s0 * sn + c0 * cs; }
  for (let i = 0; i < H1; i++) th.C[dstPort * H1 + i] = 0; th.C[dstPort * H1 + at] = th.C[srcPort * H1 + best]; th.c[dstPort] = th.c[srcPort];
}
function copyTheta(t) { const c = {}; for (const k in t) c[k] = Float64Array.from(t[k]); return c; }
function founderGenome(rng) { // a three-node chain with two longitudinal muscles and a ratchet grip; random slow weights (the gait lab sets the motor rows)
  const g = { hue: rng.random(),
    nodes: [{ x: 0, y: 0, along: 0.9, side: 0.1, back: 0.15 }, { x: -1, y: 0, along: 0.9, side: 0.1, back: 0.15 }, { x: -2, y: 0, along: 0.9, side: 0.1, back: 0.15 }],
    springs: [{ a: 0, b: 1, muscle: 1, port: 0 }, { a: 1, b: 2, muscle: 1, port: 1 }],
    freq: 2, splitAt: 1, noise: 1, horizon: 0, cells: 2, slowRate: 1, recordRate: 1, planRadius: 1, dFood: 3, dEnergy: 2, dPain: 2, dSpeed: 1,
    brain: defaultBrain(), theta: null };
  g.theta = { policy: randomTheta(PORT.policy, g.brain.policy, (rng.random() * 4294967296) >>> 0, 0.3), model: randomTheta(PORT.model, g.brain.model, (rng.random() * 4294967296) >>> 0, 0.1) }; return g;
}
function cloneGenome(g) { const c = { hue: g.hue, nodes: g.nodes.map(n => Object.assign({}, n)), springs: g.springs.map(s => Object.assign({}, s)), sleep: g.sleep === undefined ? 0 : g.sleep, freq: g.freq, splitAt: g.splitAt, noise: g.noise, horizon: g.horizon, cells: g.cells, slowRate: g.slowRate, recordRate: g.recordRate, planRadius: g.planRadius, dFood: g.dFood, dEnergy: g.dEnergy, dPain: g.dPain, dSpeed: g.dSpeed, brain: g.brain ? cloneBrain(g.brain) : defaultBrain(), theta: { policy: copyTheta(g.theta.policy), model: copyTheta(g.theta.model) } };
  if (!g.brain) { applyMask(c.theta.policy, c.brain.policy); applyMask(c.theta.model, c.brain.model); } return c; } // a genome without cortices gets the founders' layout
function stepGene(v, n, rng) { return Math.max(0, Math.min(n - 1, v + (rng.random() < 0.5 ? -1 : 1))); }
function freePort(g) { const used = new Set(g.springs.filter(s => s.muscle).map(s => s.port)); for (let p = 0; p < MAXM; p++) if (!used.has(p)) return p; return -1; }
function degree(g, i) { let d = 0; for (const s of g.springs) if (s.a === i || s.b === i) d++; return d; }
function mutateTheta(t, rng, p, sigma) { for (const k in t) { const a = t[k], s = (k === 'B' || k === 'C') ? sigma : sigma * 0.5; for (let i = 0; i < a.length; i++) if (rng.random() < p) a[i] += s * gauss(rng); } }
function mutate(g, rng, rates) {
  const c = cloneGenome(g), r = Object.assign({ gene: 0.1, body: 0.2, add: 0.12, remove: 0.06, mirror: 0.05, muscle: 0.1, spring: 0.06, grip: 0.12, theta: 0.02, sigma: 0.1, mask: 0.05, cortexAdd: 0.05, cortexRemove: 0.04, cortexResize: 0.05, cortexSplit: 0.03 }, rates || {});
  for (const k of Object.keys(GENE)) if (rng.random() < r.gene) c[k] = stepGene(c[k], GENE[k].length, rng);
  for (let i = 1; i < c.nodes.length; i++) if (rng.random() < r.body) { c.nodes[i].x += 0.2 * gauss(rng); c.nodes[i].y += 0.2 * gauss(rng); }
  for (const n of c.nodes) if (rng.random() < r.grip) { for (const k of ['along', 'side', 'back']) n[k] = Math.min(1, Math.max(0.02, n[k] * Math.exp(0.25 * gauss(rng)))); }
  if (rng.random() < r.add && c.nodes.length < MAXN) { // a new node beside an existing one, joined to it and, half the time, to another neighbour: a triangle
    const j = (rng.random() * c.nodes.length) | 0, base = c.nodes[j], ang = rng.random() * 2 * Math.PI, len = 0.7 + 0.6 * rng.random();
    const node = { x: base.x + Math.cos(ang) * len, y: base.y + Math.sin(ang) * len, along: base.along, side: base.side, back: base.back }; c.nodes.push(node); const i = c.nodes.length - 1;
    const muscle = rng.random() < 0.5 ? 1 : 0, port = muscle ? freePort(c) : -1; c.springs.push({ a: j, b: i, muscle: port >= 0 ? 1 : 0, port });
    if (rng.random() < 0.5) { let best = -1, bd = 1e9; for (let k = 0; k < i; k++) { if (k === j) continue; const d = Math.hypot(c.nodes[k].x - node.x, c.nodes[k].y - node.y); if (d < bd && d < 2.2) { bd = d; best = k; } } if (best >= 0) c.springs.push({ a: best, b: i, muscle: 0, port: -1 }); }
    if (port >= 0 && rng.random() < 0.5) { // the new muscle copies another muscle's drive with a shifted phase: a duplicated segment
      const others = c.springs.filter(s => s.muscle && s.port !== port); if (others.length) { const src = others[(rng.random() * others.length) | 0].port; duplicateChannel(c, src, port, rng, false); copyStrainColumn(c.theta, src, port); }
    }
  }
  if (rng.random() < r.mirror && c.nodes.length < MAXN) { // a limb mirrored across the body's axis: the node, its springs to mirrored neighbours, its muscles with mirrored drive
    const side = []; for (let i = 1; i < c.nodes.length; i++) if (Math.abs(c.nodes[i].y) > 0.25) side.push(i);
    if (side.length) { const i = side[(rng.random() * side.length) | 0], src = c.nodes[i], mx = src.x, my = -src.y;
      const twin = k => { let b = -1, bd = 0.3; for (let j = 0; j < c.nodes.length; j++) { const d = Math.hypot(c.nodes[j].x - c.nodes[k].x, c.nodes[j].y + c.nodes[k].y); if (d < bd) { bd = d; b = j; } } return b; };
      if (twin(i) < 0) { c.nodes.push({ x: mx, y: my, along: src.along, side: src.side, back: src.back }); const ni = c.nodes.length - 1;
        for (const sp of c.springs.slice()) { const o = sp.a === i ? sp.b : sp.b === i ? sp.a : -1; if (o < 0) continue; const tw = twin(o), to = tw >= 0 ? tw : o; if (to === ni) continue;
          let port = -1; if (sp.muscle) { port = freePort(c); if (port >= 0) { duplicateChannel(c, sp.port, port, rng, true); copyStrainColumn(c.theta, sp.port, port); } }
          c.springs.push({ a: Math.min(to, ni), b: Math.max(to, ni), muscle: port >= 0 ? 1 : 0, port }); } } }
  }
  if (rng.random() < r.remove && c.nodes.length > 2) { // a leaf goes, with its springs; indices above it shift down
    const leaves = []; for (let i = 1; i < c.nodes.length; i++) if (degree(c, i) === 1) leaves.push(i);
    if (leaves.length) { const i = leaves[(rng.random() * leaves.length) | 0]; c.nodes.splice(i, 1); c.springs = c.springs.filter(s => s.a !== i && s.b !== i).map(s => ({ a: s.a > i ? s.a - 1 : s.a, b: s.b > i ? s.b - 1 : s.b, muscle: s.muscle, port: s.port })); }
  }
  if (rng.random() < r.muscle) { const s = c.springs[(rng.random() * c.springs.length) | 0]; if (s.muscle) { s.muscle = 0; s.port = -1; } else { const p = freePort(c); if (p >= 0) { s.muscle = 1; s.port = p; } } }
  if (rng.random() < r.spring && c.nodes.length > 2) { // a cross spring between two close unconnected nodes
    const i = (rng.random() * c.nodes.length) | 0, j = (rng.random() * c.nodes.length) | 0;
    if (i !== j && !c.springs.some(s => (s.a === i && s.b === j) || (s.a === j && s.b === i)) && Math.hypot(c.nodes[i].x - c.nodes[j].x, c.nodes[i].y - c.nodes[j].y) < 2.5) c.springs.push({ a: Math.min(i, j), b: Math.max(i, j), muscle: 0, port: -1 });
  }
  mutateBrain(c, rng, r);
  mutateTheta(c.theta.policy, rng, r.theta, r.sigma); mutateTheta(c.theta.model, rng, r.theta, r.sigma); applyMask(c.theta.policy, c.brain.policy); applyMask(c.theta.model, c.brain.model);
  c.hue = (((g.hue + (rng.random() - 0.5) * 0.02) % 1) + 1) % 1;
  return c;
}
function copyStrainColumn(theta, src, dst) { for (const t of [theta.policy, theta.model]) { const H = t.b.length; for (let i = 0; i < H; i++) { t.B[i * NI + NS + dst] = t.B[i * NI + NS + src]; t.G[i * NI + NS + dst] = t.G[i * NI + NS + src]; } } }
function genomeMuscles(g) { return g.springs.filter(s => s.muscle).length; }

// ---------- a being: a soft body with a two-patch brain ----------
class Being {
  constructor(pop, id, g, x, y, heading, energy, lineage, parent) {
    const rules = pop.rules, rng = pop.rng, cfg = pop.sub.cfg;
    this.id = id; this.g = g; this.lineage = lineage === null ? id : lineage; this.parent = parent; this.born = pop.sub.tick; this.age = 0; this.hue = g.hue;
    this.energy = energy; this.reward = 0; this.pain = 0; this.eats = 0; this.bites = 0; this.bitten = 0; this.kills = 0; this.dist = 0; this.speed = 0; this.turn = 0; this.slot = -1; this.dead = false;
    this.plans = 0; this.planAccepted = 0; this.planConfirmed = 0; this.replays = 0; this.repairPending = false; this.repairPredicted = 0; this.errEarly = 0; this.errLate = 0; this.nEarly = 0; this.nLate = 0; this.effort = 0;
    const nn = g.nodes.length; this.nGenome = nn; this.n = 0; this.heading = heading;
    this.x = new Float64Array(MAXN); this.y = new Float64Array(MAXN); this.px = new Float64Array(MAXN); this.py = new Float64Array(MAXN); this.fx = new Float64Array(MAXN); this.fy = new Float64Array(MAXN);
    this.along = new Float64Array(MAXN); this.side = new Float64Array(MAXN); this.back = new Float64Array(MAXN); this.axisOf = new Int32Array(MAXN).fill(-1);
    const dev = rules.devnoise ? (cfg.devNoise === undefined ? 0.08 : cfg.devNoise) : 0; this.dev = dev;
    for (let i = 0; i < nn; i++) { const nd = g.nodes[i], j = dev ? Math.exp(dev * gauss(rng)) : 1; this.along[i] = Math.min(1, nd.along * j); this.side[i] = Math.min(1, nd.side * j); this.back[i] = Math.min(1, nd.back * j); }
    const ns = g.springs.length; this.nS = ns; this.sa = new Int32Array(ns); this.sb = new Int32Array(ns); this.rest0 = new Float64Array(ns); this.rest = new Float64Array(ns); this.muscle = new Int8Array(ns); this.port = new Int8Array(ns); this.len = new Float64Array(ns);
    for (let s = 0; s < ns; s++) { const sp = g.springs[s]; this.sa[s] = sp.a; this.sb[s] = sp.b; const a = g.nodes[sp.a], b = g.nodes[sp.b]; this.rest0[s] = Math.max(0.35, Math.hypot(a.x - b.x, a.y - b.y) * (dev ? Math.exp(dev * gauss(rng)) : 1)); this.rest[s] = this.rest0[s]; this.muscle[s] = sp.muscle; this.port[s] = sp.port; }
    for (let i = 0; i < nn; i++) { let best = 1e9; for (let s = 0; s < ns; s++) { const o = this.sa[s] === i ? this.sb[s] : this.sb[s] === i ? this.sa[s] : -1; if (o >= 0 && o < best) best = o; } this.axisOf[i] = best === 1e9 ? -1 : best; }
    this.act = new Float64Array(MAXM); this.cmd = new Float64Array(NM); this.prevCmd = new Float64Array(NM); this.prop = new Float64Array(NM);
    this.sense = new Float64Array(SN); this.prevSense = new Float64Array(SN); this.predNext = new Float64Array(SN); this.pu = new Float64Array(NI); this.mu = new Float64Array(NI); this.pred = new Float64Array(NOUT); this.target = new Float64Array(NOUT); this.known = new Float64Array(NOUT);
    this.phase = rng.random() * 2 * Math.PI; this.aheadSlot = -1; this.prevAhead = 0;
    const cells = rules.records ? G(g, 'cells') : 0, seed = (rng.random() * 4294967296) >>> 0;
    // the policy keeps no records: a repair learned one-shot became the next proposal and the gait drifted (LEARNINGS 12); its slow weights learn confirmed repairs
    this.policy = new RecordPatch(Object.assign({}, PORT.policy, { hidden: brainH(g.brain.policy), timescales: timescalesOf(g.brain.policy), seed, cells: cfg.policyCells ? cells : 0, active: ACTIVE, fanin: FANIN, recordRate: G(g, 'recordRate'), window: cfg.window, blank: true })); this.policy.setParameters(g.theta.policy); this.policy.setInputMask(inputMask(g.brain.policy));
    this.model = new RecordPatch(Object.assign({}, PORT.model, { hidden: brainH(g.brain.model), timescales: timescalesOf(g.brain.model), seed: seed ^ 0x9e3779b9, cells, active: ACTIVE, fanin: FANIN, recordRate: G(g, 'recordRate'), window: cfg.window, blank: true })); this.model.setParameters(g.theta.model); this.model.setInputMask(inputMask(g.brain.model));
    this.slowRate = rules.slow ? G(g, 'slowRate') : 0; this.horizon = rules.plan ? G(g, 'horizon') : 0; this.noise = rules.noise ? G(g, 'noise') : 0;
    this.sleeper = !!(rules.sleep && G(g, 'sleep') && this.slowRate > 0); this.sleeping = false; this.dayU = []; this.dayK = 0; this.dreams = 0; this.sleptTicks = 0; this.dayDream = null;
    this.weights = new Float64Array(NOUT); this.goal = new Float64Array(NOUT); // the drives: what the planner repairs toward
    // the model predicts the change of each reading and the energy change; the drives ask for readings to rise or fall
    this.weights[S['food ahead']] = G(g, 'dFood'); this.goal[S['food ahead']] = 1; this.weights[S['food here']] = G(g, 'dFood'); this.goal[S['food here']] = 1;
    this.weights[S.pain] = G(g, 'dPain'); this.goal[S.pain] = -1; this.weights[S.speed] = G(g, 'dSpeed'); this.goal[S.speed] = 1; this.weights[SN] = G(g, 'dEnergy'); this.goal[SN] = 1;
    this.drive = 0; for (const w of this.weights) this.drive += w;
    if (this.horizon > 0) { const T = this.horizon; this.planU = new Float64Array(T * NI); this.planGoal = new Float64Array(T * NOUT); for (let t = 0; t < T; t++) this.planGoal.set(this.goal, t * NOUT); this.lo = new Float64Array(NI); this.hi = new Float64Array(NI); }
    this.place(x, y, heading, rules.growth ? Math.min(cfg.birthNodes, nn) : nn);
    this.readUnits = g.brain.policy.reduce((s, c) => s + maskCount(c.mask), 0) + g.brain.model.reduce((s, c) => s + maskCount(c.mask), 0); this.channels = this.policy.H + this.model.H;
    this.brainPrice = POP.read * this.readUnits + POP.channel * this.channels + POP.cell * (this.policy.D + this.model.D) + (this.slowRate > 0 ? POP.slow * (this.policy.slowCount + (this.sleeper ? 0 : this.model.slowCount)) : 0) + (cells > 0 ? POP.write * 2 * ACTIVE : 0);
  }
  place(hx, hy, heading, n) { // the first n nodes of the genome in the body frame at the head, facing `heading`
    const cs = Math.cos(heading), sn = Math.sin(heading); this.n = n;
    for (let i = 0; i < n; i++) { const nd = this.g.nodes[i]; this.x[i] = wrap(hx + nd.x * cs - nd.y * sn, CFG.w); this.y[i] = wrap(hy + nd.x * sn + nd.y * cs, CFG.h); }
    this.px.set(this.x); this.py.set(this.y); this.axes();
  }
  grow() { // one more node of the genome, placed in the current head frame
    const i = this.n, nd = this.g.nodes[i], cs = Math.cos(this.heading), sn = Math.sin(this.heading);
    this.x[i] = wrap(this.x[0] + nd.x * cs - nd.y * sn, CFG.w); this.y[i] = wrap(this.y[0] + nd.x * sn + nd.y * cs, CFG.h); this.px[i] = this.x[i]; this.py[i] = this.y[i]; this.n++; this.axes();
  }
  shrink() { this.n--; const s = this.n; for (let k = 0; k < this.nS; k++) if (this.port[k] >= 0 && (this.sa[k] === s || this.sb[k] === s)) this.act[this.port[k]] = 0; this.axes(); }
  knownPath(T) { const O = this.known.length; if (!this._kp || this._kp.length !== T * O) { this._kp = new Float64Array(T * O); for (let t = 0; t < T; t++) this._kp.set(this.known, t * O); } return this._kp; }
  hasMuscle(port) { for (let s = 0; s < this.nS; s++) if (this.port[s] === port && this.sa[s] < this.n && this.sb[s] < this.n) return s; return -1; }
  axes() { // each node's head-ward direction: the head from the mean of its neighbours, every other node toward its lowest-index neighbour
    const n = this.n; let mx = 0, my = 0, c = 0;
    for (let s = 0; s < this.nS; s++) { const a = this.sa[s], b = this.sb[s]; if (a >= n || b >= n) continue; if (a === 0) { mx += ddx(this.x[0], this.x[b], CFG.w); my += ddx(this.y[0], this.y[b], CFG.h); c++; } else if (b === 0) { mx += ddx(this.x[0], this.x[a], CFG.w); my += ddx(this.y[0], this.y[a], CFG.h); c++; } }
    if (c) { const l = Math.hypot(mx, my) || 1; this.fx[0] = -mx / l; this.fy[0] = -my / l; } else { this.fx[0] = Math.cos(this.heading); this.fy[0] = Math.sin(this.heading); }
    for (let i = 1; i < n; i++) { const j = this.axisOf[i]; if (j < 0 || j >= n) { this.fx[i] = this.fx[0]; this.fy[i] = this.fy[0]; continue; } const dx = ddx(this.x[i], this.x[j], CFG.w), dy = ddx(this.y[i], this.y[j], CFG.h), l = Math.hypot(dx, dy) || 1; this.fx[i] = dx / l; this.fy[i] = dy / l; }
    this.heading = Math.atan2(this.fy[0], this.fx[0]);
  }
  get body() { return this.n; }
}

// ---------- the population ----------
class Population {
  constructor(sub, seed, founder, opts) {
    this.sub = sub; this.rules = sub.rules; this.cfg = sub.cfg; this.rng = new Mulberry(seed); this.nextId = 0; this.beings = []; this.births = 0; this.deaths = 0; this.kills = 0; this.events = []; this.onDeath = null;
    this.head = new Int32Array(N); this.next = new Int32Array(POP.max * MAXN); this.founderGenome = founder || null; this.mutRates = null;
    const initial = opts && opts.initial !== undefined ? opts.initial : POP.initial;
    this.chronicle = opts && opts.chronicle === false ? null : new Chronicle(this, opts && opts.every, opts && opts.maxEvents);
    for (let k = 0; k < initial; k++) {
      const x = this.rng.random() * CFG.w, y = this.rng.random() * CFG.h; if (sub.rock[sub.cell(x, y)]) continue;
      const g = this.founderGenome ? cloneGenome(this.founderGenome) : founderGenome(this.rng); g.hue = this.rng.random();
      this.spawn(g, x, y, this.rng.random() * 2 * Math.PI, POP.birth, null, null);
    }
  }
  spawn(g, x, y, heading, energy, lineage, parent) {
    const b = new Being(this, this.nextId++, g, x, y, heading, energy, lineage, parent);
    b.energy -= b.n; if (b.energy <= 0) { b.energy = 1; } // the body's nodes hold a unit each, paid from the share
    this.beings.push(b); return b;
  }
  get held() { let m = 0; for (const b of this.beings) m += b.energy + b.n; return m; }
  get mass() { return this.sub.mass + this.held; }
  index() { // every node into the cell grid, for contacts and the ahead sense
    this.head.fill(-1); const s = this.sub;
    this.beings.forEach((b, k) => { b.slot = k; for (let i = 0; i < b.n; i++) { const c = s.cell(b.x[i], b.y[i]), q = k * MAXN + i; this.next[q] = this.head[c]; this.head[c] = q; } });
  }
  nearNode(x, y, radius, exclude) { // the closest node of another being within radius: returns [slot, distance] or null
    const s = this.sub; let best = null, bd = radius;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      let q = this.head[s.cell(x + dx, y + dy)];
      while (q >= 0) { const k = (q / MAXN) | 0, i = q % MAXN; if (k !== exclude) { const b = this.beings[k]; const d = Math.hypot(ddx(x, b.x[i], CFG.w), ddx(y, b.y[i], CFG.h)); if (d < bd) { bd = d; best = k; } } q = this.next[q]; }
    }
    return best === null ? null : [best, bd];
  }
  sense(b) { // what the head can tell, in the reading's order
    const s = this.sub, hx = b.x[0], hy = b.y[0], cs = Math.cos(b.heading), sn = Math.sin(b.heading), sv = b.sense;
    const hc = s.cell(hx, hy), dark = s.dark(hc), ax = hx + cs * this.cfg.aheadAt, ay = hy + sn * this.cfg.aheadAt;
    // the smell of food: sampled to the left and the right of the head, and ahead of it; the nose is the head
    const left = dark ? 0 : s.scentAt(hx + cs * 0.8 - sn * 1.2, hy + sn * 0.8 + cs * 1.2), right = dark ? 0 : s.scentAt(hx + cs * 0.8 + sn * 1.2, hy + sn * 0.8 - cs * 1.2), ahead = dark ? 0 : s.scentAt(hx + cs * 1.8, hy + sn * 1.8);
    sv[0] = Math.tanh((left - right) * 4); sv[1] = Math.tanh(ahead * 1.5); sv[2] = dark ? 0 : Math.min(s.food[hc] / 4, 1); sv[3] = s.rock[s.cell(ax, ay)] ? 1 : 0;
    const near = this.nearNode(ax, ay, 0.9, b.slot); b.aheadSlot = near ? near[0] : -1; sv[4] = near ? 1 : 0;
    sv[5] = Math.min(b.energy / G(b.g, 'splitAt'), 1) * 2 - 1; sv[6] = b.pain > 0 ? 1 : 0; sv[7] = s.lightRow[hc % CFG.w] * 2 - 1; sv[8] = s.keep[hc] * 2 - 1;
    sv[9] = Math.tanh(b.speed / 0.15); sv[10] = Math.tanh(b.turn / 0.15); sv[11] = b.n / b.nGenome * 2 - 1;
    for (let p = 0; p < MAXM; p++) sv[NS + p] = 0;
    for (let k = 0; k < b.nS; k++) if (b.port[k] >= 0 && b.sa[k] < b.n && b.sb[k] < b.n) sv[NS + b.port[k]] = Math.max(-1, Math.min(1, (b.len[k] / b.rest0[k] - 1) / this.cfg.muscleAmp));
  }
  brain(b) { // the two patches: the model learns what followed; the policy proposes; the planner repairs the proposal under the model; the policy learns the repair
    const cfg = this.cfg, rules = this.rules, g = b.g, sv = b.sense, cs = Math.sin(b.phase), cc = Math.cos(b.phase);
    // (1) the model's previous step is taught what actually followed
    if (b.model.pending) { // the target is the change of the reading since the prediction was made, and the energy change
      for (let j = 0; j < SN; j++) b.target[j] = sv[j] - b.prevSense[j]; b.target[SN] = Math.max(-1, Math.min(1, b.reward)); b.known.fill(1); for (let p = 0; p < MAXM; p++) if (b.hasMuscle(p) < 0) b.known[NS + p] = 0;
      let e = 0, c = 0; for (let j = 0; j < SN; j++) if (b.known[j]) { const d = b.pred[j] - b.target[j]; e += d * d; c++; } e = c ? e / c : 0; if (b.age <= 150) { b.errEarly += e; b.nEarly++; } else { b.errLate += e; b.nLate++; } b.lastErr = e;
      if (b.model.teach(b.target, b.known)) {
        if (b.sleeper) { // by day a sleeper writes records only and keeps the window as a cue for the night
          const T = b.model.w, I = b.model.I; if (b.dayU.length < SLEEP.windows) b.dayU.push(Float64Array.from(b.model.wu.subarray(0, T * I))); else { b.dayU[b.dayK] = Float64Array.from(b.model.wu.subarray(0, T * I)); b.dayK = (b.dayK + 1) % SLEEP.windows; }
          b.model.flush(0, { clip: cfg.clip, recode: cfg.recode });
        } else b.model.flush(b.slowRate, { clip: cfg.clip, backtrack: cfg.backtrack, recode: cfg.recode });
      }
    }
    // (2) the policy's previous step is settled by readback: a repair executed last tick is learned only if what actually
    //     followed beat what the model had predicted for the unrepaired proposal; otherwise the step is skipped
    if (b.policy.pending) {
      let learn = false;
      if (b.repairPending) { let actual = 0; for (let j = 0; j < NOUT; j++) actual += b.weights[j] * (b.target[j] - b.goal[j]) ** 2; actual /= NOUT; learn = actual < b.repairPredicted; if (learn) b.planConfirmed++; }
      if (learn) { if (b.policy.teach(b.prevCmd)) b.policy.flush(b.slowRate, { clip: cfg.clip, backtrack: cfg.backtrack, recode: cfg.recode }); } else if (b.policy.skip()) b.policy.flush(b.slowRate, { clip: cfg.clip, backtrack: cfg.backtrack, recode: cfg.recode });
      b.repairPending = false;
    }
    // (3) the policy proposes the motor from the reading, the clock and the previous motor
    const pu = b.pu; pu.set(sv); pu[CLOCK] = cs; pu[CLOCK + 1] = cc; for (let m = 0; m < NM; m++) pu[MOTOR + m] = b.prevCmd[m];
    b.policy.step(pu, b.prop);
    const cmd = b.cmd; for (let m = 0; m < NM; m++) { let v = b.prop[m]; if (b.noise > 0 && m < MAXM) v += b.noise * gauss(this.rng); cmd[m] = Math.max(-1, Math.min(1, v)); }
    // (4) the planner repairs the motor ports of a private path under the model, toward the drives
    if (b.horizon > 0 && b.drive > 0) {
      const T = b.horizon, U = b.planU, r = G(g, 'planRadius'); let ph = b.phase;
      for (let t = 0; t < T; t++) { const o = t * NI; U.set(sv, o); U[o + CLOCK] = Math.sin(ph); U[o + CLOCK + 1] = Math.cos(ph); for (let m = 0; m < NM; m++) U[o + MOTOR + m] = cmd[m]; ph += G(g, 'freq'); }
      for (let j = 0; j < NI; j++) { b.lo[j] = -1; b.hi[j] = 1; } for (let m = 0; m < MAXM; m++) { b.lo[MOTOR + m] = Math.max(-1, cmd[m] - r); b.hi[MOTOR + m] = Math.min(1, cmd[m] + r); }
      const p = b.model.plan(U, T, b.planGoal, b.weights, MOTORS, b.lo, b.hi, { feedback: T > 1 ? FEEDBACK : null, feedbackAdd: true, fixedRead: cfg.fixedRead, rate: 2, maxSteps: 4, maxBacktracks: 4 });
      b.plans++; b.replays += p.replays; b.lastPlan = p;
      if (p.losses.length > 1) { // a repair: executed now, learned only after readback confirms it
        b.planAccepted++; for (let m = 0; m < MAXM; m++) cmd[m] = p.inputs[MOTOR + m];
        let predicted = 0; for (let j = 0; j < NOUT; j++) predicted += b.weights[j] * (p.initialOutput[j] - b.goal[j]) ** 2; b.repairPredicted = predicted / NOUT; b.repairPending = true;
      }
    }
    // (5) the model predicts what this motor will do
    const mu = b.mu; mu.set(sv); mu[CLOCK] = cs; mu[CLOCK + 1] = cc; for (let m = 0; m < NM; m++) mu[MOTOR + m] = cmd[m];
    b.model.step(mu, b.pred); for (let j = 0; j < SN; j++) b.predNext[j] = sv[j] + b.pred[j]; b.prevSense.set(sv);
    b.prevCmd.set(cmd); b.phase += G(g, 'freq'); if (b.phase > 2 * Math.PI) b.phase -= 2 * Math.PI;
  }
  physics(b) { // muscles set rest lengths; springs are projected; friction filters each node's displacement by its grip; rock and other bodies push back
    const cfg = this.cfg, s = this.sub, n = b.n, ns = b.nS, amp = cfg.muscleAmp, x = b.x, y = b.y, px = b.px, py = b.py; let effort = 0;
    for (let k = 0; k < ns; k++) { const p = b.port[k]; if (p >= 0) { const a = b.act[p], c = b.cmd[p]; const na = a + cfg.muscleLag * (c - a); effort += Math.abs(na - a); b.act[p] = na; b.rest[k] = b.rest0[k] * (1 + amp * na); } }
    const hx0 = x[0], hy0 = y[0], head0 = b.heading;
    for (let sub = 0; sub < cfg.substeps; sub++) {
      for (let i = 0; i < n; i++) { px[i] = x[i]; py[i] = y[i]; }
      for (let it = 0; it < cfg.iters; it++) for (let k = 0; k < ns; k++) {
        const a = b.sa[k], c = b.sb[k]; if (a >= n || c >= n) continue;
        const dx = ddx(x[a], x[c], CFG.w), dy = ddx(y[a], y[c], CFG.h), d = Math.hypot(dx, dy) || 1e-6, e = (d - b.rest[k]) / d * 0.5;
        x[a] += dx * e; y[a] += dy * e; x[c] -= dx * e; y[c] -= dy * e;
      }
      if (sub === 0) for (let i = 0; i < n; i++) { // contact with other bodies: push apart
        const near = this.nearNode(x[i], y[i], cfg.contact, b.slot); if (!near) continue; const o = this.beings[near[0]];
        let bi = 0, bd = 1e9; for (let j = 0; j < o.n; j++) { const d = Math.hypot(ddx(x[i], o.x[j], CFG.w), ddx(y[i], o.y[j], CFG.h)); if (d < bd) { bd = d; bi = j; } }
        const dx = ddx(x[i], o.x[bi], CFG.w), dy = ddx(y[i], o.y[bi], CFG.h), d = bd || 1e-6, push = (cfg.contact - d) / d * cfg.contactPush;
        x[i] -= dx * push; y[i] -= dy * push; o.x[bi] += dx * push; o.y[bi] += dy * push;
      }
      for (let i = 0; i < n; i++) { // friction: the displacement kept depends on its direction against the node's head-ward axis and on the ground
        let dx = ddx(px[i], x[i], CFG.w), dy = ddx(py[i], y[i], CFG.h); const fx = b.fx[i], fy = b.fy[i];
        const a = dx * fx + dy * fy, sd = -dx * fy + dy * fx, ka = a >= 0 ? b.along[i] : b.back[i], ground = s.keep[s.cell(px[i], py[i])];
        dx = ground * (ka * a * fx - b.side[i] * sd * fy); dy = ground * (ka * a * fy + b.side[i] * sd * fx);
        let nx = wrap(px[i] + dx, CFG.w), ny = wrap(py[i] + dy, CFG.h);
        if (s.rock[s.cell(nx, ny)]) { nx = px[i]; ny = py[i]; }
        x[i] = nx; y[i] = ny;
      }
    }
    for (let k = 0; k < ns; k++) { const a = b.sa[k], c = b.sb[k]; b.len[k] = (a < n && c < n) ? Math.hypot(ddx(x[a], x[c], CFG.w), ddx(y[a], y[c], CFG.h)) : b.rest0[k]; }
    b.axes();
    const mx = ddx(hx0, x[0], CFG.w), my = ddx(hy0, y[0], CFG.h); b.speed = mx * Math.cos(head0) + my * Math.sin(head0); b.dist += Math.hypot(mx, my);
    let dt = b.heading - head0; if (dt > Math.PI) dt -= 2 * Math.PI; else if (dt < -Math.PI) dt += 2 * Math.PI; b.turn = dt;
    return effort;
  }
  step() {
    const s = this.sub, bs = this.beings, cfg = this.cfg, rules = this.rules; this.index(); this.events.length = 0;
    const n = bs.length, order = new Int32Array(n); for (let i = 0; i < n; i++) order[i] = i;
    for (let i = n - 1; i > 0; i--) { const j = (this.rng.random() * (i + 1)) | 0; const t = order[i]; order[i] = order[j]; order[j] = t; }
    for (let q = 0; q < n; q++) {
      const b = bs[order[q]]; if (b.dead) continue; b.age++;
      this.sense(b);
      const dark = b.sleeper && s.dark(s.cell(b.x[0], b.y[0]));
      if (dark) { // asleep: no motor, no plan, no bite; one dream and one slow step per tick from the day's cues
        if (!b.sleeping) { b.sleeping = true; b.dayDream = 0; }
        b.cmd.fill(0); b.lastPlan = null; b.sleptTicks++;
        if (b.dayU.length && b.slowRate > 0) { const U = b.dayU[b.dayDream % b.dayU.length], T = U.length / b.model.I; b.dayDream++;
          if (!b.dreamOut || b.dreamOut.length < T * b.model.O) b.dreamOut = new Float64Array(T * b.model.O);
          b.model.dream(U, T, b.dreamOut); b.model.consolidate(U, T, b.dreamOut, b.slowRate, { clip: cfg.clip, known: b.knownPath(T) }); b.dreams++; b.replays++; }
      } else {
        if (b.sleeping) { // dawn: every dream is written back against the moved weights, twice; the day's cues are forgotten
          b.sleeping = false; if (b.dayU.length) { for (let pass = 0; pass < 2; pass++) for (const U of b.dayU) { const T = U.length / b.model.I; if (!b.dreamOut || b.dreamOut.length < T * b.model.O) b.dreamOut = new Float64Array(T * b.model.O); b.model.dream(U, T, b.dreamOut); b.model.reference(U, T, b.dreamOut, b.knownPath(T)); } }
          b.dayU.length = 0; b.dayK = 0; // the last awake step's outcome was never seen: the pending steps are skipped, and a full window is learned as it stands
          if (b.model.pending && b.model.skip()) b.model.flush(0, { clip: cfg.clip, recode: cfg.recode }); if (b.policy.pending && b.policy.skip()) b.policy.flush(b.slowRate, { clip: cfg.clip, backtrack: cfg.backtrack, recode: cfg.recode }); }
        this.brain(b);
      }
      const before = b.energy; let bit = false;
      if (rules.bite && b.cmd[MAXM] > 0.5 && b.aheadSlot >= 0) { // the mouth: a bite at the body ahead
        const v = bs[b.aheadSlot]; if (v && !v.dead) {
          bit = true; if (this.rng.random() < cfg.biteLands) {
            let take = 0; if (v.energy > 0) { take = Math.min(cfg.biteTake, v.energy); v.energy -= take; } else if (rules.injury && v.n > 2) { v.shrink(); take = 1; this.events.push({ kind: 'injury', b: v, by: b }); }
            if (take > 0) { b.energy += take; v.pain = 3; v.bitten++; b.bites++; v.lastBiter = b.lineage; this.events.push({ kind: 'bite', b, v, take }); }
          }
        }
      }
      const effort = this.physics(b);
      const hc = s.cell(b.x[0], b.y[0]);
      if (s.food[hc] > 0) { s.food[hc]--; b.energy++; b.eats++; this.events.push({ kind: 'eat', b }); }
      b.pain = Math.max(0, b.pain - 1);
      const price = POP.base + POP.node * b.n + POP.muscle * effort + b.brainPrice + POP.replay * (b.lastPlan ? b.lastPlan.replays : 0) + (bit ? POP.bite : 0)
        + (b.sleeper ? POP.cue * b.dayU.length * cfg.window + (b.sleeping && b.dayU.length ? POP.replay + POP.slow * b.model.slowCount : 0) : 0);
      b.lastPlan = null; b.effort = effort;
      if (this.rng.random() < price && b.energy > 0) { b.energy--; s.soil[hc]++; } // a body already at zero owes nothing more: energy never goes below zero
      b.reward = b.energy - before;
      if (rules.growth && b.n < b.nGenome && b.energy > cfg.growAt) { b.energy--; b.grow(); this.events.push({ kind: 'grow', b }); }
    }
    const survivors = [];
    for (const b of bs) {
      if (b.energy <= 0 || b.n < 2) { this.deaths++; if (b.pain > 0) this.kills++; const hc = s.cell(b.x[0], b.y[0]); s.soil[hc] += Math.max(0, b.energy); for (let i = 0; i < b.n; i++) s.food[s.cell(b.x[i], b.y[i])]++; b.energy = 0; b.n = 0; b.dead = true; b.diedAt = s.tick; this.events.push({ kind: 'die', b, killed: b.pain > 0 }); if (this.onDeath) this.onDeath(b); continue; }
      survivors.push(b);
    }
    this.beings = survivors;
    const children = []; const cand = this.beings.slice(); for (let i = cand.length - 1; i > 0; i--) { const j = (this.rng.random() * (i + 1)) | 0; const x = cand[i]; cand[i] = cand[j]; cand[j] = x; }
    if (rules.split) for (const b of cand) {
      if (b.energy < G(b.g, 'splitAt') || b.n < b.nGenome || this.beings.length + children.length >= POP.max) continue;
      const share = b.energy >> 1; if (share <= cfg.birthNodes + 1) continue; b.energy -= share;
      const tail = b.n - 1, ang = b.heading + Math.PI + (this.rng.random() - 0.5), bx = wrap(b.x[tail] + Math.cos(ang) * 1.2, CFG.w), by = wrap(b.y[tail] + Math.sin(ang) * 1.2, CFG.h);
      const child = this.spawn(mutate(b.g, this.rng, this.mutRates), bx, by, ang, share, b.lineage, b.id); this.beings.pop(); children.push(child); this.births++; this.events.push({ kind: 'birth', b: child, parent: b });
    }
    this.beings.push(...children); this.index();
    if (this.chronicle) this.chronicle.tick(this);
  }
}

// ---------- the chronicle: what every lineage did, every hundred ticks, and the events that decided it ----------
const FRAME_KEYS = ['n', 'births', 'deaths', 'starved', 'killed', 'kills', 'bitesGiven', 'bitesTaken', 'meals', 'speed', 'age', 'energy', 'nodes', 'muscles', 'channels', 'cortices', 'records', 'planning', 'plans', 'confirmed', 'ageAtDeath', 'hue'];
function serializeGenome(g) { const c = cloneGenome(g); const th = { policy: {}, model: {} }; for (const p of ['policy', 'model']) for (const k in c.theta[p]) th[p][k] = Array.from(c.theta[p][k]); c.theta = th; return c; }
function genomeSummary(g) { const s = { genes: {}, nodes: g.nodes.length, muscles: g.springs.filter(x => x.muscle).length, springs: g.springs.length, grips: { along: 0, back: 0, side: 0 }, brain: cloneBrain(g.brain), thetaNorm: {} };
  for (const k of Object.keys(GENE)) s.genes[k] = G(g, k); for (const n of g.nodes) { s.grips.along += n.along / g.nodes.length; s.grips.back += n.back / g.nodes.length; s.grips.side += n.side / g.nodes.length; }
  for (const p of ['policy', 'model']) { let n2 = 0; for (const k of ['B', 'C']) for (const v of g.theta[p][k]) n2 += v * v; s.thetaNorm[p] = Math.sqrt(n2); } return s; }
class Chronicle {
  constructor(pop, every, maxEvents) { this.every = every || 100; this.maxEvents = maxEvents || 40000; this.frames = []; this.events = []; this.samples = {}; this.acc = new Map(); this.founder = pop.founderGenome ? genomeSummary(pop.founderGenome) : null; this.founderFull = pop.founderGenome ? serializeGenome(pop.founderGenome) : null; this.deathAges = new Map(); }
  bucket(lid) { let a = this.acc.get(lid); if (!a) { a = { births: 0, deaths: 0, starved: 0, killed: 0, kills: 0, bitesGiven: 0, bitesTaken: 0, meals: 0 }; this.acc.set(lid, a); } return a; }
  tick(pop) {
    const t = pop.sub.tick;
    for (const e of pop.events) {
      if (e.kind === 'eat') { this.bucket(e.b.lineage).meals++; continue; }
      if (e.kind === 'grow') continue;
      if (e.kind === 'birth') this.bucket(e.b.lineage).births++;
      else if (e.kind === 'die') { const a = this.bucket(e.b.lineage); a.deaths++; if (e.killed) { a.killed++; if (e.b.lastBiter !== undefined) this.bucket(e.b.lastBiter).kills++; } else a.starved++; let d = this.deathAges.get(e.b.lineage); if (!d) { d = []; this.deathAges.set(e.b.lineage, d); } d.push(e.b.age); }
      else if (e.kind === 'bite') { this.bucket(e.b.lineage).bitesGiven++; this.bucket(e.v.lineage).bitesTaken++; }
      const ev = { t, kind: e.kind, id: e.b.id, lineage: e.b.lineage, x: +e.b.x[0].toFixed(1), y: +e.b.y[0].toFixed(1) };
      if (e.kind === 'die') { ev.age = e.b.age; ev.cause = e.killed ? 'killed' : 'starved'; if (e.killed && e.b.lastBiter !== undefined) ev.by = e.b.lastBiter; }
      if (e.kind === 'bite') { ev.victim = e.v.id; ev.victimLineage = e.v.lineage; ev.take = e.take; }
      if (e.kind === 'injury') { ev.by = e.by.id; ev.byLineage = e.by.lineage; }
      if (e.kind === 'birth') { ev.parent = e.parent.id; ev.nodes = e.b.nGenome; }
      this.events.push(ev);
    }
    if (this.events.length > this.maxEvents) this.events.splice(0, this.events.length - this.maxEvents);
    if (t % this.every === 0) this.frame(pop);
    if (t % (this.every * 10) === 0) this.sample(pop);
  }
  frame(pop) {
    const t = pop.sub.tick, by = new Map();
    for (const b of pop.beings) { let g = by.get(b.lineage); if (!g) { g = []; by.set(b.lineage, g); } g.push(b); }
    const lineages = {}; const all = new Set([...by.keys(), ...this.acc.keys()]);
    for (const lid of all) {
      const bs = by.get(lid) || [], a = this.acc.get(lid) || { births: 0, deaths: 0, starved: 0, killed: 0, kills: 0, bitesGiven: 0, bitesTaken: 0, meals: 0 }, n = bs.length, m = f => n ? bs.reduce((s, b) => s + f(b), 0) / n : 0;
      const ages = this.deathAges.get(lid) || [];
      lineages[lid] = [n, a.births, a.deaths, a.starved, a.killed, a.kills, a.bitesGiven, a.bitesTaken, a.meals, +m(b => Math.abs(b.speed)).toFixed(4), +m(b => b.age).toFixed(1), +m(b => b.energy).toFixed(2), +m(b => b.nGenome).toFixed(2), +m(b => b.g.springs.filter(x => x.muscle).length).toFixed(2), +m(b => b.policy.H + b.model.H).toFixed(1), +m(b => b.g.brain.policy.length + b.g.brain.model.length).toFixed(2), +m(b => b.model.D > 0 ? 1 : 0).toFixed(2), +m(b => b.horizon > 0 ? 1 : 0).toFixed(2), bs.reduce((s, b) => s + b.plans, 0), bs.reduce((s, b) => s + b.planConfirmed, 0), ages.length ? +(ages.reduce((s, v) => s + v, 0) / ages.length).toFixed(1) : null, n ? +bs[0].hue.toFixed(3) : null];
    }
    let food = 0; for (let i = 0; i < N; i++) food += pop.sub.food[i];
    this.frames.push({ t, alive: pop.beings.length, food, lineages }); this.acc.clear(); this.deathAges.clear();
  }
  sample(pop) { // the eldest of each of the ten largest lineages, summarized (no weights)
    const by = new Map(); for (const b of pop.beings) { const e = by.get(b.lineage); if (!e || b.age > e.age) by.set(b.lineage, b); }
    const counts = new Map(); for (const b of pop.beings) counts.set(b.lineage, (counts.get(b.lineage) || 0) + 1);
    const top = [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 10);
    for (const [lid, n] of top) { const b = by.get(lid); if (!this.samples[lid]) this.samples[lid] = []; this.samples[lid].push(Object.assign({ t: pop.sub.tick, n, id: b.id, age: b.age }, genomeSummary(b.g))); }
  }
  export(pop, fullGenomes) { // everything, and the full genomes of the eldest of the six largest living lineages
    const counts = new Map(); for (const b of pop.beings) counts.set(b.lineage, (counts.get(b.lineage) || 0) + 1);
    const top = [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, fullGenomes === undefined ? 6 : fullGenomes), genomes = {};
    for (const [lid] of top) { let e = null; for (const b of pop.beings) if (b.lineage === lid && (!e || b.age > e.age)) e = b; if (e) genomes[lid] = Object.assign({ id: e.id, age: e.age }, serializeGenome(e.g)); }
    return { meta: { tick: pop.sub.tick, every: this.every, rules: pop.rules, prices: Object.assign({}, POP), cfg: Object.assign({}, pop.cfg), keys: FRAME_KEYS, senses: SENSES, groups: RG_NAMES }, founder: this.founder, founderGenome: this.founderFull, frames: this.frames, events: this.events, samples: this.samples, genomes };
  }
}
const MOTORS = Int32Array.from({ length: MAXM }, (_, m) => MOTOR + m);
const FEEDBACK = Array.from({ length: SN }, (_, j) => [j, j]); // the predicted changes are added to the next moment's reading

return { Mulberry, gauss, FEEDBACK, Chronicle, serializeGenome, genomeSummary, FRAME_KEYS, RG, RG_NAMES, gbit, ALL_MASK, MAXH, SLOWEST, maskUnits, maskCount, maskNames, brainH, cortexStart, defaultBrain, RULES, CFG, POP, N, DIRS, SENSES, NS, SN, NM, NI, NOUT, HP, HM, ACTIVE, FANIN, PORT, CLOCK, MOTOR, MOTORS, MAXN, MAXM, S, GENE, G, wrap, ddx, blobs, Substrate, Being, Population, founderGenome, cloneGenome, mutate, mutateTheta, genomeMuscles, freePort, RecordPatch };
});
