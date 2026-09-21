/* The gait lab: an explicit search over bodies and the policy patch's motor rows, outside the world.
   It finds the founders (a body that crawls and turns toward food) and the locomotion ceiling per body size,
   and pretrains the founders' model patch on their own streams. The world itself keeps no fitness function.

   node sim/gaitlab.js evolve  [generations] [population] [seed]   -> receipts/gaitlab/evolve_<seed>.json (+ champion genome)
   node sim/gaitlab.js ceiling [generations] [population] [seed]   -> straight-line ceiling per node count
   node sim/gaitlab.js pretrain <champion.json> [ticks] [seed]     -> sim/founder.json (policy from the search, model learned by observation)
   node sim/gaitlab.js show <genome.json>                          -> one trial printed
*/
'use strict';
const fs = require('fs'), path = require('path');
const CB = require('./core.js');
const { Mulberry, gauss, CFG, POP, NI, NS, SN, NM, HP, CLOCK, MOTOR, MAXM, MAXN, S, GENE, G, Substrate, Population, founderGenome, cloneGenome, mutate, wrap, ddx } = CB;

const ARENA_RULES = { rock: false, mud: false, bite: false, injury: false, growth: false, devnoise: false, split: false, records: false, slow: false, plan: false, noise: false };
const ARENA_CFG = { grow: 0, decay: 0, diffuse: 0, warmup: 0 };
const SIN = CLOCK, COS = CLOCK + 1, LR = S['food L-R'], AHEAD = S['food ahead'], FAST = Math.log(0.2 / 0.8); // motor channels retain a fifth: near-immediate readout

function arena(seed) { return new Substrate(seed, ARENA_RULES, ARENA_CFG); }
function paintFood(sub, x, y, radius, units) { for (let dy = -radius; dy <= radius; dy++) for (let dx = -radius; dx <= radius; dx++) if (dx * dx + dy * dy <= radius * radius) sub.food[sub.cell(x + dx, y + dy)] = units; sub.smell(); }
function clearFood(sub) { sub.food.fill(0); sub.smell(); }

// the motor rows of the policy: for port k, channel k reads the clock, the food senses and a bias, and the readout passes channel k with a gain
function setMotorRow(th, k, amp, phase, lr, ahead, bias, gain) {
  const { B, b, C, g } = th; for (let j = 0; j < NI; j++) B[k * NI + j] = 0;
  B[k * NI + SIN] = amp * Math.cos(phase); B[k * NI + COS] = amp * Math.sin(phase); B[k * NI + LR] = lr; B[k * NI + AHEAD] = ahead; b[k] = bias; g[k] = FAST;
  for (let i = 0; i < HP; i++) C[k * HP + i] = 0; C[k * HP + k] = gain;
}
function randomDrive(g, rng) { for (let k = 0; k < MAXM; k++) setMotorRow(g.theta.policy, k, 0.8 + 1.2 * rng.random(), rng.random() * 2 * Math.PI, 0.6 * gauss(rng), 0, 0.3 * gauss(rng), 1.2 + 0.8 * rng.random()); g.theta.policy.c.fill(0); g.theta.policy.C.fill(0, MAXM * HP); }
function mutateDrive(g, rng, sigma) { // the search moves the motor rows' clock, sense and bias weights and the gain
  const th = g.theta.policy; for (let k = 0; k < MAXM; k++) { const r = k * NI;
    if (rng.random() < 0.3) { th.B[r + SIN] += sigma * gauss(rng); th.B[r + COS] += sigma * gauss(rng); }
    if (rng.random() < 0.35) th.B[r + LR] += 1.5 * sigma * gauss(rng); if (rng.random() < 0.15) th.B[r + AHEAD] += sigma * gauss(rng);
    if (rng.random() < 0.2) th.b[k] += sigma * gauss(rng); if (rng.random() < 0.15) th.C[k * HP + k] = Math.max(0.2, th.C[k * HP + k] + sigma * gauss(rng)); }
}
function offspring(g, rng, bodyToo) { const c = bodyToo ? mutate(g, rng, { gene: 0, body: 0.2, add: 0.08, remove: 0.06, muscle: 0.08, spring: 0.05, grip: 0.15, theta: 0 }) : cloneGenome(g); if (rng.random() < 0.2) c.freq = Math.max(0, Math.min(GENE.freq.length - 1, c.freq + (rng.random() < 0.5 ? -1 : 1))); mutateDrive(c, rng, 0.35); return c; }

// one trial: the being starts at the centre facing +x, a food blob sits at `bearing` and `distance`; returns progress and meals
function trial(g, bearing, distance, ticks, seed, log, nodes) {
  const sub = arena(seed), pop = new Population(sub, seed, null, { initial: 0 }); const cx = CFG.w / 2, cy = CFG.h / 2;
  const tx = wrap(cx + Math.cos(bearing) * distance, CFG.w), ty = wrap(cy + Math.sin(bearing) * distance, CFG.h); paintFood(sub, tx, ty, 1, 8);
  const b = pop.spawn(cloneGenome(g), cx, cy, 0, 5000, null, null); b.phase = 0; if (nodes && nodes < b.nGenome) b.place(cx, cy, 0, nodes); // a juvenile: the first nodes only
  const d0 = Math.hypot(ddx(cx, tx, CFG.w), ddx(cy, ty, CFG.h)); let effort = 0;
  for (let t = 0; t < ticks; t++) { pop.step(); if (!pop.beings.length) break; effort += b.effort; if (log) log.push({ t, x: Array.from(b.x.subarray(0, b.n)), y: Array.from(b.y.subarray(0, b.n)), cmd: Array.from(b.cmd), speed: b.speed }); }
  const d1 = pop.beings.length ? Math.hypot(ddx(b.x[0], tx, CFG.w), ddx(b.y[0], ty, CFG.h)) : d0;
  return { progress: d0 - d1, eats: b.eats, dist: b.dist, effort: effort / ticks };
}
const BEARINGS = [-1.5, -0.5, 0.5, 1.5], DISTANCE = 7;
function fitness(g, seed, ticks) { // the adult on four bearings, and the newborn (the first three nodes) on two: a founder must crawl before it has grown
  let f = 0; const parts = []; for (const bearing of BEARINGS) { const r = trial(g, bearing, DISTANCE, ticks, seed, null); f += r.progress + 0.5 * r.eats; parts.push(r); }
  if (g.nodes.length > CFG.birthNodes) for (const bearing of [-0.5, 0.5]) { const r = trial(g, bearing, DISTANCE, ticks, seed, null, CFG.birthNodes); f += 0.7 * (r.progress + 0.5 * r.eats); parts.push(Object.assign({ juvenile: true }, r)); }
  return { f, parts }; }
function straight(g, seed, ticks) { const sub = arena(seed), pop = new Population(sub, seed, null, { initial: 0 }); const b = pop.spawn(cloneGenome(g), CFG.w / 2, CFG.h / 2, 0, 5000, null, null); b.phase = 0; for (let t = 0; t < ticks; t++) pop.step(); return { x: ddx(CFG.w / 2, b.x[0], CFG.w), dist: b.dist }; }

const GRIP = { along: 0.9, side: 0.1, back: 0.15 };
const TEMPLATES = { // bodies the search starts from: a chain, a diamond with lateral muscles, a ladder with two rails
  chain3: { nodes: [[0, 0], [-1, 0], [-2, 0]], springs: [[0, 1, 1], [1, 2, 1]] },
  chain4: { nodes: [[0, 0], [-1, 0], [-2, 0], [-3, 0]], springs: [[0, 1, 1], [1, 2, 1], [2, 3, 1]] },
  diamond: { nodes: [[0, 0], [-1, 0.6], [-1, -0.6], [-2, 0]], springs: [[0, 1, 1], [0, 2, 1], [1, 3, 1], [2, 3, 1], [1, 2, 0]] },
  ladder: { nodes: [[0, 0], [-0.8, 0.5], [-0.8, -0.5], [-1.8, 0.5], [-1.8, -0.5], [-2.8, 0]], springs: [[0, 1, 0], [0, 2, 0], [1, 2, 0], [1, 3, 1], [2, 4, 1], [3, 4, 0], [3, 5, 1], [4, 5, 1], [1, 4, 0], [2, 3, 0]] } };
function fromTemplate(name, rng) { const g = founderGenome(rng), tp = TEMPLATES[name]; g.nodes = tp.nodes.map(([x, y]) => Object.assign({ x, y }, GRIP)); let port = 0; g.springs = tp.springs.map(([a, b, m]) => ({ a, b, muscle: m, port: m ? port++ : -1 })); return g; }
function evolve(generations, size, seed, mode) {
  const rng = new Mulberry(seed), ticks = mode === 'ceiling' ? 200 : 240, out = { mode, seed, generations, size, ticks, history: [] }, names = Object.keys(TEMPLATES);
  let pop = []; for (let i = 0; i < size; i++) { const g = fromTemplate(names[i % names.length], rng); randomDrive(g, rng); if (rng.random() < 0.5) Object.assign(g, mutate(g, rng, { gene: 0, body: 0.3, add: 0.3, remove: 0, muscle: 0.2, spring: 0.2, grip: 0.3, theta: 0 })); pop.push(g); }
  const score = g => mode === 'ceiling' ? straight(g, seed, ticks).x : fitness(g, seed, ticks).f;
  let scored = pop.map(g => ({ g, f: score(g) }));
  for (let gen = 0; gen < generations; gen++) {
    scored.sort((a, b) => b.f - a.f); const best = scored[0];
    const nodes = best.g.nodes.length, muscles = best.g.springs.filter(s => s.muscle).length;
    out.history.push({ gen, best: best.f, mean: scored.reduce((s, e) => s + e.f, 0) / scored.length, nodes, muscles, freq: G(best.g, 'freq') });
    console.log(`gen ${gen} best ${best.f.toFixed(2)} mean ${out.history[out.history.length - 1].mean.toFixed(2)} body ${nodes} nodes ${muscles} muscles freq ${G(best.g, 'freq')}`);
    const next = scored.slice(0, 4).map(e => e); // elites
    while (next.length < size) { const a = scored[(rng.random() * (size >> 1)) | 0], b = scored[(rng.random() * (size >> 1)) | 0]; const p = a.f > b.f ? a : b; const c = offspring(p.g, rng, rng.random() < 0.5); next.push({ g: c, f: score(c) }); }
    scored = next;
  }
  scored.sort((a, b) => b.f - a.f); out.champion = scored[0].g; out.championScore = scored[0].f; out.top = scored.slice(0, 8).map(e => ({ f: e.f, nodes: e.g.nodes.length, muscles: e.g.springs.filter(s => s.muscle).length }));
  out.sizes = {}; for (const e of scored) { const k = e.g.nodes.length; if (!out.sizes[k] || e.f > out.sizes[k]) out.sizes[k] = e.f; }
  return out;
}

// pretrain the founders' model patch on the champion's own streams, then measure held-out prediction against baselines
function pretrain(champion, ticks, seed) {
  const g = cloneGenome(champion); g.cells = 0; g.slowRate = 3; g.noise = 2; g.horizon = 0;
  const rules = Object.assign({}, ARENA_RULES, { slow: true, noise: true }), sub = new Substrate(seed, rules, Object.assign({}, ARENA_CFG, { window: 8, backtrack: true }));
  const rng = new Mulberry(seed + 7); const pop = new Population(sub, seed, null, { initial: 0 });
  const b = pop.spawn(g, CFG.w / 2, CFG.h / 2, 0, 1e9, null, null);
  const sprinkle = () => { clearFood(sub); for (let i = 0; i < 40; i++) paintFood(sub, rng.random() * CFG.w, rng.random() * CFG.h, 1, 6); };
  const errs = [], base = [], meanErr = [], last = new Float64Array(SN); let seen = 0; const runMean = new Float64Array(SN);
  const evalFrom = Math.floor(ticks * 0.8); let learned = 0;
  for (let t = 0; t < ticks; t++) {
    if (t % 400 === 0) sprinkle();
    if (t === evalFrom) { b.slowRate = 0; } // the last fifth is held out: the slow parameters are frozen
    const prevPred = Float64Array.from(b.pred), prevKnown = Float64Array.from(b.known), had = b.model.pending;
    pop.step(); if (!pop.beings.length) throw new Error('the being died in the lab');
    if (had && t >= evalFrom) { let e = 0, p = 0, m = 0, c = 0; for (let j = 0; j < SN; j++) { if (!b.known[j]) continue; const a = b.sense[j]; e += (last[j] + prevPred[j] - a) ** 2; p += (last[j] - a) ** 2; m += (runMean[j] - a) ** 2; c++; } errs.push(e / c); base.push(p / c); meanErr.push(m / c); }
    if (t < evalFrom) { seen++; for (let j = 0; j < SN; j++) runMean[j] += (b.sense[j] - runMean[j]) / seen; }
    last.set(b.sense); learned = b.model.updates;
  }
  const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
  const result = { ticks, evalTicks: errs.length, modelError: mean(errs), persistenceError: mean(base), meanError: mean(meanErr), updates: learned, rejected: b.model.rejected, note: 'errors on the next reading: the model adds its predicted change to the current reading; persistence predicts no change; mean predicts the running mean' };
  const founder = cloneGenome(champion); founder.theta.model = b.model.parameters(); founder.theta.policy = b.policy.parameters();
  return { result, founder };
}

function serialize(g) { const c = cloneGenome(g); c.theta = { policy: {}, model: {} }; for (const p of ['policy', 'model']) for (const k in g.theta[p]) c.theta[p][k] = Array.from(g.theta[p][k]); return c; }
function deserialize(j) { const th = { policy: {}, model: {} }; for (const p of ['policy', 'model']) for (const k in j.theta[p]) th[p][k] = Float64Array.from(j.theta[p][k]); return cloneGenome(Object.assign({}, j, { theta: th })); }

if (require.main === module) {
  const [cmd, ...args] = process.argv.slice(2); const outDir = path.join(__dirname, '..', 'receipts', 'gaitlab'); fs.mkdirSync(outDir, { recursive: true });
  if (cmd === 'evolve' || cmd === 'ceiling') {
    const generations = +(args[0] || 40), size = +(args[1] || 48), seed = +(args[2] || 1); const t0 = Date.now();
    const out = evolve(generations, size, seed, cmd); out.seconds = (Date.now() - t0) / 1000; out.champion = serialize(out.champion);
    const file = path.join(outDir, `${cmd}_${seed}.json`); fs.writeFileSync(file, JSON.stringify(out)); console.log('wrote', file, 'champion', out.championScore.toFixed(2), 'sizes', JSON.stringify(out.sizes), `${out.seconds.toFixed(0)} s`);
  } else if (cmd === 'pretrain') {
    const champion = deserialize(JSON.parse(fs.readFileSync(args[0], 'utf8')).champion || JSON.parse(fs.readFileSync(args[0], 'utf8'))); const ticks = +(args[1] || 8000), seed = +(args[2] || 1);
    const { result, founder } = pretrain(champion, ticks, seed); console.log(JSON.stringify(result));
    const file = path.join(__dirname, 'founder.json'); fs.writeFileSync(file, JSON.stringify({ founder: serialize(founder), pretrain: result, source: path.basename(args[0]) })); console.log('wrote', file);
  } else if (cmd === 'show') {
    const j = JSON.parse(fs.readFileSync(args[0], 'utf8')); const g = deserialize(j.champion || j.founder || j); const log = [];
    for (const bearing of BEARINGS) { const r = trial(g, bearing, DISTANCE, 240, 1, bearing === 0.5 ? log : null); console.log('bearing', bearing, JSON.stringify(r)); }
    for (const bearing of [-0.5, 0.5]) { const r = trial(g, bearing, DISTANCE, 240, 1, null, CFG.birthNodes); console.log('newborn, bearing', bearing, JSON.stringify(r)); }
    console.log('nodes', g.nodes.length, 'springs', g.springs.length, 'muscles', g.springs.filter(s => s.muscle).length, 'freq', G(g, 'freq'), 'dist/tick', log.length ? (function () { let d = 0; for (let i = 1; i < log.length; i++) d += Math.hypot(ddx(log[i - 1].x[0], log[i].x[0], CFG.w), ddx(log[i - 1].y[0], log[i].y[0], CFG.h)); return (d / log.length).toFixed(4); })() : '-');
  } else console.log('usage: evolve|ceiling|pretrain|show');
}
module.exports = { arena, paintFood, trial, fitness, straight, evolve, pretrain, serialize, deserialize, setMotorRow, randomDrive };
