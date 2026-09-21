/* Run a world headless and log it: node sim/probe.js '<rules json>' <ticks> <seed> ['<cfg/price overrides json>'] > out.jsonl
   With '{"chronicle":"path.json"}' in the overrides the whole chronicle (per-lineage frames, events, genome samples) is written there; sim/why.py reads it.
   Every 200 ticks one line: population, mass, births and deaths, lifetime at death by brain traits over the last window,
   the census of the genes that make a brain (records, slow learning, horizon, noise), body sizes and speeds, model error
   early against late in life, plan acceptance. The founders come from sim/founder.json unless overrides.founder is false. */
'use strict';
const fs = require('fs'), path = require('path');
const CB = require('./core.js');
const { CFG, POP, GENE, G, Substrate, Population, cloneGenome } = CB;
const { deserialize } = require('./gaitlab.js');

const rules = JSON.parse(process.argv[2] || '{}'), ticks = +(process.argv[3] || 6000), seed = +(process.argv[4] || 1), over = JSON.parse(process.argv[5] || '{}');
for (const k in over) { if (k in POP) POP[k] = over[k]; }
const cfg = {}; for (const k in over) if (k in CFG) cfg[k] = over[k];
let founder = null;
if (over.founder !== false) { const f = path.join(__dirname, 'founder.json'); if (fs.existsSync(f)) founder = deserialize(JSON.parse(fs.readFileSync(f, 'utf8')).founder); }
if (founder && over.genes) for (const k in over.genes) founder[k] = over.genes[k]; // e.g. {"cells":0} to start without records
const sub = new Substrate(seed, rules, cfg), pop = new Population(sub, seed + 1000, founder, { chronicle: !!over.chronicle });
if (over.mutRates) pop.mutRates = over.mutRates;
const m0 = pop.mass; const every = over.every || 200;
const deaths = []; pop.onDeath = b => deaths.push({ age: b.age, cells: G(b.g, 'cells'), horizon: G(b.g, 'horizon'), slow: G(b.g, 'slowRate'), noise: G(b.g, 'noise'), nodes: b.nGenome, killed: !!(b.pain > 0), errEarly: b.nEarly ? b.errEarly / b.nEarly : null, errLate: b.nLate ? b.errLate / b.nLate : null, plans: b.plans, accepted: b.planAccepted, confirmed: b.planConfirmed, eats: b.eats, bites: b.bites, dist: b.dist });
const median = a => { if (!a.length) return null; const s = a.slice().sort((x, y) => x - y); return s[s.length >> 1]; };
const mean = a => a.length ? a.reduce((x, y) => x + y, 0) / a.length : null;
const t0 = Date.now();
console.log(JSON.stringify({ kind: 'start', rules: sub.rules, ticks, seed, over, founder: founder ? { nodes: founder.nodes.length, muscles: founder.springs.filter(s => s.muscle).length, cells: G(founder, 'cells'), horizon: G(founder, 'horizon'), slow: G(founder, 'slowRate') } : null, mass: m0 }));
for (let t = 1; t <= ticks; t++) {
  sub.step(); pop.step();
  if (t % every === 0 || t === ticks) {
    const bs = pop.beings, n = bs.length, census = {};
    for (const k of ['cells', 'horizon', 'slowRate', 'noise', 'splitAt', 'freq', 'dFood', 'dEnergy', 'dPain', 'dSpeed', 'planRadius']) { const c = {}; for (const b of bs) { const v = G(b.g, k); c[v] = (c[v] || 0) + 1; } census[k] = c; }
    const withRecords = bs.filter(b => G(b.g, 'cells') > 0).length, withPlan = bs.filter(b => G(b.g, 'horizon') > 0).length, withSlow = bs.filter(b => G(b.g, 'slowRate') > 0).length;
    const lifeBy = (name, pred) => { const a = deaths.filter(d => pred(d)).map(d => d.age); return { n: a.length, median: median(a), mean: mean(a) }; };
    const line = { kind: 'tick', t, n, mass: pop.mass - m0, births: pop.births, deaths: pop.deaths, kills: pop.kills, seconds: (Date.now() - t0) / 1000,
      alive: { records: withRecords, plan: withPlan, slow: withSlow, nodesMean: mean(bs.map(b => b.n)), genomeNodesMean: mean(bs.map(b => b.nGenome)), musclesMean: mean(bs.map(b => b.g.springs.filter(s => s.muscle).length)), speedMean: mean(bs.map(b => Math.abs(b.speed))), ageMean: mean(bs.map(b => b.age)), energyMean: mean(bs.map(b => b.energy)), lineages: new Set(bs.map(b => b.lineage)).size },
      lifetime: { all: lifeBy('all', () => true), records: lifeBy('records', d => d.cells > 0), noRecords: lifeBy('noRecords', d => d.cells === 0), plan: lifeBy('plan', d => d.horizon > 0), noPlan: lifeBy('noPlan', d => d.horizon === 0), slow: lifeBy('slow', d => d.slow > 0), noSlow: lifeBy('noSlow', d => d.slow === 0), killed: lifeBy('killed', d => d.killed) },
      model: { errEarly: mean(deaths.filter(d => d.errEarly !== null).map(d => d.errEarly)), errLate: mean(deaths.filter(d => d.errLate !== null).map(d => d.errLate)), errLateRecords: mean(deaths.filter(d => d.errLate !== null && d.cells > 0).map(d => d.errLate)), errLateNoRecords: mean(deaths.filter(d => d.errLate !== null && d.cells === 0).map(d => d.errLate)) },
      plans: { total: deaths.reduce((s, d) => s + d.plans, 0), accepted: deaths.reduce((s, d) => s + d.accepted, 0), confirmed: deaths.reduce((s, d) => s + d.confirmed, 0) }, census };
    console.log(JSON.stringify(line)); deaths.length = 0;
  }
}
if (over.chronicle) fs.writeFileSync(over.chronicle, JSON.stringify(pop.chronicle.export(pop)));
console.log(JSON.stringify({ kind: 'end', ticks, seconds: (Date.now() - t0) / 1000, n: pop.beings.length, mass: pop.mass - m0, chronicle: over.chronicle || null }));
