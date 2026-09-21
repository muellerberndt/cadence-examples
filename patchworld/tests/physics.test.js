/* P0: mass is conserved; a ratchet body crawls and a symmetric one does not; the same seed replays the same world.  node tests/physics.test.js */
'use strict';
const assert = require('assert');
const CB = require('../sim/core.js');
const { CFG, S, Substrate, Population, founderGenome, cloneGenome, Mulberry, ddx } = CB;
const lab = require('../sim/gaitlab.js');

function chain(along, side, back, n) { // n nodes in a line, every link a muscle
  const g = founderGenome(new Mulberry(1)); g.nodes = []; g.springs = [];
  for (let i = 0; i < n; i++) g.nodes.push({ x: -i, y: 0, along, side, back });
  for (let i = 0; i + 1 < n; i++) g.springs.push({ a: i, b: i + 1, muscle: 1, port: i });
  for (let k = 0; k < n - 1; k++) lab.setMotorRow(g.theta.policy, k, 1.5, k * 1.2, 0, 0, 0, 1.5);
  return g;
}
// 1. one muscle between two nodes is a reciprocal motion: with a ratchet grip it crawls, with a symmetric grip it stays (the scallop theorem);
//    three nodes with two phase-lagged muscles move even with a symmetric grip (the three-sphere swimmer), and faster with the ratchet
const ratchet2 = lab.straight(chain(0.9, 0.1, 0.15, 2), 1, 300), symmetric2 = lab.straight(chain(0.5, 0.5, 0.5, 2), 1, 300);
const ratchet3 = lab.straight(chain(0.9, 0.1, 0.15, 3), 1, 300), symmetric3 = lab.straight(chain(0.5, 0.5, 0.5, 3), 1, 300);
console.log('head x after 300 ticks: two nodes ratchet', ratchet2.x.toFixed(2), 'symmetric', symmetric2.x.toFixed(2), '| three nodes ratchet', ratchet3.x.toFixed(2), 'symmetric', symmetric3.x.toFixed(2));
assert(Math.abs(ratchet2.x) > 2, 'a two-node ratchet crawls');
assert(Math.abs(symmetric2.x) < 0.2, 'a two-node symmetric body stays');
assert(Math.abs(ratchet3.x) > Math.abs(symmetric3.x) + 2, 'the ratchet chain beats the symmetric chain');
// 2. mass is conserved through eating, growth, biting, death and birth
{
  const sub = new Substrate(3, {}), pop = new Population(sub, 4); const m0 = pop.mass;
  for (let t = 0; t < 300; t++) { sub.step(); pop.step(); assert.strictEqual(pop.mass, m0, 'mass at tick ' + t); }
  console.log('mass conserved over 300 ticks:', m0, 'deaths', pop.deaths, 'births', pop.births);
}
// 3. the same seed gives the same world
{
  const run = () => { const sub = new Substrate(5, {}), pop = new Population(sub, 6); for (let t = 0; t < 120; t++) { sub.step(); pop.step(); } return pop.beings.map(b => [b.id, b.energy, +b.x[0].toFixed(9)]).join(';'); };
  assert.strictEqual(run(), run(), 'deterministic replay'); console.log('deterministic replay: ok');
}
// 4. developmental noise changes the rest lengths of two siblings of one genome
{
  const sub = new Substrate(7, { devnoise: true }), pop = new Population(sub, 8, null, { initial: 0 }); const g = founderGenome(new Mulberry(2));
  const a = pop.spawn(cloneGenome(g), 10, 10, 0, 30, null, null), b = pop.spawn(cloneGenome(g), 30, 30, 0, 30, null, null);
  assert(Math.abs(a.rest0[0] - b.rest0[0]) > 1e-6, 'developmental noise'); console.log('developmental noise: rest lengths', a.rest0[0].toFixed(3), b.rest0[0].toFixed(3));
}
console.log('physics tests passed');
