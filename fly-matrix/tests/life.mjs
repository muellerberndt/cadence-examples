// The fly's life follows the fruits the visitor moves.
//   node tests/life.mjs
// The room's placeFruit mutates a fruit's position in place; the life layer must hold that array,
// not a copy, so that the smell, a descent in progress and the perch follow the fruit. Two reports
// from the live page (2026-09-25) came from a copy: the fly kept landing and feeding where the
// bread had stood, hanging in the air there. Checked here with a room without three.js: the smell
// moves with the fruit; a descent to the banana ends on the banana where it was set down mid-descent;
// a fly standing on the bread is shaken off when the bread is moved, and a fly standing on the bare
// table is not; a fly standing where a fruit is set down is startled off.
import { DT, Flight } from "../web/body.js";
import { Life, APPETITE, FRUIT_FOOTPRINT } from "../web/life.js";

let failures = 0;
const fail = (msg) => { failures++; console.log("FAIL " + msg); };
const ok = (msg) => console.log("ok   " + msg);
const near = (a, b, tol) => Math.hypot(a[0] - b[0], a[1] - b[1]) < tol;

// a room like room.js's without the drawing: the table, the two fruits with the in-place placeFruit, the surface heights
function stubRoom() {
  const table = { x0: 2.45, x1: 3.55, y0: 1.95, y1: 2.65, z: 0.75 };
  const fruits = { banana: { pos: [2.88, 2.33, table.z + 0.045] }, bread: { pos: [3.13, 2.28, table.z + 0.0568] } };
  const room = { table, fruits };
  room.placeFruit = (name, xy) => { const F = fruits[name]; F.pos[0] = xy[0]; F.pos[1] = xy[1]; };
  room.surfaceZ = (x, y) => { if (x < table.x0 || x > table.x1 || y < table.y0 || y > table.y1) return 0; for (const F of Object.values(fruits)) if (Math.hypot(x - F.pos[0], y - F.pos[1]) < 0.06) return F.pos[2]; return table.z; };
  return room;
}
function move(room, life, name, xy) { const from = room.fruits[name].pos.slice(); room.placeFruit(name, xy); return life.fruitMoved(name, from); }

// -- the smell follows the fruit, and at a fruit its own smell dominates ------------------------
{
  const room = stubRoom(), B = room.fruits.banana.pos;
  const flight = new Flight([B[0], B[1], B[2] + 0.02], 0.0), life = new Life(flight, room, 1);
  if (life.fruits.banana.pos !== room.fruits.banana.pos || life.fruits.bread.pos !== room.fruits.bread.pos) fail("the life holds copies of the fruit positions");
  const before = life.odour(life.fruits.banana.pos).c, other = life.odour(life.fruits.bread.pos).c;
  const s0 = life.senses(null, 0); const smelledBefore = life.smelled;
  room.placeFruit("banana", [3.45, 2.05]);
  const after = life.odour(life.fruits.banana.pos).c;
  life.senses(null, 0);
  if (before < 0.9) fail(`the banana under the fly smells ${before.toFixed(2)}, expected near 1`);
  if (other > 0.3) fail(`at the banana the bread (25 cm away) smells ${other.toFixed(2)}: the two smells are not separable`);
  if (s0["orn:decaying_fruit:left"] < 0.9 || s0["orn:yeasty:left"] > 0.35) fail(`at the banana the receptor drives are ${s0["orn:decaying_fruit:left"].toFixed(2)} (decaying fruit) and ${s0["orn:yeasty:left"].toFixed(2)} (yeast)`);
  if (after > 0.3) fail(`the banana moved away still smells ${after.toFixed(2)} where it stood`);
  if (smelledBefore !== "banana") fail(`smelled ${smelledBefore} over the banana`);
  if (failures === 0) ok(`the smell follows the fruit and is separable: banana ${before.toFixed(2)}, bread ${other.toFixed(2)} at the banana; ${after.toFixed(2)} after the banana is moved 0.6 m`);
}

// -- one mushroom-body decision per search, at the decision level; an avoided smell ends with nothing
{
  const room = stubRoom(), B = room.fruits.banana.pos;
  const flight = new Flight([B[0] - 0.5, B[1], B[2] + 0.05], 0.0), life = new Life(flight, room, 7);
  life.hunger = 0.8; life.bout = 100; life.senses(null, 0);
  if (life.smelled !== "banana") fail(`half a metre from the banana the fly smells ${life.smelled}`);
  if (life.odourTick()) fail("the brain was asked at the first whiff");
  if (!life.search || life.search.fruit !== "banana" || life.search.asked) fail(`no search opened at the first whiff: ${JSON.stringify(life.search)}`);
  if (!life.valence || life.valence.action !== 0 || !life.valence.innate) fail("the instinct does not orient the fly toward the smell before the decision");
  flight.p = [B[0] - 0.1, B[1], B[2] + 0.05]; life.senses(null, 0);
  if (life.odourTick()) fail("the brain was asked 10 cm from the fruit, outside the decision reach");
  life.decide(true); if (life.landing) fail("the instinct landed on the fruit before the brain had decided");
  if (life.speed > 0.11) fail(`the drawn fly does not slow to a hover over the fruit (speed ${life.speed.toFixed(2)})`);
  flight.p = [B[0] - 0.04, B[1], B[2] + 0.05]; life.senses(null, 0);
  if (!life.odourTick()) fail("the brain was not asked 4 cm from the fruit");
  if (life.odourTick()) fail("the brain was asked twice in one search");
  // no answer from the brain: the instinct hovers for DECISION_WAIT_S (a bout ending meanwhile does not land it), then lands anyway
  life.bout = 0.05; life.decide(true); if (life.landing) fail("the instinct landed before the decision wait had passed"); life.bout = 100;
  life.clock += 3.5; life.decide(true); if (!life.landing) fail("the instinct did not land after waiting 3.5 s for the brain");
  life.landing = null; life.landingFruit = null;
  life.applyDecision({ action: 1, p: [0.3, 0.7] });
  if (!life.valence || life.valence.action !== 1 || life.lastP.banana !== 0.3) fail(`the decision was not applied: ${JSON.stringify(life.valence)}`);
  life.decide(true); // the avoid steers away, no landing
  if (life.landing) fail("an avoided smell was landed on");
  flight.p = [B[0] - 1.5, B[1], B[2] + 0.5]; life.senses(null, 0);
  life.odourTick();
  if (life.search) fail("the search did not end when the avoided smell was lost");
  if (!life.pendingReward || life.pendingReward.reward !== 0 || life.pendingReward.fresh) fail(`avoided: ${JSON.stringify(life.pendingReward)}`);
  // a smell lost before the brain was asked is dropped, nothing queued
  life.pendingReward = null; flight.p = [B[0] - 0.5, B[1], B[2] + 0.05]; life.senses(null, 0); life.odourTick();
  if (!life.search) fail("no new search on the next whiff");
  flight.p = [B[0] - 1.5, B[1], B[2] + 0.5]; life.senses(null, 0); life.odourTick();
  if (life.search || life.pendingReward) fail("a search lost before the decision was not dropped silently");
  // sugar reached after an avoid decision, or on the other fruit, is not credited to that decision
  life.pendingReward = null; life.search = { since: 0, fruit: "banana", asked: true, action: 1, landed: null };
  life.outcome(1, "sugar on the banana", "banana");
  if (!life.pendingReward || !life.pendingReward.fresh) fail("sugar after an avoid decision was credited to it");
  life.pendingReward = null; life.search = { since: 0, fruit: "banana", asked: true, action: 0, landed: null };
  life.outcome(1, "sugar on the bread", "bread");
  if (!life.pendingReward || !life.pendingReward.fresh) fail("sugar on the other fruit was credited to the banana's decision");
  life.pendingReward = null; life.search = { since: 0, fruit: "banana", asked: true, action: 0, landed: null };
  life.outcome(1, "sugar on the banana", "banana");
  if (!life.pendingReward || life.pendingReward.fresh) fail("sugar on the approached fruit was not credited to its decision");
  if (failures === 0) ok("one decision per search, at the fruit; an avoided smell ends with nothing, a lost one is dropped; chance sugar is credited afresh");
}

// -- a search that reaches a fruit without sugar stays open until the fly leaves or is struck ----
{
  const room = stubRoom(), Br = room.fruits.bread.pos;
  // landed on the bread by a search, no sugar there
  const flight = new Flight([Br[0], Br[1], Br[2] + 1.2e-3], 0.0), life = new Life(flight, room, 6);
  life.sugar = "banana"; life.hunger = 0.8;
  life.search = { since: 0, fruit: "bread", asked: true, action: 0, landed: null }; life.valence = { fruit: "bread", action: 0, p: [1, 0] };
  life.mode = "landed"; life.perch = Br.slice(); life.hold(); life.sit = 100; life.ethogram.sitStart = 0;
  // the landing bookkeeping as step() does it when the fly settles on a fruit
  const on = life.fruitAt(flight.p); if (on !== "bread") fail(`the fly is on ${on}`);
  if (life.search) { life.search.landed = on; life.valence = null; }
  // the search is not timed out while the fly sits
  life.clock = 60; life.odourTick();
  if (!life.search || life.search.landed !== "bread") fail("the open search on the bread was closed while the fly sat there");
  if (life.pendingReward) fail(`an outcome was queued while the fly sat: ${JSON.stringify(life.pendingReward)}`);
  // a blow: the search ends with -1, not fresh (a search was open)
  if (!life.punished()) fail("the blow on the bread did not count");
  if (!life.pendingReward || life.pendingReward.reward !== -1 || life.pendingReward.fresh) fail(`after the blow: ${JSON.stringify(life.pendingReward)}`);
  if (life.search) fail("the search is still open after the blow");
  life.pendingReward = null;
  // leaving the bread ends an open search with nothing
  life.search = { since: life.clock, fruit: "bread", asked: true, action: 0, landed: "bread" };
  life.takeoff("voluntary takeoff");
  if (!life.pendingReward || life.pendingReward.reward !== 0) fail(`after leaving: ${JSON.stringify(life.pendingReward)}`);
  if (life.search) fail("the search is still open after the takeoff");
  // a blow with no search open is a fresh outcome: the page asks for the approach decision it blames
  life.pendingReward = null; life.mode = "landed"; life.perch = Br.slice(); life.hold(); life.senses(null, 0);
  if (!life.punished()) fail("the blow on the bread with no search did not count");
  if (!life.pendingReward || life.pendingReward.reward !== -1 || !life.pendingReward.fresh) fail(`fresh blow: ${JSON.stringify(life.pendingReward)}`);
  if (failures === 0) ok("a sugarless landing keeps the search open: a blow ends it with -1, leaving ends it with 0, a blow with no search is fresh");
}

// -- a descent to the banana ends on the banana where it was set down during the descent ---------
{
  const room = stubRoom(), B0 = room.fruits.banana.pos.slice();
  const flight = new Flight([B0[0] - 0.08, B0[1], B0[2] + 0.25], 0.0), life = new Life(flight, room, 2);
  life.hunger = 0.8; life.senses(null, 0);
  life.search = { since: 0, fruit: "banana", asked: true, landed: null }; life.valence = { fruit: "banana", action: 0, p: [1, 0] };
  life.chooseLanding();
  if (!life.landingFruit || life.landingFruit.name !== "banana") fail("the descent is not to the banana");
  if (!near(life.landing, B0, 0.03)) fail("the landing spot is not on the banana");
  // a few steps into the descent the visitor sets the banana down 30 cm away
  for (let k = 0; k < 400; k++) life.step(DT);
  const target = [B0[0] + 0.3, B0[1] + 0.1];
  const shaken = move(room, life, "banana", target);
  if (shaken) fail("a flying fly was shaken off");
  life.step(DT);
  if (!near(life.landing, target, 0.03)) fail(`the landing spot stayed at the old place: ${life.landing.map((v) => v.toFixed(3))}`);
  let steps = 0;
  while (life.mode !== "landed" && steps < 20 / DT) { life.step(DT); steps++; }
  if (life.mode !== "landed") fail(`the fly did not land within 20 s (mode ${life.mode})`);
  else {
    const on = life.fruitAt(flight.p);
    if (on !== "banana") fail(`the fly landed on ${on} at ${flight.p.map((v) => v.toFixed(3))}, not on the moved banana at ${target}`);
    if (!near(flight.p, target, 0.05)) fail(`the fly sits at ${flight.p.map((v) => v.toFixed(3))}, the banana at ${target}`);  // the landing spot's 1.5 cm jitter and the 3 cm settling radius
    if (Math.abs(flight.p[2] - (room.fruits.banana.pos[2] + 1.2e-3)) > 0.005) fail(`the fly sits ${(100 * flight.p[2]).toFixed(1)} cm up, the banana's top is at ${(100 * room.fruits.banana.pos[2]).toFixed(1)} cm`);
    if (life.visits.banana !== 1 || life.visits.sweet !== 1) fail(`visits ${JSON.stringify(life.visits)}: the moved banana carries the sugar`);
    if (failures === 0) ok(`the descent follows the fruit: landed on the banana ${(30).toFixed(0)} cm from where the descent began, after ${(steps * DT).toFixed(1)} s`);
  }
}

// -- a fly standing on the bread is shaken off when the bread is moved ---------------------------
{
  const room = stubRoom(), Br = room.fruits.bread.pos;
  const flight = new Flight([Br[0], Br[1], Br[2] + 1.2e-3], 0.0), life = new Life(flight, room, 3);
  life.mode = "landed"; life.perch = Br.slice(); life.hold(); life.sit = 5; life.ethogram.sitStart = 0;
  life.search = { since: 0, fruit: "bread" };
  const shaken = move(room, life, "bread", [Br[0] - 0.3, Br[1] + 0.2]);
  if (!shaken) fail("the fly on the bread was not shaken off");
  if (life.mode !== "flying" || life.perch !== null) fail(`after the move the fly is ${life.mode} with perch ${life.perch}`);
  if (life.search !== null || !life.pendingReward || life.pendingReward.reward !== 0) fail("the open search was not closed with nothing");
  for (let k = 0; k < 0.5 / DT; k++) life.step(DT);
  if (life.mode !== "flying") fail("the shaken fly did not stay in the air");
  if (failures === 0) ok("a fly standing on the bread is shaken off when the bread is moved, and its search ends with nothing");
}

// -- a fly on the bare table stays; a fly where a fruit is set down is startled off --------------
{
  const room = stubRoom(), t = room.table, spot = [t.x0 + 0.1, t.y0 + 0.1];
  for (const F of Object.values(room.fruits)) if (Math.hypot(spot[0] - F.pos[0], spot[1] - F.pos[1]) < FRUIT_FOOTPRINT + 0.05) fail("the test spot is too close to a fruit");
  const flight = new Flight([spot[0], spot[1], t.z + 1.2e-3], 0.0), life = new Life(flight, room, 4);
  life.mode = "landed"; life.perch = [spot[0], spot[1], t.z]; life.hold(); life.sit = 5; life.ethogram.sitStart = 0;
  if (move(room, life, "banana", [t.x1 - 0.1, t.y0 + 0.1])) fail("a fly on the bare table was shaken off by a banana moved elsewhere");
  if (life.mode !== "landed") fail("the fly on the table did not stay");
  if (!move(room, life, "banana", [spot[0] + 0.02, spot[1] - 0.02])) fail("a fly where the banana is set down was not startled");
  if (life.mode !== "flying") fail("the startled fly is not flying");
  if (failures === 0) ok("a fly on the bare table stays; a fly where a fruit is set down is startled off");
}

// -- a landing on the bare table lands at the table's height, where a fruit is not ----------------
{
  const room = stubRoom(), t = room.table;
  const flight = new Flight([t.x0 + 0.2, t.y0 + 0.2, t.z + 0.3], 0.0), life = new Life(flight, room, 5);
  life.hunger = 0.0; life.senses(null, 0);
  life.chooseLanding();
  if (life.landingFruit) fail("a landing with no appetite is to a fruit");
  if (Math.abs(life.landing[2] - t.z) > 1e-9) fail(`the table landing is at ${life.landing[2]}, the table at ${t.z}`);
  if (failures === 0) ok("a landing on the bare table is at the table's height");
}

if (failures) { console.log(`${failures} failure(s)`); process.exit(1); }
console.log("all passed");
