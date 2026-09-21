// The body under stress: irritants crowded round the worm, pokes and treats at random,
// several plates. After every physics step the centreline must be one body length, inside
// the plate, and nowhere bent tighter than the body can bend.
//   node worm/tests/body.mjs
import { readFileSync, existsSync } from "node:fs";
import { Life, mulberry32 } from "../web/life.js";

const root = new URL("..", import.meta.url).pathname;
const load = (f) => JSON.parse(readFileSync(root + f));
const spec = load("web/data/brain.json"), p = load("web/data/params.json");
const L = p.body_length, DS = 0.05;

function resample(tr, ds) {          // points every ds along the centreline
  const out = [tr[0]]; let want = ds, acc = 0;
  for (let k = 1; k < tr.length; k++) {
    const seg = Math.hypot(tr[k][0] - tr[k - 1][0], tr[k][1] - tr[k - 1][1]);
    while (seg > 0 && acc + seg >= want) { const f = (want - acc) / seg; out.push([tr[k - 1][0] + f * (tr[k][0] - tr[k - 1][0]), tr[k - 1][1] + f * (tr[k][1] - tr[k - 1][1])]); want += ds; }
    acc += seg;
  }
  return out;
}
function check(life, where) {
  const tr = life.body.trail; let len = 0;
  for (let k = 1; k < tr.length; k++) len += Math.hypot(tr[k][0] - tr[k - 1][0], tr[k][1] - tr[k - 1][1]);
  if (Math.abs(len - L) > 1e-9) throw new Error(`${where}: the body is ${len} mm long, not ${L}`);
  for (const [x, y] of tr) if (!(x >= 0 && x <= life.w && y >= 0 && y <= life.h)) throw new Error(`${where}: the body left the plate at ${x}, ${y}`);
  if (tr[0][0] !== life.body.x || tr[0][1] !== life.body.y) throw new Error(`${where}: the head is not the first point of the body`);
  // exact: no vertex turns by more than one physics step at the tightest bend allows
  const most = p.max_curvature * Math.max(p.crawl_speed, p.reverse_speed) * p.physics_step * (1 + 1e-9);
  for (let k = 1; k < tr.length - 1; k++) {
    const ax = tr[k][0] - tr[k - 1][0], ay = tr[k][1] - tr[k - 1][1], bx = tr[k + 1][0] - tr[k][0], by = tr[k + 1][1] - tr[k][1];
    if (Math.hypot(ax, ay) < 1e-7 || Math.hypot(bx, by) < 1e-7) continue;     // a trimmed end segment has no reliable direction
    const turn = Math.abs(Math.atan2(ax * by - ay * bx, ax * bx + ay * by));
    if (turn > most) throw new Error(`${where}: a kink of ${(turn * 180 / Math.PI).toFixed(1)} degrees at point ${k} of ${tr.length}`);
  }
  const r = resample(tr, DS); let sharpest = 0;
  for (let k = 1; k < r.length - 1; k++) {
    const a = Math.atan2(r[k][1] - r[k - 1][1], r[k][0] - r[k - 1][0]), b = Math.atan2(r[k + 1][1] - r[k][1], r[k + 1][0] - r[k][0]);
    let d = Math.abs(a - b); if (d > Math.PI) d = 2 * Math.PI - d;
    sharpest = Math.max(sharpest, d);
  }
  return sharpest / DS;               // the tightest curvature along the body, rad/mm
}

const steps = Math.round(p.tick / p.physics_step);
let worst = 0;
for (const [seed, scene] of [[1, "ring"], [2, "ring"], [3, "ring"], [4, "corner"], [5, "carpet"], [6, "plain"]]) {
  const life = new Life(spec, p, seed), hand = mulberry32(1000 + seed);
  if (scene === "ring") for (let k = 0; k < 6; k++) life.place("noxious", life.body.x + 0.9 * Math.cos(k / 6 * 2 * Math.PI), life.body.y + 0.9 * Math.sin(k / 6 * 2 * Math.PI), "B");
  if (scene === "carpet") for (let x = 1; x < life.w; x += 0.7) for (let y = 1; y < life.h; y += 0.7) life.place("noxious", x, y, "B");
  if (scene === "corner") { Object.assign(life.body, { heading: -2.4 }); life.place("food", 0.5, 0.5, "A", true); }
  life.place("food", life.body.x + 0.3, life.body.y, "A", true);
  let reversals = 0, mode = "forward", tight = 0, edge = 1e9;
  for (let k = 0; k < 1500; k++) {
    for (let s = 0; s < steps; s++) {
      life.step(); tight = Math.max(tight, check(life, `seed ${seed} ${scene} tick ${k}`));
      if (life.body.mode === "reverse" && mode === "forward") reversals++; mode = life.body.mode;
      for (const [x, y] of life.body.trail) edge = Math.min(edge, x, y, life.w - x, life.h - y);
    }
    const r = life.think();
    for (const v of life.activity) if (!Number.isFinite(v)) throw new Error(`seed ${seed}: activity is not finite`);
    if (life.brain.growth() > p.stability + 1e-9) throw new Error(`seed ${seed}: recurrent gain ${life.brain.growth()}`);
    if (hand() < 0.03) life.poke(); if (hand() < 0.02) life.treat();
  }
  worst = Math.max(worst, tight);
  console.log(`${scene.padEnd(6)} seed ${seed}: ${reversals} reversals, ${life.lessons} lessons, tightest bend ${tight.toFixed(1)} rad/mm, nearest the edge ${edge.toFixed(3)} mm, gain ${life.brain.growth().toFixed(3)}`);
}
const bound = p.max_curvature * 1.4;   // chords of 0.05 mm across 0.014 mm steps overstate a bounded curvature by up to 28%
if (worst > bound) throw new Error(`a bend of ${worst.toFixed(1)} rad/mm exceeds ${bound}`);

// the Python reference of the body lays the same track
if (existsSync(root + "tests/body_cases.json")) {
  for (const c of load("tests/body_cases.json")) {
    const life = new Life(spec, p, c.seed); let worstGap = 0;
    for (const [what, a, b] of c.script) {
      if (what === "step") for (let k = 0; k < a; k++) life.physics(p.physics_step, false);
      if (what === "reverse") life.reverse(a, b);
      if (what === "steer") life.body.heading += a;
      if (what === "move") { const dx = a - life.body.x, dy = b - life.body.y; life.body.trail = life.body.trail.map(([x, y]) => [x + dx, y + dy]); life.body.x = a; life.body.y = b; }
    }
    if (life.body.trail.length !== c.trail.length) throw new Error(`body case ${c.seed}: ${life.body.trail.length} points against Python's ${c.trail.length}`);
    life.body.trail.forEach(([x, y], k) => { worstGap = Math.max(worstGap, Math.abs(x - c.trail[k][0]), Math.abs(y - c.trail[k][1])); });
    if (worstGap > 1e-9) throw new Error(`body case ${c.seed}: the track differs from Python's by ${worstGap} mm`);
    console.log(`body case seed ${c.seed}: ${c.trail.length} points, largest difference from the Python body ${worstGap.toExponential(1)} mm`);
  }
}
console.log("BODY OK");
