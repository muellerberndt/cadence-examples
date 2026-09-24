// The JavaScript body against the Python body, and the invariants of the flight.
//   node tests/body.mjs
// Replays every script in tests/body_cases.json through web/body.js and requires positions and
// quaternions to agree with the Python samples to a relative 1e-12. The twin sine, cosine and
// arctangent must agree bit for bit. Along the way: the fly never leaves the room, the speed stays
// below 3 m/s, the pilot holds hover altitude within 5 mm for 5 s, a 90 degree saccade is inside 5
// degrees within 60 ms, a fly with its wings off lands, and open-loop hover diverges.
import { readFileSync } from "node:fs";
import { DT, Flight, HandPilot, ROOM, hoverTrim, sin_, cos_, atan2_, asin_, wrap_ } from "../web/body.js";

const root = new URL("..", import.meta.url).pathname;
const data = JSON.parse(readFileSync(root + "tests/body_cases.json"));
const TOL = 1e-12;
let failures = 0;
const fail = (msg) => { failures++; console.log("FAIL " + msg); };
const ok = (msg) => console.log("ok   " + msg);

// -- the twin functions, bit for bit ----------------------------------------------------------
{
  const m = data.math;
  let bad = 0;
  for (let i = 0; i < m.x.length; i++) { if (sin_(m.x[i]) !== m.sin[i]) bad++; if (cos_(m.x[i]) !== m.cos[i]) bad++; }
  for (let i = 0; i < m.yx.length; i++) if (atan2_(m.yx[i][0], m.yx[i][1]) !== m.atan2[i]) bad++;
  for (let i = 0; i < m.u.length; i++) if (asin_(m.u[i]) !== m.asin[i]) bad++;
  if (bad) fail(`twin math: ${bad} values differ from Python`); else ok(`twin math: ${2 * m.x.length + m.yx.length + m.u.length} values identical`);
}

// -- one case through the JavaScript body -----------------------------------------------------
function tiltDeg(fl) { const c = fl.rotation()[8]; return Math.acos(Math.max(-1, Math.min(1, c))) * 180 / Math.PI; }

function runCase(cs) {
  const st = cs.start;
  const fl = new Flight(st.position, st.heading ?? 0.0);
  if (st.velocity) fl.v = st.velocity.slice();
  if (st.attitude) fl.setAttitude(...st.attitude);
  let pilot = null;
  if (cs.pilot) { pilot = new HandPilot(cs.pilot.target, cs.pilot.heading ?? 0.0, cs.pilot.speed ?? 0.3); if (cs.pilot.route) pilot.flyRoute(cs.pilot.route); }
  let fixed = { ...(cs.controls ?? hoverTrim()) };
  const events = new Map();
  for (const ev of cs.events ?? []) { if (!events.has(ev.step)) events.set(ev.step, []); events.get(ev.step).push(ev); }
  const inv = { maxSpeed: 0, outside: 0, altDev: 0, tiltMax: 0, saccadeAt: null, saccadeDone: null, landedAt: null, headingErr: [] };
  const samples = [];
  for (let k = 0; k < cs.steps; k++) {
    for (const ev of events.get(k) ?? []) {
      if ("saccade" in ev) { pilot.saccade(ev.saccade); inv.saccadeAt = k; }
      if ("swat" in ev) fl.swat(...ev.swat);
      if ("kick" in ev) fl.kick(...ev.kick);
      if ("target" in ev) pilot.target = ev.target.slice();
      if ("controls" in ev) fixed = { ...ev.controls };
    }
    const c = pilot ? pilot.controls(fl) : fixed;
    fl.step(DT, c);
    if ((k + 1) % cs.every === 0) samples.push([k + 1, fl.p.slice(), fl.q.slice(), fl.v.slice(), fl.w.slice()]);
    // invariants, every step
    for (let i = 0; i < 3; i++) if (!(fl.p[i] >= 0 && fl.p[i] <= ROOM[i])) inv.outside++;
    inv.maxSpeed = Math.max(inv.maxSpeed, fl.speed());
    inv.altDev = Math.max(inv.altDev, Math.abs(fl.p[2] - st.position[2]));
    inv.tiltMax = Math.max(inv.tiltMax, tiltDeg(fl));
    if (inv.saccadeAt !== null) inv.headingErr.push(Math.abs(wrap_(pilot.heading - fl.euler()[2])));
    if (fl.landed && inv.landedAt === null) inv.landedAt = fl.t;
    for (const x of [...fl.p, ...fl.q, ...fl.v, ...fl.w]) if (!Number.isFinite(x)) throw new Error(`${cs.name}: state is not finite at step ${k}`);
  }
  if (inv.saccadeAt !== null) {    // the last moment the heading error was above 5 degrees
    let last = -1;
    inv.headingErr.forEach((e, i) => { if (e > 5 * Math.PI / 180) last = i; });
    inv.saccadeDone = (last + 1) * DT;
  }
  return { samples, inv, fl };
}

function compare(cs, samples) {
  let worstP = 0, worstQ = 0;
  for (let i = 0; i < cs.samples.length; i++) {
    const [step, p, q] = cs.samples[i], [step2, p2, q2] = samples[i];
    if (step !== step2) throw new Error(`${cs.name}: sample ${i} is step ${step2}, Python has ${step}`);
    const dp = Math.sqrt((p[0] - p2[0]) ** 2 + (p[1] - p2[1]) ** 2 + (p[2] - p2[2]) ** 2) / Math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2);
    const dq = Math.sqrt((q[0] - q2[0]) ** 2 + (q[1] - q2[1]) ** 2 + (q[2] - q2[2]) ** 2 + (q[3] - q2[3]) ** 2);
    worstP = Math.max(worstP, dp); worstQ = Math.max(worstQ, dq);
  }
  return { worstP, worstQ };
}

for (const cs of data.cases) {
  const { samples, inv } = runCase(cs);
  if (samples.length !== cs.samples.length) { fail(`${cs.name}: ${samples.length} samples, Python has ${cs.samples.length}`); continue; }
  const { worstP, worstQ } = compare(cs, samples);
  const exact = worstP === 0 && worstQ === 0;
  if (worstP > TOL || worstQ > TOL) fail(`${cs.name}: parity, position ${worstP.toExponential(2)} quaternion ${worstQ.toExponential(2)}`);
  else ok(`${cs.name.padEnd(8)} parity over ${cs.steps} steps: ${exact ? "bit-identical" : `position ${worstP.toExponential(2)}, quaternion ${worstQ.toExponential(2)}`}`);
  if (inv.outside) fail(`${cs.name}: the fly left the room in ${inv.outside} steps`);
  if (inv.maxSpeed >= 3) fail(`${cs.name}: speed reached ${inv.maxSpeed.toFixed(2)} m/s`);
  if (cs.name === "hover") {
    if (inv.altDev > 5e-3) fail(`hover: altitude wandered ${(inv.altDev * 1e3).toFixed(2)} mm`); else ok(`hover    altitude held within ${(inv.altDev * 1e3).toFixed(3)} mm over ${(cs.steps * DT).toFixed(0)} s`);
  }
  if (cs.name === "saccade") {
    if (inv.saccadeDone > 0.060) fail(`saccade: inside 5 degrees only after ${(inv.saccadeDone * 1e3).toFixed(1)} ms`); else ok(`saccade  90 degrees inside 5 degrees from ${(inv.saccadeDone * 1e3).toFixed(1)} ms on`);
  }
  if (cs.name === "landing") {
    if (inv.landedAt === null) fail("landing: the fly never came to rest on the floor"); else ok(`landing  at rest on the floor after ${(inv.landedAt * 1e3).toFixed(0)} ms`);
  }
  if (cs.name === "tumble") {
    if (inv.tiltMax < 60) fail(`tumble: open-loop hover tilted only ${inv.tiltMax.toFixed(1)} degrees`); else ok(`tumble   open-loop hover diverged to ${inv.tiltMax.toFixed(0)} degrees of tilt within ${(cs.steps * DT).toFixed(1)} s`);
  }
  if (cs.name === "burst") ok(`burst    top speed ${inv.maxSpeed.toFixed(2)} m/s`);
  if (cs.name === "lap") ok(`lap      top speed ${inv.maxSpeed.toFixed(2)} m/s, altitude within ${(inv.altDev * 1e3).toFixed(1)} mm`);
}

if (failures) { console.log(`${failures} failure(s)`); process.exit(1); }
console.log("all passed");
