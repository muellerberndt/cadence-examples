// Pure actual-trail and constant-command guide checks. No rendering or learned
// navigation claim; projection is intentionally separate from body mechanics.
// Run: node tests/path-history.mjs
import assert from "node:assert/strict";
import { PathHistory, projectCommand, PATH_MAX_POINTS, PATH_MAX_EVENTS, PROJECTION_MAX_POINTS } from "../web/path-history.js";

const command = (forwardSpeed = .2, yawRate = 0, verticalSpeed = 0) => ({ forwardSpeed, yawRate, verticalSpeed });
const frame = (time, overrides = {}) => ({ generation: 1, time, position: [1, 2, 3], heading: 0,
  command: command(), active: true, attention: null, action: null, observationId: 1, deadline: 2, ...overrides });
const near = (a, b, epsilon = 1e-12) => assert.ok(Math.abs(a - b) <= epsilon, `${a} != ${b}`);
const endpoint = points => points.at(-1);
const guide = input => { const history = new PathHistory(); history.record(input); return projectCommand(history.snapshot()); };
let tests = 0;
function test(name, run) { run(); tests++; console.log(`ok ${name}`); }

test("trail stores measured positions even when they disagree with the command", () => {
  const history = new PathHistory();
  history.record(frame(0));
  history.record(frame(.025, { position: [20, 21, 22] }));
  history.record(frame(.05, { position: [-5, -6, -7] }));
  const snapshot = history.snapshot();
  assert.equal(snapshot.points.length, 2);
  assert.deepEqual(snapshot.points[1].position, [-5, -6, -7]);
  assert.equal(snapshot.points[1].command.forwardSpeed, .2);
  const projected = projectCommand(snapshot);
  assert.deepEqual(projected[0].position, [-5, -6, -7]);
  near(endpoint(projected).position[0], -4.8);
  assert.deepEqual(history.snapshot().points, snapshot.points, "projection must never append speculative history");
});

test("fixed cadence uses actual calls and never invents missed positions", () => {
  const history = new PathHistory();
  for (let k = 0; k <= 200; k++) history.record(frame(k * .0005, { position: [k, 0, 0] }));
  assert.deepEqual(history.snapshot().points.map(p => p.position[0]), [0, 100, 200]);
  history.record(frame(1, { position: [999, 0, 0] }));
  assert.deepEqual(history.snapshot().points.map(p => p.position[0]), [0, 100, 200, 999]);
  history.record(frame(1, { position: [1000, 0, 0] }));
  assert.equal(history.snapshot().points.length, 4, "repeated render timestamps do not add actual trail points");
});

test("command reversals and target/action changes are recorded between position samples", () => {
  const history = new PathHistory();
  history.record(frame(0));
  history.record(frame(.01, { command: command(0), attention: "bread", action: 0 }));
  history.record(frame(.02, { command: command(-.2), attention: "bread", action: 1, observationId: 2 }));
  const snapshot = history.snapshot();
  assert.equal(snapshot.points.length, 1); assert.equal(snapshot.events.length, 3);
  assert.ok(snapshot.events[1].changes.includes("command"));
  assert.ok(snapshot.events[1].changes.includes("attention"));
  assert.ok(snapshot.events[2].changes.includes("action"));
  assert.equal(snapshot.events[2].command.forwardSpeed, -.2);
  assert.equal(snapshot.current.time, .02);
});

test("brain-off/pause preserves actual trail and immediately removes the future guide", () => {
  const history = new PathHistory(); history.record(frame(0));
  assert.ok(projectCommand(history.snapshot()).length > 1);
  history.record(frame(.05, { position: [1.01, 2, 3], active: false, command: command(0), observationId: null, deadline: null }));
  const stopped = history.snapshot();
  assert.equal(stopped.points[0].active, true); assert.equal(stopped.points[1].active, false);
  assert.deepEqual(projectCommand(stopped), []);
  history.record(frame(.07, { position: [1.04, 2, 3], active: false, command: command(0), observationId: null, deadline: null }));
  assert.deepEqual(projectCommand(history.snapshot()), [], "stabilizer/inertial movement is not a neural future route");
});

test("an interruption between active endpoints marks the sampled segment and unsampled tail inactive", () => {
  const history = new PathHistory({ maxEvents: 1 });
  history.record(frame(0));
  history.record(frame(.01, { active: false }));
  history.record(frame(.02));
  let snapshot = history.snapshot();
  assert.equal(snapshot.points.length, 1, "the interruption must not invent a position sample");
  assert.equal(snapshot.events.length, 1, "the interruption need not remain in the bounded event log");
  assert.equal(snapshot.current.active, true);
  assert.equal(snapshot.current.intervalActive, false, "an active endpoint cannot erase the interrupted tail");
  history.record(frame(.05));
  snapshot = history.snapshot();
  assert.deepEqual(snapshot.points.map(p => p.active), [true, true]);
  assert.deepEqual(snapshot.points.map(p => p.intervalActive), [true, false]);
  assert.equal(snapshot.current.intervalActive, false);
  history.record(frame(.075));
  assert.equal(history.snapshot().current.intervalActive, true, "the next uninterrupted tail can recover");
  history.record(frame(.1));
  assert.deepEqual(history.snapshot().points.map(p => p.intervalActive), [true, false, true]);
});

test("same-time authority changes cannot rewrite saved interval flags", () => {
  const history = new PathHistory();
  history.record(frame(0)); history.record(frame(.05));
  const before = history.snapshot();
  history.record(frame(.05, { active: false }));
  history.record(frame(.05));
  let snapshot = history.snapshot();
  assert.equal(snapshot.current.intervalActive, false);
  assert.deepEqual(snapshot.points, before.points, "recorded endpoints retain their original interval status");
  history.record(frame(.1));
  snapshot = history.snapshot();
  assert.equal(snapshot.points.at(-1).intervalActive, false);
  assert.equal(before.points.at(-1).intervalActive, true, "detached snapshots remain unchanged");
  assert.throws(() => { snapshot.current.intervalActive = true; }, TypeError);
});

test("inactive endpoints and generation resets set the interval convention without trusting caller flags", () => {
  const history = new PathHistory();
  history.record(frame(0, { active: false, intervalActive: true }));
  assert.equal(history.snapshot().points[0].intervalActive, false);
  history.record(frame(.01)); history.record(frame(.05));
  assert.equal(history.snapshot().points[1].intervalActive, false, "the previous inactive endpoint counts");
  history.record(frame(0, { generation: 2, intervalActive: false }));
  assert.equal(history.snapshot().points[0].intervalActive, true, "new generations do not inherit interruptions");
  history.record(frame(.01, { generation: 2, active: false }));
  history.reset(2); history.record(frame(0, { generation: 2 }));
  assert.equal(history.snapshot().current.intervalActive, true);
});

test("constant yaw follows the analytic arc for signed velocity and yaw", () => {
  for (const speed of [-.3, .3]) for (const yaw of [-1.2, 1.2]) {
    const input = frame(0, { heading: .4, command: command(speed, yaw, -.1) });
    const end = endpoint(guide(input));
    near(end.position[0], 1 + speed / yaw * (Math.sin(.4 + yaw) - Math.sin(.4)));
    near(end.position[1], 2 + speed / yaw * (Math.cos(.4) - Math.cos(.4 + yaw)));
    near(end.position[2], 2.9); near(end.heading, .4 + yaw);
  }
});

test("straight and near-zero-yaw guides are continuous through a signed-speed zero crossing", () => {
  for (const speed of [-.2, 0, .2]) for (const yaw of [-1e-12, 0, 1e-12]) {
    const projected = guide(frame(0, { heading: Math.PI / 2, command: command(speed, yaw) }));
    if (speed === 0) assert.deepEqual(projected, []);
    else { near(endpoint(projected).position[0], 1); near(endpoint(projected).position[1], 2 + speed); }
  }
  assert.deepEqual(guide(frame(0, { command: command(0, 2, 0) })), [], "turn-in-place is not a fake spatial path");
  const vertical = endpoint(guide(frame(0, { command: command(0, 2, .12) })));
  assert.deepEqual(vertical.position.slice(0, 2), [1, 2]); near(vertical.position[2], 3.12);
});

test("projection is limited by remaining command life and the one-second display horizon", () => {
  const history = new PathHistory(); history.record(frame(5, { deadline: 5.12 }));
  near(endpoint(projectCommand(history.snapshot(), { horizon: 99 })).time, 5.12);
  near(endpoint(projectCommand(history.snapshot(), { horizon: .03 })).time, 5.03);
  history.record(frame(5.12, { deadline: 5.12 }));
  assert.deepEqual(projectCommand(history.snapshot()), []);
  assert.deepEqual(guide(frame(5, { deadline: 4 })), []);
  assert.deepEqual(guide(frame(0, { deadline: null })), []);
  const unexpired = new PathHistory(); unexpired.record(frame(0, { deadline: 50 }));
  assert.equal(endpoint(projectCommand(unexpired.snapshot(), { horizon: 50 })).time, 1);
  assert.ok(projectCommand(unexpired.snapshot(), { step: 1e-30 }).length <= PROJECTION_MAX_POINTS);
});

test("empty, inactive, invalid and mismatched-generation snapshots cannot create a guide", () => {
  assert.deepEqual(projectCommand(new PathHistory().snapshot()), []);
  assert.deepEqual(projectCommand(null), []);
  assert.deepEqual(projectCommand({ generation: 2, current: frame(0) }), []);
  assert.deepEqual(projectCommand({ generation: 1, current: frame(0, { heading: NaN }) }), []);
  assert.deepEqual(guide(frame(0, { active: false })), []);
});

test("generation reset breaks the trail and stale generations/time rewinds cannot contaminate it", () => {
  const history = new PathHistory(); history.record(frame(10));
  history.record(frame(0, { generation: 2, position: [100, 100, 100] }));
  const clean = history.snapshot();
  assert.equal(clean.generation, 2); assert.equal(clean.points.length, 1); assert.equal(clean.events.length, 1);
  assert.deepEqual(clean.points[0].position, [100, 100, 100]);
  assert.throws(() => history.record(frame(11)), /stale_path_generation/);
  history.record(frame(1, { generation: 2 }));
  assert.throws(() => history.record(frame(.5, { generation: 2 })), /path_time_rewound/);
  history.reset(2); history.record(frame(0, { generation: 2 }));
  assert.equal(history.snapshot().points.length, 1);
});

test("positions and events stay bounded and snapshots own frozen copies", () => {
  const history = new PathHistory({ maxPoints: 3, maxEvents: 2 }), original = frame(0);
  history.record(original); original.position[0] = 999; original.command.forwardSpeed = 999;
  assert.equal(history.snapshot().current.position[0], 1);
  for (let k = 1; k <= 100; k++) history.record(frame(k * .05, { position: [k, 0, 0], observationId: k }));
  const snapshot = history.snapshot();
  assert.equal(snapshot.points.length, 3); assert.equal(snapshot.events.length, 2);
  assert.deepEqual(snapshot.points.map(p => p.position[0]), [98, 99, 100]);
  assert.throws(() => { snapshot.points[0].position[0] = -100; }, TypeError);
  assert.throws(() => { snapshot.current.command.forwardSpeed = -100; }, TypeError);
  assert.throws(() => { snapshot.events[0].changes.push("invented"); }, TypeError);
  assert.throws(() => { snapshot.points.push(frame(2)); }, TypeError);
  history.record(frame(6, { position: [200, 0, 0] }));
  assert.deepEqual(snapshot.points.map(p => p.position[0]), [98, 99, 100]);
});

test("geometry metadata cannot alter the command-only guide", () => {
  const input = frame(0), baseline = guide(input);
  const changed = guide({ ...input, fruitPosition: [100, -5, 50], targetBearing: 2, room: { width: 1000 } });
  assert.deepEqual(changed, baseline);
});

test("invalid recorder and projection inputs reject without mutating saved history", () => {
  for (const config of [{ interval: 0 }, { interval: NaN }, { maxPoints: 0 }, { maxPoints: PATH_MAX_POINTS + 1 },
    { maxEvents: PATH_MAX_EVENTS + 1 }, { maxEvents: 1.5 }]) assert.throws(() => new PathHistory(config));
  const history = new PathHistory(); history.record(frame(0)); const before = history.snapshot();
  for (const bad of [frame(-1), frame(.01, { position: [1, 2] }), frame(.01, { position: [NaN, 2, 3] }),
    frame(.01, { command: command(Infinity) }), frame(.01, { generation: 1.5 }), frame(.01, { active: 1 }),
    frame(.01, { action: 2 }), frame(.01, { observationId: -1 }), frame(.01, { deadline: NaN })]) {
    assert.throws(() => history.record(bad)); assert.deepEqual(history.snapshot(), before);
  }
  for (const options of [{ horizon: -1 }, { horizon: NaN }, { step: 0 }, { step: Infinity }]) {
    assert.throws(() => projectCommand(before, options));
  }
});

console.log(`${tests} path history/projection groups passed`);
