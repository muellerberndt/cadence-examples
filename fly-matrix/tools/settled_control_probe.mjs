#!/usr/bin/env node
// Bounded real-payload integration diagnostic. No renderer, training, pilot or policy fallback.
// Run: node tools/settled_control_probe.mjs --out receipts/settled_control_probe.json
// Check bindings/invariants without rerunning: node tools/settled_control_probe.mjs --verify receipts/settled_control_probe.json
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { SettlingBrain } from "../web/brain.js";
import { Flight, DT, CONTROL_KEYS, HandPilot } from "../web/body.js";
import { SettledLife } from "../web/settled-life.js";
import { SETTLED_MOTOR_GROUPS } from "../web/motor.js";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const SOURCE_PATHS = ["tools/settled_control_probe.mjs", "web/brain.js", "web/body.js", "web/settled-life.js", "web/motor.js", "web/senses.js", "web/data/brain.json"];
const digest = value => createHash("sha256").update(value).digest("hex");
const sourceHashes = () => SOURCE_PATHS.map(path => ({ path, sha256: digest(readFileSync(resolve(ROOT, path))) }));

// Frozen before the first run. This one deterministic initial condition is a smoke diagnostic,
// not a selected successful trajectory or a behavioral acquisition/generalization experiment.
const PROTOCOL = {
  schema: "cadence.fly-matrix.settled-control-protocol/v1",
  seed_label: 1, rng_used: false,
  duration_s: 0.5, decision_period_s: 0.02, physics_dt_s: 0.0005,
  max_steps: 256, equation_tolerance: 1e-6, command_ttl_s: 0.12,
  initial: { position: [2, 1.5, 0.5], attitude_radians: [0, 0, 0], velocity: [0, 0, 0], angular_velocity: [0, 0, 0] },
  room: { dimensions_m: [4, 3, 2.6], geometry: "existing Flight floor, walls and ceiling only", fruits: { banana: [1.75, 1.5, 0], bread: [2.25, 1.5, 0] } },
  arms: ["measured", "zero_motor"],
  intervention: "zero_motor replaces all declared motor readouts by zero after the same real-payload solve; it is an output lesion, not a shuffled or whole-brain lesion",
  sensory_source: "SettledLife.senses(null, decision_period_s), frozen throughout each solve",
  scheduling: "synchronous offline sample-and-hold; simulated physics pauses during compute; 40 body steps follow each command",
  learning: false, teacher: false, renderer: false,
  descriptive_thresholds: { lost_height_m: 0.1, severe_tilt_radians: Math.PI / 2 },
  scope: "short integration and actuator-authority diagnostic; no flight competence, real-time performance, unique-equilibrium, whole-connectome or biological-fidelity claim",
};
const count = Math.round(PROTOCOL.duration_s / PROTOCOL.decision_period_s);
const bodySteps = Math.round(PROTOCOL.decision_period_s / PROTOCOL.physics_dt_s);
const zeroControls = () => Object.fromEntries(CONTROL_KEYS.map(key => [key, 0]));
const tilt = flight => Math.acos(Math.min(1, Math.max(-1, flight.rotation()[8])));
const state = flight => ({ position: [...flight.p], velocity: [...flight.v], quaternion: [...flight.q], angular_velocity: [...flight.w], phase: flight.phase, tilt_radians: tilt(flight), speed_m_s: flight.speed(), touching: flight.touching, on_floor: flight.onFloor, landed: flight.landed });
const finiteState = flight => [...flight.p, ...flight.v, ...flight.q, ...flight.w, flight.phase].every(Number.isFinite);
const summarizeLatency = values => {
  const sorted = [...values].sort((a, b) => a - b);
  return { total: values.reduce((a, b) => a + b, 0), mean: values.reduce((a, b) => a + b, 0) / values.length, median: sorted[Math.floor(sorted.length / 2)], maximum: sorted.at(-1) };
};

function checkReceipt(receipt) {
  assert.equal(receipt.schema, "cadence.fly-matrix.settled-control-probe/v1");
  const { sha256, ...body } = receipt;
  assert.equal(sha256, digest(JSON.stringify(body)), "receipt body changed");
  assert.deepEqual(receipt.protocol, PROTOCOL, "protocol differs from the frozen source");
  assert.deepEqual(receipt.sources, sourceHashes(), "source or payload binding changed");
  assert.deepEqual(receipt.results.map(r => r.arm), PROTOCOL.arms, "missing or reordered arm");
  for (const arm of receipt.results) {
    assert.equal(arm.rows.length, count, "missing scheduled solve");
    assert.equal(arm.body_steps, count * bodySteps);
    for (const [index, row] of arm.rows.entries()) {
      assert.equal(row.request_id, index + 1);
      assert.equal(row.t_s, index * PROTOCOL.decision_period_s);
      assert.equal(row.solve.tolerance, PROTOCOL.equation_tolerance);
      assert.ok(Number.isInteger(row.solve.iterations) && row.solve.iterations >= 0 && row.solve.iterations <= PROTOCOL.max_steps);
      assert.ok(Number.isFinite(row.solve_ms) && row.solve_ms >= 0);
      if (row.solve.converged) {
        assert.ok(Number.isFinite(row.solve.residual) && row.solve.residual >= 0 && row.solve.residual <= row.solve.tolerance);
        assert.equal(row.accepted, true);
      } else assert.equal(row.accepted, false);
      assert.ok(Object.values(row.controls).every(Number.isFinite));
      if (!row.accepted || arm.arm === "zero_motor") {
        assert.ok(Object.values(row.controls).every(value => value === 0));
        assert.equal(row.proboscis, 0);
      }
    }
    assert.equal(arm.finite_body, true);
    assert.equal(arm.solver_failures, arm.rows.filter(row => !row.solve.converged).length);
  }
  const zero = receipt.results.find(r => r.arm === "zero_motor");
  assert.equal(zero.active_command_body_steps, 0);
  assert.equal(zero.passive_trajectory_exact_match, true);
  assert.equal(receipt.flight_competency_established, false);
}

function runArm(payload, arm) {
  const brain = new SettlingBrain(payload), initial = PROTOCOL.initial;
  const flight = new Flight(initial.position, initial.attitude_radians[2]);
  flight.setAttitude(...initial.attitude_radians);
  flight.v = [...initial.velocity]; flight.w = [...initial.angular_velocity];
  const room = { fruits: Object.fromEntries(Object.entries(PROTOCOL.room.fruits).map(([name, pos]) => [name, { pos: [...pos] }])), surfaceZ: () => 0 };
  const life = new SettledLife(flight, room, PROTOCOL.seed_label, { generation: 1, commandTTL: PROTOCOL.command_ttl_s, tolerance: PROTOCOL.equation_tolerance });
  const passive = arm === "zero_motor" ? new Flight(initial.position, initial.attitude_radians[2]) : null;
  if (passive) { passive.setAttitude(...initial.attitude_radians); passive.v = [...initial.velocity]; passive.w = [...initial.angular_velocity]; }
  for (const name of SETTLED_MOTOR_GROUPS) assert.ok(brain.sets[name]?.length > 0, `payload omits required motor group ${name}`);
  const rows = [], initialState = state(flight);
  let maxTilt = 0, maxSpeed = 0, minimumHeight = initial.position[2], firstContact = null, severeTiltAt = null;
  let activeSteps = 0, passiveMatches = true, maxFrequency = 0, maxAmplitude = 0, maxProboscis = 0;
  const started = performance.now();
  for (let tick = 0; tick < count; tick++) {
    const request = { requestId: tick + 1, generation: 1, steps: PROTOCOL.max_steps, tolerance: PROTOCOL.equation_tolerance };
    assert.equal(life.expectControl(request), true);
    const senses = life.senses(null, PROTOCOL.decision_period_s);
    brain.clearStimuli();
    for (const [name, level] of Object.entries(senses)) {
      assert.ok(brain.sets[name]?.length > 0, `payload omits sensory group ${name}`);
      assert.ok(Number.isFinite(level) && level >= 0 && level <= 1);
      brain.stimulate(name, level);
    }
    const start = performance.now(), solve = brain.settleControl(PROTOCOL.max_steps, PROTOCOL.equation_tolerance);
    const solveMs = performance.now() - start;
    const neuralReadouts = Object.fromEntries(SETTLED_MOTOR_GROUPS.map(name => [name, brain.mean(name)]));
    const deliveredReadouts = arm === "zero_motor" ? Object.fromEntries(SETTLED_MOTOR_GROUPS.map(name => [name, 0])) : neuralReadouts;
    const envelope = { type: "control", kind: "control", ...request, ...solve };
    if (solve.converged) envelope.readouts = deliveredReadouts;
    const accepted = life.applyControl(envelope);
    assert.equal(accepted, solve.converged, "controller rejected an otherwise valid solve envelope");
    const row = { request_id: request.requestId, t_s: tick * PROTOCOL.decision_period_s, senses, solve, solve_ms: solveMs, accepted,
      neural_motor_readouts: neuralReadouts, motor_intervention: arm === "zero_motor", controls: { ...life.controls }, proboscis: life.proboscis,
      command_status: { ...life.commandStatus } };
    for (let step = 0; step < bodySteps; step++) {
      life.step(DT);
      assert.ok(finiteState(flight), `${arm}: nonfinite body state at tick ${tick}, step ${step}`);
      if (passive) {
        passive.step(DT, zeroControls());
        assert.deepEqual(state(flight), state(passive), "zero-motor body differs from passive physics");
      }
      if (Object.values(life.controls).some(value => value !== 0) || life.proboscis !== 0) activeSteps++;
      maxFrequency = Math.max(maxFrequency, life.controls.f); maxAmplitude = Math.max(maxAmplitude, life.controls.aL, life.controls.aR);
      maxProboscis = Math.max(maxProboscis, life.proboscis);
      maxTilt = Math.max(maxTilt, tilt(flight)); maxSpeed = Math.max(maxSpeed, flight.speed()); minimumHeight = Math.min(minimumHeight, flight.p[2]);
      if (flight.touching > 0 && firstContact === null) firstContact = life.clock;
      if (tilt(flight) > PROTOCOL.descriptive_thresholds.severe_tilt_radians && severeTiltAt === null) severeTiltAt = life.clock;
    }
    row.after = state(flight); rows.push(row);
  }
  const end = state(flight);
  return { arm, duration_simulated_s: life.clock, body_steps: flight.steps, wall_time_ms: performance.now() - started,
    solve_latency_ms: summarizeLatency(rows.map(row => row.solve_ms)), solver_failures: rows.filter(row => !row.solve.converged).length,
    accepted_commands: rows.filter(row => row.accepted).length, active_command_body_steps: activeSteps,
    maximum_frequency_hz: maxFrequency, maximum_amplitude: maxAmplitude, maximum_proboscis: maxProboscis,
    maximum_tilt_radians: maxTilt, maximum_speed_m_s: maxSpeed, minimum_height_m: minimumHeight,
    height_lost_m: initial.position[2] - end.position[2], exceeded_height_loss_threshold: initial.position[2] - minimumHeight > PROTOCOL.descriptive_thresholds.lost_height_m,
    first_contact_s: firstContact, severe_tilt_at_s: severeTiltAt, finite_body: true,
    passive_trajectory_exact_match: passive ? passiveMatches : null,
    initial: initialState, final: end, unsupported: { ...life.unsupported }, rows };
}

const args = process.argv.slice(2);
assert.ok(args.length === 0 || (args.length === 2 && ["--out", "--verify"].includes(args[0])), "usage: settled_control_probe.mjs [--out FILE | --verify FILE]");
if (args[0] === "--verify") {
  const receipt = JSON.parse(readFileSync(resolve(args[1]), "utf8"));
  checkReceipt(receipt);
  console.log("settled-control receipt integrity, source bindings, protocol coverage and actuator invariants verified; this is not a trajectory rerun");
} else {
  assert.equal(DT, PROTOCOL.physics_dt_s);
  assert.equal(count * PROTOCOL.decision_period_s, PROTOCOL.duration_s);
  assert.equal(bodySteps * DT, PROTOCOL.decision_period_s);
  // Any accidental call into the legacy flight controller fails this actual execution.
  HandPilot.prototype.controls = () => { throw new Error("legacy pilot invoked during strict probe"); };
  const sources = sourceHashes();
  const payload = JSON.parse(readFileSync(resolve(ROOT, "web/data/brain.json"), "utf8"));
  const results = PROTOCOL.arms.map(arm => runArm(payload, arm));
  assert.deepEqual(sourceHashes(), sources, "source changed while the probe ran; refuse to bind mixed versions");
  const body = { schema: "cadence.fly-matrix.settled-control-probe/v1", created_utc: new Date().toISOString(),
    protocol: PROTOCOL, sources, runtime: { node: process.version, platform: process.platform, architecture: process.arch },
    payload: { neurons: payload.n, edges: payload.edges, synapses: payload.synapses, whole: payload.whole, model: payload.model, recruitment: payload.recruitment },
    results, flight_competency_established: false,
    limitations: ["One initial condition and half a simulated second, no training or held-out behavior experiment.",
      "Synchronous offline solves pause simulated physics; elapsed solve time is measured but no real-time scheduling deadline is passed.",
      "The actual 60000-neuron recruited payload is used, not all 150802 neurons of the whole-brain atlas.",
      "Residual qualification is approximate equation agreement, not a global contraction, uniqueness or biological fidelity certificate.",
      "Walking, leg forces and grooming are unsupported; wingbeat is a supplied stroke-averaged actuator carrier.",
      "Worker message transport, renderer behavior, learning and external surface geometry are outside this probe."] };
  const receipt = { ...body, sha256: digest(JSON.stringify(body)) };
  checkReceipt(receipt);
  if (args[0] === "--out") { const out = resolve(args[1]); mkdirSync(dirname(out), { recursive: true }); writeFileSync(out, JSON.stringify(receipt, null, 2) + "\n"); }
  console.log(JSON.stringify({ protocol: { duration_s: PROTOCOL.duration_s, decisions_per_arm: count },
    results: results.map(({ rows, ...summary }) => summary), flight_competency_established: false }, null, 2));
}
