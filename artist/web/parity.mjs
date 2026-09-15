#!/usr/bin/env node
// Parity of artist/web/artist.js against the Python artist on a recorded fixture:
//
//     node artist/web/parity.mjs runs/artist/parity [--no-sidecar] [--limit N] [--quiet]
//
// Loads the snapshot the fixture was exported from, rebuilds the life (the engine of
// ../../web/engine.js with its records head, the canvas world, the intention search and the
// torque search of ./artist.js), replays the recorded drawings from their targets and body
// poses, and holds every step to the Python numbers: the moment the JavaScript world produces
// (every sensor, the ink window, the reward and the episode flags), the intention the slow
// level commits, the action and the controller exactly, and the predictions, the policy, the
// objective and the record tables to the tolerances the fixture records. After the last drawing
// the records tables, the running mean, the write counts, the ledger, the generators and the
// measures of every drawing are compared with final.json. Exits non-zero on any difference.

import { existsSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { ExperienceAgent, unpackF64 } from "../../web/engine.js";
import { ArtistBrain, CanvasWorld, canvasBody, canvasConfig } from "./artist.js";

const FORMAT = "cadence-artist-parity/1";

function args(argv) {
  const out = { fixture: null, limit: Infinity, quiet: false, sidecar: true };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--limit") out.limit = Number(argv[++i]);
    else if (a === "--quiet") out.quiet = true;
    else if (a === "--no-sidecar") out.sidecar = false;
    else if (out.fixture === null) out.fixture = a;
    else throw Error(`unexpected argument ${a}`);
  }
  if (!out.fixture) throw Error("usage: node parity.mjs <fixture dir> [--no-sidecar] [--limit N] [--quiet]");
  return out;
}

const fmt = (x) => (x === 0 ? "0" : Number(x).toExponential(3));

class Differences {
  constructor(tolerances) { this.tol = tolerances; this.rows = []; this.worst = {}; this.count = 0; }
  fail(where, what, mine, theirs) {
    this.count += 1;
    if (this.rows.length < 12) this.rows.push(`${where}: ${what} is ${JSON.stringify(mine)} here and ${JSON.stringify(theirs)} in Python`);
    else if (this.rows.length === 12) this.rows.push("…");
  }
  exact(where, what, mine, theirs) { if (mine !== theirs) this.fail(where, what, mine, theirs); }
  exactList(where, what, mine, theirs) {
    if (mine.length !== theirs.length) return this.fail(where, `${what} length`, mine.length, theirs.length);
    for (let k = 0; k < mine.length; k++) if (Number(mine[k]) !== Number(theirs[k])) return this.fail(where, `${what}[${k}]`, Number(mine[k]), Number(theirs[k]));
  }
  close(where, what, mine, theirs, kind = "read") {
    const tol = this.tol[kind] ?? 1e-9, scale = Math.max(1, Math.abs(theirs));
    const d = Math.abs(mine - theirs);
    this.worst[kind] = Math.max(this.worst[kind] || 0, d / scale);
    if (!(d <= tol * scale)) this.fail(where, what, mine, theirs);
  }
  closeList(where, what, mine, theirs, kind = "read") {
    if (mine.length !== theirs.length) return this.fail(where, `${what} length`, mine.length, theirs.length);
    for (let k = 0; k < mine.length; k++) this.close(where, `${what}[${k}]`, Number(mine[k]), Number(theirs[k]), kind);
  }
}

function main() {
  const opt = args(process.argv.slice(2));
  const dir = resolve(opt.fixture);
  const snapshot = JSON.parse(readFileSync(join(dir, "snapshot.json"), "utf8"));
  const final = JSON.parse(readFileSync(join(dir, "final.json"), "utf8"));
  const book = JSON.parse(readFileSync(join(dir, "drawings.json"), "utf8"));
  if (final.format !== FORMAT || book.format !== FORMAT) throw Error(`${dir} is not a ${FORMAT} fixture`);
  const lines = readFileSync(join(dir, "stream.jsonl"), "utf8").trim().split("\n").map((l) => JSON.parse(l));
  const diff = new Differences(final.tolerances || {});
  const sidecarPath = join(dir, "records_f64.json");
  const sidecar = opt.sidecar && existsSync(sidecarPath) ? JSON.parse(readFileSync(sidecarPath, "utf8")) : null;

  const extra = snapshot.extra || {};
  const agent = new ExperienceAgent(snapshot, sidecar ? { records: sidecar } : {});
  const brain = new ArtistBrain(agent, { canvas: extra.canvas, intention: extra.intention, planner: extra.planner, intention_interval: extra.intention_interval, intention_gain: extra.intention_gain });
  const config = canvasConfig({ ...extra.canvas, max_decisions: book.per_drawing }); // the fixture caps every drawing
  const body = canvasBody(extra.body, { gain: (extra.body && extra.body.gain) ?? 1.0 });
  const world = new CanvasWorld(config, body, { seed: 1, lifeId: "parity" });
  console.log(`snapshot: ${(snapshot.ledger.real_transitions || 0).toLocaleString()} decisions lived · ${agent.n} neurons, ${agent.E.toLocaleString()} synapses, ${agent.records.granules.toLocaleString()} granules (${agent.recordsSource})`);
  console.log(`fixture: ${book.drawings.length} drawings, ${lines.length} steps, ${lines.filter((l) => l.decision).length} decisions`);

  const byDrawing = new Map();
  for (const line of lines) { if (!byDrawing.has(line.drawing)) byDrawing.set(line.drawing, []); byDrawing.get(line.drawing).push(line); }
  let decisions = 0, steps = 0, seconds = 0, slowest = 0;

  for (const [index, record] of book.drawings.entries()) {
    if (index >= opt.limit) break;
    const expected = byDrawing.get(index) || [];
    const target = Uint8Array.from(record.target);
    brain.detach();
    let moment = world.reset({ target, theta: record.theta, family: record.family });
    brain.begin(world.view());
    for (let at = 0; at < expected.length; at++) {
      const line = expected[at], where = `drawing ${index} step ${at}`;
      compareMoment(diff, where, moment, line.moment);
      compareIntention(diff, `${where} before`, brain, line.intention_before);
      const t0 = process.hrtime.bigint();
      const decision = brain.step(moment, world.view());
      const took = Number(process.hrtime.bigint() - t0) / 1e6;
      seconds += took / 1000;
      if (decision !== null) { slowest = Math.max(slowest, took); decisions += 1; }
      steps += 1;
      compareIntention(diff, where, brain, line.intention);
      diff.close(where, "sheet value", brain.sheet.value, line.sheet_value, "sheet");
      diff.closeList(where, "pen pixel", brain.penPixel, line.pen_pixel, "sheet");
      diff.exact(where, "a decision", decision !== null, line.decision !== null);
      if (decision !== null && line.decision !== null) {
        const e = line.decision;
        diff.exact(where, "action", decision.action, e.action);
        diff.exact(where, "controller", decision.controller, e.controller);
        diff.exact(where, "decision_id", decision.decision_id, e.decision_id);
        diff.exact(where, "event_id", decision.event_id, e.event_id);
        diff.exact(where, "parameter_version", decision.parameter_version, e.parameter_version);
        for (const [name, value] of Object.entries(e.budget)) diff.exact(where, `budget.${name}`, decision.budget[name] | 0, value);
        for (const [name, value] of Object.entries(e.prediction)) diff.closeList(where, `prediction.${name}`, decision.prediction[name], value);
        diff.closeList(where, "probabilities", decision.probabilities, e.probabilities);
        diff.close(where, "log probability", decision.log_probability, e.log_probability);
      }
      diff.exact(where, "record writes", agent.ledger.record_writes, line.ledger.record_writes);
      diff.exact(where, "imagined", agent.ledger.imagined, line.ledger.imagined);
      diff.exact(where, "real transitions", agent.ledger.real_transitions, line.ledger.real_transitions);
      diff.exact(where, "parameter version", agent.parameterVersion, line.parameter_version);
      diff.close(where, "discrepancy", world.discrepancy, line.discrepancy, "moment");
      if (decision === null) { if (at !== expected.length - 1) diff.fail(where, "the drawing ended early", at, expected.length - 1); break; }
      moment = world.act(decision.decision_id, decision.action);
    }
    const measure = world.measure(), expect = final.measures[index];
    if (expect) {
      diff.close(`drawing ${index}`, "chamfer", measure.chamfer, expect.chamfer, "moment");
      diff.close(`drawing ${index}`, "f1", measure.f1, expect.f1, "moment");
      diff.exact(`drawing ${index}`, "ink", measure.ink, expect.ink);
      diff.exact(`drawing ${index}`, "strokes", measure.strokes, expect.strokes);
    }
    if (!opt.quiet) console.log(`  drawing ${index} (${record.family}): ${expected.length - 1} decisions, chamfer ${measure.chamfer.toFixed(4)}, f1 ${measure.f1.toFixed(3)}, ${measure.ink} pixels inked`);
  }

  // the records head, the ledger and the generators after the last drawing
  const block = final.records;
  diff.closeList("final", "records mean", agent.records.mean, unpackF64(block.mean), "table");
  diff.exact("final", "records writes", agent.records.writes, block.writes);
  diff.exact("final", "records seen", agent.records.seen, block.seen);
  for (const name of Object.keys(block.tables)) {
    const mine = agent.records.tables[name], theirs = unpackF64(block.tables[name]);
    if (mine.length !== theirs.length) { diff.fail("final", `${name} length`, mine.length, theirs.length); continue; }
    let worst = 0, atWorst = -1;
    for (let k = 0; k < mine.length; k++) { const d = Math.abs(mine[k] - theirs[k]); if (d > worst) { worst = d; atWorst = k; } }
    diff.worst.table = Math.max(diff.worst.table || 0, worst);
    if (!(worst <= (final.tolerances.table ?? 1e-9))) diff.fail("final", `records ${name} (largest difference at cell ${atWorst})`, worst, 0);
    console.log(`  final records/${name}: ${mine.length.toLocaleString()} records, largest difference ${fmt(worst)}`);
  }
  diff.closeList("final", "efficacy", agent.efficacy, final.efficacy, "table");
  diff.closeList("final", "bias", agent.bias, final.bias, "table");
  diff.closeList("final", "critic w", agent.wCritic, final.critic_w, "table");
  diff.close("final", "critic b", agent.bCritic, final.critic_b, "table");
  diff.closeList("final", "context trace", agent.context.trace, final.context_trace, "table");
  diff.exact("final", "generator", agent.rng[0].state, final.rng.action);
  diff.exact("final", "parameter version", agent.parameterVersion, final.parameter_version);
  for (const name of ["record_writes", "imagined", "real_transitions"]) diff.exact("final", `ledger.${name}`, agent.ledger[name], final.ledger[name]);

  console.log(`steps ${steps}, decisions ${decisions}; ${((seconds * 1000) / Math.max(1, steps)).toFixed(1)} ms per step, slowest decision ${slowest.toFixed(1)} ms`);
  console.log(`largest relative differences: moments ${fmt(diff.worst.moment || 0)}, reads ${fmt(diff.worst.read || 0)}, intentions ${fmt(diff.worst.intention || 0)}, objective ${fmt(diff.worst.sheet || 0)}, tables ${fmt(diff.worst.table || 0)}`);
  if (diff.rows.length) {
    console.log(`PARITY FAILED (${diff.count} differences)`);
    for (const row of diff.rows) console.log(`  ${row}`);
    return 1;
  }
  console.log("PARITY HELD: every moment, intention, decision, prediction and record table matches the Python artist");
  return 0;
}

function compareMoment(diff, where, mine, theirs) {
  diff.exact(where, "life_id", mine.life_id, theirs.life_id);
  diff.exact(where, "episode_id", mine.episode_id, theirs.episode_id);
  diff.exact(where, "event_id", mine.event_id, theirs.event_id);
  diff.exact(where, "tick", mine.tick, theirs.tick);
  diff.exact(where, "feedback_for", mine.feedback_for, theirs.feedback_for);
  diff.exact(where, "executed", mine.executed, theirs.executed);
  diff.exact(where, "terminated", mine.terminated, theirs.terminated);
  diff.exact(where, "truncated", mine.truncated, theirs.truncated);
  diff.exact(where, "reward_known", mine.reward_known, theirs.reward_known);
  diff.close(where, "reward", mine.reward, theirs.reward, "moment");
  for (const [name, value] of Object.entries(theirs.observation)) diff.closeList(where, `observation.${name}`, mine.observation[name], value, "moment");
  diff.exact(where, "a final observation", mine.final_observation !== null, theirs.final_observation !== null);
  if (theirs.final_observation && mine.final_observation) for (const [name, value] of Object.entries(theirs.final_observation)) diff.closeList(where, `final_observation.${name}`, mine.final_observation[name], value, "moment");
}

function compareIntention(diff, where, brain, theirs) {
  const i = brain.intention;
  diff.exact(where, "intention pen", i.pen, theirs.pen);
  diff.exact(where, "intention valid", i.valid, theirs.valid);
  diff.exact(where, "intention countdown", brain.countdown, theirs.countdown);
  diff.closeList(where, "intention endpoint", i.endpoint, theirs.endpoint, "intention");
  diff.closeList(where, "intention direction", i.direction, theirs.direction, "intention");
  diff.close(where, "intention distance", i.distance, theirs.distance, "intention");
  diff.close(where, "intention score", i.score, theirs.score, "intention");
  diff.closeList(where, "intention goal", brain.dress({ goal: null }).goal, theirs.goal, "intention");
}

process.exit(main());
