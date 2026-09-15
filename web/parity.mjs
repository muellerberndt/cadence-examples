#!/usr/bin/env node
// Parity of the JavaScript engine against the Python agent on the S01 arm fixture:
//
//     node web/parity.mjs runs/arm/parity [--limit N] [--quiet] [--no-sidecar]
//
// Loads snapshot.json, installs the S01 planner, replays stream.jsonl moment by moment and
// reports: decisions matching (action, controller), the largest prediction difference,
// parameter-version agreement per line, efficacy sha256 agreement per line (byte-exact only
// when the arithmetic order matches; the max abs difference is what the tolerance judges),
// and the final differences of efficacy, bias, world velocity, critic, context trace and
// the generator states against final.json. Exits non-zero if any decision differs or a
// difference exceeds the tolerances in final.json.
//
// The snapshot carries the pending decision and the ring's warm potentials (float32, as
// `ring.precision` says) and continues the stream by itself.
//
// A snapshot with a records world head carries its tables in float32; the fixture writes
// them in float64 to <fixture>/records_f64.json (the sidecar of web.export), which the
// harness loads when its efficacy digest matches (`--no-sidecar` runs from the float32
// tables). After the stream the records mean, every records table and the write count are
// compared with final.json.

import { readFileSync, existsSync } from "node:fs";
import { createHash } from "node:crypto";
import { resolve, join, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { ExperienceAgent, unpackF64, snapshotCarriesLife, RECORDS_FORMAT } from "./engine.js";
import { ModelPlanner } from "../arm/web/planner.js";

const HERE = dirname(fileURLToPath(import.meta.url));

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
  if (!out.fixture) throw Error("usage: node parity.mjs <fixture dir> [--limit N] [--quiet] [--no-sidecar]");
  return out;
}

const sha256 = (typed) => createHash("sha256").update(Buffer.from(typed.buffer, typed.byteOffset, typed.byteLength)).digest("hex");
const maxAbs = (a, b) => { if (a.length !== b.length) return Infinity; let m = 0; for (let i = 0; i < a.length; i++) { const d = Math.abs(a[i] - b[i]); if (!(d <= m)) m = Number.isNaN(d) ? Infinity : d; } return m; };
const fmt = (x) => (x === 0 ? "0" : x.toExponential(3));

/** The float64 records sidecar next to the fixture, when the snapshot has a records head and the digests match; null otherwise. */
export function loadRecordsSidecar(dir, snapshot, wanted) {
  if (!snapshot.records || !wanted) return { sidecar: null, note: snapshot.records ? "snapshot tables (float32)" : "no records head" };
  const path = join(dir, snapshot.records.sidecar || "records_f64.json");
  if (!existsSync(path)) return { sidecar: null, note: `snapshot tables (${snapshot.records.precision}); no sidecar at ${path}` };
  const sidecar = JSON.parse(readFileSync(path, "utf8"));
  if (sidecar.format !== RECORDS_FORMAT) throw Error(`${path} is not a ${RECORDS_FORMAT} file`);
  const digest = sha256(unpackF64(snapshot.efficacy));
  if (sidecar.snapshot_efficacy_sha256 !== digest) throw Error(`${path} belongs to another snapshot (efficacy sha256 ${sidecar.snapshot_efficacy_sha256} != ${digest})`);
  return { sidecar, note: `sidecar ${path} (float64 tables)` };
}

/** The final records comparison: the largest difference of the mean and of every table, and the write counts. */
export function recordsFinals(state, final) {
  if (!state.records || !final.records) return { rows: [], meanDiff: null, tableDiff: null, tables: {}, writes: [state.records ? state.records.writes : null, final.records ? final.records.writes : null] };
  const tables = {};
  let tableDiff = 0;
  for (const name of Object.keys(final.records.tables)) { const d = maxAbs(state.records.tables[name] || new Float64Array(0), unpackF64(final.records.tables[name])); tables[name] = d; if (!(d <= tableDiff)) tableDiff = d; }
  const meanDiff = maxAbs(state.records.mean, unpackF64(final.records.mean));
  const blockDiff = final.records.block_norm ? maxAbs(state.records.block_norm, unpackF64(final.records.block_norm)) : null;
  return { meanDiff, tableDiff, blockDiff, tables, writes: [state.records.writes, final.records.writes], seen: [state.records.seen, final.records.seen] };
}

function main() {
  const opt = args(process.argv.slice(2));
  const dir = resolve(opt.fixture);
  const snapshot = JSON.parse(readFileSync(join(dir, "snapshot.json"), "utf8"));
  const lines = readFileSync(join(dir, "stream.jsonl"), "utf8").split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l));
  const final = JSON.parse(readFileSync(join(dir, "final.json"), "utf8"));
  const tol = final.tolerances || { activation: 1e-9, efficacy: 1e-8, prediction: 1e-8 };
  const complete = snapshotCarriesLife(snapshot);
  if (!complete) {
    console.error("the snapshot carries no pending record or no ring potentials: regenerate the fixture with the stage's tools/parity_fixture.py");
    return 2;
  }
  const planner = new ModelPlanner(snapshot.extra.planner);
  const records = loadRecordsSidecar(dir, snapshot, opt.sidecar);
  const agent = new ExperienceAgent(snapshot, { planner: planner.forAgent(), records: records.sidecar });
  console.log(`engine: ${agent.n} neurons, ${agent.E} synapses, ${agent.layout.ranges} ranges, ${agent.layout.pairs} blocks`);
  if (agent.records) console.log(`records head: ${agent.records.granules} granules, ${agent.records.config.active} active, ${agent.reading.length} reading neurons, ${agent.records.fields.length} fields, ${agent.records.parameters().toLocaleString()} record parameters; tables from ${records.note}; ${agent.records.writes} writes so far`);
  console.log(`snapshot: ${complete ? "carries the pending decision and the ring potentials" : "older export without them"}${snapshot.ring && snapshot.ring.precision ? ` (ring precision: ${snapshot.ring.precision})` : ""}`);
  console.log(`life records: ring from ${agent.ringSource}; pending from ${agent.pendingSource}`);
  console.log(`stream: ${lines.length} moments; tolerances ${JSON.stringify(tol)}`);
  const started = Date.now();
  let decisions = 0, matched = 0, versionsAgree = 0, hashesAgree = 0, predictionMax = 0, probabilityMax = 0, logpMax = 0, learningMax = 0, hashLines = 0, writesAgree = 0, writeLines = 0;
  const mismatches = [];
  const count = Math.min(lines.length, opt.limit);
  for (let k = 0; k < count; k++) {
    const line = lines[k];
    let decision;
    try { decision = agent.step(line.moment); } catch (err) { console.error(`line ${k} (event ${line.moment.event_id}): engine error: ${err.message}`); return 1; }
    const expected = line.decision;
    if ((decision === null) !== (expected === null)) mismatches.push(`line ${k}: decision ${decision === null ? "null" : "present"} vs expected ${expected === null ? "null" : "present"}`);
    else if (decision !== null) {
      decisions += 1;
      const same = decision.action === expected.action && decision.controller === expected.controller;
      if (same) matched += 1; else mismatches.push(`line ${k} (event ${line.moment.event_id}): action ${decision.action}/${decision.controller} vs expected ${expected.action}/${expected.controller}`);
      for (const name of Object.keys(expected.prediction)) predictionMax = Math.max(predictionMax, maxAbs(decision.prediction[name] || [], expected.prediction[name]));
      probabilityMax = Math.max(probabilityMax, maxAbs(decision.probabilities, expected.probabilities));
      logpMax = Math.max(logpMax, Math.abs(decision.log_probability - expected.log_probability));
      if (decision.decision_id !== expected.decision_id || decision.context_version !== expected.context_version || decision.parameter_version !== expected.parameter_version) mismatches.push(`line ${k}: ids differ (decision ${decision.decision_id}/${expected.decision_id}, versions ${decision.parameter_version}/${expected.parameter_version}, context ${decision.context_version}/${expected.context_version})`);
      for (const key of Object.keys(expected.budget || {})) if (decision.budget[key] !== expected.budget[key]) mismatches.push(`line ${k}: budget ${key} ${decision.budget[key]} vs ${expected.budget[key]}`);
    }
    if (agent.parameterVersion === line.parameter_version) versionsAgree += 1; else mismatches.push(`line ${k}: parameter version ${agent.parameterVersion} vs expected ${line.parameter_version}`);
    if (line.efficacy_sha256) { hashLines += 1; if (sha256(agent.efficacy) === line.efficacy_sha256) hashesAgree += 1; }
    for (const [key, value] of Object.entries(line.learning || {})) { const mine = agent.lastReport.learning[key]; if (mine === undefined) mismatches.push(`line ${k}: learning report lacks ${key}`); else learningMax = Math.max(learningMax, Math.abs(mine - value)); }
    if (line.record_writes !== undefined) { writeLines += 1; if (agent.ledger.record_writes === line.record_writes && (line.imagined === undefined || agent.ledger.imagined === line.imagined)) writesAgree += 1; else mismatches.push(`line ${k}: record writes ${agent.ledger.record_writes} vs ${line.record_writes}, imagined ${agent.ledger.imagined} vs ${line.imagined}`); }
    if (!opt.quiet && (k % 10 === 9 || k === count - 1)) console.log(`  ${k + 1}/${count} moments · ${matched}/${decisions} decisions match · version ${agent.parameterVersion} · prediction max ${fmt(predictionMax)} · ${((Date.now() - started) / 1000).toFixed(1)} s`);
  }
  const state = agent.snapshotState();
  const rec = recordsFinals(state, final);
  const finals = {
    efficacy: [maxAbs(state.efficacy, final.efficacy), tol.efficacy],
    bias: [maxAbs(state.bias, final.bias), tol.efficacy],
    world_velocity: [maxAbs(state.world_velocity, final.world_velocity), tol.efficacy],
    critic_w: [maxAbs(state.critic_w, final.critic_w), tol.efficacy],
    critic_b: [Math.abs(state.critic_b - final.critic_b), tol.efficacy],
    context_trace: [maxAbs(state.context_trace, final.context_trace), tol.activation],
  };
  const full = count === lines.length;
  console.log("");
  console.log(`decisions matching (action, controller): ${matched}/${decisions}`);
  console.log(`parameter version agreement per line: ${versionsAgree}/${count}; final ${state.parameter_version} vs ${final.parameter_version}`);
  console.log(`efficacy sha256 agreement per line: ${hashesAgree}/${hashLines}`);
  console.log(`max abs prediction difference: ${fmt(predictionMax)} (tolerance ${tol.prediction})`);
  console.log(`max abs probability difference: ${fmt(probabilityMax)}; log-probability: ${fmt(logpMax)}; learning-report numbers: ${fmt(learningMax)}`);
  for (const [name, [diff, limit]] of Object.entries(finals)) console.log(`final max abs difference ${name}: ${fmt(diff)} (tolerance ${limit})${full ? "" : " [partial stream]"}`);
  if (rec.tableDiff !== null) {
    console.log(`record writes agreement per line (writes, imagined): ${writesAgree}/${writeLines}; final writes ${rec.writes[0]} vs ${rec.writes[1]} (ledger ${state.ledger.record_writes} vs ${final.ledger.record_writes}; imagined ${state.ledger.imagined} vs ${final.ledger.imagined})`);
    console.log(`final max abs difference records mean: ${fmt(rec.meanDiff)}; pathway norms: ${rec.blockDiff === null ? "n/a" : fmt(rec.blockDiff)} (${state.records.block_norm.length} pathways); records tables: ${fmt(rec.tableDiff)} (tolerance ${tol.records ?? 1e-10})${full ? "" : " [partial stream]"}`);
    console.log(`  per table: ${Object.entries(rec.tables).map(([name, d]) => `${name} ${fmt(d)}`).join(" · ")}`);
  }
  console.log(`rng states: action ${state.rng.action} vs ${final.rng.action}; replay ${state.rng.replay} vs ${final.rng.replay}`);
  console.log(`ring: stored ${state.ring_stored} vs ${final.ring_stored}; at ${state.ring_at} vs ${final.ring_at}`);
  console.log(`ledger phases: ${JSON.stringify(state.ledger.phases)}`);
  console.log(`expected phases: ${JSON.stringify(final.ledger.phases)}`);
  console.log(`elapsed ${((Date.now() - started) / 1000).toFixed(1)} s`);
  let failed = mismatches.length > 0 || matched !== decisions;
  if (predictionMax > tol.prediction) { failed = true; mismatches.push(`prediction difference ${predictionMax} exceeds ${tol.prediction}`); }
  if (full) {
    for (const [name, [diff, limit]] of Object.entries(finals)) if (!(diff <= limit)) { failed = true; mismatches.push(`final ${name} difference ${diff} exceeds ${limit}`); }
    if (state.rng.action !== final.rng.action || state.rng.replay !== final.rng.replay) { failed = true; mismatches.push("rng states differ"); }
    if (state.parameter_version !== final.parameter_version) { failed = true; mismatches.push("final parameter version differs"); }
    if (state.ring_stored !== final.ring_stored || state.ring_at !== final.ring_at) { failed = true; mismatches.push("ring cursors differ"); }
    if (rec.tableDiff !== null) {
      const limit = tol.records ?? 1e-10;
      if (!(rec.tableDiff <= limit)) { failed = true; mismatches.push(`records tables difference ${rec.tableDiff} exceeds ${limit}`); }
      if (!(rec.meanDiff <= limit)) { failed = true; mismatches.push(`records mean difference ${rec.meanDiff} exceeds ${limit}`); }
      if (rec.blockDiff !== null && !(rec.blockDiff <= limit)) { failed = true; mismatches.push(`records pathway-norm difference ${rec.blockDiff} exceeds ${limit}`); }
      if (rec.writes[0] !== rec.writes[1] || rec.seen[0] !== rec.seen[1] || state.ledger.record_writes !== final.ledger.record_writes || state.ledger.imagined !== final.ledger.imagined) { failed = true; mismatches.push("record write or imagined-read counts differ"); }
    }
  }
  if (mismatches.length) { console.log(""); console.log(`${mismatches.length} mismatch(es):`); for (const m of mismatches.slice(0, 40)) console.log("  " + m); }
  const onlyTolerances = failed && matched === decisions && mismatches.every((m) => /exceeds/.test(m));
  if (onlyTolerances && agent.ringSource.includes("float32")) console.log("note: every decision matches and only the numerical tolerances are exceeded; the ring came from the snapshot in float32, so its replayed rows differ from Python's float64 ring at rounding level.");
  console.log(failed ? "PARITY FAILED" : "PARITY OK");
  return failed ? 1 : 0;
}

// run as the entry point only: parity_s02.mjs imports the records helpers from here
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) process.exit(main());
