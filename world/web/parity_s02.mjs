#!/usr/bin/env node
// Parity of the S02 port against the Python fixture of tools/parity_fixture.py:
//
//     node world/web/parity_s02.mjs runs/world/parity [--limit N] [--quiet]
//
// Brain section: loads snapshot.json, builds the engine agent and the Brain around it (stores,
// planner, bookkeeping and recall read from `extra`), replays stream.jsonl moment by moment
// with the exploration rate each line records, and reports: decisions matching (action,
// controller, ids, the search's expansions), the largest prediction difference, the parameter
// version per line, the plan and the planner counters (searches, plan reuses, expansions,
// fallbacks, pruned moves, the fallback generator's state) per line, the largest store
// difference and the largest recall-port difference per line, and the final differences of
// efficacy, bias, world velocity, critic, context trace, stores, the records head (mean,
// every table, the write and imagined-read counts; the float64 tables come from the
// records_f64.json sidecar next to the fixture) and generator states against final.json.
// World section: loads the world from `extra.world`, applies every operation and
// event of world.jsonl through the same World API the page's curriculum uses, and compares
// the moment each event produced and the private state after it with the Python record;
// the deadline operations check `requestDeadline` and every request's start cell is checked
// against `farCandidates`. Exits non-zero if anything differs beyond the tolerances.

import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { resolve, join } from "node:path";
import { ExperienceAgent, normalizeMoment, snapshotCarriesLife } from "../../web/engine.js";
import { loadRecordsSidecar, recordsFinals } from "../../web/parity.mjs";
import { Curriculum, World, requestDeadline, sameCell } from "./world.js";
import { Brain } from "./planner.js";

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
  if (!out.fixture) throw Error("usage: node parity_s02.mjs <fixture dir> [--limit N] [--quiet] [--no-sidecar]");
  return out;
}

const sha256 = (typed) => createHash("sha256").update(Buffer.from(typed.buffer, typed.byteOffset, typed.byteLength)).digest("hex");
const maxAbs = (a, b) => { if (a.length !== b.length) return Infinity; let m = 0; for (let i = 0; i < a.length; i++) { const d = Math.abs(a[i] - b[i]); if (!(d <= m)) m = Number.isNaN(d) ? Infinity : d; } return m; };
const fmt = (x) => (x === 0 ? "0" : x.toExponential(3));
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

/** The largest difference between two store records (the shapes of Brain.stores()). */
function storeDiff(mine, theirs) {
  let m = 0;
  const walk = (a, b) => {
    if (Array.isArray(a)) { if (!Array.isArray(b) || a.length !== b.length) { m = Infinity; return; } for (let k = 0; k < a.length; k++) walk(a[k], b[k]); }
    else if (typeof a === "object" && a !== null) { for (const k of Object.keys(a)) { if (!(k in b)) { m = Infinity; return; } walk(a[k], b[k]); } }
    else { const d = Math.abs(Number(a) - Number(b)); if (!(d <= m)) m = Number.isNaN(d) ? Infinity : d; }
  };
  walk(mine, theirs);
  return m;
}

/** A moment produced by the JS world against the recorded one: every field the agent sees. */
function momentDiff(mine, theirs) {
  const a = normalizeMoment(mine), b = normalizeMoment(theirs), out = [];
  for (const key of ["life_id", "episode_id", "event_id", "tick", "dt", "feedback_for", "executed", "reward", "reward_known", "terminated", "truncated"]) if (a[key] !== b[key]) out.push(`${key} ${a[key]} vs ${b[key]}`);
  if (!same(Array.from(a.action_mask), Array.from(b.action_mask))) out.push("action_mask");
  if (!same(Array.from(a.goal || []), Array.from(b.goal || []))) out.push("goal");
  const names = new Set([...Object.keys(a.observation), ...Object.keys(b.observation)]);
  for (const name of names) {
    if (!(name in a.observation) || !(name in b.observation)) { out.push(`field ${name} missing`); continue; }
    if (!same(Array.from(a.observation[name]), Array.from(b.observation[name]))) out.push(`observation ${name}`);
    if (!same(Array.from(a.observed[name]), Array.from(b.observed[name]))) out.push(`observed ${name}`);
  }
  if ((a.final_observation === null) !== (b.final_observation === null)) out.push("final_observation presence");
  else if (a.final_observation !== null) for (const name of Object.keys(b.final_observation)) if (!same(Array.from(a.final_observation[name]), Array.from(b.final_observation[name]))) out.push(`final_observation ${name}`);
  return out;
}

function stateDiff(mine, theirs) {
  const out = [];
  for (const key of Object.keys(theirs)) { if (key === "seed" || key === "life_id") continue; if (!same(mine[key], theirs[key])) out.push(key); }
  return out;
}

function applyOp(world, op, checks) {
  switch (op.op) {
    case "place_object": world.placeObject(op.object, op.cell); break;
    case "set_word": world.setWord(op.word); break;
    case "set_word_rule": world.setWordRule(op.rule); break;
    case "set_goal": world.setGoal(op.goal); break;
    case "set_door_rule": world.setDoorRule(op.rule); break;
    case "set_mask": world.setMask(op.legal); break;
    case "reward_rule": world.rewardRule = op.rule === null ? null : { ...op.rule }; break;
    case "set_deadline": {
      if (world.rewardRule && world.rewardRule.kind === "reach") { const mine = requestDeadline(world, world.rewardRule.target); checks.deadlines += 1; if (mine !== op.steps) checks.mismatches.push(`request deadline ${mine} vs ${op.steps} at tick ${world.tick}`); }
      world.setDeadline(op.steps);
      break;
    }
    default: throw Error(`unknown world operation ${op.op}`);
  }
}

function worldSection(snapshot, events, opt) {
  const started = Date.now();
  const world = World.fromState(snapshot.extra.world);
  const cur = new Curriculum(world, 0, { cueRule: snapshot.extra.curriculum ? snapshot.extra.curriculum.cue_rule : 0 });
  const checks = { deadlines: 0, starts: 0, mismatches: [] };
  let momentsOk = 0, statesOk = 0, count = 0;
  const initial = stateDiff(world.state(), snapshot.extra.world);
  if (initial.length) checks.mismatches.push(`the loaded state differs from the dump in ${initial.join(", ")}`);
  for (const line of events) {
    if (count >= opt.limit) break;
    count += 1;
    for (const op of line.ops) applyOp(world, op, checks);
    let m;
    if (line.event.op === "reset") {
      if (world.rewardRule && world.rewardRule.kind === "reach") { checks.starts += 1; if (!cur.farCandidates(world.rewardRule.target).some((c) => sameCell(c, line.event.agent))) checks.mismatches.push(`event ${count}: the request start ${line.event.agent} is not a far candidate`); }
      m = world.reset({ agent: line.event.agent });
    } else if (line.event.op === "act") m = world.act(line.event.decision_id, line.event.action);
    else throw Error(`unknown world event ${line.event.op}`);
    const md = momentDiff(m, line.moment);
    if (!md.length) momentsOk += 1; else checks.mismatches.push(`event ${count} (${line.event.op}): moment differs in ${md.join(", ")}`);
    const sd = stateDiff(world.state(), line.state);
    if (!sd.length) statesOk += 1; else checks.mismatches.push(`event ${count} (${line.event.op}): state differs in ${sd.join(", ")}`);
  }
  console.log("");
  console.log(`world: ${count} events replayed from the snapshot's state; moments identical ${momentsOk}/${count}; states identical ${statesOk}/${count}; request deadlines checked ${checks.deadlines}; request starts checked ${checks.starts}; ${((Date.now() - started) / 1000).toFixed(2)} s`);
  return checks.mismatches;
}

function main() {
  const opt = args(process.argv.slice(2));
  const dir = resolve(opt.fixture);
  const snapshot = JSON.parse(readFileSync(join(dir, "snapshot.json"), "utf8"));
  const lines = readFileSync(join(dir, "stream.jsonl"), "utf8").split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l));
  const events = readFileSync(join(dir, "world.jsonl"), "utf8").split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l));
  const final = JSON.parse(readFileSync(join(dir, "final.json"), "utf8"));
  const tol = final.tolerances || { activation: 1e-9, efficacy: 1e-8, prediction: 1e-8, stores: 1e-12, recall: 1e-12, records: 1e-10 };
  if (!snapshotCarriesLife(snapshot)) { console.error("the snapshot carries no pending record or no ring potentials; regenerate it with tools/parity_fixture.py"); return 2; }
  const records = loadRecordsSidecar(dir, snapshot, opt.sidecar);
  const agent = new ExperienceAgent(snapshot, { records: records.sidecar });
  const brain = new Brain(agent, snapshot.extra);
  console.log(`engine: ${agent.n} neurons, ${agent.E} synapses, ${agent.layout.ranges} ranges, ${agent.layout.pairs} blocks; recall port ${agent.ports.recall.length}`);
  if (agent.records) console.log(`records head: ${agent.records.granules} granules, ${agent.records.config.active} active, ${agent.reading.length} reading neurons, ${agent.records.fields.length} fields, ${agent.records.parameters().toLocaleString()} record parameters; tables from ${records.note}; ${agent.records.writes} writes so far`);
  console.log(`life records: ring from ${agent.ringSource}; pending from ${agent.pendingSource}`);
  console.log(`planner: max_nodes ${brain.planner.maxNodes}, depth ${brain.planner.depth}; stores loaded (place writes ${brain.places.writes}, map writes ${brain.map.writes}, word writes ${brain.words.writes}); bookkeeping: ${brain.searches} searches, ${brain.reused} reuses so far`);
  console.log(`stream: ${lines.length} moments; world: ${events.length} events; tolerances ${JSON.stringify(tol)}`);
  const started = Date.now();
  let decisions = 0, matched = 0, versionsAgree = 0, hashesAgree = 0, hashLines = 0, predictionMax = 0, probabilityMax = 0, logpMax = 0, learningMax = 0, storesMax = 0, recallMax = 0, countersAgree = 0, plansAgree = 0, searches = 0, reuses = 0, greedy = 0, writesAgree = 0, writeLines = 0;
  const mismatches = [];
  const count = Math.min(lines.length, opt.limit);
  for (let k = 0; k < count; k++) {
    const line = lines[k];
    brain.setEpsilon(line.epsilon);
    let decision;
    try { decision = brain.step(line.moment); } catch (err) { console.error(`line ${k} (event ${line.moment.event_id}): engine error: ${err.stack || err.message}`); return 1; }
    const expected = line.decision;
    if ((decision === null) !== (expected === null)) mismatches.push(`line ${k}: decision ${decision === null ? "null" : "present"} vs expected ${expected === null ? "null" : "present"}`);
    else if (decision !== null) {
      decisions += 1;
      const ok = decision.action === expected.action && decision.controller === expected.controller;
      if (ok) matched += 1; else mismatches.push(`line ${k} (event ${line.moment.event_id}, ${line.phase}): action ${decision.action}/${decision.controller} vs expected ${expected.action}/${expected.controller}`);
      for (const name of Object.keys(expected.prediction)) predictionMax = Math.max(predictionMax, maxAbs(decision.prediction[name] || [], expected.prediction[name]));
      probabilityMax = Math.max(probabilityMax, maxAbs(decision.probabilities, expected.probabilities));
      logpMax = Math.max(logpMax, Math.abs(decision.log_probability - expected.log_probability));
      if (decision.decision_id !== expected.decision_id || decision.context_version !== expected.context_version || decision.parameter_version !== expected.parameter_version) mismatches.push(`line ${k}: ids differ (decision ${decision.decision_id}/${expected.decision_id}, versions ${decision.parameter_version}/${expected.parameter_version}, context ${decision.context_version}/${expected.context_version})`);
      for (const key of Object.keys(expected.budget || {})) if (decision.budget[key] !== expected.budget[key]) mismatches.push(`line ${k}: budget ${key} ${decision.budget[key]} vs ${expected.budget[key]}`);
      if (expected.controller === "planner") { if (line.reused) reuses += 1; else if (line.searched) searches += 1; else greedy += 1; }
    }
    if (agent.parameterVersion === line.parameter_version) versionsAgree += 1; else mismatches.push(`line ${k}: parameter version ${agent.parameterVersion} vs expected ${line.parameter_version}`);
    if (line.efficacy_sha256) { hashLines += 1; if (sha256(agent.efficacy) === line.efficacy_sha256) hashesAgree += 1; }
    for (const [key, value] of Object.entries(line.learning || {})) { const mine = agent.lastReport.learning[key]; if (mine === undefined) mismatches.push(`line ${k}: learning report lacks ${key}`); else learningMax = Math.max(learningMax, Math.abs(mine - value)); }
    const counters = brain.counters();
    if (same(counters, line.planner)) countersAgree += 1; else mismatches.push(`line ${k}: planner counters ${JSON.stringify(counters)} vs expected ${JSON.stringify(line.planner)}`);
    const plan = { plan: brain.plan.slice(), cells: brain.planCells.map((c) => [c[0], c[1]]), key: brain.planKey === null ? null : [brain.planKey[0], [brain.planKey[1][0], brain.planKey[1][1]]] };
    if (same(plan, { plan: line.plan, cells: line.plan_cells, key: line.plan_key })) plansAgree += 1; else mismatches.push(`line ${k}: plan ${JSON.stringify(plan)} vs expected ${JSON.stringify({ plan: line.plan, cells: line.plan_cells, key: line.plan_key })}`);
    storesMax = Math.max(storesMax, storeDiff(brain.stores(), line.stores));
    recallMax = Math.max(recallMax, maxAbs(agent.lastRecall, line.recall));
    if (line.record_writes !== undefined) { writeLines += 1; if (agent.ledger.record_writes === line.record_writes && (line.imagined === undefined || agent.ledger.imagined === line.imagined)) writesAgree += 1; else mismatches.push(`line ${k}: record writes ${agent.ledger.record_writes} vs ${line.record_writes}, imagined ${agent.ledger.imagined} vs ${line.imagined}`); }
    if (!opt.quiet && (k % 10 === 9 || k === count - 1)) console.log(`  ${k + 1}/${count} moments · ${matched}/${decisions} decisions match · version ${agent.parameterVersion} · prediction max ${fmt(predictionMax)} · stores max ${fmt(storesMax)} · ${((Date.now() - started) / 1000).toFixed(1)} s`);
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
    stores: [storeDiff(brain.stores(), final.stores), tol.stores ?? 1e-12],
    recall: [maxAbs(agent.lastRecall, final.recall), tol.recall ?? 1e-12],
  };
  if (rec.tableDiff !== null) { finals.records_mean = [rec.meanDiff, tol.records ?? 1e-10]; if (rec.blockDiff !== null) finals.records_pathway_norms = [rec.blockDiff, tol.records ?? 1e-10]; finals.records_tables = [rec.tableDiff, tol.records ?? 1e-10]; }
  const full = count === lines.length;
  console.log("");
  console.log(`decisions matching (action, controller): ${matched}/${decisions} (${searches} searches, ${reuses} plan reuses, ${greedy} reward-greedy cue decisions, ${decisions - searches - reuses - greedy} exploration)`);
  console.log(`parameter version agreement per line: ${versionsAgree}/${count}; final ${state.parameter_version} vs ${final.parameter_version}`);
  console.log(`efficacy sha256 agreement per line: ${hashesAgree}/${hashLines}`);
  console.log(`planner counters agreement per line (searches, reuses, expansions, fallbacks, pruned, rng): ${countersAgree}/${count}; plan agreement per line: ${plansAgree}/${count}`);
  console.log(`max abs prediction difference: ${fmt(predictionMax)} (tolerance ${tol.prediction})`);
  console.log(`max abs probability difference: ${fmt(probabilityMax)}; log-probability: ${fmt(logpMax)}; learning-report numbers: ${fmt(learningMax)}`);
  console.log(`max abs store difference per line: ${fmt(storesMax)} (tolerance ${tol.stores}); recall port: ${fmt(recallMax)} (tolerance ${tol.recall})`);
  for (const [name, [diff, limit]] of Object.entries(finals)) console.log(`final max abs difference ${name}: ${fmt(diff)} (tolerance ${limit})${full ? "" : " [partial stream]"}`);
  if (rec.tableDiff !== null) {
    console.log(`  per table: ${Object.entries(rec.tables).map(([name, d]) => `${name} ${fmt(d)}`).join(" · ")}`);
    console.log(`record writes agreement per line (writes, imagined): ${writesAgree}/${writeLines}; final writes ${rec.writes[0]} vs ${rec.writes[1]} (ledger ${state.ledger.record_writes} vs ${final.ledger.record_writes}; imagined ${state.ledger.imagined} vs ${final.ledger.imagined})`);
  }
  console.log(`rng states: action ${state.rng.action} vs ${final.rng.action}; replay ${state.rng.replay} vs ${final.rng.replay}; planner fallback ${brain.planner.rng.state} vs ${final.planner.rng}`);
  console.log(`ring: stored ${state.ring_stored} vs ${final.ring_stored}; at ${state.ring_at} vs ${final.ring_at}`);
  console.log(`counters: ${JSON.stringify(brain.counters())}`);
  console.log(`expected: ${JSON.stringify(final.planner)}`);
  console.log(`ledger phases: ${JSON.stringify(state.ledger.phases)}`);
  console.log(`expected phases: ${JSON.stringify(final.ledger.phases)}`);
  console.log(`elapsed ${((Date.now() - started) / 1000).toFixed(1)} s`);
  let failed = mismatches.length > 0 || matched !== decisions;
  if (predictionMax > tol.prediction) { failed = true; mismatches.push(`prediction difference ${predictionMax} exceeds ${tol.prediction}`); }
  if (storesMax > (tol.stores ?? 1e-12)) { failed = true; mismatches.push(`store difference ${storesMax} exceeds ${tol.stores}`); }
  if (recallMax > (tol.recall ?? 1e-12)) { failed = true; mismatches.push(`recall difference ${recallMax} exceeds ${tol.recall}`); }
  if (full) {
    for (const [name, [diff, limit]] of Object.entries(finals)) if (!(diff <= limit)) { failed = true; mismatches.push(`final ${name} difference ${diff} exceeds ${limit}`); }
    if (state.rng.action !== final.rng.action || state.rng.replay !== final.rng.replay || brain.planner.rng.state !== final.planner.rng) { failed = true; mismatches.push("rng states differ"); }
    if (state.parameter_version !== final.parameter_version) { failed = true; mismatches.push("final parameter version differs"); }
    if (state.ring_stored !== final.ring_stored || state.ring_at !== final.ring_at) { failed = true; mismatches.push("ring cursors differ"); }
    if (!same(brain.counters(), final.planner)) { failed = true; mismatches.push("final planner counters differ"); }
    if (!same(state.ledger.phases, final.ledger.phases)) { failed = true; mismatches.push("ledger phase counts differ"); }
    if (rec.tableDiff !== null && (rec.writes[0] !== rec.writes[1] || rec.seen[0] !== rec.seen[1] || state.ledger.record_writes !== final.ledger.record_writes || state.ledger.imagined !== final.ledger.imagined)) { failed = true; mismatches.push("record write or imagined-read counts differ"); }
  }
  const worldMismatches = worldSection(snapshot, events, opt);
  if (worldMismatches.length) failed = true;
  const all = [...mismatches, ...worldMismatches];
  if (all.length) { console.log(""); console.log(`${all.length} mismatch(es):`); for (const m of all.slice(0, 40)) console.log("  " + m); }
  console.log(failed ? "PARITY FAILED" : "PARITY OK");
  return failed ? 1 : 0;
}

process.exit(main());
