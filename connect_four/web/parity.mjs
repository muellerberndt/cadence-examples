#!/usr/bin/env node
// Parity of connect_four/web/brain.js against the Python brain on a recorded fixture:
//
//     node connect_four/web/parity.mjs runs/connect_four/parity [--limit N] [--quiet]
//
// Loads the checkpoint the fixture was recorded from, replays its games move by move through
// the JavaScript brain (the same games against the same opponent columns, learning on) and
// holds every step to the Python numbers: the chosen column, the predicted landings, the
// imagined next board and every counter exactly, the read values, the search value and the
// moments of every record table to 1e-9. After the last game the record tables, the running
// means, the write counts and the state of the planner's generator are compared with
// final.json. Exits non-zero on the first kind of difference it finds.

import { existsSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { unpackF64 } from "../../web/engine.js";
import { gameConfig, World } from "./game.js";
import { Brain, CANDIDATE, viewOf } from "./brain.js";

const FORMAT = "cadence-connect-four-parity/1";

function args(argv) {
  const out = { fixture: null, limit: Infinity, quiet: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--limit") out.limit = Number(argv[++i]);
    else if (a === "--quiet") out.quiet = true;
    else if (out.fixture === null) out.fixture = a;
    else throw Error(`unexpected argument ${a}`);
  }
  if (!out.fixture) throw Error("usage: node parity.mjs <fixture dir> [--limit N] [--quiet]");
  return out;
}

const fmt = (x) => (x === 0 ? "0" : Number(x).toExponential(3));

class Differences {
  constructor(tolerances) { this.tol = tolerances; this.rows = []; this.worst = {}; }
  fail(where, what, mine, theirs) { if (this.rows.length < 12) this.rows.push(`${where}: ${what} is ${JSON.stringify(mine)} here and ${JSON.stringify(theirs)} in Python`); else if (this.rows.length === 12) this.rows.push("…"); this.count = (this.count || 0) + 1; }
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
  const fixture = JSON.parse(readFileSync(join(dir, "fixture.json"), "utf8"));
  if (fixture.format !== FORMAT) throw Error(`${dir}/fixture.json is not a ${FORMAT} fixture`);
  const checkpoint = JSON.parse(readFileSync(join(dir, fixture.checkpoint), "utf8"));
  const cortices = existsSync(join(dir, "cortices.json")) ? JSON.parse(readFileSync(join(dir, "cortices.json"), "utf8")) : null;
  const final = JSON.parse(readFileSync(join(dir, "final.json"), "utf8"));
  const diff = new Differences(fixture.tolerances || { table: 1e-9, read: 1e-9, moment: 1e-9 });

  const brain = new Brain(checkpoint, { cortices });
  const config = gameConfig(checkpoint.brain.game);
  console.log(`checkpoint: ${checkpoint.label} · ${checkpoint.brain.counts.transitions.toLocaleString()} transitions, ${brain.agent.n} neurons, ${brain.agent.E} synapses`);
  console.log(`cortices: drop ${brain.drop.records.cells} cells (${brain.drop.records.inputs} inputs), lines ${brain.lines.records.cells} cells (${brain.lines.records.inputs} inputs); ${brain.drop.records.writes + brain.lines.records.writes} writes so far`);
  console.log(`fixture: ${fixture.games.length} games, ${fixture.steps.length} steps, recorded in ${fixture.wall_seconds.toFixed(1)} s`);
  console.log(`expansion: ${cortices ? "from cortices.json" : "regenerated from the seed"}; largest difference against the seed's own draw: drop ${fmt(brain.expansion.drop || 0)}, lines ${fmt(brain.expansion.lines || 0)}`);

  const world = new World(config, { lifeId: fixture.life_id });
  brain.newWorld();
  let at = 0, decisions = 0, moves = 0, slowest = 0, seconds = 0;
  const step = (moment, expect) => {
    const t0 = process.hrtime.bigint();
    const decision = brain.step(moment);
    const took = Number(process.hrtime.bigint() - t0) / 1e6;
    seconds += took / 1000;
    if (moment.action_mask.some((x) => x)) { slowest = Math.max(slowest, took); }
    const where = `game ${expect.game} step ${at}`;
    diff.exact(where, "kind", expect.kind === "decision" && decision === null ? "decision" : expect.kind, expect.kind); // the kinds line up by construction
    diff.exact(where, "event", moment.event_id, expect.event);
    diff.exact(where, "episode", moment.episode_id, expect.episode);
    diff.exact(where, "a decision", decision !== null, expect.decision !== null);
    if (decision !== null && expect.decision !== null) {
      const info = brain.lastSearch, e = expect.decision;
      decisions += 1;
      diff.exact(where, "column", decision.action, e.column);
      diff.exact(where, "controller", decision.controller, e.controller);
      diff.exact(where, "decision_id", decision.decision_id, e.decision_id);
      diff.exact(where, "expansions", decision.budget.expansions, e.budget.expansions);
      diff.exact(where, "depth", decision.budget.depth, e.budget.depth);
      diff.exact(where, "invalid", decision.budget.invalid, e.budget.invalid);
      diff.exactList(where, "next_board", Array.from(decision.prediction.next_board), e.prediction.next_board);
      diff.exactList(where, "valid", Array.from(decision.prediction.valid), e.prediction.valid);
      diff.closeList(where, "value", Array.from(decision.prediction.value), e.prediction.value);
      diff.exactList(where, "landings unchosen", info.landings[0], e.landings[0]);
      diff.exactList(where, "landings chosen", info.landings[1], e.landings[1]);
      diff.exactList(where, "columns", info.columns, e.columns);
      diff.closeList(where, "read", info.read, e.read);
    }
    for (const [name, value] of Object.entries(expect.counts)) diff.exact(where, `counts.${name}`, brain.counts[name], value);
    diff.exact(where, "planner.nodes", brain.planner.nodes, expect.planner.nodes);
    diff.exact(where, "planner.searches", brain.planner.searches, expect.planner.searches);
    diff.exact(where, "planner.invalid", brain.planner.invalid, expect.planner.invalid);
    diff.exact(where, "validity samples", brain.validity.length, expect.validity);
    diff.exact(where, "extended", brain.extended, expect.extended);
    const moments = tableMomentsOf(brain);
    for (const [name, values] of Object.entries(expect.moments)) diff.closeList(where, `moments.${name}`, moments[name], values, "moment");
    at += 1;
    return decision;
  };

  for (const game of fixture.games.slice(0, opt.limit)) {
    let moment = world.reset({ first: game.first, opening: game.opening });
    let reply = 0;
    for (;;) {
      const decision = step(moment, fixture.steps[at]);
      if (decision === null) break;
      moves += 1;
      const mid = world.act(decision.decision_id, decision.action);
      if (step(mid, fixture.steps[at]) !== null) throw Error("the intermediate moment produced a decision");
      if (mid.terminated) break;
      moment = world.reply(game.replies[reply++]);
    }
    const played = world.board.moves;
    diff.exactList(`game ${game.index}`, "moves", played, game.moves);
    diff.exact(`game ${game.index}`, "outcome", world.outcome(), game.outcome);
    if (!opt.quiet) console.log(`  game ${game.index}: ${played.length} moves, outcome ${world.outcome()}${played.join("") === game.moves.join("") ? "" : " (DIFFERENT)"}`);
  }

  // the tables, the means and the generator after the last game
  for (const [name, cortex] of [["drop", brain.drop.records], ["lines", brain.lines.records]]) {
    const block = final.cortices[name];
    diff.closeList(`final ${name}`, "mean", cortex.mean, unpackF64(block.mean), "table");
    diff.exact(`final ${name}`, "seen", cortex.seen, block.seen);
    diff.exact(`final ${name}`, "writes", cortex.writes, block.writes);
    for (const field of Object.keys(block.tables)) {
      const mine = cortex.tables[field], theirs = unpackF64(block.tables[field]);
      if (mine.length !== theirs.length) { diff.fail(`final ${name}`, `${field} length`, mine.length, theirs.length); continue; }
      let worst = 0, atWorst = -1;
      for (let k = 0; k < mine.length; k++) { const d = Math.abs(mine[k] - theirs[k]); if (d > worst) { worst = d; atWorst = k; } }
      diff.worst.table = Math.max(diff.worst.table || 0, worst);
      if (!(worst <= (fixture.tolerances.table ?? 1e-9))) diff.fail(`final ${name}`, `${field} (largest difference at cell ${atWorst})`, worst, 0);
      console.log(`  final ${name}/${field}: ${mine.length.toLocaleString()} records, largest difference ${fmt(worst)}`);
    }
  }
  const generator = brain.planner.rng.toDict();
  diff.exact("final", "generator state", generator.state, final.planner_generator.state);
  diff.exact("final", "generator inc", generator.inc, final.planner_generator.inc);
  for (const [name, value] of Object.entries(final.counts)) diff.exact("final", `counts.${name}`, brain.counts[name], value);
  diff.exact("final", "planner.nodes", brain.planner.nodes, final.planner.nodes);

  console.log(`steps ${at}, decisions ${decisions}, brain moves ${moves}; ${(seconds * 1000 / Math.max(1, at)).toFixed(1)} ms per step, slowest decision ${slowest.toFixed(1)} ms`);
  console.log(`largest differences: tables ${fmt(diff.worst.table || 0)}, reads ${fmt(diff.worst.read || 0)}, moments ${fmt(diff.worst.moment || 0)}`);
  if (diff.rows.length) {
    console.log(`PARITY FAILED (${diff.count} differences)`);
    for (const row of diff.rows) console.log(`  ${row}`);
    return 1;
  }
  console.log("PARITY HELD: every move, landing, prediction and record table matches the Python brain");
  return 0;
}

/** The moments of every record table, as connect_four/tools/parity_fixture.py records them. */
function tableMomentsOf(brain) {
  const out = {};
  for (const [name, records] of [["drop", brain.drop.records], ["lines", brain.lines.records]]) {
    for (const f of records.fields) {
      const table = records.tables[f.name];
      let sum = 0.0, absolute = 0.0, square = 0.0, largest = 0.0;
      for (let k = 0; k < table.length; k++) { const v = table[k], a = v < 0 ? -v : v; sum += v; absolute += a; square += v * v; if (a > largest) largest = a; }
      out[`${name}/${f.name}`] = [sum, absolute, square, largest];
    }
    let mean = 0.0;
    for (const v of records.mean) mean += v;
    out[`${name}/mean`] = [mean, records.seen, records.writes];
  }
  return out;
}

process.exit(main());
