// The page's engine in real games, every one of its moves graded by the perfect solver.
//
//   node connect4/web/arena.mjs --brain <dir with brain.json> --opponents one_ply solver_070 solver_100 \
//        alphazero=<checkpoint>@25 --games 40 --out <receipt.json> [--budget 2000 --late-stones 18 --late-budget 200000 --visits 1000000]
//
// Games are paired (the brain opens the even games; both games of a pair share the opponent's
// seed). Pascal Pons' solver grades each column the brain chose: optimal, slip (worse, the
// theoretical result kept) or blunder (the result changes). Blunder rates are over positions
// that are not already lost, by the stones on the board. `--subject minimax` measures the
// rules-only search with the same budget of visited positions, as the control.
import { readFileSync, writeFileSync } from "node:fs";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { ValuePatch, mulberry32 } from "./patch.js";
import { Brain, Position, ORDER, CELLS } from "./brain.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PONS = path.join(HERE, "..", "external", "pons");
const INVALID = -1000;
const BANDS = [[0, 8], [8, 16], [16, 24], [24, 42]];

// a subprocess that answers one line per request line
class Lines {
  constructor(command, args, options = {}) {
    this.child = spawn(command, args, { stdio: ["pipe", "pipe", "inherit"], ...options });
    this.waiting = []; this.buffer = "";
    // a helper that dies must stop the arena, not leave it waiting for an answer that never comes
    this.child.on("exit", (code) => { if (code) { console.error(`${command} ${args.slice(0, 3).join(" ")} exited with code ${code}`); process.exit(1); } });
    this.child.stdout.on("data", (chunk) => {
      this.buffer += chunk;
      let at;
      while ((at = this.buffer.indexOf("\n")) >= 0) {
        const line = this.buffer.slice(0, at); this.buffer = this.buffer.slice(at + 1);
        const resolve = this.waiting.shift(); if (resolve) resolve(line);
      }
    });
  }
  ask(line) { return new Promise((resolve) => { this.waiting.push(resolve); this.child.stdin.write(line + "\n"); }); }
  line() { return new Promise((resolve) => this.waiting.push(resolve)); }
  close() { this.child.stdin.end(); }
}

const boardLine = (position) => { let s = ""; const side = position.side; for (let k = 0; k < CELLS; k++) { const v = position.cells[k]; s += v === 0 ? "." : v === side ? "x" : "o"; } return s; };

class Solver {
  constructor() { this.lines = new Lines(path.join(PONS, "c4bridge"), [path.join(PONS, "7x6.book")]); this.cache = new Map(); }
  async scores(position) {
    const line = boardLine(position);
    if (!this.cache.has(line)) this.cache.set(line, (await this.lines.ask(line)).trim().split(/\s+/).slice(0, 7).map(Number));
    return this.cache.get(line);
  }
}

const pick = (random, list) => list[Math.floor(random() * list.length)];

function makeOpponent(name, seed, shared) {
  const random = mulberry32(seed);
  if (name === "random") return async (position) => pick(random, position.legal());
  if (name === "one_ply") return async (position) => {
    const legal = position.legal(), wins = legal.filter((c) => position.wins(c));
    if (wins.length) return pick(random, wins);
    position.plies++;                                    // the other side's view of the same board
    const blocks = legal.filter((c) => position.wins(c));
    position.plies--;
    return pick(random, blocks.length ? blocks : legal);
  };
  if (name.startsWith("solver_")) {
    const strength = Number(name.split("_")[1]) / 100;
    return async (position) => {
      const legal = position.legal();
      if (random() >= strength) return pick(random, legal);
      const scores = await shared.solver.scores(position), best = Math.max(...legal.map((c) => scores[c]));
      return pick(random, legal.filter((c) => scores[c] === best));
    };
  }
  if (name.startsWith("alphazero=")) {
    const server = shared.servers.get(name);
    return async (position) => Number(await server.ask(`${seed} ${boardLine(position)}`));
  }
  throw new Error(`unknown opponent ${name}`);
}

// the control: alpha-beta over the rules alone, deepened until `nodes` positions are visited
function minimaxSubject(nodes) {
  let visited = 0;
  const search = (position, depth, alpha, beta) => {
    if (++visited > nodes) throw new Error("spent");
    const legal = position.legal();
    for (const c of legal) if (position.wins(c)) return 1.0 + depth / 100;
    if (depth === 0 || position.plies >= 41) return 0.0;
    let best = -2.0;
    for (const c of legal) {
      position.play(c); let v; try { v = -search(position, depth - 1, -beta, -alpha); } finally { position.undo(); }
      best = Math.max(best, v); alpha = Math.max(alpha, best); if (alpha >= beta) break;
    }
    return best;
  };
  return { think(position) {
    const legal = position.legal(); visited = 0; let chosen = legal[0], reached = 0;
    for (let depth = 1; depth <= CELLS - position.plies; depth++) {
      try {
        let top = -Infinity, column = legal[0];
        for (const c of legal) {
          let v; if (position.wins(c)) v = 2.0; else { position.play(c); try { v = -search(position, depth - 1, -2.0, 2.0); } finally { position.undo(); } }
          if (v > top) { top = v; column = c; }
        }
        chosen = column; reached = depth;
      } catch { break; }
    }
    return { column: chosen, depth: reached, reads: visited };
  } };
}

const quality = (scores, column) => { const best = Math.max(...scores.filter((v) => v !== INVALID)), chosen = scores[column]; return chosen === best ? "optimal" : Math.sign(chosen) === Math.sign(best) ? "slip" : "blunder"; };
const rates = (moves) => { const open = moves.filter((m) => m.best >= 0); return { moves: moves.length, optimal: moves.length ? moves.filter((m) => m.quality === "optimal").length / moves.length : null, not_lost_moves: open.length, blunder_when_not_lost: open.length ? open.filter((m) => m.quality === "blunder").length / open.length : null }; };

async function main() {
  const args = process.argv.slice(2), opt = { opponents: [], games: 40, budget: 2000, "late-stones": 18, "late-budget": 200000, visits: 1000000, seed: 0, subject: "brain" };
  for (let i = 0; i < args.length; i++) {
    const key = args[i].replace(/^--/, "");
    if (key === "opponents") { while (args[i + 1] && !args[i + 1].startsWith("--")) opt.opponents.push(args[++i]); } else opt[key] = args[++i];
  }
  for (const k of ["games", "budget", "late-stones", "late-budget", "visits", "seed"]) opt[k] = Number(opt[k]);
  const shared = { solver: new Solver(), servers: new Map() };
  for (const name of opt.opponents) if (name.startsWith("alphazero=")) {
    const [checkpoint, sims] = name.slice("alphazero=".length).split("@");
    const server = new Lines(process.env.PYTHON || "python3", [path.join(HERE, "..", "bench", "alphazero", "serve.py"), checkpoint, sims || "25"]);   // by path: "-m connect4..." would import this package, whose name alpha-zero-general's game package shares
    await server.line(); shared.servers.set(name, server);
  }
  const subject = opt.subject === "minimax" ? minimaxSubject(opt.budget)
    : new Brain(new ValuePatch(JSON.parse(readFileSync(path.join(opt.brain, "brain.json"), "utf8"))), { reads: opt.budget, lateStones: opt["late-stones"], lateReads: opt["late-budget"], visits: opt.visits });
  const receipt = { format: "cadence-examples.connect4.arena/1", subject: opt.subject, brain: opt.brain || null, search: { reads: opt.budget, late_stones: opt["late-stones"], late_reads: opt["late-budget"], visits: opt.visits }, games_per_opponent: opt.games, opponents: {} };
  for (const [kOpponent, name] of opt.opponents.entries()) {
    const started = Date.now(), results = { first: [0, 0, 0], second: [0, 0, 0] }, moves = [], games = [], latency = [];
    for (let g = 0; g < opt.games; g++) {
      const opponent = makeOpponent(name, 1000 * (opt.seed + 1) + 100 * kOpponent + (g >> 1), shared), subjectFirst = g % 2 === 0, position = new Position();
      let winner = -1;
      while (true) {
        const mine = (position.plies % 2 === 0) === subjectFirst;
        let column;
        if (mine) {
          const scores = await shared.solver.scores(position), t0 = performance.now();
          const thought = subject.think(position);
          latency.push(performance.now() - t0); column = thought.column;
          moves.push({ stones: position.plies, quality: quality(scores, column), best: Math.max(...scores.filter((v) => v !== INVALID)), depth: thought.depth, reads: thought.reads });
        } else column = await opponent(position);
        if (!position.playable(column)) throw new Error(`${mine ? "the subject" : name} chose the full column ${column}`);
        position.play(column);
        if (position.lost) { winner = (position.plies - 1) % 2; break; }
        if (position.full) break;
      }
      const side = subjectFirst ? "first" : "second";
      results[side][winner === -1 ? 1 : winner === (subjectFirst ? 0 : 1) ? 0 : 2]++;
      games.push({ columns: position.columns.join(""), subject_first: subjectFirst, winner });
    }
    latency.sort((a, b) => a - b);
    const entry = { win_draw_loss: results, all: rates(moves), by_stones: Object.fromEntries(BANDS.map(([lo, hi]) => [`${lo}-${hi}`, rates(moves.filter((m) => lo <= m.stones && m.stones < hi))])),
      latency_ms: { mean: latency.reduce((a, b) => a + b, 0) / latency.length, p95: latency[Math.floor(0.95 * (latency.length - 1))], max: latency[latency.length - 1] }, seconds: (Date.now() - started) / 1000, games };
    receipt.opponents[name] = entry;
    const bands = Object.entries(entry.by_stones).map(([b, v]) => `${b}: ${v.blunder_when_not_lost === null ? "-" : v.blunder_when_not_lost.toFixed(3)}`).join("  ");
    console.log(`${name.slice(0, 34).padEnd(34)} first ${JSON.stringify(results.first)} second ${JSON.stringify(results.second)}  optimal ${entry.all.optimal.toFixed(3)}  blunders when not lost ${entry.all.blunder_when_not_lost.toFixed(3)} of ${entry.all.not_lost_moves}  [${bands}]  ${entry.latency_ms.mean.toFixed(0)} ms/move (max ${entry.latency_ms.max.toFixed(0)})  ${entry.seconds.toFixed(0)} s`);
  }
  if (opt.out) writeFileSync(opt.out, JSON.stringify(receipt, null, 1));
  shared.solver.lines.close(); for (const s of shared.servers.values()) s.close();
}
main();
