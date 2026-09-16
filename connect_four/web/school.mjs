#!/usr/bin/env node
// School: the page's brain plays the perfect solver, game after game, remembering every position
// its searches prove, so a line it lost is proven further back each time until the losing move is
// known. Games are played through the same brain.js the page runs.
//
//     node connect_four/web/school.mjs <checkpoint.json> <cortices.json> <bridge> <book> \
//         --side first|second|both --games N --depth 10 --budget 131072 --seed 1 --out school.json
//
// The perfect solver plays a score-maximal column, ties broken by the seeded generator; with
// --strength below 1 it plays a random column the rest of the time. The output holds the memory
// (boards, sides, results) and one row per game (result, plies, the first ply a result was thrown).

import { readFileSync, writeFileSync } from "node:fs";
import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import { Brain, CANDIDATE, OPPONENT } from "./brain.js";

const argv = process.argv.slice(2);
const opt = { side: "first", games: 10, depth: 10, budget: 131072, seed: 1, strength: 1.0, out: "school.json", memory: null, threat: null, parity: null, band: null, first: 1.0, second: 1.0, lateStones: 99, lateDepth: 16, lateBudget: 1048576, tutor: "" };
// --tutor first|second|both (with --side watch): the tutored side plays the column the memory holds for the board
// when it holds one, and the perfect column where it holds none, so every watched game extends the brain's own lines
// --side watch: the solver plays itself, the first player at --first and the second at --second (a perfect column with
// that probability, else a random one), and the brain remembers the winner's columns without playing
const positional = [];
for (let i = 0; i < argv.length; i++) {
  const a = argv[i];
  if (a.startsWith("--")) opt[a.slice(2).replace(/-([a-z])/g, (_, c) => c.toUpperCase())] = argv[++i]; else positional.push(a);
}
const [checkpointFile, corticesFile, bridge, book] = positional;
opt.games = Number(opt.games); opt.depth = Number(opt.depth); opt.budget = Number(opt.budget); opt.seed = Number(opt.seed); opt.strength = Number(opt.strength); opt.first = Number(opt.first); opt.second = Number(opt.second); opt.lateStones = Number(opt.lateStones); opt.lateDepth = Number(opt.lateDepth); opt.lateBudget = Number(opt.lateBudget);

const brain = new Brain(JSON.parse(readFileSync(checkpointFile, "utf8")), { cortices: JSON.parse(readFileSync(corticesFile, "utf8")), learning: false });
brain.planner.remember = true;
brain.planner.imitate = true;
if (opt.threat !== null) brain.imagination.threatWeight = Number(opt.threat);
if (opt.parity !== null) brain.imagination.parityWeight = Number(opt.parity);
if (opt.band !== null) brain.planner.tieBand = Number(opt.band);
if (opt.memory) {  // a memory from earlier schooling, merged in
  const m = JSON.parse(readFileSync(opt.memory, "utf8"));
  for (let i = 0; i < m.entries; i++) brain.planner.memory.set(brain.planner._key(Int8Array.from(m.boards[i]), m.sides[i]), m.results[i]);
  for (const w of m.wins || []) for (let n = 0; n < w[2]; n++) brain.planner.rememberWin(Int8Array.from(w[0]), w[1]);
}
const cols = brain.game.cols, rows = brain.game.rows, cells = rows * cols;

let s = (opt.seed * 2654435761) >>> 0 || 1;
const rnd = () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };

const proc = spawn(bridge, book ? [book] : [], { stdio: ["pipe", "pipe", "ignore"] });
const lines = createInterface({ input: proc.stdout });
const queue = [];
lines.on("line", (line) => { const r = queue.shift(); if (r) r(line); });
const cache = new Map();
function solve(rel) {  // rel: the board as the side to move sees it (0 empty, 1 own, 2 other)
  let key = ""; for (let i = 0; i < cells; i++) key += ".xo"[rel[i]];
  const hit = cache.get(key); if (hit) return Promise.resolve(hit);
  return new Promise((resolve) => { queue.push((line) => { const a = line.trim().split(/\s+/).slice(0, 7).map(Number); if (cache.size < 200000) cache.set(key, a); resolve(a); }); proc.stdin.write(key + "\n"); });
}
const INVALID = -1000;

function relative(board, side) { return side === 1 ? board : Int8Array.from(board, (v) => (v === 0 ? 0 : 3 - v)); }
function winner(board, cell) {  // whether the stone at cell completes four
  const r = Math.floor(cell / cols), c = cell % cols, me = board[cell];
  for (const [dr, dc] of [[0, 1], [1, 0], [1, 1], [1, -1]]) {
    let n = 1;
    for (const d of [1, -1]) for (let k = 1; k < 4; k++) { const rr = r + d * k * dr, cc = c + d * k * dc; if (rr < 0 || rr >= rows || cc < 0 || cc >= cols || board[rr * cols + cc] !== me) break; n++; }
    if (n >= 4) return true;
  }
  return false;
}

const games = [];
const t0 = Date.now();
for (let g = 0; g < opt.games; g++) {
  const watching = opt.side === "watch";
  const brainFirst = opt.side === "first" || (opt.side === "both" && g % 2 === 0);
  const board = new Int8Array(cells), h = new Array(cols).fill(0); let side = 1, moves = [], result = null, throwAt = null;
  while (result === null) {
    const rel = relative(board, side);
    const legal = Uint8Array.from({ length: cols }, (_, c) => (h[c] < rows ? 1 : 0));
    const brainTurn = !watching && (moves.length % 2 === 0) === brainFirst;
    let col;
    const scores = await solve(rel);
    let best = -Infinity; for (let c = 0; c < cols; c++) if (scores[c] !== INVALID && scores[c] > best) best = scores[c];
    const strength = watching ? (side === 1 ? opt.first : opt.second) : opt.strength;
    const tutored = watching && (opt.tutor === "both" || (opt.tutor === "first" && side === 1) || (opt.tutor === "second" && side === 2));
    const remembered = tutored ? brain.planner.wins.get(String.fromCharCode.apply(null, rel)) : null;
    if (remembered) {
      let bestCount = 0; col = -1;
      for (const [c, n] of remembered) if (legal[c] && n > bestCount) { bestCount = n; col = c; }
      if (col < 0) { const top = []; for (let c = 0; c < cols; c++) if (scores[c] === best) top.push(c); col = top[Math.floor(rnd() * top.length)]; }
    } else if (tutored) {
      const top = []; for (let c = 0; c < cols; c++) if (scores[c] === best) top.push(c); col = top[Math.floor(rnd() * top.length)];
    } else if (brainTurn) {
      const late = moves.length >= opt.lateStones;
      col = brain.planner.search(rel, legal, late ? opt.lateDepth : opt.depth, late ? opt.lateBudget : opt.budget).column;
      if (throwAt === null && best >= 0 && scores[col] < 0) throwAt = [moves.length, best, scores[col]];
    } else if (rnd() >= strength) {
      const open = []; for (let c = 0; c < cols; c++) if (legal[c]) open.push(c); col = open[Math.floor(rnd() * open.length)];
    } else {
      const top = []; for (let c = 0; c < cols; c++) if (scores[c] === best) top.push(c); col = top[Math.floor(rnd() * top.length)];
    }
    const cell = h[col] * cols + col; board[cell] = side; h[col]++; moves.push(col);
    if (winner(board, cell)) result = watching ? (side === 1 ? "first" : "second") : (side === 1) === brainFirst ? "win" : "loss";
    else if (moves.length === cells) result = "draw";
    side = 3 - side;
  }
  // the winner's moves, on the boards as the winner saw them, join the memory of winning columns
  if (result !== "draw") {
    const winnerSide = watching ? (result === "first" ? 1 : 2) : (result === "win") === brainFirst ? 1 : 2;
    const b = new Int8Array(cells), hh = new Array(cols).fill(0); let sd = 1;
    for (const col of moves) {
      if (sd === winnerSide) brain.planner.rememberWin(relative(b, sd), col);
      b[hh[col] * cols + col] = sd; hh[col]++; sd = 3 - sd;
    }
  }
  games.push({ game: g, brain_first: brainFirst, result, plies: moves.length, thrown_at: throwAt, memory: brain.planner.memory.size, wins: brain.planner.wins.size, moves });
  if (watching && g % 100 !== 99) continue;
  console.log(`g${String(g).padStart(3, "0")} ${watching ? "watch " : brainFirst ? "first " : "second"} ${result.padEnd(6)} ${String(moves.length).padStart(2)} plies thrown_at=${throwAt ? throwAt.join("/") : "-"} memory=${brain.planner.memory.size} wins=${brain.planner.wins.size} ${((Date.now() - t0) / 1000).toFixed(0)}s`);
}
const boards = [], sides = [], results = [];
for (const [key, value] of brain.planner.memory) { const b = []; for (let i = 0; i < cells; i++) b.push(key.charCodeAt(i)); boards.push(b); sides.push(Number(key[cells])); results.push(value); }
const wins = [];
for (const [key, counts] of brain.planner.wins) { const b = []; for (let i = 0; i < cells; i++) b.push(key.charCodeAt(i)); for (const [col, n] of counts) wins.push([b, col, n]); }
writeFileSync(opt.out, JSON.stringify({ format: "cadence-connect-four-school/1", options: opt, entries: boards.length, boards, sides, results, wins, games }));
const tally = { win: 0, draw: 0, loss: 0, first: 0, second: 0 }; for (const g of games) tally[g.result]++;
console.log(`DONE ${opt.side} vs solver at ${opt.strength}: ${JSON.stringify(tally)} memory=${brain.planner.memory.size}`);
proc.stdin.end();
