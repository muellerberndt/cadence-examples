import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { drop, winner, legal, reason, evaluate, Monitor } from "./brain.js";
const rng = (seed) => () => {
  seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
  return seed / 2 ** 32;
};
function start(seed, plies = 4) {
  const random = rng(seed);
  let b = Array(42).fill(0),
    p = 1;
  for (let i = 0; i < plies; i++) {
    const moves = legal(b);
    b = drop(b, moves[Math.floor(random() * moves.length)], p);
    p = -p;
  }
  return { b, p };
}
function exact(b, p, d) {
  const win = winner(b);
  if (win) return win * p * (100000 + d);
  if (!legal(b).length) return 0;
  if (!d) return evaluate(b, p);
  return Math.max(...legal(b).map((c) => -exact(drop(b, c, p), -p, d - 1)));
}
const conformance = [];
for (let seed = 11; seed < 31; seed++) {
  const { b, p } = start(seed, 10);
  if (winner(b)) continue;
  const values = legal(b).map((c) => [c, -exact(drop(b, c, p), -p, 2)]),
    best = Math.max(...values.map((x) => x[1]));
  const result = reason(b, p, {
    depth: 3,
    maxNodes: 100000,
    monitoring: false,
  });
  // Earlier proven immediate wins have different mate-distance scores but the same winning choice.
  assert.ok(values.some(([c, v]) => c === result.column && v === best));
  conformance.push({
    seed,
    column: result.column,
    optimal: values.filter((x) => x[1] === best).map((x) => x[0]),
  });
}
const games = [];
for (const seed of [11, 23, 37, 51])
  for (const side of [1, -1])
    for (const control of ["one_ply", "four_ply"]) {
      let { b, p } = start(seed),
        moves = 0,
        nodes = 0,
        extensions = 0,
        controlNodes = 0;
      while (!winner(b) && legal(b).length) {
        const us = p === side,
          r = reason(b, p, {
            depth: us ? 6 : control === "one_ply" ? 1 : 4,
            maxNodes: 80000,
            monitoring: us,
          });
        assert.ok(legal(b).includes(r.column));
        b = drop(b, r.column, p);
        p = -p;
        moves++;
        if (us) {
          nodes += r.nodes;
          extensions += +(r.depth > 4);
        } else controlNodes += r.nodes;
      }
      games.push({
        seed,
        side,
        control,
        outcome: winner(b) * side,
        moves,
        nodes,
        controlNodes,
        extensions,
      });
    }
// The monitor must alter the budget on an ambiguous opening.
const b = Array(42).fill(0),
  on = reason(b, 1, { monitoring: true }),
  off = reason(b, 1, { monitoring: false });
assert.ok(on.depth > off.depth);
assert.equal(off.depth, 4);
const report = {
  schema: "cadence.connect-four/v1",
  conformance,
  games,
  monitor_control: {
    on_depth: on.depth,
    off_depth: off.depth,
    on_nodes: on.nodes,
    off_nodes: off.nodes,
  },
  sources: {},
  boundary:
    "Supplied game rules and threat evaluator. Adversarial search is explicit; conventional minimax with the same evaluator can match it. Finite-depth play, not a solved or learned world model.",
};
for (const path of [
  "connect-four/brain.js",
  "connect-four/benchmark.mjs",
  "shared/nervous_system.js",
])
  report.sources[path] = createHash("sha256")
    .update(readFileSync(new URL("../" + path, import.meta.url)))
    .digest("hex");
writeFileSync(
  new URL("./evidence.json", import.meta.url),
  JSON.stringify(report, null, 2) + "\n",
);
console.log(
  JSON.stringify({
    conformance: conformance.length,
    monitor: report.monitor_control,
    controls: Object.fromEntries(
      ["one_ply", "four_ply"].map((c) => {
        const rows = games.filter((g) => g.control === c);
        return [
          c,
          {
            games: rows.length,
            wins: rows.filter((g) => g.outcome === 1).length,
            draws: rows.filter((g) => g.outcome === 0).length,
            losses: rows.filter((g) => g.outcome === -1).length,
          },
        ];
      }),
    ),
  }),
);
