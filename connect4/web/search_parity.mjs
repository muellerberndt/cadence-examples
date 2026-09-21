// node connect4/web/search_parity.mjs <dir with brain.json and search.json>
import { readFileSync } from "node:fs";
import { ValuePatch } from "./patch.js";
import { Brain, Position } from "./brain.js";
const dir = process.argv[2];
const patch = new ValuePatch(JSON.parse(readFileSync(`${dir}/brain.json`, "utf8")));
const fx = JSON.parse(readFileSync(`${dir}/search.json`, "utf8"));
const brain = new Brain(patch, { reads: fx.budget, lateStones: fx.late_stones, lateReads: fx.late_budget, visits: fx.visits });
let bad = 0, worst = 0;
for (const [i, p] of fx.positions.entries()) {
  const position = new Position();
  for (const c of p.columns) position.play(c);
  const before = position.plies;
  const t = brain.think(position);
  const same = t.column === p.column && t.depth === p.depth && t.reads === p.reads && t.visits === p.visits && t.proven === p.proven && Math.abs(t.value - p.value) < 1e-9 && JSON.stringify(t.line) === JSON.stringify(p.line);
  worst = Math.max(worst, Math.abs(t.value - p.value));
  if (position.plies !== before) { console.log(`position ${i}: the board was left with ${position.plies} stones, began with ${before}`); bad++; }
  else if (!same) { bad++; console.log(`position ${i} (${p.columns.length} stones): python`, JSON.stringify({ c: p.column, v: p.value, d: p.depth, r: p.reads, l: p.line }), "js", JSON.stringify({ c: t.column, v: t.value, d: t.depth, r: t.reads, l: t.line })); }
}
console.log(`searched ${fx.positions.length} positions: ${bad} differ; worst |value| ${worst.toExponential(2)}`);
console.log(bad ? "SEARCH PARITY FAILED" : "SEARCH PARITY HELD");
process.exit(bad ? 1 : 0);
