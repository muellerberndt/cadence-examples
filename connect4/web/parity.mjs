// node connect4/web/parity.mjs <dir with brain.json and parity.json>
// The page's value patch must say what cadence says: the same active cells, the same value.
import { readFileSync } from "node:fs";
import { ValuePatch } from "./patch.js";

const dir = process.argv[2];
const brain = JSON.parse(readFileSync(`${dir}/brain.json`, "utf8"));
const { positions } = JSON.parse(readFileSync(`${dir}/parity.json`, "utf8"));
const patch = new ValuePatch(brain);
let worstValue = 0, worstSlow = 0, worstActivity = 0, differentCells = 0;
for (const p of positions) {
  const trace = {};
  const value = patch.value(p.reading, trace);
  worstValue = Math.max(worstValue, Math.abs(value - p.value));
  worstSlow = Math.max(worstSlow, Math.abs(trace.slow - p.slow));
  const mine = new Map();
  for (let i = 0; i < trace.cells.length; i++) if (trace.activity[i] > 0) mine.set(trace.cells[i], trace.activity[i]);
  const theirs = new Map(p.cells.map((c, i) => [c, p.activity[i]]));
  let same = mine.size === theirs.size;
  for (const [c, a] of theirs) {
    if (!mine.has(c)) { same = false; continue; }
    worstActivity = Math.max(worstActivity, Math.abs(mine.get(c) - a));
  }
  if (!same) differentCells++;
}
console.log(`positions ${positions.length}: cells differ in ${differentCells}; worst |value| ${worstValue.toExponential(2)}, |slow| ${worstSlow.toExponential(2)}, |activity| ${worstActivity.toExponential(2)}`);
if (differentCells > 0 || worstValue > 1e-9) { console.log("PARITY FAILED"); process.exit(1); }
console.log("PARITY HELD");
