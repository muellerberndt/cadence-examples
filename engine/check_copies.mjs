// The examples that carry a copy of the engine carry it byte for byte.
import { readFileSync } from "node:fs";
const pairs = [["engine/brain.js", "fly-matrix/web/brain.js"], ["engine/learner.js", "fly-matrix/web/learner.js"], ["engine/export.py", "fly-matrix/tools/brain_payload.py"]];
let ok = true;
for (const [a, b] of pairs) {
  if (readFileSync(a, "utf8") !== readFileSync(b, "utf8")) { console.error(`${b} differs from ${a}`); ok = false; }
}
if (!ok) process.exit(1);
console.log(`${pairs.length} copies of the engine are exact`);
