// Parity: the browser engine against cadence 0.50.0 along one continued life.
// Each case's ticks feed the brain's own previous settled state back in, so
// the whole trajectory must match, then each lesson's retained parameters.
import { readFileSync } from "node:fs";
import { Brain } from "../web/brain.js";

const dir = new URL(".", import.meta.url).pathname;
const spec = JSON.parse(readFileSync(dir + "../web/data/brain.json"));
const cases = JSON.parse(readFileSync(dir + "parity_cases.json"));
const params = JSON.parse(readFileSync(dir + "../web/data/params.json"));
const rel = (a, b) => {
  let n = 0, d = 0;
  for (let k = 0; k < a.length; k++) { n += (a[k] - b[k]) ** 2; d += b[k] ** 2; }
  return Math.sqrt(n / Math.max(d, 1e-300));
};

const brain = new Brain(spec);
brain.p = params;
let worst = 0, ok = true;
for (const c of cases) {
  const t0 = performance.now();
  const outputs = c.ticks.map((u) => Array.from(brain.sense(u)));
  const lesson = brain.learn(c.kinds);
  const ms = performance.now() - t0;
  const checks = {
    outputs: rel(outputs.flat(), c.outputs.flat()),
    state: rel(brain.state, c.state),
    weights: rel(brain.weights, c.lesson.weights),
    biases: rel(brain.biases, c.lesson.biases),
    energy: Math.abs(lesson.energy - c.lesson.energy) / Math.max(1e-300, Math.abs(c.lesson.energy)),
  };
  const same = lesson.updated === c.lesson.accepted && lesson.rows === c.lesson.rows;
  const bad = Object.entries(checks).filter(([, v]) => !(v < 1e-6));
  worst = Math.max(worst, ...Object.values(checks));
  ok = ok && same && bad.length === 0;
  console.log(`${c.kinds.join("+").padEnd(9)} accepted ${lesson.updated}/${c.lesson.accepted} ` +
    Object.entries(checks).map(([k, v]) => `${k} ${v.toExponential(1)}`).join("  ") + `  | ${ms.toFixed(0)} ms`);
}
console.log(ok ? `PARITY OK (worst relative difference ${worst.toExponential(2)})` : "PARITY FAILED");
process.exit(ok ? 0 : 1);
