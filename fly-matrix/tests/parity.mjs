// The browser brain against the library: replay the parity cases through web/brain.js.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { SettlingBrain } from "../web/brain.js";

const here = dirname(fileURLToPath(import.meta.url));
globalThis.atob ??= (b) => Buffer.from(b, "base64").toString("binary");
const payload = JSON.parse(readFileSync(join(here, "..", "web", "data", "brain.json"), "utf8"));
const cases = JSON.parse(readFileSync(join(here, "parity_cases.json"), "utf8"));
const brain = new SettlingBrain(payload);
let worst = 0;
for (const c of cases.cases) {
  brain.reset(); brain.clearStimuli();   // the state to rest, and the senses of the previous case off
  for (const [pop, level] of Object.entries(c.stimulus)) brain.stimulate(pop, level);
  for (let t = 0; t < c.per_step_means.length; t++) {
    brain.step();
    c.readouts.forEach((r, k) => { const d = Math.abs(brain.mean(r) - c.per_step_means[t][k]); if (d > worst) worst = d; });
  }
  const active = brain.activeCount(0.5);
  if (active !== c.final_active) { console.error(`${c.name}: active count ${active} differs from the library's ${c.final_active}`); process.exit(1); }
  console.log(`${c.name}: ${c.per_step_means.length} steps, ${active} active, worst deviation so far ${worst.toExponential(2)}`);
}
if (!(worst < 1e-9)) { console.error(`parity failed: worst deviation ${worst}`); process.exit(1); }
console.log(`parity ok: ${brain.n} neurons, ${brain.edges} synapse classes, worst deviation ${worst.toExponential(2)}`);
