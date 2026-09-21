// Parity: the browser brain against cadence's TemporalPatchNet on the same paths.
import { readFileSync } from "node:fs";
import { Brain } from "../web/brain.js";

const dir = new URL(".", import.meta.url).pathname;
const spec = JSON.parse(readFileSync(dir + "../web/data/brain.json"));
const cases = JSON.parse(readFileSync(dir + "parity_cases.json"));
const params = JSON.parse(readFileSync(dir + "../web/data/params.json"));
const rel = (a, b) => { let n = 0, d = 0; for (let k = 0; k < a.length; k++) { n += (a[k] - b[k]) ** 2; d += b[k] ** 2; } return Math.sqrt(n / Math.max(d, 1e-300)); };
const flat = (m) => Float64Array.from(m.flat());
let worst = 0, ok = true;
for (const c of cases) {
  const brain = new Brain(spec);
  if (c.warmup.length) brain.advance(c.warmup.map((r) => Float64Array.from(r)));
  const U = c.inputs.map((r) => Float64Array.from(r));
  const t0 = performance.now();
  const r = brain.observe(U, flat(c.target), params.beta, params.rate);
  const ms = performance.now() - t0;
  const checks = {
    free_output: rel(r.free.output, flat(c.free_output)),
    plus_hidden: rel(r.plus.hidden, flat(c.plus_hidden)),
    minus_hidden: rel(r.minus.hidden, flat(c.minus_hidden)),
    delta_A: rel(r.delta.A, Float64Array.from(c.delta_A)),
    after_A: rel(brain.parameters().A, Float64Array.from(c.after_A)),
    growth: Math.abs(brain.growth() - c.growth_after) / c.growth_after,
  };
  const same = r.updated === c.updated && (!c.updated || Math.abs(r.step - c.step) <= 1e-12 * c.step);
  const bad = Object.entries(checks).filter(([, v]) => !(v < 1e-6));
  worst = Math.max(worst, ...Object.values(checks));
  ok = ok && same && bad.length === 0;
  console.log(`T=${String(c.T).padStart(2)} updated ${r.updated}/${c.updated} step ${r.step ?? 0}/${c.step} ` +
    Object.entries(checks).map(([k, v]) => `${k} ${v.toExponential(1)}`).join("  ") + `  | ${ms.toFixed(0)} ms`);
}
console.log(ok ? `PARITY OK (worst relative difference ${worst.toExponential(2)})` : "PARITY FAILED");
process.exit(ok ? 0 : 1);
