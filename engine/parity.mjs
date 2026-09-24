// The browser engine against the library: settle cases (per-step population means) and a
// recorded run of the actor-critic (actions, dopamine, the plastic efficacies at the end).
//   node engine/parity.mjs <brain.json> [cases.json] [lessons.json]
import { readFileSync } from "node:fs";
import { SettlingBrain } from "./brain.js";
import { ActorCriticLearner } from "./learner.js";

globalThis.atob ??= (b) => Buffer.from(b, "base64").toString("binary");
const [payloadPath, casesPath, lessonsPath] = process.argv.slice(2);
if (!payloadPath) { console.error("usage: node engine/parity.mjs <brain.json> [cases.json] [lessons.json]"); process.exit(2); }
const payload = JSON.parse(readFileSync(payloadPath, "utf8"));
const TOL = 1e-9;
let ok = true;

if (casesPath) {
  const cases = JSON.parse(readFileSync(casesPath, "utf8"));
  const brain = new SettlingBrain(payload);
  let worst = 0;
  for (const c of cases.cases) {
    brain.reset(); brain.clearStimuli();
    for (const [pop, level] of Object.entries(c.stimulus)) brain.stimulate(pop, level);
    for (let t = 0; t < c.per_step_means.length; t++) {
      brain.step();
      c.readouts.forEach((r, k) => { const d = Math.abs(brain.mean(r) - c.per_step_means[t][k]); if (d > worst) worst = d; });
    }
    const active = brain.activeCount(0.5);
    if (active !== c.final_active) { console.error(`${c.name}: active count ${active} differs from the library's ${c.final_active}`); ok = false; }
  }
  console.log(`settle: ${cases.cases.length} cases, worst deviation ${worst.toExponential(2)}`);
  if (!(worst < TOL)) ok = false;
}

if (lessonsPath) {
  const rec = JSON.parse(readFileSync(lessonsPath, "utf8"));
  const brain = new SettlingBrain(payload);
  const cfg = rec.config;
  const learner = new ActorCriticLearner(brain, { outputs: cfg.outputs, plastic: cfg.plastic, critic: cfg.critic, plasticNeurons: cfg.plasticNeurons, beta: cfg.beta, temperature: cfg.temperature, nudgedSteps: cfg.nudgedSteps, tolerance: cfg.tolerance, gamma: cfg.gamma, lam: cfg.lam, eta: cfg.eta, etaBias: cfg.etaBias, etaCritic: cfg.etaCritic, cap: cfg.cap, dopamineCap: cfg.dopamineCap });
  const setDrive = (pairs) => { brain.clearStimuli(); for (const [i, level] of pairs) brain.setDrive(i, level); };
  let worstDelta = 0, mismatches = 0;
  brain.reset();
  rec.decisions.forEach((d, t) => {
    setDrive(d.drive);
    if (t === 0) brain.settleFree(cfg.freeSteps, cfg.tolerance);            // the first free phase from rest; later ones are the settled next states
    const out = learner.act(false, d.u);
    if (out.action !== d.action) mismatches++;
    const next = t + 1 < rec.decisions.length ? rec.decisions[t + 1].drive : rec.next_drive;
    if (d.done) brain.reset();                                            // a finished stream starts its next life from rest
    setDrive(next); brain.settleFree(cfg.freeSteps, cfg.tolerance);
    const lesson = learner.learn(d.reward, d.done);
    brain.settleFree(cfg.freeSteps, cfg.tolerance);                      // the library's act refreshes the free state under the updated weights
    worstDelta = Math.max(worstDelta, Math.abs(lesson.delta - d.dopamine));
  });
  let worstEff = 0; rec.efficacy.forEach((e, k) => { worstEff = Math.max(worstEff, Math.abs(learner.efficacy[k] - e)); });
  let worstCritic = Math.abs(learner.bCritic - rec.b_critic); rec.w_critic.forEach((w, k) => { worstCritic = Math.max(worstCritic, Math.abs(learner.wCritic[k] - w)); });
  let worstBias = 0; (rec.bias || []).forEach((b, k) => { worstBias = Math.max(worstBias, Math.abs(brain.bias[cfg.plasticNeurons[k]] - b)); });
  console.log(`lessons: ${rec.decisions.length} decisions, ${mismatches} action mismatches, dopamine within ${worstDelta.toExponential(2)}, efficacies within ${worstEff.toExponential(2)}, biases within ${worstBias.toExponential(2)}, critic within ${worstCritic.toExponential(2)}`);
  if (mismatches || !(worstDelta < TOL) || !(worstEff < TOL) || !(worstBias < TOL) || !(worstCritic < TOL)) ok = false;
}

if (!ok) { console.error("parity failed"); process.exit(1); }
console.log(`parity ok: ${payload.n} neurons, ${payload.edges} synapse classes`);
