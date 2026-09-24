// A smoke test of the browser learner on the page's payload: decisions and rewards change the
// plastic synapses, nothing is NaN, and the cost per decision is printed.
import { readFileSync } from "node:fs";
import { FlyBrain } from "../web/brain.js";
import { FlyLearner } from "../web/learner.js";
globalThis.atob = globalThis.atob || ((b) => Buffer.from(b, "base64").toString("binary"));
const payload = JSON.parse(readFileSync(new URL("../web/data/brain.json", import.meta.url)));
const brain = new FlyBrain(payload);
const learner = new FlyLearner(brain, { outputs: ["mbon:MBON11:right", "mbon:MBON05:left"], actions: [0, 1], plastic: "kc>mbon", critic: "kc", nudgedSteps: 10, tonic: {} });
let seed = 7; learner.rng = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
console.log(`payload ${brain.n} neurons, plastic ${learner.edges.length} KC>MBON synapses, critic ${learner.criticIndex.length} KCs`);
const odours = [["orn:decaying_fruit:left", "orn:decaying_fruit:right"], ["orn:yeasty:left", "orn:yeasty:right"]];
let t0 = performance.now(), ms = 0;
for (let trial = 0; trial < 6; trial++) {
  const k = trial % 2;
  brain.clearStimuli(); for (const name of odours[k]) brain.stimulate(name, 0.6);
  for (let i = 0; i < 40; i++) brain.step();
  const a = performance.now();
  const d = learner.act(false);
  const lesson = learner.learn(k === 0 ? 1 : 0, true);
  ms += performance.now() - a;
  console.log(`trial ${trial} ${k === 0 ? "fruit" : "yeast"}: p ${d.p.map((x) => x.toFixed(3)).join("/")} choice ${d.choice} value ${d.value.toFixed(3)} -> dopamine ${lesson.delta.toFixed(3)} moved ${lesson.moved} changed ${lesson.changed} mean |change| ${lesson.meanAbsChange.toFixed(4)}`);
}
const bad = Array.from(learner.efficacy).some((x) => !Number.isFinite(x)) || Array.from(brain.s).some((x) => !Number.isFinite(x));
console.log(`${(ms / 6).toFixed(0)} ms per decision with lesson (nudged 10 steps x 2), ${((performance.now() - t0) / 6).toFixed(0)} ms per trial in all; finite: ${!bad}`);
if (bad) process.exit(1);
