// The brain in a worker: the flight sub-net settles here, off the drawing thread.
// Messages in: {type:"init", url} then {type:"run", stimuli:{channelSet: level}, steps:K}.
// Messages out: {type:"ready", n, edges, members, populations} then, per run,
// {type:"state", readouts:{group:mean}, active, s: Float32Array (transferred), steps}.
import { SettlingBrain } from "./brain.js";
import { ActorCriticLearner } from "./learner.js";
import { GROUPS } from "./motor.js";

let brain = null, readoutNames = [], learner = null, learnConfig = null, seedState = 1;
const rnd = () => { seedState = (seedState + 0x6d2b79f5) >>> 0; let t = seedState; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };

function makeLearner(keep) { // the learner on the brain's current wiring; `keep` carries efficacies across a rewire
  if (!learnConfig) return;
  learner = new ActorCriticLearner(brain, learnConfig); learner.rng = rnd;
  if (keep && keep.length === learner.efficacy.length) { learner.efficacy.set(keep); learner.applyWeights(); }
  self.postMessage({ type: "learn:ready", plastic: learner.edges.length, outputs: learner.outputs, critic: learner.criticIndex.length, ...learner.stats() });
}
function stateMessage(extra) {
  const readouts = {};
  for (const name of readoutNames) readouts[name] = brain.mean(name);
  const s = Float32Array.from(brain.s);
  self.postMessage({ type: "state", readouts, active: brain.activeCount(0.5), s, steps: brain.steps, ...extra }, [s.buffer]);
}

self.onmessage = async (e) => {
  const m = e.data;
  if (m.type === "init") {
    const payload = await (await fetch(m.url)).json();
    brain = new SettlingBrain(payload);
    readoutNames = [];
    for (const g of GROUPS) for (const side of ["left", "right"]) if (brain.sets[`${g}:${side}`]) readoutNames.push(`${g}:${side}`);
    for (const extra of ["steering:left", "steering:right", "neck_mn", "dn", "mn:haltere:left", "mn:haltere:right", "haltere:left", "haltere:right", "ocelli:left", "ocelli:right",
                         "gf", "dn:landing", "dn:grooming", "dn:DNa02:left", "dn:DNa02:right", "dn:DNp09", "dn:DNp09:left", "mn9", "kc", "mbon", "dan:pam", "dan:ppl1", "vis:LC4", "vis:LPLC2",
                         "mbon:MBON11:right", "mbon:MBON05:left", "mbon:approach", "mbon:avoid", "pn", "apl",
                         "orn:decaying_fruit:left", "orn:decaying_fruit:right", "orn:yeasty:left", "orn:yeasty:right", "orn:fruity:left", "orn:fruity:right", "grn:sugar:labellum", "grn:bitter:labellum", "leg_touch", "lptc:hs:left", "lptc:vs:left"]) if (brain.sets[extra] && !readoutNames.includes(extra)) readoutNames.push(extra);
    if (Array.isArray(m.readouts)) for (const name of m.readouts) if (brain.sets[name] && !readoutNames.includes(name)) readoutNames.push(name);
    self.postMessage({ type: "ready", n: brain.n, edges: brain.edges, synapses: payload.synapses, members: brain.members, populations: Object.keys(brain.sets), model: payload.model, whole: payload.whole, recruitment: payload.recruitment });
    return;
  }
  if (!brain) return;
  if (m.type === "reset") { brain.reset(); return; }
  if (m.type === "shuffle") { // the same neurons with the wiring shuffled: postsynaptic endpoints permuted, counts and signs kept
    if (!brain.original) brain.original = { rowPtr: brain.rowPtr, pre: brain.pre, w: brain.w };
    const keep = learner ? Float64Array.from(learner.efficacy) : null;
    if (!m.on) { brain.rowPtr = brain.original.rowPtr; brain.pre = brain.original.pre; brain.w = Float64Array.from(brain.original.w); brain.reset(); makeLearner(brain.learnedOnOriginal); return; }
    const o = brain.original, E = brain.edges, n = brain.n;
    const post = new Int32Array(E); for (let i = 0; i < n; i++) for (let e = o.rowPtr[i]; e < o.rowPtr[i + 1]; e++) post[e] = i;
    let a = (m.seed || 1) >>> 0; const rnd = () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    for (let e = E - 1; e > 0; e--) { const j = Math.floor(rnd() * (e + 1)); const t = post[e]; post[e] = post[j]; post[j] = t; }
    const count = new Int32Array(n + 1); for (let e = 0; e < E; e++) count[post[e] + 1]++;
    const rowPtr = new Int32Array(n + 1); for (let i = 0; i < n; i++) rowPtr[i + 1] = rowPtr[i] + count[i + 1];
    const fill = rowPtr.slice(0, n), pre = new Int32Array(E), w = new Float64Array(E);
    for (let e = 0; e < E; e++) { const k = fill[post[e]]++; pre[k] = o.pre[e]; w[k] = o.w[e]; }
    brain.learnedOnOriginal = keep;
    brain.rowPtr = rowPtr; brain.pre = pre; brain.w = w; brain.reset(); makeLearner(null); return;
  }
  if (m.type === "learned") { // learned efficacies on some synapses (data/learned.json); off restores the signs
    if (!brain.original) brain.original = { rowPtr: brain.rowPtr, pre: brain.pre, w: brain.w };
    const w = Float64Array.from(brain.original.w);
    brain.w = w;
    if (learner) learner.reset();
    if (m.on && m.edges) { if (learner) learner.load(m.edges, m.efficacy); else for (let k = 0; k < m.edges.length; k++) brain.setEfficacy(m.edges[k], m.efficacy[k]); }
    brain.reset(); if (learner) self.postMessage({ type: "lesson", ...learner.stats(), loaded: !!m.on }); return;
  }
  if (m.type === "learn:init") { // the actor-critic on this wiring: outputs, actions, plastic set, critic, constants
    learnConfig = m.config; seedState = (m.seed || 1) >>> 0;
    if (!brain.original) brain.original = { rowPtr: brain.rowPtr, pre: brain.pre, w: Float64Array.from(brain.w) };
    makeLearner(null); return;
  }
  if (m.type === "learn:reset") { if (learner) { learner.reset(); self.postMessage({ type: "lesson", ...learner.stats(), reset: true }); } return; }
  if (m.type === "learn:set") { if (learner) { Object.assign(learner.cfg, m.config || {}); } return; }
  if (m.type === "decide") { // settle under the senses as in run, then sample an action and keep its eligibility
    brain.clearStimuli();
    for (const [set, level] of Object.entries(m.stimuli || {})) brain.stimulate(set, level);
    const t0 = performance.now();
    for (let k = 0; k < (m.steps || 1); k++) brain.step();
    // m.u replays a uniform draw (0 forces the first action): the page blames a blow on the approach that brought the fly to the fruit
    const decision = learner ? learner.act(!!m.greedy, typeof m.u === "number" ? m.u : undefined) : null;
    if (decision && m.blame) decision.blame = true;
    stateMessage({ ms: performance.now() - t0, ran: m.steps || 1, decision, learnMs: performance.now() - t0 });
    return;
  }
  if (m.type === "reward") { // the outcome of the last decision, with the live state as the next state
    if (!learner) return;
    const t0 = performance.now();
    const lesson = learner.learn(m.reward || 0, !!m.done);
    if (lesson) self.postMessage({ type: "lesson", ...lesson, ms: performance.now() - t0, why: m.why || "" });
    else self.postMessage({ type: "lesson", ...learner.stats(), dropped: learner.dropped, why: m.why || "", lost: true });  // an outcome with no decision: the page shows it
    return;
  }
  if (m.type === "run") {
    brain.clearStimuli();
    for (const [set, level] of Object.entries(m.stimuli || {})) brain.stimulate(set, level);
    const t0 = performance.now();
    for (let k = 0; k < (m.steps || 1); k++) brain.step();
    const ms = performance.now() - t0;
    stateMessage({ ms, ran: m.steps || 1, baseline: !!m.baseline });
  }
};
