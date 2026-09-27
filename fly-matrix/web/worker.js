// The connected brain settles here in a worker, off the drawing thread.
// Current assisted path: rendered retina + other senses -> checked descending
// and motor readouts; assist:decide/selectTarget -> both settled odor scores,
// checked free/+/- phases and a sampled MBON choice; accept + matching later reward
// -> local KC->MBON learning. Tests: tests/settlement.mjs, tests/settled-learning.mjs
// and tests/assisted-worker.mjs check the residuals, local update and reward credit.
// We assist with input encoders, an action sampler and a reward/critic helper;
// the main thread supplies a stabilizing pilot and physical body mechanics.
// A small residual certifies a numerical approximate fixed point of this frozen
// input, not global convergence, biological fidelity or a complete fly mind.
// See ../docs/FLY_MATRIX_BRAIN for the measured claims and supplied assistance.
// Legacy fixed-step API: {type:"init", url} then
// {type:"run", stimuli:{channelSet: level}, steps:K}; replies are
// {type:"ready", n, edges, members, populations} then
// {type:"state", readouts:{group:mean}, active, s: Float32Array (transferred), steps}.
import { SettlingBrain } from "./brain.js";
import { ActorCriticLearner } from "./learner.js";
import { GROUPS } from "./motor.js";
import { createRetina } from "./retina.js";
import { validateTraceOptions } from "./settlement-trace.js";
import { validateReplayOptions, replayTransferables } from "./neural-replay.js";
import { TARGET_CANDIDATES, POLICY_OUTPUTS, NAVIGATION_READOUTS, attendedSenses,
  candidateScore, targetProbabilities, selectTarget } from "./neural-policy.js";

let brain = null, readoutNames = [], learner = null, learnConfig = null, seedState = 1;
// The strict runtime's declared observation ports. Named populations elsewhere in the
// payload (including motor neurons, reward cells and internal readouts) are not inputs.
export const CONTROL_SENSORY_POPULATIONS = Object.freeze([
  "haltere:left", "haltere:right", "ocelli:left", "ocelli:right", "vis:LC4", "vis:LPLC2",
  "grn:sugar:labellum", "grn:sugar:front_leg", "leg_touch",
  "lptc:hs:left", "lptc:vs:left", "lptc:hs:right", "lptc:vs:right",
  "jo:C:left", "jo:C:right", "jo:E:left", "jo:E:right",
  "orn:decaying_fruit:left", "orn:decaying_fruit:right", "orn:yeasty:left", "orn:yeasty:right",
]);
const controlSensoryNames = new Set(CONTROL_SENSORY_POPULATIONS);
let motorMembers = new Uint8Array(0);
let assistedProposal = null;
let retinaEncoder = null;
let progressPhase = null, progressAttention = null;
// Only this worker installs the observer; requests supply a boolean opt-in.
// Interim copied iterates are diagnostic, including states that later fail.
function enableProgress(message) {
  if (message.progress !== true || !["assist:observe", "assist:decide", "settle_control"].includes(message.type)) return;
  progressPhase = message.type === "assist:decide" ? "free" : "observation";
  progressAttention = message.attention ?? null;
  let lastSent = -Infinity;
  brain.onSolveProgress = sample => {
    const now = performance.now();
    if (now - lastSent < 200) return;
    lastSent = now;
    const activity = Float32Array.from(sample.activity), deltaV = Float32Array.from(sample.deltaV);
    let maxAbsDeltaV = 0, valid = Number.isFinite(sample.residual);
    for (let i = 0; i < sample.n; i++) {
      if (!Number.isFinite(activity[i]) || !Number.isFinite(deltaV[i])) valid = false;
      else maxAbsDeltaV = Math.max(maxAbsDeltaV, Math.abs(deltaV[i]));
    }
    self.postMessage({ type: "settlement_progress", requestId: message.requestId,
      generation: message.generation, searchToken: message.searchToken ?? null,
      phase: sample.phase === "free" ? progressPhase : sample.phase,
      attention: progressAttention, iteration: sample.iteration, residual: sample.residual,
      tolerance: sample.tolerance, n: sample.n, activity, deltaV,
      maxAbsDeltaV: valid ? maxAbsDeltaV : null, valid, authoritative: false },
      [activity.buffer, deltaV.buffer]);
  };
}
const assistedSensoryNames = new Set([...CONTROL_SENSORY_POPULATIONS, "dan:pam", "dan:ppl1"]);
function setAssistedInput(senses, retina = undefined, captureInput = false) {
  // INPUT "CLAMPING" = sample-and-hold drive, NOT a hard clamp on neural state.
  // stimulate adds d_i = amplitude * max(levels of named sets containing i).
  // Recurrent input still changes v_i and s_i, including in stimulated cells.
  if (!senses || typeof senses !== "object" || Array.isArray(senses)) throw new Error("invalid_senses");
  for (const [name, level] of Object.entries(senses)) {
    if (!assistedSensoryNames.has(name)) throw new Error(`undeclared_sensory_population:${name}`);
    const indices = brain.sets[name];
    if (!Array.isArray(indices) || !indices.length) throw new Error(`unknown_or_empty_sensory_population:${name}`);
    if (!Number.isFinite(level) || level < 0 || level > 1) throw new Error(`invalid_sensory_level:${name}`);
    for (const i of indices) if (!Number.isInteger(i) || i < 0 || i >= brain.n || motorMembers[i]) throw new Error(`invalid_sensory_membership:${name}`);
  }
  // Decode the actual rendered pixels before mutating any live drive. The
  // supplied mapping can stimulate only retained photoreceptors, never motors.
  if (retina !== undefined && !retinaEncoder) throw new Error("retina_unavailable");
  const retinal = retina === undefined ? null : retinaEncoder.encode(retina);
  if (retinal) for (const i of retinal.indices) {
    if (motorMembers[i]) throw new Error("retinal_motor_overlap");
  }
  brain.clearStimuli();
  for (const [name, level] of Object.entries(senses)) brain.stimulate(name, level);
  if (retinal) for (let k = 0; k < retinal.indices.length; k++) {
    const i = retinal.indices[k];
    brain.drive[i] = Math.max(brain.drive[i], brain.amplitude * retinal.levels[k]);
  }
  return retinal ? { neurons: retinal.indices.length,
    width: retina.width, height: retina.height,
    mean: retinal.levels.reduce((a, b) => a + b, 0) / retinal.levels.length,
    min: Math.min(...retinal.levels), max: Math.max(...retinal.levels),
    ...(captureInput ? { capture: { width: retina.width, height: retina.height, origin: retina.origin,
      rgba: retina.rgba.slice(), indices: retinal.indices, levels: retinal.levels, mapping: retinaEncoder.mapping } } : {}) } : null;
}

function bindTrace(solve, message, senses, retinal, source) {
  if (solve.replay) solve.replay.identity = { requestId: message.requestId, generation: message.generation, source };
  if (!solve.trace) return;
  solve.trace.identity = { requestId: message.requestId, generation: message.generation, source };
  const capture = retinal?.capture;
  solve.trace.retinal = capture ? { frame: { width: capture.width, height: capture.height, origin: capture.origin, rgba: capture.rgba },
    indices: capture.indices, levels: capture.levels, mapping: capture.mapping } : null;
  solve.trace.input = { senses: { ...senses },
    semantics: "frozen additive drive; recurrent sensory states are not hard-clamped" };
}

// A counterfactual diagnostic is isolated from control, target sampling and all
// learner eligibility/reward state. Both cases start from the identical live
// state. Only rendered RGB changes; finally restores even temporary caches and
// their original object identities/existence when validation or solving fails.
function inspectRetina(message) {
  if (!message.retina) throw new Error("missing_retinal_frame");
  const traceOptions = message.trace ?? true;
  validateTraceOptions(traceOptions, brain.n);
  const keys = ["v", "s", "drive", "total", "controlActivation", "controlDefect"];
  const saved = keys.map(key => ({ key, exists: Object.hasOwn(brain, key), original: brain[key], values: brain[key]?.slice() }));
  const steps = brain.steps;
  const restore = () => {
    for (const { key, exists, original, values } of saved) {
      if (exists) { brain[key] = original; if (values) original.set(values); }
      else delete brain[key];
    }
    brain.steps = steps;
  };
  let response, failure;
  try {
    const senses = message.attention == null ? message.senses : attendedSenses(message.senses, message.attention);
    const run = (retina, source) => {
      const retinal = setAssistedInput(senses, retina, true);
      const solve = brain.settleControl(message.steps ?? 1024, message.tolerance ?? 1e-6, traceOptions);
      bindTrace(solve, message, senses, retinal, source);
      const { trace, ...diagnostic } = solve;
      return { solve: diagnostic, trace, retinal, readouts: Object.fromEntries(readoutNames.map(name => [name, brain.mean(name)])) };
    };
    const actual = run(message.retina, "retina-diagnostic-actual"), actualActivity = brain.s.slice();
    restore();
    const blackFrame = { ...message.retina, rgba: message.retina.rgba.slice() };
    for (let k = 0; k < blackFrame.rgba.length; k += 4) blackFrame.rgba[k] = blackFrame.rgba[k + 1] = blackFrame.rgba[k + 2] = 0;
    const black = run(blackFrame, "retina-diagnostic-black");
    const receptors = new Set(brain.sets.photoreceptor), nonreceptor = { count: 0, changedCount: 0, maxAbsDelta: 0, maxIndex: null };
    for (let i = 0; i < brain.n; i++) if (!receptors.has(i)) {
      nonreceptor.count++;
      const delta = Math.abs(actualActivity[i] - brain.s[i]);
      if (delta !== 0) nonreceptor.changedCount++;
      if (!Number.isFinite(delta)) { nonreceptor.maxAbsDelta = null; nonreceptor.maxIndex = i; }
      else if (nonreceptor.maxAbsDelta !== null && delta > nonreceptor.maxAbsDelta) { nonreceptor.maxAbsDelta = delta; nonreceptor.maxIndex = i; }
    }
    response = { type: "retina_diagnostic", kind: "diagnostic", diagnosticOnly: true,
      requestId: message.requestId, generation: message.generation,
      comparable: actual.solve.converged && black.solve.converged, actual, black,
      deltaReadouts: Object.fromEntries(readoutNames.map(name => [name, actual.readouts[name] - black.readouts[name]])),
      nonreceptor, meaning: "same initial state, nonvisual drives and weights; actual RGB minus black RGB; no control or learning credit" };
  } catch (error) { failure = error; }
  finally { restore(); }
  if (failure) return { type: "retina_diagnostic", kind: "diagnostic", diagnosticOnly: true,
    requestId: message.requestId, generation: message.generation, comparable: false, stateRestored: true,
    reason: failure instanceof Error ? failure.message : String(failure) };
  return { ...response, stateRestored: true };
}

// Each candidate uses the SAME initial neural state and frozen retinal/body
// snapshot. Only the attended odor differs. The network supplies the two scores;
// the attention protocol and sampling decoder are supplied, not anatomy-derived.
// No fruit coordinate, distance, sugar location or reward enters this selection.
function chooseNeuralTarget(m) {
  const v = brain.v.slice(), s = brain.s.slice(), steps = brain.steps;
  const candidates = [], states = [], temperature = learner.cfg.temperature;
  try {
    for (const { fruit } of TARGET_CANDIDATES) {
      brain.v.set(v); brain.s.set(s); brain.steps = steps;
      setAssistedInput(attendedSenses(m.senses, fruit), m.retina);
      progressPhase = `target:${fruit}`; progressAttention = fruit;
      const solve = brain.settleControl(m.steps ?? 1024, m.tolerance ?? 1e-6);
      const readouts = Object.fromEntries(POLICY_OUTPUTS.map(name => [name, brain.mean(name)]));
      candidates.push({ fruit, solve, readouts, score: candidateScore(readouts) });
      states.push({ v: brain.v.slice(), s: brain.s.slice(), steps: brain.steps });
    }
  } finally {
    brain.v.set(v); brain.s.set(s); brain.steps = steps;
    setAssistedInput(m.senses, m.retina);
    progressPhase = "free"; progressAttention = m.attention ?? null;
  }
  if (candidates.some(c => !c.solve.converged)) return { selection: { accepted: false, candidates, reason: "target_phase_failed" } };
  const p = targetProbabilities(candidates, temperature), draw = rnd();
  const fruit = selectTarget(candidates, draw, temperature);
  return { selection: { accepted: true, candidates, p, draw, fruit, temperature },
    state: states[TARGET_CANDIDATES.findIndex(c => c.fruit === fruit)] };
}
function sameProposal(m) {
  return assistedProposal && ["requestId", "generation", "searchToken"].every(k => m[k] === assistedProposal[k]);
}
function cancelProposal() {
  if (assistedProposal && learner) {
    for (const key of ["trace", "traceBias", "traceCritic"]) learner[key].set(assistedProposal[key]);
    learner.pending = null;
  }
  assistedProposal = null;
}
function annotatedMotorMembers(b) {
  const members = new Uint8Array(b.n), families = ["power", "steering", "tension", "neck_mn", "leg_mn"];
  for (const [name, indices] of Object.entries(b.sets)) {
    if (name !== "motor" && name !== "mn9" && !name.startsWith("mn:") && !families.some(family => name === family || name.startsWith(`${family}:`))) continue;
    for (const index of indices) {
      if (!Number.isInteger(index) || index < 0 || index >= b.n) throw new Error(`invalid_motor_population:${name}`);
      members[index] = 1;
    }
  }
  return members;
}
const rnd = () => { seedState = (seedState + 0x6d2b79f5) >>> 0; let t = seedState; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };

const EDGE_ARRAYS = ["pre", "w", "count", "sign", "gainPre", "efficacy0"];
function rememberOriginal() {
  if (!brain.original) brain.original = { rowPtr: brain.rowPtr.slice(),
    ...Object.fromEntries(EDGE_ARRAYS.map(key => [key, brain[key]?.slice() ?? null])) };
}
// Full-edge efficacy values, keyed by payload edge identity rather than a learner's
// plastic-subset position. A population-selected plastic set can change after rewiring.
function originalEfficacies() {
  const values = new Float64Array(brain.edges);
  for (let e = 0; e < brain.edges; e++) {
    const factor = brain.gainPre ? brain.gainPre[e] : brain.gain * (brain.count ? brain.count[e] : 1);
    const baseline = brain.efficacy0 ? brain.efficacy0[e] : brain.sign ? brain.sign[e] : 0;
    values[brain.edgeIdentity ? brain.edgeIdentity[e] : e] = factor === 0 || factor * baseline === brain.w[e]
      ? baseline : brain.w[e] / factor;
  }
  if (learner) for (let k = 0; k < learner.edges.length; k++) {
    const e = learner.edges[k]; values[brain.edgeIdentity ? brain.edgeIdentity[e] : e] = learner.efficacy[k];
  }
  return values;
}
function currentEdgeIndices(edges) {
  if (!edges || typeof edges[Symbol.iterator] !== "function") throw new Error("invalid_edge_ids");
  for (const e of edges) if (!Number.isSafeInteger(e) || e < 0 || e >= brain.edges) throw new Error(`invalid_edge_id:${String(e)}`);
  return brain.edgePosition ? Array.from(edges, e => brain.edgePosition[e]) : edges;
}
function makeLearner(keep) { // `keep` is a full efficacy vector in original payload edge order
  if (!learnConfig) return;
  const config = Array.isArray(learnConfig.plastic) ? { ...learnConfig, plastic: currentEdgeIndices(learnConfig.plastic) } : learnConfig;
  learner = new ActorCriticLearner(brain, config); learner.rng = rnd;
  // The matching weights were already copied/remapped; do not round-trip them through
  // division and multiplication. Keep reset baselines separate in sign/efficacy0.
  if (keep) for (let k = 0; k < learner.edges.length; k++) {
    const e = learner.edges[k]; learner.efficacy[k] = keep[brain.edgeIdentity ? brain.edgeIdentity[e] : e];
  }
  self.postMessage({ type: "learn:ready", plastic: learner.edges.length, outputs: learner.outputs, critic: learner.criticIndex.length, ...learner.stats() });
}
function stateMessage(extra) {
  // Physical/decision readouts use Float64 population means. Float32 s below is
  // a separate display copy; the atlas does not compute or select an action.
  const readouts = {};
  for (const name of readoutNames) readouts[name] = brain.mean(name);
  const s = Float32Array.from(brain.s);
  self.postMessage({ type: "state", readouts, active: brain.activeCount(0.5), s, steps: brain.steps, ...extra },
    [s.buffer, ...replayTransferables(extra.replay)]);
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
  if (["init", "reset", "learn:init", "learn:reset", "learned", "shuffle"].includes(m.type)) cancelProposal();
  if (m.type === "init") {
    const payload = await (await fetch(m.url)).json();
    const candidate = new SettlingBrain(payload), candidateMotorMembers = annotatedMotorMembers(candidate);
    brain = candidate;
    retinaEncoder = brain.sets.photoreceptor?.length
      ? createRetina({ n: brain.n, photoreceptors: brain.sets.photoreceptor }) : null;
    learner = null; learnConfig = null;
    motorMembers = candidateMotorMembers;
    readoutNames = [];
    for (const g of GROUPS) for (const side of ["left", "right"]) if (brain.sets[`${g}:${side}`]) readoutNames.push(`${g}:${side}`);
    for (const extra of ["steering:left", "steering:right", "neck_mn", "dn", "mn:haltere:left", "mn:haltere:right", "haltere:left", "haltere:right", "ocelli:left", "ocelli:right",
                         "gf", "dn:landing", "dn:grooming", "dn:DNa02:left", "dn:DNa02:right", "dn:DNp09", "dn:DNp09:left", "mn9", "mn:ttm:left", "mn:ttm:right", "kc", "mbon", "dan:pam", "dan:ppl1", "vis:LC4", "vis:LPLC2",
                         "mbon:MBON11:right", "mbon:MBON05:left", "mbon:approach", "mbon:avoid", "pn", "apl",
                         "orn:decaying_fruit:left", "orn:decaying_fruit:right", "orn:yeasty:left", "orn:yeasty:right", "orn:fruity:left", "orn:fruity:right", "grn:sugar:labellum", "grn:bitter:labellum", "leg_touch", "lptc:hs:left", "lptc:vs:left"]) if (brain.sets[extra] && !readoutNames.includes(extra)) readoutNames.push(extra);
    if (Array.isArray(m.readouts)) for (const name of m.readouts) if (brain.sets[name] && !readoutNames.includes(name)) readoutNames.push(name);
    for (const name of NAVIGATION_READOUTS) if (brain.sets[name] && !readoutNames.includes(name)) readoutNames.push(name);
    self.postMessage({ type: "ready", requestId: m.requestId, generation: m.generation, n: brain.n, edges: brain.edges, synapses: payload.synapses, members: brain.members, populations: Object.keys(brain.sets), model: payload.model, whole: payload.whole, recruitment: payload.recruitment });
    return;
  }
  if (!brain) {
    if (m.type === "settle_control" || m.type === "inspect:retina" || m.type.startsWith("assist:")) throw new Error("brain_not_initialized");
    return;
  }
  enableProgress(m);
  if (m.type === "inspect:retina") { self.postMessage(inspectRetina(m)); return; }
  if (m.type === "assist:cancel") { if (sameProposal(m)) cancelProposal(); return; }
  if (m.type === "assist:accept") { if (sameProposal(m)) assistedProposal.accepted = true; return; }
  if (m.type === "assist:observe" || m.type === "assist:decide") {
    const senses = m.attention == null ? m.senses : attendedSenses(m.senses, m.attention);
    const traceOptions = m.type === "assist:observe" ? m.trace ?? null : null;
    const replayOptions = m.type === "assist:observe" ? m.replay ?? null : null;
    if (traceOptions !== null && traceOptions !== false) validateTraceOptions(traceOptions, brain.n);
    if (replayOptions !== null && replayOptions !== false) validateReplayOptions(replayOptions, brain.n);
    const retinal = setAssistedInput(senses, m.retina, traceOptions !== null && traceOptions !== false);
    const t0 = performance.now();
    if (m.type === "assist:observe") {
      // A capped numerical attempt is diagnostic, not the next admitted neural
      // state. Retain the previous state so a transient contact input cannot
      // seed every later observation with its unresolved iterates. Inputs,
      // weights, tolerance and failure reports are unchanged; no retry is
      // silently reported as success.
      const previousV = brain.v.slice(), previousS = brain.s.slice();
      const solve = brain.settleControl(m.steps ?? 1024, m.tolerance ?? 1e-6, traceOptions, replayOptions);
      bindTrace(solve, m, senses, retinal, "control");
      if (!solve.converged) { brain.v.set(previousV); brain.s.set(previousS); }
      const reply = { type: "control", kind: "control", assisted: true, requestId: m.requestId, generation: m.generation,
        attention: m.attention ?? null, searchToken: m.searchToken ?? null, retinal, ...solve,
        stateRestored: !solve.converged, ms: performance.now() - t0 };
      if (solve.converged) stateMessage(reply); else self.postMessage(reply, replayTransferables(reply.replay));
    } else {
      // The free equilibrium supplies the action probabilities. The two softly
      // nudged equilibria supply local eligibility, never the displayed action.
      if (!learner) throw new Error("learner_not_initialized");
      if (!Number.isSafeInteger(m.requestId) || !Number.isSafeInteger(m.generation) || typeof m.searchToken !== "string" || !m.searchToken) throw new Error("invalid_decision_identity");
      cancelProposal();
      const target = m.selectTarget === true ? chooseNeuralTarget(m) : null;
      const targetSelection = target?.selection ?? null;
      if (targetSelection && !targetSelection.accepted) {
        self.postMessage({ type: "assisted_decision", kind: "decision", requestId: m.requestId, generation: m.generation,
          searchToken: m.searchToken, accepted: false, converged: false, reason: targetSelection.reason, targetSelection,
          ms: performance.now() - t0 }); return;
      }
      const previous = target ? { v: brain.v.slice(), s: brain.s.slice(), steps: brain.steps } : null;
      if (targetSelection) {
        setAssistedInput(attendedSenses(m.senses, targetSelection.fruit), m.retina);
        progressAttention = targetSelection.fruit;
        // This exact input has already settled in the selected target probe.
        // Reuse its state, then independently check the free residual again in
        // actSettled. Both nudged phases still run; no tolerance is relaxed.
        // Internal states never enter the worker reply or its evidence packet.
        brain.v.set(target.state.v); brain.s.set(target.state.s); brain.steps = target.state.steps;
      }
      const saved = Object.fromEntries(["trace", "traceBias", "traceCritic"].map(k => [k, learner[k].slice()]));
      const decision = learner.actSettled({ maxSteps: m.steps ?? 1024, tolerance: m.tolerance ?? 1e-6 });
      if (!decision.accepted && previous) {
        brain.v.set(previous.v); brain.s.set(previous.s); brain.steps = previous.steps;
      }
      if (decision.accepted) assistedProposal = { ...saved, requestId: m.requestId, generation: m.generation, searchToken: m.searchToken, accepted: false };
      const reply = { type: "assisted_decision", kind: "decision", requestId: m.requestId, generation: m.generation, searchToken: m.searchToken,
        ...decision.solves?.free, accepted: decision.accepted, reason: decision.reason, decision, targetSelection, retinal, ms: performance.now() - t0 };
      if (decision.accepted) stateMessage(reply); else self.postMessage(reply);
    }
    return;
  }
  if (m.type === "assist:reward") {
    // The network stores learned memory on its KC->MBON synapses after an
    // admitted choice is used and its matching outcome arrives. We assist
    // with the critic that converts that outcome's reward to delta.
    if (!learner || !sameProposal(m) || !assistedProposal.accepted || m.neuralCredit !== true) {
      self.postMessage({ type: "lesson", accepted: false, lost: true, reason: "no_matching_executed_decision", generation: m.generation }); return;
    }
    // Terminal search outcomes need no fabricated next state. Nonterminal callers
    // must supply the real next observation and qualify its equilibrium as well.
    if (typeof m.done !== "boolean") throw new Error("invalid_terminal_flag");
    if (!m.done) setAssistedInput(m.senses, m.retina);
    const result = learner.learnSettled(m.reward, m.done, { maxSteps: m.steps ?? 1024, tolerance: m.tolerance ?? 1e-6 });
    if (!result.accepted) cancelProposal(); else assistedProposal = null;
    self.postMessage({ type: "lesson", ...learner.stats(), ...result.lesson, accepted: result.accepted, reason: result.reason,
      lost: !result.accepted, why: m.why || "", generation: m.generation, requestId: m.requestId, searchToken: m.searchToken });
    return;
  }
  if (m.type === "reset") { brain.reset(); self.postMessage({ type: "reset", requestId: m.requestId, generation: m.generation }); return; }
  if (m.type === "settle_control") {
    const senses = m.senses === undefined ? {} : m.senses;
    if (!senses || typeof senses !== "object" || Array.isArray(senses)) throw new Error("invalid_senses");
    // Validate the complete frozen input before changing the live drive. Missing populations
    // are errors: silently dropping a sensory channel would change the declared controller.
    for (const [name, level] of Object.entries(senses)) {
      if (!controlSensoryNames.has(name)) throw new Error(`undeclared_sensory_population:${name}`);
      if (!Object.prototype.hasOwnProperty.call(brain.sets, name) || !Array.isArray(brain.sets[name]) || !brain.sets[name].length) throw new Error(`unknown_or_empty_sensory_population:${name}`);
      if (!Number.isFinite(level) || level < 0 || level > 1) throw new Error(`invalid_sensory_level:${name}`);
      for (const index of brain.sets[name]) {
        if (!Number.isInteger(index) || index < 0 || index >= brain.n) throw new Error(`invalid_sensory_population:${name}`);
        if (motorMembers[index]) throw new Error(`sensory_motor_overlap:${name}`);
      }
    }
    const traceOptions = m.trace ?? null;
    const replayOptions = m.replay ?? null;
    if (traceOptions !== null && traceOptions !== false) validateTraceOptions(traceOptions, brain.n);
    if (replayOptions !== null && replayOptions !== false) validateReplayOptions(replayOptions, brain.n);
    brain.clearStimuli();
    for (const [name, level] of Object.entries(senses)) brain.stimulate(name, level);
    const t0 = performance.now();
    const diagnostic = brain.settleControl(m.steps === undefined ? 256 : m.steps, m.tolerance === undefined ? 1e-6 : m.tolerance, traceOptions, replayOptions);
    bindTrace(diagnostic, m, senses, null, "control");
    const reply = { type: "control", kind: "control", requestId: m.requestId, generation: m.generation,
      ...diagnostic, ms: performance.now() - t0 };
    if (diagnostic.converged) stateMessage(reply);
    else self.postMessage(reply, replayTransferables(reply.replay)); // failure has diagnostics, never an authoritative readout
    return;
  }
  if (m.type === "shuffle") { // the same neurons with the wiring shuffled: postsynaptic endpoints permuted, counts and signs kept
    rememberOriginal();
    // Repeated control shuffles always start from the saved original graph. Learning
    // within a shuffled control is discarded on unshuffle, never copied to other edges.
    if (!brain.edgeIdentity) brain.learnedOnOriginal = { w: brain.w.slice(), efficacy: originalEfficacies() };
    if (!m.on) {
      brain.rowPtr = brain.original.rowPtr.slice();
      for (const key of EDGE_ARRAYS) brain[key] = brain.original[key]?.slice() ?? null;
      brain.w = brain.learnedOnOriginal.w.slice();
      brain.edgeIdentity = null; brain.edgePosition = null;
      brain.reset(); makeLearner(brain.learnedOnOriginal.efficacy); return;
    }
    const o = brain.original, E = brain.edges, n = brain.n;
    const post = new Int32Array(E); for (let i = 0; i < n; i++) for (let e = o.rowPtr[i]; e < o.rowPtr[i + 1]; e++) post[e] = i;
    let a = (m.seed || 1) >>> 0; const rnd = () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    for (let e = E - 1; e > 0; e--) { const j = Math.floor(rnd() * (e + 1)); const t = post[e]; post[e] = post[j]; post[j] = t; }
    const count = new Int32Array(n + 1); for (let e = 0; e < E; e++) count[post[e] + 1]++;
    const rowPtr = new Int32Array(n + 1); for (let i = 0; i < n; i++) rowPtr[i + 1] = rowPtr[i] + count[i + 1];
    const fill = rowPtr.slice(0, n), edgeIdentity = new Int32Array(E), edgePosition = new Int32Array(E);
    for (let e = 0; e < E; e++) { const k = fill[post[e]]++; edgeIdentity[k] = e; edgePosition[e] = k; }
    for (const key of EDGE_ARRAYS) {
      const source = key === "w" ? brain.learnedOnOriginal.w : o[key];
      brain[key] = source ? source.constructor.from(edgeIdentity, e => source[e]) : null;
    }
    brain.rowPtr = rowPtr; brain.edgeIdentity = edgeIdentity; brain.edgePosition = edgePosition;
    brain.reset(); makeLearner(brain.learnedOnOriginal.efficacy); return;
  }
  if (m.type === "learned") { // learned efficacies on some synapses (data/learned.json); off restores the signs
    const edges = m.on && m.edges ? currentEdgeIndices(m.edges) : null; // reject invalid payload IDs before changing weights
    rememberOriginal();
    const w = brain.edgeIdentity ? Float64Array.from(brain.edgeIdentity, e => brain.original.w[e]) : brain.original.w.slice();
    brain.w = w;
    if (learner) learner.reset();
    if (edges) {
      if (learner) learner.load(edges, m.efficacy); else for (let k = 0; k < edges.length; k++) brain.setEfficacy(edges[k], m.efficacy[k]);
    }
    brain.reset(); if (learner) self.postMessage({ type: "lesson", ...learner.stats(), loaded: !!m.on }); return;
  }
  if (m.type === "learn:init") { // the actor-critic on this wiring: outputs, actions, plastic set, critic, constants
    if (Array.isArray(m.config?.plastic)) currentEdgeIndices(m.config.plastic);
    learnConfig = m.config; seedState = (m.seed || 1) >>> 0;
    rememberOriginal();
    makeLearner(originalEfficacies()); return;
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
  } catch (error) {
    const control = m && ["settle_control", "assist:observe"].includes(m.type), decision = m?.type === "assist:decide";
    if (m?.type === "inspect:retina") {
      self.postMessage({ type: "retina_diagnostic", kind: "diagnostic", diagnosticOnly: true,
        requestId: m.requestId, generation: m.generation, comparable: false,
        reason: error instanceof Error ? error.message : String(error) }); return;
    }
    if (decision || m?.type === "assist:reward") cancelProposal();
    self.postMessage({ type: control ? "control" : decision ? "assisted_decision" : "error", kind: control ? "control" : decision ? "decision" : "error",
      requestId: m && m.requestId, generation: m && m.generation,
      searchToken: m && m.searchToken, accepted: false,
      converged: false, iterations: 0, residual: null,
      tolerance: m && Number.isFinite(m.tolerance) ? m.tolerance : null,
      reason: error instanceof Error ? error.message : String(error) });
  } finally {
    if (brain) brain.onSolveProgress = null;
    progressPhase = null; progressAttention = null;
  }
};
