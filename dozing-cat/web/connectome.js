// The brain as a connectome for the viewer: senses (the readings and the push), the encoded
// evidence, the belief, the record cells, the prediction, the habit's two motor units, and,
// with the governor patch on, its readback port, cortex and motor as their own regions with
// the port drawn as synapses from the prediction and the belief and the mode returning to it.
// The synapses are the patch's own weights above a small threshold, as the experiment's demo
// page draws them; the activity of every neuron each decision comes from the moment's
// computation (the readings, the push, the evidence, the repaired belief, the record code, the
// readout, the habit's push and the governor's settled state). Nothing here changes a brain.

import { ACTIONS, OUTPUTS, PAW, READINGS } from "./cat.js";
import { CORTEX, MOTOR, READBACK } from "./governor.js";

export const ROLES = { senses: "sensory", evidence: "association", belief: "association", records: "memory", prediction: "vision", habit: "motor", readback: "value", "governor cortex": "value", "governor motor": "value" };

export function regionIndex(patch, withGovernor) {
  const nS = READINGS + ACTIONS, cells = patch.records.cells;
  const ix = {};
  let at = 0;
  const add = (name, count) => { ix[name] = { start: at, stop: at + count }; at += count; };
  add("senses", nS); add("evidence", patch.encoded); add("belief", patch.belief); add("records", cells); add("prediction", OUTPUTS); add("habit", 2);
  if (withGovernor) { add("readback", READBACK.length); add("governor cortex", CORTEX); add("governor motor", MOTOR.length); }
  return { ix, n: at };
}

/** The synapse table {key: weight} of the brain as drawn: every entry is one synapse. */
export function synapseTable(patch, habit, governor) {
  const { ix } = regionIndex(patch, !!governor);
  const P = patch.P, nb = patch.belief, En = patch.encoded, In = patch.inputs;
  const table = new Map();
  const put = (p, q, w) => table.set(p * 1048576 + q, w);
  const E = P.E; // (encoded x inputs)
  for (let j = 0; j < En; j++) for (let i = 0; i < In; i++) if (Math.abs(E[j * In + i]) > 0.08) put(ix.senses.start + i, ix.evidence.start + j, E[j * In + i]);
  const F = P.F, T = P.T, za = nb + ACTIONS, fi = patch.fi;
  for (let j = 0; j < nb; j++) {
    for (let k = 0; k < ACTIONS; k++) put(ix.senses.start + READINGS + k, ix.belief.start + j, T[j * za + nb + k]);
    for (let k = 0; k < nb; k++) if (k !== j) put(ix.belief.start + k, ix.belief.start + j, T[j * za + k] + F[j * fi + k]);
    for (let i = 0; i < En; i++) put(ix.evidence.start + i, ix.belief.start + j, F[j * fi + nb + i]);
  }
  const R = patch.records, cells = R.cells, proj = R.projection; // (encoded + belief) x cells
  for (let r = 0; r < En + nb; r++) {
    const source = r < En ? ix.evidence.start + r : ix.belief.start + (r - En);
    const row = r * cells;
    for (let c = 0; c < cells; c++) if (Math.abs(proj[row + c]) > 0.18) put(source, ix.records.start + c, proj[row + c]);
  }
  // the store's read into the belief: table_y (cells x width) times the read block of F, transposed
  const W = R.width, readOffset = nb + En + nb;
  for (let c = 0; c < cells; c++) {
    let any = 0.0; for (let w = 0; w < W; w++) any += Math.abs(R.table[c * W + w]);
    if (any <= 0) continue;
    for (let j = 0; j < nb; j++) {
      let v = 0.0; for (let w = 0; w < W; w++) v += R.table[c * W + w] * F[j * fi + readOffset + w];
      if (Math.abs(v) > 1e-4) put(ix.records.start + c, ix.belief.start + j, v);
    }
  }
  const C = P.C;
  for (let o = 0; o < OUTPUTS; o++) for (let j = 0; j < nb; j++) if (Math.abs(C[o * nb + j]) > 0.02) put(ix.belief.start + j, ix.prediction.start + o, C[o * nb + j]);
  for (let k = 0; k < 2; k++) put(ix.senses.start + PAW + k, ix.habit.start + k, -habit.drift);
  if (governor) {
    const rb = ix.readback.start, gc = ix["governor cortex"].start, gm = ix["governor motor"].start;
    const local = (i) => (i < READBACK.length ? rb + i : i < READBACK.length + CORTEX ? gc + (i - READBACK.length) : gm + (i - READBACK.length - CORTEX));
    for (const [p, q, w] of governor.table) put(local(p), local(q), w);
    for (let o = 0; o < OUTPUTS; o++) { put(ix.prediction.start + o, rb + 0, 0.05); put(ix.prediction.start + o, rb + 1, 0.05); }
    for (let j = 0; j < nb; j++) put(ix.belief.start + j, rb + 2, 0.03);
    for (let k = 0; k < MOTOR.length; k++) put(gm + k, rb + 3 + k, 0.5);
  }
  return { table, ix };
}

/** The connectome the viewer lays out: pre, post, |weight| per synapse (sorted by post then
 *  pre, as the library sorts its connectomes), the region of every neuron, and the keys of the
 *  synapses so the weights can be refreshed after learning without a new layout. */
export function buildConnectome(patch, habit, governor) {
  const { table, ix } = synapseTable(patch, habit, governor);
  const { n } = regionIndex(patch, !!governor);
  const keys = Array.from(table.keys()).sort((a, b) => (a % 1048576) - (b % 1048576) || Math.floor(a / 1048576) - Math.floor(b / 1048576));
  const pre = new Uint32Array(keys.length), post = new Uint32Array(keys.length), weight = new Float32Array(keys.length);
  keys.forEach((k, e) => { pre[e] = Math.floor(k / 1048576); post[e] = k % 1048576; weight[e] = Math.abs(table.get(k)); });
  const groups = new Array(n);
  for (const name in ix) for (let i = ix[name].start; i < ix[name].stop; i++) groups[i] = name;
  return { n, pre, post, weight, keys, groups, ix };
}

/** The current |weights| of a built connectome's synapses, in its order. */
export function weightsOf(connectome, patch, habit, governor) {
  const { table } = synapseTable(patch, habit, governor);
  return Float32Array.from(connectome.keys, (k) => { const v = table.get(k); return v === undefined ? 0 : Math.abs(v); });
}

/** The activity of every neuron at one decision, in [0, 1] mostly: the readings, the push, the
 *  evidence, the belief, the record code, the readout, the habit's push, the governor. */
export function activityOf(connectome, out, r, action, e, z, codeDense, y, habitPush, gov) {
  const ix = connectome.ix;
  const set = (name, values, f = (v) => Math.abs(v)) => { const s = ix[name].start; for (let i = 0; i < values.length; i++) out[s + i] = f(values[i]); };
  set("senses", r); for (let k = 0; k < ACTIONS; k++) out[ix.senses.start + READINGS + k] = Math.abs(action[k]);
  set("evidence", e); set("belief", z);
  set("records", codeDense, (v) => (v > 0 ? 1 : 0));
  set("prediction", y, (v) => Math.min(1, Math.abs(v) / 3.0));
  set("habit", habitPush);
  if (gov && ix.readback) { set("readback", gov.readback, (v) => v); set("governor cortex", gov.cortex, (v) => v); set("governor motor", gov.motor, (v) => v); }
  return out;
}
