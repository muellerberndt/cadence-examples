// Observe actual local-equation updates without changing the solver. A patch
// here is one receiving neuron's equation and its incoming CSR neighborhood.
// Neighborhoods overlap when they read the same presynaptic state; this engine
// does not maintain independent copies or run a separate overlap-consensus rule.
export const TRACE_MAX_PATCHES = 64, TRACE_MAX_EDGES = 256, TRACE_MAX_FRAMES = 32;
const finite = x => Number.isFinite(x) ? x : null;

export function validateTraceOptions(options, n) {
  if (options === true) options = {};
  if (!options || typeof options !== "object" || Array.isArray(options)) throw new TypeError("invalid_trace_options");
  const limits = { maxPatches: TRACE_MAX_PATCHES, maxEdges: TRACE_MAX_EDGES, maxFrames: TRACE_MAX_FRAMES };
  for (const [key, limit] of Object.entries(limits)) {
    const value = options[key] ?? limit;
    if (!Number.isSafeInteger(value) || value < (key === "maxFrames" ? 2 : 1) || value > limit) throw new RangeError(`invalid_trace_${key}`);
    limits[key] = value;
  }
  if (options.focus !== undefined && (!(Array.isArray(options.focus) || ArrayBuffer.isView(options.focus))
    || !Number.isSafeInteger(options.focus.length) || options.focus.length < 1 || options.focus.length > limits.maxPatches))
    throw new RangeError("invalid_trace_focus");
  const focus = options.focus === undefined ? null : Array.from(options.focus);
  if (focus && (!focus.length || focus.length > limits.maxPatches || new Set(focus).size !== focus.length
    || !focus.every(i => Number.isSafeInteger(i) && i >= 0 && i < n))) throw new RangeError("invalid_trace_focus");
  return { ...limits, focus };
}

function chooseFocus(brain, maximum) {
  if (brain.n <= maximum) return Array.from({ length: brain.n }, (_, i) => i);
  const ids = new Set(), add = id => { if (ids.size < maximum && Number.isInteger(id) && id >= 0 && id < brain.n) ids.add(id); };
  const sample = (name, count) => {
    const members = brain.sets[name] || [];
    for (let k = 0; k < Math.min(count, members.length); k++) add(members[Math.floor(k * members.length / Math.min(count, members.length))]);
  };
  sample("photoreceptor", 6);
  const receptors = new Set(ids);
  // Reserve outputs before filling graph neighbors: a small display budget
  // must not systematically truncate the motor/descending end of the graph.
  for (const name of ["mbon:MBON11:right", "mbon:MBON05:left", "dn:DNa02:left", "dn:DNa02:right", "dn:DNp09",
    "dn:DNp03", "dn:landing", "power:left", "power:right", "steering:left", "steering:right",
    "lptc:hs:left", "lptc:hs:right", "vis:LC4", "pn", "kc"]) sample(name, 1);
  const before = ids.size;
  for (let post = 0; post < brain.n && ids.size < Math.min(maximum, before + 12); post++) {
    for (let e = brain.rowPtr[post]; e < brain.rowPtr[post + 1]; e++) {
      if (brain.w[e] !== 0 && receptors.has(brain.pre[e])) { add(post); break; }
    }
  }
  for (const name of ["photoreceptor", "mbon", "dn", "motor"]) sample(name, name === "photoreceptor" ? 8 : 2);
  for (let k = 0; k < maximum; k++) add(Math.floor(k * brain.n / maximum));
  return [...ids];
}

/** The solver calls capture after each real residual calculation. Unsaved
 * iterations still retain the immediately preceding v/defect for honest last-
 * update measurements. No full-network state history is stored. */
export function createSettlementTrace(brain, options, maxSteps, tolerance) {
  const limits = validateTraceOptions(options, brain.n);
  const focus = limits.focus || chooseFocus(brain, limits.maxPatches), focusSet = new Set(focus);
  const labels = new Map(focus.map(id => [id, []]));
  for (const [name, members] of Object.entries(brain.sets)) for (const id of members) {
    const names = labels.get(id); if (names) names.push(name);
  }
  const labelPriority = name => name === "photoreceptor" || name.includes(":") ? 0 : 1;
  for (const [id, names] of labels) labels.set(id, names.sort((a, b) => labelPriority(a) - labelPriority(b) || a.localeCompare(b)).slice(0, 8));
  const queues = focus.map(post => {
    const chosen = [];
    // Prefer displayed shared states, then retain CSR order for the remaining
    // ports. All omitted edges still contribute to the full synapticInput.
    for (const inFocus of [true, false]) for (let e = brain.rowPtr[post]; e < brain.rowPtr[post + 1] && chosen.length < limits.maxEdges; e++)
      if (focusSet.has(brain.pre[e]) === inFocus) chosen.push(e);
    return chosen;
  });
  const edges = [];
  for (let depth = 0; edges.length < limits.maxEdges; depth++) {
    let added = false;
    for (let k = 0; k < focus.length && edges.length < limits.maxEdges; k++) {
      const e = queues[k][depth];
      if (e !== undefined) { edges.push({ id: e, pre: brain.pre[e], post: focus[k], weight: brain.w[e] }); added = true; }
    }
    if (!added) break;
  }
  const shared = new Map();
  for (const edge of edges) { if (!shared.has(edge.pre)) shared.set(edge.pre, new Set()); shared.get(edge.pre).add(edge.post); }
  const trace = { schema: "cadence-settlement-trace/v1", phase: "free", n: brain.n, edgeCount: brain.edges,
    dt: brain.dt, tolerance, maxSteps, sampling: "bounded adaptive decimation; deltaV is the immediately preceding solver update",
    selection: limits.focus ? "requested neuron indices; bounded incoming CSR sample" : "deterministic population/index sample; bounded incoming CSR sample",
    patches: focus.map(id => ({ id, labels: labels.get(id), incomingCount: brain.rowPtr[id + 1] - brain.rowPtr[id],
      sampledIncomingCount: edges.filter(e => e.post === id).length })), edges,
    sharedInputs: [...shared].filter(([, posts]) => posts.size > 1).map(([id, posts]) => ({ id, posts: [...posts] })), frames: [] };
  let stride = 1;
  let previousPotential = null, previousDefect = null, last = null;
  function capture(iteration, residual, force = false) {
    const potential = focus.map(i => finite(brain.v[i]));
    const valid = Number.isFinite(residual);
    const defect = focus.map(i => valid ? finite(brain.controlDefect[i]) : null);
    last = { iteration, residual, potential, defect, previousPotential, previousDefect };
    previousPotential = potential; previousDefect = defect;
    if (force || iteration === 0 || (limits.maxFrames > 2 && iteration % stride === 0)) save(last);
  }
  function save(state) {
    if (!state || trace.frames.at(-1)?.iteration === state.iteration) return;
    if (trace.frames.length === limits.maxFrames) {
      // Preserve the initial state and every other retained frame. Sampling
      // starts densely enough to reveal short solves, then progressively thins.
      trace.frames = trace.frames.filter((_, k) => k % 2 === 0);
      stride *= 2;
    }
    const valid = Number.isFinite(state.residual);
    let settledCount = null;
    if (valid) {
      settledCount = 0;
      for (let i = 0; i < brain.n; i++) if (Math.abs(brain.controlDefect[i]) <= tolerance) settledCount++;
    }
    const synapticInput = focus.map(i => {
      let sum = 0;
      for (let e = brain.rowPtr[i]; e < brain.rowPtr[i + 1]; e++) sum += brain.s[brain.pre[e]] * brain.w[e];
      return finite(sum);
    });
    trace.frames.push({ iteration: state.iteration, residual: finite(state.residual), settledCount,
      localMaxDefect: valid ? Math.max(...state.defect.map(Math.abs)) : null,
      potential: state.potential, activity: focus.map(i => finite(brain.s[i])), drive: focus.map(i => finite(brain.drive[i])),
      bias: focus.map(i => finite(brain.bias[i])), synapticInput, defect: state.defect,
      previousPotential: state.previousPotential, previousDefect: state.previousDefect,
      deltaV: state.potential.map((v, k) => state.previousPotential === null ? 0 : v === null || state.previousPotential[k] === null ? null : finite(v - state.previousPotential[k])),
      edgeActivity: edges.map(e => finite(brain.s[e.pre])), contribution: edges.map(e => finite(brain.s[e.pre] * brain.w[e.id])) });
  }
  return {
    capture,
    finish(result) { save(last); return { ...trace, ...result }; },
  };
}
