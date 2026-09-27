// Recorded numerical Euler updates, never synthetic propagation or biological spikes.
// Float32 display copies do not participate in the Float64 solver or its residual test.
export const REPLAY_MAX_FRAMES = 7;
export const REPLAY_MAX_BYTES = 8_500_000;
export const REPLAY_SCHEMA = "cadence-neural-replay/v1";

export function validateReplayOptions(options, n) {
  if (options === true) options = {};
  if (!options || typeof options !== "object" || Array.isArray(options)) throw new Error("invalid_replay_options");
  const maxFrames = options.maxFrames ?? REPLAY_MAX_FRAMES;
  if (!Number.isInteger(maxFrames) || maxFrames < 2 || maxFrames > REPLAY_MAX_FRAMES
      || !Number.isInteger(n) || n < 1 || n * maxFrames * 8 > REPLAY_MAX_BYTES) throw new Error("invalid_replay_budget");
  return { maxFrames };
}

export function createNeuralReplay(brain, options, tolerance) {
  const { maxFrames } = validateReplayOptions(options, brain.n);
  // This scratch vector records v_after - v_before at EVERY actual solver step.
  // Sparse display sampling therefore never substitutes a many-step difference.
  const deltaV = new Float32Array(brain.n), frames = [], scheduled = new Set([0]);
  for (let iteration = 1; scheduled.size < maxFrames - 1; iteration *= 2) scheduled.add(iteration);
  const save = (iteration, residual) => {
    if (frames.at(-1)?.iteration === iteration) return;
    const activity = Float32Array.from(brain.s), change = deltaV.slice();
    let valid = Number.isFinite(residual), maxAbsDeltaV = 0;
    for (let i = 0; i < brain.n; i++) {
      if (!Number.isFinite(activity[i]) || !Number.isFinite(change[i])) valid = false;
      else maxAbsDeltaV = Math.max(maxAbsDeltaV, Math.abs(change[i]));
    }
    frames.push({ iteration, residual: Number.isFinite(residual) ? residual : null,
      activity, deltaV: change, valid, maxAbsDeltaV: valid ? maxAbsDeltaV : null });
  };
  return {
    deltaV,
    capture(iteration, residual) { if (scheduled.has(iteration)) save(iteration, residual); },
    finish(diagnostic) {
      save(diagnostic.iterations, diagnostic.residual);
      return { schema: REPLAY_SCHEMA, n: brain.n, dt: brain.dt, tolerance,
        representation: "Float32 display copies of Float64 solver activity and actual last-step potential change",
        schedule: "initial, powers of two through the frame budget, final; no interpolated states",
        frames, ...diagnostic,
        maxAbsDeltaV: Math.max(0, ...frames.filter(f => f.valid).map(f => f.maxAbsDeltaV)),
        arrayBytes: frames.reduce((sum, f) => sum + f.activity.byteLength + f.deltaV.byteLength, 0) };
    },
  };
}

// Transfer only detached display copies; no live brain, learner or member array is sent.
export function replayTransferables(replay) {
  if (!replay) return [];
  const buffers = [];
  for (const frame of replay.frames) for (const values of [frame.activity, frame.deltaV]) {
    if (!(values instanceof Float32Array)) throw new Error("invalid_replay_array");
    buffers.push(values.buffer);
  }
  if (new Set(buffers).size !== buffers.length) throw new Error("aliased_replay_buffers");
  return buffers;
}

/** Map one recorded state into anatomical atlas indices. Heat is a display scale of
 * |actual last-step delta v| with a FIXED peak over the replay, not a residual,
 * activity derivative, learned weight update, or inferred propagation speed. */
export function replayScanFrame(replay, frameIndex, members = null, atlasCount = replay?.n) {
  if (!replay || replay.schema !== REPLAY_SCHEMA || !Number.isInteger(replay.n) || replay.n < 1
      || !Array.isArray(replay.frames) || !replay.frames.length || replay.frames.length > REPLAY_MAX_FRAMES
      || !Number.isInteger(frameIndex) || frameIndex < 0 || frameIndex >= replay.frames.length
      || !Number.isInteger(atlasCount) || atlasCount < replay.n
      || !Number.isFinite(replay.maxAbsDeltaV) || replay.maxAbsDeltaV < 0) throw new Error("invalid_replay");
  const frame = replay.frames[frameIndex], n = replay.n;
  if (!frame.valid || !(frame.activity instanceof Float32Array) || !(frame.deltaV instanceof Float32Array)
      || frame.activity.length !== n || frame.deltaV.length !== n
      || !Number.isFinite(frame.residual) || frame.residual < 0
      || !Number.isInteger(frame.iteration) || frame.iteration < 0 || frame.iteration > replay.iterations
      || !Number.isFinite(frame.maxAbsDeltaV) || frame.maxAbsDeltaV < 0
      || frame.maxAbsDeltaV > replay.maxAbsDeltaV) throw new Error("invalid_replay_frame");
  if (members !== null && (!ArrayBuffer.isView(members) && !Array.isArray(members) || members.length !== n)) throw new Error("invalid_replay_members");
  const activity = new Float32Array(atlasCount), heat = new Float32Array(atlasCount);
  const seen = members === null ? null : new Uint8Array(atlasCount), peak = replay.maxAbsDeltaV;
  for (let i = 0; i < n; i++) {
    const index = members === null ? i : members[i], s = frame.activity[i], change = Math.abs(frame.deltaV[i]);
    if (!Number.isInteger(index) || index < 0 || index >= atlasCount || seen?.[index]) throw new Error("invalid_replay_members");
    if (!Number.isFinite(s) || !Number.isFinite(change) || change > peak) throw new Error("invalid_replay_values");
    if (seen) seen[index] = 1;
    activity[index] = s;
    heat[index] = peak > 0 ? Math.log1p(99 * change / peak) / Math.log(100) : 0;
  }
  return { activity, heat, identity: replay.identity ? { ...replay.identity } : null,
    frameIndex, frameCount: replay.frames.length, iteration: frame.iteration, iterations: replay.iterations,
    residual: frame.residual, converged: replay.converged, maxAbsDeltaV: peak,
    frameMaxAbsDeltaV: frame.maxAbsDeltaV, mappedCount: n, n,
    label: `Recorded solve #${replay.identity?.requestId ?? "?"} · step ${frame.iteration}/${replay.iterations}`,
    encoding: `All ${n.toLocaleString("en-US")} cells · |Δv| fixed relative log scale` };
}
