// Read-only diagnostic replay. Never feeds rendered values back into a controller.
export function repairTrace(source, release = false) {
  if (source.recordedTrace && !release) return source.recordedTrace;
  const n = source.state.length,
    mask = source.mask ?? Array(n).fill(1),
    dt = source.dt ?? 1;
  let state = release
    ? source.state.slice()
    : (source.initialState?.slice() ?? Array(n).fill(0));
  let potential = release
    ? (source.potential?.slice() ??
      state.map((x) =>
        Math.atanh(Math.max(-0.999999999, Math.min(0.999999999, x))),
      ))
    : (source.initialPotential?.slice() ?? Array(n).fill(0));
  const drive = release ? Array(n).fill(0) : source.drive;
  for (let i = source.recurrentCount ?? n; i < n; i++)
    state[i] = source.state[i];
  const frames = [state.slice()],
    residuals = [0],
    potentials = [potential.slice()];
  const steps = release
    ? Math.max(80, Math.min(source.steps ?? 200, 400))
    : source.steps;
  for (let t = 0; t < steps; t++) {
    const inbox = Array(n).fill(0);
    for (const [a, b, w] of source.edges) inbox[b] += w * state[a];
    potential = potential.map((v, i) =>
      !mask[i]
        ? 0
        : i >= (source.recurrentCount ?? n)
          ? Math.atanh(
              Math.max(-0.999999999, Math.min(0.999999999, source.state[i])),
            )
          : dt === 1
            ? inbox[i] + drive[i]
            : v + dt * (inbox[i] + drive[i] - v),
    );
    const next = potential.map((v, i) =>
      i >= (source.recurrentCount ?? n)
        ? source.state[i]
        : mask[i] * Math.tanh(v),
    );
    residuals.push(Math.max(...next.map((v, i) => Math.abs(v - state[i]))));
    state = next;
    frames.push(state.slice());
    potentials.push(potential.slice());
  }
  const mismatches = frames.map((activity, t) => {
    const inbox = Array(n).fill(0);
    for (const [a, b, w] of source.edges) inbox[b] += w * activity[a];
    return potentials[t].map((v, i) =>
      mask[i] ? inbox[i] + drive[i] - v : -v,
    );
  });
  const groups = [...new Set(source.groups ?? Array(n).fill("patch"))];
  const populations = Object.fromEntries(
    groups.map((g) => {
      const ids = Array.from({ length: n }, (_, i) => i).filter(
        (i) => (source.groups?.[i] ?? "patch") === g && mask[i],
      );
      const average = (a) =>
        a.reduce((sum, v) => sum + v, 0) / Math.max(1, a.length);
      return [
        g,
        frames.map((activity, t) => ({
          mean: average(ids.map((i) => activity[i])),
          rms: Math.sqrt(average(ids.map((i) => activity[i] ** 2))),
          mismatch: Math.sqrt(average(ids.map((i) => mismatches[t][i] ** 2))),
        })),
      ];
    }),
  );
  return { frames, residuals, release, potentials, mismatches, populations };
}
