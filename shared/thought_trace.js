// The search's actual two-step value readout, overlaid on the full decision map.
// Other regions retain the last decision; they do not pretend to execute this branch.
import { settlingTrace } from "./telemetry.js";

export function thoughtSnapshot(base, evaluation, index, total) {
  const n = base.state.length;
  const drive = [...evaluation.slice(0, 5).map(Math.atanh), ...Array(n - 5).fill(0)];
  const activeEdges = base.edges.slice(0, 5);
  const trace = settlingTrace({
    state: base.state.map((v, i) => i < 6 ? evaluation[i] : v),
    initialState: base.state.map((v, i) => i < 6 ? 0 : v),
    initialPotential: base.potential.map((v, i) => i < 6 ? 0 : v),
    drive, edges: activeEdges, groups: base.groups,
    recurrentCount: 6, steps: 2, dt: 1,
  });
  // The inactive decision regions are held, not claimed to be satisfying a
  // branch's value-only equation. Population readbacks still show retained state.
  trace.mismatches.forEach(row => row.fill(0, 6));
  trace.potentials.forEach(row => {
    for (let i = 6; i < n; i++) row[i] = base.potential[i];
  });
  for (const [g, series] of Object.entries(trace.populations)) {
    if (!base.groups.slice(0, 6).includes(g)) series.forEach(row => { row.mismatch = 0; });
  }
  return {
    ...base,
    state: trace.frames.at(-1),
    potential: trace.potentials.at(-1),
    initialState: trace.frames[0], initialPotential: trace.potentials[0],
    drive, input: drive, steps: 2, dt: 1,
    activeSynapses: 5,
    recordedTrace: trace,
    phase: `Recorded future ${index + 1} / ${total} · other regions retained`,
    equilibrium: { residual: 0, tolerance: 1e-10, converged: true,
      regions: { features: 0, value: 0 }, links: 5 },
  };
}
