// Display-only snapshots of actual solver iterates. Intermediate states never
// supply motor commands, probabilities, eligibility or a convergence verdict.
export const PROGRESS_REPAIR_SCALE = .1; // fixed potential units, never per-frame normalization
const phases = new Set(['observation', 'target:banana', 'target:bread', 'free', 'plus', 'minus']);
export function progressScanFrame(message, members, atlasCount) {
  const m = message;
  if (m?.type !== 'settlement_progress' || m.authoritative !== false || m.valid !== true || !phases.has(m.phase)
    || !Number.isSafeInteger(m.requestId) || !Number.isSafeInteger(m.generation)
    || !Number.isInteger(m.n) || m.n < 1 || !Number.isInteger(atlasCount) || atlasCount < m.n
    || !Number.isInteger(m.iteration) || m.iteration < 0 || m.iteration > 1024
    || !Number.isFinite(m.residual) || m.residual < 0 || !Number.isFinite(m.tolerance) || m.tolerance <= 0
    || !(m.activity instanceof Float32Array) || !(m.deltaV instanceof Float32Array)
    || m.activity.length !== m.n || m.deltaV.length !== m.n || members?.length !== m.n)
    throw new Error('invalid_settlement_progress');
  const activity = new Float32Array(atlasCount), heat = new Float32Array(atlasCount), seen = new Uint8Array(atlasCount);
  for (let i = 0; i < m.n; i++) {
    const index = members[i], a = m.activity[i], d = m.deltaV[i];
    if (!Number.isInteger(index) || index < 0 || index >= atlasCount || seen[index] || !Number.isFinite(a) || !Number.isFinite(d))
      throw new Error('invalid_settlement_progress_values');
    seen[index] = 1; activity[index] = a;
    heat[index] = Math.log1p(99 * Math.min(1, Math.abs(d) / PROGRESS_REPAIR_SCALE)) / Math.log(100);
  }
  return { activity, heat };
}
