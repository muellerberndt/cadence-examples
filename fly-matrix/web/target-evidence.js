// Shared admission check for actual settled odor competition. This validates
// worker evidence; it never chooses a target or changes the neural state.
import { TARGET_CANDIDATES, candidateScore, targetProbabilities, selectTarget } from './neural-policy.js';
const finite = x => typeof x === 'number' && Number.isFinite(x);
const qualified = (s, request) => s?.converged === true && finite(s.residual) && s.residual >= 0
  && finite(s.tolerance) && s.tolerance > 0 && s.tolerance <= request.tolerance && s.residual <= s.tolerance
  && Number.isSafeInteger(s.iterations) && s.iterations >= 0 && s.iterations <= request.steps;

export function validTargetEvidence(message, request) {
  const t = message?.targetSelection;
  if (!request || message?.requestId !== request.requestId || message.generation !== request.generation
    || message.searchToken !== request.searchToken || t?.temperature !== request.temperature
    || !Array.isArray(t.candidates) || t.candidates.length !== TARGET_CANDIDATES.length) return false;
  try {
    if (!t.candidates.every((c, i) => c.fruit === TARGET_CANDIDATES[i].fruit && qualified(c.solve, request)
      && finite(c.score) && Math.abs(c.score - candidateScore(c.readouts)) <= 1e-12)) return false;
    const p = targetProbabilities(t.candidates, request.temperature);
    return Array.isArray(t.p) && t.p.length === p.length && t.p.every((v, i) => finite(v) && Math.abs(v - p[i]) <= 1e-12)
      && selectTarget(t.candidates, t.draw, request.temperature) === t.fruit;
  } catch { return false; }
}
