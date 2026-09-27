// Fair service for one serial brain worker. A checked observation may be valid
// adapter input while too old for raw motor trim. That distinction must not
// permanently starve the actor's multi-phase decision job. This selector grants
// no neural/learning authority; controller and solver admission remain separate.
export function chooseWorkerJob({ observationDue, wantDecision, qualifiedObservationId, lastDecisionId,
  decisionCertifiesInput = false }) {
  const observedSinceDecision = Number.isSafeInteger(qualifiedObservationId) && qualifiedObservationId >= 0
    && Number.isSafeInteger(lastDecisionId) && qualifiedObservationId > lastDecisionId;
  // Goal jobs independently qualify both attended inputs and every free/nudged
  // actor phase. Requiring a different, unattended input to settle first can
  // starve those valid jobs. This explicit scheduling option grants no motor
  // authority: the caller still supplies its current sensed frame, serializes
  // worker jobs, and validates the returned request identity and phase evidence.
  if (wantDecision === true && (decisionCertifiesInput === true || observedSinceDecision)) return "decision";
  if (observationDue === true) return "observation";
  return null;
}

// Use only a currently admitted source. A failed, revoked, detached or expired
// source cannot stand in for the recovery observation required before a choice.
export function qualifiedObservationId(life, generation) {
  if (life.neuralEnabled === false) return null;
  const source = life._motionSource;
  if (source && !source.detached && source.generation === generation
      && Number.isFinite(source.holdUntil) && Number.isFinite(life.clock) && life.clock < source.holdUntil)
    return source.requestId;
  const route = life._routeObservation;
  if (route && route.generation === generation && Number.isFinite(route.deadline)
      && Number.isFinite(life.clock) && life.clock < route.deadline) return route.requestId;
  const admitted = life.lastQualifiedObservation;
  if (admitted && admitted.generation === generation && Number.isFinite(admitted.deadline)
      && Number.isFinite(life.clock) && life.clock < admitted.deadline) return admitted.requestId;
  return null;
}
