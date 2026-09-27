// Unadopted peripheral-encoding candidate. These are hand-set candidate genes,
// not measurements of fly receptor kinetics. The existing linear encoder is
// the control. No food identity, sugar label, neural output or position enters.
export const ODOR_TRANSDUCTION_GENES = Object.freeze({
  fedHalfConcentration: 1e-3,
  hungerSensitivityRatio: 100,
  hillExponent: 1,
});

/** Supplied saturating receptor response to a nonnegative bilateral odor level.
 * Its hunger-dependent half-response spans the measured room-plume range.
 * This is NOT an implementation of endogenous insulin/sNPF signaling. */
export function odorReceptorResponse(concentration, hunger) {
  if (!Number.isFinite(concentration) || concentration < 0
      || !Number.isFinite(hunger) || hunger < 0 || hunger > 1)
    throw new RangeError('finite nonnegative concentration and hunger in [0,1] required');
  if (concentration === 0) return 0;
  const half = ODOR_TRANSDUCTION_GENES.fedHalfConcentration
    * ODOR_TRANSDUCTION_GENES.hungerSensitivityRatio ** (-hunger);
  return 1 / (1 + half / concentration);
}
