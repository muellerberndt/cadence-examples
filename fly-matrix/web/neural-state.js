// The brain's initial all-neuron equation defect measures input mismatch at
// the start of a frozen-input solve. It can also include prior disequilibrium;
// it is neither a learned novelty score nor evidence of an emotion. This
// observer never writes a neuron, samples an action or creates a reward.
// The optional r/(scale+r) meter is supplied presentation scaling, not biology.
export const BRAIN_STATE_MAX_AGE_SECONDS = 1, BRAIN_STATE_DISPLAY_SCALE = 1;
const id = value => Number.isSafeInteger(value) && value >= 0;
const nonnegative = value => Number.isFinite(value) && value >= 0;

export class BrainStateTelemetry {
  #generation;
  #lastRequestId = -1;
  #sample = null;
  #reason = "waiting for fresh input";
  #maxAge;
  #scale;

  constructor({ generation = 0, maxAgeSeconds = BRAIN_STATE_MAX_AGE_SECONDS, scale = BRAIN_STATE_DISPLAY_SCALE } = {}) {
    if (!id(generation) || !Number.isFinite(maxAgeSeconds) || maxAgeSeconds <= 0
      || !Number.isFinite(scale) || scale <= 0) throw new RangeError("invalid_brain_state_configuration");
    this.#generation = generation; this.#maxAge = maxAgeSeconds; this.#scale = scale;
    Object.freeze(this);
  }

  reset(generation) {
    if (!id(generation) || generation < this.#generation) throw new RangeError("invalid_brain_state_generation");
    this.#generation = generation; this.#lastRequestId = -1; this.#sample = null;
    this.#reason = "waiting for fresh input";
  }

  // Supply the highest issued request ID on a disturbance/revoke so even a
  // previously unseen but already in-flight old observation cannot repopulate it.
  invalidate(reason = "waiting for fresh input", throughRequestId = null) {
    if (throughRequestId !== null && !id(throughRequestId)) throw new RangeError("invalid_brain_state_request_id");
    if (throughRequestId !== null) this.#lastRequestId = Math.max(this.#lastRequestId, throughRequestId);
    this.#sample = null; this.#reason = String(reason);
  }

  observe(message, { generation, requestId, observationTime, neuronCount = null } = {}) {
    if (!message || message.type !== "control" || message.diagnosticOnly === true
      || (message.kind !== undefined && message.kind !== "control")
      || !id(generation) || generation !== this.#generation || message.generation !== generation
      || !id(requestId) || message.requestId !== requestId || requestId <= this.#lastRequestId
      || !nonnegative(observationTime) || (this.#sample && observationTime < this.#sample.observationTime)
      || (neuronCount !== null && (!id(neuronCount) || neuronCount < 1))) return false;
    this.#lastRequestId = requestId;
    // A current failed/nonfinite initial measurement replaces stale old data
    // with unavailable status; capped finite solves still measured a mismatch.
    if (!nonnegative(message.initialResidual) || typeof message.converged !== "boolean"
      || !id(message.iterations) || !Number.isFinite(message.tolerance) || message.tolerance <= 0
      || !(message.residual === null || nonnegative(message.residual))
      || (message.active !== undefined && !id(message.active))
      || (message.active !== undefined && neuronCount !== null && message.active > neuronCount)
      || (message.converged && (message.residual === null || message.residual > message.tolerance))) {
      this.#sample = null; this.#reason = "neural mismatch unavailable"; return false;
    }
    this.#sample = Object.freeze({ generation, requestId, observationTime, neuronCount,
      // Worker state replies count activity >=0.5 over the entire graph. Failed
      // solves omit that count; missing telemetry must never masquerade as zero.
      active: message.active ?? null, initialResidual: message.initialResidual, residual: message.residual,
      converged: message.converged, iterations: message.iterations, tolerance: message.tolerance,
      reason: typeof message.reason === "string" ? message.reason : "" });
    this.#reason = "";
    return true;
  }

  read({ time, generation = this.#generation, enabled = true, paused = false } = {}) {
    const sample = this.#sample;
    const matching = generation === this.#generation && nonnegative(time) && sample && time >= sample.observationTime;
    const ageSeconds = matching ? time - sample.observationTime : null;
    const state = !enabled ? "off" : !matching ? "waiting" : ageSeconds >= this.#maxAge ? "stale" : paused ? "paused" : "live";
    const available = state === "live" || state === "paused", raw = available ? sample.initialResidual : null;
    return Object.freeze({ state, available, raw,
      // This form avoids overflow when scale and residual are both very large.
      level: available ? (raw === 0 ? 0 : 1 / (1 + this.#scale / raw)) : null,
      ageSeconds, sample: sample && generation === this.#generation ? Object.freeze({ ...sample }) : null,
      reason: state === "waiting" ? this.#reason || "waiting for fresh input"
        : state === "stale" ? "last input snapshot expired" : state === "paused" ? "recorded input while paused"
        : state === "off" ? "neural observation disabled" : "",
      scale: this.#scale, maxAgeSeconds: this.#maxAge });
  }
}
