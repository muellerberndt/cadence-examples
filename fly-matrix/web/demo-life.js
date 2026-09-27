// Checked connectome motor readouts and MBON approach/avoid decisions retain
// their causal roles. The public hybrid demo supplies the previously tested
// flight routes, fruit selection, landing, perching, feeding and grooming from
// AssistedLife/Life. Those body routines are not learned connectome behavior.
import { AssistedLife, ASSISTED_MOTOR_TTL } from './assisted-life.js';
import { EPISODE_S } from './life.js';

export const DEMO_OBSERVATION_TTL = 4;
export const DEMO_DECISION_TTL = 10;
const neutral = () => ({ yawRate: 0, forwardSpeed: 0, verticalSpeed: 0, feed: 0 });

export class DemoLife extends AssistedLife {
  constructor(flight, room, seed = 1, options = {}) {
    super(flight, room, seed, options);
    this.source = 'hybrid-demo'; this.neuralEnabled = options.neuralEnabled !== false;
    this.decisionWaitSeconds = DEMO_DECISION_TTL;
    this._routeObservation = null; this.lastQualifiedObservation = null;
    this.authority.mode = 'hybrid-demo';
    this.authority.neuralDecision = 'checked MBON approach/avoid for the supplied current fruit search; credited only after the choice is used';
    this.authority.retina = 'rendered photoreceptor drive alongside the declared analytic sensory encoders';
    this.authority.navigation = { command: neutral(), active: false, framesApplied: 0, attention: null };
    this.authority.motionSupport = { enabled: false, config: null, phase: 'inactive', active: false,
      source: null, sourceAgeSeconds: null, originalCommand: neutral(), desiredVelocityWorld: [0, 0, 0],
      velocityWorld: [0, 0, 0], appliedHeading: null, appliedFrames: 0, freshFrames: 0,
      delayedFrames: 0, heldFrames: 0, stoppingFrames: 0 };
    this.authority.limits.observationAdmissionSeconds = DEMO_OBSERVATION_TTL;
    this.authority.limits.targetDecisionSeconds = DEMO_DECISION_TTL;
  }

  expectObservation(requestId, generation = this.generation, options = {}) {
    if (!this.neuralEnabled) return false;
    const request = super.expectObservation(requestId, generation, options);
    if (!request) return request;
    Object.assign(this._pendingObservation, { observationTime: this.clock,
      freshUntil: this.clock + ASSISTED_MOTOR_TTL, deadline: this.clock + DEMO_OBSERVATION_TTL,
      attention: null, searchToken: null });
    return { ...this._pendingObservation };
  }

  applyObservation(message) {
    const request = this._pendingObservation;
    if (!this.neuralEnabled || !request || message?.requestId !== request.requestId
      || message.generation !== request.generation) return this._discard('observations', 'stale or disabled demo observation');
    if (message.attention !== null || message.searchToken !== null) {
      this._pendingObservation = null; this._observation = null; this.lastQualifiedObservation = null;
      return this._discard('observations', 'unexpected demo observation attention');
    }
    const accepted = super.applyObservation(message);
    if (!accepted) { this._observation = null; this.lastQualifiedObservation = null; return false; }
    // The wider window permits displaying an honestly aged checked state and
    // fair worker scheduling. It does not extend the direct motor-trim lease.
    this._observation.deadline = request.freshUntil;
    if (this.clock >= request.freshUntil) this._observation = null;
    this.lastQualifiedObservation = { requestId: request.requestId, generation: request.generation,
      observationTime: request.observationTime, acceptedAt: this.clock, deadline: request.deadline };
    this.authority.retinal = message.retinal ? structuredClone(message.retinal) : null;
    return true;
  }

  openDecision(requestId, generation = this.generation, options = {}) {
    if (!this.neuralEnabled || !this.search) return null;
    const search = this.search;
    const deadline = Math.min(search.askedAt + DEMO_DECISION_TTL, search.since + EPISODE_S);
    if (!Number.isFinite(deadline) || this.clock >= deadline) return null;
    // Keep the inherited identity/valence validation. The same supplied latency
    // allowance is also read by Life's near-fruit waiting/landing routine.
    const request = super.openDecision(requestId, generation, options);
    if (!request) return request;
    this._pendingDecision.deadline = request.deadline = deadline;
    return { ...request, selectTarget: false, attention: search.fruit };
  }

  applyAssistedDecision(message) {
    if (!this.neuralEnabled) return this._discard('choices', 'neural demo decisions disabled');
    return super.applyAssistedDecision(message);
  }

  revokeNeural(reason = 'neural authority revoked') {
    this.lastQualifiedObservation = null; this._routeObservation = null;
    return super.revokeNeural(reason);
  }
}
