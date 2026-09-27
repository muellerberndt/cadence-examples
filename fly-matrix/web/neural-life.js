// Neural target attention and continuous navigation, with a supplied stabilizer.
// Life's fruit-bearing routes, wall turns, random bouts and automatic takeoffs
// are bypassed. Geometry enters only sensory transduction and physical contact.
import { AssistedLife, ASSISTED_MOTOR_TTL } from "./assisted-life.js";
import { APPETITE, DECISION_S, EPISODE_S, HUNGER_RATE } from "./life.js";
import { HandPilot, CONTROL_KEYS, RADIUS, wrap_ } from "./body.js";
import { NAVIGATION_READOUTS, decodeNavigation, NAVIGATION_GAINS } from "./neural-policy.js";
import { validTargetEvidence } from "./target-evidence.js";
import { motionSupportOptions, smoothVelocity, MOTION_SUPPORT_EPSILON } from "./motion-support.js";
export { MOTION_SUPPORT_DEFAULTS } from "./motion-support.js";

// Full-graph candidate competition and both nudged action phases need a bounded
// response window of their own. Ten simulated seconds is a supplied latency
// budget, not a learned parameter or a relaxed numerical solve. The inherited
// three-second window serves the legacy landing wait. Motor input stays fresh
// for only one second, and execution still needs a new attended observation.
export const NEURAL_TARGET_DECISION_TTL = 10;

const finite = x => typeof x === "number" && Number.isFinite(x);
const zero = () => Object.fromEntries(CONTROL_KEYS.map(k => [k, 0]));
const neutral = () => ({ yawRate: 0, forwardSpeed: 0, verticalSpeed: 0, feed: 0 });

export class NeuralLife extends AssistedLife {
  constructor(flight, room, seed = 1, options = {}) {
    super(flight, room, seed, options);
    this.source = "neural-navigation";
    this.neuralEnabled = options.neuralEnabled !== false;
    this.speed = 0; this.altitude = flight.p[2]; this.pilot.route = null;
    this._facingDirection = 1; this.facingHeading = this.heading;
    this._routeObservation = null; this._lastContactFruit = null; this._feeding = false;
    this._executionEvidence = null;
    this._nextSearchAt = 0; this._navigation = neutral(); this.proboscis = 0;
    this.motionSupport = motionSupportOptions(options.motionSupport);
    this._motionSource = null; this._motionVelocity = [0, 0, 0]; this._lastFreshAppliedObservationId = null;
    this.authority.mode = "neural-navigation";
    this.authority.navigationScript = "none; settled descending and wing-plane motor outputs request local velocity and yaw";
    this.authority.behaviorFallback = "zero translational/yaw request; HandPilot retains balance and velocity damping";
    this.authority.neuralDecision = "settled candidate MBON scores select odor attention, then checked approach/avoid modulates neural navigation";
    this.authority.targetSelection = { requested: 0, accepted: 0, last: null };
    this.authority.execution = { current: null, last: null, lastInvalidation: null };
    this.authority.navigation = { gains: { ...NAVIGATION_GAINS }, command: neutral(), active: false, framesApplied: 0, attention: null,
      facingHeading: this.facingHeading };
    this.authority.facingAssistance = this.motionSupport
      ? "supplied cruise, smoothing and body alignment follow the checked neural direction and yaw; the original neural command is recorded separately, and real wing torques turn the body"
      : "supplied candidate: orient the body along requested travel; signed neural velocity and yaw remain unchanged, and real wing torques turn the body";
    this.authority.limits.targetDecisionSeconds = NEURAL_TARGET_DECISION_TTL;
    this.authority.bodyRoutines = "passive contact; contact-only taste/reward; neural MN9 feeding extension; no supplied route or takeoff impulse";
    this.authority.retina = "page-supplied rendered receptor observations; no visual target/bearing input to controller";
    this.authority.motionSupport = { enabled: !!this.motionSupport, config: this.motionSupport ? { ...this.motionSupport } : null,
      phase: "inactive", active: false, source: null, sourceAgeSeconds: null, originalCommand: neutral(),
      desiredVelocityWorld: [0, 0, 0], velocityWorld: [0, 0, 0], appliedHeading: null,
      appliedFrames: 0, freshFrames: 0, delayedFrames: 0, heldFrames: 0, stoppingFrames: 0 };
  }

  // The inherited encoder may display the strongest odor, but that diagnostic
  // does not select attention or a route. Taste is restricted to actual contact.
  senses(handPos = null, dt = 0) {
    const senses = super.senses(handPos, dt);
    this._contact();
    senses["grn:sugar:labellum"] = senses["grn:sugar:front_leg"] = this.onSugar ? 1 : 0;
    senses.leg_touch = this.flight.touching > 0 ? 0.6 : 0;
    return senses;
  }

  expectObservation(requestId, generation = this.generation, options = {}) {
    if (!this.neuralEnabled) return false;
    const request = super.expectObservation(requestId, generation, options);
    if (!request) return request;
    const record = this.search?.neuralChoice;
    // Attention begins on admission. Waiting for execution here would deadlock:
    // execution itself requires a fresh observation of the attended odor.
    const attention = record && !record.revoked ? record.fruit : null;
    Object.assign(this._pendingObservation, { attention, searchToken: attention ? record.searchToken : null,
      observationTime: this.clock, freshUntil: this.clock + ASSISTED_MOTOR_TTL });
    if (this.motionSupport) this._pendingObservation.deadline = this.clock + this.motionSupport.maxObservationAgeSeconds;
    return { ...this._pendingObservation };
  }

  applyObservation(message) {
    const request = this._pendingObservation;
    if (!this.neuralEnabled || !request || message?.requestId !== request.requestId || message.generation !== request.generation)
      return this._discard("observations", "stale or disabled navigation observation");
    if (message?.attention !== request.attention || message.searchToken !== request.searchToken || !NAVIGATION_READOUTS.every(name => finite(message?.readouts?.[name]))) {
      this._clearExecution("invalid matching observation");
      this._pendingObservation = null; this._observation = null; this._routeObservation = null;
      this._stopRoute();
      return this._discard("observations", "missing navigation readouts or mismatched odor attention");
    }
    if (request.attention !== null && (!this.search?.neuralChoice || this.search.neuralChoice.revoked
      || this.search.neuralChoice.searchToken !== request.searchToken || this.search.neuralChoice.fruit !== request.attention)) {
      this._clearExecution("attended episode closed");
      this._pendingObservation = null; this._observation = null; this._routeObservation = null; this._stopRoute();
      return this._discard("observations", "attended neural episode has closed");
    }
    if (!super.applyObservation(message)) {
      this._clearExecution("matching observation failed admission");
      this._pendingObservation = null; this._observation = null; this._routeObservation = null; this._clearMotionSupport();
      this._stopRoute(true, this.clock >= request.deadline);
      return false;
    }
    // Wider opt-in admission grants only motion-adapter input. Raw motor trim
    // and fresh neural/learning authority keep the original one-second deadline.
    this._observation.deadline = request.freshUntil;
    if (this.clock >= request.freshUntil) this._observation = null;
    this._routeObservation = this.clock < request.freshUntil
      ? { ...request, acceptedAt: this.clock, deadline: request.freshUntil, readouts: { ...message.readouts } } : null;
    if (this.motionSupport) {
      const record = this.search?.neuralChoice;
      this._motionSource = { requestId: request.requestId, generation: request.generation,
        attention: request.attention, searchToken: request.searchToken, observationTime: request.observationTime,
        acceptedAt: this.clock, freshUntil: request.freshUntil, holdUntil: this.clock + this.motionSupport.holdSeconds,
        detached: false, command: decodeNavigation(message.readouts, record && !record.revoked ? record.action : 0) };
    }
    this.authority.retinal = message.retinal ? structuredClone(message.retinal) : null;
    return true;
  }

  odourTick() {
    if (!this.neuralEnabled) return false;
    if (this.search && this.clock - this.search.since >= EPISODE_S) {
      this.outcome(0, "neural attention episode ended", null, { episodeTimeout: true });
      this._nextSearchAt = this.clock + DECISION_S;
      return false;
    }
    if (this._pendingDecision && this.clock >= this._pendingDecision.deadline) {
      this.revokeNeural("target decision expired"); this._nextSearchAt = this.clock + DECISION_S;
    }
    return !this.search && !this.pendingReward && this.clock >= this._nextSearchAt && this.mode === "flying" && this.hunger > APPETITE;
  }

  openDecision(requestId, generation = this.generation, options = {}) {
    if (!this.neuralEnabled || this.search || this.mode !== "flying" || this.hunger <= APPETITE || this.pendingReward) return null;
    // Fruit is deliberately absent until the checked target competition replies.
    this.search = { since: this.clock, askedAt: this.clock, asked: true, landed: null, fruit: null };
    this.valence = { fruit: null, action: 0, p: null, innate: true };
    const request = super.openDecision(requestId, generation, options);
    if (!request) { this.search = null; this.valence = null; return null; }
    const deadline = Math.min(this.search.askedAt + NEURAL_TARGET_DECISION_TTL, this.search.since + EPISODE_S);
    request.deadline = this._pendingDecision.deadline = deadline;
    this.authority.targetSelection.requested++;
    return { ...request, selectTarget: true };
  }

  applyAssistedDecision(message) {
    const request = this._pendingDecision, t = message?.targetSelection;
    const valid = this.neuralEnabled && validTargetEvidence(message, request);
    if (!valid) return this._discard("choices", "invalid neural target-selection evidence");
    // The inherited action validator now binds the selected fruit and its token.
    const previousFruit = this.search?.fruit;
    if (this.search !== request.search) return this._discard("choices", "target search replaced");
    this.search.fruit = request.fruit = this.valence.fruit = t.fruit;
    if (!super.applyAssistedDecision(message)) {
      if (this.search === request.search) { this.search.fruit = previousFruit; if (this.valence) this.valence.fruit = previousFruit; }
      return false;
    }
    this._clearExecution("new actor choice");
    this._observation = null; this._routeObservation = null; this._pendingObservation = null;
    if (this._motionSource) this._motionSource.detached = true;
    this._stopRoute(true, !!this.motionSupport);
    this.authority.targetSelection.accepted++;
    this.authority.targetSelection.last = structuredClone(t);
    this.authority.lastChoice.targetSelection = structuredClone(t);
    return true;
  }

  // Route authority is evaluated on every physics step, not at a scripted bout.
  decide() { this.ethogram.decisions++; }
  chooseLanding() { return false; }
  landingPoint() { return null; }
  startEscape() { return false; }
  takeoff() { return false; }
  startGrooming() { return false; }
  hold() { return false; }
  fruitMoved(name) { if (!Object.hasOwn(this.fruits, name)) return false; this._contact(); return false; }
  punished() {
    // The actual user-applied blow is the event; no nearest-fruit lookup assigns
    // its credit. Only an executed current neural episode can receive it.
    this.visits.punished++;
    if (!this.search) return false;
    this.outcome(-1, "external physical blow"); return true;
  }

  _clearMotionSupport() {
    this._motionSource = null; this._motionVelocity = [0, 0, 0]; this._lastFreshAppliedObservationId = null;
    if (this.authority.motionSupport) Object.assign(this.authority.motionSupport, {
      phase: "inactive", active: false, source: null, sourceAgeSeconds: null, originalCommand: neutral(),
      desiredVelocityWorld: [0, 0, 0], velocityWorld: [0, 0, 0], appliedHeading: null });
  }

  _applyMotionSupport(dt, fresh, freshCommand) {
    const config = this.motionSupport, f = this.flight, telemetry = this.authority.motionSupport;
    if (!this.neuralEnabled || (this._motionSource && this.clock >= this._motionSource.holdUntil)) this._motionSource = null;
    const source = this._motionSource;
    const original = source ? source.command : neutral();
    // A held source is openly supplied execution of the last checked command;
    // it is never counted as a new neural update or a newly executed choice.
    const phase = !source ? "inactive" : fresh && !source.detached ? "fresh"
      : !source.detached && this.clock < source.observationTime + config.maxObservationAgeSeconds ? "delayed_checked" : "held";
    if (source && !fresh) this.heading = wrap_(this.heading + original.yawRate * dt);
    const forward = source ? Math.sign(original.forwardSpeed) * config.cruiseSpeed : 0;
    const desired = [forward === 0 ? 0 : forward * Math.cos(this.heading),
      forward === 0 ? 0 : forward * Math.sin(this.heading), original.verticalSpeed];
    this._motionVelocity = smoothVelocity(this._motionVelocity, desired, dt, config);
    const horizontal = Math.hypot(this._motionVelocity[0], this._motionVelocity[1]);
    if (source && original.forwardSpeed !== 0) this._facingDirection = Math.sign(original.forwardSpeed);
    this.facingHeading = horizontal > MOTION_SUPPORT_EPSILON ? Math.atan2(this._motionVelocity[1], this._motionVelocity[0])
      : source && original.yawRate !== 0 ? wrap_(this.heading + (this._facingDirection < 0 ? Math.PI : 0)) : f.euler()[2];
    this.pilot.heading = this.facingHeading; this.pilot.speed = horizontal;
    this.pilot.target = f.p.map((x, i) => x + this._motionVelocity[i] / HandPilot.K_POS);
    const moving = Math.hypot(...this._motionVelocity) > MOTION_SUPPORT_EPSILON;
    Object.assign(telemetry, {
      phase: !source && moving ? "stopping" : phase,
      source: source ? { requestId: source.requestId, generation: source.generation, attention: source.attention,
        searchToken: source.searchToken, observationTime: source.observationTime, acceptedAt: source.acceptedAt,
        freshUntil: source.freshUntil, holdUntil: source.holdUntil, detached: source.detached } : null,
      sourceAgeSeconds: source ? this.clock - source.observationTime : null,
      originalCommand: { ...original }, desiredVelocityWorld: desired, velocityWorld: this._motionVelocity.slice(),
      appliedHeading: moving || source?.command.yawRate ? this.facingHeading : null,
    });
    // Mouth extension, direct wing trim and learning remain fresh-only.
    this.proboscis = fresh ? freshCommand.feed : 0;
  }

  _stopRoute(refreshControls = true, preserveCourse = false) {
    if (!preserveCourse) this._clearMotionSupport();
    this._navigation = neutral(); this.speed = 0; this.pilot.speed = 0; this.pilot.route = null;
    const bodyHeading = this.flight.euler()[2];
    // A communication gap withdraws actuation, not the signed course's reference
    // frame. Rebasing it to a reverse-facing body would invert the next identical
    // command. Explicit revocation and episode/choice boundaries still rebase.
    if (!preserveCourse) { this.heading = bodyHeading; this._facingDirection = 1; }
    this.pilot.heading = this.facingHeading = bodyHeading;
    this.pilot.target = this.flight.p.slice();
    if (this.authority.navigation) Object.assign(this.authority.navigation, { command: neutral(), active: false, attention: null,
      facingHeading: this.facingHeading });
    if (refreshControls) {
      this.wingsOff = this.flight.touching > 0 && this.flight.onFloor === true;
      this.controls = this.wingsOff ? zero() : this.pilot.controls(this.flight);
      if (this.wingsOff) {
        this.authority.pilotControls = zero(); this.authority.neuralTrim = zero();
        this.authority.requestedNeuralTrim = zero(); this.authority.executedControls = zero();
      }
    }
  }

  revokeNeural(reason = "neural authority revoked") {
    this._clearExecution(reason);
    const record = this.search?.neuralChoice;
    const cancellation = super.revokeNeural(reason) ?? (record ? { requestId: record.requestId, generation: record.generation, searchToken: record.searchToken, reason } : null);
    this._routeObservation = null; this._stopRoute();
    this.search = null; this.valence = null; this.wantDecision = false;
    this.authority.lastCancellation = cancellation;
    return cancellation;
  }

  _contact() {
    const f = this.flight;
    this.onWhich = null;
    if (f.touching > 0 && f.onFloor === true) for (const [name, fruit] of Object.entries(this.fruits)) {
      if (Math.hypot(f.p[0] - fruit.pos[0], f.p[1] - fruit.pos[1]) < 0.06 && Math.abs(f.p[2] - RADIUS - fruit.pos[2]) < 0.003) { this.onWhich = name; break; }
    }
    this.onFruit = this.onWhich !== null; this.onSugar = this.onFruit && this.onWhich === this.sugar;
  }

  _clearExecution(reason) {
    if (this._executionEvidence) this.authority.execution.lastInvalidation = {
      reason, at: this.clock, decisionRequestId: this._executionEvidence.decisionRequestId,
      controlRequestId: this._executionEvidence.controlRequestId };
    this._executionEvidence = null;
    this.authority.execution.current = null;
  }

  // Learning follows an executed action through its bounded episode, not the
  // freshness of whichever sensory packet happens to exist at reward time.
  // Delayed checked execution through the supplied actuator is named explicitly;
  // it never increments fresh navigation/trim counters. Held motion cannot create
  // a record, and numerical/identity failure or hard revocation clears one.
  _recordChoiceExecution(source, command, phase) {
    const choice = this.search?.neuralChoice;
    if (this._executionEvidence || !choice || choice.revoked || !this.neuralEnabled || this.wingsOff
      || !source || source.detached || !["fresh", "delayed_checked"].includes(phase)
      || source.generation !== this.generation || choice.generation !== this.generation
      || source.searchToken !== choice.searchToken || source.attention !== choice.fruit
      || this.search.neuralToken !== choice.searchToken || this.clock >= this.search.since + EPISODE_S
      || ![command.yawRate, command.forwardSpeed, command.verticalSpeed].every(finite)
      || ![command.yawRate, command.forwardSpeed, command.verticalSpeed].some(v => v !== 0)) return;
    const age = this.clock - source.observationTime;
    const maxAge = this.motionSupport ? this.motionSupport.maxObservationAgeSeconds : ASSISTED_MOTOR_TTL;
    if (!finite(age) || age < 0 || age >= maxAge
      || (this.motionSupport && this.clock >= source.holdUntil)) return;
    const evidence = Object.freeze({ decisionRequestId: choice.requestId, controlRequestId: source.requestId,
      generation: this.generation, searchToken: choice.searchToken, fruit: choice.fruit, action: choice.action,
      executedAt: this.clock, observationTime: source.observationTime, acceptedAt: source.acceptedAt,
      sourceAgeSeconds: age, phase, episodeStartedAt: this.search.since, episodeDeadline: this.search.since + EPISODE_S,
      originalCommand: Object.freeze({ ...command }),
      velocityWorld: Object.freeze(this.motionSupport ? this._motionVelocity.slice()
        : [command.forwardSpeed * Math.cos(this.heading), command.forwardSpeed * Math.sin(this.heading), command.verticalSpeed]) });
    this._executionEvidence = evidence;
    this.authority.execution.current = structuredClone(evidence);
    this.authority.execution.last = structuredClone(evidence);
    if (!choice.executed) {
      choice.executed = true; this.authority.choices.executed++;
      const key = phase === "fresh" ? "freshExecuted" : "delayedExecuted";
      this.authority.choices[key] = (this.authority.choices[key] || 0) + 1;
    }
    this.authority.lastChoice = { ...choice, executedAt: evidence.executedAt, executionEvidence: structuredClone(evidence) };
  }

  outcome(reward, why, on = null, options = {}) {
    const choice = this.search?.neuralChoice, evidence = this._executionEvidence;
    // odourTick runs every DECISION_S. Its first zero-reward timeout may land
    // just after the deadline; other outcomes must occur strictly before it.
    const timeout = reward === 0 && options.episodeTimeout === true && evidence
      && this.clock >= evidence.episodeDeadline && this.clock <= evidence.episodeDeadline + DECISION_S + 1e-9;
    const executedCredit = evidence && choice && !choice.revoked && this.neuralEnabled
      && evidence.decisionRequestId === choice.requestId && evidence.generation === this.generation
      && choice.generation === this.generation && evidence.searchToken === choice.searchToken
      && this.search.neuralToken === choice.searchToken && evidence.fruit === choice.fruit && evidence.action === choice.action
      && evidence.episodeStartedAt === this.search.since && this.clock >= evidence.executedAt
      && (this.clock < evidence.episodeDeadline || timeout);
    super.outcome(reward, why, on);
    if (this.pendingReward?.neuralCredit && !executedCredit) {
      this.pendingReward = { ...this.pendingReward, fresh: true, neuralCredit: false, requestId: null, searchToken: null };
      this.authority.choices.creditedOutcomes--; this.authority.choices.uncreditedOutcomes++;
    }
    if (this.pendingReward?.neuralCredit) this.pendingReward.executionEvidence = structuredClone(evidence);
    this._clearExecution("episode outcome recorded");
    this._routeObservation = null; this._observation = null; this._pendingObservation = null;
    if (this._motionSource) this._motionSource.detached = true;
    this._stopRoute(true, !!this.motionSupport);
  }

  step(dt) {
    if (!finite(dt) || dt <= 0) throw new RangeError("invalid neural-navigation timestep");
    const f = this.flight;
    this.clock += dt; this.since += dt;
    if (this.motionSupport && !this.neuralEnabled) {
      this._clearExecution("neural control disabled");
      this._observation = null; this._routeObservation = null; this._stopRoute(false);
    }
    if (this.reward > 0) this.reward = Math.max(0, this.reward - dt);
    if (this.punish > 0) this.punish = Math.max(0, this.punish - dt);
    if (this._observation && this.clock >= this._observation.deadline) this._observation = null;
    if (this._routeObservation && this.clock >= this._routeObservation.deadline) { this._routeObservation = null; this._stopRoute(false, true); }
    const observation = this._routeObservation, record = this.search?.neuralChoice;
    const attended = !observation?.attention || (record && !record.revoked && observation.attention === record.fruit && observation.searchToken === record.searchToken);
    const active = this.neuralEnabled && !!observation && attended;
    const command = active ? decodeNavigation(observation.readouts, record && !record.revoked ? record.action : 0) : neutral();
    this._navigation = command; this.proboscis = command.feed;
    if (active) this.heading = wrap_(this.heading + command.yawRate * dt);
    this.speed = command.forwardSpeed;
    // Supplied facing assistance: keep the neural signed velocity vector in
    // its existing reference frame, but let the physical nose face that travel
    // direction. Negative forward requests therefore add pi to the stabilizer's
    // attitude target, not to the neural heading or world velocity. HandPilot
    // must produce real torques; neither quaternion nor rendered pose is set.
    // At zero speed retain the previous facing side, without suppressing a
    // neural yaw request. During an observation gap only the course is retained:
    // no stored neural yaw/translation target reaches the stabilizer. Hard
    // revocation resets both frames to the current body.
    if (active) {
      if (command.forwardSpeed !== 0) this._facingDirection = Math.sign(command.forwardSpeed);
      this.facingHeading = wrap_(this.heading + (this._facingDirection < 0 ? Math.PI : 0));
    } else this.facingHeading = f.euler()[2];
    this.pilot.route = null; this.pilot.heading = this.facingHeading;
    this.pilot.speed = Math.abs(command.forwardSpeed);
    // Invert the pilot's local velocity servo exactly: K_POS*(target-position)
    // equals the neural velocity request. No absolute waypoint is stored.
    this.pilot.target = [f.p[0] + command.forwardSpeed * Math.cos(this.heading) / HandPilot.K_POS,
      f.p[1] + command.forwardSpeed * Math.sin(this.heading) / HandPilot.K_POS,
      f.p[2] + command.verticalSpeed / HandPilot.K_POS];
    if (this.motionSupport) this._applyMotionSupport(dt, active, command);
    this.altitude = this.pilot.target[2];
    Object.assign(this.authority.navigation, { command: { ...command }, active, attention: observation?.attention ?? null,
      facingHeading: this.facingHeading });
    const contact = f.touching > 0 && f.onFloor === true;
    this.wingsOff = contact && (this.motionSupport ? this._motionVelocity[2] : command.verticalSpeed) <= 0;
    this.controls = this.wingsOff ? zero() : this.pilot.controls(f);
    if (this.wingsOff) {
      this.authority.pilotControls = zero(); this.authority.neuralTrim = zero();
      this.authority.requestedNeuralTrim = zero(); this.authority.executedControls = zero();
    }
    const routeEnacted = active && !this.wingsOff && [command.yawRate, command.forwardSpeed, command.verticalSpeed].some(v => v !== 0);
    if (routeEnacted) this.authority.navigation.framesApplied++;
    if (routeEnacted) this._lastFreshAppliedObservationId = observation.requestId;
    if (this.motionSupport) {
      const support = this.authority.motionSupport;
      support.active = !this.wingsOff && (Math.hypot(...support.velocityWorld) > MOTION_SUPPORT_EPSILON
        || !!(support.source && support.originalCommand.yawRate !== 0));
      if (support.active) {
        support.appliedFrames++;
        support[support.phase === "fresh" ? "freshFrames" : support.phase === "delayed_checked" ? "delayedFrames"
          : support.phase === "held" ? "heldFrames" : "stoppingFrames"]++;
      }
    }
    f.step(dt, this.controls);
    if (this.motionSupport) {
      const support = this.authority.motionSupport;
      if (support.active) this._recordChoiceExecution(this._motionSource, support.originalCommand, support.phase);
    } else if (routeEnacted) this._recordChoiceExecution(observation, command, "fresh");
    this._contact();
    const feeding = this.onSugar && command.feed > 0;
    if (this.onWhich && this.onWhich !== this._lastContactFruit) {
      this.visits[this.onWhich]++;
      if (this.onSugar) { this.visits.sweet++; this.outcome(1, `physical sugar contact on the ${this.onWhich}`, this.onWhich); }
      else if (this.search) this.search.landed = this.onWhich;
    }
    if (feeding && !this._feeding) this.ethogram.feeds++;
    const contactAfter = f.touching > 0 && f.onFloor === true;
    if (contactAfter && this.mode === "flying") this.ethogram.landings++;
    this.mode = feeding ? "feeding" : contactAfter ? "landed" : "flying";
    if (contactAfter) this.ethogram.sitting_s += dt; else this.ethogram.flying_s += dt;
    this.hunger = Math.max(0, Math.min(1, this.hunger + dt * (feeding ? -0.12 * command.feed : HUNGER_RATE)));
    this._feeding = feeding; this._lastContactFruit = this.onWhich;
    if (this.since >= DECISION_S) { this.since %= DECISION_S; return true; }
    return false;
  }

  pose() { return { neural: true, assisted: true, mode: this.mode, t: this.clock, groom: null, feed: this.proboscis, feeding: this._feeding, hunger: this.hunger }; }
}
