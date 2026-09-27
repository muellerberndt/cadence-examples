// PATCH NET SETTLES: sensory drive propagates through the connected recurrent
// brain; accepted states meet the checked potential-equation residual threshold.
// PATCH NET CONTROLS: bounded left/right wing-plane and stroke-shift trim, read
// from settled motor neurons, changes aerodynamic forces and the fly's motion.
// PATCH NET DECIDES: settled MBON activities determine the approach/avoid
// probabilities for the current food search. Free and BOTH nudged phases qualify.
// PATCH NET LEARNS: local KC->MBON efficacy changes, credited only after the chosen
// action was used and its later outcome matches, change later neural responses.
// EVIDENCE: tests/assisted-life.mjs compares actual-payload motor outputs with a
// zero-output lesion at identical body/pilot state and checks decision execution.
// tests/assisted-worker.mjs checks qualified phases and matching reward credit;
// receipts/assisted_evidence_1024.json records a learned yeast response that is
// removed/recovered by restoring/reinstating the neural weights.
// Together these implement a sensory -> decision -> action -> memory loop.
// This earlier controller remains a tested base. The default NeuralLife subclass
// replaces its scripted target/route/body routines with neural navigation.
// WE ASSIST by supplying encoders, the action sampler, reward/critic helper,
// power/frequency, balance, routes/fruit selection, bouts, landing/perch hold,
// takeoff impulses, feeding and grooming. The pilot and routines remain active
// during successful solves as well as fallback; neural authority is as above.
// Compilation, equations, input forcing, authority and evidence: ../docs/FLY_MATRIX_BRAIN
import { Life, APPETITE, DECISION_WAIT_S } from "./life.js";
import { CONTROL_KEYS, BETA_MAX, S_MAX } from "./body.js";
import { settledMotor, SETTLED_MOTOR_GROUPS } from "./motor.js";

export const ASSISTED_MOTOR_TTL = 1; // simulated seconds from the observation request
export const ASSISTED_PLANE_LIMIT = 0.05; // rad; declared candidate, not trained/tuned
export const ASSISTED_SHIFT_LIMIT = 0.00005; // m; declared candidate, not trained/tuned
export const ASSISTED_OUTPUTS = ["mbon:MBON11:right", "mbon:MBON05:left"];
const finite = x => typeof x === "number" && Number.isFinite(x);
const integer = x => Number.isSafeInteger(x) && x >= 0;
const clip = (x, limit) => Math.max(-limit, Math.min(limit, x));
const zero = () => Object.fromEntries(CONTROL_KEYS.map(key => [key, 0]));
const qualified = (solve, request) => solve && solve.converged === true && finite(solve.residual)
  && solve.residual >= 0 && finite(solve.tolerance) && solve.tolerance > 0
  && solve.tolerance <= request.tolerance && solve.residual <= solve.tolerance
  && integer(solve.iterations) && solve.iterations <= request.steps;

export class AssistedLife extends Life {
  constructor(flight, room, seed = 1, options = {}) {
    super(flight, room, seed);
    this.generation = options.generation ?? 0;
    if (!integer(this.generation)) throw new RangeError("invalid assisted generation");
    this.source = "assisted";
    this.baseline = null;
    this._observation = null; this._pendingObservation = null; this._pendingDecision = null;
    this._lastObservationId = -1; this._lastDecisionId = -1; this._searchNumber = 0;
    this.authority = {
      mode: "assisted", autonomous: false,
      pilotWings: "HandPilot supplies power, frequency and stabilizing wing commands",
      navigationScript: "fruit-bearing turns, wall avoidance, speed, altitude and landing targets",
      behaviorFallback: "bouts, saccades, takeoff impulse, perch position/attitude hold, sitting, grooming and feeding",
      senses: "engineered transduction; HS/VS and ocellar projection inputs bypass upstream visual processing",
      neuralDecision: "checked MBON approach/avoid choice for its still-open search only",
      neuralMotor: "checked motor-neuron plane/shift trim; pilot retains amplitude/frequency",
      limits: { planeRadians: ASSISTED_PLANE_LIMIT, shiftMetres: ASSISTED_SHIFT_LIMIT, ttlSeconds: ASSISTED_MOTOR_TTL },
      observations: { requested: 0, accepted: 0, discarded: 0, framesApplied: 0 },
      choices: { requested: 0, accepted: 0, discarded: 0, executed: 0, creditedOutcomes: 0, uncreditedOutcomes: 0 },
      pilotControls: { ...this.controls }, requestedNeuralTrim: zero(), neuralTrim: zero(), executedControls: { ...this.controls },
      lastChoice: null, lastDiscard: null, lastCancellation: null,
    };
    const pilotControls = this.pilot.controls.bind(this.pilot);
    this.pilot.controls = body => {
      // Qualified motor-neuron activities provide the neural wing trim; the
      // causal body-motion comparison is in tests/assisted-life.mjs. We assist
      // with HandPilot's baseline commands and declared actuator limits.
      const base = pilotControls(body), out = { ...base }, requested = zero(), applied = zero();
      if (this._observation && this.clock < this._observation.deadline) {
        const wings = this._observation.motors.wings;
        for (const [key, index, limit, bodyLimit] of [["betaL", 2, ASSISTED_PLANE_LIMIT, BETA_MAX], ["betaR", 3, ASSISTED_PLANE_LIMIT, BETA_MAX], ["sL", 4, ASSISTED_SHIFT_LIMIT, S_MAX], ["sR", 5, ASSISTED_SHIFT_LIMIT, S_MAX]]) {
          requested[key] = clip(wings[index], limit);
          out[key] = clip(base[key] + requested[key], bodyLimit);
          applied[key] = out[key] - base[key];
        }
        if (Object.values(applied).some(value => value !== 0)) this.authority.observations.framesApplied++;
      }
      this.authority.pilotControls = { ...base };
      this.authority.requestedNeuralTrim = requested;
      this.authority.neuralTrim = applied;
      this.authority.executedControls = { ...out };
      return out;
    };
  }

  _discard(kind, reason) {
    this.authority[kind].discarded++;
    this.authority.lastDiscard = { kind, reason, t: this.clock };
    return false;
  }

  _request(requestId, generation, options, lastId, ttl) {
    const steps = options.steps ?? 1024, tolerance = options.tolerance ?? 1e-6;
    if (!integer(requestId) || requestId <= lastId || generation !== this.generation || !integer(steps) || steps < 1 || steps > 1024 || !finite(tolerance) || tolerance <= 0 || tolerance > 1e-6) return null;
    return { requestId, generation, steps, tolerance, deadline: this.clock + ttl };
  }

  expectObservation(requestId, generation = this.generation, options = {}) {
    const request = this._request(requestId, generation, options, this._lastObservationId, ASSISTED_MOTOR_TTL);
    if (!request) return this._discard("observations", "invalid observation request");
    this._lastObservationId = requestId; this._pendingObservation = request;
    this.authority.observations.requested++;
    return { ...request };
  }

  applyObservation(message) {
    const request = this._pendingObservation;
    if (!request || message?.type !== "control" || message.kind !== "control"
      || message.requestId !== request.requestId || message.generation !== request.generation)
      return this._discard("observations", "stale or unrequested observation");
    this._pendingObservation = null;
    const reads = message.readouts;
    if (this.clock >= request.deadline || !qualified(message, request) || !reads || typeof reads !== "object"
      || Array.isArray(reads) || !Object.values(reads).every(finite)
      || !SETTLED_MOTOR_GROUPS.every(name => Object.hasOwn(reads, name) && finite(reads[name]))) {
      this._observation = null;
      return this._discard("observations", "expired, unresolved or incomplete observation");
    }
    this._observation = { motors: settledMotor(reads), deadline: request.deadline };
    this.readouts = { ...reads };
    this.authority.observations.accepted++;
    return true;
  }

  openDecision(requestId, generation = this.generation, options = {}) {
    // The network determines approach/avoid probabilities for this open search.
    // We assist by selecting the fruit, request time and route geometry.
    const wait = this.decisionWaitSeconds ?? DECISION_WAIT_S;
    const request = this._request(requestId, generation, options, this._lastDecisionId, wait);
    const temperature = options.temperature ?? 0.3;
    if (!request || !finite(temperature) || temperature <= 0 || !this.search || !this.search.asked
      || this.search.landed || this.search.neuralChoice || this.mode !== "flying" || this.landing
      || !this.valence?.innate) return null;
    this.search.neuralToken ??= `${this.generation}:${++this._searchNumber}`;
    request.searchToken = this.search.neuralToken; request.temperature = temperature;
    request.deadline = Math.min(request.deadline, this.search.askedAt + wait);
    if (!finite(request.deadline) || this.clock >= request.deadline) return null;
    this._lastDecisionId = requestId;
    this._pendingDecision = { ...request, search: this.search, fruit: this.search.fruit };
    this.authority.choices.requested++;
    return { ...request };
  }

  applyAssistedDecision(message) {
    // Admitting a checked sampled choice is not yet executing it. The ownership
    // record below must survive until decide() actually uses its valence.
    const request = this._pendingDecision;
    if (!request || message?.type !== "assisted_decision" || message.kind !== "decision"
      || message.generation !== request.generation || message.requestId !== request.requestId
      || message.searchToken !== request.searchToken) return this._discard("choices", "stale or unrequested choice");
    this._pendingDecision = null;
    if (this.clock >= request.deadline || this.search !== request.search || this.search.landed
      || this.search.fruit !== request.fruit || this.search.neuralChoice || this.mode !== "flying"
      || this.landing || !this.valence?.innate || this.hunger <= APPETITE)
      return this._discard("choices", "search no longer permits a neural choice");
    const decision = message.decision, solves = decision?.solves ?? message.solves;
    if (message.accepted !== true || !qualified(message, request)
      || !["free", "plus", "minus"].every(phase => qualified(solves?.[phase], request))
      || !decision || decision.greedy === true || ![0, 1].includes(decision.action)
      || decision.choice !== decision.action || !Array.isArray(decision.p) || decision.p.length !== 2
      || !decision.p.every(p => finite(p) && p >= 0 && p <= 1)
      || Math.abs(decision.p[0] + decision.p[1] - 1) > 1e-12
      || !finite(decision.draw) || decision.draw < 0 || decision.draw >= 1
      || decision.action !== (decision.draw > decision.p[0] ? 1 : 0)
      || !ASSISTED_OUTPUTS.every(name => finite(message.readouts?.[name])))
      return this._discard("choices", "invalid free/nudged phase or decision evidence");
    const logits = ASSISTED_OUTPUTS.map(name => message.readouts[name] / request.temperature);
    const peak = Math.max(...logits), weights = logits.map(x => Math.exp(x - peak));
    const expectedP = weights[0] / (weights[0] + weights[1]);
    if (!finite(expectedP) || Math.abs(expectedP - decision.p[0]) > 1e-10) return this._discard("choices", "probabilities do not match settled MBON readouts");
    const record = { requestId: request.requestId, generation: request.generation, searchToken: request.searchToken,
      fruit: request.fruit, action: decision.action, executed: false, p: [...decision.p] };
    this.search.neuralChoice = record;
    super.applyDecision(decision);
    this.valence.neuralToken = request.searchToken;
    this.authority.choices.accepted++;
    this.authority.lastChoice = { ...record };
    return true;
  }

  // Raw legacy replies never receive decision or descending-neuron authority.
  applyDecision() { return this._discard("choices", "unchecked legacy decision"); }
  setBaseline() { this.baseline = null; }

  decide() {
    // The neural approach/avoid choice changes the heading or landing request.
    // We assist with Life's geometric translation of that choice. Legacy
    // DN/GF/MN9 overrides have no authority here.
    const record = this.search?.neuralChoice;
    const executes = record && !record.executed && !record.revoked && this.mode === "flying" && this.escape <= 0
      && !this.search.landed && this.valence?.neuralToken === record.searchToken && this.hunger > APPETITE;
    const reads = this.readouts;
    this.readouts = {}; this.baseline = null;
    try { super.decide(true); } finally { this.readouts = reads; }
    if (executes) {
      record.executed = true; this.authority.choices.executed++;
      this.authority.lastChoice = { ...record, executedAt: this.clock };
    }
  }

  outcome(reward, why, on = null) {
    // The neural actor learns from the outcome of its matching executed choice.
    // We assist with landing and reward transduction; an assistance-only landing
    // without that neural choice supplies no actor-learning credit.
    const record = this.search?.neuralChoice;
    const credit = !!record?.executed && !record.revoked && (on === null || (record.action === 0 && record.fruit === on));
    super.outcome(reward, why, on);
    this.pendingReward = { ...this.pendingReward, fresh: !credit, neuralCredit: credit,
      requestId: credit ? record.requestId : null, generation: this.generation,
      searchToken: credit ? record.searchToken : null };
    this._pendingDecision = null;
    this.authority.choices[credit ? "creditedOutcomes" : "uncreditedOutcomes"]++;
  }

  revokeNeural(reason = "neural authority revoked") {
    const pending = this._pendingDecision, accepted = this.search?.neuralChoice;
    const cancel = pending ?? (accepted && !accepted.executed ? accepted : null);
    const cancellation = cancel ? { requestId: cancel.requestId, generation: cancel.generation,
      searchToken: cancel.searchToken, reason } : null;
    this._pendingObservation = null; this._pendingDecision = null; this._observation = null;
    this.readouts = {}; this.baseline = null;
    // A paused renderer may never call step(). Withdraw the actual current trim
    // immediately while retaining only the last supplied pilot command in flight.
    const fallback = this.mode === "flying" && !this.wingsOff ? { ...this.authority.pilotControls } : zero();
    this.authority.pilotControls = { ...fallback };
    this.authority.requestedNeuralTrim = zero();
    this.authority.neuralTrim = zero();
    this.authority.executedControls = { ...fallback };
    this.controls = { ...fallback };
    if (this.search?.neuralChoice) {
      this.search.neuralChoice.revoked = true;
      if (this.valence) this.valence = { fruit: this.search.fruit, action: 0, p: null, innate: true };
    }
    this.authority.lastDiscard = { kind: "all", reason, t: this.clock };
    this.authority.lastCancellation = cancellation;
    return cancellation;
  }

  step(dt) {
    if (this._observation && this.clock + dt >= this._observation.deadline) this._observation = null;
    this.authority.pilotControls = zero(); this.authority.requestedNeuralTrim = zero();
    this.authority.neuralTrim = zero(); this.authority.executedControls = zero();
    return super.step(dt);
  }

  pose() { return { ...super.pose(), assisted: true }; }
}
