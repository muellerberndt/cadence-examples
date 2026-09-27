// The patch net chooses the odor goal and MBON approach/avoid action. A supplied
// body controller interprets that intent using room geometry and the existing
// wing-force integrator. Goal tracking, clearance, landing/perching and the
// fixed feeding/grooming/takeoff sequence are assistance, not learned behavior.
import { DemoLife } from './demo-life.js';
import { APPETITE, DECISION_S, HOVER_HEIGHT, LAND_REACH, GROOM_S, FEED_S } from './life.js';
import { ROOM, HandPilot, CONTROL_KEYS, wrap_ } from './body.js';
import { validTargetEvidence } from './target-evidence.js';

export const GOAL_BODY = Object.freeze({ speed: .3, clearance: .15, retreat: .6,
  avoidClimb: .15, reached: .04, restSeconds: 2, retrySeconds: .5, maxExecutionSeconds: 20 });
export const GOAL_GROOM_SEQUENCE = Object.freeze(['head', 'legs', 'wings']);
export const GOAL_STARTUP = Object.freeze({ speed: GOAL_BODY.speed, distance: 1.2,
  maxSeconds: 6, clearance: GOAL_BODY.clearance, reached: .015 });
export const GOAL_WAITING = Object.freeze({ speed: .3, turnMargin: .35, innerClearance: .15,
  turnRate: 2.5, reactionSeconds: .15, minRecoveryHeight: .4 });
export const GOAL_DECISION_TTL = 20;
export const GOAL_OBSERVATION_TTL = 10;
const clip = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const zeroControls = () => Object.fromEntries(CONTROL_KEYS.map(key => [key, 0]));

export class GoalLife extends DemoLife {
  constructor(flight, room, seed = 1, options = {}) {
    super(flight, room, seed, options);
    this.source = 'neural-goal'; this._nextSearchAt = 0;
    this._mechanicalTarget = flight.p.slice(); this._goalApplied = null;
    this._perchedAt = null; this._primitive = null; this._groomIndex = 0;
    this.decisionWaitSeconds = GOAL_DECISION_TTL;
    this.authority.limits.targetDecisionSeconds = GOAL_DECISION_TTL;
    this.authority.limits.observationAdmissionSeconds = GOAL_OBSERVATION_TTL;
    this.authority.mode = 'neural-goal';
    this.authority.navigationScript = 'supplied navigation/clearance/landing executes only the checked brain-selected fruit and approach/avoid action';
    this.authority.neuralDecision = 'settled odor competition chooses the fruit; checked free/plus/minus MBON evidence chooses approach or avoid';
    this.authority.bodyRoutines = 'supplied initial-heading launch, current-heading waiting cruise, room-clearance turns, goal tracking, landing/perching, contact feeding, grooming rotation, takeoff and physical nudge recovery; not learned motor competence';
    this.authority.targetSelection = { requested: 0, accepted: 0, last: null };
    this.authority.goal = { phase: 'waiting', fruit: null, action: null, decisionRequestId: null,
      generation: this.generation, searchToken: null, target: null, executed: false,
      framesApplied: 0, clearanceLimited: false, config: { ...GOAL_BODY } };
    if (options.startup !== undefined && typeof options.startup !== 'boolean') throw new TypeError('startup must be a boolean');
    const enabled = options.startup !== false && this.neuralEnabled;
    const origin = flight.p.slice(), heading = flight.euler()[2], direction = [Math.cos(heading), Math.sin(heading)];
    let distance = GOAL_STARTUP.distance;
    for (let axis = 0; axis < 2; axis++) if (Math.abs(direction[axis]) > 1e-12) {
      const boundary = direction[axis] > 0 ? ROOM[axis] - GOAL_STARTUP.clearance : GOAL_STARTUP.clearance;
      distance = Math.min(distance, Math.max(0, (boundary - origin[axis]) / direction[axis]));
    }
    const target = [origin[0] + distance * direction[0], origin[1] + distance * direction[1], origin[2]];
    const launch = enabled && distance > GOAL_STARTUP.reached && !flight.onFloor && !flight.landed;
    this.authority.startup = { enabled, phase: launch ? 'launching' : enabled ? 'blocked' : 'disabled', active: false,
      heading, origin, target, startedAt: this.clock, deadline: this.clock + GOAL_STARTUP.maxSeconds,
      endedAt: launch ? null : this.clock, reason: launch ? null : enabled ? 'no forward clearance or grounded' : 'disabled',
      framesApplied: 0, config: { ...GOAL_STARTUP } };
    if (launch) this._mechanicalTarget = target.slice();
    if (options.waiting !== undefined && typeof options.waiting !== 'boolean') throw new TypeError('waiting must be a boolean');
    this._waitingSuspended = !this.neuralEnabled;
    this.authority.waitingSupport = { enabled: options.waiting !== false, active: false, phase: 'waiting',
      heading, desiredHeading: heading, altitude: origin[2], velocityWorld: [0, 0, 0],
      framesApplied: 0, turns: 0, lastTurn: null, config: { ...GOAL_WAITING } };
    this.authority.physicalReaction = { active: false, until: null, startedAt: null, events: 0, framesApplied: 0 };
    this.authority.behaviorFallback = 'supplied current-heading cruise and room-clearance turns between brain goals; fixed contact feeding, grooming rotation, perch and takeoff primitives';
    // Life.step still owns physical integration and landing/perch mechanics.
    // Replace only its unselected forward waypoint before the pilot reads it.
    const controls = this.pilot.controls.bind(this.pilot);
    this.pilot.controls = body => {
      if (this.authority.physicalReaction.active) {
        const off = zeroControls();
        this.authority.pilotControls = { ...off }; this.authority.neuralTrim = { ...off };
        this.authority.requestedNeuralTrim = { ...off }; this.authority.executedControls = { ...off };
        return off;
      }
      if (this.mode === 'flying' && !this.landing) {
        this.pilot.target = this._mechanicalTarget.slice(); this.pilot.speed = GOAL_BODY.speed;
        const dx = this.pilot.target[0] - body.p[0], dy = this.pilot.target[1] - body.p[1];
        if (Math.hypot(dx, dy) > 1e-6) this.heading = Math.atan2(dy, dx);
        this.pilot.heading = this.heading;
      }
      const choice = this.search?.neuralChoice, goal = this.authority.goal;
      if (this.mode === 'flying' && choice && !choice.revoked && goal.decisionRequestId === choice.requestId
        && ['approach', 'avoid', 'landing'].includes(goal.phase)
        && Math.hypot(...this.pilot.target.map((x, i) => x - body.p[i])) > 1e-6)
        this._goalApplied = choice;
      return controls(body);
    };
  }

  odourTick() {
    if (!this.neuralEnabled || this.authority.physicalReaction.active) return false;
    if (this.search?.neuralChoice && !this.search.landed && this.clock >= this.search.goalDeadline) {
      this.outcome(0, 'brain-selected goal episode ended'); return false;
    }
    if (this._pendingDecision && this.clock >= this._pendingDecision.deadline) {
      this._retryGoal('goal decision expired'); return false;
    }
    return !this.search && !this.pendingReward && this.mode === 'flying'
      && this.hunger > APPETITE && this.clock >= this._nextSearchAt;
  }

  expectObservation(requestId, generation = this.generation, options = {}) {
    const request = super.expectObservation(requestId, generation, options);
    if (!request) return request;
    // An aged qualified observation may unblock high-level goal scheduling.
    // DemoLife.applyObservation still expires direct motor trim at freshUntil,
    // one second from this request, regardless of this scheduling allowance.
    request.deadline = this._pendingObservation.deadline = this.clock + GOAL_OBSERVATION_TTL;
    return request;
  }

  openDecision(requestId, generation = this.generation, options = {}) {
    if (!this.neuralEnabled || this.authority.physicalReaction.active || this.search || this.pendingReward || this.mode !== 'flying' || this.hunger <= APPETITE) return null;
    this.search = { since: this.clock, askedAt: this.clock, fruit: null, asked: true, landed: null };
    this.valence = { fruit: null, action: 0, p: null, innate: true };
    const request = super.openDecision(requestId, generation, options);
    if (!request) { this.search = null; this.valence = null; return null; }
    // The goal is a bounded historical intent, not a current motor reading.
    // Body travel during computation does not extend the raw trim's one-second
    // lease or relax any free/candidate/nudged residual requirement.
    request.deadline = this._pendingDecision.deadline = this.clock + GOAL_DECISION_TTL;
    this.authority.targetSelection.requested++;
    return { ...request, selectTarget: true, attention: null };
  }

  applyAssistedDecision(message) {
    const request = this._pendingDecision, selection = message?.targetSelection;
    const matching = request && this.search === request.search && message?.requestId === request.requestId
      && message.generation === request.generation && message.searchToken === request.searchToken;
    if (!this.neuralEnabled || !validTargetEvidence(message, request)) {
      this._discard('choices', 'invalid neural goal evidence');
      if (matching) this._retryGoal('current neural goal evidence rejected');
      return false;
    }
    if (this.search !== request.search) return this._discard('choices', 'goal search replaced');
    const previous = this.search.fruit;
    this.search.fruit = request.fruit = this.valence.fruit = selection.fruit;
    if (!super.applyAssistedDecision(message)) {
      if (this.search === request.search) {
        this.search.fruit = previous; if (this.valence) this.valence.fruit = previous;
        this._retryGoal('current neural goal action rejected');
      }
      return false;
    }
    this._finishStartup('qualified neural goal admitted', false);
    this.authority.waitingSupport.active = false; this.authority.waitingSupport.phase = 'goal';
    this.authority.waitingSupport.velocityWorld = [0, 0, 0];
    this._pendingObservation = null; this._observation = null; this.lastQualifiedObservation = null;
    const choice = this.search.neuralChoice;
    // The bounded body-execution window starts when the choice is admitted.
    // Worker latency must not spend the body's travel/landing budget. This
    // does not extend the independent observation or decision-admission TTL.
    this.search.goalDeadline = this.clock + GOAL_BODY.maxExecutionSeconds;
    Object.assign(this.authority.goal, { phase: choice.action === 0 ? 'approach' : 'avoid', fruit: choice.fruit,
      action: choice.action, decisionRequestId: choice.requestId, searchToken: choice.searchToken,
      generation: choice.generation, executed: false, clearanceLimited: false });
    this.authority.targetSelection.accepted++;
    this.authority.targetSelection.last = structuredClone(selection);
    this.authority.lastChoice.targetSelection = structuredClone(selection);
    if (choice.action === 1) this._setRetreat(); else this._trackGoal();
    return true;
  }

  _boundedTarget(target) {
    const bounded = target.map((x, i) => clip(x, GOAL_BODY.clearance, ROOM[i] - GOAL_BODY.clearance));
    this.authority.goal.clearanceLimited = target.some((x, i) => x !== bounded[i]);
    this._mechanicalTarget = bounded; this.authority.goal.target = bounded.slice();
  }

  _retryGoal(reason) {
    // A refused current solve must not leave an asked, target-null search with
    // no pending deadline. Stale identities never enter this branch. Revocation
    // also ends the one-shot launch; retrying a brain choice cannot restart it.
    this.revokeNeural(reason); this.resumeBodySupport();
    this._nextSearchAt = this.clock + GOAL_BODY.retrySeconds;
  }

  _setRetreat() {
    const p = this.flight.p, fruit = this.fruits[this.search.neuralChoice.fruit].pos;
    const dx = p[0] - fruit[0], dy = p[1] - fruit[1], length = Math.hypot(dx, dy);
    // A finite retreat stops at the room's inner clearance boundary instead of
    // continually commanding outward speed against a wall. At exact overlap,
    // the supplied body heading supplies the mechanical axis, not a new goal.
    const ux = length > 1e-9 ? dx / length : Math.cos(this.flight.euler()[2]);
    const uy = length > 1e-9 ? dy / length : Math.sin(this.flight.euler()[2]);
    this._boundedTarget([p[0] + GOAL_BODY.retreat * ux, p[1] + GOAL_BODY.retreat * uy, p[2] + GOAL_BODY.avoidClimb]);
  }

  _trackGoal() {
    const goal = this.authority.goal, choice = this.search?.neuralChoice;
    if (!choice || choice.revoked || goal.decisionRequestId !== choice.requestId) return;
    if (choice.action === 0 && !this.landing) {
      const fruit = this.fruits[choice.fruit].pos;
      this._boundedTarget([fruit[0], fruit[1], fruit[2] + HOVER_HEIGHT]);
      if (Math.hypot(fruit[0] - this.flight.p[0], fruit[1] - this.flight.p[1]) < LAND_REACH) this.chooseLanding();
    } else if (choice.action === 1 && Math.hypot(...this._mechanicalTarget.map((x, i) => x - this.flight.p[i])) < GOAL_BODY.reached
      && this.flight.speed() < .1) this.outcome(0, 'completed chosen avoidance');
  }

  chooseLanding() {
    const choice = this.search?.neuralChoice;
    if (!choice || choice.revoked || choice.action !== 0) return false;
    this.landingFruit = { name: choice.fruit, dx: 0, dy: 0 };
    this.landing = this.landingPoint();
    this.authority.goal.phase = 'landing'; this.authority.goal.target = this.landing.slice();
    this.log(`body descends to brain-selected ${choice.fruit}`); return true;
  }

  decide() {
    this.ethogram.decisions++;
    if (this.mode === 'flying') { this._trackGoal(); return; }
    // These fixed body primitives intentionally replace the legacy random
    // bouts. They neither choose a fruit nor create neural execution credit.
    if (this.mode === 'landed' && this.clock - (this._perchedAt ?? this.clock) >= GOAL_BODY.restSeconds) {
      if (this.feedingContact() && this.hunger > .3 && this._primitive !== 'fed') { this.startFeeding(); this._primitive = 'feeding'; }
      else this.startGrooming();
    } else if (this.mode === 'feeding' || this.mode === 'grooming') {
      this.episode -= DECISION_S;
      if (this.episode <= 0) {
        if (this.mode === 'feeding') { this._primitive = 'fed'; this.startGrooming(); }
        else this.takeoff('supplied takeoff after body routine');
      }
    }
  }

  startGrooming() {
    this.mode = 'grooming'; this.episode = GROOM_S; this.ethogram.grooms++;
    // Declared cosmetic motor primitives rotate across visits. This sequence
    // does not choose food, alter actor probabilities or consume a random draw.
    this.groomTarget = GOAL_GROOM_SEQUENCE[this._groomIndex++ % GOAL_GROOM_SEQUENCE.length];
    this.log(`supplied ${this.groomTarget} grooming`);
  }

  startFeeding() { super.startFeeding(); this.episode = FEED_S; }

  takeoff(why) {
    super.takeoff(why);
    this._mechanicalTarget = [this.flight.p[0], this.flight.p[1], this.altitude];
    this._perchedAt = null; this._primitive = null; this.authority.goal.phase = 'waiting'; this.authority.goal.target = null;
    this.resumeBodySupport();
  }

  _stopGoal() {
    this._finishStartup('body goal stopped', false);
    this.landing = null; this.landingFruit = null; this._mechanicalTarget = this.flight.p.slice();
    this.heading = this.flight.euler()[2]; this.pilot.target = this.flight.p.slice(); this.pilot.heading = this.heading;
    this.pilot.speed = 0; this._goalApplied = null;
    Object.assign(this.authority.goal, { phase: this.mode === 'flying' ? 'waiting' : 'perched', fruit: null,
      action: null, decisionRequestId: null, searchToken: null, target: null, executed: false, clearanceLimited: false });
  }

  outcome(reward, why, on = null) {
    super.outcome(reward, why, on); this._stopGoal();
    this._pendingObservation = null; this._observation = null; this.lastQualifiedObservation = null;
    this._nextSearchAt = this.clock + GOAL_BODY.retrySeconds;
  }

  revokeNeural(reason = 'goal authority revoked') {
    this._waitingSuspended = true;
    if (this.authority.waitingSupport) Object.assign(this.authority.waitingSupport,
      { active: false, phase: 'suspended', velocityWorld: [0, 0, 0] });
    if (this.authority.physicalReaction) Object.assign(this.authority.physicalReaction, { active: false, until: null });
    this._finishStartup(reason, false);
    const cancellation = super.revokeNeural(reason);
    this.search = null; this.valence = null; this.wantDecision = false; this._stopGoal();
    // Pausing may run no further physics step: withdraw the selected target now.
    if (this.mode === 'flying') this.controls = this.pilot.controls(this.flight);
    return cancellation;
  }

  step(dt) {
    if (!Number.isFinite(dt) || dt <= 0) throw new RangeError('invalid goal timestep');
    const startup = this.authority.startup;
    if (startup.phase === 'launching') {
      if (!this.neuralEnabled || this.mode !== 'flying' || this.flight.onFloor || this.flight.landed)
        this._finishStartup('launch disabled or grounded');
      else if (this.clock + dt >= startup.deadline) this._finishStartup('launch time limit');
      else if (Math.hypot(...startup.target.map((x, i) => x - this.flight.p[i])) < GOAL_STARTUP.reached
        && this.flight.speed() < .05) this._finishStartup('launch endpoint reached');
      else if (this.authority.waitingSupport.enabled && !this._waitingSuspended
        && Math.hypot(...startup.target.map((x, i) => x - this.flight.p[i])) < .2)
        this._finishStartup('continuous waiting cruise', false);
    }
    if (!this.neuralEnabled && (this.search || this.authority.goal.decisionRequestId !== null)) this.revokeNeural('neural goals disabled');
    const reaction = this.authority.physicalReaction;
    if (reaction.active && this.clock >= reaction.until) {
      reaction.active = false; reaction.until = null;
      this.resumeBodySupport();
      this.authority.waitingSupport.altitude = Math.max(GOAL_WAITING.minRecoveryHeight, this.flight.p[2]);
    }
    this._updateWaiting(dt);
    this._goalApplied = null;
    const before = this.mode, decides = super.step(dt), applied = this._goalApplied;
    startup.active = startup.phase === 'launching' && before === 'flying' && this.controls.f > 0;
    if (startup.active) startup.framesApplied++;
    if (reaction.active) reaction.framesApplied++;
    const waiting = this.authority.waitingSupport;
    if (waiting.active && this.controls.f > 0) waiting.framesApplied++;
    if (applied && this.search?.neuralChoice === applied && !applied.revoked && this.controls.f > 0) {
      this.authority.goal.framesApplied++;
      if (!applied.executed) {
        applied.executed = true; this.authority.choices.executed++;
        this.authority.lastChoice = { ...applied, executedAt: this.clock, targetSelection: structuredClone(this.authority.targetSelection.last) };
      }
      this.authority.goal.executed = true;
    }
    if (before === 'flying' && this.mode === 'landed') { this._perchedAt = this.clock; this._primitive = null; this.authority.goal.phase = 'perched'; }
    return decides;
  }

  _finishStartup(reason, holdPosition = true) {
    const startup = this.authority.startup;
    if (!startup || startup.phase !== 'launching') return;
    startup.phase = reason === 'qualified neural goal admitted' ? 'handoff' : 'finished';
    startup.active = false; startup.endedAt = this.clock; startup.reason = reason;
    // This launch is a one-shot body primitive, not a new actor decision. Keep
    // its record after stopping so later waits, pause/resume and food changes
    // cannot silently restart it. Only a new controller/reset starts anew.
    if (holdPosition) this._mechanicalTarget = this.flight.p.slice();
  }

  resumeBodySupport() {
    this._waitingSuspended = false;
    const support = this.authority.waitingSupport;
    support.active = false; support.phase = 'waiting';
    support.heading = support.desiredHeading = this.flight.euler()[2];
    support.altitude = this.flight.p[2]; support.velocityWorld = [0, 0, 0];
  }

  _updateWaiting(dt) {
    const support = this.authority.waitingSupport, f = this.flight;
    const available = support.enabled && this.neuralEnabled && !this._waitingSuspended
      && this.mode === 'flying' && !this.search?.neuralChoice && !this.landing
      && !this.authority.physicalReaction.active && this.authority.startup.phase !== 'launching';
    if (!available) {
      support.active = false; support.velocityWorld = [0, 0, 0];
      support.phase = this.authority.physicalReaction.active ? 'reaction' : this._waitingSuspended ? 'suspended'
        : this.search?.neuralChoice ? 'goal' : this.mode !== 'flying' ? 'perched'
          : this.authority.startup.phase === 'launching' ? 'startup' : 'disabled';
      return;
    }
    if (!support.active) {
      support.heading = support.desiredHeading = this.heading;
      support.altitude = Math.max(GOAL_WAITING.minRecoveryHeight, f.p[2]);
    }
    const before = support.desiredHeading, vector = [Math.cos(before), Math.sin(before)], walls = [];
    for (let axis = 0; axis < 2; axis++) {
      if (f.p[axis] < GOAL_WAITING.turnMargin && vector[axis] < 0) {
        vector[axis] *= -1; walls.push(axis === 0 ? 'left' : 'near');
      } else if (f.p[axis] > ROOM[axis] - GOAL_WAITING.turnMargin && vector[axis] > 0) {
        vector[axis] *= -1; walls.push(axis === 0 ? 'right' : 'far');
      }
      if (f.p[axis] < GOAL_WAITING.innerClearance && vector[axis] < .5) vector[axis] = .5;
      else if (f.p[axis] > ROOM[axis] - GOAL_WAITING.innerClearance && vector[axis] > -.5) vector[axis] = -.5;
    }
    support.desiredHeading = Math.atan2(vector[1], vector[0]);
    if (walls.length) {
      support.turns++; support.lastTurn = { time: this.clock, walls, before, after: support.desiredHeading };
    }
    support.heading = wrap_(support.heading + clip(wrap_(support.desiredHeading - support.heading),
      -GOAL_WAITING.turnRate * dt, GOAL_WAITING.turnRate * dt));
    support.velocityWorld = [GOAL_WAITING.speed * Math.cos(support.heading), GOAL_WAITING.speed * Math.sin(support.heading), 0];
    this._mechanicalTarget = [f.p[0] + support.velocityWorld[0] / HandPilot.K_POS,
      f.p[1] + support.velocityWorld[1] / HandPilot.K_POS, support.altitude];
    support.active = true; support.phase = 'cruise';
  }

  respondToNudge() {
    // Called after the page applies its actual impulse and cancels old credit.
    // Release contact holds without setting position, attitude or velocity, and
    // briefly let Flight.step integrate the impulse before stabilization resumes.
    this.revokeNeural('physical nudge');
    this.perch = null; this.landing = null; this.landingFruit = null;
    this.mode = 'flying'; this.wingsOff = false; this.episode = 0;
    this._perchedAt = null; this._primitive = null;
    this.flight.landed = false; this.flight.onFloor = false; this.flight.touching = 0;
    const reaction = this.authority.physicalReaction;
    Object.assign(reaction, { active: true, startedAt: this.clock, until: this.clock + GOAL_WAITING.reactionSeconds,
      events: reaction.events + 1 });
    this._nextSearchAt = reaction.until;
    this.controls = zeroControls();
    this.authority.pilotControls = zeroControls(); this.authority.neuralTrim = zeroControls();
    this.authority.requestedNeuralTrim = zeroControls(); this.authority.executedControls = zeroControls();
  }
}
