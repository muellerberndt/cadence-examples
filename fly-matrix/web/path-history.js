// Record measured body positions and the neural navigation commands actually
// applied at those times. A separate constant-command guide is kinematic only:
// it does not predict the stabilizer/body dynamics or invent a planned route.
// No fruit positions, target bearings, Three objects or DOM state enter here.
export const PATH_INTERVAL = 0.05, PATH_MAX_POINTS = 1200, PATH_MAX_EVENTS = 256;
export const PROJECTION_MAX_SECONDS = 1, PROJECTION_MAX_POINTS = 256;
const finite = x => typeof x === "number" && Number.isFinite(x);
const integer = x => Number.isSafeInteger(x) && x >= 0;
const EMPTY_GUIDE = Object.freeze([]);
const commandKeys = ["forwardSpeed", "yawRate", "verticalSpeed"];

function frameOf(input) {
  if (!input || !integer(input.generation) || !finite(input.time) || input.time < 0
    || !finite(input.heading) || typeof input.active !== "boolean") throw new TypeError("invalid_path_frame");
  const position = input.position;
  if ((!Array.isArray(position) && !ArrayBuffer.isView(position)) || position.length !== 3
    || !Array.from(position).every(finite)) throw new TypeError("invalid_path_position");
  if (!input.command || !commandKeys.every(key => finite(input.command[key]))) throw new TypeError("invalid_path_command");
  const attention = input.attention ?? null, action = input.action ?? null;
  const observationId = input.observationId ?? null, deadline = input.deadline ?? null;
  if (attention !== null && (typeof attention !== "string" || attention.length > 256)) throw new TypeError("invalid_path_attention");
  if (action !== null && action !== 0 && action !== 1) throw new TypeError("invalid_path_action");
  if (observationId !== null && !integer(observationId)) throw new TypeError("invalid_path_observation_id");
  if (deadline !== null && !finite(deadline)) throw new TypeError("invalid_path_deadline");
  return { generation: input.generation, time: input.time, position: Array.from(position),
    heading: input.heading, command: Object.fromEntries(commandKeys.map(key => [key, input.command[key]])),
    active: input.active, attention, action, observationId, deadline };
}

function frozenFrame(frame) {
  return Object.freeze({ ...frame, position: Object.freeze([...frame.position]), command: Object.freeze({ ...frame.command }),
    ...(frame.changes ? { changes: Object.freeze([...frame.changes]) } : {}) });
}

function changesSince(previous, current) {
  if (!previous) return ["start"];
  const changes = [];
  if (commandKeys.some(key => previous.command[key] !== current.command[key])) changes.push("command");
  for (const key of ["active", "attention", "action", "observationId", "deadline"]) {
    if (previous[key] !== current[key]) changes.push(key);
  }
  return changes;
}

export class PathHistory {
  #generation = null;
  #points = [];
  #events = [];
  #current = null;
  #origin = null;
  #nextSample = null;
  #intervalActive = true;
  #interval;
  #maxPoints;
  #maxEvents;

  constructor({ interval = PATH_INTERVAL, maxPoints = PATH_MAX_POINTS, maxEvents = PATH_MAX_EVENTS } = {}) {
    if (!finite(interval) || interval <= 0 || !integer(maxPoints) || maxPoints < 1 || maxPoints > PATH_MAX_POINTS
      || !integer(maxEvents) || maxEvents < 1 || maxEvents > PATH_MAX_EVENTS) throw new RangeError("invalid_path_history_limits");
    this.#interval = interval; this.#maxPoints = maxPoints; this.#maxEvents = maxEvents;
    Object.freeze(this);
  }

  /** Explicit reset permits a new session/time origin. Within a session,
   * generation numbers increase; late records from an older generation reject. */
  reset(generation = null) {
    if (generation !== null && !integer(generation)) throw new TypeError("invalid_path_generation");
    this.#generation = generation;
    this.#points = []; this.#events = []; this.#current = null;
    this.#origin = null; this.#nextSample = null; this.#intervalActive = true;
  }

  /** Call on physics steps, and immediately when pause/revoke changes applied
   * authority. Unsampled calls still update current state and command events.
   * Missing physics samples are never filled with projected/interpolated points.
   * intervalActive includes every recorded active flag since the preceding
   * position sample, including both endpoints. The first sample uses its own
   * active flag. An interruption makes the whole interval conservatively grey;
   * an unsampled current frame carries the same flag for the visible tail. */
  record(input) {
    const current = frameOf(input);
    if (this.#generation !== null && current.generation < this.#generation) throw new RangeError("stale_path_generation");
    if (current.generation !== this.#generation) this.reset(current.generation);
    if (this.#current && current.time < this.#current.time) throw new RangeError("path_time_rewound_without_reset");
    current.intervalActive = this.#intervalActive && current.active;
    this.#intervalActive = current.intervalActive;
    const changes = changesSince(this.#current, current);
    if (changes.length) {
      this.#events.push({ ...current, changes });
      if (this.#events.length > this.#maxEvents) this.#events.shift();
    }
    this.#current = current;
    if (this.#origin === null) { this.#origin = current.time; this.#nextSample = current.time; }
    const epsilon = 8 * Number.EPSILON * Math.max(1, Math.abs(current.time));
    const sampled = current.time + epsilon >= this.#nextSample;
    if (sampled) {
      this.#points.push(current);
      if (this.#points.length > this.#maxPoints) this.#points.shift();
      const elapsedIntervals = Math.floor((current.time - this.#origin + epsilon) / this.#interval);
      this.#nextSample = this.#origin + (elapsedIntervals + 1) * this.#interval;
      // Start the next interval at this endpoint, without changing the saved
      // frame. Same-time pause/resume calls must not rewrite the prior segment.
      this.#intervalActive = current.active;
    }
    return sampled;
  }

  snapshot() {
    return Object.freeze({ generation: this.#generation, interval: this.#interval,
      maxPoints: this.#maxPoints, maxEvents: this.#maxEvents,
      points: Object.freeze(this.#points.map(frozenFrame)), events: Object.freeze(this.#events.map(frozenFrame)),
      current: this.#current ? frozenFrame(this.#current) : null });
  }
}

/** Where the requested velocity would point if held, not a simulated body path.
 * `heading` is the controller's current desired heading (life.heading). World
 * coordinates are x/y horizontal and z up. Expiry uses simulation time only. */
export function projectCommand(snapshot, { horizon = PROJECTION_MAX_SECONDS, step = PATH_INTERVAL } = {}) {
  if (!finite(horizon) || horizon < 0 || !finite(step) || step <= 0) throw new RangeError("invalid_path_projection_options");
  if (!snapshot?.current) return EMPTY_GUIDE;
  let current;
  try { current = frameOf(snapshot.current); } catch { return EMPTY_GUIDE; }
  if (snapshot.generation !== current.generation || !current.active || current.deadline === null) return EMPTY_GUIDE;
  const duration = Math.min(horizon, PROJECTION_MAX_SECONDS, current.deadline - current.time);
  const { forwardSpeed: speed, yawRate, verticalSpeed } = current.command;
  if (duration <= 0 || (speed === 0 && verticalSpeed === 0)) return EMPTY_GUIDE;
  const segments = Math.min(PROJECTION_MAX_POINTS - 1, Math.max(1, Math.ceil(duration / step)));
  const points = [];
  for (let k = 0; k <= segments; k++) {
    const t = duration * k / segments, halfTurn = yawRate * t / 2;
    const square = halfTurn * halfTurn;
    const sinc = Math.abs(halfTurn) < 1e-4 ? 1 - square / 6 + square * square / 120 : Math.sin(halfTurn) / halfTurn;
    const distance = speed * t * sinc, angle = current.heading + halfTurn;
    const position = [current.position[0] + distance * Math.cos(angle),
      current.position[1] + distance * Math.sin(angle), current.position[2] + verticalSpeed * t];
    const heading = current.heading + yawRate * t;
    if (![...position, heading, current.time + t].every(finite)) return EMPTY_GUIDE;
    points.push(Object.freeze({ time: current.time + t, position: Object.freeze(position), heading }));
  }
  return Object.freeze(points);
}
