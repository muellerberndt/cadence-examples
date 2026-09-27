// Supplied motion-adapter candidates, not fitted biology or neural outputs.
// The strict controller does not use them unless motionSupport is opted in.
export const MOTION_SUPPORT_DEFAULTS = Object.freeze({
  cruiseSpeed: .3, maxObservationAgeSeconds: 4, holdSeconds: 8,
  responseSeconds: .25, maxAcceleration: 1,
});
export const MOTION_SUPPORT_EPSILON = 1e-8;

export function motionSupportOptions(value = false) {
  if (value === false || value === undefined) return null;
  if (value !== true && (!value || typeof value !== "object" || Array.isArray(value))) throw new TypeError("invalid motion support options");
  const supplied = value === true ? {} : value;
  if (Object.keys(supplied).some(k => !Object.hasOwn(MOTION_SUPPORT_DEFAULTS, k))) throw new TypeError("unknown motion support option");
  const p = { ...MOTION_SUPPORT_DEFAULTS, ...supplied };
  if (!Object.values(p).every(x => Number.isFinite(x) && x > 0) || p.cruiseSpeed > 1
    || p.maxObservationAgeSeconds < 1 || p.maxObservationAgeSeconds > 10 || p.holdSeconds > 15
    || p.responseSeconds > 5 || p.maxAcceleration > 10) throw new RangeError("invalid motion support bounds");
  return Object.freeze(p);
}

/** One exact exponential response followed by a vector acceleration bound.
 * Inputs are world velocities only: no room, target, body pose or destination.
 * A numerical snap below 1e-8 m/s makes a stopped target exactly representable. */
export function smoothVelocity(current, desired, dt, config) {
  if (!Number.isFinite(dt) || dt <= 0 || current.length !== 3 || desired.length !== 3
    || ![...current, ...desired].every(Number.isFinite)) throw new RangeError("invalid motion velocity");
  const difference = desired.map((x, i) => x - current[i]), distance = Math.hypot(...difference);
  if (distance === 0) return desired.slice();
  const step = Math.min(distance * -Math.expm1(-dt / config.responseSeconds), config.maxAcceleration * dt);
  if (distance <= MOTION_SUPPORT_EPSILON && distance <= config.maxAcceleration * dt) return desired.slice();
  return current.map((x, i) => x + difference[i] * step / distance);
}
