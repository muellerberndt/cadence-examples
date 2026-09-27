// The muscles: motor-neuron group activation onto the six wing controls. Twin of fruitfly/motor.py.
export const HOVER = { amplitude: 1.0, tilt: 0.0, shift: 0.0, frequency: 200.0 };
// each gain = the largest excursion the hand pilot used to recover from the same kicks
export const GAIN_AMPLITUDE = 0.02, GAIN_TILT = 0.227, GAIN_SHIFT = 0.149;
export const GROUPS = ["power", "mn:wing:b1", "mn:wing:b2", "mn:wing:b3", "mn:wing:i1", "mn:wing:i2", "mn:wing:iii1", "mn:wing:iii3", "mn:wing:iii4"];

export function wingControls(activation) {
  const g = (name) => activation[name] || 0;
  const out = {};
  for (const side of ["left", "right"]) {
    out[`amplitude_${side}`] = HOVER.amplitude + GAIN_AMPLITUDE * (g(`power:${side}`) + g(`mn:wing:b1:${side}`) + g(`mn:wing:b2:${side}`) - g(`mn:wing:b3:${side}`) - g(`mn:wing:i1:${side}`));
    out[`tilt_${side}`] = HOVER.tilt + GAIN_TILT * (g(`mn:wing:b2:${side}`) - g(`mn:wing:i2:${side}`) - 0.5 * g(`mn:wing:iii3:${side}`));
    out[`shift_${side}`] = HOVER.shift + GAIN_SHIFT * (g(`mn:wing:iii1:${side}`) - g(`mn:wing:iii3:${side}`) - g(`mn:wing:iii4:${side}`));
  }
  out.frequency = HOVER.frequency * (0.9 + 0.1 * (g("power:left") + g("power:right")));
  return out;
}

// Absolute settled motor activities only. The hover-assisted function above is a
// historical control; it must not be used in the connectome-only body path.
export const SETTLED_MOTOR_GROUPS = [
  ...GROUPS.flatMap(group => ["left", "right"].map(side => `${group}:${side}`)),
  "mn9", "mn:ttm:left", "mn:ttm:right",
];

export function settledMotor(readouts) {
  const activity = {};
  for (const name of SETTLED_MOTOR_GROUPS) {
    const value = Object.hasOwn(readouts, name) ? readouts[name] : 0;
    if (typeof value !== "number" || !Number.isFinite(value))
      throw new RangeError(`motor readout ${name} must be a finite number`);
    activity[name] = Math.min(1, Math.max(0, value));
  }
  const amplitudes = [], planes = [], shifts = [];
  for (const side of ["left", "right"]) {
    const power = activity[`power:${side}`];
    const [b1, b2, b3, i1, i2, iii1, iii3, iii4] =
      ["b1", "b2", "b3", "i1", "i2", "iii1", "iii3", "iii4"].map(name => activity[`mn:wing:${name}:${side}`]);
    amplitudes.push(Math.min(1.3, Math.max(0, Math.sqrt(power) * (1 + GAIN_AMPLITUDE * (b1 + b2 - b3 - i1)))));
    planes.push(Math.min(0.5, Math.max(-0.5, GAIN_TILT * (b2 - i2 - 0.5 * iii3))));
    shifts.push(Math.min(0.0005, Math.max(-0.0005, 1e-3 * GAIN_SHIFT * (iii1 - iii3 - iii4))));
  }
  // Candidate model, not fitted biology: activity is a normalized stroke-force
  // fraction. A supplied 200 Hz mechanical carrier is powered by motor activity;
  // a silent side has zero amplitude/force. No neural oscillator or hover offset.
  // With zero steering, equal power p produces p times body weight at level rest.
  const frequency = activity["power:left"] > 0 || activity["power:right"] > 0 ? 200 : 0;
  return {
    wings: [...amplitudes, ...planes, ...shifts, frequency],
    proboscis: activity.mn9,
    // Extensor activations, not a supplied gait or a jump/landing policy.
    legs: { ttm_left: activity["mn:ttm:left"], ttm_right: activity["mn:ttm:right"] },
  };
}
