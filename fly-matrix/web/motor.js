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
