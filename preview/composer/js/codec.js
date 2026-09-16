// The S06 codec in the browser: 28 slot indices to the register writes the chip runs.
//
// A port of codec.py, field for field and table for table. The bin tables are the same exact
// tables: pitch p is MIDI p + 11 at equal temperament, pulse width w is the 12 bit value
// 256 w + 128, the eight envelope bins are the chip nibbles (0, 1, 2, 3, 5, 8, 11, 15), cutoff c
// is the 11 bit value 66 c, resonance r is the nibble 2 r, routing is the three filter bits and
// volume is the low nibble of 0x18 with the filter mode fixed to low-pass.
//
// The frequency register of a pitch divides by the clock the resampler realizes, which the
// nominal clock misses by about 56 ppm at 22,050 Hz. That number is not derived here: the
// caller passes `cyclesPerSecond` from the stage, and the parity harness checks all 97 registers
// against the Python codec's.

export const VOICES = 3;
export const VOICE_FIELDS = ["pitch", "pulse_width", "waveform", "gate", "attack", "decay", "sustain", "release"];
export const VOICE_SLOTS = [97, 16, 4, 2, 8, 8, 16, 8];
export const SHARED_FIELDS = ["cutoff", "resonance", "routing", "volume"];
export const SHARED_SLOTS = [32, 8, 8, 16];
export const FIELDS = [];
for (let v = 0; v < VOICES; v++) for (const name of VOICE_FIELDS) FIELDS.push(`voice${v + 1}_${name}`);
for (const name of SHARED_FIELDS) FIELDS.push(name);
export const SLOTS = [];
for (let v = 0; v < VOICES; v++) SLOTS.push(...VOICE_SLOTS);
SLOTS.push(...SHARED_SLOTS);
export const FIELD_COUNT = SLOTS.length; // 28
export const MOTOR_NEURONS = SLOTS.reduce((a, b) => a + b, 0); // 541
export const SLOT_BOUNDS = [];
{
  let at = 0;
  for (const width of SLOTS) { SLOT_BOUNDS.push([at, at + width]); at += width; }
}

export const REST = 0;
export const PITCH_MIDI_BASE = 11;
export const PITCHES = VOICE_SLOTS[0];
export const MAX_FREQUENCY_REGISTER = 0xffff;
export const ACCUMULATOR_BITS = 24;

// the chip's waveform bits, in the codec's order
export const TRIANGLE_BIT = 0x10, SAWTOOTH_BIT = 0x20, PULSE_BIT = 0x40, NOISE_BIT = 0x80, GATE_BIT = 0x01;
export const WAVEFORM_BITS = [TRIANGLE_BIT, SAWTOOTH_BIT, PULSE_BIT, NOISE_BIT];
export const ENVELOPE_NIBBLES = [0, 1, 2, 3, 5, 8, 11, 15];
export const PULSE_WIDTH_STEP = 256, PULSE_WIDTH_OFFSET = 128;
export const CUTOFF_STEP = 66, RESONANCE_STEP = 2, FILTER_MODE = 0x10;
export const FILTER_CUTOFF_LOW = 0x15, FILTER_CUTOFF_HIGH = 0x16, FILTER_RESONANCE_ROUTING = 0x17, VOLUME_MODE = 0x18;
export const REGISTER_COUNT = 25;

export function fieldIndex(voice, name) {
  const at = VOICE_FIELDS.indexOf(name);
  if (at < 0) throw Error(`a voice has no field ${name}`);
  return voice * VOICE_FIELDS.length + at;
}

export function sharedIndex(name) {
  const at = SHARED_FIELDS.indexOf(name);
  if (at < 0) throw Error(`the shared fields have no ${name}`);
  return VOICES * VOICE_FIELDS.length + at;
}

/** A gesture: 28 integers, each inside its slot. */
export function validate(gesture) {
  const values = Int32Array.from(gesture);
  if (values.length !== FIELD_COUNT) throw Error(`a gesture is ${FIELD_COUNT} integer fields`);
  for (let f = 0; f < FIELD_COUNT; f++) {
    if (!Number.isInteger(values[f]) || values[f] < 0 || values[f] >= SLOTS[f]) {
      throw Error(`a gesture field falls outside its slot: field ${f} is ${values[f]}`);
    }
  }
  return values;
}

/** Every voice resting, the filter open, the volume at the chip's maximum. */
export function silence() {
  const gesture = new Int32Array(FIELD_COUNT);
  for (let v = 0; v < VOICES; v++) {
    gesture[fieldIndex(v, "pulse_width")] = 8;
    gesture[fieldIndex(v, "waveform")] = 2;
    gesture[fieldIndex(v, "sustain")] = 15;
  }
  gesture[sharedIndex("cutoff")] = SHARED_SLOTS[0] - 1;
  gesture[sharedIndex("volume")] = SHARED_SLOTS[3] - 1;
  return gesture;
}

/** The gesture the instrument runs: a rest holds its voice's gate down. */
export function executed(gesture) {
  const values = validate(gesture);
  for (let v = 0; v < VOICES; v++) {
    if (values[fieldIndex(v, "pitch")] === REST) values[fieldIndex(v, "gate")] = 0;
  }
  return values;
}

/** Whether a voice's sound passes through the filter, which is the routing bit of that voice. */
export function routed(gesture, voice) {
  const values = validate(gesture);
  return ((values[sharedIndex("routing")] >> voice) & 1) === 1;
}

export function hertz(pitch, { concertAHz = 440.0, tuningCents = 0.0 } = {}) {
  return concertAHz * Math.pow(2.0, (pitch + PITCH_MIDI_BASE - 69) / 12.0 + tuningCents / 1200.0);
}

/** The 97 frequency registers of the pitches, pitch 0 (the rest) reading zero. */
export function frequencyTable(cyclesPerSecond, options = {}) {
  const table = new Int32Array(PITCHES);
  for (let pitch = 1; pitch < PITCHES; pitch++) {
    const value = hertz(pitch, options) * Math.pow(2, ACCUMULATOR_BITS) / cyclesPerSecond;
    table[pitch] = roundHalfEven(value);
  }
  return table;
}

/** Python's round(): half to even, which is what the register table was built with. */
export function roundHalfEven(value) {
  const floor = Math.floor(value);
  const rest = value - floor;
  if (rest > 0.5) return floor + 1;
  if (rest < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
}

/** Over the 541 motor neurons: every value of every field except a pitch the register cannot hold. */
export function legalMask(frequencies) {
  const mask = new Uint8Array(MOTOR_NEURONS).fill(1);
  for (let v = 0; v < VOICES; v++) {
    const [start] = SLOT_BOUNDS[fieldIndex(v, "pitch")];
    for (let pitch = 0; pitch < PITCHES; pitch++) {
      const reaches = pitch === REST || (frequencies[pitch] >= 1 && frequencies[pitch] <= MAX_FREQUENCY_REGISTER);
      if (!reaches) mask[start + pitch] = 0;
    }
  }
  return mask;
}

export class Codec {
  /** @param cyclesPerSecond the clock the resampler realizes, from the stage. */
  constructor(cyclesPerSecond, options = {}) {
    this.cyclesPerSecond = cyclesPerSecond;
    this.options = { concertAHz: 440.0, tuningCents: 0.0, ...options };
    this.frequencies = frequencyTable(cyclesPerSecond, this.options);
    this.mask = legalMask(this.frequencies);
  }

  legalMask() { return this.mask; }

  /** The whole register image of a gesture played from silence, in write order. */
  registers(gesture) { return this.writes(gesture, null).writes; }

  /**
   * The register writes of a gesture, the gesture as executed, and the new image.
   *
   * Only registers whose value changes are written. The control register of a voice is written
   * after its frequency, its pulse width and its envelope, so a gate that rises finds the note it
   * is meant to start. A rest leaves the frequency register alone, so the note that is sounding
   * releases at its own pitch.
   */
  writes(gesture, image) {
    const values = executed(gesture);
    const known = new Map(image ? (image instanceof Map ? image : Object.entries(image).map(([k, v]) => [Number(k), Number(v)])) : []);
    const wanted = new Map();
    const order = [];
    const want = (register, value) => { wanted.set(register, value); order.push(register); };

    const cutoff = CUTOFF_STEP * values[sharedIndex("cutoff")];
    const resonance = RESONANCE_STEP * values[sharedIndex("resonance")];
    const routing = values[sharedIndex("routing")];
    const volume = values[sharedIndex("volume")];
    want(FILTER_CUTOFF_LOW, cutoff & 0x07);
    want(FILTER_CUTOFF_HIGH, cutoff >> 3);
    want(FILTER_RESONANCE_ROUTING, (resonance << 4) | routing);
    want(VOLUME_MODE, volume | FILTER_MODE);

    for (let voice = 0; voice < VOICES; voice++) {
      const base = 7 * voice;
      const pitch = values[fieldIndex(voice, "pitch")];
      const width = PULSE_WIDTH_STEP * values[fieldIndex(voice, "pulse_width")] + PULSE_WIDTH_OFFSET;
      const waveform = WAVEFORM_BITS[values[fieldIndex(voice, "waveform")]];
      const gate = values[fieldIndex(voice, "gate")];
      const attack = ENVELOPE_NIBBLES[values[fieldIndex(voice, "attack")]];
      const decay = ENVELOPE_NIBBLES[values[fieldIndex(voice, "decay")]];
      const sustain = values[fieldIndex(voice, "sustain")];
      const release = ENVELOPE_NIBBLES[values[fieldIndex(voice, "release")]];
      if (pitch !== REST) {
        const frequency = this.frequencies[pitch];
        if (!(frequency >= 1 && frequency <= MAX_FREQUENCY_REGISTER)) {
          throw Error(`pitch ${pitch} falls outside the chip's frequency register`);
        }
        want(base + 0, frequency & 0xff);
        want(base + 1, frequency >> 8);
      }
      want(base + 2, width & 0xff);
      want(base + 3, width >> 8);
      want(base + 5, (attack << 4) | decay);
      want(base + 6, (sustain << 4) | release);
      want(base + 4, waveform | (gate ? GATE_BIT : 0));
    }

    const writes = [];
    for (const register of order) {
      const value = wanted.get(register);
      if (known.get(register) !== value) writes.push([register, value]);
    }
    for (const [register, value] of wanted) known.set(register, value);
    return { writes, executed: values, image: known };
  }
}

/** The one-hot of a gesture over the 541 motor neurons. */
export function oneHot(gesture) {
  const values = validate(gesture);
  const out = new Float64Array(MOTOR_NEURONS);
  for (let f = 0; f < FIELD_COUNT; f++) out[SLOT_BOUNDS[f][0] + values[f]] = 1;
  return out;
}

/** The gesture a motor layer names: the largest value of each field's slot. */
export function fromOneHot(vector) {
  if (vector.length !== MOTOR_NEURONS) throw Error(`the motor layer holds ${MOTOR_NEURONS} neurons`);
  const out = new Int32Array(FIELD_COUNT);
  for (let f = 0; f < FIELD_COUNT; f++) {
    const [start, stop] = SLOT_BOUNDS[f];
    let best = start, top = -Infinity;
    for (let i = start; i < stop; i++) if (vector[i] > top) { top = vector[i]; best = i; }
    out[f] = best - start;
  }
  return out;
}

/** One mixed-radix integer over the 28 fields, row major. Exact, and about 2.6e78 wide. */
export function joint(gesture) {
  const values = validate(gesture);
  let out = 0n;
  for (let f = 0; f < FIELD_COUNT; f++) out = out * BigInt(SLOTS[f]) + BigInt(values[f]);
  return out;
}

export function fromJoint(index) {
  let rest = BigInt(index);
  const out = new Int32Array(FIELD_COUNT);
  for (let f = FIELD_COUNT - 1; f >= 0; f--) {
    const width = BigInt(SLOTS[f]);
    out[f] = Number(rest % width);
    rest /= width;
  }
  if (rest !== 0n) throw Error("the joint index falls outside the gesture space");
  return out;
}
