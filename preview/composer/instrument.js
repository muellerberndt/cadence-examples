// The stage's instrument in the browser: four chips, the mix and one monitor a voice.
//
// A port of `s05_sid.instrument.ThreeVoiceInstrument` over the WebAssembly build of the pinned
// reSIDfp. The mix chip takes every register the codec writes; a monitor chip takes the shared
// registers and its own voice's, with the other two voices muted, because muting alone does not
// isolate a voice on this emulator: the binding's mute holds the waveform down and leaves the
// voice's direct term, which the envelope still scales.
//
// The audio is the chip's int16 divided by 32,768, which is exact and cannot clip. The block
// clock is the stage's: a block is 20 ms of the PAL clock, carried as a fraction so the cycles
// never drift.

import * as ear from "./js/hearing.js";

export const MIX = 0;
export const MONITOR = [1, 2, 3];
export const CLOCK_HZ = 985248.0; // PAL
export const SAMPLE_RATE = 22050;
export const BLOCK_MS = 20;
export const MODEL_8580 = 2;
export const METHOD_RESAMPLE = 2;
export const SHARED_REGISTERS = [0x15, 0x16, 0x17, 0x18];
const FULL_SCALE = 32768.0;

export class Instrument {
  /** @param module the loaded sid.mjs module. */
  constructor(module, { clockHz = CLOCK_HZ, sampleRate = SAMPLE_RATE, blockMs = BLOCK_MS } = {}) {
    this.module = module;
    this.clockHz = clockHz;
    this.sampleRate = sampleRate;
    this.blockMs = blockMs;
    this.cap = 8192;
    this.pointer = module._malloc(this.cap * 2);
    this.carry = 0;
    this.blocks = 0;
    this.history = [MIX, ...MONITOR].map(() => new Float64Array(ear.WINDOW_SAMPLES));
    this.image = new Map(); // the registers each chip has been given
    this.images = [new Map(), new Map(), new Map(), new Map()];
  }

  static async open(url, options = {}) {
    const factory = (await import(url)).default;
    const module = await factory();
    return new Instrument(module, options);
  }

  /** Four fresh chips, the monitors with their other voices muted. */
  reset() {
    const m = this.module;
    for (const handle of [MIX, ...MONITOR]) {
      if (!m._sid_open(handle, MODEL_8580, METHOD_RESAMPLE, this.clockHz, this.sampleRate)) {
        throw Error(`the chip did not open: ${m.UTF8ToString(m._sid_last_error())}`);
      }
      m._sid_enable_filter_at(handle, 1);
    }
    for (let voice = 0; voice < 3; voice++) {
      const handle = MONITOR[voice];
      for (let other = 0; other < 3; other++) if (other !== voice) m._sid_mute_at(handle, other, 1);
    }
    this.carry = 0;
    this.blocks = 0;
    this.images = [new Map(), new Map(), new Map(), new Map()];
    for (const row of this.history) row.fill(0);
  }

  /** Which chips a register reaches: the shared registers reach every chip, a voice's its own. */
  _handlesOf(register) {
    if (SHARED_REGISTERS.includes(register)) return [MIX, ...MONITOR];
    const voice = Math.floor(register / 7);
    return voice < 3 ? [MIX, MONITOR[voice]] : [MIX];
  }

  /** One block: the writes, the clock, and what each channel emitted. `added` is mixed in. */
  play(writes, added = null) {
    const m = this.module;
    for (const [register, value] of writes) {
      for (const handle of this._handlesOf(register)) {
        const image = this.images[handle];
        if (image.get(register) === value) continue;
        image.set(register, value);
        m._sid_write_at(handle, register, value);
      }
    }
    this.carry += (this.clockHz * this.blockMs) / 1000;
    const cycles = Math.floor(this.carry);
    this.carry -= cycles;
    const out = [];
    for (const handle of [MIX, ...MONITOR]) {
      const n = m._sid_clock_at(handle, cycles, this.pointer, this.cap);
      if (n < 0) throw Error(`the chip threw: ${m.UTF8ToString(m._sid_last_error())}`);
      const samples = new Float64Array(Math.min(n, this.cap));
      const heap = m.HEAP16.subarray(this.pointer >> 1, (this.pointer >> 1) + samples.length);
      for (let i = 0; i < samples.length; i++) samples[i] = heap[i] / FULL_SCALE;
      out.push(samples);
    }
    if (added) for (let i = 0; i < out[MIX].length && i < added.length; i++) out[MIX][i] += added[i];
    for (let channel = 0; channel < out.length; channel++) this._append(channel, out[channel]);
    this.blocks++;
    return { mix: out[MIX], voices: [out[1], out[2], out[3]], cycles };
  }

  _append(channel, samples) {
    const row = this.history[channel];
    const keep = row.length - samples.length;
    row.copyWithin(0, samples.length);
    for (let i = 0; i < samples.length && i < row.length; i++) row[keep + i] = samples[i];
  }

  /** The analysis window of one channel: the last 2,646 samples. */
  window(channel) { return this.history[channel]; }

  /** The whole hearing of the last block: the mix, and each voice on its own, as (4, 68). */
  hear() {
    const out = new Float64Array(4 * ear.WIDTH);
    for (let channel = 0; channel < 4; channel++) out.set(ear.features(this.history[channel]), channel * ear.WIDTH);
    return out;
  }

  /** The fine spectrum of the mix, at the resolution the pitch reader wants. */
  fine(bands = 128) { return ear.fineSpectrum(this.history[MIX], bands); }
}
