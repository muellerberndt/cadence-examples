// The supplied ear in the browser: 68 numbers per channel, for the mix and for each voice.
//
// A port of hearing.py. The analysis window holds the last 2,646 samples, which is one 20 ms
// block and the 100 ms before it; one channel gives 64 log-power bands from 100 Hz to 10 kHz, the
// level of the block, how far it stands above the 100 ms before it, and a one-hot silence flag.
// Every analysed series is made mean-free first and its slope is removed with the mean, which is
// what keeps the emulator's power-on offset out of a decision.
//
// The transform here is a radix-2 FFT written out below rather than the one numpy calls, so the
// two agree to the arithmetic of doubles and not to the bit. The stage's tolerances are the bar:
// TOLERANCE 0.05 for a normalised value and BAND_TOLERANCE 0.02 for a band, against the 0.018 and
// 0.007 the emulator's own spread measures.

export const SAMPLE_RATE = 22050;
export const BLOCK_SAMPLES = 441;
export const HISTORY_SAMPLES = 2205;
export const WINDOW_SAMPLES = HISTORY_SAMPLES + BLOCK_SAMPLES;
export const RISE_SAMPLES = 1323;
export const FALL_SAMPLES = 221;
export const N_FFT = 4096;
export const BANDS = 64;
export const BAND_LOW_HZ = 100.0;
export const BAND_HIGH_HZ = 10000.0;
export const SPECTRUM_DB = [-70.0, 0.0];
export const RMS_DBFS = [-80.0, 0.0];
export const ONSET_DB = [0.0, 60.0];
export const SILENCE_DBFS = -50.0;
export const ONSET_THRESHOLD_DB = 6.0;
export const TOLERANCE = 0.05;
export const BAND_TOLERANCE = 0.02;
export const CHANNELS = ["mix", "voice1", "voice2", "voice3"];
export const FIELD_WIDTHS = { spectrum: BANDS, rms: 1, onset: 1, silence: 2 };
export const WIDTH = BANDS + 1 + 1 + 2; // 68
export const INDEX = { spectrum: [0, 64], rms: [64, 65], onset: [65, 66], silence: [66, 68] };

const POWER_FLOOR = Math.pow(10.0, RMS_DBFS[0] / 10.0);

/** A raised cosine rise, a flat middle and a short raised cosine fall. */
export function analysisWindow() {
  const window = new Float64Array(WINDOW_SAMPLES).fill(1);
  for (let i = 0; i < RISE_SAMPLES; i++) window[i] = 0.5 - 0.5 * Math.cos((Math.PI * (i + 1)) / (RISE_SAMPLES + 1));
  for (let k = 0; k < FALL_SAMPLES; k++) {
    window[WINDOW_SAMPLES - FALL_SAMPLES + k] = 0.5 - 0.5 * Math.cos((Math.PI * (FALL_SAMPLES - k)) / (FALL_SAMPLES + 1));
  }
  return window;
}

/** The band edges: geometric from 100 Hz to 10 kHz, the ends exact. */
export function bandEdges() {
  const count = BANDS + 2;
  const low = Math.log10(BAND_LOW_HZ), high = Math.log10(BAND_HIGH_HZ);
  const edges = new Float64Array(count);
  for (let i = 0; i < count; i++) edges[i] = Math.pow(10, low + ((high - low) * i) / (count - 1));
  edges[0] = BAND_LOW_HZ;
  edges[count - 1] = BAND_HIGH_HZ;
  return edges;
}

/** Triangles of peak one over the FFT bins, one per band, as a sparse list per band. */
export function filterbank() {
  const bins = N_FFT / 2 + 1;
  const edges = bandEdges();
  const bank = [];
  for (let band = 0; band < BANDS; band++) {
    const low = edges[band], centre = edges[band + 1], high = edges[band + 2];
    const index = [], weight = [];
    for (let k = 0; k < bins; k++) {
      const f = (k * SAMPLE_RATE) / N_FFT;
      let w = 0;
      if (f > low && f <= centre) w = (f - low) / (centre - low);
      else if (f > centre && f < high) w = (high - f) / (high - centre);
      if (w !== 0) { index.push(k); weight.push(w); }
    }
    bank.push({ index: Int32Array.from(index), weight: Float64Array.from(weight) });
  }
  return bank;
}

export const WINDOW = analysisWindow();
export const FILTERBANK = filterbank();
export const REFERENCE_POWER = (() => {
  let sum = 0;
  for (let i = 0; i < WINDOW.length; i++) sum += WINDOW[i] * WINDOW[i];
  return (N_FFT * sum) / 4.0;
})();

/** Remove the mean and the slope, which is where the chip's power-on offset lives. */
export function meanFree(values, start = 0, end = values.length) {
  const count = end - start;
  const out = new Float64Array(count);
  let mean = 0;
  for (let i = 0; i < count; i++) mean += values[start + i];
  mean /= count;
  if (count < 2) {
    for (let i = 0; i < count; i++) out[i] = values[start + i] - mean;
    return out;
  }
  const middle = (count - 1) / 2.0;
  let tt = 0, tc = 0;
  for (let i = 0; i < count; i++) {
    const t = i - middle;
    const c = values[start + i] - mean;
    out[i] = c;
    tt += t * t;
    tc += t * c;
  }
  const slope = tc / tt;
  for (let i = 0; i < count; i++) out[i] -= (i - middle) * slope;
  return out;
}

const normalize = (value, [low, high]) => Math.min(1, Math.max(0, (value - low) / (high - low)));

// ---------------------------------------------------------------------------- the transform
const BITREV = (() => {
  const table = new Int32Array(N_FFT);
  const bits = Math.log2(N_FFT);
  for (let i = 0; i < N_FFT; i++) {
    let x = i, r = 0;
    for (let b = 0; b < bits; b++) { r = (r << 1) | (x & 1); x >>= 1; }
    table[i] = r;
  }
  return table;
})();
const TWIDDLE_COS = new Float64Array(N_FFT / 2);
const TWIDDLE_SIN = new Float64Array(N_FFT / 2);
for (let i = 0; i < N_FFT / 2; i++) {
  TWIDDLE_COS[i] = Math.cos((-2 * Math.PI * i) / N_FFT);
  TWIDDLE_SIN[i] = Math.sin((-2 * Math.PI * i) / N_FFT);
}

/** |rfft(x, N_FFT)|^2 over the N_FFT/2 + 1 bins, x zero-padded. Radix 2, in place. */
export function powerSpectrum(values) {
  const re = new Float64Array(N_FFT), im = new Float64Array(N_FFT);
  // the bit-reversed permutation of the zero-padded input
  for (let i = 0; i < N_FFT; i++) {
    const j = BITREV[i];
    re[i] = j < values.length ? values[j] : 0;
    im[i] = 0;
  }
  for (let size = 2; size <= N_FFT; size <<= 1) {
    const half = size >> 1, step = N_FFT / size;
    for (let i = 0; i < N_FFT; i += size) {
      for (let k = 0; k < half; k++) {
        const c = TWIDDLE_COS[k * step], s = TWIDDLE_SIN[k * step];
        const a = i + k, b = a + half;
        const tr = re[b] * c - im[b] * s;
        const ti = re[b] * s + im[b] * c;
        re[b] = re[a] - tr;
        im[b] = im[a] - ti;
        re[a] += tr;
        im[a] += ti;
      }
    }
  }
  const bins = N_FFT / 2 + 1;
  const power = new Float64Array(bins);
  for (let k = 0; k < bins; k++) power[k] = re[k] * re[k] + im[k] * im[k];
  return power;
}

// ---------------------------------------------------------------------------- the features
/** The 68 features of one analysis window of one channel, as a flat Float64Array. */
export function features(window) {
  if (window.length !== WINDOW_SAMPLES) throw Error(`the audio holds ${window.length} samples where ${WINDOW_SAMPLES} are expected`);
  const whole = meanFree(window);
  const block = meanFree(window, WINDOW_SAMPLES - BLOCK_SAMPLES, WINDOW_SAMPLES);
  const before = meanFree(window, 0, HISTORY_SAMPLES);

  const windowed = new Float64Array(WINDOW_SAMPLES);
  for (let i = 0; i < WINDOW_SAMPLES; i++) windowed[i] = whole[i] * WINDOW[i];
  const spectrum = powerSpectrum(windowed);

  const out = new Float64Array(WIDTH);
  for (let band = 0; band < BANDS; band++) {
    const { index, weight } = FILTERBANK[band];
    let sum = 0;
    for (let k = 0; k < index.length; k++) sum += weight[k] * spectrum[index[k]];
    const decibels = 10.0 * Math.log10(sum / REFERENCE_POWER + 1e-30);
    out[band] = normalize(decibels, SPECTRUM_DB);
  }
  let blockPower = 0, beforePower = 0;
  for (let i = 0; i < block.length; i++) blockPower += block[i] * block[i];
  blockPower /= block.length;
  for (let i = 0; i < before.length; i++) beforePower += before[i] * before[i];
  beforePower /= before.length;
  const level = 10.0 * Math.log10(blockPower + 1e-30);
  const onset = 10.0 * Math.log10((blockPower + POWER_FLOOR) / (beforePower + POWER_FLOOR));
  out[INDEX.rms[0]] = normalize(level, RMS_DBFS);
  out[INDEX.onset[0]] = normalize(onset, ONSET_DB);
  const silent = level < SILENCE_DBFS;
  out[INDEX.silence[0]] = silent ? 0.0 : 1.0;
  out[INDEX.silence[0] + 1] = silent ? 1.0 : 0.0;
  return out;
}

/** The whole hearing of one block: the mix, and each voice on its own. */
export function listen(mix, voices) {
  if (voices.length !== 3) throw Error("three voice windows of the analysis length are expected");
  return { mix: features(mix), voices: voices.map((voice) => features(voice)) };
}

export function isSilent(vector) { return vector[INDEX.silence[0] + 1] > vector[INDEX.silence[0]]; }

export function hasOnset(vector, thresholdDb = ONSET_THRESHOLD_DB) {
  return vector[INDEX.onset[0]] * (ONSET_DB[1] - ONSET_DB[0]) + ONSET_DB[0] >= thresholdDb;
}

export function levelDbfs(vector) { return vector[INDEX.rms[0]] * (RMS_DBFS[1] - RMS_DBFS[0]) + RMS_DBFS[0]; }

/** The root mean square difference of the 64 normalised band values. */
export function spectralDistance(heard, target) {
  let sum = 0;
  for (let band = 0; band < BANDS; band++) {
    const d = heard[band] - target[band];
    sum += d * d;
  }
  return Math.sqrt(sum / BANDS);
}

// ---------------------------------------------------------------------------- the fine ear
const FINE_BANKS = new Map();

/** A geometric band spectrum of one window at the resolution transcription needs (FineEar). */
export function fineSpectrum(window, bands = 128) {
  let bank = FINE_BANKS.get(bands);
  if (!bank) {
    const count = bands + 2;
    const low = Math.log10(BAND_LOW_HZ), high = Math.log10(BAND_HIGH_HZ);
    const edges = new Float64Array(count);
    for (let i = 0; i < count; i++) edges[i] = Math.pow(10, low + ((high - low) * i) / (count - 1));
    edges[0] = BAND_LOW_HZ;
    edges[count - 1] = BAND_HIGH_HZ;
    bank = [];
    const bins = N_FFT / 2 + 1;
    for (let band = 0; band < bands; band++) {
      const lo = edges[band], centre = edges[band + 1], hi = edges[band + 2];
      const index = [], weight = [];
      for (let k = 0; k < bins; k++) {
        const f = (k * SAMPLE_RATE) / N_FFT;
        let w = 0;
        if (f > lo && f <= centre) w = (f - lo) / (centre - lo);
        else if (f > centre && f < hi) w = (hi - f) / (hi - centre);
        if (w !== 0) { index.push(k); weight.push(w); }
      }
      bank.push({ index: Int32Array.from(index), weight: Float64Array.from(weight) });
    }
    FINE_BANKS.set(bands, bank);
  }
  const whole = meanFree(window.length > WINDOW_SAMPLES ? window.subarray(window.length - WINDOW_SAMPLES) : window);
  const windowed = new Float64Array(WINDOW_SAMPLES);
  for (let i = 0; i < WINDOW_SAMPLES; i++) windowed[i] = whole[i] * WINDOW[i];
  const spectrum = powerSpectrum(windowed);
  const out = new Float64Array(bands);
  for (let band = 0; band < bands; band++) {
    const { index, weight } = bank[band];
    let sum = 0;
    for (let k = 0; k < index.length; k++) sum += weight[k] * spectrum[index[k]];
    const decibels = 10.0 * Math.log10(sum / REFERENCE_POWER + 1e-30);
    out[band] = normalize(decibels, SPECTRUM_DB);
  }
  return out;
}

/** int16 samples as the stage converts them: divided by 32,768, which is exact. */
export function fromInt16(samples) {
  const out = new Float64Array(samples.length);
  for (let i = 0; i < samples.length; i++) out[i] = samples[i] / 32768.0;
  return out;
}
