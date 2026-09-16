// The composer's search in the browser: the notes a spectrum carries, and the gesture to play.
//
// A port of `PitchReader` and `GesturePlanner` as round six leaves them (NOTES.md §25): the
// proposal grid is the codec's, a one-semitone grid around the note the reader named, and the cost
// is one of three, switched by the same config keys as the Python:
//
//   monitor_notes  the note the candidate's own imagined monitor would carry (config_monitor.json)
//   synthesis      the mix the model imagines of the candidate  (config_synthesis.json)
//   otherwise      the note the gesture itself names, the grid cost (config.json)
//
// Nothing here renders and nothing here settles: the costs read `S06Brain.imagine`, which reads
// the records cortex.

import { BANDS, HEARD, HEARING_WIDTH, decodeArray, harden } from "./s06_brain.js";
import * as cdc from "./codec.js";

const ONSET_DB = [0.0, 60.0];
const ONSET_THRESHOLD_DB = 6.0;
const REST = cdc.REST;

export class NoteTarget {
  constructor(pitches, silent, onset) {
    this.pitches = pitches;
    this.silent = !!silent;
    this.onset = !!onset;
  }
  sounding() { return this.pitches.filter((p) => p !== null).length; }
}

export class PitchReader {
  /** @param spec the fixture's reader block: the pitches it considers and its combs. */
  constructor(spec) {
    this.pitchFloor = spec.pitch_floor | 0;
    this.harmonics = spec.harmonics | 0;
    this.leastShare = spec.least_share;
    this.mostNotes = spec.most_notes | 0;
    this.bands = spec.bands | 0;
    this.shortlist = spec.shortlist | 0;
    this.joint = !!spec.joint;
    this.pitches = Int32Array.from(spec.pitches);
    this.combs = {};
    for (const [width, comb] of Object.entries(spec.combs)) {
      this.combs[Number(width)] = decodeArray(comb); // (pitches, width) row major
    }
  }

  /** The comb scores of one spectrum: one per pitch. */
  _scores(values) {
    const comb = this.combs[values.length];
    if (!comb) return null;
    const rows = this.pitches.length, width = values.length;
    const out = new Float64Array(rows);
    for (let r = 0; r < rows; r++) {
      let sum = 0;
      const at = r * width;
      for (let k = 0; k < width; k++) sum += comb[at + k] * values[k];
      out[r] = sum;
    }
    return out;
  }

  /** The one note each spectrum carries, by the comb: the read that scores a candidate. */
  bestPitches(spectra) {
    const out = new Int32Array(spectra.length);
    for (let b = 0; b < spectra.length; b++) {
      const scores = this._scores(spectra[b]);
      if (!scores) { out[b] = -1; continue; }
      let best = 0;
      for (let r = 1; r < scores.length; r++) if (scores[r] > scores[best]) best = r;
      out[b] = this.pitches[best];
    }
    return out;
  }

  /** The pitches the comb finds in one spectrum: the greedy peel, or the joint fit. */
  notes(spectrum, want = null) {
    const values = spectrum;
    const comb = this.combs[values.length];
    if (!comb) return [];
    const width = values.length;
    let count = want === null ? this.mostNotes : want;
    count = Math.max(1, Math.min(count, this.mostNotes));
    const single = this._scores(values);
    if (!this.joint) {
      const left = Float64Array.from(values);
      const out = [];
      let first = 0;
      for (let index = 0; index < count; index++) {
        const score = this._scores(left);
        let row = 0;
        for (let r = 1; r < score.length; r++) if (score[r] > score[row]) row = r;
        const best = score[row];
        if (index === 0) first = best;
        else if (first <= 0 || best < this.leastShare * first) break;
        out.push(this.pitches[row]);
        const at = row * width;
        for (let k = 0; k < width; k++) left[k] = Math.max(left[k] - best * comb[at + k] * width, 0);
      }
      return out;
    }
    // the joint fit: the combination of fundamentals whose summed combs leave the least residual
    const order = Array.from(single.keys()).sort((a, b) => single[b] - single[a]);
    const top = order.slice(0, Math.max(count, this.shortlist));
    let whole = 0;
    for (let k = 0; k < width; k++) whole += values[k] * values[k];
    let chosen = [this.pitches[top[0]]], previous = whole;
    for (let size = 1; size <= count; size++) {
      let best = null, least = Infinity;
      for (const combination of combinations(top, size)) {
        const residual = this._residual(combination, values, width, comb);
        if (residual < least) { best = combination; least = residual; }
      }
      if (best === null) break;
      if (size > 1 && least > (1.0 - this.leastShare) * previous) break;
      chosen = best.map((index) => this.pitches[index]);
      previous = least;
    }
    return chosen;
  }

  /** The residual of the non-negative least squares of these combs against the spectrum. */
  _residual(rows, values, width, comb) {
    const order = rows.length;
    // the normal equations of A x = b with A = comb[rows].T, which is (width, order)
    const ata = new Float64Array(order * order), atb = new Float64Array(order);
    for (let i = 0; i < order; i++) {
      const ai = rows[i] * width;
      for (let j = 0; j < order; j++) {
        const aj = rows[j] * width;
        let sum = 0;
        for (let k = 0; k < width; k++) sum += comb[ai + k] * comb[aj + k];
        ata[i * order + j] = sum;
      }
      let sum = 0;
      for (let k = 0; k < width; k++) sum += comb[ai + k] * values[k];
      atb[i] = sum;
    }
    const weights = solve(ata, atb, order);
    let residual = 0;
    for (let k = 0; k < width; k++) {
      let fit = 0;
      for (let i = 0; i < order; i++) {
        const w = weights[i] > 0 ? weights[i] : 0; // np.maximum(weights, 0.0)
        fit += w * comb[rows[i] * width + k];
      }
      const d = values[k] - fit;
      residual += d * d;
    }
    return residual;
  }

  /** A demonstration heard voice by voice: each pass names one voice's note and nothing else. */
  readVoices(rows, fines) {
    const pitches = [], silent = [];
    let onset = false;
    for (let voice = 0; voice < cdc.VOICES; voice++) {
      const row = rows[voice];
      const quiet = row[BANDS + 3] > row[BANDS + 2];
      const onsetDb = row[BANDS + 1] * (ONSET_DB[1] - ONSET_DB[0]) + ONSET_DB[0];
      onset = onset || (!quiet && onsetDb >= ONSET_THRESHOLD_DB);
      if (quiet) { pitches.push(null); silent.push(true); continue; }
      const fine = fines ? fines[voice] : null;
      const spectrum = fine === null || fine === undefined ? row.subarray(0, BANDS) : fine;
      const found = this.notes(spectrum, 1);
      pitches.push(found.length ? found[0] : null);
      silent.push(!found.length);
    }
    return new NoteTarget([pitches[0], pitches[1], pitches[2]], silent.every(Boolean), onset);
  }

  /** One block of a heard phrase, as a note per voice, beside what the body is playing. */
  read(target, playing = [null, null, null], fine = null, want = null) {
    const row = target;
    const silent = row[BANDS + 3] > row[BANDS + 2];
    const onsetDb = row[BANDS + 1] * (ONSET_DB[1] - ONSET_DB[0]) + ONSET_DB[0];
    const onset = onsetDb >= ONSET_THRESHOLD_DB;
    if (silent) return new NoteTarget([null, null, null], true, onset);
    const spectrum = fine === null ? row.subarray(0, BANDS) : fine;
    const found = this.notes(spectrum, want).slice().sort((a, b) => a - b);
    if (!found.length) return new NoteTarget([null, null, null], true, onset);
    return new NoteTarget(this.assign(found, playing), false, onset);
  }

  /** Which voice carries which of the notes found, by what is playing and by the corpus's order. */
  assign(found, playing = [null, null, null], tolerance = 1) {
    const places = [null, null, null];
    const left = found.slice();
    const taken = new Set();
    for (let voice = 0; voice < playing.length; voice++) {
      const note = playing[voice];
      if (note === null || note === undefined || !left.length) continue;
      let near = left[0];
      for (const p of left) if (Math.abs(p - note) < Math.abs(near - note)) near = p;
      if (Math.abs(near - note) <= tolerance) {
        places[voice] = near;
        left.splice(left.indexOf(near), 1);
        taken.add(voice);
      }
    }
    const orders = { 1: [0], 2: [0, 2], 3: [0, 2, 1] };
    const order = orders[found.length] || [0, 2, 1];
    const free = order.filter((v) => !taken.has(v));
    for (const note of left.slice().sort((a, b) => b - a)) {
      if (!free.length) break;
      places[free.shift()] = note;
    }
    return places;
  }
}

/** Every combination of `size` of the values, in the order itertools.combinations gives them. */
export function* combinations(values, size) {
  const n = values.length;
  if (size > n) return;
  const index = Array.from({ length: size }, (_, i) => i);
  while (true) {
    yield index.map((i) => values[i]);
    let i = size - 1;
    while (i >= 0 && index[i] === i + n - size) i--;
    if (i < 0) return;
    index[i]++;
    for (let j = i + 1; j < size; j++) index[j] = index[j - 1] + 1;
  }
}

/** A small dense solve by Gaussian elimination with partial pivoting. */
export function solve(a, b, n) {
  const m = Float64Array.from(a), x = Float64Array.from(b);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++) if (Math.abs(m[row * n + col]) > Math.abs(m[pivot * n + col])) pivot = row;
    if (pivot !== col) {
      for (let k = 0; k < n; k++) { const t = m[col * n + k]; m[col * n + k] = m[pivot * n + k]; m[pivot * n + k] = t; }
      const t = x[col]; x[col] = x[pivot]; x[pivot] = t;
    }
    const head = m[col * n + col];
    if (Math.abs(head) < 1e-300) continue;
    for (let row = col + 1; row < n; row++) {
      const factor = m[row * n + col] / head;
      if (factor === 0) continue;
      for (let k = col; k < n; k++) m[row * n + k] -= factor * m[col * n + k];
      x[row] -= factor * x[col];
    }
  }
  const out = new Float64Array(n);
  for (let row = n - 1; row >= 0; row--) {
    let sum = x[row];
    for (let k = row + 1; k < n; k++) sum -= m[row * n + k] * out[k];
    const head = m[row * n + row];
    out[row] = Math.abs(head) < 1e-300 ? 0 : sum / head;
  }
  return out;
}

const channelRow = (hearing, index) => hearing.subarray(index * HEARING_WIDTH, (index + 1) * HEARING_WIDTH);

export class GesturePlanner {
  /** @param spec the planner's settings, as the fixture records them from the Python dataclass. */
  constructor(spec, reader) {
    Object.assign(this, {
      rollout: 3, beam: 6, pitch_floor: 1, pitch_span: 4, pitch_weight: 1.0, onset_weight: 0.5,
      rest_weight: 0.5, spectral_tiebreak: 0.1, read_notes: true, monitor_notes: false,
      hold_tolerance: 1, hold_margin: 0.05, note_envelope: [0, 0, 15, 0], note_volume: 15,
      attack_on_onset: true, synthesis: false, note_tiebreak: 0.05, monitor_weight: 0.0,
      pitch_stride: 3, refine: 2, waveforms: 4, shared_every: 4, epsilon: 0.0, mix_spectral: 1.0,
      voice_spectral: 0.5, silence: 0.5, onset: 0.25, action_change: 0.01, preference_weight: 1.0,
    }, spec);
    this.reader = reader;
    this.turn = 0;
    this.mode = "match";
    this.note = null;
    this.forced = null;
    this.target = null;
    this.target_channels = 1;
    this.fine = null;
    this.heard_fine = null;
    this.voices = null;
    this.voice_fines = null;
    this.echo = null;
    this.note_target = null;
    this.held_target = null;
    this.searched = 0;
  }

  /** What each voice is playing now, as a pitch index or nothing. */
  soundingPitches(base) {
    const out = [];
    for (let voice = 0; voice < cdc.VOICES; voice++) {
      const pitch = base[cdc.fieldIndex(voice, "pitch")];
      const gate = base[cdc.fieldIndex(voice, "gate")];
      out.push(pitch !== REST && gate === 1 ? pitch : null);
    }
    return out;
  }

  /** What each voice is heard to be playing now, off its own monitor. */
  monitorPitches(hearing) {
    const out = [];
    for (let voice = 0; voice < cdc.VOICES; voice++) {
      const row = channelRow(hearing, 1 + voice);
      if (!this.reader || row[BANDS + 3] > row[BANDS + 2]) { out.push(null); continue; }
      out.push(this.reader.bestPitches([row.subarray(0, BANDS)])[0]);
    }
    return out;
  }

  /** The gesture with an envelope that holds a note for as long as the gate is up. */
  sustaining(base, voice, mask) {
    const out = Int32Array.from(base);
    const [attack, decay, sustain, release] = this.note_envelope;
    for (const [name, value] of [["attack", attack], ["decay", decay], ["sustain", sustain], ["release", release]]) {
      const index = cdc.fieldIndex(voice, name);
      const [start] = cdc.SLOT_BOUNDS[index];
      if (mask[start + value]) out[index] = value;
    }
    const volume = cdc.sharedIndex("volume");
    const [start] = cdc.SLOT_BOUNDS[volume];
    if (out[volume] < this.note_volume && mask[start + this.note_volume]) out[volume] = this.note_volume;
    return out;
  }

  /** One voice's choices: a one-semitone grid around the note the reader named, and a rest. */
  noteCandidates(base, mask, voice, wanted) {
    const pitchAt = cdc.fieldIndex(voice, "pitch"), gateAt = cdc.fieldIndex(voice, "gate");
    const [start] = cdc.SLOT_BOUNDS[pitchAt];
    const out = [Int32Array.from(base)];
    const rest = Int32Array.from(base);
    rest[pitchAt] = REST;
    rest[gateAt] = 0;
    out.push(rest);
    if (wanted === null || wanted === undefined) return out;
    for (let step = -this.pitch_span; step <= this.pitch_span; step++) {
      const pitch = wanted + step;
      if (!(pitch >= this.pitch_floor && pitch < cdc.PITCHES) || !mask[start + pitch]) continue;
      const candidate = this.sustaining(base, voice, mask);
      candidate[pitchAt] = pitch;
      candidate[gateAt] = 1;
      out.push(candidate);
    }
    return out;
  }

  /** What a gesture costs against the notes the reader found, read off the gesture itself. */
  noteCost(gestures, base, target) {
    const span = Math.max(1.0, this.pitch_span);
    const out = new Float64Array(gestures.length);
    for (let voice = 0; voice < cdc.VOICES; voice++) {
      const pitchAt = cdc.fieldIndex(voice, "pitch"), gateAt = cdc.fieldIndex(voice, "gate");
      const was = base[gateAt] === 1 ? base[pitchAt] : REST;
      const wanted = target.pitches[voice];
      for (let g = 0; g < gestures.length; g++) {
        const pitch = gestures[g][pitchAt];
        const sounds = pitch !== REST && gestures[g][gateAt] === 1;
        const fresh = sounds && pitch !== was;
        if (wanted === null) {
          out[g] += this.rest_weight * (sounds ? 1 : 0);
          out[g] += this.onset_weight * (fresh ? 1 : 0);
          continue;
        }
        const error = Math.min(Math.abs(pitch - wanted), span) / span;
        out[g] += sounds ? this.pitch_weight * error : this.pitch_weight + this.rest_weight;
        const moved = was === REST || Math.abs(was - wanted) > this.hold_tolerance;
        out[g] += (target.onset && moved ? -1 : 1) * this.onset_weight * (fresh ? 1 : 0);
      }
    }
    for (let g = 0; g < out.length; g++) out[g] /= cdc.VOICES;
    return out;
  }

  /** What a candidate costs by the note its own voice would be heard to play. */
  monitorCost(brain, hearing, attending, hypothetical, base, gestures, voice, target, held) {
    const predicted = brain.imagine(hearing, attending, hypothetical, base, gestures, { fine: this.heard_fine });
    const heard = brain.imaginedHearing(hearing, predicted);
    const span = Math.max(1.0, this.pitch_span);
    const wanted = target.pitches[voice];
    const out = new Float64Array(gestures.length);
    const rows = heard.map((row) => channelRow(row, 1 + voice));
    const spectra = rows.map((row) => row.subarray(0, BANDS));
    const pitches = wanted === null ? null : (this.reader ? this.reader.bestPitches(spectra) : new Int32Array(rows.length));
    for (let g = 0; g < rows.length; g++) {
      const row = rows[g];
      const sounds = !(row[BANDS + 3] > row[BANDS + 2]);
      const onsetDb = row[BANDS + 1] * (ONSET_DB[1] - ONSET_DB[0]) + ONSET_DB[0];
      const onset = onsetDb >= ONSET_THRESHOLD_DB && sounds;
      if (wanted === null) {
        out[g] = this.rest_weight * (sounds ? 1 : 0) + this.onset_weight * (onset ? 1 : 0);
        continue;
      }
      const error = Math.min(Math.abs(pitches[g] - wanted), span) / span;
      let value = sounds ? this.pitch_weight * error : this.pitch_weight + this.rest_weight;
      const was = held[voice];
      const moved = was === null || Math.abs(was - wanted) > this.hold_tolerance;
      value += (target.onset && moved ? -1 : 1) * this.onset_weight * (onset ? 1 : 0);
      out[g] = value;
    }
    return out;
  }

  /** The same cost over all three voices at once, which is what separates a beam. */
  monitorTotal(brain, hearing, attending, hypothetical, base, gestures, target, held) {
    const predicted = brain.imagine(hearing, attending, hypothetical, base, gestures, { fine: this.heard_fine });
    const heard = brain.imaginedHearing(hearing, predicted);
    const span = Math.max(1.0, this.pitch_span);
    const total = new Float64Array(heard.length);
    for (let voice = 0; voice < cdc.VOICES; voice++) {
      const rows = heard.map((row) => channelRow(row, 1 + voice));
      const wanted = target.pitches[voice];
      const pitches = wanted === null ? null : (this.reader ? this.reader.bestPitches(rows.map((r) => r.subarray(0, BANDS))) : new Int32Array(rows.length));
      for (let g = 0; g < rows.length; g++) {
        const row = rows[g];
        const sounds = !(row[BANDS + 3] > row[BANDS + 2]);
        const onsetDb = row[BANDS + 1] * (ONSET_DB[1] - ONSET_DB[0]) + ONSET_DB[0];
        const onset = onsetDb >= ONSET_THRESHOLD_DB && sounds;
        if (wanted === null) {
          total[g] += this.rest_weight * (sounds ? 1 : 0) + this.onset_weight * (onset ? 1 : 0);
          continue;
        }
        const error = Math.min(Math.abs(pitches[g] - wanted), span) / span;
        total[g] += sounds ? this.pitch_weight * error : this.pitch_weight + this.rest_weight;
        const was = held[voice];
        const moved = was === null || Math.abs(was - wanted) > this.hold_tolerance;
        total[g] += (target.onset && moved ? -1 : 1) * this.onset_weight * (onset ? 1 : 0);
      }
    }
    for (let g = 0; g < total.length; g++) total[g] /= cdc.VOICES;
    return total;
  }

  /** How far the mix the model imagines of a candidate stands from the mix that was heard. */
  synthesisCost(brain, hearing, predicted, gestures, base, target) {
    const imagined = brain.imaginedFine(this.heard_fine, predicted);
    const wanted = this.fine;
    if (!imagined || !wanted || imagined[0].length !== wanted.length) return this.noteCost(gestures, base, target);
    const out = new Float64Array(gestures.length);
    for (let g = 0; g < imagined.length; g++) {
      let sum = 0;
      for (let k = 0; k < wanted.length; k++) { const d = imagined[g][k] - wanted[k]; sum += d * d; }
      out[g] = Math.sqrt(sum / wanted.length);
    }
    if (this.monitor_weight) {
      const heard = brain.imaginedHearing(hearing, predicted);
      let asked = 0;
      for (const p of target.pitches) if (p === null) asked++;
      asked /= cdc.VOICES;
      for (let g = 0; g < heard.length; g++) {
        let silent = 0;
        for (let voice = 0; voice < cdc.VOICES; voice++) {
          const row = channelRow(heard[g], 1 + voice);
          silent += row[BANDS + 3] > row[BANDS + 2] ? 1 : 0;
        }
        out[g] += this.monitor_weight * Math.abs(silent / cdc.VOICES - asked);
      }
    }
    const notes = this.noteCost(gestures, base, target);
    for (let g = 0; g < out.length; g++) out[g] += this.note_tiebreak * notes[g];
    return out;
  }

  /** Whether what the body is heard to play is already inside the tolerance of what is asked. */
  playingMonitor(held, target) {
    for (let voice = 0; voice < cdc.VOICES; voice++) {
      const was = held[voice], wanted = target.pitches[voice];
      if ((wanted === null) !== (was === null)) return false;
      if (wanted !== null && Math.abs(was - wanted) > this.hold_tolerance) return false;
    }
    return true;
  }

  /** Whether what the body is playing is already inside the tolerance of what is asked. */
  playing(base, target) {
    for (let voice = 0; voice < cdc.VOICES; voice++) {
      const pitchAt = cdc.fieldIndex(voice, "pitch"), gateAt = cdc.fieldIndex(voice, "gate");
      const sounds = base[pitchAt] !== REST && base[gateAt] === 1;
      const wanted = target.pitches[voice];
      if ((wanted === null) === sounds) return false;
      if (wanted !== null && Math.abs(base[pitchAt] - wanted) > this.hold_tolerance) return false;
    }
    return true;
  }

  /** Whether the phrase still asks for what it asked for, inside the hold tolerance. */
  steady(held, now) {
    for (let voice = 0; voice < cdc.VOICES; voice++) {
      const a = held.pitches[voice], b = now.pitches[voice];
      if ((a === null) !== (b === null)) return false;
      if (a !== null && b !== null && Math.abs(a - b) > this.hold_tolerance) return false;
    }
    return true;
  }

  /** The block a new note needs before it: the gate of the voice that changes, held down. */
  released(base, choice) {
    const out = Int32Array.from(base);
    let wanted = false;
    for (let voice = 0; voice < cdc.VOICES; voice++) {
      const pitch = cdc.fieldIndex(voice, "pitch"), gate = cdc.fieldIndex(voice, "gate");
      if (choice[gate] === 1 && choice[pitch] !== REST && base[gate] === 1 && base[pitch] !== REST && choice[pitch] !== base[pitch]) {
        out[gate] = 0;
        wanted = true;
      }
    }
    return wanted ? out : null;
  }

  /** Revise all three voices against the notes, then let the bands separate what ties. */
  byNotes(brain, hearing, attending, hypothetical, base, mask, target, quiet = false) {
    const synthesis = this.synthesis && this.fine !== null && this.heard_fine !== null
      && brain.fields.some((f) => f.name === "d_fine_mix");
    const monitors = this.monitor_notes && this.reader !== null;
    const heldNow = monitors ? this.monitorPitches(hearing) : [null, null, null];
    let running = Int32Array.from(base);
    let expansions = 0;
    const beam = [];
    this.trace = [];
    for (let voice = 0; voice < cdc.VOICES; voice++) {
      const wanted = target.pitches[voice];
      if (quiet && wanted !== null && this.soundingPitches(running)[voice] !== null) continue;
      let gestures = this.noteCandidates(running, mask, voice, wanted).filter((g) => legal(g, mask));
      if (!gestures.length) continue;
      expansions += gestures.length;
      let total;
      if (monitors) {
        total = this.monitorCost(brain, hearing, attending, hypothetical, running, gestures, voice, target, heldNow);
      } else if (synthesis) {
        const predicted = brain.imagine(hearing, attending, hypothetical, running, gestures, { fine: this.heard_fine });
        total = this.synthesisCost(brain, hearing, predicted, gestures, running, target);
      } else {
        total = this.noteCost(gestures, base, target);
      }
      total[0] -= this.hold_margin * (synthesis ? this.note_tiebreak : 1.0);
      // how many candidates share the least cost: the monitor cost reads the same note off every
      // imagined monitor in most blocks (NOTES.md section 25), and then the winner is whatever the
      // sort's tie-break gives, which numpy's introsort and a stable sort do differently
      let least = total[0], ties = 0;
      for (let k = 1; k < total.length; k++) if (total[k] < least) least = total[k];
      for (let k = 0; k < total.length; k++) if (Math.abs(total[k] - least) <= 1e-12) ties++;
      this.trace.push({ voice, gestures: gestures.map((g) => Int32Array.from(g)), total: Float64Array.from(total), ties, least });
      const order = argsort(total);
      running = Int32Array.from(gestures[order[0]]);
      if (order.length > 1 && total[order[1]] - total[order[0]] < 1e-9) beam.push(Int32Array.from(gestures[order[1]]));
    }
    const gestures = [running].concat(beam.filter((g) => legal(g, mask)));
    const predicted = brain.imagine(hearing, attending, hypothetical, base, gestures, { fine: this.heard_fine });
    let winner = 0;
    if (gestures.length > 1) {
      if (monitors) {
        winner = argmin(this.monitorTotal(brain, hearing, attending, hypothetical, base, gestures, target, heldNow));
      } else if (synthesis) {
        winner = argmin(this.synthesisCost(brain, hearing, predicted, gestures, base, target));
      } else if (this.spectral_tiebreak) {
        const heard = brain.imaginedHearing(hearing, predicted);
        const coarse = this.target.subarray(0, BANDS);
        const notes = this.noteCost(gestures, base, target);
        const total = new Float64Array(gestures.length);
        for (let g = 0; g < heard.length; g++) {
          let sum = 0;
          for (let k = 0; k < BANDS; k++) { const d = heard[g][k] - coarse[k]; sum += d * d; }
          total[g] = notes[g] + this.spectral_tiebreak * Math.sqrt(sum / BANDS);
        }
        winner = argmin(total);
      }
    }
    this.searched += 1;
    this.trace.push({ voice: null, gestures: gestures.map((g) => Int32Array.from(g)), total: null, winner });
    const budget = { candidates: expansions, expansions: expansions + gestures.length, rollout: 1 };
    const choice = gestures[winner];
    const release = this.released(base, choice);
    if (release !== null && legal(release, mask)) {
      this.note = Int32Array.from(choice);
      return { values: release, budget, controller: "release" };
    }
    return { values: choice, budget, controller: "notes" };
  }

  /**
   * One decision: the 28 field values to play now.
   * @param inputs {hearing, attending, hypothetical, mask, base, fine (the target's), heard_fine}
   */
  plan(brain, inputs) {
    const { hearing, attending, hypothetical, mask } = inputs;
    this.heard_fine = inputs.heard_fine ?? null;
    let base = this.forced !== null ? this.forced : inputs.base;
    const plays = mayPlay(mask);
    if (this.note !== null && plays && legal(this.note, mask)) {
      const note = this.note;
      this.note = null;
      return { values: note, budget: { candidates: 1, expansions: 1 }, controller: "note" };
    }
    this.note = null;
    if (!plays) base = silentLike(base);
    if (this.forced !== null || !plays) {
      this.note_target = null;
      this.held_target = null;
      return { values: base, budget: { candidates: 1, expansions: 1 }, controller: "held" };
    }
    if (this.read_notes && this.mode === "match" && this.target !== null && this.reader !== null) {
      const target = this.voices !== null
        ? this.reader.readVoices(this.voices, this.voice_fines || [null, null, null])
        : this.reader.read(this.target.subarray(0, HEARING_WIDTH), this.soundingPitches(base), this.fine);
      const held = this.held_target;
      this.note_target = target;
      this.held_target = target;
      const quiet = this.attack_on_onset && !target.onset && held !== null;
      const settled = (this.monitor_notes && this.reader !== null)
        ? this.playingMonitor(this.monitorPitches(hearing), target)
        : this.playing(base, target);
      if (held !== null && !target.onset && this.steady(held, target) && settled) {
        return { values: base, budget: { candidates: 1, expansions: 1 }, controller: "held-note" };
      }
      return this.byNotes(brain, hearing, attending, hypothetical, base, mask, target, quiet);
    }
    // the band objective and the compose path are the stage's other two costs and are not ported
    // here: with no note to match, this build holds what the body is playing and says so
    return { values: base, budget: { candidates: 1, expansions: 1 }, controller: "no-target" };
  }
}

export function argsort(values) {
  const order = Array.from(values.keys());
  order.sort((a, b) => (values[a] - values[b]) || (a - b)); // numpy's stable argsort
  return order;
}

export function argmin(values) {
  let best = 0;
  for (let k = 1; k < values.length; k++) if (values[k] < values[best]) best = k;
  return best;
}

/** Whether the mask leaves this body a note: during a demonstration it leaves the rest alone. */
export function mayPlay(mask) {
  const [start, stop] = cdc.SLOT_BOUNDS[cdc.fieldIndex(0, "pitch")];
  for (let i = start + 1; i < stop; i++) if (mask[i]) return true;
  return false;
}

/** Whether every field of a gesture is a value the mask allows. */
export function legal(gesture, mask) {
  for (let f = 0; f < cdc.FIELD_COUNT; f++) if (!mask[cdc.SLOT_BOUNDS[f][0] + gesture[f]]) return false;
  return true;
}

/** The same gesture with every voice resting, which is what a body that may not play does. */
export function silentLike(gesture) {
  const out = Int32Array.from(gesture);
  for (let voice = 0; voice < cdc.VOICES; voice++) {
    out[cdc.fieldIndex(voice, "pitch")] = REST;
    out[cdc.fieldIndex(voice, "gate")] = 0;
  }
  return out;
}
