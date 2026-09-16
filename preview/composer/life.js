// The composer's life in the browser: it hears a phrase voice by voice, then answers it.
//
// A port of what `ComposerLife` does around one imitation, and of the part of `World` the page
// needs: the demonstration is played on its own chips and its audio is added to what this body
// hears, the body is held to a rest while it plays, and when it stops the phrase is latched and
// the attempt begins. The decision itself is `GesturePlanner.plan` over `S06Brain`, which the
// decision parity measures against the Python stage.
//
// What is not here: learning, the phrase store, the preference store, the compose path and the
// band objective. The page listens and answers, which is the match path.

import * as cdc from "./js/codec.js";
import * as ear from "./js/hearing.js";
import { S06Brain } from "./js/s06_brain.js";
import { GesturePlanner, PitchReader, mayPlay, silentLike } from "./js/s06_planner.js";
import { Instrument, MIX } from "./instrument.js";

export const STEP_BLOCKS = 6;
export const PHRASE_STEPS = 32;

/** A written piece as a stream of gestures, the way corpus.gestures_of expands it. */
export function gesturesOf(piece) {
  const blocks = piece.blocks | 0;
  const out = [];
  for (let b = 0; b < blocks; b++) {
    const gesture = cdc.silence();
    for (const [name, value] of Object.entries(piece.shared || {})) gesture[cdc.sharedIndex(name)] = value | 0;
    for (let voice = 0; voice < piece.voices.length; voice++) {
      const settings = piece.voices[voice].settings || {};
      for (const [name, value] of Object.entries(settings)) gesture[cdc.fieldIndex(voice, name)] = value | 0;
      gesture[cdc.fieldIndex(voice, "pitch")] = cdc.REST;
      gesture[cdc.fieldIndex(voice, "gate")] = 0;
    }
    out.push(gesture);
  }
  for (let voice = 0; voice < piece.voices.length; voice++) {
    const waveform = (piece.voices[voice].settings || {}).waveform ?? 2;
    for (const [start, span, pitch] of piece.voices[voice].notes || []) {
      for (let b = start; b < Math.min(start + span, blocks); b++) {
        out[b][cdc.fieldIndex(voice, "pitch")] = pitch;
        out[b][cdc.fieldIndex(voice, "gate")] = 1;
        out[b][cdc.fieldIndex(voice, "waveform")] = waveform;
      }
    }
  }
  return out;
}

/** The same stream with every voice but one resting: the solo pass of the listen phase. */
export function soloPass(gestures, voice) {
  return gestures.map((gesture) => {
    const out = Int32Array.from(gesture);
    for (let other = 0; other < cdc.VOICES; other++) {
      if (other === voice) continue;
      out[cdc.fieldIndex(other, "pitch")] = cdc.REST;
      out[cdc.fieldIndex(other, "gate")] = 0;
    }
    return out;
  });
}

/** The mask of a block: a rest is the only note while a demonstration plays. */
export function maskOf(brain, plays) {
  const mask = new Uint8Array(cdc.MOTOR_NEURONS).fill(1);
  const legal = brain.legalMask;
  if (legal) for (let i = 0; i < mask.length; i++) mask[i] = legal[i];
  if (plays) return mask;
  for (let voice = 0; voice < cdc.VOICES; voice++) {
    const [start, stop] = cdc.SLOT_BOUNDS[cdc.fieldIndex(voice, "pitch")];
    for (let i = start + 1; i < stop; i++) mask[i] = 0;
    const [gateStart, gateStop] = cdc.SLOT_BOUNDS[cdc.fieldIndex(voice, "gate")];
    for (let i = gateStart + 1; i < gateStop; i++) mask[i] = 0;
  }
  return mask;
}

export class LiveComposer {
  /**
   * @param spec the page's brain asset (s06-page-brain/1)
   * @param instrument the four-chip instrument
   * @param demonstrator a second instrument, for the phrase this body hears
   */
  constructor(spec, instrument, demonstrator) {
    this.spec = spec;
    this.clock = spec.clock;
    this.brain = new S06Brain(spec);
    this.reader = new PitchReader(spec.reader);
    this.planner = new GesturePlanner(spec.planner, this.reader);
    this.codec = new cdc.Codec(spec.clock.cycles_per_second || 985248.0 * (1 + 0), {});
    this.instrument = instrument;
    this.demonstrator = demonstrator;
    // the reading takes nothing from the connectome: measured over 4,240 decisions of the
    // acceptance brain, the hypothetical drive over the reading is zero in every one of them
    this.hypothetical = new Float64Array(spec.reading.length);
    this.reset();
  }

  reset() {
    this.phase = "idle";
    this.schedule = this.schedule || [];
    this.leg = this.schedule.length;  // an empty schedule is a life that has heard nothing yet
    this.legBlock = 0;
    this.block = 0;
    this.played = 0;
    this.listened = 0;
    this.heard = [];
    this.heardFine = [];
    this.latched = new Map();
    this.latchedFine = new Map();
    this.latchedVoice = [new Map(), new Map(), new Map()];
    this.latchedVoiceFine = [new Map(), new Map(), new Map()];
    this.base = cdc.silence();
    this.image = null;
    this.played_notes = [];
    this.planner.note = null;
    this.planner.forced = null;
    this.planner.held_target = null;
    this.planner.note_target = null;
    this.planner.mode = "match";
    this.planner.target = null;
    this.planner.voices = null;
    this.planner.voice_fines = null;
    this.planner.fine = null;
    this.planner.heard_fine = null;
    this.planner.echo = null;
  }

  /** The phrase to hear: its gestures, and the schedule of passes the stage listens through. */
  setPhrase(gestures, { passes = 4 } = {}) {
    this.phrase = gestures;
    this.passes = Math.max(1, passes);
    this.reset();
    this.schedule = [];
    if (this.passes > 1) {
      for (let voice = 0; voice < Math.min(this.passes - 1, cdc.VOICES); voice++) {
        this.schedule.push({ phase: "listen", stream: soloPass(gestures, voice), label: `listening to voice ${voice + 1}` });
      }
    }
    this.schedule.push({ phase: "listen", stream: gestures, label: "listening to the phrase" });
    this.schedule.push({ phase: "gap", blocks: this.clock.gap_blocks | 0, label: "the ear clears" });
    this.schedule.push({ phase: "attempt", blocks: gestures.length, label: "playing" });
    this.leg = 0;
    this.legBlock = 0;
    this.phase = "listen";
    this.instrument.reset();
    this.demonstrator.reset();
    this.demoImage = null;
  }

  get done() { return this.leg >= this.schedule.length; }

  get label() { return this.done ? "done" : this.schedule[this.leg].label; }

  /** One 20 ms block: hear what sounds, decide, play it, and return what the page draws. */
  step() {
    if (this.done) return null;
    const leg = this.schedule[this.leg];
    const listening = leg.phase !== "attempt";
    const clock = { demonstration: 0, decision: 0, chips: 0, ear: 0 };
    let mark = performance.now();

    // the demonstration, on its own chips, added to what this body hears
    let added = null;
    if (leg.phase === "listen") {
      const gesture = leg.stream[this.legBlock];
      const { writes, image } = this.codec.writes(gesture, this.demoImage);
      this.demoImage = image;
      added = this.demonstrator.play(writes).mix;
    }
    clock.demonstration = performance.now() - mark;
    mark = performance.now();

    const mask = maskOf(this.brain, !listening);
    const base = listening ? silentLike(this.base) : this.base;

    // what this body plays this block
    let values = base, controller = "held", budget = { candidates: 1, expansions: 1 };
    const hearing = this.hearing || new Float64Array(4 * ear.WIDTH);
    if (!listening) {
      const step = Math.floor(this.played / STEP_BLOCKS);
      this.planner.mode = "match";
      this.planner.target = this.targetFor(step);
      this.planner.fine = this.fineFor(step);
      const [voices, fines] = this.voicesFor(step);
      this.planner.voices = voices;
      this.planner.voice_fines = fines;
      this.planner.forced = null;
      const decision = this.planner.plan(this.brain, {
        hearing, attending: Float64Array.from([1, 0]), hypothetical: this.hypothetical,
        mask, base, heard_fine: this.fine || null,
      });
      values = decision.values;
      controller = decision.controller;
      budget = decision.budget;
      this.played++;
    } else {
      this.planner.forced = silentLike(this.base);
      this.planner.target = null;
      this.planner.voices = this.planner.voice_fines = null;
      values = silentLike(this.base);
    }

    clock.decision = performance.now() - mark;
    mark = performance.now();
    const executed = cdc.executed(values);
    const { writes, image } = this.codec.writes(executed, this.image);
    this.image = image;
    const played = this.instrument.play(writes, added);
    this.base = executed;
    clock.chips = performance.now() - mark;
    mark = performance.now();

    // what it heard of the block it just played
    this.hearing = this.instrument.hear();
    this.fine = this.instrument.fine(this.spec.reader.bands);
    clock.ear = performance.now() - mark;
    if (listening) {
      this.heard.push(Float64Array.from(this.hearing));
      this.heardFine.push(Float64Array.from(this.fine));
    }

    this.block++;
    this.legBlock++;
    const blocks = leg.phase === "listen" ? leg.stream.length : leg.blocks;
    if (this.legBlock >= blocks) {
      this.leg++;
      this.legBlock = 0;
      if (!this.done && this.schedule[this.leg].phase === "attempt") this.latch();
      this.phase = this.done ? "done" : this.schedule[this.leg].phase;
    }
    return { phase: leg.phase, values: executed, controller, budget, audio: played.mix, hearing: this.hearing, clock };
  }

  /** A stretch of blocks as one hearing per step of the phrase grid: the step's mean. */
  stepsOf(blocks, width) {
    const out = new Map();
    for (let step = 0; step < Math.ceil(blocks.length / STEP_BLOCKS); step++) {
      const window = blocks.slice(step * STEP_BLOCKS, (step + 1) * STEP_BLOCKS);
      if (!window.length || step >= PHRASE_STEPS) continue;
      const mean = new Float64Array(width);
      for (const row of window) for (let k = 0; k < width; k++) mean[k] += row[k] / window.length;
      out.set(step, mean);
    }
    return out;
  }

  /** The end of what was heard: the passes are cut apart, the gap trimmed, the phrase kept. */
  latch() {
    const passes = this.passes;
    if (passes > 1 && this.heard.length >= passes * STEP_BLOCKS) {
      const unit = Math.floor(this.heard.length / passes);
      for (let voice = 0; voice < Math.min(passes - 1, cdc.VOICES); voice++) {
        this.latchedVoice[voice] = this.stepsOf(this.heard.slice(voice * unit, (voice + 1) * unit), 4 * ear.WIDTH);
        if (this.heardFine.length) {
          this.latchedVoiceFine[voice] = this.stepsOf(this.heardFine.slice(voice * unit, (voice + 1) * unit), this.spec.reader.bands);
        }
      }
      this.heard = this.heard.slice((passes - 1) * unit);
      if (this.heardFine.length) this.heardFine = this.heardFine.slice((passes - 1) * unit);
    }
    while (this.heard.length > 1 && this.heard[this.heard.length - 1][ear.BANDS + 3] > this.heard[this.heard.length - 1][ear.BANDS + 2]) {
      this.heard.pop();
      this.heardFine.pop();
    }
    let sounded = 0;
    for (const row of this.heard) if (row[ear.BANDS + 2] >= row[ear.BANDS + 3]) sounded++;
    if (this.heard.length < STEP_BLOCKS || !sounded) { this.heard = []; return; }
    this.latched = this.stepsOf(this.heard, 4 * ear.WIDTH);
    this.latchedFine = this.heardFine.length ? this.stepsOf(this.heardFine, this.spec.reader.bands) : new Map();
    this.listened = this.heard.length;
  }

  targetFor(step) {
    const wanted = step % PHRASE_STEPS;
    return this.latched.get(wanted) || null;
  }

  fineFor(step) { return this.latchedFine.get(step % PHRASE_STEPS) || null; }

  voicesFor(step) {
    const wanted = step % PHRASE_STEPS;
    if (!this.latchedVoice[0].size) return [null, null];
    const rows = [], fines = [];
    for (let voice = 0; voice < cdc.VOICES; voice++) {
      const row = this.latchedVoice[voice].get(wanted);
      if (!row) return [null, null];
      rows.push(row.subarray(0, ear.WIDTH));
      fines.push(this.latchedVoiceFine[voice].get(wanted) || null);
    }
    return [rows, fines];
  }
}

export { Instrument, MIX, mayPlay };
