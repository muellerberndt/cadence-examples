// The four stores of world/brain.py in JavaScript: `PlaceStore`
// (object to place with a known flag), `MapStore` (cell to its four passages with a known
// flag: the witnessed map), `CueStore` (the episode's cue word) and `WordStore` (word to
// referent co-occurrence), with the arithmetic of the cadence classes they are built on,
// for one stream: `FastSynapses` with rule "delta" (cadence/src/cadence/stream.py:
// `_delta_unit` keys, decay once per write, the residual write `M += rate * outer(k, value -
// k @ M)`, the unscaled read) and `SynapticMemory` (cadence/src/cadence/memory.py: the
// consolidated matrix plus a per-stream residual, the slow consolidation at rate
// `consolidation` and the fast correction at `rate`). Every operation is written out
// element by element in the order numpy evaluates it, so a life's store contents agree with
// the Python life bit for bit on one-hot keys (see parity_s02.mjs).

import { npSum } from "../../web/engine.js";
import { CELLS, DIRECTION_NAMES, OBJECTS, PASSAGE, SIZE, WORDS, argmaxOf } from "./world.js";

export function cellIndex(x, y) { return y * SIZE + x; }

/** FastSynapses._delta_unit on one key: scale by the largest magnitude, then divide by the norm (a one-hot key is its own unit). */
export function deltaUnit(x) {
  let scale = 0.0;
  for (let k = 0; k < x.length; k++) { const a = Math.abs(x[k]); if (a > scale) scale = a; }
  const scaled = Float64Array.from(x, (v) => v / (scale > 0 ? scale : 1.0));
  const norm = Math.sqrt(npSum(Float64Array.from(scaled, (v) => v * v)));
  return Float64Array.from(scaled, (v) => v / (norm > 0 ? norm : 1.0));
}

const matrix = (rows, cols) => Array.from({ length: rows }, () => new Float64Array(cols));
const anyNonzero = (x) => { for (let k = 0; k < x.length; k++) if (x[k] !== 0.0) return true; return false; };

/** cadence.stream.FastSynapses(rule="delta") for one stream: `strength` is (keys, values). */
export class DeltaSynapses {
  constructor(keys, values, { decay = 1.0, rate = 1.0, amplitude = 1.0 } = {}) {
    if (!(decay >= 0 && decay <= 1)) throw Error("decay lies in [0, 1]");
    if (!(rate >= 0 && rate <= 1)) throw Error("delta rate lies in [0, 1]");
    this.keys = keys | 0; this.values = values | 0; this.decay = decay; this.rate = rate; this.amplitude = amplitude;
    this.strength = matrix(this.keys, this.values);
    this.mass = 0.0;
    this.writes = 0;
  }
  reset() { this.strength = matrix(this.keys, this.values); this.mass = 0.0; }
  /** `cue @ strength` (numpy matmul over the key axis, summed in index order). */
  _read(cue, strength) {
    const out = new Float64Array(this.values);
    for (let j = 0; j < this.values; j++) { let acc = 0.0; for (let i = 0; i < this.keys; i++) acc += cue[i] * strength[i][j]; out[j] = acc; }
    return out;
  }
  /** Read a key as a value without changing memory; delta uses unit keys and does not divide by the count of writes. */
  recall(key) {
    if (key.length !== this.keys) throw Error(`key must have ${this.keys} entries`);
    const out = this._read(deltaUnit(key), this.strength);
    return Float64Array.from(out, (x) => this.amplitude * x);
  }
  /** Decay once, then write the residual of the unit key toward the value (a zero key writes nothing). */
  observe(key, value) {
    if (key.length !== this.keys || value.length !== this.values) throw Error("key and value must match the ports");
    const strength = this.strength.map((row) => Float64Array.from(row, (x) => x * this.decay));
    const mass = this.mass * this.decay;
    if (!anyNonzero(key)) { this.strength = strength; this.mass = mass; return; }
    const a = deltaUnit(key);
    const predicted = this._read(a, strength);
    const b = Float64Array.from(value, (v, j) => v - predicted[j]);
    for (let i = 0; i < this.keys; i++) { const ra = this.rate * a[i]; for (let j = 0; j < this.values; j++) strength[i][j] += ra * b[j]; }
    for (const row of strength) for (let j = 0; j < this.values; j++) if (!Number.isFinite(row[j])) throw Error("synaptic update overflowed; scale the observed keys and values");
    this.strength = strength; this.mass = mass + this.rate;
    this.writes += 1;
  }
  toJSON() { return { strength: this.strength.map((row) => Array.from(row)), mass: [this.mass], writes: this.writes }; }
  load(d) {
    if (d.strength.length !== this.keys || d.strength.some((row) => row.length !== this.values)) throw Error("the stored strength does not match the ports");
    this.strength = d.strength.map((row) => Float64Array.from(row, Number)); this.mass = Number(Array.isArray(d.mass) ? d.mass[0] : d.mass); this.writes = d.writes | 0;
  }
}

/** cadence.memory.SynapticMemory for one stream: a consolidated matrix shared across streams plus this stream's residual. */
export class ConsolidatingSynapses extends DeltaSynapses {
  constructor(keys, values, { decay = 0.9, rate = 1.0, amplitude = 1.0, consolidation = 0.05 } = {}) {
    super(keys, values, { decay, rate, amplitude });
    if (!(consolidation >= 0 && consolidation <= 1)) throw Error("consolidation must lie in [0, 1]");
    this.consolidation = consolidation;
    this.consolidated = matrix(this.keys, this.values);
  }
  /** Forget the transient residual; persistent synapses survive. */
  reset() { super.reset(); this.strength = this.consolidated.map((row) => Float64Array.from(row)); }
  /** Erase both persistent and transient associations. */
  clear() { this.consolidated = matrix(this.keys, this.values); this.reset(); }
  /** Consolidate one real observation (salience 0, every value observed): the slow update at `consolidation`, the fast correction at `rate`. */
  observe(key, value, { salience = 0.0 } = {}) {
    if (key.length !== this.keys || value.length !== this.values) throw Error("key and value must match the ports");
    const C = this.consolidated;
    // W = C + F: the residual F decays, C never does
    const strength = this.strength.map((row, i) => Float64Array.from(row, (x, j) => C[i][j] + this.decay * (x - C[i][j])));
    const mass = this.mass * this.decay;
    if (!anyNonzero(key)) { this.strength = strength; this.mass = mass; return; }
    const cue = deltaUnit(key);
    let rate = this.consolidation + Math.min(1.0, this.consolidation * salience);
    rate = Math.min(1.0, rate);
    const read = this._read(cue, C);
    const error = Float64Array.from(value, (v, j) => (v - read[j]) * 1.0);
    const change = matrix(this.keys, this.values);
    for (let i = 0; i < this.keys; i++) { const rc = rate * cue[i]; for (let j = 0; j < this.values; j++) change[i][j] = (rc * error[j]) / 1; }
    const consolidated = C.map((row, i) => Float64Array.from(row, (x, j) => x + change[i][j]));
    for (let i = 0; i < this.keys; i++) for (let j = 0; j < this.values; j++) strength[i][j] += change[i][j];
    const prediction = this._read(cue, strength);
    const correction = Float64Array.from(value, (v, j) => (v - prediction[j]) * 1.0);
    for (let i = 0; i < this.keys; i++) { const rc = this.rate * cue[i]; for (let j = 0; j < this.values; j++) strength[i][j] += rc * correction[j]; }
    for (const rows of [strength, consolidated]) for (const row of rows) for (let j = 0; j < this.values; j++) if (!Number.isFinite(row[j])) throw Error("synaptic update overflowed; scale the observed values");
    this.consolidated = consolidated; this.strength = strength; this.mass = mass + this.rate;
    this.writes += 1;
  }
  toJSON() { return { ...super.toJSON(), consolidated: this.consolidated.map((row) => Array.from(row)) }; }
  load(d) { super.load(d); if (d.consolidated.length !== this.keys) throw Error("the stored consolidated matrix does not match the ports"); this.consolidated = d.consolidated.map((row) => Float64Array.from(row, Number)); }
}

const oneHotKey = (index, width) => { const out = new Float64Array(width); out[index] = 1.0; return out; };

/** object -> (place one-hot over the cells, known flag); one exposure writes it, a visit that finds nothing erases it. */
export class PlaceStore {
  constructor() { this.memory = new DeltaSynapses(OBJECTS, CELLS + 1, { rule: "delta" }); this.writes = 0; }
  see(k, cell) {
    const value = new Float64Array(CELLS + 1);
    value[cellIndex(cell[0], cell[1])] = 1.0;
    value[CELLS] = 1.0;
    this.memory.observe(oneHotKey(k, OBJECTS), value);
    this.writes += 1;
  }
  forget(k) { this.memory.observe(oneHotKey(k, OBJECTS), new Float64Array(CELLS + 1)); this.writes += 1; }
  /** {where: [x, y] | null, known} */
  recall(k) {
    const read = this.memory.recall(oneHotKey(k, OBJECTS));
    const known = read[CELLS];
    if (known < 0.5) return { where: null, known };
    const c = argmaxOf(read.subarray(0, CELLS));
    return { where: [c % SIZE, Math.floor(c / SIZE)], known };
  }
  drive(k) { return k === null || k === undefined ? new Float64Array(CELLS + 1) : this.memory.recall(oneHotKey(k, OBJECTS)); }
  erase() { this.memory.reset(); }
  /** Every object's record, for a page: {object, where, known, strength}. */
  records() { return Array.from({ length: OBJECTS }, (_, k) => ({ object: k, ...this.recall(k), strength: Float64Array.from(this.memory.strength[k]) })); }
  toJSON() { return { memory: this.memory.toJSON(), writes: this.writes }; }
  load(d) { this.memory.load(d.memory); this.writes = d.writes | 0; }
}

/** cell -> its four passages with a known flag: the witnessed map. Standing in a cell shows its passages; one exposure writes them, a later exposure overwrites (a door that changed). */
export class MapStore {
  constructor() { this.memory = new DeltaSynapses(CELLS, 13, { rule: "delta" }); this.writes = 0; }
  see(cell, passages) {
    const value = new Float64Array(13);
    passages.forEach((kind, d) => { value[3 * d + PASSAGE.indexOf(kind)] = 1.0; });
    value[12] = 1.0;
    this.memory.observe(oneHotKey(cellIndex(cell[0], cell[1]), CELLS), value);
    this.writes += 1;
  }
  /** {passages: [kind x 4] | null, known} */
  recall(cell) {
    const read = this.memory.recall(oneHotKey(cellIndex(cell[0], cell[1]), CELLS));
    const known = read[12];
    if (known < 0.5) return { passages: null, known };
    return { passages: DIRECTION_NAMES.map((_, d) => PASSAGE[argmaxOf(read.subarray(3 * d, 3 * d + 3))]), known };
  }
  erase() { this.memory.reset(); }
  /** Every known cell's passages, for a page: {cell, passages, known}. */
  records() { const out = []; for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) { const r = this.recall([x, y]); if (r.passages !== null) out.push({ cell: [x, y], ...r }); } return out; }
  toJSON() { return { memory: this.memory.toJSON(), writes: this.writes }; }
  load(d) { this.memory.load(d.memory); this.writes = d.writes | 0; }
}

/** The episode's cue word, held from the moment it was seen until the episode ends. */
export class CueStore {
  constructor() { this.value = new Float64Array(WORDS + 1); }
  see(word) { if (word > 0) this.value = oneHotKey(word, WORDS + 1); }
  reset() { this.value = new Float64Array(WORDS + 1); }
  get word() { return anyNonzero(this.value) ? argmaxOf(this.value) : 0; }
  toJSON() { return { value: Array.from(this.value) }; }
  load(d) { this.value = Float64Array.from(d.value, Number); }
}

/** word -> referent by accumulated co-occurrence with visible objects across scenes. */
export class WordStore {
  constructor(rate = 0.25) { this.memory = new ConsolidatingSynapses(WORDS + 1, OBJECTS, { decay: 1.0, rate }); this.writes = 0; }
  scene(word, visible) {
    if (word <= 0 || !visible.length) return;
    const value = new Float64Array(OBJECTS);
    for (const k of visible) value[k] = 1.0;
    this.memory.observe(oneHotKey(word, WORDS + 1), value);
    this.writes += 1;
  }
  /** {object: k | null, strength}: the strongest referent when it leads the runner-up by 0.05 (numpy argsort of -read: stable for these small arrays). */
  referent(word) {
    const read = this.memory.recall(oneHotKey(word, WORDS + 1));
    const order = Array.from(read, (_, k) => k).sort((a, b) => (read[b] > read[a] ? 1 : read[b] < read[a] ? -1 : a - b));
    if (read[order[0]] <= 0 || read[order[0]] - read[order[1]] < 0.05) return { object: null, strength: read[order[0]] };
    return { object: order[0], strength: read[order[0]] };
  }
  drive(word) { return word <= 0 ? new Float64Array(OBJECTS) : this.memory.recall(oneHotKey(word, WORDS + 1)); }
  erase() { this.memory.clear(); }
  /** word (1..WORDS) -> strengths over the objects, for a page. */
  table() { return Array.from({ length: WORDS }, (_, w) => Float64Array.from(this.memory.strength[w + 1])); }
  toJSON() { return { memory: this.memory.toJSON(), writes: this.writes }; }
  load(d) { this.memory.load(d.memory); this.writes = d.writes | 0; }
}

export const RECALL_WIDTH = (CELLS + 1) + (WORDS + 1) + OBJECTS; // place read, cue, word read
