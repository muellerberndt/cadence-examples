// The Connect Four brain in the browser: the JavaScript port of connect_four/brain.py.
//
// Two record cortices (the port of cadence.Records) hold what followed each reading. The drop
// cortex reads one column of the board and holds the row a dropped stone lands in; the line
// cortex reads one window of four cells and holds whether the window is a completed line and
// the outcome of the games that held it. Imagination composes the next board from the
// predicted landings, a validator rejects a board that is not the old board plus one stone,
// and negamax with alpha-beta searches the imagined boards within a budget. Every witnessed
// move is written into the records the reading touched, so the brain keeps learning from the
// games a visitor plays against it, exactly as the Python brain learns from its own.
//
// The S00 part of the brain (the workspace and dynamics regions, the event transaction, the
// critic) is the shared engine in web/engine.js, instantiated from the agent snapshot with
// this planner as its controller, so every settling step is drawn by web/brain_scan.js while
// the page runs.
//
// Numbers: float64 everywhere. The expansion of each cortex is regenerated from its seed with
// the Mulberry32 generator cadence.Records draws it from, so the cells of the page are the
// cells of the Python brain. The planner's tie-break generator is numpy's PCG64 continued from
// the state the checkpoint saved, so a tie among equal moves falls the way Python's falls.

import { ExperienceAgent, Mulberry32, npSum, topK, unpackArray, unpackF64 } from "../../web/engine.js";
import { Board, cellsOf, gameConfig } from "./game.js";

export const CHECKPOINT_FORMAT = "cadence-connect-four-checkpoint/1";
export const BRAIN_FORMAT = "cadence-connect-four-web/1";
export const CANDIDATE = 1, OPPONENT = 2;      // stone values of a board relative to the candidate
export const WIN_BONUS = 1e-4;                  // a proven result one ply sooner scores this much better
export const LEAF_MARGIN = 0.01;                // leaf values stay inside [margin, 1 - margin]
export const TIE = 1e-9;

/** A proven search value without its ply bonus: 1 for a win, 0 for a loss, 0.5 for a draw. */
export function plainResult(value) { return value > 0.9 ? 1.0 : value < 0.1 ? 0.0 : 0.5; }

/** A remembered result as the search scores it at `ply`: as if decided one ply on. */
export function provenValue(result, ply) { return result > 0.9 ? 1.0 - WIN_BONUS * (ply + 1) : result < 0.1 ? WIN_BONUS * (ply + 1) : 0.5; }
export const VIEW = [[0, 1, 2], [0, 1, 2], [0, 2, 1]]; // VIEW[side][cell]: side 1 keeps, side 2 swaps
export const MOVER_CANDIDATE = 0;               // env.MOVER.index("candidate")

const clamp = (x, lo, hi) => (x < lo ? lo : x > hi ? hi : x);

/** The first index of the largest entry (numpy argmax). */
export function argmaxOf(values, start = 0, count = values.length - start) {
  let arg = 0;
  for (let k = 1; k < count; k++) if (values[start + k] > values[start + arg]) arg = k;
  return arg;
}

// ---------------------------------------------------------------- numpy's generator

const M128 = (1n << 128n) - 1n, M64 = (1n << 64n) - 1n;
const PCG_MULT = (2549297995355413924n << 64n) + 4865540595714422341n;

/**
 * numpy's PCG64 and `Generator.integers(n)` for n below 2^32, continued from the state a
 * checkpoint saved (`bit_generator.state`). `integers` follows numpy's bounded path: no draw
 * when n is 1, otherwise Lemire's method on 32-bit draws, which take the halves of a 64-bit
 * draw in turn (the cached half is part of the state).
 */
export class PCG64 {
  constructor(state) {
    this.state = BigInt(state.state) & M128;
    this.inc = BigInt(state.inc) & M128;
    this.hasUint32 = state.has_uint32 | 0;
    this.uinteger = (state.uinteger || 0) >>> 0;
  }

  toDict() { return { state: this.state.toString(), inc: this.inc.toString(), has_uint32: this.hasUint32, uinteger: this.uinteger }; }

  next64() {
    this.state = (this.state * PCG_MULT + this.inc) & M128;
    const s = this.state, v = ((s >> 64n) ^ (s & M64)) & M64, r = Number(s >> 122n);
    return r === 0 ? v : ((v >> BigInt(r)) | (v << BigInt(64 - r))) & M64;
  }

  next32() {
    if (this.hasUint32) { this.hasUint32 = 0; return this.uinteger; }
    const next = this.next64();
    this.hasUint32 = 1;
    this.uinteger = Number(next >> 32n) >>> 0;
    return Number(next & 0xffffffffn) >>> 0;
  }

  /** A draw from 0 to n - 1. */
  integers(n) {
    const rng = (n | 0) - 1;
    if (rng <= 0) return 0;
    const excl = rng + 1;
    let m = this.next32() * excl, leftover = m % 4294967296;
    if (leftover < excl) {
      const threshold = (4294967296 - excl) % excl;
      while (leftover < threshold) { m = this.next32() * excl; leftover = m % 4294967296; }
    }
    return Math.floor(m / 4294967296);
  }
}

// ---------------------------------------------------------------- the records cortex

/**
 * cadence.Records: a reading of `inputs` units, each unit's running mean subtracted
 * (habituation), a fixed random expansion onto `cells` cells regenerated from the seed, the
 * `active` largest cells kept and normalised to unit length, and one record table per field.
 * A read is the sum of the active cells' records weighted by their activity; a write moves
 * exactly those records toward the witnessed outcome by the delta rule. The cortices of this
 * stage declare no pathways and no task sets, so one code serves every field and the valued
 * field differs only in its write rate.
 */
export class Records {
  constructor(inputs, fields, { cells = 2000, active = 20, rate = 0.2, valued = [], valuedRate = 1.0, habituation = 1e-5, bias = 0.3, seed = 0 } = {}) {
    this.inputs = inputs | 0;
    this.cells = cells | 0;
    this.active = active | 0;
    this.fields = fields.map((f) => ({ name: f.name, width: f.width | 0 }));
    this.valued = new Set(valued);
    this.rate = rate;
    this.valuedRate = valuedRate;
    this.habituation = habituation;
    this.bias = bias;
    this.seed = seed >>> 0;
    const generator = new Mulberry32(this.seed);
    const draws = generator.normals(this.inputs * this.cells), scale = Math.sqrt(this.inputs);
    this.projection = new Float64Array(this.inputs * this.cells);
    for (let k = 0; k < this.projection.length; k++) this.projection[k] = draws[k] / scale;
    const offsets = generator.normals(this.cells);
    this.offset = new Float64Array(this.cells);
    for (let g = 0; g < this.cells; g++) this.offset[g] = offsets[g] * this.bias;
    this.mean = new Float64Array(this.inputs);
    this.seen = 0;
    this.tables = {};
    for (const f of this.fields) this.tables[f.name] = new Float64Array(this.cells * f.width);
    this.writes = 0;
    this._drive = new Float64Array(this.cells);
    this._square = new Float64Array(this.active);
  }

  parameters() { let out = 0; for (const f of this.fields) out += this.tables[f.name].length; return out; }

  fieldOf(name) { return this.fields.find((f) => f.name === name); }

  /** The winner code of one row of a flat (batch, inputs) array: {index, values} over the kept cells. */
  _winners(x, offset) {
    const I = this.inputs, G = this.cells, drive = this._drive, W = this.projection;
    drive.set(this.offset);
    for (let i = 0; i < I; i++) {
      const xi = x[offset + i];
      if (xi === 0) continue;
      const row = i * G;
      for (let g = 0; g < G; g++) drive[g] += xi * W[row + g];
    }
    const index = topK(drive, Math.min(this.active, G)), values = new Float64Array(index.length), square = this._square;
    for (let k = 0; k < index.length; k++) { const v = drive[index[k]] > 0.0 ? drive[index[k]] : 0.0; values[k] = v; square[k] = v * v; }
    const norm = Math.sqrt(npSum(square, 0, index.length));
    if (norm > 0) { const d = Math.max(norm, 1e-12); for (let k = 0; k < values.length; k++) values[k] = values[k] / d; }
    return { index, values, cells: G };
  }

  /**
   * The codes of a flat (batch, inputs) array of readings, one {index, values} per row.
   * `adapt` first moves each unit's running mean by the rows (witnessed readings adapt,
   * imagined readings do not).
   */
  code(readings, batch, adapt = false, keys = null) {
    const I = this.inputs;
    if (readings.length !== batch * I) throw Error("readings must have one entry per input and row");
    const x = Float64Array.from(readings);
    if (this.habituation > 0) {
      if (adapt) {
        for (let b = 0; b < batch; b++) {
          this.seen += 1;
          const rate = Math.max(this.habituation, 1.0 / this.seen);
          for (let i = 0; i < I; i++) this.mean[i] += rate * (x[b * I + i] - this.mean[i]);
        }
      }
      for (let b = 0; b < batch; b++) for (let i = 0; i < I; i++) x[b * I + i] -= this.mean[i];
    }
    this.lastReadings = x;
    this.lastReading = x.subarray(0, I);
    // every row of one call shares the running mean, so two rows of the same reading have the
    // same code: `keys` names the reading of each row and the expansion runs once per reading
    const out = [], seen = keys === null ? null : new Map();
    for (let b = 0; b < batch; b++) {
      if (seen !== null) {
        const hit = seen.get(keys[b]);
        if (hit !== undefined) { out.push(hit); continue; }
      }
      const code = this._winners(x, b * I);
      if (seen !== null) seen.set(keys[b], code);
      out.push(code);
    }
    return out;
  }

  /** The read of one field through one code: the active cells' records, weighted by their activity. */
  read(code, name) {
    const f = this.fieldOf(name), table = this.tables[name], W = f.width, out = new Float64Array(W);
    for (let k = 0; k < code.index.length; k++) {
      const row = code.index[k] * W, c = code.values[k];
      if (c === 0) continue;
      for (let j = 0; j < W; j++) out[j] += c * table[row + j];
    }
    return out;
  }

  /** The delta rule on the active cells of one code; returns the fields written and their largest errors. */
  write(code, targets) {
    let written = 0;
    const report = [];
    for (const f of this.fields) {
      if (!(f.name in targets)) continue;
      const target = targets[f.name], W = f.width, table = this.tables[f.name];
      const read = this.read(code, f.name), error = new Float64Array(W);
      for (let j = 0; j < W; j++) error[j] = target[j] - read[j];
      const rate = this.valued.has(f.name) ? this.valuedRate : this.rate;
      for (let k = 0; k < code.index.length; k++) {
        const c = code.values[k];
        if (c === 0) continue;  // the write touches the records of the active cells only
        const row = code.index[k] * W;
        for (let j = 0; j < W; j++) table[row + j] += rate * (c * error[j]);
      }
      let largest = 0.0;
      for (let j = 0; j < W; j++) if (Math.abs(error[j]) > largest) largest = Math.abs(error[j]);
      report.push({ name: f.name, rate, reward: this.valued.has(f.name), valued: this.valued.has(f.name), error: largest });
      written += 1;
    }
    this.writes += written;
    return { written, report };
  }

  /**
   * The fixed cells and the learned state of an export block: the projection and the offsets
   * when the block carries them (a life recorded on another platform draws them a few last
   * bits apart, because log and cos round differently there), the habituation mean, the
   * tables, the readings seen and the write count. What the seed regenerates here is compared
   * with what arrives and the largest difference is kept in `regenerated`.
   */
  load(block) {
    this.regenerated = null;
    if (block.projection) {
      const projection = unpackF64(block.projection), offset = block.offset ? unpackF64(block.offset) : this.offset;
      if (projection.length !== this.inputs * this.cells) throw Error("the records projection does not match the cortex");
      if (offset.length !== this.cells) throw Error("the records offsets do not match the cortex");
      let worst = 0;
      for (let k = 0; k < projection.length; k++) { const d = Math.abs(projection[k] - this.projection[k]); if (d > worst) worst = d; }
      for (let g = 0; g < offset.length; g++) { const d = Math.abs(offset[g] - this.offset[g]); if (d > worst) worst = d; }
      this.regenerated = worst;
      this.projection = projection;
      this.offset = offset;
    } else if (block.probe) {
      // no arrays in the block: hold the regenerated expansion to the numbers the export took from it
      let sum = 0.0;
      for (const v of this.projection) sum += v;
      let offsets = 0.0;
      for (const v of this.offset) offsets += v;
      const seen = [this.projection[0], this.projection[this.projection.length - 1], sum], want = block.probe.projection;
      for (let k = 0; k < 3; k++) if (!(Math.abs(seen[k] - want[k]) <= 1e-9 * Math.max(1, Math.abs(want[k])))) throw Error(`the regenerated projection differs from the one the checkpoint was written with (${seen[k]} against ${want[k]})`);
      if (!(Math.abs(offsets - block.probe.offset[2]) <= 1e-9 * Math.max(1, Math.abs(block.probe.offset[2])))) throw Error("the regenerated offsets differ from the ones the checkpoint was written with");
      this.regenerated = 0;
    }
    if (block.mean) {
      const mean = unpackF64(block.mean);
      if (mean.length !== this.inputs) throw Error("the records mean does not match the reading");
      this.mean = mean;
    }
    for (const f of this.fields) {
      const spec = block.tables ? block.tables[f.name] : null;
      if (!spec) throw Error(`the records block lacks the table of ${f.name}`);
      const table = unpackF64(spec);
      if (table.length !== this.cells * f.width) throw Error(`the records table of ${f.name} does not match the cortex`);
      this.tables[f.name] = table;
    }
    if (block.seen !== undefined) this.seen = block.seen | 0;
    if (block.writes !== undefined) this.writes = block.writes | 0;
  }
}

// ---------------------------------------------------------------- the receptive fields

/** The supplied layout: the cells of each column bottom to top, and the windows of `connect` cells. */
export class Layout {
  constructor(game) {
    this.game = game;
    this.rows = game.rows; this.cols = game.cols; this.connect = game.connect; this.cells = game.cells;
    this.columns = [];
    for (let c = 0; c < game.cols; c++) { const column = new Int32Array(game.rows); for (let r = 0; r < game.rows; r++) column[r] = r * game.cols + c; this.columns.push(column); }
    const windows = [];
    for (const [dr, dc] of [[0, 1], [1, 0], [1, 1], [1, -1]]) {
      for (let r = 0; r < game.rows; r++) {
        for (let c = 0; c < game.cols; c++) {
          const spots = [];
          let fits = true;
          for (let k = 0; k < game.connect; k++) {
            const rr = r + k * dr, cc = c + k * dc;
            if (rr < 0 || rr >= game.rows || cc < 0 || cc >= game.cols) { fits = false; break; }
            spots.push(rr * game.cols + cc);
          }
          if (fits) windows.push(Int32Array.from(spots));
        }
      }
    }
    this.windows = windows;
    this.top = Int32Array.from({ length: game.cols }, (_, c) => (game.rows - 1) * game.cols + c);
    this.columnPowers = Int32Array.from({ length: game.rows }, (_, r) => 3 ** r);
    this.windowPowers = Int32Array.from({ length: game.connect }, (_, k) => 3 ** k);
    this.columnPatterns = 3 ** game.rows;
    this.windowPatterns = 3 ** game.connect;
    this.columnOf = Int32Array.from({ length: game.cells }, (_, i) => i % game.cols);
    this.rowOf = Int32Array.from({ length: game.cells }, (_, i) => Math.floor(i / game.cols));
    this.floorCells = Uint8Array.from({ length: game.cells }, (_, i) => (i - game.cols < 0 ? 1 : 0));
    this.belowCell = Int32Array.from({ length: game.cells }, (_, i) => (i - game.cols < 0 ? 0 : i - game.cols));
    this.throughWindows = [];  // the windows through each cell
    for (let cell = 0; cell < game.cells; cell++) this.throughWindows.push(Int32Array.from(windows.map((w, k) => (w.includes(cell) ? k : -1)).filter((k) => k >= 0)));
  }

  /** Column pattern ids of a board seen by one side. */
  columnIds(rel, out = new Int32Array(this.cols)) {
    for (let c = 0; c < this.cols; c++) {
      const column = this.columns[c];
      let id = 0;
      for (let r = 0; r < this.rows; r++) id += rel[column[r]] * this.columnPowers[r];
      out[c] = id;
    }
    return out;
  }

  /** Window pattern ids of a board seen by one side. */
  windowIds(rel, out = new Int32Array(this.windows.length)) {
    for (let w = 0; w < this.windows.length; w++) {
      const cells = this.windows[w];
      let id = 0;
      for (let k = 0; k < this.connect; k++) id += rel[cells[k]] * this.windowPowers[k];
      out[w] = id;
    }
    return out;
  }
}

/** The board as `side` sees it: 0 empty, 1 its own stone, 2 the other's. */
export function viewOf(side, board, out = new Int8Array(board.length)) {
  const view = VIEW[side];
  for (let i = 0; i < board.length; i++) out[i] = view[board[i]];
  return out;
}

/** The categorical reading of pattern ids: one one-hot of width 3 per base-3 digit. */
export function oneHotDigits(ids, digits, out = null) {
  const n = ids.length, width = 3 * digits;
  const rows = out || new Float64Array(n * width);
  rows.fill(0);
  for (let k = 0; k < n; k++) {
    let id = ids[k];
    for (let j = 0; j < digits; j++) { rows[k * width + j * 3 + (id % 3)] = 1.0; id = Math.floor(id / 3); }
  }
  return rows;
}

// ---------------------------------------------------------------- the two cortices

/** Records over column readings; `landing` is the row a dropped stone occupies, or none. */
export class DropRecords {
  constructor(layout, cfg, seed) {
    this.layout = layout;
    this.gain = Number(cfg.chosen_gain);
    this.records = new Records(3 * layout.rows + 1, [{ name: "landing", width: layout.rows + 1 }], {
      cells: cfg.cells | 0, active: cfg.active | 0, rate: Number(cfg.rate), habituation: Number(cfg.habituation), bias: Number(cfg.bias), seed,
    });
    this.width = layout.rows + 1;
    this._reads = new Float64Array(2 * layout.columnPatterns * this.width);
    this._known = new Uint8Array(2 * layout.columnPatterns);
    this._ids = new Int32Array(layout.cols);
    this._pair = new Int32Array(2 * layout.cols);
  }

  /** Readings of column ids: the content digits as one-hot and the chosen flag (an id offset). */
  readings(ids) {
    const rows = this.layout.rows, patterns = this.layout.columnPatterns, width = 3 * rows + 1;
    const plain = new Int32Array(ids.length);
    for (let k = 0; k < ids.length; k++) plain[k] = ids[k] % patterns;
    const out = new Float64Array(ids.length * width);
    const digits = oneHotDigits(plain, rows);
    for (let k = 0; k < ids.length; k++) {
      out.set(digits.subarray(k * 3 * rows, (k + 1) * 3 * rows), k * width);
      out[k * width + 3 * rows] = Math.floor(ids[k] / patterns) * this.gain;
    }
    return out;
  }

  /** The landing reads of column ids, cached until the records change. */
  reads(ids) {
    const need = [];
    for (const id of ids) if (!this._known[id] && !need.includes(id)) need.push(id);
    if (need.length) {
      need.sort((a, b) => a - b);
      const codes = this.records.code(this.readings(Int32Array.from(need)), need.length, false);
      for (let k = 0; k < need.length; k++) {
        const read = this.records.read(codes[k], "landing");
        this._reads.set(read, need[k] * this.width);
        this._known[need[k]] = 1;
      }
    }
    return this._reads;
  }

  /** The predicted landing row per column of one board seen by the side to move, unchosen and chosen. */
  landings(rel) {
    const lay = this.layout, ids = lay.columnIds(rel, this._ids), pair = this._pair;
    for (let c = 0; c < lay.cols; c++) { pair[c] = ids[c]; pair[lay.cols + c] = ids[c] + lay.columnPatterns; }
    const reads = this.reads(pair), unchosen = new Int32Array(lay.cols), chosen = new Int32Array(lay.cols);
    for (let c = 0; c < lay.cols; c++) {
      unchosen[c] = argmaxOf(reads, pair[c] * this.width, this.width);
      chosen[c] = argmaxOf(reads, pair[lay.cols + c] * this.width, this.width);
    }
    return [unchosen, chosen];
  }

  /** One witnessed move: the board before it seen by the mover, the chosen column, the observed row per column. */
  learn(rel, column, rows, adapt, report = null) {
    const lay = this.layout, ids = lay.columnIds(rel);
    for (let c = 0; c < lay.cols; c++) if (c === column) ids[c] += lay.columnPatterns;
    const codes = this.records.code(this.readings(ids), lay.cols, adapt, ids);
    if (report) report.code(this, codes[column], this.records.lastReadings.subarray(column * this.records.inputs, (column + 1) * this.records.inputs), { column });
    const eye = [];
    for (let r = 0; r <= lay.rows; r++) { const one = new Float64Array(lay.rows + 1); one[r] = 1.0; eye.push(one); }
    for (let j = 0; j < lay.cols; j++) {
      const out = this.records.write(codes[j], { landing: eye[rows[j]] });
      if (report) report.write(this, codes[j], out, j === column);
    }
    this._known.fill(0);
    return lay.cols;
  }
}

/** Records over window readings: `complete` at the consequence rate, the valued `value` at the value rate. */
export class LineRecords {
  constructor(layout, cfg, seed) {
    this.layout = layout;
    this.records = new Records(3 * layout.connect, [{ name: "complete", width: 2 }, { name: "value", width: 1 }], {
      cells: cfg.cells | 0, active: cfg.active | 0, rate: Number(cfg.rate), valued: ["value"], valuedRate: Number(cfg.value_rate),
      habituation: Number(cfg.habituation), bias: Number(cfg.bias), seed,
    });
    this.patternIds = Int32Array.from({ length: layout.windowPatterns }, (_, i) => i);
    this.patterns = oneHotDigits(this.patternIds, layout.connect);
    this.complete = new Float64Array(layout.windowPatterns);
    this.value = new Float64Array(layout.windowPatterns).fill(0.5);
    this._fresh = false;
  }

  /** The reads of every window pattern under the current records. */
  refresh() {
    if (this._fresh) return;
    const codes = this.records.code(this.patterns, this.layout.windowPatterns, false);
    for (let i = 0; i < codes.length; i++) {
      const complete = this.records.read(codes[i], "complete"), value = this.records.read(codes[i], "value");
      const a = complete[0] > 0 ? complete[0] : 0.0, b = complete[1] > 0 ? complete[1] : 0.0, total = a + b;
      this.complete[i] = total > 0 ? b / Math.max(total, 1e-12) : 0.0;
      this.value[i] = 0.5 + value[0];
    }
    this._fresh = true;
  }

  /** The readings of a board's windows: the pattern rows of its window ids. */
  _readingsOf(ids) {
    const width = 3 * this.layout.connect, out = new Float64Array(ids.length * width);
    for (let k = 0; k < ids.length; k++) out.set(this.patterns.subarray(ids[k] * width, (ids[k] + 1) * width), k * width);
    return out;
  }

  /**
   * One observed board after a move, seen by the mover and by the other side. A board the game
   * went on from has no completed window; a won board has at least one completed window of the
   * mover among the windows through the new stone.
   */
  learnLines(relMover, newCell, won, adapt, report = null) {
    const lay = this.layout;
    let written = 0;
    const through = new Uint8Array(lay.windows.length);
    for (const w of lay.throughWindows[newCell]) through[w] = 1;
    const views = [[true, relMover], [false, viewOf(OPPONENT, relMover)]];
    for (const [moverView, rel] of views) {
      const ids = lay.windowIds(rel);
      const codes = this.records.code(this._readingsOf(ids), ids.length, adapt, ids);
      const first = new Map();
      for (let k = ids.length - 1; k >= 0; k--) first.set(ids[k], k);  // the lowest index of each pattern id
      let incomplete = new Set(first.keys());
      if (moverView && won) {
        const elsewhere = new Set();
        for (let k = 0; k < ids.length; k++) if (!through[k]) elsewhere.add(ids[k]);
        const candidates = [];
        for (let k = 0; k < ids.length; k++) if (through[k] && !elsewhere.has(ids[k]) && !candidates.includes(ids[k])) candidates.push(ids[k]);
        candidates.sort((a, b) => a - b);
        incomplete = elsewhere;
        if (candidates.length) {
          const probability = candidates.map((id) => {
            const read = this.records.read(codes[first.get(id)], "complete");
            const a = read[0] > 0 ? read[0] : 0.0, b = read[1] > 0 ? read[1] : 0.0;
            return b / Math.max(a + b, 1e-12);
          });
          const top = Math.max(...probability);
          for (let k = 0; k < candidates.length; k++) {
            if (probability[k] >= top - TIE) {
              const code = codes[first.get(candidates[k])];
              const out = this.records.write(code, { complete: Float64Array.from([0.0, 1.0]) });
              if (report) report.write(this, code, out, moverView);
              written += 1;
            }
          }
        }
      }
      const rest = Array.from(incomplete).sort((a, b) => a - b);
      if (report && moverView) {
        const shown = [], width = this.records.inputs;
        let reading = this.records.lastReadings.subarray(0, width);
        for (let k = 0; k < ids.length; k++) if (through[k]) { if (!shown.length) reading = this.records.lastReadings.subarray(k * width, (k + 1) * width); shown.push(codes[k]); }
        report.lineCodes(this, shown, reading);
      }
      for (const id of rest) {
        const code = codes[first.get(id)];
        const out = this.records.write(code, { complete: Float64Array.from([1.0, 0.0]) });
        if (report) report.write(this, code, out, moverView);
        written += 1;
      }
    }
    this._fresh = false;
    return written;
  }

  /** A finished game: each window pattern of the game once, toward the mean outcome of the boards that held it. */
  learnValues(boards, report = null) {
    const total = new Map(), count = new Map();
    for (const [ids, outcome] of boards) {
      const seen = new Set();
      for (const id of ids) seen.add(id);
      for (const id of seen) { total.set(id, (total.get(id) || 0) + outcome); count.set(id, (count.get(id) || 0) + 1); }
    }
    if (!total.size) return 0;
    const order = Array.from(total.keys()).sort((a, b) => a - b);
    const codes = this.records.code(this._readingsOf(Int32Array.from(order)), order.length, false);
    for (let k = 0; k < order.length; k++) {
      const out = this.records.write(codes[k], { value: Float64Array.from([total.get(order[k]) / count.get(order[k]) - 0.5]) });
      if (report) report.write(this, codes[k], out, true);
    }
    this._fresh = false;
    return order.length;
  }
}

// ---------------------------------------------------------------- imagination

/** Imagined consequences read from the records: composed boards, their validity, terminal status and value. Reads only. */
export class Imagination {
  constructor(layout, drop, lines) {
    this.layout = layout; this.drop = drop; this.lines = lines;
    this.threatWeight = 0.0;  // 0 keeps the plain window mean as the leaf value
    this.parityWeight = 0.0;  // the rows a standing threat has to stand on to be reached
    this._rel = new Int8Array(layout.cells);
    this._mine = new Uint8Array(layout.cells);
    this._theirs = new Uint8Array(layout.cells);
  }

  /**
   * The cells that complete a line for the side read as 1 in `rel`, read from the learned complete
   * field alone: a window holding one empty cell whose filled pattern the line records rate as a
   * completed line puts a threat on that cell.
   */
  threatCells(rel, out) {
    const lay = this.layout;
    out.fill(0);
    for (let w = 0; w < lay.windows.length; w++) {
      const cells = lay.windows[w];
      let id = 0, empties = 0, spot = -1;
      for (let k = 0; k < lay.connect; k++) {
        const v = rel[cells[k]];
        id += v * lay.windowPowers[k];
        if (v === 0) { empties += 1; if (spot < 0) spot = k; }
      }
      if (empties !== 1) continue;  // a full window has no spot: the lookup is masked, as in Python
      if (this.lines.complete[id + lay.windowPowers[spot]] > 0.5) out[cells[spot]] = 1;
    }
    return out;
  }

  /** How the threats stand on `board` just moved by `mover`: reachable at once, and in total, per side. */
  threatSummary(board, mover) {
    const lay = this.layout, other = mover === CANDIDATE ? OPPONENT : CANDIDATE;
    const mine = this.threatCells(viewOf(mover, board, this._rel), this._mine);
    const theirs = this.threatCells(viewOf(other, board, this._rel), this._theirs);
    let mineNow = 0, mineAll = 0, theirsNow = 0, theirsAll = 0;
    for (let i = 0; i < lay.cells; i++) {
      const playable = board[i] === 0 && (lay.floorCells[i] === 1 || board[lay.belowCell[i]] !== 0);
      if (mine[i]) { mineAll += 1; if (playable) mineNow += 1; }
      if (theirs[i]) { theirsAll += 1; if (playable) theirsNow += 1; }
    }
    return { mineNow, mineAll, theirsNow, theirsAll };
  }

  /**
   * Standing threats on the rows that fall to each side when the columns fill up: the side that
   * opened takes the odd rows counted from the floor (row 0, 2, 4), the other side the even ones.
   */
  parityThreats(board, mover) {
    const lay = this.layout, other = mover === CANDIDATE ? OPPONENT : CANDIDATE;
    let stones = 0;
    for (let i = 0; i < lay.cells; i++) if (board[i] !== 0) stones += 1;
    const moverOpened = stones % 2 !== 0;  // the opener moves on even counts; the mover has just moved
    const mine = this.threatCells(viewOf(mover, board, this._rel), this._mine);
    const theirs = this.threatCells(viewOf(other, board, this._rel), this._theirs);
    let mineOdd = 0, theirsOdd = 0;
    for (let i = 0; i < lay.cells; i++) {
      const openerRow = lay.rowOf[i] % 2 === 0;
      const mineRow = moverOpened ? openerRow : !openerRow;
      if (mine[i] && mineRow) mineOdd += 1;
      if (theirs[i] && !mineRow) theirsOdd += 1;
    }
    return { mineOdd, theirsOdd };
  }

  /**
   * The leaf value of `board` just moved by `mover` with the threats that stand on it: one threat
   * of the other side's playable at once decides the position, two reachable threats of the mover's
   * decide it the other way, and otherwise the standing threats tilt the window mean.
   */
  threatValue(board, mover, base) {
    const { mineNow, mineAll, theirsNow, theirsAll } = this.threatSummary(board, mover);
    const w = this.threatWeight;
    let tilt = Math.min(Math.max(Math.min(mineAll, 3) - Math.min(theirsAll, 3), -3), 3) / 3.0;
    if (this.parityWeight) {
      const { mineOdd, theirsOdd } = this.parityThreats(board, mover);
      tilt = tilt + this.parityWeight * Math.min(Math.max(mineOdd - theirsOdd, -2), 2) / 2.0;
    }
    let value = clamp(base + w * 0.5 * tilt, LEAF_MARGIN, 1.0 - LEAF_MARGIN);
    if (mineNow >= 2) value = 1.0 - LEAF_MARGIN;
    if (theirsNow >= 1) value = LEAF_MARGIN;
    return value;
  }

  /** The columns of an imagined board whose top cell is empty. */
  legal(board) {
    const out = [];
    for (let c = 0; c < this.layout.cols; c++) if (board[this.layout.top[c]] === 0) out.push(c);
    return Int32Array.from(out);
  }

  /** The imagined boards after `side` drops a stone in each of `columns`, and whether the validator accepts each. */
  compose(board, side, columns) {
    const lay = this.layout, n = columns.length;
    const [unchosen, chosen] = this.drop.landings(viewOf(side, board));
    const children = [];
    for (let k = 0; k < n; k++) children.push(Int8Array.from(board));
    for (let j = 0; j < lay.cols; j++) {
      for (let k = 0; k < n; k++) {
        const row = columns[k] === j ? chosen[j] : unchosen[j];
        if (row < lay.rows) children[k][row * lay.cols + j] = side;
      }
    }
    const valid = new Uint8Array(n);
    for (let k = 0; k < n; k++) {
      let changed = 0, cell = 0;
      for (let i = 0; i < lay.cells; i++) if (children[k][i] !== board[i]) { if (changed === 0) cell = i; changed += 1; }
      valid[k] = changed === 1 && lay.columnOf[cell] === columns[k] && board[cell] === 0 ? 1 : 0;
    }
    return { children, valid };
  }

  /** For boards just moved by `mover`: a completed window of the mover, a full board, and the value for the mover. */
  outcome(boards, mover) {
    this.lines.refresh();
    const lay = this.layout, n = boards.length;
    const won = new Uint8Array(n), full = new Uint8Array(n), value = new Float64Array(n);
    const ids = new Int32Array(lay.windows.length), buffer = new Float64Array(lay.windows.length), rel = new Int8Array(lay.cells);
    for (let k = 0; k < n; k++) {
      viewOf(mover, boards[k], rel);
      lay.windowIds(rel, ids);
      let top = -Infinity, empty = false;
      for (let w = 0; w < ids.length; w++) { const c = this.lines.complete[ids[w]]; if (c > top) top = c; buffer[w] = this.lines.value[ids[w]]; }
      for (let i = 0; i < lay.cells; i++) if (boards[k][i] === 0) { empty = true; break; }
      won[k] = top > 0.5 ? 1 : 0;
      full[k] = empty ? 0 : 1;
      value[k] = npSum(buffer, 0, ids.length) / ids.length;
      if (this.threatWeight) value[k] = this.threatValue(boards[k], mover, value[k]);
    }
    return { won, full, value };
  }

  /** The value of a board for `side`, the side to move (the other side just moved). */
  static_(board, side) {
    const { value } = this.outcome([board], side === CANDIDATE ? OPPONENT : CANDIDATE);
    return clamp(1.0 - value[0], LEAF_MARGIN, 1.0 - LEAF_MARGIN);
  }

  /** The learned value of each column's imagined board for `side`, without search. */
  oneStep(board, side, columns) {
    const { children, valid } = this.compose(board, side, columns);
    const { won, full, value } = this.outcome(children, side);
    const out = new Float64Array(columns.length);
    let fallback = null;
    for (let k = 0; k < columns.length; k++) {
      if (!valid[k]) { if (fallback === null) fallback = this.static_(board, side); out[k] = fallback; continue; }
      out[k] = won[k] ? 1.0 : full[k] ? 0.5 : clamp(value[k], LEAF_MARGIN, 1.0 - LEAF_MARGIN);
    }
    return out;
  }
}

const SPENT = { spent: true };  // the decision's budget of imagined transitions is spent

/** Negamax with alpha-beta pruning and iterative deepening over imagined boards. */
export class Planner {
  constructor(imagination, generator) {
    this.imagination = imagination;
    this.rng = generator;
    this.nodes = 0; this.searches = 0; this.invalid = 0;
    this.depths = new Map();
    this.budget = 0;
    this._cache = new Map();
    this._table = new Map();  // board+side -> {depth, value, bound, best}: what earlier visits established
    this.tieBand = TIE;  // root values this close count as equal; the column nearest the middle is played
    this.memory = new Map();  // board+side to move -> a proven result: 1 win, 0.5 draw, 0 loss
    this.remember = false;  // write what a search proves into the memory and read it back at every visit
    this.memoryLimit = 200000;
    this._exact = false;  // whether the last _negamax value was proven
    this._spent = 0;
    this._limit = 0;
  }

  get depth() {
    let n = 0, total = 0;
    for (const [d, k] of this.depths) { n += k; total += d * k; }
    return n ? total / n : 0.0;
  }

  stats() {
    const depths = {};
    for (const d of Array.from(this.depths.keys()).sort((a, b) => a - b)) depths[String(d)] = this.depths.get(d);
    return { nodes: this.nodes, searches: this.searches, invalid: this.invalid, depth: this.depth, depths, budget: this.budget, memory: this.memory.size };
  }

  _key(board, side) { return String.fromCharCode.apply(null, board) + side; }

  _expand(board, side) {
    const key = this._key(board, side), hit = this._cache.get(key);
    if (hit !== undefined) return hit;
    const im = this.imagination, columns = im.legal(board);
    if (this._spent + columns.length > this._limit) throw SPENT;
    this._spent += columns.length;
    const { children, valid } = im.compose(board, side, columns);
    for (let k = 0; k < valid.length; k++) if (!valid[k]) this.invalid += 1;
    const { won, full, value } = im.outcome(children, side);
    const { lost, wins } = this._decided(children, side, won, full);
    const known = new Float64Array(columns.length).fill(NaN);  // the memory's result for the side to move on each child
    if (this.remember && this.memory.size) {
      const other = side === CANDIDATE ? OPPONENT : CANDIDATE;
      for (let k = 0; k < columns.length; k++) {
        if (!valid[k] || won[k] || full[k] || lost[k] || wins[k]) continue;
        const hit = this.memory.get(this._key(children[k], other));
        if (hit !== undefined) known[k] = 1.0 - hit;
      }
    }
    const clipped = new Float64Array(value.length);
    for (let k = 0; k < value.length; k++) clipped[k] = clamp(value[k], LEAF_MARGIN, 1.0 - LEAF_MARGIN);
    // the move order: proven wins first, then the higher value, then the column nearest the middle (np.lexsort)
    const centre = (im.layout.cols - 1) / 2;
    const primary = new Float64Array(columns.length), secondary = new Float64Array(columns.length);
    for (let k = 0; k < columns.length; k++) { primary[k] = -(won[k] ? 2.0 : wins[k] ? 1.9 : lost[k] ? -1.0 : Number.isNaN(known[k]) ? value[k] : known[k]); secondary[k] = Math.abs(columns[k] - centre); }
    const order = Int32Array.from({ length: columns.length }, (_, k) => k).sort((a, b) => (primary[a] - primary[b]) || (secondary[a] - secondary[b]) || (a - b));
    const entry = { columns, children, valid, won, full, value: clipped, order, lost, wins, known };
    this._cache.set(key, entry);
    return entry;
  }

  /** Keep a proven result; when the memory is full, the positions with the most stones go first,
   *  since the search re-proves them cheapest. */
  _remember(key, result) {
    if (this.memory.size >= this.memoryLimit && !this.memory.has(key)) {
      const stones = (k) => { let n = 0; for (let i = 0; i < k.length - 1; i++) if (k.charCodeAt(i) !== 0) n++; return n; };
      const keys = Array.from(this.memory.keys()).sort((a, b) => stones(b) - stones(a));
      for (const k of keys.slice(0, Math.floor(this.memoryLimit / 10))) this.memory.delete(k);
    }
    this.memory.set(key, result);
  }

  /** Children the threats standing on them decide before any expansion: lost when the other side
   *  can complete a line at once, won when the mover holds two threats it can reach at once and the
   *  other side none. Read off the records; off with the threat weight. */
  _decided(children, side, won, full) {
    const n = children.length, lost = new Uint8Array(n), wins = new Uint8Array(n);
    if (!this.imagination.threatWeight) return { lost, wins };
    for (let k = 0; k < n; k++) {
      if (won[k] || full[k]) continue;
      const { mineNow, theirsNow } = this.imagination.threatSummary(children[k], side);
      if (theirsNow >= 1) lost[k] = 1;
      else if (mineNow >= 2) wins[k] = 1;
    }
    return { lost, wins };
  }

  /** The value of `board` for `side` to move, searched `depth` plies. */
  _negamax(board, side, depth, alpha, beta, ply) {
    const expanded = this._expand(board, side);
    const { columns, children, valid, won, full, value, lost, wins, known } = expanded;
    let order = expanded.order;
    if (!columns.length) { this._exact = true; return 0.5; }
    const key = this._key(board, side), entry = this._table.get(key);
    const alpha0 = alpha;
    if (entry !== undefined) {
      if (entry.depth >= depth) {
        if (entry.bound === 0) { this._exact = false; return entry.value; }
        if (entry.bound === 1) { if (entry.value > alpha) alpha = entry.value; }
        else if (entry.bound === -1) { if (entry.value < beta) beta = entry.value; }
        if (alpha >= beta) { this._exact = false; return entry.value; }
      }
      order = [entry.best, ...Array.from(order).filter((k) => k !== entry.best)];
    }
    const other = side === CANDIDATE ? OPPONENT : CANDIDATE;
    let best = -1.0, bestK = order[0], bestExact = false, allExact = true, cut = false, uncertain = null;
    for (const k of order) {
      let v, exact = true;
      if (!valid[k]) { if (uncertain === null) uncertain = this.imagination.static_(board, side); v = uncertain; exact = false; }
      else if (won[k]) v = 1.0 - WIN_BONUS * ply;
      else if (full[k]) v = 0.5;
      else if (lost[k]) v = WIN_BONUS * (ply + 1);  // the other side completes a line next
      else if (wins[k]) v = 1.0 - WIN_BONUS * (ply + 2);  // two threats to reach, one block
      else if (!Number.isNaN(known[k])) v = provenValue(known[k], ply + 1);  // what an earlier search proved of this child
      else if (depth <= 1) { v = value[k]; exact = false; }
      else { v = 1.0 - this._negamax(children[k], other, depth - 1, 1.0 - beta, 1.0 - alpha, ply + 1); exact = this._exact; }
      if (v > best) { best = v; bestK = k; bestExact = exact; }
      allExact = allExact && exact;
      if (v > alpha) alpha = v;
      if (alpha >= beta) { cut = true; break; }
    }
    // the value is proven when its best child is a proven win, or every child is proven and none was cut off
    this._exact = bestExact && (best > 0.9 || (allExact && !cut));
    if (this._exact && this.remember) this._remember(key, plainResult(best));
    const bound = best <= alpha0 ? -1 : best >= beta ? 1 : 0;  // an upper bound, a lower bound, or the value
    this._table.set(key, { depth, value: best, bound, best: bestK });
    return best;
  }

  /** The candidate's column for `board` and what the search saw. */
  search(board, legal, depth, budget) {
    this._cache = new Map();
    this._table = new Map();
    this._spent = 0;
    this._limit = budget | 0;
    this.budget = budget | 0;
    const invalidBefore = this.invalid;
    const { columns, children, valid, won, full, value, order, lost, wins, known } = this._expand(board, CANDIDATE);
    const rootExact = new Map();
    const rank = new Map();
    order.forEach((k, i) => rank.set(columns[k], i));
    const root = [];
    for (let c = 0; c < legal.length; c++) if (legal[c]) root.push(c);
    let choice = [...root], completed = 0, rootValue = 0.5;
    let values = new Map(), sequence = [...root];
    for (let d = 1; d <= depth; d++) {
      // the order of the next deepening: the best value of the last one first, then the expansion's order
      sequence = [...root].sort((a, b) => ((values.get(b) || 0.0) - (values.get(a) || 0.0)) || ((rank.get(a) || 0) - (rank.get(b) || 0)) || (a - b));
      const found = new Map();
      let best = -1.0, spent = false;
      try {
        for (const c of sequence) {
          let k = -1;
          for (let j = 0; j < columns.length; j++) if (columns[j] === c) { k = j; break; }
          let v, exact = true;
          if (!valid[k]) { v = this.imagination.static_(board, CANDIDATE); exact = false; }
          else if (won[k]) v = 1.0 - WIN_BONUS;
          else if (full[k]) v = 0.5;
          else if (lost[k]) v = WIN_BONUS * 2;
          else if (wins[k]) v = 1.0 - WIN_BONUS * 3;
          else if (!Number.isNaN(known[k])) v = provenValue(known[k], 2);
          else if (d === 1) { v = value[k]; exact = false; }
          // a move cut off by this window is worse than the best by more than a tie
          else { v = 1.0 - this._negamax(children[k], OPPONENT, d - 1, 0.0, 1.0 - (best - 2 * TIE), 2); exact = this._exact; }
          found.set(c, v);
          rootExact.set(c, exact);
          if (v > best) best = v;
        }
      } catch (error) { if (error !== SPENT) throw error; spent = true; }
      if (spent) break;
      values = found;
      let top = -Infinity;
      for (const v of values.values()) if (v > top) top = v;
      if (this.remember && values.size === root.length) {
        let topC = root[0];
        for (const [c, v] of values) if (v > values.get(topC)) topC = c;
        if (rootExact.get(topC) && (top > 0.9 || root.every((c) => rootExact.get(c)))) this._remember(this._key(board, CANDIDATE), plainResult(top));
      }
      choice = sequence.filter((c) => values.get(c) >= top - this.tieBand);
      if (this.tieBand > TIE) {
        const centre = (this.imagination.layout.cols - 1) / 2;
        choice = [choice.slice().sort((a, b) => (Math.abs(a - centre) - Math.abs(b - centre)) || (a - b))[0]];  // the middle among near-equal columns
      }
      completed = d;
      rootValue = top;
      if (top >= 1.0 - WIN_BONUS * depth - TIE || top <= WIN_BONUS * depth + TIE) break;  // a proven result
    }
    const column = choice[this.rng.integers(choice.length)];
    let k = -1;
    for (let j = 0; j < columns.length; j++) if (columns[j] === column) { k = j; break; }
    this.searches += 1;
    this.nodes += this._spent;
    this.depths.set(completed, (this.depths.get(completed) || 0) + 1);
    // the value the records read for each legal column without search (Imagination.one_step)
    const read = new Map();
    let fallback = null;
    for (let j = 0; j < columns.length; j++) {
      if (!valid[j]) { if (fallback === null) fallback = this.imagination.static_(board, CANDIDATE); read.set(columns[j], fallback); }
      else read.set(columns[j], won[j] ? 1.0 : full[j] ? 0.5 : value[j]);
    }
    const info = {
      value: rootValue, depth: completed, expansions: this._spent, invalid: this.invalid - invalidBefore,
      next_board: Float64Array.from(children[k]), valid: valid[k] ? 1.0 : 0.0,
      column, columns: [...root], searched: root.map((c) => (values.has(c) ? values.get(c) : null)),
      read: root.map((c) => read.get(c)), sequence: [...sequence], choice: [...choice], budget: this.budget,
    };
    this._cache = new Map();
    this._table = new Map();
    return info;
  }
}

// ---------------------------------------------------------------- the brain

/** The side that moves next: from the goal when given, else the other side of `mover`. */
export function sideToMove(observation, goal) {
  if (goal) return goal[0] >= goal[1] ? CANDIDATE : OPPONENT;
  return argmaxOf(observation.mover) === MOVER_CANDIDATE ? OPPONENT : CANDIDATE;
}

/**
 * The candidate behind the one `step` contract: the record cortices, the imagination and the
 * planner of brain.py, with the S00 agent of web/engine.js as the event transaction.
 */
export class Brain {
  constructor(checkpoint, options = {}) {
    if (checkpoint.format !== CHECKPOINT_FORMAT) throw Error(`not a ${CHECKPOINT_FORMAT} checkpoint`);
    const block = checkpoint.brain;
    if (block.format !== BRAIN_FORMAT) throw Error(`not a ${BRAIN_FORMAT} brain block`);
    this.block = block;
    this.seed = block.seed | 0;
    this.learning = options.learning === undefined ? !!block.learning : !!options.learning;
    this.game = gameConfig(block.game);
    this.layout = new Layout(this.game);
    this.cfg = block.brain;
    this.report = options.report || null;  // a page's records view: {code, lineCodes, write}
    this.drop = new DropRecords(this.layout, this.cfg.drop_records, block.cortices.drop.seed);
    this.lines = new LineRecords(this.layout, this.cfg.line_records, block.cortices.lines.seed);
    const cortices = options.cortices || checkpoint.cortices || null;  // the fixed expansion of this life, shared by its checkpoints
    this.drop.records.load({ ...block.cortices.drop, ...(cortices ? cortices.drop : {}) });
    this.lines.records.load({ ...block.cortices.lines, ...(cortices ? cortices.lines : {}) });
    this.expansion = { drop: this.drop.records.regenerated, lines: this.lines.records.regenerated };
    this.imagination = new Imagination(this.layout, this.drop, this.lines);
    this.imagination.threatWeight = Number((this.cfg.planner || {}).threat_weight || 0);
    this.imagination.parityWeight = Number((this.cfg.planner || {}).parity_weight || 0);
    this.planner = new Planner(this.imagination, new PCG64(block.planner_generator));
    this.planner.tieBand = Number((this.cfg.planner || {}).tie_band || TIE);
    this.planner.remember = !!(this.cfg.planner || {}).remember && this.learning;
    this.planner.memoryLimit = Number((this.cfg.planner || {}).memory_limit || 200000);
    if (block.memory && block.memory.entries) {
      const boards = unpackArray(block.memory.boards), sides = unpackArray(block.memory.sides), results = unpackArray(block.memory.results);
      const n = this.layout.cells;
      for (let i = 0; i < block.memory.entries; i++) this.planner.memory.set(this.planner._key(boards.subarray(i * n, (i + 1) * n), sides[i]), results[i]);
    }
    this.depth = block.planning.depth | 0;
    this.budget = block.planning.budget | 0;
    this.extendedDepth = block.planning.extended_depth | 0;
    this.extendedBudget = block.planning.extended_budget | 0;
    this.validityGate = Number(block.validity_gate);
    this.validityWindow = this.cfg.planner.validity_window | 0;
    this.validityMin = this.cfg.planner.validity_min | 0;
    this.validity = (block.validity || []).slice(-this.validityWindow).map((v) => !!v);
    this.counts = { ...block.counts };
    this.planner.nodes = block.planner.nodes | 0;
    this.planner.searches = block.planner.searches | 0;
    this.planner.invalid = block.planner.invalid | 0;
    for (const [d, k] of Object.entries(block.planner.depths || {})) this.planner.depths.set(Number(d), k | 0);
    this.agent = new ExperienceAgent(checkpoint.agent, { ...options.engine, planner: (agent, row, moment, drive, free) => this._plan(agent, row, moment, drive, free) });
    this.lastSearch = null;
    this._episode = -1;
    this._board = null;
    this._held = null;
    this._boards = [];
  }

  parameters() { return { drop_records: this.drop.records.parameters(), line_records: this.lines.records.parameters() }; }

  get extended() {
    if (this.validity.length < this.validityMin) return false;
    let sum = 0;
    for (const v of this.validity) sum += v ? 1 : 0;
    return sum / this.validity.length >= this.validityGate;
  }

  state() {
    let mean = null;
    if (this.validity.length) { let sum = 0; for (const v of this.validity) sum += v ? 1 : 0; mean = sum / this.validity.length; }
    return { ...this.counts, validity: mean, validity_samples: this.validity.length, extended: this.extended, planner: this.planner.stats() };
  }

  /** Attach to another world: fresh event cursors and no board from the last one. */
  newWorld() {
    this.agent.newStream(0);
    this._episode = -1; this._board = null; this._held = null; this._boards = [];
  }

  setLearning(value) {
    this.learning = !!value;
    this.agent.config.learning = !!value;
  }

  // -- the event stream

  step(moment) {
    const newEpisode = moment.episode_id !== this._episode;
    const intermediate = moment.feedback_for !== null && moment.feedback_for !== undefined && !moment.terminated && !moment.truncated;
    if (newEpisode && this._held !== null) throw Error("a decision awaits its reply moment; a new episode cannot start");
    const cells = cellsOf(moment.observation, this.game);
    if (newEpisode) { this._episode = moment.episode_id; this._board = null; this._boards = []; }
    else if (this._board !== null) this._observe(this._board, cells, moment);
    this._board = cells;
    if (intermediate) { this._held = [moment.feedback_for | 0, moment.executed | 0]; return null; }
    if (this._held !== null && (moment.feedback_for === null || moment.feedback_for === undefined)) {
      moment = { ...moment, feedback_for: this._held[0], executed: this._held[1], reward: moment.reward, reward_known: true };
    }
    this._held = null;
    if (moment.terminated) this._finish(moment);
    else if (moment.truncated) this._boards = [];
    return this.agent.step(moment);
  }

  /** One observed transition: the stone that appeared, its column and its mover. */
  _observe(before, after, moment) {
    const lay = this.layout;
    let cell = -1, changed = 0;
    for (let i = 0; i < lay.cells; i++) if (after[i] !== before[i]) { cell = i; changed += 1; }
    if (changed !== 1 || before[cell] !== 0) return;
    const mover = argmaxOf(moment.observation.mover) === MOVER_CANDIDATE ? CANDIDATE : OPPONENT;
    if (after[cell] !== mover) return;
    const hasFeedback = moment.feedback_for !== null && moment.feedback_for !== undefined;
    const column = hasFeedback && mover === CANDIDATE ? moment.executed | 0 : lay.columnOf[cell];
    const won = !!moment.terminated && moment.reward !== 0.0;
    const { children, valid } = this.imagination.compose(before, mover, Int32Array.from([column]));
    let exact = !!valid[0];
    if (exact) for (let i = 0; i < lay.cells; i++) if (children[0][i] !== after[i]) { exact = false; break; }
    this.validity.push(exact);
    while (this.validity.length > this.validityWindow) this.validity.shift();
    this.counts.validity_scored += 1;
    this.counts.transitions += 1;
    if (!this.learning) return;
    const relBefore = viewOf(mover, before);
    const rows = new Int32Array(lay.cols).fill(lay.rows);
    rows[lay.columnOf[cell]] = Math.floor(cell / lay.cols);
    this.counts.drop_writes += this.drop.learn(relBefore, column, rows, true, this.report);
    const relAfter = viewOf(mover, after);
    this.counts.line_writes += this.lines.learnLines(relAfter, cell, won, true, this.report);
    this._boards.push([this.layout.windowIds(relAfter), mover]);
  }

  /** A finished game: its boards' window patterns take the outcome of their movers. */
  _finish(moment) {
    if (this.learning && this._boards.length) {
      const sign = Math.sign(moment.reward);
      const candidate = sign > 0 ? 1.0 : sign < 0 ? 0.0 : 0.5;
      const boards = this._boards.map(([ids, mover]) => [ids, mover === CANDIDATE ? candidate : 1.0 - candidate]);
      this.counts.value_writes += this.lines.learnValues(boards, this.report);
      this.counts.games_written += 1;
    }
    this._boards = [];
  }

  // -- deciding

  _plan(agent, row, moment, drive, free) {
    const board = cellsOf(moment.observation, this.game);
    const depth = this.extended ? this.extendedDepth : this.depth;
    const budget = this.extended ? this.extendedBudget : this.budget;
    const legal = moment.action_mask;
    const [unchosen, chosen] = this.drop.landings(viewOf(CANDIDATE, board));  // what the drop records predict for this board
    const info = this.planner.search(board, legal, depth, budget);
    info.landings = [Array.from(unchosen), Array.from(chosen)];
    info.board = Int8Array.from(board);
    this.lastSearch = info;
    const prediction = { next_board: info.next_board, valid: Float64Array.from([info.valid]), value: Float64Array.from([info.value]) };
    return { action: info.column, prediction, budget: { expansions: info.expansions | 0, depth: info.depth | 0, invalid: info.invalid | 0 } };
  }

  /** The next board the records imagine for one column, without search (the page's preview). */
  predictBoard(board, column, side) {
    const { children, valid } = this.imagination.compose(board, side, Int32Array.from([column]));
    return { board: children[0], valid: !!valid[0] };
  }

  /** The state a fixture compares: the tables, the mean and the counters of both cortices. */
  snapshot() {
    return {
      drop: { mean: Float64Array.from(this.drop.records.mean), tables: { landing: Float64Array.from(this.drop.records.tables.landing) }, seen: this.drop.records.seen, writes: this.drop.records.writes },
      lines: { mean: Float64Array.from(this.lines.records.mean), tables: { complete: Float64Array.from(this.lines.records.tables.complete), value: Float64Array.from(this.lines.records.tables.value) }, seen: this.lines.records.seen, writes: this.lines.records.writes },
      counts: { ...this.counts }, planner: this.planner.stats(), validity: this.validity.length,
    };
  }
}

/** The moments of both cortices a fixture holds to the Python numbers after every write. */
export function tableMoments(brain) {
  const out = {};
  for (const [name, records] of [["drop", brain.drop.records], ["lines", brain.lines.records]]) {
    for (const f of records.fields) {
      const table = records.tables[f.name];
      let sum = 0.0, absolute = 0.0, square = 0.0, largest = 0.0;
      for (let k = 0; k < table.length; k++) { const v = table[k]; sum += v; absolute += Math.abs(v); square += v * v; if (Math.abs(v) > largest) largest = Math.abs(v); }
      out[`${name}/${f.name}`] = [sum, absolute, square, largest];
    }
    let mean = 0.0;
    for (const v of records.mean) mean += v;
    out[`${name}/mean`] = [mean, records.seen, records.writes];
  }
  return out;
}
