// The composer's world model in the browser: the records cortex and the imagination over it.
//
// A port of what one decision of `S06Agent` reads. A decision takes no settling: `plan` reads the
// moment and `imagine`, and `imagine` reads the records cortex alone, so what the connectome
// carries into a decision is one vector, the hypothetical drive over the reading, which holds the
// goal port, the context trace and the recall. The page keeps that vector; this file does the rest.
//
// The arithmetic is `Records`: the reading is made mean-free, projected through the fixed random
// expansion, the `active` largest cells are kept, their values are clipped at zero and normalised,
// and each field's table is read through them. Candidates are read as deltas on the base gesture's
// drive, which is what makes a grid of nine candidates one matrix-vector product and nine gathers.

export const CHANNELS = ["mix", "voice1", "voice2", "voice3"];
export const BANDS = 64;
export const HEARD = 4;
export const HEARING_WIDTH = 68;

const TYPES = { f4: Float32Array, f8: Float64Array, u4: Uint32Array, i4: Int32Array, u2: Uint16Array, i2: Int16Array, u1: Uint8Array, i1: Int8Array };

/** A {dtype, shape, b64} array, as the fixture and the atlas exporter write them. */
export function decodeArray(spec) {
  if (spec == null) return null;
  if (ArrayBuffer.isView(spec)) return spec;
  if (Array.isArray(spec)) return Float64Array.from(spec);
  const T = TYPES[spec.dtype];
  if (!T) throw Error(`unsupported dtype ${spec.dtype}`);
  const bytes = Uint8Array.from(atob(spec.b64), (c) => c.charCodeAt(0));
  const copy = new Uint8Array(bytes.length);
  copy.set(bytes);
  return new T(copy.buffer, 0, copy.byteLength / T.BYTES_PER_ELEMENT);
}

export class RecordsCortex {
  /** @param spec {cells, active, inputs, mean, projection, offset, tables} from the fixture. */
  constructor(spec, fields) {
    this.cells = spec.cells | 0;
    this.active = spec.active | 0;
    this.inputs = spec.inputs | 0;
    this.mean = Float64Array.from(decodeArray(spec.mean));
    this.projection = decodeArray(spec.projection); // (inputs, cells), row major
    this.offset = Float64Array.from(decodeArray(spec.offset));
    this.fields = fields;
    this.tables = {};
    for (const field of fields) {
      const flat = decodeArray(spec.tables[field.name]);
      if (flat.length !== this.cells * field.width) throw Error(`${field.name}: the table is ${flat.length} where ${this.cells * field.width} is expected`);
      this.tables[field.name] = flat; // (cells, width), row major
    }
  }

  /** The cell drives of one reading: (reading - mean) @ projection + offset. */
  rawDrive(reading) {
    const { cells, inputs, projection, offset, mean } = this;
    const out = new Float64Array(cells);
    out.set(offset);
    for (let i = 0; i < inputs; i++) {
      const x = reading[i] - mean[i];
      if (x === 0) continue;
      const row = i * cells;
      for (let g = 0; g < cells; g++) out[g] += x * projection[row + g];
    }
    return out;
  }

  /** The `active` largest cells of a drive, their normalised code, and each field's read. */
  fieldsOf(drives, { valued = false } = {}) {
    const batch = drives.length;
    const out = {};
    for (const field of this.fields) {
      if ((field.source === "reward" || field.source === "terminal") && !valued) continue;
      out[field.name] = new Float64Array(batch * field.width);
    }
    const keep = new Int32Array(this.active);
    const code = new Float64Array(this.active);
    for (let b = 0; b < batch; b++) {
      this._winners(drives[b], keep, code);
      for (const field of this.fields) {
        if (!(field.name in out)) continue;
        const table = this.tables[field.name], width = field.width;
        const read = new Float64Array(width);
        for (let k = 0; k < this.active; k++) {
          const c = code[k];
          if (c === 0) continue;
          const row = keep[k] * width;
          for (let w = 0; w < width; w++) read[w] += c * table[row + w];
        }
        const at = b * width;
        if (field.kind === "categorical") {
          let total = 0;
          for (let w = 0; w < width; w++) { const v = read[w] > 0 ? read[w] : 0; read[w] = v; total += v; }
          for (let w = 0; w < width; w++) out[field.name][at + w] = total > 0 ? read[w] / Math.max(total, 1e-300) : 1.0 / width;
        } else {
          for (let w = 0; w < width; w++) {
            const clipped = Math.min(0.8, Math.max(0.05, read[w]));
            out[field.name][at + w] = field.lo + ((clipped - 0.05) / 0.75) * (field.hi - field.lo);
          }
        }
      }
    }
    return { fields: out, batch };
  }

  /** The active largest entries of a drive (the set, as argpartition gives it), and the code. */
  _winners(drive, keep, code) {
    const active = this.active, cells = this.cells;
    // a partial selection: keep the `active` largest by repeatedly replacing the smallest kept
    let filled = 0, smallest = 0;
    for (let g = 0; g < cells; g++) {
      const value = drive[g];
      if (filled < active) {
        keep[filled] = g;
        code[filled] = value;
        filled++;
        if (filled === active) {
          smallest = 0;
          for (let k = 1; k < active; k++) if (code[k] < code[smallest]) smallest = k;
        }
        continue;
      }
      if (value <= code[smallest]) continue;
      keep[smallest] = g;
      code[smallest] = value;
      smallest = 0;
      for (let k = 1; k < active; k++) if (code[k] < code[smallest]) smallest = k;
    }
    let norm = 0;
    for (let k = 0; k < active; k++) {
      const v = code[k] > 0 ? code[k] : 0;
      code[k] = v;
      norm += v * v;
    }
    norm = Math.sqrt(norm);
    if (norm > 0) for (let k = 0; k < active; k++) code[k] = code[k] / Math.max(norm, 1e-12);
  }
}

export class S06Brain {
  /**
   * @param spec the fixture's once-only block: ports, reading, graph gains, records, fields.
   */
  constructor(spec) {
    this.spec = spec;
    this.fields = spec.fields;
    this.cortex = new RecordsCortex(spec.records, spec.fields);
    this.inputGain = spec.graph.input_gain;
    this.gain = spec.graph.gain;
    this.reading = Int32Array.from(spec.reading);
    this.actionAt = spec.action_at | 0;
    // where each neuron of the reading sits, so the drive is built in reading space
    const position = new Map();
    for (let k = 0; k < this.reading.length; k++) position.set(this.reading[k], k);
    this.position = position;
    this.observationFields = {};
    for (const [name, neurons] of Object.entries(spec.ports.observation_fields)) {
      this.observationFields[name] = Int32Array.from(neurons.map((n) => (position.has(n) ? position.get(n) : -1)));
    }
    this.actionSlots = spec.ports.action_slots.map((slot) => Int32Array.from(slot.map((n) => position.get(n))));
    this.imagined = 0;
  }

  /** The sensory drive of one hearing, in reading space, as `_encode_observation` builds it. */
  observationDrive(hearing, attending, fine = null) {
    const out = new Float64Array(this.reading.length);
    const put = (name, values, offset = 0, count = values.length) => {
      const slots = this.observationFields[name];
      if (!slots) return;
      for (let k = 0; k < slots.length; k++) {
        if (slots[k] < 0) continue;
        out[slots[k]] = values[offset + k];
      }
    };
    const clip = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
    if (fine && this.observationFields.fine_mix) {
      const values = new Float64Array(fine.length);
      for (let i = 0; i < fine.length; i++) values[i] = this.inputGain * clip(fine[i]);
      put("fine_mix", values);
    }
    for (let index = 0; index < CHANNELS.length; index++) {
      const channel = CHANNELS[index], at = index * HEARING_WIDTH;
      const spectrum = new Float64Array(BANDS);
      for (let k = 0; k < BANDS; k++) spectrum[k] = this.inputGain * clip(hearing[at + k]);
      put(`spectrum_${channel}`, spectrum);
      const level = new Float64Array(2);
      for (let k = 0; k < 2; k++) level[k] = this.inputGain * clip(hearing[at + BANDS + k]);
      put(`level_${channel}`, level);
      const silence = new Float64Array(2);
      for (let k = 0; k < 2; k++) silence[k] = this.inputGain * hearing[at + BANDS + 2 + k];
      put(`silence_${channel}`, silence);
    }
    const attend = new Float64Array(attending.length);
    for (let k = 0; k < attending.length; k++) attend[k] = this.inputGain * attending[k];
    put("attending", attend);
    return out;
  }

  /** One efference copy: the chosen neuron of every slot at the action gain. */
  fieldDrive(drive, gesture) {
    const out = Float64Array.from(drive);
    for (const slot of this.actionSlots) for (let k = 0; k < slot.length; k++) out[slot[k]] = 0;
    for (let f = 0; f < this.actionSlots.length; f++) out[this.actionSlots[f][gesture[f]]] = this.gain;
    return out;
  }

  /**
   * The consequences the records predict for candidate gestures under one hearing.
   * `hypothetical` is the drive the rest of the brain carries into this decision, in reading space.
   */
  imagine(hearing, attending, hypothetical, base, gestures, { fine = null } = {}) {
    const drive = this.observationDrive(hearing, attending, fine);
    for (let k = 0; k < drive.length; k++) drive[k] += hypothetical[k];
    const withBase = this.fieldDrive(drive, base);
    const raw = this.cortex.rawDrive(withBase);
    const cells = this.cortex.cells;
    const drives = [];
    for (let g = 0; g < gestures.length; g++) {
      const row = Float64Array.from(raw);
      const gesture = gestures[g];
      for (let f = 0; f < this.actionSlots.length; f++) {
        if (gesture[f] === base[f]) continue;
        const to = this.actionSlots[f][gesture[f]], from = this.actionSlots[f][base[f]];
        const toRow = to * cells, fromRow = from * cells;
        for (let c = 0; c < cells; c++) row[c] += this.gain * (this.cortex.projection[toRow + c] - this.cortex.projection[fromRow + c]);
      }
      drives.push(row);
    }
    this.imagined += gestures.length;
    return this.cortex.fieldsOf(drives);
  }

  /** `imagine` for one hearing per gesture, which a rollout past its first block needs. */
  imagineMany(hearings, attending, hypothetical, gestures) {
    const drives = [];
    for (let g = 0; g < gestures.length; g++) {
      const drive = this.observationDrive(hearings[g], attending, null);
      for (let k = 0; k < drive.length; k++) drive[k] += hypothetical[k];
      drives.push(this.cortex.rawDrive(this.fieldDrive(drive, gestures[g])));
    }
    this.imagined += gestures.length;
    return this.cortex.fieldsOf(drives);
  }

  /** This block's hearing plus the predicted change, as a batch of (4, 68). */
  imaginedHearing(hearing, predicted) {
    const batch = predicted.batch;
    const out = [];
    for (let b = 0; b < batch; b++) {
      const row = new Float64Array(HEARD * HEARING_WIDTH);
      for (let index = 0; index < CHANNELS.length; index++) {
        const channel = CHANNELS[index], at = index * HEARING_WIDTH;
        const spectrum = predicted.fields[`d_spectrum_${channel}`];
        const level = predicted.fields[`d_level_${channel}`];
        const silence = predicted.fields[`next_silence_${channel}`];
        for (let k = 0; k < BANDS; k++) row[at + k] = hearing[at + k] + spectrum[b * BANDS + k];
        for (let k = 0; k < 2; k++) row[at + BANDS + k] = hearing[at + BANDS + k] + level[b * 2 + k];
        for (let k = 0; k < 2; k++) row[at + BANDS + 2 + k] = silence[b * 2 + k];
      }
      out.push(harden(row));
    }
    return out;
  }

  /** This block's fine mix plus the change the records predict of it. */
  imaginedFine(fine, predicted) {
    const change = predicted.fields.d_fine_mix;
    if (!change || !fine) return null;
    const width = this.fields.find((f) => f.name === "d_fine_mix").width;
    const out = [];
    for (let b = 0; b < predicted.batch; b++) {
      const row = new Float64Array(width);
      for (let k = 0; k < width; k++) {
        const v = fine[k] + change[b * width + k];
        row[k] = v < 0 ? 0 : v > 1 ? 1 : v;
      }
      out.push(row);
    }
    return out;
  }
}

/** An imagined hearing read back as an observation: values in range, the flag one of two. */
export function harden(row) {
  const out = Float64Array.from(row);
  for (let k = 0; k < out.length; k++) out[k] = out[k] < 0 ? 0 : out[k] > 1 ? 1 : out[k];
  for (let index = 0; index < HEARD; index++) {
    const at = index * HEARING_WIDTH + BANDS + 2;
    const silent = out[at] < out[at + 1];
    out[at] = silent ? 0 : 1;
    out[at + 1] = silent ? 1 : 0;
  }
  return out;
}
