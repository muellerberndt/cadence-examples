// The records cortex on a page: the region of the whole brain that brain_scan.js does not
// draw, because its cells are not neurons of the connectome. A Canvas2D next to the scan
// shows the granule raster (every cell of the fixed expansion in a compact grid, the active
// set of the executed reading's plain code lit by its code value, the valued code's cells in
// the value colour, imagined reads as brief grey flickers) with the writes flashing on the
// active cells when the outcome arrives (rate-scaled; consequence records in the learning
// colour on the plain code's cells, reward records in the value colour on the valued code's
// cells), and under it the habituated reading the expansion saw, grouped by port, with the
// running norm of every pathway (the valued code divides by them) named under the strip.
// The per-field record reads (the predicted class distributions or values) are rendered as
// HTML rows by `readsHTML`. No activity is invented here: every value drawn comes from the
// engine's `onRecords` events ({kind: "code" | "imagine" | "write"}).

export const RECORDS_COLOR = [96, 165, 250]; // web.py RECORDS_COLOR: the records cortex on a page
const WRITE_COLOR = [255, 140, 90]; // the learning colour of the pages (synapse flashes)
const REWARD_COLOR = [237, 129, 182]; // the value colour (goal, dopamine)
const IMAGINE_COLOR = [200, 214, 228];

/** The reading vector's port groups from a snapshot: [{name, start, end}] in reading order. */
export function readingGroups(snapshot) {
  const rc = snapshot.records;
  if (!rc || !rc.reading) return [];
  const P = snapshot.ports, position = new Map(rc.reading.map((neuron, k) => [neuron, k]));
  const groups = [];
  const add = (name, neurons) => {
    const at = neurons.map((i) => position.get(i)).filter((k) => k !== undefined);
    if (at.length) groups.push({ name, start: Math.min(...at), end: Math.max(...at) + 1 });
  };
  const values = [], flags = [];
  for (const list of Object.values(P.observation_fields || {})) values.push(...list);
  for (const list of Object.values(P.observation_flags || {})) flags.push(...list);
  add("sensors", values); add("missing flags", flags); add("goal", P.goal || []); add("action", P.action || []); add("context", P.context || []); add("recall", P.recall || []);
  return groups.sort((a, b) => a.start - b.start);
}

/** The names of the input pathways the head normalises, in the order Agent.__init__ builds them (observation, goal, action, recall, context; empty ones dropped). */
export function pathwayNames(snapshot) {
  const rc = snapshot.records;
  if (!rc || !rc.reading) return [];
  const P = snapshot.ports, position = new Set(rc.reading);
  return [["observation", P.observation], ["goal", P.goal], ["action", P.action], ["recall", P.recall], ["context", P.context]].filter(([, port]) => (port || []).some((n) => position.has(n))).map(([name]) => name);
}

export class RecordsView {
  /**
   * @param canvas the Canvas2D element
   * @param spec {granules, active, inputs, groups, rate, rewardRate, habituation, color}
   */
  constructor(canvas, spec) {
    this.canvas = canvas;
    this.spec = { color: RECORDS_COLOR, groups: [], rate: 0.2, rewardRate: 1.0, habituation: 0.0, ...spec };
    const G = this.spec.granules | 0;
    this.granules = G;
    this.level = new Float32Array(G); // the executed reading's plain code, held until the next one
    this.valued = new Float32Array(G); // the executed reading's valued code (reward and terminal records read it)
    this.imagined = new Float32Array(G); // imagined reads, decaying
    this.glow = new Float32Array(G); // writes, decaying
    this.glowKind = new Uint8Array(G); // 1 a consequence write, 2 a reward write
    this.activeIndex = new Int32Array(0);
    this.reading = null;
    this.readingScale = 2.0;
    this.blockNorm = null; // the running norm of every pathway, as the last code saw it
    this.lastWrite = null;
    this.counts = { codes: 0, imagined: 0, writes: 0, cells: 0 };
    this.cols = 0; this.rows = 0; this.image = null;
    this.frames = 0;
  }

  /** A code of the executed reading ("decide": held) or of an imagined batch ("imagine": a flicker). */
  code(item, kind) {
    let top = 1e-12;
    for (let k = 0; k < item.values.length; k++) if (item.values[k] > top) top = item.values[k];
    if (kind === "decide") {
      this.level.fill(0); this.valued.fill(0);
      for (let k = 0; k < item.index.length; k++) this.level[item.index[k]] = item.values[k] / top;
      if (item.valuedIndex) { let vtop = 1e-12; for (let k = 0; k < item.valuedValues.length; k++) if (item.valuedValues[k] > vtop) vtop = item.valuedValues[k]; for (let k = 0; k < item.valuedIndex.length; k++) this.valued[item.valuedIndex[k]] = item.valuedValues[k] / vtop; }
      this.activeIndex = Int32Array.from(item.index);
      this.counts.codes += 1;
    } else {
      for (let k = 0; k < item.index.length; k++) { const g = item.index[k], v = 0.5 + 0.5 * (item.values[k] / top); if (v > this.imagined[g]) this.imagined[g] = v; }
      this.counts.imagined += item.batch || 1;
    }
  }

  /** The reading the expansion saw for the executed decision (mean-free, pathways normalised) and the pathway norms in force. */
  setReading(values, blockNorm = null) {
    this.reading = Float32Array.from(values);
    let top = 0; for (const v of this.reading) if (Math.abs(v) > top) top = Math.abs(v);
    this.readingScale = Math.max(1e-6, top);
    this.blockNorm = blockNorm ? Float64Array.from(blockNorm) : null;
  }

  /** A write into the records of the active cells: consequence records flash the plain code's cells at their rate, reward records the valued code's cells in the value colour. */
  write(item) {
    const full = Math.max(this.spec.rewardRate, this.spec.rate, 1e-9);
    const flash = (index, values, rate, kind) => {
      if (!index || !index.length) return;
      let top = 1e-12; for (let k = 0; k < values.length; k++) if (values[k] > top) top = values[k];
      const strength = Math.min(1, rate / full);
      for (let k = 0; k < index.length; k++) { const g = index[k], v = strength * (0.55 + 0.45 * values[k] / top); if (v >= this.glow[g]) { this.glow[g] = v; this.glowKind[g] = kind; } }
    };
    let plainRate = 0, rewardRate = 0;
    for (const f of item.fields) { if (f.reward) rewardRate = Math.max(rewardRate, f.rate); else plainRate = Math.max(plainRate, f.rate); }
    if (plainRate > 0) flash(item.index, item.values, plainRate, 1);
    if (rewardRate > 0) flash(item.valuedIndex || item.index, item.valuedValues || item.values, rewardRate, 2);
    this.lastWrite = item;
    this.counts.writes += item.written || item.fields.length;
    this.counts.cells = item.index.length;
  }

  _layout(width, height) {
    const G = this.granules, aw = Math.max(1, width - 8), ah = Math.max(1, height);
    const cols = Math.max(1, Math.ceil(Math.sqrt((G * aw) / ah))), rows = Math.max(1, Math.ceil(G / cols));
    if (cols !== this.cols || rows !== this.rows) { this.cols = cols; this.rows = rows; this.image = new ImageData(cols, rows); }
  }

  draw() {
    const canvas = this.canvas, r = canvas.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2);
    if (!r.width || !r.height) return;
    const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#05090e"; ctx.fillRect(0, 0, r.width, r.height);
    const stripH = Math.min(116, Math.max(80, r.height * 0.29)), top = 18, rasterH = r.height - stripH - top - 30;
    this._layout(r.width, rasterH);
    // the raster: tissue in the region colour, the active set lit, imagined reads flickering, writes flashing
    const G = this.granules, c = this.spec.color, data = this.image.data, decay = 0.9, imagineDecay = 0.82;
    for (let g = 0; g < G; g++) {
      const lv = this.level[g], vl = this.valued[g], im = this.imagined[g], gl = this.glow[g], kind = this.glowKind[g];
      let R = c[0] * 0.14, Gc = c[1] * 0.14, B = c[2] * 0.14;
      if (lv > 0) { const a = 0.35 + 0.65 * lv; R += c[0] * a; Gc += c[1] * a; B += c[2] * a; }
      if (vl > 0) { const a = 0.3 + 0.5 * vl; R += REWARD_COLOR[0] * a; Gc += REWARD_COLOR[1] * a * 0.6; B += REWARD_COLOR[2] * a * 0.8; }
      if (im > 0.02) { R += IMAGINE_COLOR[0] * 0.55 * im; Gc += IMAGINE_COLOR[1] * 0.55 * im; B += IMAGINE_COLOR[2] * 0.55 * im; this.imagined[g] = im * imagineDecay; }
      if (gl > 0.02) { const wc = kind === 2 ? REWARD_COLOR : WRITE_COLOR; R = R * (1 - gl) + wc[0] * gl * 1.1; Gc = Gc * (1 - gl) + wc[1] * gl * 1.1; B = B * (1 - gl) + wc[2] * gl * 1.1; this.glow[g] = gl * decay; }
      const o = g * 4;
      data[o] = R > 255 ? 255 : R; data[o + 1] = Gc > 255 ? 255 : Gc; data[o + 2] = B > 255 ? 255 : B; data[o + 3] = 255;
    }
    for (let g = G; g < this.cols * this.rows; g++) { const o = g * 4; data[o] = 5; data[o + 1] = 9; data[o + 2] = 14; data[o + 3] = 255; }
    if (!this._scratch || this._scratch.width !== this.cols || this._scratch.height !== this.rows) { this._scratch = document.createElement("canvas"); this._scratch.width = this.cols; this._scratch.height = this.rows; }
    this._scratch.getContext("2d").putImageData(this.image, 0, 0);
    const cell = Math.min((r.width - 8) / this.cols, rasterH / this.rows), rw = cell * this.cols, rh = cell * this.rows, rx = 4 + ((r.width - 8) - rw) / 2, ry = top;
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(this._scratch, rx, ry, rw, rh);
    ctx.strokeStyle = `rgba(${c[0]},${c[1]},${c[2]},0.35)`; ctx.lineWidth = 1; ctx.strokeRect(rx + 0.5, ry + 0.5, rw - 1, rh - 1);
    ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`; ctx.font = "600 11px ui-monospace, SFMono-Regular, Menlo, monospace"; ctx.textAlign = "left";
    ctx.fillText(`granules ${G.toLocaleString()} · ${this.spec.active} active · ${this.cols}×${this.rows}`, 4, 12);
    ctx.fillStyle = "#6f8397"; ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
    const lw = this.lastWrite;
    const writeText = lw ? `write ${lw.fields.length} field${lw.fields.length === 1 ? "" : "s"}${lw.fields.some((f) => f.reward) ? " +reward" : ""}` : "no write yet";
    ctx.fillText(`codes ${this.counts.codes} · imagined ${this.counts.imagined} · ${writeText}`, 4, ry + rh + 12);
    // the habituated reading: bars from a midline, the ports separated by ticks and named in one line under the strip
    const sy = ry + rh + 20, sh = stripH - 6, barsTop = sy + 14, barsBottom = sy + sh - 26, mid = barsTop + (barsBottom - barsTop) * 0.55;
    ctx.fillStyle = "#0a1119"; ctx.fillRect(4, sy, r.width - 8, sh);
    ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`; ctx.font = "600 10px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.fillText(`${this.spec.habituation > 0 ? "habituated " : ""}reading (${this.spec.inputs} units${this.spec.habituation > 0 ? ", mean-free" : ""})`, 6, sy + 10);
    const values = this.reading, n = this.spec.inputs, bw = (r.width - 8) / Math.max(1, n);
    if (values) {
      const scale = this.readingScale, up = mid - barsTop, down = barsBottom - mid;
      for (let k = 0; k < values.length; k++) {
        const v = values[k] / scale, x = 4 + k * bw;
        if (v >= 0) { ctx.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},${0.35 + 0.65 * Math.min(1, v)})`; ctx.fillRect(x, mid - v * up, Math.max(1, bw - 0.5), Math.max(1, v * up)); }
        else { ctx.fillStyle = `rgba(143,163,184,${0.35 + 0.65 * Math.min(1, -v)})`; ctx.fillRect(x, mid, Math.max(1, bw - 0.5), Math.max(1, -v * down)); }
      }
    } else { ctx.fillStyle = "#4f6479"; ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace"; ctx.fillText("no reading yet · press Play", 6, mid + 3); }
    ctx.strokeStyle = "#1c2a3a"; ctx.beginPath(); ctx.moveTo(4, mid + 0.5); ctx.lineTo(r.width - 4, mid + 0.5); ctx.stroke();
    for (const g of this.spec.groups) { const x1 = 4 + g.end * bw; ctx.strokeStyle = "#24364a"; ctx.beginPath(); ctx.moveTo(x1, barsTop - 2); ctx.lineTo(x1, barsBottom + 2); ctx.stroke(); }
    ctx.fillStyle = "#6f8397"; ctx.font = "9px ui-monospace, SFMono-Regular, Menlo, monospace";
    const withSizes = this.spec.groups.map((g) => `${g.name} ${g.end - g.start}`).join(" · "), namesOnly = this.spec.groups.map((g) => g.name).join(" · ");
    const legend = ctx.measureText(withSizes).width <= r.width - 12 ? withSizes : namesOnly;
    ctx.fillText(legend, 6, sy + sh - 15);
    if (this.blockNorm && this.blockNorm.length) {
      const short = { observation: "obs", goal: "goal", action: "act", recall: "recall", context: "ctx" }, names = this.spec.pathways || [];
      const norms = Array.from(this.blockNorm, (v, k) => `${short[names[k]] || names[k] || `p${k + 1}`} ${v.toFixed(2)}`).join(" · ");
      ctx.fillText(ctx.measureText(`valued code ÷ ${norms}`).width <= r.width - 12 ? `valued code ÷ ${norms}` : `valued ÷ ${Array.from(this.blockNorm, (v) => v.toFixed(2)).join(" ")}`, 6, sy + sh - 4);
    } else ctx.fillText("valued code: no pathway normalisation", 6, sy + sh - 4);
    this.frames += 1;
  }
}

/** The per-field record reads as HTML rows: class bars (the largest class marked) or a value on its sensor range, with the error of the last write when one arrived. */
export function readsHTML(prediction, fields, lastWrite = null, colors = { color: RECORDS_COLOR, write: WRITE_COLOR, reward: REWARD_COLOR }) {
  if (!prediction) return `<div class="read muted">no read yet</div>`;
  const errors = new Map((lastWrite ? lastWrite.fields : []).map((f) => [f.name, f]));
  const c = colors.color;
  return fields.map((f) => {
    const p = prediction[f.name];
    if (!p) return "";
    const e = errors.get(f.name);
    const err = e ? `<em title="the largest error of the last write at rate ${e.rate}${e.reward ? " (a reward record)" : ""}" style="color:rgb(${(e.reward ? colors.reward : colors.write).join(",")})">err ${e.error.toFixed(2)}</em>` : "";
    const label = f.name.replace(/^next_/, "").replace(/^passage_(\w)\w*$/, (_, d) => `passage ${d.toUpperCase()}`);
    if (f.kind === "categorical") {
      let arg = 0; for (let k = 1; k < p.length; k++) if (p[k] > p[arg]) arg = k;
      const bars = Array.from(p, (v, k) => `<i title="class ${k}: ${v.toFixed(3)}" style="height:${Math.round(3 + 15 * Math.min(1, Math.max(0, v)))}px;background:${k === arg ? `rgb(${c.join(",")})` : `rgba(${c.join(",")},0.35)`}"></i>`).join("");
      return `<div class="read"><span class="name" title="${f.name}">${label}</span><span class="bars">${bars}</span><b title="the largest class and its probability">${arg} · ${p[arg].toFixed(2)}</b>${err}</div>`;
    }
    const parts = Array.from(p, (v) => { const t = Math.min(1, Math.max(0, (v - f.lo) / (f.hi - f.lo))); return `<span class="track" title="${f.lo} to ${f.hi}"><i style="left:${(t * 100).toFixed(1)}%;background:rgb(${c.join(",")})"></i></span><b>${v.toFixed(3)}</b>`; }).join("");
    return `<div class="read"><span class="name" title="${f.name}">${label}</span>${parts}${err}</div>`;
  }).join("");
}
