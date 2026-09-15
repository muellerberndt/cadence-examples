// The artist in JavaScript: a port of artist/env.py and artist/brain.py. The canvas world over
// the S01 arm body (rasterisation at one pixel width, the pen as one more action bit, the
// pen-centred windows, the symmetric Chamfer discrepancy and the foreground F1), the planning
// objective `Sheet`, the slow level that chooses a stroke intention by imagining its marks, and
// the fast level that scores the eighteen choices through the learned model. Float64 everywhere,
// numpy's pairwise sums through `npSum`, numpy's round-half-to-even through `rint`, and a
// Mulberry32 stream where Python seeds a numpy generator (the page runs its own life, so the two
// generators need not agree; every decision, intention, prediction and record write does).
//
// The fast level's leaf objective is `ModelPlanner` from ../../arm/web/planner.js, the port of
// the planner the Python stage imports from S01. The body is `CanvasArm` here rather than `Arm`
// from ../../arm/web/arm.js: that port wraps an angle with two moduli, which costs an ulp of 2pi
// per body step against `(angle + pi) % (2 pi) - pi`, and 210 body steps of drift are enough to
// move a pen pixel and part two searches that agree to 1e-13. Everything else is the same body.

import { Mulberry32, npMean, npSum } from "../../web/engine.js";
import { ModelPlanner, compose } from "../../arm/web/planner.js";

export const CANVAS_SIZE = 32;
export const MARK_WINDOW = 8;
export const TORQUE_PAIRS = 9;
export const CHOICES = 2 * TORQUE_PAIRS; // nine torque pairs by pen up or down
export const CHOICE_FIELDS = [TORQUE_PAIRS, 2];
export const PREDICTED_FIELDS = ["dd_hand", "d_velocity", "d_angles", "mark"];
export const CANVAS_FAMILIES = ["segment", "two_segments", "polygon", "curve", "composition"];
export const FAMILY_LABELS = { segment: "segment", two_segments: "two segments", polygon: "polygon", curve: "curve", composition: "composition" };

export const CANVAS_DEFAULTS = { size: CANVAS_SIZE, window: MARK_WINDOW, extent: 0.7, centre: [0.0, 0.5], margin: 3, max_decisions: 200, patience: 60, finish: 0.02, torque_cost: 0.001, pen_offset: [0.0, 0.0], cap: 4.0 };

/** The canvas geometry with its derived measures (env.py CanvasConfig). */
export function canvasConfig(over = {}) {
  const c = { ...CANVAS_DEFAULTS, ...over };
  c.centre = [...(over.centre || CANVAS_DEFAULTS.centre)];
  c.pen_offset = [...(over.pen_offset || CANVAS_DEFAULTS.pen_offset)];
  c.pixel = c.extent / c.size;
  c.diagonal = Math.hypot(c.size, c.size);
  return c;
}

/** The joint action index as the brain reads it: the torque pair and the pen state (row-major). */
export function splitChoice(action) {
  const a = action | 0;
  if (!(a >= 0 && a < CHOICES)) throw Error("action outside the eighteen choices");
  return [(a / 2) | 0, a % 2];
}
export function joinChoice(torque, pen) { return (torque | 0) * 2 + (pen | 0); }

/** numpy's round-half-to-even, which is also Python's `round` on a float. */
export function rint(x) {
  const f = Math.floor(x), d = x - f;
  if (d < 0.5) return f;
  if (d > 0.5) return f + 1;
  return f % 2 === 0 ? f : f + 1;
}

// -- geometry and rasterisation

/** Continuous pixel coordinates (row, column) of a hand position; integers are pixel centres. */
export function pixelOf(hand, config) {
  const middle = (config.size - 1) / 2.0;
  const column = (hand[0] - config.centre[0]) / config.pixel + middle;
  const row = middle - (hand[1] - config.centre[1]) / config.pixel;
  return Float64Array.of(row, column);
}

export function handOfPixel(pixel, config) {
  const middle = (config.size - 1) / 2.0;
  return Float64Array.of(config.centre[0] + (pixel[1] - middle) * config.pixel, config.centre[1] - (pixel[0] - middle) * config.pixel);
}

/** The one-pixel-wide line between two pixel centres, endpoints included (env.py bresenham). */
export function bresenham(start, end) {
  let r0 = rint(start[0]), c0 = rint(start[1]);
  const r1 = rint(end[0]), c1 = rint(end[1]);
  const dr = Math.abs(r1 - r0), dc = Math.abs(c1 - c0);
  const sr = r1 > r0 ? 1 : -1, sc = c1 > c0 ? 1 : -1;
  let error = dr - dc;
  const points = [r0, c0];
  while (r0 !== r1 || c0 !== c1) {
    const twice = 2 * error;
    if (twice > -dc) { error -= dc; r0 += sr; }
    if (twice < dr) { error += dr; c0 += sc; }
    points.push(r0, c0);
  }
  return Int32Array.from(points);
}

/** Ink the points that lie on the canvas; returns the flat (row, column) pairs actually inked. */
export function drawPoints(canvas, points, size) {
  const out = [];
  for (let k = 0; k < points.length; k += 2) {
    const r = points[k], c = points[k + 1];
    if (r >= 0 && r < size && c >= 0 && c < size) { canvas[r * size + c] = 1; out.push(r, c); }
  }
  return Int32Array.from(out);
}

/** Polylines of (row, column) points rasterised onto a fresh canvas. */
export function renderPaths(paths, size) {
  const canvas = new Uint8Array(size * size);
  for (const path of paths) for (let k = 1; k < path.length; k++) drawPoints(canvas, bresenham(path[k - 1], path[k]), size);
  return canvas;
}

export function windowOrigin(pixel, window) { return [rint(pixel[0]) - (window >> 1), rint(pixel[1]) - (window >> 1)]; }

/** The `window` by `window` patch at `origin`; outside the canvas reads empty. */
export function cropWindow(canvas, origin, window, size) {
  const out = new Float64Array(window * window);
  const [r0, c0] = origin;
  for (let i = 0; i < window; i++) {
    const r = r0 + i;
    if (r < 0 || r >= size) continue;
    for (let j = 0; j < window; j++) {
      const c = c0 + j;
      if (c < 0 || c >= size) continue;
      out[i * window + j] = canvas[r * size + c];
    }
  }
  return out;
}

// -- the supplied measurements

/** The foreground of a canvas as (row, column) coordinates, in numpy's row-major nonzero order. */
export function coordinatesOf(canvas, size) {
  const rows = [], cols = [];
  for (let r = 0; r < size; r++) for (let c = 0; c < size; c++) if (canvas[r * size + c]) { rows.push(r); cols.push(c); }
  return { n: rows.length, rows: Float64Array.from(rows), cols: Float64Array.from(cols) };
}

/** `_pairwise(a, b).min(axis=1)`: the distance from every point of `a` to the nearest of `b`. */
export function nearestDistances(a, b) {
  const out = new Float64Array(a.n);
  for (let i = 0; i < a.n; i++) {
    let best = Infinity;
    const ar = a.rows[i], ac = a.cols[i];
    for (let j = 0; j < b.n; j++) { const dr = ar - b.rows[j], dc = ac - b.cols[j], d = Math.sqrt(dr * dr + dc * dc); if (d < best) best = d; }
    out[i] = best;
  }
  return out;
}

/** Symmetric Chamfer distance of the foregrounds, normalised by the canvas diagonal (env.py chamfer). */
export function chamferOf(drawn, target, size) {
  const d = coordinatesOf(drawn, size), t = coordinatesOf(target, size);
  if (!d.n && !t.n) return 0.0;
  if (!d.n || !t.n) return 1.0;
  const diagonal = Math.hypot(size, size);
  return 0.5 * (npMean(nearestDistances(d, t)) + npMean(nearestDistances(t, d))) / diagonal;
}

/** Foreground precision, recall and F1 within `tolerance` pixels (env.py foreground_f1). */
export function foregroundF1(drawn, target, size, tolerance = 1.0) {
  const d = coordinatesOf(drawn, size), t = coordinatesOf(target, size);
  if (!d.n && !t.n) return { precision: 1.0, recall: 1.0, f1: 1.0 };
  if (!d.n || !t.n) return { precision: 0.0, recall: 0.0, f1: 0.0 };
  const bar = tolerance + 1e-9;
  const precision = npMean(Float64Array.from(nearestDistances(d, t), (x) => (x <= bar ? 1.0 : 0.0)));
  const recall = npMean(Float64Array.from(nearestDistances(t, d), (x) => (x <= bar ? 1.0 : 0.0)));
  const total = precision + recall;
  return { precision, recall, f1: total > 0 ? (2 * precision * recall) / total : 0.0 };
}

/** Distance from every pixel to the nearest target pixel; the diagonal when the target is empty. */
export function distanceMap(target, size) {
  const t = coordinatesOf(target, size), out = new Float64Array(size * size);
  if (!t.n) return out.fill(Math.hypot(size, size));
  for (let r = 0; r < size; r++) for (let c = 0; c < size; c++) {
    let best = Infinity;
    for (let j = 0; j < t.n; j++) { const dr = r - t.rows[j], dc = c - t.cols[j], d = Math.sqrt(dr * dr + dc * dc); if (d < best) best = d; }
    out[r * size + c] = best;
  }
  return out;
}

/**
 * The planner's objective on one drawing (env.py Sheet): the discrepancy with the target term
 * capped, so ink further than `cap` pixels from the target earns nothing. Witnessed ink is
 * certain; imagined ink carries the weight the learned model gives it, and the objective is the
 * expectation under independent cells.
 */
export class Sheet {
  constructor(target, size, cap = 4.0, floor = 1e-3) {
    this.target = target; this.size = size; this.cap = cap; this.floor = floor;
    this.distances = Float64Array.from(distanceMap(target, size), (d) => Math.min(d, cap) / cap);
    this.targetPoints = coordinatesOf(target, size);
    this.mass = new Float64Array(size * size);
    this.ink = 0.0;
    this.nearest = new Float64Array(this.targetPoints.n).fill(1.0);
  }

  /** Set the objective to a witnessed canvas (the brain's belief of what is drawn). */
  observe(canvas) {
    const size = this.size;
    this.mass = Float64Array.from(canvas, (v) => (v ? 1.0 : 0.0));
    const drawn = this.drawn;
    const points = coordinatesOf(drawn, size);
    const marked = [];
    for (let k = 0; k < size * size; k++) if (this.mass[k] >= 0.5) marked.push(this.distances[k]);
    this.ink = npSum(Float64Array.from(marked));
    if (this.targetPoints.n && points.n) this.nearest = Float64Array.from(nearestDistances(this.targetPoints, points), (d) => Math.min(d / this.cap, 1.0));
    else this.nearest = new Float64Array(this.targetPoints.n).fill(1.0);
  }

  get drawn() { return Uint8Array.from(this.mass, (v) => (v >= 0.5 ? 1 : 0)); }

  get value() {
    const n = Math.max(this.targetPoints.n, 1);
    return (this.ink + npSum(this.nearest)) / n;
  }

  /** The objective's change if `points` were inked, with the new ink, reach and applied weights. */
  peek(points, weights = null) {
    const count = points.length / 2;
    if (!count) return { change: 0.0, ink: this.ink, nearest: this.nearest, applied: new Float64Array(0) };
    const size = this.size, room = new Float64Array(count);
    for (let k = 0; k < count; k++) room[k] = 1.0 - this.mass[points[2 * k] * size + points[2 * k + 1]];
    let ink = this.ink, nearest = this.nearest, applied;
    if (weights === null) {
      applied = room;
      const terms = new Float64Array(count);
      for (let k = 0; k < count; k++) terms[k] = room[k] * this.distances[points[2 * k] * size + points[2 * k + 1]];
      ink = this.ink + npSum(terms);
      if (this.targetPoints.n) {
        const set = { n: count, rows: Float64Array.from({ length: count }, (_, k) => points[2 * k]), cols: Float64Array.from({ length: count }, (_, k) => points[2 * k + 1]) };
        const reach = nearestDistances(this.targetPoints, set);
        nearest = Float64Array.from(this.nearest, (v, i) => Math.min(v, Math.min(reach[i] / this.cap, 1.0)));
      }
    } else {
      applied = new Float64Array(count);
      for (let k = 0; k < count; k++) applied[k] = Math.min(1.0, Math.max(0.0, weights[k])) * room[k];
      // np.argsort(-applied): decreasing weight, ties in index order
      const order = Array.from({ length: count }, (_, k) => k).sort((x, y) => applied[y] - applied[x] || x - y);
      nearest = this.nearest;
      let copied = false;
      for (const index of order) {
        const weight = applied[index];
        if (weight <= this.floor) break;
        ink += weight * this.distances[points[2 * index] * size + points[2 * index + 1]];
        if (this.targetPoints.n) {
          if (!copied) { nearest = Float64Array.from(nearest); copied = true; }
          const pr = points[2 * index], pc = points[2 * index + 1];
          for (let j = 0; j < this.targetPoints.n; j++) {
            const dr = this.targetPoints.rows[j] - pr, dc = this.targetPoints.cols[j] - pc;
            const reach = Math.min(Math.sqrt(dr * dr + dc * dc) / this.cap, 1.0);
            nearest[j] = nearest[j] - weight * (nearest[j] - Math.min(nearest[j], reach));
          }
        }
      }
    }
    const n = Math.max(this.targetPoints.n, 1);
    return { change: (ink + npSum(nearest)) / n - this.value, ink, nearest, applied };
  }

  apply(points, weights = null) {
    const out = this.peek(points, weights);
    const count = points.length / 2, size = this.size;
    if (count) {
      const values = new Float64Array(count);
      for (let k = 0; k < count; k++) values[k] = Math.min(this.mass[points[2 * k] * size + points[2 * k + 1]] + out.applied[k], 1.0);
      for (let k = 0; k < count; k++) this.mass[points[2 * k] * size + points[2 * k + 1]] = values[k]; // numpy assigns the whole right-hand side: a repeated pixel takes the last value
    }
    this.ink = out.ink; this.nearest = out.nearest;
    return out.change;
  }

  copy() {
    const out = Object.create(Sheet.prototype);
    out.target = this.target; out.size = this.size; out.cap = this.cap; out.floor = this.floor;
    out.distances = this.distances; out.targetPoints = this.targetPoints;
    out.mass = Float64Array.from(this.mass); out.ink = this.ink; out.nearest = Float64Array.from(this.nearest);
    return out;
  }

  /** The target pixels no ink has reached within the cap; the search approaches these. */
  uncovered() {
    const rows = [], cols = [];
    for (let j = 0; j < this.targetPoints.n; j++) if (this.nearest[j] >= 1.0 - 1e-9) { rows.push(this.targetPoints.rows[j]); cols.push(this.targetPoints.cols[j]); }
    return { n: rows.length, rows: Float64Array.from(rows), cols: Float64Array.from(cols) };
  }
}

// -- target families and splits

/** The little random interface the page's own targets are drawn from (Python seeds numpy here). */
export class PageRandom {
  constructor(seed) { this.rng = new Mulberry32(seed); }
  random() { return this.rng.random(); }
  uniform(lo = 0.0, hi = 1.0) { return lo + (hi - lo) * this.rng.random(); }
  integers(lo, hi) { return lo + Math.floor(this.rng.random() * (hi - lo)); }
}

const rotatePoint = (p, c, s) => [p[0] * c - p[1] * s, p[0] * s + p[1] * c];

function placePaths(paths, rng, config) {
  let all = paths.flat();
  let low = [Math.min(...all.map((p) => p[0])), Math.min(...all.map((p) => p[1]))];
  let high = [Math.max(...all.map((p) => p[0])), Math.max(...all.map((p) => p[1]))];
  const box = config.size - 1 - 2 * config.margin;
  const shrink = Math.min(1.0, box / Math.max(Math.max(high[0] - low[0], high[1] - low[1]), 1e-9));
  if (shrink < 1.0) {
    const middle = [(low[0] + high[0]) / 2, (low[1] + high[1]) / 2];
    paths = paths.map((path) => path.map((p) => [(p[0] - middle[0]) * shrink + middle[0], (p[1] - middle[1]) * shrink + middle[1]]));
    all = paths.flat();
    low = [Math.min(...all.map((p) => p[0])), Math.min(...all.map((p) => p[1]))];
    high = [Math.max(...all.map((p) => p[0])), Math.max(...all.map((p) => p[1]))];
  }
  const lo = [config.margin - low[0], config.margin - low[1]];
  const hi = [config.size - 1 - config.margin - high[0], config.size - 1 - config.margin - high[1]];
  const shift = [rng.uniform(lo[0], hi[0]), rng.uniform(lo[1], hi[1])];
  return paths.map((path) => path.map((p) => [p[0] + shift[0], p[1] + shift[1]]));
}

function arcPoints(radius, span, start, count = 24) {
  return Array.from({ length: count }, (_, k) => { const a = start + (span * k) / (count - 1); return [radius * Math.sin(a), radius * Math.cos(a)]; });
}

/** The polylines of one target shape in pixel coordinates (row, column) (env.py shape). */
export function shapeOf(family, rng, rotation, scale, config) {
  const span = config.size - 2 * config.margin - 4;
  const length = scale * span;
  let paths;
  if (family === "segment") paths = [[[-length / 2, 0.0], [length / 2, 0.0]]];
  else if (family === "two_segments") {
    const angle = rng.uniform(Math.PI / 3, (2 * Math.PI) / 3), reach = length * rng.uniform(0.6, 1.0);
    paths = [[[-length / 2, 0.0], [0.0, 0.0], [Math.cos(angle) * reach, Math.sin(angle) * reach]]];
  } else if (family === "polygon") {
    const sides = rng.integers(3, 5), angles = Array.from({ length: sides }, () => rng.uniform(0, 2 * Math.PI)).sort((a, b) => a - b);
    const radius = length / 2;
    const corners = angles.map((a) => [radius * Math.cos(a), radius * Math.sin(a)]);
    paths = [[...corners, corners[0]]];
  } else if (family === "curve") paths = [arcPoints(length / 2, rng.uniform(Math.PI * 0.7, Math.PI * 1.5), rng.uniform(0, 2 * Math.PI))];
  else if (family === "composition") {
    const first = [[-length * 0.35, 0.0], [length * 0.35, 0.0]];
    const second = arcPoints(length / 3, rng.uniform(Math.PI * 0.6, Math.PI), rng.uniform(0, 2 * Math.PI));
    let offset = [rng.uniform(-1, 1), rng.uniform(-1, 1)];
    const norm = Math.max(Math.sqrt(offset[0] * offset[0] + offset[1] * offset[1]), 1e-9);
    offset = [(offset[0] / norm) * length * 0.55, (offset[1] / norm) * length * 0.55];
    paths = [first, second.map((p) => [p[0] + offset[0], p[1] + offset[1]])];
  } else throw Error(`unknown family ${family}`);
  const c = Math.cos(rotation), s = Math.sin(rotation);
  return placePaths(paths.map((path) => path.map((p) => rotatePoint(p, c, s))), rng, config);
}

/** A body configuration whose hand starts somewhere on the canvas (env.py start_angles). */
export function startAngles(rng, config, lengths = [0.5, 0.5]) {
  for (let k = 0; k < 500; k++) {
    const theta = [rng.uniform(-Math.PI, Math.PI), rng.uniform(-Math.PI, Math.PI)];
    const hand = [lengths[0] * Math.cos(theta[0]) + lengths[1] * Math.cos(theta[0] + theta[1]), lengths[0] * Math.sin(theta[0]) + lengths[1] * Math.sin(theta[0] + theta[1])];
    const pixel = pixelOf(hand, config);
    if (pixel[0] >= config.margin && pixel[1] >= config.margin && pixel[0] <= config.size - 1 - config.margin && pixel[1] <= config.size - 1 - config.margin) return theta;
  }
  return [rng.uniform(-Math.PI, Math.PI), rng.uniform(-Math.PI, Math.PI)];
}

/** One target of a family, rotated and scaled, with the body pose the drawing starts from. */
export function familyDrawing(family, rng, config, { rotation = null, scale = null } = {}) {
  const turn = rotation === null ? rng.uniform(0, 2 * Math.PI) : rotation;
  const size = scale === null ? rng.uniform(0.55, 0.85) : scale;
  const paths = shapeOf(family, rng, turn, size, config);
  return { name: FAMILY_LABELS[family] || family, family, rotation: turn, scale: size, paths, target: renderPaths(paths, config.size), theta: startAngles(rng, config) };
}

/** A visitor's strokes, in canvas pixel coordinates, as a target. */
export function drawnTarget(strokes, config, theta) {
  const paths = strokes.filter((s) => s.length).map((s) => s.map((p) => [p[0], p[1]]));
  return { name: "your figure", family: "drawn", rotation: 0, scale: 0, paths, target: renderPaths(paths, config.size), theta };
}

// -- the world

/** One canvas, one target, one drawing at a time (env.py CanvasWorld). */
export class CanvasWorld {
  constructor(config, body, { seed = 0, lifeId = "artist" } = {}) {
    this.config = config; this.body = body; this.lifeId = lifeId;
    this.rng = new PageRandom(seed);
    this._episode = -1; this._event = 0; this._tick = 0;
    this._canvas = new Uint8Array(config.size * config.size);
    this._target = new Uint8Array(config.size * config.size);
    this._pen = 0;
    this._mark = new Float64Array(config.window * config.window);
    this._discrepancy = 1.0; this._best = 1.0; this._sinceBest = 0; this._pending = null; this._decisionAction = null;
    this.strokes = 0; this.drawing = null; this.lastStroke = null;
  }

  get canvas() { return Uint8Array.from(this._canvas); }
  get target() { return Uint8Array.from(this._target); }
  get discrepancy() { return this._discrepancy; }
  get penState() { return this._pen; }
  get tick() { return this._tick; }
  get episode() { return this._episode; }
  get penPixel() { return pixelOf(this.body.hand, this.config); }

  /** The visual channels the supplied planner reads: both aligned images and the gaze. */
  view() { return { canvas: this.canvas, target: this.target, pen: this.penPixel, pen_state: this._pen }; }

  measure() {
    let ink = 0;
    for (const v of this._canvas) ink += v;
    return { chamfer: chamferOf(this._canvas, this._target, this.config.size), ink, strokes: this.strokes, ...foregroundF1(this._canvas, this._target, this.config.size) };
  }

  reset(drawing, theta = null) {
    this.drawing = drawing;
    this._target = Uint8Array.from(drawing.target);
    this._canvas = new Uint8Array(this.config.size * this.config.size);
    this._episode += 1; this._tick = 0; this._pen = 0; this.strokes = 0;
    this._mark = new Float64Array(this.config.window * this.config.window);
    this._pending = null; this.lastStroke = null;
    this.body.reset({ theta: theta === null ? drawing.theta : theta });
    this._discrepancy = chamferOf(this._canvas, this._target, this.config.size);
    this._best = this._discrepancy; this._sinceBest = 0;
    return this._moment({ feedback: false, reward: 0.0, terminated: false, truncated: false });
  }

  observation() {
    const origin = windowOrigin(this.penPixel, this.config.window), size = this.config.size;
    return {
      ...this.body.observation(),
      pen: Float64Array.of(this._pen),
      mark: Float64Array.from(this._mark),
      canvas_local: cropWindow(this._canvas, origin, this.config.window, size),
      target_local: cropWindow(this._target, origin, this.config.window, size),
    };
  }

  _moment({ feedback, reward, terminated, truncated }) {
    const observation = this.observation();
    const m = {
      __moment: true, life_id: this.lifeId, episode_id: this._episode, event_id: this._event, tick: this._tick,
      dt: this.body.config.dt * this.body.config.substeps, observation,
      observed: Object.fromEntries(Object.entries(observation).map(([k, v]) => [k, new Uint8Array(v.length).fill(1)])),
      action_mask: new Uint8Array(CHOICES).fill(1),
      feedback_for: feedback ? this._pending : null, executed: feedback ? this._decisionAction : null,
      reward, reward_known: feedback, terminated, truncated,
      final_observation: truncated ? this.observation() : null,
      goal: new Float64Array(4), // the intention the brain commits is written in by its own loop
      replay_of: null, has_feedback: feedback, any_legal: true,
    };
    this._event += 1;
    return m;
  }

  act(decisionId, action) {
    if (this._pending !== null) throw Error("the previous decision has not been fed back");
    const [torque, pen] = splitChoice(action);
    this._pending = decisionId; this._decisionAction = action | 0;
    const w = this.config.window, size = this.config.size;
    const origin = windowOrigin(this.penPixel, w);
    const before = cropWindow(this._canvas, origin, w, size);
    const from = this.penPixel, start = Float64Array.of(from[0] + this.config.pen_offset[0], from[1] + this.config.pen_offset[1]);
    this.body.act(decisionId, torque);
    const to = this.penPixel, end = Float64Array.of(to[0] + this.config.pen_offset[0], to[1] + this.config.pen_offset[1]);
    this.lastStroke = null;
    if (pen) {
      const drawn = drawPoints(this._canvas, bresenham(start, end), size);
      if (drawn.length && !this._pen) this.strokes += 1;
      this.lastStroke = drawn;
    }
    this._pen = pen;
    const after = cropWindow(this._canvas, origin, w, size);
    this._mark = Float64Array.from(after, (v, k) => v - before[k]);
    this._tick += 1;
    const discrepancy = chamferOf(this._canvas, this._target, size);
    const t = TORQUE_VECTORS[torque];
    const cost = this.config.torque_cost * (t[0] * t[0] + t[1] * t[1]);
    const reward = this._discrepancy - discrepancy - cost;
    this._discrepancy = discrepancy;
    if (discrepancy < this._best - 1e-9) { this._best = discrepancy; this._sinceBest = 0; }
    else if (this._canvas.some((v) => v)) this._sinceBest += 1; // the clock starts at the first mark
    const terminated = discrepancy <= this.config.finish;
    const truncated = !terminated && (this._tick >= this.config.max_decisions || this._sinceBest >= this.config.patience);
    const m = this._moment({ feedback: true, reward, terminated, truncated });
    this._pending = null;
    return m;
  }
}

export const TORQUE_VECTORS = [];
for (const a of [-1.0, 0.0, 1.0]) for (const b of [-1.0, 0.0, 1.0]) TORQUE_VECTORS.push([a, b]);

/** Python's `(angle + pi) % (2 pi) - pi`: one modulus, the sign taken from the divisor. */
export function wrapAngle(angle) {
  const two = 2 * Math.PI;
  let r = (angle + Math.PI) % two;
  if (r < 0) r += two;
  return r - Math.PI;
}

/**
 * The S01 two-link arm as the canvas world holds it (arm/env.py Arm with its own goal switched
 * off: no target, no success radius, no horizon, since this world supplies the reward). Links of
 * 0.5, angles wrapped to [-pi, pi], velocities clipped to +-max_velocity, a 50 Hz body clock with
 * semi-implicit Euler, one torque pair held for `substeps` body steps. The joint state moves by
 * additions, multiplications and `wrapAngle` alone, so it stays bit-identical to the Python body;
 * the sines, cosines and the hand position are read out from it. `trace` keeps the joint angles
 * after every body step, which the page draws the pen's path from.
 */
export class CanvasArm {
  constructor(config = {}) {
    this.config = { lengths: [...(config.lengths || [0.5, 0.5])], dt: config.dt ?? 0.02, substeps: config.substeps ?? 5, max_velocity: config.max_velocity ?? 2.0, friction: config.friction ?? 0.1, gain: config.gain ?? 1.0 };
    this.theta = new Float64Array(2); this.omega = new Float64Array(2);
    this._flags = new Float64Array(2);
    this._previous = { hand: new Float64Array(2), omega: new Float64Array(2), theta: new Float64Array(2), d_hand: new Float64Array(2) };
    this.trace = null;
  }

  handOf(theta) {
    const [l1, l2] = this.config.lengths;
    return Float64Array.of(l1 * Math.cos(theta[0]) + l2 * Math.cos(theta[0] + theta[1]), l1 * Math.sin(theta[0]) + l2 * Math.sin(theta[0] + theta[1]));
  }
  elbowOf(theta) { const l1 = this.config.lengths[0]; return Float64Array.of(l1 * Math.cos(theta[0]), l1 * Math.sin(theta[0])); }
  get hand() { return this.handOf(this.theta); }
  get elbow() { return this.elbowOf(this.theta); }

  reset({ theta = null } = {}) {
    this.theta = theta === null ? new Float64Array(2) : Float64Array.from(theta);
    this.omega = new Float64Array(2);
    this._flags = new Float64Array(2);
    this.trace = null;
    this._previous = { hand: this.hand, omega: Float64Array.from(this.omega), theta: Float64Array.from(this.theta), d_hand: new Float64Array(2) };
    return this;
  }

  observation() {
    const hand = this.hand, p = this._previous, c = this.config;
    const dHand = Float64Array.of(hand[0] - p.hand[0], hand[1] - p.hand[1]);
    return {
      angles: Float64Array.of(Math.sin(this.theta[0]), Math.cos(this.theta[0]), Math.sin(this.theta[1]), Math.cos(this.theta[1])),
      velocity: Float64Array.of(this.omega[0] / c.max_velocity, this.omega[1] / c.max_velocity),
      hand,
      d_hand: dHand,
      dd_hand: Float64Array.of(dHand[0] - p.d_hand[0], dHand[1] - p.d_hand[1]),
      d_velocity: Float64Array.of(this.omega[0] - p.omega[0], this.omega[1] - p.omega[1]),
      d_angles: Float64Array.of(wrapAngle(this.theta[0] - p.theta[0]), wrapAngle(this.theta[1] - p.theta[1])),
      flags: Float64Array.from(this._flags),
    };
  }

  /** Hold one torque pair for `substeps` body steps; record velocity-limit flags and every pose. */
  integrate(action) {
    const c = this.config, pair = TORQUE_VECTORS[action | 0];
    const torque = [pair[0] * c.gain, pair[1] * c.gain];
    const flags = new Float64Array(2), trace = new Float64Array(2 * c.substeps);
    for (let k = 0; k < c.substeps; k++) {
      for (let j = 0; j < 2; j++) {
        this.omega[j] = this.omega[j] + c.dt * (torque[j] - c.friction * this.omega[j]);
        if (Math.abs(this.omega[j]) >= c.max_velocity) flags[j] = Math.max(flags[j], 1.0);
        this.omega[j] = Math.min(c.max_velocity, Math.max(-c.max_velocity, this.omega[j]));
      }
      for (let j = 0; j < 2; j++) this.theta[j] = wrapAngle(this.theta[j] + c.dt * this.omega[j]);
      trace[2 * k] = this.theta[0]; trace[2 * k + 1] = this.theta[1];
    }
    this._flags = flags;
    this.trace = trace;
  }

  act(decisionId, action) {
    const a = action | 0;
    if (!(a >= 0 && a < TORQUE_VECTORS.length)) throw Error("action outside the nine torque pairs");
    const hand = this.hand, p = this._previous;
    this._previous = { hand, omega: Float64Array.from(this.omega), theta: Float64Array.from(this.theta), d_hand: Float64Array.of(hand[0] - p.hand[0], hand[1] - p.hand[1]) };
    this.integrate(a);
  }
}

/** The S01 body with its own goal switched off: this world supplies the reward (env.py body_config). */
export function canvasBody(body = {}, { gain = 1.0 } = {}) {
  return new CanvasArm({ lengths: [...(body.lengths || [0.5, 0.5])], substeps: body.substeps ?? 5, gain });
}

// -- imagined canvases

/** The canvas pixels a predicted window expects ink in, with the weight it gives each. */
export function markWeights(predicted, origin, floor, size, window) {
  const points = [], weights = [];
  for (let i = 0; i < window; i++) for (let j = 0; j < window; j++) {
    const value = predicted[i * window + j];
    if (!(value >= floor)) continue;
    const r = origin[0] + i, c = origin[1] + j;
    if (r < 0 || c < 0 || r >= size || c >= size) continue;
    points.push(r, c); weights.push(value);
  }
  return { points: Int32Array.from(points), weights: Float64Array.from(weights) };
}

/** The imagined next observation: the S01 body composition plus the pen, the ink and the windows. */
export function composeCanvas(observation, deltas, pen, canvas, target, config) {
  const out = compose(observation, deltas);
  const origin = windowOrigin(pixelOf(out.hand, config), config.window);
  out.pen = Float64Array.of(pen);
  out.mark = Float64Array.from(deltas.mark, (v) => Math.min(1.0, Math.max(0.0, v)));
  out.canvas_local = cropWindow(canvas, origin, config.window, config.size);
  out.target_local = cropWindow(target, origin, config.window, config.size);
  return out;
}

// -- the slow level: an intention

/** A proposed stroke: where the pen should go next and whether it draws on the way. */
export function makeIntention(endpoint, pen, direction, distance, score = 0.0, valid = true) {
  return { endpoint, pen, direction, distance, score, valid };
}
export function emptyIntention() { return makeIntention(Float64Array.of(0, 0), 0, Float64Array.of(0, 0), 0.0, 0.0, false); }

/** The intention as the goal port reads it (brain.py Intention.encode). */
export function encodeIntention(intention, maxDistance, gain) {
  return Float64Array.of(
    gain * (0.5 + 0.5 * intention.direction[0]),
    gain * (0.5 + 0.5 * intention.direction[1]),
    gain * Math.min(intention.distance / maxDistance, 1.0),
    gain * intention.pen,
  );
}

/** Lexicographically sorted unique (row, column) pairs, as `np.unique(points, axis=0)`. */
function uniqueRows(flat) {
  const count = flat.length / 2;
  const order = Array.from({ length: count }, (_, k) => k).sort((x, y) => flat[2 * x] - flat[2 * y] || flat[2 * x + 1] - flat[2 * y + 1]);
  const out = [];
  for (const k of order) {
    const r = flat[2 * k], c = flat[2 * k + 1];
    if (out.length && out[out.length - 2] === r && out[out.length - 1] === c) continue;
    out.push(r, c);
  }
  return Int32Array.from(out);
}

/** Supplied search over local endpoints, scored by the canvases the learned ink model imagines. */
export class IntentionSearch {
  constructor({ directions = 8, distances = [3.0, 7.0, 12.0], beam = 4, horizon = 2, approach = 0.1, travel = 0.01, step = 1.2, relative = 0.35 } = {}) {
    this.directions = directions | 0; this.distances = [...distances]; this.beam = beam | 0; this.horizon = horizon | 0;
    this.approach = approach; this.travel = travel; this.step = step; this.relative = relative;
    this.lastStamps = null; this.lastChoice = null;
  }

  units() {
    return Array.from({ length: this.directions }, (_, k) => { const a = (k * 2 * Math.PI) / this.directions; return [Math.sin(a), Math.cos(a)]; });
  }

  /** What the learned model says one decision of motion leaves, per direction and pen state. */
  stamps(model, observation, config, goal) {
    const observations = [], actions = [], keys = [];
    for (const [k, unit] of this.units().entries()) {
      const delta = Float64Array.of(unit[1] * this.step * config.pixel, -unit[0] * this.step * config.pixel);
      for (const pen of [0, 1]) {
        const imagined = {};
        for (const [name, value] of Object.entries(observation)) imagined[name] = Float64Array.from(value);
        imagined.d_hand = Float64Array.from(delta);
        imagined.dd_hand = new Float64Array(2);
        imagined.pen = Float64Array.of(pen);
        observations.push(imagined); actions.push(joinChoice(4, pen)); keys.push([k, pen]);
      }
    }
    const predictions = model.predictBatch(observations, actions, goal);
    const windows = predictions.map((p) => p.mark);
    let peak = -Infinity;
    for (const w of windows) for (const v of w) if (v > peak) peak = v;
    const floor = Math.max(this.relative * peak, 1e-6);
    const window = config.window;
    return keys.map(([k, pen], index) => {
      const offsets = [];
      const w = windows[index];
      for (let i = 0; i < window; i++) for (let j = 0; j < window; j++) if (w[i * window + j] >= floor) offsets.push(i - (window >> 1), j - (window >> 1));
      return { k, pen, offsets: Int32Array.from(offsets) };
    });
  }

  /** The pixels a stroke of that length would ink, the stamp repeated along the path. */
  marks(offsets, start, unit, distance, size) {
    if (!offsets.length) return new Int32Array(0);
    const count = distance > 0 ? Math.ceil((distance + 1e-9) / this.step) : 1;
    const flat = [];
    for (let s = 0; s < count; s++) {
      const t = distance > 0 ? s * this.step : 0.0;
      const cr = rint(start[0] + t * unit[0]), cc = rint(start[1] + t * unit[1]);
      for (let k = 0; k < offsets.length; k += 2) {
        const r = cr + offsets[k], c = cc + offsets[k + 1];
        if (r < 0 || c < 0 || r >= size || c >= size) continue;
        flat.push(r, c);
      }
    }
    return uniqueRows(Int32Array.from(flat));
  }

  choose(model, observation, sheet, penPixel, config, goal) {
    const stamps = this.stamps(model, observation, config, goal);
    const units = this.units(), size = config.size;
    let expansions = 2 * this.directions;
    let beam = [{ state: sheet.copy(), position: Float64Array.from(penPixel), score: 0.0, first: null }];
    let best = null;
    let level = [];
    for (let depth = 0; depth < this.horizon; depth++) {
      const children = [];
      for (const entry of beam) {
        for (const stamp of stamps) {
          for (const distance of [0.0, ...this.distances]) {
            const unit = units[stamp.k];
            const end = Float64Array.of(entry.position[0] + unit[0] * distance, entry.position[1] + unit[1] * distance);
            if (end[0] < 0 || end[1] < 0 || end[0] > size - 1 || end[1] > size - 1) continue;
            const child = entry.state.copy();
            const gain = -child.apply(this.marks(stamp.offsets, entry.position, unit, distance, size));
            const total = entry.score + gain - (this.travel * distance) / size;
            const leaf = total - (this.approach * IntentionSearch.reach(child, end)) / size;
            const intention = makeIntention(end, stamp.pen, Float64Array.from(unit), distance, leaf);
            children.push({ state: child, position: end, score: total, first: entry.first === null ? intention : entry.first, leaf, intention });
            expansions += 1;
          }
        }
      }
      if (!children.length) break;
      children.sort((x, y) => y.leaf - x.leaf); // Python sorts by -leaf, stably
      const kept = children.slice(0, this.beam);
      for (const child of kept) if (best === null || child.leaf > best.leaf) best = { leaf: child.leaf, first: child.first };
      if (depth === 0) level = kept.map((c) => ({ endpoint: [c.intention.endpoint[0], c.intention.endpoint[1]], pen: c.intention.pen, leaf: c.leaf }));
      beam = kept.map((c) => ({ state: c.state, position: c.position, score: c.score, first: c.first }));
    }
    this.lastStamps = stamps;
    const intention = best === null ? emptyIntention() : best.first;
    this.lastChoice = { intention, beam: level };
    return { intention, budget: { intentions: expansions } };
  }

  /** Pixels from a position to the nearest target pixel no ink has reached. */
  static reach(state, position) {
    const uncovered = state.uncovered();
    if (!uncovered.n) return 0.0;
    let best = Infinity;
    for (let j = 0; j < uncovered.n; j++) {
      const dr = uncovered.rows[j] - position[0], dc = uncovered.cols[j] - position[1];
      const d = Math.sqrt(dr * dr + dc * dc);
      if (d < best) best = d;
    }
    return best;
  }
}

// -- the fast level: torques and the pen

/** Bounded search over the eighteen choices through the learned body and ink models. */
export class CanvasPlanner {
  constructor({ rollout = 2, canvas_weight = 3.0, relative = 0.25, objective = "velocity", coast = 3, gain = 4.0, max_speed = 0.18, decision_seconds = 0.1 } = {}) {
    this.rollout = rollout | 0; this.canvasWeight = canvas_weight; this.relative = relative;
    this.objective = objective; this.coast = coast; this.gain = gain; this.maxSpeed = max_speed; this.decisionSeconds = decision_seconds;
    this.lastImagined = null;
  }

  tracker() {
    return new ModelPlanner({ depth: 1, beam: 1, rollout: 1, objective: this.objective, coast: this.coast, gain: this.gain, max_speed: this.maxSpeed, decision_seconds: this.decisionSeconds });
  }

  plan(model, observation, sheet, penPixel, intention, config, canvas, target, goal) {
    const tracker = this.tracker();
    const endpoint = handOfPixel(intention.valid ? intention.endpoint : penPixel, config);
    const actions = Array.from({ length: CHOICES }, (_, k) => k);
    let observations = actions.map(() => observation);
    const sheets = actions.map(() => sheet.copy());
    const pixels = actions.map(() => Float64Array.from(penPixel));
    const scores = new Float64Array(CHOICES);
    const imagined = actions.map(() => ({ points: [], pen: 0 }));
    let first = [], expansions = 0;
    const rollout = Math.max(1, this.rollout);
    for (let step = 0; step < rollout; step++) {
      const predictions = model.predictBatch(observations, actions, goal);
      expansions += actions.length;
      if (step === 0) first = predictions.slice();
      let peak = -Infinity;
      for (const p of predictions) for (const v of p.mark) if (v > peak) peak = v;
      const floor = Math.max(this.relative * peak, 1e-6);
      const composed = [];
      for (let k = 0; k < actions.length; k++) {
        const { points, weights } = markWeights(predictions[k].mark, windowOrigin(pixels[k], config.window), floor, config.size, config.window);
        scores[k] -= this.canvasWeight * sheets[k].apply(points, weights);
        const pen = actions[k] % 2;
        const next = composeCanvas(observations[k], predictions[k], pen, sheets[k].drawn, target, config);
        if (step === rollout - 1) scores[k] += tracker.leafScore(next, endpoint);
        composed.push(next);
        pixels[k] = pixelOf(next.hand, config);
        if (step === 0) { imagined[k].pen = pen; for (let j = 0; j < points.length; j += 2) imagined[k].points.push([points[j], points[j + 1], weights[j / 2]]); }
      }
      observations = composed;
    }
    let action = 0;
    for (let k = 1; k < CHOICES; k++) if (scores[k] > scores[action]) action = k;
    this.lastImagined = { chosen: action, scores: Array.from(scores), candidates: imagined, endpoint: [endpoint[0], endpoint[1]] };
    return { action, prediction: first[action], budget: { expansions, rollout } };
  }
}

// -- the candidate life

/** One artist: the agent, its belief of the canvas, its intention and the supplied search (brain.py ArtistLife). */
export class ArtistBrain {
  constructor(agent, config) {
    this.agent = agent;
    this.geometry = canvasConfig(config.canvas || {});
    this.search = new IntentionSearch(config.intention || {});
    this.planner = new CanvasPlanner(config.planner || {});
    this.replanEvery = config.intention_interval | 0;
    this.intentionGain = config.intention_gain ?? 0.3;
    this.maxDistance = Math.max(...this.search.distances);
    this.sheet = null;
    this.belief = new Uint8Array(this.geometry.size * this.geometry.size);
    this.target = new Uint8Array(this.geometry.size * this.geometry.size);
    this.penPixel = new Float64Array(2);
    this.intention = emptyIntention();
    this.countdown = 0;
    this.budget = {};
    this.decision = null;
    this.lastPrediction = null;
    this.replanned = false;
    agent.setPlanner((a, row, moment) => {
      if (this.sheet === null) throw Error("the artist plans on a drawing it has seen");
      const out = this.planner.plan(a, moment.observation, this.sheet, this.penPixel, this.intention, this.geometry, this.belief, this.target, moment.goal);
      return { ...out, budget: { ...out.budget, ...this.budget } };
    });
  }

  /** A new drawing: the target is seen, the belief starts from the canvas in front of it. */
  begin(view) {
    this.target = Uint8Array.from(view.target);
    this.belief = Uint8Array.from(view.canvas);
    this.sheet = new Sheet(this.target, this.geometry.size, this.geometry.cap);
    this.sheet.observe(this.belief);
    this.penPixel = Float64Array.from(view.pen);
    this.intention = emptyIntention();
    this.countdown = 0;
  }

  observe(view) {
    this.penPixel = Float64Array.from(view.pen);
    this.belief = Uint8Array.from(view.canvas);
    if (this.sheet !== null) this.sheet.observe(this.belief);
  }

  /** The moment the agent reads: the committed intention in the goal port. */
  dress(moment) { return { ...moment, goal: encodeIntention(this.intention, this.maxDistance, this.intentionGain) }; }

  step(moment, view = null) {
    if (view !== null) this.observe(view);
    this.replanned = false;
    if (this.sheet !== null && this.countdown <= 0 && !(moment.terminated || moment.truncated)) {
      const goal = encodeIntention(this.intention, this.maxDistance, this.intentionGain);
      const out = this.search.choose(this.agent, moment.observation, this.sheet, this.penPixel, this.geometry, goal);
      this.intention = out.intention; this.budget = out.budget;
      this.countdown = this.replanEvery;
      this.replanned = true;
    }
    this.countdown -= 1;
    const decision = this.agent.step(this.dress(moment));
    this.decision = decision;
    this.lastPrediction = decision === null ? null : decision.prediction;
    return decision;
  }

  /** Leave one environment for another: a decision without its outcome is dropped, the cursors start afresh. */
  detach() { this.agent.abandon(0); this.agent.newStream(0); }
}
