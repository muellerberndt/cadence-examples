// The whole 1943 brain as one scan: every core neuron and synapse, every stored route
// address and the value readout, laid out by the standard Cadence atlas and drawn by the
// standard brain-scan renderer. Nothing is sampled: the recording's arrays feed the view.
import { BrainScan, layoutAtlas } from './brain_scan.js';

const TITLES = {
  retina: 'Retina', afterimage: 'Afterglow', v1_fovea: 'Foveal vision', v1_periphery: 'Peripheral vision',
  context: 'Reverberating context', association: 'Association', recall: 'Route recall', motor: 'Motor decisions',
  body: 'Body', efference: 'Motor trace', habit: 'Fixed habit', habit_cue: 'Habit cue',
  'route store': 'Route store', 'value / dopamine': 'Value / dopamine',
};
const ROLES = { efference: 'memory', habit: 'motor', habit_cue: 'motor', 'route store': 'memory', 'value / dopamine': 'value', unassigned: 'other' };
// The eye: three channels (on, off, warmth) of a 48-by-48 fovea, then of a 56-by-60 periphery,
// channel-major. V1: eight detectors per receptive-field position, feature-major.
const FOVEA_SIDE = 48, PERIPHERY = [56, 60], CHANNELS = 3, V1 = { v1_fovea: [15, 15], v1_periphery: [14, 15] };

function retinaPositions(count) {
  const f = FOVEA_SIDE * FOVEA_SIDE, p = PERIPHERY[0] * PERIPHERY[1];
  if (count !== CHANNELS * (f + p)) return null;
  const out = new Array(count), fovea = CHANNELS * f;
  for (let i = 0; i < count; i++) {
    if (i < fovea) {
      const c = Math.floor(i / f), rem = i % f, r = Math.floor(rem / FOVEA_SIDE), col = rem % FOVEA_SIDE, o = 0.28 * (c - 1);
      out[i] = [((col + 0.5 + o) / FOVEA_SIDE) * 0.9 - 1.0, 0.45 - ((r + 0.5 + o) / FOVEA_SIDE) * 0.9];
    } else {
      const j = i - fovea, c = Math.floor(j / p), rem = j % p, r = Math.floor(rem / PERIPHERY[1]), col = rem % PERIPHERY[1], o = 0.28 * (c - 1);
      out[i] = [(col + 0.5 + o) / PERIPHERY[1], 0.5 - (r + 0.5 + o) / PERIPHERY[0]];
    }
  }
  return out;
}
function v1Positions(count, [rows, cols]) {
  const cells = rows * cols, features = Math.round(count / cells);
  if (features * cells !== count || features < 1) return null;
  const out = new Array(count);
  for (let i = 0; i < count; i++) {
    const j = Math.floor(i / cells), rem = i % cells, py = Math.floor(rem / cols), px = rem % cols, a = (2 * Math.PI * j) / features;
    out[i] = [((px + 0.5 + 0.3 * Math.cos(a)) / cols) * 2 - 1, 1 - ((py + 0.5 + 0.3 * Math.sin(a)) / rows) * 2];
  }
  return out;
}
const intensity = (x, gain) => Math.min(1, Math.log1p(Math.abs(x) * gain) / 3);

export class Circuit {
  constructor(canvas, labels, g) {
    this.canvas = canvas; this.labels = labels; this.g = g;
    this.n = g.meta.neurons; this.e = g.meta.synapses;
    this.addresses = g.meta.addresses; this.r = this.addresses.length;
    this.recall = g.meta.populations.find((p) => p.name === 'recall')?.indices || [];
    this.ports = this.recall.length; // one drawn synapse per stored value, address to recall neuron
    this.critic = this.n + this.r; this.total = this.critic + 2;
    this.totalEdges = this.e + this.r * this.ports + g.meta.critic_index.length;
    // Regions: the recorded populations, the route store (one vertex per address) and the value readout.
    const groups = new Array(this.total).fill('unassigned');
    for (const p of g.meta.populations) for (const i of p.indices) groups[i] = p.name;
    for (let i = this.n; i < this.critic; i++) groups[i] = 'route store';
    groups[this.critic] = groups[this.critic + 1] = 'value / dopamine';
    const positions = {};
    for (const p of g.meta.populations) {
      const shape = V1[p.name];
      const given = p.name === 'retina' || p.name === 'afterimage' ? retinaPositions(p.indices.length) : shape ? v1Positions(p.indices.length, shape) : null;
      if (given) positions[p.name] = given;
    }
    const pre = new Uint32Array(this.totalEdges), post = new Uint32Array(this.totalEdges);
    pre.set(g.arrays.pre); post.set(g.arrays.post);
    let at = this.e;
    for (let i = 0; i < this.r; i++) for (let j = 0; j < this.ports; j++) { pre[at] = this.n + i; post[at] = this.recall[j]; at++; }
    for (const id of g.meta.critic_index) { pre[at] = id; post[at] = this.critic; at++; }
    const names = [...new Set(groups)];
    const atlas = layoutAtlas({
      n: this.total, pre, post, groups, roles: ROLES, positions, seed: 1943,
      labels: Object.fromEntries(names.map((name) => [name, TITLES[name] ?? name])),
    });
    this.names = groups.map((name) => TITLES[name] ?? name);
    this.scan = new BrainScan(canvas, atlas, { labels, montageRows: 0, heatDecay: 0.8 });
    if (!this.scan.enabled) throw Error('This whole-circuit view needs WebGL 2. Try a browser with hardware acceleration.');
    this.weight = new Float32Array(this.totalEdges);
    this.level = new Float32Array(this.total); this.heat = new Float32Array(this.total); this.message = new Float32Array(this.total);
    this.signals = { repair: new Float32Array(this.total), activity: new Float32Array(this.total), plasticity: new Float32Array(this.total), adaptation: new Float32Array(this.total) };
    this.mode = 'repair'; this.last = null; this.maxRepair = 0; this.changed = 0; this.lastDraw = 0; this.drawCost = 0;
    this.scan.onhover = (hit) => this.inspect(hit);
    new ResizeObserver(() => this.draw()).observe(canvas.parentElement);
  }
  fit() { this.scan.fit(); }
  update({ v, s, previous, previousActivation, adaptation, weights, oldWeights, bias, oldBias, memory, critic, oldCritic, criticBias = 0, dopamine = 0 }) {
    const { repair, activity, plasticity, adaptation: adapt } = this.signals;
    repair.fill(0); activity.fill(0); plasticity.fill(0); adapt.fill(0);
    let maxRepair = 0, changed = 0, maxActivation = 1e-12;
    for (let i = 0; i < this.n; i++) {
      const d = previous ? v[i] - previous[i] : 0;
      repair[i] = d; activity[i] = s[i]; adapt[i] = adaptation?.[i] || 0;
      maxRepair = Math.max(maxRepair, Math.abs(d)); maxActivation = Math.max(maxActivation, Math.abs(s[i]));
    }
    for (let i = 0; i < this.e; i++) {
      const w = weights[i], d = oldWeights ? w - oldWeights[i] : 0;
      this.weight[i] = w;
      if (d !== 0) {
        changed++;
        const a = this.g.arrays.pre[i], b = this.g.arrays.post[i];
        if (Math.abs(d) > Math.abs(plasticity[a])) plasticity[a] = d;
        if (Math.abs(d) > Math.abs(plasticity[b])) plasticity[b] = d;
      }
    }
    const m = memory.arrays, read = memory.meta.read;
    let edge = this.e;
    for (let i = 0; i < this.r; i++) {
      const id = this.n + i;
      activity[id] = i === read ? 1 : 0;
      let delta = 0;
      for (let j = 0; j < this.ports; j++) {
        const k = i * this.ports + j;
        this.weight[edge++] = m.memory[k] || 0;
        const dj = (m.consolidated[k] || 0) - (m.before_consolidated[k] || 0);
        if (Math.abs(dj) > Math.abs(delta)) delta = dj;
      }
      plasticity[id] = delta;
      repair[id] = i === read && memory !== this.last?.memory ? maxRepair : 0;
    }
    let value = criticBias;
    for (let i = 0; i < this.g.meta.critic_index.length; i++) {
      const w = critic?.[i] || 0;
      this.weight[edge++] = w;
      value += w * s[this.g.meta.critic_index[i]];
      const d = oldCritic ? w - oldCritic[i] : 0;
      if (Math.abs(d) > Math.abs(plasticity[this.critic])) plasticity[this.critic] = d;
    }
    activity[this.critic] = value; activity[this.critic + 1] = dopamine;
    repair[this.critic + 1] = dopamine ? maxRepair : 0;
    // messages: the activations sent along synapses, on the brain's own scale
    for (let i = 0; i < this.total; i++) this.message[i] = activity[i] / maxActivation;
    // the glow: what changed since the previous iteration (or since the last decision)
    const change = previous ? repair : previousActivation ? s.map((x, i) => x - previousActivation[i]) : null;
    let maxChange = 1e-12;
    if (change) for (let i = 0; i < this.n; i++) maxChange = Math.max(maxChange, Math.abs(change[i]));
    const gain = 20 / maxChange;
    for (let i = 0; i < this.n; i++) this.heat[i] = change ? intensity(change[i], gain) : 0;
    for (let i = this.n; i < this.total; i++) this.heat[i] = repair[i] ? 1 : 0;
    // The resting raster is re-rasterised only when a parameter, memory or critic packet changed.
    const relearned = !this.last || this.last.weights !== weights || this.last.memory !== memory || this.last.critic !== critic;
    this.last = { v, s, previous, weights, bias, memory, dopamine, read, critic };
    this.maxRepair = maxRepair; this.changed = changed;
    if (relearned) this.scan.setWeights(this.weight);
    this.refresh();
  }
  /** The brightness shown follows the chosen signal on a logarithmic scale over the brain. */
  refresh() {
    const signal = this.signals[this.mode] ?? this.signals.repair;
    let max = 1e-12;
    for (let i = 0; i < this.total; i++) max = Math.max(max, Math.abs(signal[i]));
    const gain = 20 / max;
    for (let i = 0; i < this.total; i++) this.level[i] = Math.sign(signal[i]) * intensity(signal[i], gain);
    this.scan.show(this.message, this.heat, { level: this.level, potential: this.last?.v });
    this.lastDraw = performance.now();
  }
  draw() { if (this.last) this.refresh(); else this.scan.draw(); }
  /** Animate the messages between updates, at a rate the machine can afford. */
  animate(now) {
    // A software renderer (no GPU) keeps the recording's own pace instead of animating.
    if (!this.last || this.scan.software || this.drawCost > 400 || now - this.lastDraw < Math.max(40, 3 * this.drawCost)) return;
    const start = performance.now();
    this.scan.draw(now);
    this.drawCost = performance.now() - start;
    this.lastDraw = now;
  }
  inspect(hit) {
    if (!hit || !this.last) return;
    const id = hit.neuron, l = this.last, fmt = (x) => Number(x || 0).toExponential(5);
    let text = `${this.names[id]} · ${id < this.n ? 'neuron' : 'port'} ${id}\n`;
    if (id < this.n) {
      let n = 0, net = 0;
      for (let i = 0; i < this.e; i++) if (this.g.arrays.post[i] === id) { n++; net += l.weights[i] * l.s[this.g.arrays.pre[i]]; }
      text += `v ${fmt(l.v[id])}  activity ${fmt(l.s[id])}  bias ${fmt(l.bias?.[id])}  repair ${fmt(this.signals.repair[id])}\n${n} incoming synapses · derived input ${fmt(net)}`;
    } else if (id < this.critic) {
      const a = id - this.n, address = this.addresses[a];
      const stored = l.memory.arrays.memory.subarray(a * this.ports, (a + 1) * this.ports);
      let mass = 0; for (const w of stored) mass += Math.abs(w);
      text += `address ${address[2]} · ${address[0]} ${address[1] ? 'mirrored' : 'played'} · ${a === l.read ? 'reading' : 'inactive'}\n${this.ports} stored values · |weight| sum ${fmt(mass)}`;
    } else if (id === this.critic) text += `value ${fmt(this.signals.activity[id])} · linear readout of ${this.g.meta.critic_index.length} association neurons`;
    else text += `dopamine ${fmt(this.signals.activity[id])} · global modulator`;
    document.getElementById('inspector').textContent = text;
  }
}
