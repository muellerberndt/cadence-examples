// Drawing the worm: a translucent body with its whole nervous system inside,
// the 302 neurons at their measured positions, every synapse and gap junction,
// the transmitter each synapse releases, and each lesson as a wave of local
// change spreading from the command neurons. A second view shows the head.
// Two tones. Structure is cold: the body, the cords, the wiring and every neuron at
// rest are pale steel blue on navy. Activity is hot: a neuron's glow runs from deep
// red through orange to white-hot. What a synapse releases keeps its transmitter's hue.
export const COLORS = {
  structure: [168, 204, 234],
  hot: [255, 98, 40],
  excitatory: [255, 122, 48],   // glutamate, acetylcholine
  inhibitory: [64, 214, 255],   // GABA
  dopamine: [255, 72, 190],
  modulatory: [176, 128, 255],  // serotonin, octopamine, tyramine, peptides
  none: [255, 122, 48],
  gap: [222, 240, 255],         // electrical coupling
  learning: [124, 255, 160],     // plasticity: the one green inside the worm
  odourA: [46, 230, 196],
  odourB: [167, 139, 250],
  food: [217, 249, 157],
  noxious: [255, 64, 110],
  forward: [155, 215, 255],
  reverse: [255, 120, 80],
};
const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
// Activity spans six decades: a sensory cell that smells food sits near 1, a
// motor neuron in the tail near 1e-5. `lum` is its logarithm on 0..1.
export const lum = (a) => { const x = Math.abs(a); return x < 1e-6 ? 0 : Math.min(1, 1 + Math.log10(x) / 6); };
// `fire` is what is drawn: nothing below 3e-5, then a steep rise, so a quiet
// worm is cold and a signal stands out as it spreads and fades hop by hop.
export const fire = (a) => { const L = lum(a); return L <= 0.25 ? 0 : Math.pow((L - 0.25) / 0.75, 1.7); };
const RAMP = [[96, 22, 12], [255, 98, 40], [255, 232, 184]];
export const hot = (g) => { const k = g < 0.5 ? 0 : 1, f = g < 0.5 ? g / 0.5 : (g - 0.5) / 0.5, a = RAMP[k], b = RAMP[k + 1]; return [0, 1, 2].map((i) => Math.round(a[i] + (b[i] - a[i]) * f)); };
const AP_TIP = 0.137, AP_SPAN = 1.162, VENTRAL = 0.62, DORSAL = -0.72, RING = 0.045, HEAD = 0.1;

function sprite(c) {
  const s = document.createElement("canvas"); s.width = s.height = 64;
  const g = s.getContext("2d"), r = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  const hot = c.map((v) => Math.round(v + 0.55 * (255 - v)));
  r.addColorStop(0, rgba(hot, 1));
  r.addColorStop(0.14, rgba(c, 0.9));
  r.addColorStop(0.4, rgba(c, 0.26));
  r.addColorStop(1, rgba(c, 0));
  g.fillStyle = r; g.fillRect(0, 0, 64, 64); return s;
}

export class Renderer {
  constructor(canvas, connectome, spec, params, inset) {
    this.c = canvas; this.g = canvas.getContext("2d"); this.p = params;
    // somata at their measured positions; across the head the dorsal-ventral axis is
    // drawn 2.2 times wider, so the ganglia around the nerve ring can be told apart
    this.cells = connectome.neurons.map((n, i) => {
      const s = Math.min(0.995, Math.max(0.004, (AP_TIP - n.ap) / AP_SPAN));
      const spread = s < 0.12 ? 2.2 : s < 0.2 ? 2.2 - 1.2 * (s - 0.12) / 0.08 : 1;
      return { i, name: n.name, role: n.role, kind: n.kind, transmitter: n.transmitter, family: n.family ?? "none",
               s, dv: Math.max(-0.85, Math.min(0.85, (n.dv / 0.05) * 0.8 * spread)),
               dorsal: /^(DA|DB|DD|AS)\d/.test(n.name) };   // motor neurons whose commissures reach the dorsal cord
    });
    this.sprites = {}; for (const k of Object.keys(COLORS)) this.sprites[k] = sprite(COLORS[k]);
    // every connection, keyed into the brain's A entries for its live weight. Its path
    // follows the anatomy: between head cells through the nerve ring, along the body
    // through the ventral cord, between neighbours directly.
    const key = new Map(); spec.A.rows.forEach((r, k) => key.set(r * spec.H + spec.A.cols[k], k));
    const edge = (a, b, gap) => {
      const ca = this.cells[a], cb = this.cells[b], ds = Math.abs(ca.s - cb.s);
      const via = ca.s < HEAD && cb.s < HEAD ? "ring" : ds > 0.06 ? "cord" : "direct";
      return { a, b, k: key.get(b * spec.H + a), gap, family: gap ? "gap" : ca.family, via,
               n: via === "cord" ? Math.min(14, 3 + Math.ceil(ds * 14)) : via === "ring" ? 7 : 1 };
    };
    this.synapses = connectome.chemical.map(([pre, post]) => edge(pre, post, false)).filter((e) => e.k !== undefined && e.a !== e.b);
    this.gaps = connectome.gap.map(([a, b]) => edge(a, b, true)).filter((e) => e.k !== undefined);
    this.edges = this.synapses.concat(this.gaps);
    // hop distance from the command neurons: the order in which a lesson's
    // correction reaches each cell through local couplings
    const adj = Array.from({ length: spec.H }, () => []);
    for (const e of this.edges) { adj[e.a].push(e.b); adj[e.b].push(e.a); }
    const start = Object.values(params.outputs).flat().map((n) => spec.names.indexOf(n));
    this.hops = new Int32Array(spec.H).fill(99); const q = [...start]; start.forEach((i) => { this.hops[i] = 0; });
    while (q.length) { const i = q.shift(); for (const j of adj[i]) if (this.hops[j] > this.hops[i] + 1) { this.hops[j] = this.hops[i] + 1; q.push(j); } }
    this.cam = { x: 0, y: 0, scale: 0, ready: false };
    this.zoom = 1; this.speckles = new Map();
    this.resize();
    this.inset = inset ? new BrainView(this, inset.canvas, inset.foot) : null;
  }

  resize() {
    const dpr = window.devicePixelRatio || 1, r = this.c.getBoundingClientRect();
    this.c.width = Math.round(r.width * dpr); this.c.height = Math.round(r.height * dpr);
    this.W = r.width; this.H = r.height; this.dpr = dpr;
    this.inset?.resize();
  }
  baseScale() { return Math.min(0.5 * this.W, 0.6 * this.H) / this.p.body_length; }     // the whole body fits, whichever way it lies
  get cx() { return this.W / 2; }
  get cy() { return this.H / 2 - Math.min(50, this.H * 0.05); }
  toScreen(x, y) { return [(x - this.cam.x) * this.cam.scale + this.cx, (y - this.cam.y) * this.cam.scale + this.cy]; }
  toWorld(sx, sy) { return [(sx - this.cx) / this.cam.scale + this.cam.x, (sy - this.cy) / this.cam.scale + this.cam.y]; }

  // ---- body geometry ----------------------------------------------------------------
  frame(trail, N = 96) {
    const L = this.p.body_length, pts = [];
    let seg = 0, acc = 0;
    for (let k = 0; k < N; k++) {
      const want = (k / (N - 1)) * L;
      while (seg < trail.length - 1) {
        const [ax, ay] = trail[seg], [bx, by] = trail[seg + 1], d = Math.hypot(bx - ax, by - ay);
        if (acc + d >= want) { const f = d ? (want - acc) / d : 0; pts.push([ax + f * (bx - ax), ay + f * (by - ay)]); break; }
        acc += d; seg++;
      }
      if (pts.length < k + 1) pts.push(trail[trail.length - 1]);
    }
    const nx = [], ny = [];
    for (let k = 0; k < N; k++) {
      const a = pts[Math.max(0, k - 1)], b = pts[Math.min(N - 1, k + 1)];
      const tx = b[0] - a[0], ty = b[1] - a[1], n = Math.hypot(tx, ty);
      if (n > 1e-9) { nx.push(-ty / n); ny.push(tx / n); } else { nx.push(k ? nx[k - 1] : 0); ny.push(k ? ny[k - 1] : 1); }
    }
    return { pts, nx, ny, N };
  }
  radius(s) {
    const R = 0.043;
    const head = s < 0.08 ? 0.03 + 0.97 * Math.sqrt(s / 0.08) : 1;          // a blunt, rounded nose
    const tail = s > 0.55 ? Math.max(0.06, Math.pow((1 - s) / 0.45, 0.85)) : 1;
    return R * head * tail;
  }
  at(f, s, off) {                 // body coordinate (s along, off across in radii) -> world
    s = Math.min(1, Math.max(0, s));
    const x = s * (f.N - 1), k = Math.min(f.N - 2, Math.floor(x)), t = x - k;
    const px = f.pts[k][0] * (1 - t) + f.pts[k + 1][0] * t, py = f.pts[k][1] * (1 - t) + f.pts[k + 1][1] * t;
    const nx = f.nx[k] * (1 - t) + f.nx[k + 1] * t, ny = f.ny[k] * (1 - t) + f.ny[k + 1] * t;
    const r = this.radius(s) * off;
    return [px + nx * r, py + ny * r];
  }
  // a point on a connection's path
  along(f, T, e, t) {
    const ca = this.cells[e.a], cb = this.cells[e.b];
    if (e.via === "ring") {        // an arc through the nerve ring
      const u = 1 - t, m = 0.25 * (ca.dv + cb.dv);
      return T(...this.at(f, u * u * ca.s + 2 * u * t * RING + t * t * cb.s, u * u * ca.dv + 2 * u * t * m + t * t * cb.dv));
    }
    const base = ca.dv + (cb.dv - ca.dv) * t, w = e.via === "cord" ? Math.min(1, 7 * t * (1 - t)) : 0;
    return T(...this.at(f, ca.s + (cb.s - ca.s) * t, base + (VENTRAL - base) * w));
  }
  route(path, f, T, e, pos) {
    if (e.via === "direct") { const a = pos[e.a], b = pos[e.b]; path.moveTo(a[0], a[1]); path.lineTo(b[0], b[1]); return; }
    for (let j = 0; j <= e.n; j++) { const w = this.along(f, T, e, j / e.n); j ? path.lineTo(w[0], w[1]) : path.moveTo(w[0], w[1]); }
  }
  commissure(path, f, T, c) {     // from a motor neuron round the body wall to the dorsal cord
    for (let j = 0; j <= 6; j++) { const t = j / 6, w = T(...this.at(f, c.s + 0.012 * Math.sin(Math.PI * t), c.dv + (DORSAL - c.dv) * t)); j ? path.lineTo(w[0], w[1]) : path.moveTo(w[0], w[1]); }
  }

  // ---- one frame --------------------------------------------------------------------------
  draw(life, view) {
    const g = this.g, { W, H } = this, p = this.p;
    g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    const f = this.frame(life.body.trail);
    const ends = [0, 0.5, 1].map((s) => this.at(f, s, 0)), mid = [(ends[0][0] + ends[1][0] * 2 + ends[2][0]) / 4, (ends[0][1] + ends[1][1] * 2 + ends[2][1]) / 4];
    const target = this.baseScale() * this.zoom;
    if (!this.cam.ready) Object.assign(this.cam, { x: mid[0], y: mid[1], scale: target, ready: true });
    const k = 1 - Math.exp(-view.dt * 2.2);
    this.cam.x += (mid[0] - this.cam.x) * k; this.cam.y += (mid[1] - this.cam.y) * k;
    this.cam.scale += (target - this.cam.scale) * (1 - Math.exp(-view.dt * 5));
    const S = this.cam.scale, T = (x, y) => this.toScreen(x, y);

    // background: deep navy, a world-anchored blueprint grid, vignette
    g.globalCompositeOperation = "source-over";
    g.fillStyle = "#0c182b"; g.fillRect(0, 0, W, H);
    this.grid(g, 0.1, 0.05); this.grid(g, 0.5, 0.11);
    this.plateEdge(g, life);
    const vig = g.createRadialGradient(W / 2, H / 2, Math.min(W, H) * 0.3, W / 2, H / 2, Math.max(W, H) * 0.75);
    vig.addColorStop(0, "rgba(4,10,22,0)"); vig.addColorStop(1, "rgba(3,8,18,0.6)");
    g.fillStyle = vig; g.fillRect(0, 0, W, H);

    // smells: a faint tint, and contour rings one and two falloff lengths out, so the ground stays dark
    for (const it of life.items) if (it.odour && it.amount > 0) {
      const [x, y] = this.toScreen(it.x, it.y), R = 3 * p.odour_sigma * S;
      if (x < -R || y < -R || x > W + R || y > H + R) continue;
      const c = COLORS[it.odour === "A" ? "odourA" : "odourB"], gr = g.createRadialGradient(x, y, 0, x, y, R), a = 0.075 * Math.sqrt(it.amount);
      gr.addColorStop(0, rgba(c, a)); gr.addColorStop(0.35, rgba(c, a * 0.4)); gr.addColorStop(1, rgba(c, 0));
      g.fillStyle = gr; g.beginPath(); g.arc(x, y, R, 0, Math.PI * 2); g.fill();
      g.lineWidth = 1; g.setLineDash([3, 7]);
      for (const [n, al] of [[1, 0.3], [2, 0.16]]) { g.strokeStyle = rgba(c, al * Math.sqrt(it.amount)); g.beginPath(); g.arc(x, y, n * p.odour_sigma * S, 0, Math.PI * 2); g.stroke(); }
      g.setLineDash([]);
    }
    for (const it of life.items) this.item(g, it, view.now);

    // the body, then the nervous system inside it
    const zs = Math.min(1.7, Math.max(0.3, S / 700));
    this.body(g, f, T, view, S);
    this.nervous(g, f, T, life, view, { W, H, zs, head: 0.5, patch: 3.4, minPatch: 1.15, lw: Math.min(1.4, Math.max(0.5, S / 600)), lines: 700, pulses: 520, glow: 1, ringLines: false, edges: this.edges });

    // events at the mouth
    g.globalCompositeOperation = "lighter";
    const head = this.toScreen(...this.at(f, 0.01, 0));
    for (const e of view.effects) {
      const age = (view.now - e.at) / 1000; if (age > e.dur) continue;
      const t = age / e.dur, c = COLORS[e.color];
      const R = (18 + 120 * t) * Math.min(1.6, S / 700);
      const gr = g.createRadialGradient(head[0], head[1], 0, head[0], head[1], R);
      gr.addColorStop(0, rgba(c, 0.55 * (1 - t))); gr.addColorStop(0.5, rgba(c, 0.18 * (1 - t))); gr.addColorStop(1, rgba(c, 0));
      g.fillStyle = gr; g.beginPath(); g.arc(head[0], head[1], R, 0, Math.PI * 2); g.fill();
      g.strokeStyle = rgba(c, 0.7 * (1 - t)); g.lineWidth = 1.5;
      g.beginPath(); g.arc(head[0], head[1], R * 0.8, 0, Math.PI * 2); g.stroke();
    }
    g.globalCompositeOperation = "source-over";
    this.decor(g);
    this.inset?.draw(life, view);
  }

  // the whole nervous system: wiring, cords, live connections, vesicles, lessons, neurons
  nervous(g, f, T, life, view, o) {
    const act = view.activity, A = life.brain.val, cells = this.cells;
    const pos = cells.map((c) => T(...this.at(f, c.s, c.dv)));
    this.pos = pos;

    // the wiring the brain is allowed to use, cold and thin: synapses and gap junctions in the
    // head and between neighbours, commissures; the long axons are the cords themselves
    g.globalCompositeOperation = "source-over"; g.lineCap = "round";
    const wire = new Path2D();
    for (const e of o.edges) if (e.via !== "cord") this.route(wire, f, T, e, pos);
    for (const c of cells) if (c.dorsal) this.commissure(wire, f, T, c);
    g.strokeStyle = rgba(COLORS.structure, 0.1); g.lineWidth = 0.6 * o.lw; g.stroke(wire);

    // the nerve cords: a cold line, hot where the cells along it fire
    const NB = 72, bin = new Float32Array(NB);
    for (const c of cells) { const b = Math.min(NB - 1, Math.floor(c.s * NB)); bin[b] = Math.max(bin[b], fire(act[c.i])); }
    for (let k = 1; k < NB; k++) bin[k] = Math.max(bin[k], 0.8 * bin[k - 1]);
    for (let k = NB - 2; k >= 0; k--) bin[k] = Math.max(bin[k], 0.8 * bin[k + 1]);
    for (const [side, gain] of [[VENTRAL, 1], [DORSAL, 0.6]]) {
      const cord = new Path2D();
      for (let k = 2; k <= NB - 1; k++) { const a = T(...this.at(f, k / NB, side)); k > 2 ? cord.lineTo(a[0], a[1]) : cord.moveTo(a[0], a[1]); }
      g.globalCompositeOperation = "source-over"; g.strokeStyle = rgba(COLORS.structure, 0.38 * gain); g.lineWidth = 1.0 * o.lw; g.stroke(cord);
      g.globalCompositeOperation = "lighter";
      for (let k = 2; k < NB - 1; k++) {
        const x = bin[k] * gain; if (x < 0.03) continue;
        const a = T(...this.at(f, k / NB, side)), b = T(...this.at(f, (k + 1) / NB, side));
        g.strokeStyle = rgba(hot(x), 0.15 + 0.8 * x); g.lineWidth = (1.0 + 2.4 * x) * o.lw;
        g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.stroke();
      }
    }

    // what each connection carries now, in its transmitter's colour; the long axons
    // show it as pulses running along the cord
    const live = [];
    for (const e of o.edges) {
      const x = e.gap ? Math.abs(A[e.k]) * Math.max(Math.abs(act[e.a]), Math.abs(act[e.b])) : Math.abs(A[e.k] * act[e.a]);
      const L = fire(x * 10); if (L > 0.04) live.push([L, e]);
    }
    live.sort((p, q) => q[0] - p[0]);
    const LV = 6, paths = new Map();
    for (let n = 0, drawn = 0; n < live.length && drawn < o.lines; n++) {
      const [L, e] = live[n]; if (e.via === "cord" || (e.via === "ring" && !o.ringLines)) continue;
      drawn++;
      const key = e.family + "|" + Math.min(LV - 1, Math.floor(L * LV));
      let p = paths.get(key); if (!p) paths.set(key, (p = new Path2D()));
      this.route(p, f, T, e, pos);
    }
    g.globalCompositeOperation = "screen";
    for (const [key, p] of paths) {
      const [fam, lv] = key.split("|"), x = (Number(lv) + 1) / LV;
      g.strokeStyle = rgba(COLORS[fam], (fam === "gap" ? 0.55 : 1) * o.glow * (0.07 + 0.8 * x * x)); g.lineWidth = (0.5 + 1.5 * x) * o.lw;
      g.stroke(p);
    }
    for (const c of cells) if (c.dorsal) {       // commissures carry their motor neuron's activity
      const L = fire(act[c.i]); if (L < 0.04) continue;
      const p = new Path2D(); this.commissure(p, f, T, c);
      g.strokeStyle = rgba(hot(L), 0.15 + 0.7 * L); g.lineWidth = (0.6 + 1.3 * L) * o.lw; g.stroke(p);
    }
    // vesicles: one pulse per chemical synapse per tick, from the presynaptic cell to its target
    g.globalCompositeOperation = "lighter";
    for (let n = 0, drawn = 0; n < live.length && drawn < o.pulses; n++) {
      const [L, e] = live[n]; if (e.gap || L < 0.1) continue;
      const t = (view.phase + (e.k % 5) * 0.03) % 1;
      const q = e.via !== "direct" ? this.along(f, T, e, t) : [pos[e.a][0] + (pos[e.b][0] - pos[e.a][0]) * t, pos[e.a][1] + (pos[e.b][1] - pos[e.a][1]) * t];
      if (q[0] < -20 || q[1] < -20 || q[0] > o.W + 20 || q[1] > o.H + 20) continue;
      const sz = Math.max(3, (4 + 12 * L) * o.zs);
      g.globalAlpha = Math.min(1, 0.25 + 0.75 * L);
      g.drawImage(this.sprites[e.family], q[0] - sz / 2, q[1] - sz / 2, sz, sz);
      drawn++;
    }
    g.globalAlpha = 1;

    // the neurons, every one: a cold patch at rest, filled and blooming as it fires
    const order = cells.map((c) => [fire(act[c.i]), c]).sort((a, b) => a[0] - b[0]);
    for (const [L, c] of order) {
      const [x, y] = pos[c.i]; if (x < -30 || y < -30 || x > o.W + 30 || y > o.H + 30) continue;
      const z = o.zs * (c.s < 0.12 ? o.head : 1), r = Math.max(o.minPatch, o.patch * z);
      if (L > 0.08) {               // the bloom belongs to cells that really fire
        g.globalCompositeOperation = "lighter";
        const sz = Math.max(r * 3, (6 + 44 * L) * z);
        g.globalAlpha = Math.min(1, 1.15 * Math.pow(L, 1.4)); g.drawImage(this.sprites.hot, x - sz / 2, y - sz / 2, sz, sz); g.globalAlpha = 1;
      }
      g.globalCompositeOperation = "source-over";
      g.beginPath(); g.arc(x, y, r * (1 + 0.35 * L), 0, Math.PI * 2);
      g.fillStyle = "rgba(12,24,43,0.9)"; g.fill();
      if (L > 0.02) { g.fillStyle = rgba(hot(L), 0.25 + 0.75 * L); g.fill(); }
      const w = Math.min(1, L * 2.5), sc = hot(Math.min(1, L + 0.2)).map((v, i) => Math.round(COLORS.structure[i] + (v - COLORS.structure[i]) * w));
      g.strokeStyle = rgba(sc, 0.62 + 0.38 * w); g.lineWidth = Math.max(0.7, 0.9 * o.lw); g.stroke();
      if (r > 2.2 && L <= 0.02) { g.fillStyle = rgba(COLORS.structure, 0.6); g.beginPath(); g.arc(x, y, r * 0.3, 0, Math.PI * 2); g.fill(); }
    }
    // the lesson: changed connections flare, then a ring moves outward cell by cell
    const Ls = view.lesson;
    if (Ls && Ls.applied) {
      const age = (view.now - Ls.shownAt) / 1000, fade = Math.max(0, 1 - age / 3.2);
      if (fade > 0) {
        g.globalCompositeOperation = "source-over";     // drawn last and opaque, so the green stays green over the heat
        for (const [d, e] of Ls.top) {          // the connections that changed most, lit along their paths
          const x = d / Ls.max, reach = Math.min(1, Math.max(0, age * 2.5 - this.hops[e.a] * 0.35)); if (reach <= 0) continue;
          const p = new Path2D(); this.route(p, f, T, e, pos);
          g.strokeStyle = rgba(COLORS.learning, 0.9 * Math.sqrt(x) * fade * reach); g.lineWidth = (0.9 + 1.6 * x) * o.lw; g.stroke(p);
        }
        for (const c of cells) {                 // each cell's share of the correction, arriving hop by hop
          const cr = Ls.credit[c.i] / Ls.creditMax; if (cr < 0.05) continue;
          const t = age - this.hops[c.i] * 0.16; if (t < 0 || t > 1.4) continue;
          const [x, y] = pos[c.i], r = (3 + 26 * t) * Math.max(0.55, o.zs), a = cr * (1 - t / 1.4);
          g.strokeStyle = rgba(COLORS.learning, Math.min(1, 1.2 * a)); g.lineWidth = 1.6;
          g.beginPath(); g.arc(x, y, r, 0, Math.PI * 2); g.stroke();
          const sz = (10 + 30 * cr) * Math.max(0.55, o.zs);
          g.globalAlpha = Math.min(1, 0.7 * a); g.drawImage(this.sprites.learning, x - sz / 2, y - sz / 2, sz, sz); g.globalAlpha = 1;
        }
      }
    }

    g.globalCompositeOperation = "source-over";
  }

  grid(g, step, alpha) {
    const S = this.cam.scale, [x0, y0] = this.toWorld(0, 0), [x1, y1] = this.toWorld(this.W, this.H);
    g.strokeStyle = `rgba(120,165,215,${alpha})`; g.lineWidth = 1; g.beginPath();
    for (let x = Math.floor(x0 / step) * step; x <= x1; x += step) { const sx = (x - this.cam.x) * S + this.cx; g.moveTo(sx, 0); g.lineTo(sx, this.H); }
    for (let y = Math.floor(y0 / step) * step; y <= y1; y += step) { const sy = (y - this.cam.y) * S + this.cy; g.moveTo(0, sy); g.lineTo(this.W, sy); }
    g.stroke();
  }
  plateEdge(g, life) {
    const [x0, y0] = this.toScreen(0, 0), [x1, y1] = this.toScreen(life.w, life.h);
    g.save(); g.shadowColor = rgba(COLORS.structure, 0.7); g.shadowBlur = 12;
    g.strokeStyle = rgba(COLORS.structure, 0.55); g.lineWidth = 1.4;
    g.beginPath(); g.roundRect(x0, y0, x1 - x0, y1 - y0, 0.4 * this.cam.scale); g.stroke(); g.restore();
  }
  item(g, it, now) {
    if (it.amount <= 0) return;
    const [x, y] = this.toScreen(it.x, it.y), S = this.cam.scale;
    if (it.kind === "food") {
      const R = this.p.food_radius * S * (0.55 + 0.45 * Math.sqrt(it.amount));
      const gr = g.createRadialGradient(x, y, 0, x, y, R);
      gr.addColorStop(0, rgba(COLORS.food, 0.22)); gr.addColorStop(0.75, rgba(COLORS.food, 0.10)); gr.addColorStop(1, rgba(COLORS.food, 0));
      g.fillStyle = gr; g.beginPath(); g.arc(x, y, R, 0, Math.PI * 2); g.fill();
      if (!this.speckles.has(it)) {         // a lawn of bacteria, fixed per drop
        let s = (it.x * 997 + it.y * 131) | 0; const r = () => ((s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
        this.speckles.set(it, Array.from({ length: 70 }, () => [Math.sqrt(r()) * 0.92, r() * Math.PI * 2, r()]));
      }
      g.fillStyle = rgba(COLORS.food, 0.55);
      for (const [d, a, z] of this.speckles.get(it)) {
        g.beginPath(); g.arc(x + Math.cos(a) * d * R, y + Math.sin(a) * d * R, 0.6 + z * 1.1, 0, Math.PI * 2); g.fill();
      }
      g.strokeStyle = rgba(COLORS.food, 0.35); g.lineWidth = 1; g.beginPath(); g.arc(x, y, R, 0, Math.PI * 2); g.stroke();
    } else if (it.kind === "noxious") {
      const R = this.p.noxious_radius * S, pulse = 0.5 + 0.5 * Math.sin(now / 420);
      const gr = g.createRadialGradient(x, y, 0, x, y, R * 1.6);
      gr.addColorStop(0, rgba(COLORS.noxious, 0.16)); gr.addColorStop(1, rgba(COLORS.noxious, 0));
      g.fillStyle = gr; g.beginPath(); g.arc(x, y, R * 1.6, 0, Math.PI * 2); g.fill();
      g.beginPath();
      for (let k = 0; k < 12; k++) { const a = (k / 12) * Math.PI * 2, rr = R * (k % 2 ? 0.55 : 1); g[k ? "lineTo" : "moveTo"](x + Math.cos(a) * rr, y + Math.sin(a) * rr); }
      g.closePath(); g.fillStyle = rgba(COLORS.noxious, 0.12); g.fill();
      g.strokeStyle = rgba(COLORS.noxious, 0.8); g.lineWidth = 1.2; g.stroke();
      g.strokeStyle = rgba(COLORS.noxious, 0.25 * pulse); g.beginPath(); g.arc(x, y, R * (1.3 + 0.35 * pulse), 0, Math.PI * 2); g.stroke();
    }
  }
  body(g, f, T, view, S) {
    // sampled finely at the nose, where the profile rounds off
    const ss = [];
    for (let j = 0; j < 10; j++) ss.push(0.02 * (j / 10) ** 2);
    for (let k = 2; k < f.N; k++) ss.push(k / (f.N - 1));
    const N = ss.length, left = ss.map((s) => T(...this.at(f, s, 1))), right = ss.map((s) => T(...this.at(f, s, -1)));
    const outline = () => {
      g.beginPath(); g.moveTo(left[0][0], left[0][1]);
      for (let k = 1; k < N; k++) g.lineTo(left[k][0], left[k][1]);
      for (let k = N - 1; k >= 0; k--) g.lineTo(right[k][0], right[k][1]);
      g.closePath();
    };
    // translucent flesh
    g.globalCompositeOperation = "source-over";
    outline();
    const [hx, hy] = T(...this.at(f, 0, 0)), [tx, ty] = T(...this.at(f, 1, 0));
    const lg = g.createLinearGradient(hx, hy, tx, ty);
    lg.addColorStop(0, "rgba(58,98,146,0.34)"); lg.addColorStop(0.5, "rgba(44,80,124,0.27)"); lg.addColorStop(1, "rgba(40,72,112,0.2)");
    g.fillStyle = lg; g.fill();
    // cuticle rings
    g.strokeStyle = rgba(COLORS.structure, 0.08); g.lineWidth = 1;
    for (let k = 12; k < N - 3; k += 2) { g.beginPath(); g.moveTo(left[k][0], left[k][1]); g.lineTo(right[k][0], right[k][1]); g.stroke(); }
    // an inner contour, as the body wall
    g.strokeStyle = rgba(COLORS.structure, 0.22); g.lineWidth = 1;
    for (const side of [0.84, -0.84]) {
      g.beginPath(); ss.forEach((s, k) => { if (s < 0.012 || s > 0.97) return; const w = T(...this.at(f, s, side)); g.lineTo(w[0], w[1]); }); g.stroke();
    }
    // pharynx: a tube with two bulbs
    g.strokeStyle = rgba(COLORS.structure, 0.4); g.lineWidth = 1;
    const bulb = (s, w) => {
      const c = T(...this.at(f, s, 0)), e = T(...this.at(f, s, w));
      g.beginPath(); g.arc(c[0], c[1], Math.hypot(e[0] - c[0], e[1] - c[1]), 0, Math.PI * 2); g.stroke();
    };
    g.beginPath(); for (let j = 0; j <= 10; j++) { const w = T(...this.at(f, 0.004 + j * 0.0105, 0)); j ? g.lineTo(w[0], w[1]) : g.moveTo(w[0], w[1]); } g.stroke();
    bulb(0.03, 0.34); bulb(0.078, 0.42);
    // nerve ring: cold when the head is quiet, hot when it is busy
    let head = 0, n = 0;
    for (const c of this.cells) if (c.s < 0.1) { head = Math.max(head, fire(view.activity[c.i])); n++; }
    const ring = T(...this.at(f, 0.045, 0)), edge = T(...this.at(f, 0.045, 0.95));
    const rr = Math.hypot(edge[0] - ring[0], edge[1] - ring[1]), rc = head > 0.05 ? hot(head) : COLORS.structure;
    g.save(); g.globalCompositeOperation = "lighter";
    g.shadowColor = rgba(rc, 0.9); g.shadowBlur = 6 + 26 * head;
    g.strokeStyle = rgba(rc, 0.55 + 0.45 * head); g.lineWidth = 1.4 + 2.2 * head;
    g.beginPath(); g.ellipse(ring[0], ring[1], rr * 0.86, rr * 0.3, Math.atan2(edge[1] - ring[1], edge[0] - ring[0]), 0, Math.PI * 2); g.stroke();
    g.restore();
    // the cuticle: one crisp pale line
    g.save(); g.shadowColor = rgba(COLORS.structure, 0.6); g.shadowBlur = 8;
    outline(); g.strokeStyle = rgba([196, 222, 244], 0.92); g.lineWidth = 1.5; g.lineJoin = "round"; g.stroke();
    g.restore();
  }
  decor(g) {       // quiet instrument marks at the corners
    const m = 18, l = 16; g.strokeStyle = rgba(COLORS.structure, 0.4); g.lineWidth = 1;
    for (const [x, y, sx, sy] of [[m, m, 1, 1], [this.W - m, m, -1, 1], [m, this.H - m, 1, -1], [this.W - m, this.H - m, -1, -1]]) {
      g.beginPath(); g.moveTo(x, y + sy * l); g.lineTo(x, y); g.lineTo(x + sx * l, y); g.stroke();
    }
  }
}

// ---- the brain at scale: the head straightened, anterior to the left, dorsal up -------------
const GROUPS = [
  { cls: "AWA", text: "AWA · smell A", color: "odourA", input: 0, top: true },
  { cls: "AWC", text: "AWC · smell B", color: "odourB", input: 1, top: true },
  { cls: "ASH", text: "ASH · pain", color: "noxious", input: 3, top: true },
  { cls: "CEP", text: "CEP · food", color: "food", input: 2, top: true },
  { cls: "ADE", text: "ADE · food", color: "food", input: 2, top: true },
  { cls: "AVB", text: "AVB · forward", color: "forward", output: 0 },
  { cls: "AVA", text: "AVA · reverse", color: "reverse", output: 1 },
  { cls: "AVD", text: "AVD · reverse", color: "reverse", output: 1 },
  { cls: "AVE", text: "AVE · reverse", color: "reverse", output: 1 },
];

class BrainView {
  constructor(R, canvas, foot) {
    this.R = R; this.c = canvas; this.g = canvas.getContext("2d"); this.foot = foot;
    const N = 96, L = R.p.body_length;
    this.f = { pts: Array.from({ length: N }, (_, k) => [(k / (N - 1)) * L, 0]), nx: new Array(N).fill(0), ny: new Array(N).fill(1), N };
    this.s0 = -0.01; this.s1 = 0.138; this.ex = 1.2;
    this.edges = R.edges.filter((e) => Math.min(R.cells[e.a].s, R.cells[e.b].s) < this.s1);
    this.groups = GROUPS.map((gr) => ({ ...gr, cells: R.cells.filter((c) => c.name.startsWith(gr.cls) && /^[A-Z]{3}[DV]?[LR]?$/.test(c.name)).map((c) => c.i) }));
    this.hover = -1; this.pos = [];
    canvas.addEventListener("mousemove", (ev) => {
      const r = canvas.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top;
      let best = -1, bd = 12;
      this.pos.forEach((p, i) => { const d = Math.hypot(p[0] - x, p[1] - y); if (d < bd && R.cells[i].s < this.s1) { bd = d; best = i; } });
      this.hover = best;
    });
    canvas.addEventListener("mouseleave", () => { this.hover = -1; });
    this.resize();
  }
  resize() {
    const dpr = window.devicePixelRatio || 1, r = this.c.getBoundingClientRect();
    this.c.width = Math.round(r.width * dpr); this.c.height = Math.round(r.height * dpr);
    this.W = r.width; this.H = r.height; this.dpr = dpr;
    this.pad = 10;
    const L = this.R.p.body_length, R0 = this.R.radius(0.2);
    this.k = Math.min((this.W - 2 * this.pad) / ((this.s1 - this.s0) * L), (this.H / 2 - 30) / (R0 * this.ex));
  }
  T(x, y) { return [this.pad + (x - this.s0 * this.R.p.body_length) * this.k, this.H / 2 + 4 + y * this.k * this.ex]; }

  draw(life, view) {
    const g = this.g, R = this.R, W = this.W, H = this.H, T = (x, y) => this.T(x, y);
    if (!W) return;
    g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    g.globalCompositeOperation = "source-over"; g.clearRect(0, 0, W, H);
    R.body(g, this.f, T, view, this.k);
    R.nervous(g, this.f, T, life, view, { W, H, zs: 0.6, head: 1, patch: 4.2, minPatch: 2.2, lw: 0.9, lines: 650, pulses: 320, glow: 0.8, ringLines: true, edges: this.edges });
    this.pos = R.pos;

    // the cells that sense and the cells that command, named
    g.font = "500 10px Inter, sans-serif"; g.textBaseline = "middle";
    const u = life.input ?? [0, 0, 0, 0], y = life.readout ?? [0, 0];
    for (const top of [true, false]) {
      const row = this.groups.filter((gr) => !!gr.top === top).map((gr) => {
        const ps = gr.cells.map((i) => this.pos[i]), ax = ps.reduce((a, p) => a + p[0], 0) / ps.length, ay = ps.reduce((a, p) => a + p[1], 0) / ps.length;
        const on = gr.input !== undefined ? Math.min(1, u[gr.input]) : Math.max(0, Math.min(1, y[gr.output]));
        return { ...gr, ax, ay, on, w: g.measureText(gr.text).width };
      }).sort((a, b) => a.ax - b.ax);
      let end = 6;
      for (const l of row) { l.x = Math.max(end, Math.min(W - l.w - 6, l.ax - l.w / 2)); end = l.x + l.w + 12; }
      if (row.length) row[row.length - 1].x = Math.min(row[row.length - 1].x, W - row[row.length - 1].w - 6);
      for (let j = row.length - 2; j >= 0; j--) row[j].x = Math.min(row[j].x, row[j + 1].x - row[j].w - 12);
      const ly = top ? 11 : H - 10;
      for (const l of row) {
        const c = COLORS[l.color], mx = l.x + l.w / 2;
        g.strokeStyle = rgba(c, 0.16 + 0.5 * l.on); g.lineWidth = 0.7;
        g.beginPath(); g.moveTo(mx, ly + (top ? 7 : -7)); g.lineTo(l.ax, l.ay); g.stroke();
        g.fillStyle = rgba(c, 0.6 + 0.4 * l.on); g.fillText(l.text, l.x, ly);
        if (l.on > 0.02) {           // the input arriving, or the command being given
          g.globalCompositeOperation = "lighter";
          for (const i of l.cells) {
            const [px, py] = this.pos[i];
            g.strokeStyle = rgba(c, 0.85 * l.on); g.lineWidth = 1.4;
            g.beginPath(); g.arc(px, py, 6 + 3 * l.on, 0, Math.PI * 2); g.stroke();
          }
          g.globalCompositeOperation = "source-over";
        }
      }
    }
    // scale bar: 10 micrometres along the body
    const bar = 0.01 * this.k;
    g.strokeStyle = rgba(COLORS.structure, 0.6); g.lineWidth = 1;
    g.beginPath(); g.moveTo(W - 14 - bar, H - 26); g.lineTo(W - 14, H - 26); g.stroke();
    g.fillStyle = "rgba(169,188,211,0.7)"; g.textAlign = "right"; g.fillText("10 µm", W - 14, H - 36); g.textAlign = "start";

    // a neuron under the pointer
    if (this.hover >= 0) {
      const c = R.cells[this.hover], [px, py] = this.pos[this.hover], a = view.activity[c.i];
      g.strokeStyle = "rgba(235,248,255,0.9)"; g.lineWidth = 1.2; g.beginPath(); g.arc(px, py, 8, 0, Math.PI * 2); g.stroke();
      const tr = c.transmitter ? c.transmitter.replaceAll("_", " + ") : "no transmitter listed";
      this.say(`<b>${c.name}</b> · ${c.kind.split(";")[0].toLowerCase()} · ${tr} · activity ${Math.abs(a) < 1e-9 ? "0" : a.toExponential(1)}`);
    } else this.say("Heat is activity on a log scale, 10<sup>−4.5</sup> to 1. Point at a neuron to name it.");
  }
  say(html) { if (html !== this.said) { this.foot.innerHTML = html; this.said = html; } }
}
