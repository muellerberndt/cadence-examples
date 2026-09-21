// Drawing the worm: a translucent body with its 302 neurons at their measured
// positions, synaptic transmission in each transmitter's colour, and each
// lesson as a wave of local change spreading from the command neurons.
export const COLORS = {
  excitatory: [255, 176, 80],   // glutamate, acetylcholine
  inhibitory: [94, 200, 255],   // GABA
  dopamine: [255, 111, 216],
  modulatory: [184, 155, 255],  // serotonin, octopamine, tyramine, peptides
  none: [160, 190, 225],
  learning: [255, 241, 184],
  odourA: [46, 230, 196],
  odourB: [167, 139, 250],
  food: [217, 249, 157],
  noxious: [255, 90, 78],
  body: [150, 220, 255],
};
const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
const glow = (a) => Math.min(1, 1.15 * Math.pow(Math.min(1, Math.abs(a)), 0.55));
const AP_TIP = 0.137, AP_SPAN = 1.162;

function sprite(c) {
  const s = document.createElement("canvas"); s.width = s.height = 64;
  const g = s.getContext("2d"), r = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  r.addColorStop(0, "rgba(255,255,255,0.95)");
  r.addColorStop(0.1, rgba(c, 0.95));
  r.addColorStop(0.32, rgba(c, 0.32));
  r.addColorStop(1, rgba(c, 0));
  g.fillStyle = r; g.fillRect(0, 0, 64, 64); return s;
}

export class Renderer {
  constructor(canvas, connectome, spec, params) {
    this.c = canvas; this.g = canvas.getContext("2d"); this.p = params;
    this.cells = connectome.neurons.map((n, i) => ({
      i, name: n.name, role: n.role, family: n.family ?? "none",
      s: Math.min(0.995, Math.max(0.004, (AP_TIP - n.ap) / AP_SPAN)),
      dv: Math.max(-1, Math.min(1, n.dv / 0.05)) * 0.8,
    }));
    this.sprites = {}; for (const k of Object.keys(COLORS)) this.sprites[k] = sprite(COLORS[k]);
    // chemical synapses, keyed into the brain's A entries for their live weight
    const key = new Map(); spec.A.rows.forEach((r, k) => key.set(r * spec.H + spec.A.cols[k], k));
    this.synapses = connectome.chemical.map(([pre, post, count, sign, label]) => ({
      pre, post, k: key.get(post * spec.H + pre), family: this.cells[pre].family,
    })).filter((s) => s.k !== undefined && s.pre !== s.post);
    this.gaps = connectome.gap.map(([a, b]) => ({ a, b }));
    // hop distance from the command neurons: the order in which a lesson's
    // correction reaches each cell through local couplings
    const adj = Array.from({ length: spec.H }, () => []);
    for (const s of this.synapses) { adj[s.pre].push(s.post); adj[s.post].push(s.pre); }
    for (const g of this.gaps) { adj[g.a].push(g.b); adj[g.b].push(g.a); }
    const start = Object.values(params.outputs).flat().map((n) => spec.names.indexOf(n));
    this.hops = new Int32Array(spec.H).fill(99); const q = [...start]; start.forEach((i) => { this.hops[i] = 0; });
    while (q.length) { const i = q.shift(); for (const j of adj[i]) if (this.hops[j] > this.hops[i] + 1) { this.hops[j] = this.hops[i] + 1; q.push(j); } }
    this.cam = { x: 0, y: 0, scale: 0, ready: false };
    this.zoom = 1; this.speckles = new Map();
    this.resize();
  }

  resize() {
    const dpr = window.devicePixelRatio || 1, r = this.c.getBoundingClientRect();
    this.c.width = Math.round(r.width * dpr); this.c.height = Math.round(r.height * dpr);
    this.W = r.width; this.H = r.height; this.dpr = dpr;
  }
  baseScale() { return 0.5 * Math.min(this.W, 1.6 * this.H) / this.p.body_length; }
  get cy() { return this._lensOrigin ? this._lensOrigin[1] : this.H / 2 - Math.min(70, this.H * 0.07); }
  get lensR() { return Math.min(165, this.H * 0.2, this.W * 0.13); }
  get lensOn() { return this.W > 900 && this.zoom > 0.45; }
  get cx() { return this._lensOrigin ? this._lensOrigin[0] : this.W / 2 + (this.lensOn ? this.lensR * 0.95 : 0); }
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
      const tx = b[0] - a[0], ty = b[1] - a[1], n = Math.hypot(tx, ty) || 1;
      nx.push(-ty / n); ny.push(tx / n);
    }
    return { pts, nx, ny, N };
  }
  radius(s) {
    const R = 0.043;
    const head = s < 0.06 ? 0.45 + 0.55 * Math.sin((s / 0.06) * Math.PI / 2) : 1;
    const tail = s > 0.55 ? Math.max(0.06, Math.pow((1 - s) / 0.45, 0.85)) : 1;
    return R * head * tail;
  }
  at(f, s, off) {                 // body coordinate (s along, off across in radii) -> world
    const x = s * (f.N - 1), k = Math.min(f.N - 2, Math.floor(x)), t = x - k;
    const px = f.pts[k][0] * (1 - t) + f.pts[k + 1][0] * t, py = f.pts[k][1] * (1 - t) + f.pts[k + 1][1] * t;
    const nx = f.nx[k] * (1 - t) + f.nx[k + 1] * t, ny = f.ny[k] * (1 - t) + f.ny[k + 1] * t;
    const r = this.radius(s) * off;
    return [px + nx * r, py + ny * r];
  }

  // ---- one frame --------------------------------------------------------------------------
  draw(life, view) {
    const g = this.g;
    g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    const f = this.frame(life.body.trail);
    this.lensPass = false;
    this.scene(life, view, f, true);
    this.lens(life, view, f);
    g.globalCompositeOperation = "source-over";
    this.decor(g);
  }

  lens(life, view, f) {
    // a magnifier on the head: the nerve ring and head ganglia at four times scale
    const g = this.g, R = this.lensR;
    if (!this.lensOn) return;                           // narrow screen or zoomed out: no lens
    const head = this.at(f, 0.055, 0), hs = this.toScreen(...head);
    // anchored at the left, between the title and the controls; the camera keeps the worm to its right
    const cx = R + 44, cy = Math.max(R + 96, Math.min(this.H - R - 130, this.H * 0.47));
    this.lensAt = { cx, cy, R };
    // leader line from the head to the lens
    g.save(); g.globalCompositeOperation = "source-over";
    g.strokeStyle = "rgba(127,231,255,0.35)"; g.lineWidth = 1;
    const ang = Math.atan2(cy - hs[1], cx - hs[0]);
    g.beginPath(); g.arc(hs[0], hs[1], 0.07 * this.cam.scale, 0, Math.PI * 2); g.stroke();
    g.beginPath(); g.moveTo(hs[0] + Math.cos(ang) * 0.07 * this.cam.scale, hs[1] + Math.sin(ang) * 0.07 * this.cam.scale);
    g.lineTo(cx - Math.cos(ang) * R, cy - Math.sin(ang) * R); g.stroke();
    g.beginPath(); g.arc(cx, cy, R, 0, Math.PI * 2); g.clip();
    const saved = { ...this.cam }, W0 = this.W, cy0 = this.cy;
    this.cam = { ...saved, x: head[0], y: head[1], scale: saved.scale * 4 };
    this._lensOrigin = [cx, cy];
    this.lensPass = true;
    this.scene(life, view, f, false);
    this.lensPass = false;
    this.cam = saved; this._lensOrigin = null;
    g.restore();
    g.save(); g.strokeStyle = "rgba(160,235,255,0.6)"; g.lineWidth = 1.2; g.shadowColor = "rgba(127,231,255,0.8)"; g.shadowBlur = 14;
    g.beginPath(); g.arc(cx, cy, R, 0, Math.PI * 2); g.stroke(); g.restore();
    g.fillStyle = "rgba(169,188,211,0.8)"; g.font = "500 10px Inter, sans-serif"; g.textAlign = "center";
    g.fillText("NERVE RING  ×4", cx, cy + R + 16); g.textAlign = "start";
  }

  scene(life, view, f, main) {
    const g = this.g, { W, H } = this, p = this.p, now = view.now;
    if (main) {
    const mid = this.at(f, 0.3, 0);
    const target = this.baseScale() * this.zoom;
    if (!this.cam.ready) Object.assign(this.cam, { x: mid[0], y: mid[1], scale: target, ready: true });
    const k = 1 - Math.exp(-view.dt * 2.2);
    this.cam.x += (mid[0] - this.cam.x) * k; this.cam.y += (mid[1] - this.cam.y) * k;
    this.cam.scale += (target - this.cam.scale) * (1 - Math.exp(-view.dt * 5));
    }
    const S = this.cam.scale;

    // background: deep navy, a world-anchored blueprint grid, vignette
    g.globalCompositeOperation = "source-over";
    g.fillStyle = "#050b16"; g.fillRect(0, 0, W, H);
    this.grid(g, 0.1, 0.035); this.grid(g, 0.5, 0.07);
    this.plateEdge(g, life);
    const vig = g.createRadialGradient(W / 2, H / 2, Math.min(W, H) * 0.25, W / 2, H / 2, Math.max(W, H) * 0.75);
    vig.addColorStop(0, "rgba(10,30,60,0.0)"); vig.addColorStop(1, "rgba(0,0,0,0.55)");
    g.fillStyle = vig; g.fillRect(0, 0, W, H);

    // smells, then what gives them off
    g.globalCompositeOperation = "lighter";
    for (const it of life.items) if (it.odour && it.amount > 0) {
      const [x, y] = this.toScreen(it.x, it.y), R = 3 * p.odour_sigma * S;
      if (x < -R || y < -R || x > W + R || y > H + R) continue;
      const c = COLORS[it.odour === "A" ? "odourA" : "odourB"], gr = g.createRadialGradient(x, y, 0, x, y, R), a = 0.16 * Math.sqrt(it.amount);
      gr.addColorStop(0, rgba(c, a)); gr.addColorStop(0.35, rgba(c, a * 0.45)); gr.addColorStop(1, rgba(c, 0));
      g.fillStyle = gr; g.beginPath(); g.arc(x, y, R, 0, Math.PI * 2); g.fill();
    }
    g.globalCompositeOperation = "source-over";
    for (const it of life.items) this.item(g, it, now);

    // the body
    this.body(g, f, life, view);

    // synapses, gap junctions, neurons: additive light inside the body
    g.globalCompositeOperation = "lighter";
    const act = view.activity, pos = this.cells.map((c) => this.toScreen(...this.at(f, c.s, c.dv)));
    g.strokeStyle = "rgba(130,200,255,0.035)"; g.lineWidth = 0.7; g.beginPath();
    for (const syn of this.synapses) { if (Math.abs(this.cells[syn.pre].s - this.cells[syn.post].s) > 0.06) continue; const a = pos[syn.pre], b = pos[syn.post]; g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); }
    g.stroke();
    const A = life.brain.val, live = [];
    for (const syn of this.synapses) {
      const m = glow(A[syn.k] * act[syn.pre] * 4) * 0.35;
      if (m > 0.02) live.push([m, syn]);
    }
    live.sort((a, b) => b[0] - a[0]);
    const phase = view.phase;
    for (let n = 0; n < Math.min(live.length, 700); n++) {
      const [m, syn] = live[n], a = pos[syn.pre], b = pos[syn.post], c = COLORS[syn.family];
      const alpha = Math.min(0.55, m * 1.6);
      g.strokeStyle = rgba(c, alpha * 0.55); g.lineWidth = Math.max(0.6, S * 0.0012);
      const ca = this.cells[syn.pre], cb = this.cells[syn.post], long = Math.abs(ca.s - cb.s) > 0.12;
      g.beginPath();
      if (long) {           // long-range: follow the body, as the cords do
        for (let j = 0; j <= 8; j++) {
          const t = j / 8, w = this.toScreen(...this.at(f, ca.s + (cb.s - ca.s) * t, (ca.dv + (cb.dv - ca.dv) * t) * 0.6));
          j ? g.lineTo(w[0], w[1]) : g.moveTo(w[0], w[1]);
        }
      } else { g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); }
      g.stroke();
      if (n < 260) {        // a pulse travelling from the presynaptic cell to its target
        const t = (phase + (syn.k % 7) * 0.013) % 1;
        const q = long ? this.toScreen(...this.at(f, ca.s + (cb.s - ca.s) * t, (ca.dv + (cb.dv - ca.dv) * t) * 0.6))
                       : [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
        const sz = (5 + 14 * Math.min(1, m * 2.5)) * Math.min(1.6, S / 700);
        g.globalAlpha = Math.min(1, alpha * 1.6);
        g.drawImage(this.sprites[syn.family], q[0] - sz / 2, q[1] - sz / 2, sz, sz);
        g.globalAlpha = 1;
      }
    }
    for (const gp of this.gaps) {
      const m = Math.min(act[gp.a], act[gp.b]);
      if (m < 0.2 || Math.abs(this.cells[gp.a].s - this.cells[gp.b].s) > 0.06) continue;
      const a = pos[gp.a], b = pos[gp.b];
      g.strokeStyle = `rgba(220,245,255,${0.25 * m})`; g.lineWidth = 0.8;
      g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.stroke();
    }

    // the lesson: changed synapses flare, then a ring moves outward cell by cell
    const L = view.lesson;
    if (L && L.applied) {
      const age = (now - L.shownAt) / 1000, fade = Math.max(0, 1 - age / 2.6);
      if (fade > 0) {
        for (const [d, syn] of L.top) {
          const x = d / L.max, ca = this.cells[syn.pre], cb = this.cells[syn.post];
          g.strokeStyle = rgba(COLORS.learning, 0.6 * x * fade); g.lineWidth = 0.7 + 1.3 * x;
          g.beginPath();
          for (let j = 0; j <= 6; j++) {
            const t = j / 6, w = this.toScreen(...this.at(f, ca.s + (cb.s - ca.s) * t, (ca.dv + (cb.dv - ca.dv) * t) * 0.6));
            j ? g.lineTo(w[0], w[1]) : g.moveTo(w[0], w[1]);
          }
          g.stroke();
        }
        for (const c of this.cells) {
          const cr = L.credit[c.i] / L.creditMax; if (cr < 0.08) continue;
          const t = age - this.hops[c.i] * 0.16; if (t < 0 || t > 1.1) continue;
          const [x, y] = pos[c.i], r = (3 + 22 * t) * Math.min(1.5, S / 700);
          g.strokeStyle = rgba(COLORS.learning, 0.8 * cr * (1 - t / 1.1)); g.lineWidth = 1.2;
          g.beginPath(); g.arc(x, y, r, 0, Math.PI * 2); g.stroke();
        }
      }
    }

    // neurons last, brightest on top
    const zs = this.lensPass ? 1 : Math.min(1.7, Math.max(0.55, S / 700));
    for (const c of this.cells) {
      const [x, y] = pos[c.i]; if (x < -30 || y < -30 || x > W + 30 || y > H + 30) continue;
      const a = glow(act[c.i]);
      g.globalCompositeOperation = "source-over";
      g.strokeStyle = rgba(COLORS.body, 0.22 + 0.2 * a); g.lineWidth = 0.8;
      const rz = this.lensPass ? 2.2 : 1;
      g.beginPath(); g.arc(x, y, 1.9 * zs * rz, 0, Math.PI * 2); g.stroke();
      g.fillStyle = rgba(COLORS.body, 0.3); g.beginPath(); g.arc(x, y, 0.7 * zs * rz, 0, Math.PI * 2); g.fill();
      if (a > 0.05) {
        g.globalCompositeOperation = "lighter";
        const sz = (5 + 17 * a) * zs * (this.lensPass ? 2.2 : 1);
        g.globalAlpha = Math.min(1, 0.25 + 0.95 * a);
        g.drawImage(this.sprites[c.family], x - sz / 2, y - sz / 2, sz, sz);
        g.globalAlpha = 1;
      }
    }

    // events at the mouth
    g.globalCompositeOperation = "lighter";
    const head = this.toScreen(...this.at(f, 0.01, 0));
    for (const e of view.effects) {
      const age = (now - e.at) / 1000; if (age > e.dur) continue;
      const t = age / e.dur, c = COLORS[e.color];
      const R = (18 + 120 * t) * Math.min(1.6, S / 700);
      const gr = g.createRadialGradient(head[0], head[1], 0, head[0], head[1], R);
      gr.addColorStop(0, rgba(c, 0.55 * (1 - t))); gr.addColorStop(0.5, rgba(c, 0.18 * (1 - t))); gr.addColorStop(1, rgba(c, 0));
      g.fillStyle = gr; g.beginPath(); g.arc(head[0], head[1], R, 0, Math.PI * 2); g.fill();
      g.strokeStyle = rgba(c, 0.7 * (1 - t)); g.lineWidth = 1.5;
      g.beginPath(); g.arc(head[0], head[1], R * 0.8, 0, Math.PI * 2); g.stroke();
    }
    g.globalCompositeOperation = "source-over";
  }

  grid(g, step, alpha) {
    const S = this.cam.scale, [x0, y0] = this.toWorld(0, 0), [x1, y1] = this.toWorld(this.W, this.H);
    g.strokeStyle = `rgba(90,150,230,${alpha})`; g.lineWidth = 1; g.beginPath();
    for (let x = Math.floor(x0 / step) * step; x <= x1; x += step) { const sx = (x - this.cam.x) * S + this.cx; g.moveTo(sx, 0); g.lineTo(sx, this.H); }
    for (let y = Math.floor(y0 / step) * step; y <= y1; y += step) { const sy = (y - this.cam.y) * S + this.cy; g.moveTo(0, sy); g.lineTo(this.W, sy); }
    g.stroke();
  }
  plateEdge(g, life) {
    const [x0, y0] = this.toScreen(0, 0), [x1, y1] = this.toScreen(life.w, life.h);
    g.strokeStyle = "rgba(127,231,255,0.28)"; g.lineWidth = 1.2;
    g.beginPath(); g.roundRect(x0, y0, x1 - x0, y1 - y0, 0.4 * this.cam.scale); g.stroke();
  }
  item(g, it, now) {
    if (it.amount <= 0) return;
    const [x, y] = this.toScreen(it.x, it.y), S = this.cam.scale;
    if (it.kind === "food") {
      const R = this.p.food_radius * S * (0.55 + 0.45 * Math.sqrt(it.amount));
      const gr = g.createRadialGradient(x, y, 0, x, y, R);
      gr.addColorStop(0, rgba(COLORS.food, 0.30)); gr.addColorStop(0.75, rgba(COLORS.food, 0.14)); gr.addColorStop(1, rgba(COLORS.food, 0));
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
      gr.addColorStop(0, rgba(COLORS.noxious, 0.32)); gr.addColorStop(1, rgba(COLORS.noxious, 0));
      g.fillStyle = gr; g.beginPath(); g.arc(x, y, R * 1.6, 0, Math.PI * 2); g.fill();
      g.beginPath();
      for (let k = 0; k < 12; k++) { const a = (k / 12) * Math.PI * 2, rr = R * (k % 2 ? 0.55 : 1); g[k ? "lineTo" : "moveTo"](x + Math.cos(a) * rr, y + Math.sin(a) * rr); }
      g.closePath(); g.fillStyle = rgba(COLORS.noxious, 0.20); g.fill();
      g.strokeStyle = rgba(COLORS.noxious, 0.8); g.lineWidth = 1.2; g.stroke();
      g.strokeStyle = rgba(COLORS.noxious, 0.25 * pulse); g.beginPath(); g.arc(x, y, R * (1.3 + 0.35 * pulse), 0, Math.PI * 2); g.stroke();
    }
  }
  body(g, f, life, view) {
    const N = f.N, left = [], right = [];
    for (let k = 0; k < N; k++) {
      const s = k / (N - 1);
      left.push(this.toScreen(...this.at(f, s, 1))); right.push(this.toScreen(...this.at(f, s, -1)));
    }
    const outline = () => {
      g.beginPath(); g.moveTo(left[0][0], left[0][1]);
      for (let k = 1; k < N; k++) g.lineTo(left[k][0], left[k][1]);
      for (let k = N - 1; k >= 0; k--) g.lineTo(right[k][0], right[k][1]);
      g.closePath();
    };
    // translucent flesh
    outline();
    const [hx, hy] = this.toScreen(...this.at(f, 0, 0)), [tx, ty] = this.toScreen(...this.at(f, 1, 0));
    const lg = g.createLinearGradient(hx, hy, tx, ty);
    lg.addColorStop(0, "rgba(60,130,200,0.22)"); lg.addColorStop(0.5, "rgba(40,95,160,0.15)"); lg.addColorStop(1, "rgba(40,95,160,0.08)");
    g.fillStyle = lg; g.fill();
    // cuticle rings
    g.strokeStyle = "rgba(150,210,255,0.07)"; g.lineWidth = 1;
    for (let k = 3; k < N - 3; k += 2) { g.beginPath(); g.moveTo(left[k][0], left[k][1]); g.lineTo(right[k][0], right[k][1]); g.stroke(); }
    // pharynx: a tube with two bulbs
    g.strokeStyle = "rgba(170,225,255,0.20)"; g.lineWidth = 1;
    const bulb = (s, w) => {
      const c = this.toScreen(...this.at(f, s, 0)), e = this.toScreen(...this.at(f, s, w));
      g.beginPath(); g.arc(c[0], c[1], Math.hypot(e[0] - c[0], e[1] - c[1]), 0, Math.PI * 2); g.stroke();
    };
    g.beginPath(); for (let j = 0; j <= 10; j++) { const w = this.toScreen(...this.at(f, 0.004 + j * 0.0105, 0)); j ? g.lineTo(w[0], w[1]) : g.moveTo(w[0], w[1]); } g.stroke();
    bulb(0.03, 0.34); bulb(0.078, 0.42);
    // nerve ring: brightest when the head is busy
    let headActivity = 0, headCount = 0;
    for (const c of this.cells) if (c.s < 0.1) { headActivity += glow(view.activity[c.i]); headCount++; }
    headActivity /= Math.max(1, headCount);
    const ring = this.toScreen(...this.at(f, 0.045, 0)), ringEdge = this.toScreen(...this.at(f, 0.045, 0.95));
    const rr = Math.hypot(ringEdge[0] - ring[0], ringEdge[1] - ring[1]);
    g.save(); g.globalCompositeOperation = "lighter";
    g.shadowColor = "rgba(127,231,255,0.9)"; g.shadowBlur = 10 + 30 * headActivity;
    g.strokeStyle = `rgba(160,235,255,${0.35 + 0.6 * headActivity})`; g.lineWidth = 1.6 + 2 * headActivity;
    g.beginPath(); g.arc(ring[0], ring[1], rr * 0.82, 0, Math.PI * 2); g.stroke();
    g.restore();
    // nerve cords along the ventral and dorsal midlines
    g.strokeStyle = "rgba(150,215,255,0.13)"; g.lineWidth = 1;
    for (const side of [0.72, -0.72]) {
      g.beginPath(); for (let k = 4; k < N - 2; k++) { const w = this.toScreen(...this.at(f, k / (N - 1), side)); k > 4 ? g.lineTo(w[0], w[1]) : g.moveTo(w[0], w[1]); } g.stroke();
    }
    // glowing cuticle edge
    g.save(); g.globalCompositeOperation = "lighter";
    g.shadowColor = "rgba(127,231,255,0.8)"; g.shadowBlur = 16;
    outline(); g.strokeStyle = "rgba(150,225,255,0.55)"; g.lineWidth = 1.4; g.stroke();
    g.restore();
  }
  decor(g) {       // quiet instrument marks at the corners
    const m = 18, l = 16; g.strokeStyle = "rgba(127,231,255,0.35)"; g.lineWidth = 1;
    for (const [x, y, sx, sy] of [[m, m, 1, 1], [this.W - m, m, -1, 1], [m, this.H - m, 1, -1], [this.W - m, this.H - m, -1, -1]]) {
      g.beginPath(); g.moveTo(x, y + sy * l); g.lineTo(x, y); g.lineTo(x + sx * l, y); g.stroke();
    }
  }
}
