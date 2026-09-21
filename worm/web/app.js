import { Life } from "./life.js";
import { Renderer, COLORS, lum, hot } from "./render.js";

const load = (f) => fetch(`data/${f}`).then((r) => r.json());
const [spec, connectome, params] = await Promise.all([load("brain.json"), load("connectome.json"), load("params.json")]);
const seed = Number(new URLSearchParams(location.search).get("seed") ?? 1);
const life = new Life(spec, params, seed);
{ // born within smelling distance of a first drop of food
  const h = life.body.heading, x = life.body.x + 1.5 * Math.cos(h + 0.45), y = life.body.y + 1.5 * Math.sin(h + 0.45);
  life.place("food", Math.min(life.w - 0.6, Math.max(0.6, x)), Math.min(life.h - 0.6, Math.max(0.6, y)), "A", true);
}
life.place("food", ...life.freeSpot(), "A", true);
life.place("noxious", ...life.freeSpot(), "B", true);

const $ = (id) => document.getElementById(id);
const canvas = $("world");
const R = new Renderer(canvas, connectome, spec, params, { canvas: document.querySelector("#brain canvas"), foot: $("brainfoot") });
addEventListener("resize", () => { R.resize(); tl.resize(); mm.resize(); });
const state = { speed: 3, paused: false, tool: null, smell: "A", effects: [], lesson: null, inspect: null };
const stepsPerTick = Math.round(params.tick / params.physics_step);
const order = R.cells.slice().sort((a, b) => a.s - b.s).map((c) => c.i), ROWS = 34;     // the strip's rows: cells from head to tail
let acc = 0, steps = 0, last = performance.now();
// The full life: every neuron's activity at every tick, and the strip's rows ready to draw. Past
// CAP entries the record keeps every second tick, then every fourth, so memory stays bounded.
const history = { activity: [], rows: [], events: [], stride: 1 }, CAP = 14400;

// ---- controls -----------------------------------------------------------------------------
document.querySelectorAll("[data-tool]").forEach((b) => b.onclick = () => {
  state.tool = state.tool === b.dataset.tool ? null : b.dataset.tool;
  document.querySelectorAll("[data-tool]").forEach((x) => x.classList.toggle("on", x.dataset.tool === state.tool));
  canvas.classList.toggle("placing", !!state.tool);
});
document.querySelectorAll("[data-smell]").forEach((b) => b.onclick = () => {
  state.smell = b.dataset.smell;
  document.querySelectorAll("[data-smell]").forEach((x) => x.classList.toggle("on", x === b));
});
document.querySelectorAll("[data-speed]").forEach((b) => b.onclick = () => setSpeed(Number(b.dataset.speed)));
function setSpeed(v) { state.speed = v; document.querySelectorAll("[data-speed]").forEach((x) => x.classList.toggle("on", Number(x.dataset.speed) === v)); }
const pause = (v = !state.paused) => {
  state.paused = v; $("pause").textContent = v ? "▶" : "❚❚";
  if (!v && state.inspect !== null) { state.inspect = null; $("inspect").style.display = "none"; }     // running again means back to the present
};
$("pause").onclick = () => pause();
$("treat").onclick = () => { life.treat(); effect("food", 1.6); };
$("poke").onclick = () => { life.poke(); effect("noxious", 1.2); };
$("live").onclick = () => pause(false);
addEventListener("keydown", (e) => {
  if (e.code === "Space") { e.preventDefault(); pause(); }
  if (e.key === "1") setSpeed(1); if (e.key === "2") setSpeed(3); if (e.key === "3") setSpeed(10);
});
canvas.addEventListener("click", (e) => {
  const [x, y] = R.toWorld(e.clientX, e.clientY);
  if (e.altKey || !state.tool) {           // alt-click (or no tool): remove the nearest item
    const near = life.items.map((it) => [Math.hypot(it.x - x, it.y - y), it]).sort((a, b) => a[0] - b[0])[0];
    if (near && near[0] < 0.6 && e.altKey) life.remove(near[1]);
    return;
  }
  if (x < 0.2 || y < 0.2 || x > life.w - 0.2 || y > life.h - 0.2) return;
  life.place(state.tool, x, y, state.smell);
  hideHint();
});
canvas.addEventListener("contextmenu", (e) => {
  e.preventDefault(); const [x, y] = R.toWorld(e.clientX, e.clientY);
  const near = life.items.map((it) => [Math.hypot(it.x - x, it.y - y), it]).sort((a, b) => a[0] - b[0])[0];
  if (near && near[0] < 0.6) life.remove(near[1]);
});
canvas.addEventListener("wheel", (e) => { e.preventDefault(); R.zoom = Math.min(2.4, Math.max(0.12, R.zoom * Math.exp(-e.deltaY * 0.0012))); }, { passive: false });
let hintTimer = setTimeout(hideHint, 9000);
function hideHint() { $("hint").style.opacity = 0; clearTimeout(hintTimer); }

function effect(color, dur) { state.effects.push({ color, dur, at: performance.now() }); }
function toast(html) { const t = $("toast"); t.innerHTML = html; t.style.opacity = 1; clearTimeout(toast.h); toast.h = setTimeout(() => (t.style.opacity = 0), 4200); }

// ---- the life ---------------------------------------------------------------------------------
function record(n) {
  if (n % history.stride) return;
  const a = Float32Array.from(life.activity), rows = new Float32Array(ROWS);
  for (let r = 0; r < ROWS; r++) {           // mean log-activity of the row's cells, head at the top
    const i0 = Math.floor(r * order.length / ROWS), i1 = Math.floor((r + 1) * order.length / ROWS);
    let m = 0; for (let i = i0; i < i1; i++) m += lum(a[order[i]]); rows[r] = m / (i1 - i0);
  }
  history.activity.push(a); history.rows.push(rows);
  if (history.activity.length >= 2 * CAP) {
    history.activity = history.activity.filter((_, k) => k % 2 === 0); history.rows = history.rows.filter((_, k) => k % 2 === 0);
    history.stride *= 2; if (state.inspect !== null) state.inspect = Math.floor(state.inspect / 2);
  }
}
function tick(res) {
  record(res.tick);
  for (const kind of res.arrived) {
    effect(kind === "food" ? "food" : "noxious", kind === "food" ? 1.8 : 1.3);
    history.events.push({ tick: res.tick, event: kind });
  }
  if (res.lesson) {
    const L = res.lesson;
    history.events.push({ tick: res.tick, event: "lesson" });
    if (L.applied) {
      const top = R.edges.map((s) => [Math.abs(L.applied[s.k]), s]).sort((a, b) => b[0] - a[0]).slice(0, 160);
      state.lesson = { ...L, top, max: top[0][0] || 1, creditMax: Math.max(...L.credit) || 1, shownAt: performance.now() };
      const what = L.kinds.includes("pain") ? "pain" : "food";
      toast(`<b>Learned from ${what}.</b> The ${L.length} ticks before it were relived with the command neurons nudged toward ${what === "pain" ? "reversing" : "moving on"}; every connection changed by its own two ends.`);
    }
  }
  for (const e of life.events.splice(0)) if (e.event === "treat" || e.event === "poke") history.events.push({ tick: e.tick, event: e.event });
}

function frame(now) {
  const dt = Math.min(0.1, (now - last) / 1000); last = now;
  if (!state.paused && state.inspect === null) {
    acc += dt * state.speed;
    let guard = 0;
    while (acc >= params.physics_step && guard++ < 600) {
      life.step(); acc -= params.physics_step; steps++;
      if (steps % stepsPerTick === 0) tick(life.think());
    }
  }
  const phase = Math.min(1, ((steps % stepsPerTick) + acc / params.physics_step) / stepsPerTick);
  let activity;
  if (state.inspect !== null) activity = history.activity[state.inspect];
  else { activity = new Float64Array(life.activity.length); for (let i = 0; i < activity.length; i++) activity[i] = life.previousActivity[i] + (life.activity[i] - life.previousActivity[i]) * phase; }
  state.effects = state.effects.filter((e) => now - e.at < e.dur * 1000);
  R.draw(life, { now, dt, phase, activity, lesson: state.lesson, effects: state.effects });
  requestAnimationFrame(frame);
}

// ---- the panel, the minimap and the life strip, a few times a second -----------------------------
function panel() {
  const set = (id, v) => { const el = $(id), x = Math.max(-1, Math.min(1, v)); el.style.left = `${50 + Math.min(0, x) * 50}%`; el.style.width = `${Math.abs(x) * 50}%`; el.style.background = x >= 0 ? "#9bd7ff" : "#ff7850"; };
  const a = life.probe("odour_A"), b = life.probe("odour_B");
  set("learnA", (a[0] - a[1]) * 1.4); set("learnB", (b[0] - b[1]) * 1.4);
  $("fwd").style.width = `${Math.max(0, Math.min(1, life.readout[0])) * 100}%`;
  $("rev").style.width = `${Math.max(0, Math.min(1, life.readout[1])) * 100}%`;
  const m = Math.floor(life.t / 60), s = String(Math.floor(life.t % 60)).padStart(2, "0");
  $("stats").textContent = `age ${m}:${s} · ${life.lessons} lessons · ${life.brain.growth().toFixed(2)} recurrent gain`;
}
setInterval(panel, 250);

const mm = { c: document.querySelector("#minimap canvas"), resize() { const d = devicePixelRatio || 1, r = this.c.getBoundingClientRect(); this.c.width = r.width * d; this.c.height = r.height * d; this.d = d; this.W = r.width; this.H = r.height; } };
mm.resize();
function minimap() {
  if (!mm.W) return;               // hidden on narrow screens
  const g = mm.c.getContext("2d"); g.setTransform(mm.d, 0, 0, mm.d, 0, 0); g.clearRect(0, 0, mm.W, mm.H);
  const pad = 10, s = Math.min((mm.W - 2 * pad) / life.w, (mm.H - 2 * pad) / life.h), ox = (mm.W - life.w * s) / 2, oy = (mm.H - life.h * s) / 2;
  g.strokeStyle = "rgba(168,204,234,.4)"; g.strokeRect(ox, oy, life.w * s, life.h * s);
  for (const it of life.items) {
    if (it.odour) { const c = COLORS[it.odour === "A" ? "odourA" : "odourB"]; g.fillStyle = `rgba(${c},.18)`; g.beginPath(); g.arc(ox + it.x * s, oy + it.y * s, params.odour_sigma * 1.6 * s, 0, 7); g.fill(); }
    const c = COLORS[it.kind === "food" ? "food" : "noxious"]; g.fillStyle = `rgba(${c},.9)`; g.beginPath(); g.arc(ox + it.x * s, oy + it.y * s, 2.4, 0, 7); g.fill();
  }
  g.strokeStyle = "rgba(255,150,90,.95)"; g.lineWidth = 1.6; g.beginPath();
  life.body.trail.forEach(([x, y], k) => k ? g.lineTo(ox + x * s, oy + y * s) : g.moveTo(ox + x * s, oy + y * s)); g.stroke();
  const [cx0, cy0] = R.toWorld(0, 0), [cx1, cy1] = R.toWorld(R.W, R.H);
  g.strokeStyle = "rgba(168,204,234,.4)"; g.lineWidth = 1; g.strokeRect(ox + cx0 * s, oy + cy0 * s, (cx1 - cx0) * s, (cy1 - cy0) * s);
}
setInterval(minimap, 120);

// the full life: every neuron's activity (head at the top, tail at the bottom), every outcome and lesson
const tl = { c: document.querySelector("#timeline canvas"), resize() { const d = devicePixelRatio || 1, r = this.c.getBoundingClientRect(); this.c.width = r.width * d; this.c.height = r.height * d; this.d = d; this.W = r.width; this.H = r.height; } };
tl.resize();
function lifestrip() {
  const g = tl.c.getContext("2d"), n = history.rows.length; g.setTransform(tl.d, 0, 0, tl.d, 0, 0); g.clearRect(0, 0, tl.W, tl.H);
  const x0 = 14, w = tl.W - 28, top = 14, h = tl.H - 20, cols = Math.min(n, Math.floor(w));
  if (!n) return;
  const per = n / Math.max(1, cols), rowH = h / ROWS, colW = w / Math.max(cols, 1);
  for (let c = 0; c < cols; c++) {
    const a0 = Math.floor(c * per), a1 = Math.max(a0 + 1, Math.floor((c + 1) * per)), jump = Math.max(1, Math.floor((a1 - a0) / 4));
    for (let r = 0; r < ROWS; r++) {
      let v = 0; for (let k = a0; k < a1; k += jump) v = Math.max(v, history.rows[k][r]);
      v = (v - 0.25) / 0.75; if (v < 0.03) continue;          // the same heat as the worm's
      g.fillStyle = `rgba(${hot(Math.pow(v, 1.7))},${Math.min(1, 0.3 + v)})`;
      g.fillRect(x0 + c * colW, top + r * rowH, Math.ceil(colW), Math.ceil(rowH));
    }
  }
  const mark = { food: COLORS.food, pain: COLORS.noxious, treat: [255, 224, 138], poke: [255, 138, 90], lesson: COLORS.learning }, span = n * history.stride;
  for (const e of history.events) {
    const x = x0 + (e.tick / span) * w, c = mark[e.event];
    g.fillStyle = `rgb(${c})`;
    if (e.event === "lesson") { g.globalAlpha = 0.55; g.fillRect(x - 0.5, top - 2, 1, h + 2); g.globalAlpha = 1; }
    else { g.beginPath(); g.arc(x, 7, 3, 0, 7); g.fill(); }
  }
  if (state.inspect !== null) { const x = x0 + (state.inspect / n) * w; g.strokeStyle = "rgba(222,240,255,.95)"; g.beginPath(); g.moveTo(x, 2); g.lineTo(x, tl.H - 2); g.stroke(); }
}
setInterval(lifestrip, 400);
tl.c.addEventListener("click", (e) => {
  const r = tl.c.getBoundingClientRect(), f = (e.clientX - r.left - 14) / (r.width - 28), n = history.activity.length;
  if (!n) return;
  state.inspect = Math.max(0, Math.min(n - 1, Math.round(f * (n - 1))));
  const t = state.inspect * history.stride * params.tick; $("inspectLabel").textContent = `The brain at ${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")} of its life`;
  $("inspect").style.display = "flex"; pause(true);
});

window.__worm = { life, R, state, history, get steps() { return steps; } };
const warm = Number(new URLSearchParams(location.search).get("warm") ?? 0);
for (let k = 0; k < warm / params.tick; k++) { for (let j = 0; j < stepsPerTick; j++) { life.step(); steps++; } tick(life.think()); }
if (warm) hideHint();
requestAnimationFrame(frame);
