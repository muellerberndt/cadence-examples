import { Brain } from "./learner.js";
import { Dish, ODOURS } from "./dish.js";
import { PAGE } from "./net.js";

const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const COLORS = { food: css("--food"), diacetyl: css("--diacetyl"), butanone: css("--butanone"), warn: css("--warn"),
  ink: css("--ink"), muted: css("--muted"), line: css("--line"), faint: css("--faint"), panel: css("--panel") };
const index = Object.fromEntries(PAGE.names.map((n, i) => [n, i]));

const state = {
  running: true, speed: 8, learning: true, decisions: 0, searches: 0,
  outcomes: [], curve: [], swaps: [], initialScale: null,
};
let brain, dish, obs, frame = 0;

function makeBrain(params) {
  const net = { ...PAGE.net, ...(params ?? {}) };
  const b = new Brain(net);
  state.initialScale = Float64Array.from(PAGE.net.sign);
  applyLesion(b);
  return b;
}

function drive(o) {
  const d = new Float64Array(PAGE.net.n);
  o.forEach((x, c) => { for (const owner of PAGE.sensors[c]) d[owner] = PAGE.sense * x; });
  return d;
}

function applyLesion(b) {
  b.mask.fill(1);
  const value = $("lesion").value;
  if (value) for (const name of value.split(",")) b.mask[index[name]] = 0;
  b.free = null;
  b.pending = null;
}

function reset(params) {
  brain = makeBrain(params);
  dish = new Dish({ seed: 7, meaning: dish?.meaning ?? 0 });
  obs = dish.observe();
  Object.assign(state, { decisions: 0, searches: 0, outcomes: [], curve: [], swaps: [] });
  updateMeaning();
}

function decide() {
  const d = drive(obs);
  const learn = state.learning;
  if (!learn || !brain.free) brain.free = brain.settle(d, brain.free, brain.config.free_steps);
  const { choice } = brain.act(d, Math.random(), { learn });
  const r = dish.step(Math.floor(choice / 2));
  if (r.finished) {
    state.searches += 1;
    state.outcomes.push(r.outcome);
    if (state.outcomes.length > 400) state.outcomes.shift();
    const last = state.outcomes.slice(-20);
    state.curve.push({ food: share(last, "food"), empty: share(last, "empty") });
    if (state.curve.length > 600) { state.curve.shift(); state.swaps = state.swaps.map((s) => s - 1).filter((s) => s >= 0); }
    dish.reset();
  }
  obs = dish.observe();
  if (learn) brain.learn(r.reward + PAGE.shaping * r.approach, r.finished, drive(obs));
  else if (r.finished) brain.free = null;
  state.decisions += 1;
}

const share = (list, key) => (list.length ? list.filter((x) => x === key).length / list.length : 0);

// -- dish

const dishCanvas = $("dish"), dctx = dishCanvas.getContext("2d");
function drawDish() {
  const W = dishCanvas.width, R = W / 2 - 6, cx = W / 2, cy = W / 2;
  const toX = (x) => cx + x * R, toY = (y) => cy - y * R;
  dctx.clearRect(0, 0, W, W);
  dctx.save();
  dctx.beginPath(); dctx.arc(cx, cy, R, 0, Math.PI * 2); dctx.clip();
  dctx.globalCompositeOperation = "lighter";
  dish.spots.forEach(([sx, sy], k) => {
    const g = dctx.createRadialGradient(toX(sx), toY(sy), 0, toX(sx), toY(sy), R * dish.c.sigma * 1.9);
    const c = k === 0 ? COLORS.diacetyl : COLORS.butanone;
    g.addColorStop(0, hexA(c, 0.42)); g.addColorStop(0.45, hexA(c, 0.14)); g.addColorStop(1, hexA(c, 0));
    dctx.fillStyle = g; dctx.fillRect(0, 0, W, W);
  });
  dctx.globalCompositeOperation = "source-over";
  // trail
  const trail = dish.trail;
  dctx.strokeStyle = hexA(COLORS.ink, 0.16); dctx.lineWidth = 2; dctx.beginPath();
  trail.forEach(([x, y], i) => (i ? dctx.lineTo(toX(x), toY(y)) : dctx.moveTo(toX(x), toY(y))));
  dctx.stroke();
  // spots
  dish.spots.forEach(([sx, sy], k) => {
    const isFood = k === dish.food;
    dctx.lineWidth = 3;
    dctx.setLineDash(isFood ? [] : [6, 6]);
    dctx.strokeStyle = isFood ? COLORS.food : hexA(COLORS.muted, 0.8);
    dctx.beginPath(); dctx.arc(toX(sx), toY(sy), dish.c.contact * R, 0, Math.PI * 2); dctx.stroke();
    dctx.setLineDash([]);
    if (isFood) {
      dctx.fillStyle = hexA(COLORS.food, 0.18);
      dctx.beginPath(); dctx.arc(toX(sx), toY(sy), dish.c.contact * R, 0, Math.PI * 2); dctx.fill();
    }
    dctx.font = "500 22px 'IBM Plex Mono', ui-monospace, monospace";
    dctx.fillStyle = k === 0 ? COLORS.diacetyl : COLORS.butanone;
    dctx.textAlign = "center";
    dctx.fillText(`${ODOURS[k]}${isFood ? " · food" : ""}`, toX(sx), toY(sy) - dish.c.contact * R - 12);
  });
  // body: the last few positions, thick at the head
  const seg = trail.slice(-9);
  for (let i = 1; i < seg.length; i++) {
    dctx.strokeStyle = COLORS.ink; dctx.lineCap = "round";
    dctx.lineWidth = 4 + (i / seg.length) * 9;
    dctx.beginPath(); dctx.moveTo(toX(seg[i - 1][0]), toY(seg[i - 1][1])); dctx.lineTo(toX(seg[i][0]), toY(seg[i][1])); dctx.stroke();
  }
  const [hx, hy] = dish.pos;
  dctx.fillStyle = COLORS.ink;
  dctx.beginPath(); dctx.arc(toX(hx + Math.cos(dish.heading) * 0.02), toY(hy + Math.sin(dish.heading) * 0.02), 8, 0, Math.PI * 2); dctx.fill();
  dctx.restore();
  dctx.strokeStyle = COLORS.line; dctx.lineWidth = 3;
  dctx.beginPath(); dctx.arc(cx, cy, R, 0, Math.PI * 2); dctx.stroke();
}

function hexA(hex, a) {
  const h = hex.replace("#", "");
  const n = parseInt(h.length === 3 ? h.split("").map((c) => c + c).join("") : h, 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
}

// -- probe: the policy for each sensory pattern, settled from rest with the current seams

const PROBES = [
  { odour: 0, side: "left", obs: [1, 0, 0, 0, 0, 0] },
  { odour: 0, side: "right", obs: [0, 1, 0, 0, 0, 0] },
  { odour: 1, side: "left", obs: [0, 0, 1, 0, 0, 0] },
  { odour: 1, side: "right", obs: [0, 0, 0, 1, 0, 0] },
];
const probeRoot = $("probe");
probeRoot.innerHTML = PROBES.map((p, i) => `
  <div class="probe-row">
    <span class="who" style="color:${p.odour === 0 ? "var(--diacetyl)" : "var(--butanone)"}">${ODOURS[p.odour]} <span class="side-tag">on ${p.side}</span></span>
    <span class="bar" id="bar-${i}"><span class="l"></span><span class="r"></span><span class="f"></span></span>
    <span class="verdict" id="verdict-${i}">–</span>
  </div>`).join("");

function drawProbe() {
  PROBES.forEach((p, i) => {
    const st = brain.settle(drive(p.obs), null, brain.config.free_steps);
    const q = brain.probabilities(st);
    const act = [q[0] + q[1], q[2] + q[3], q[4] + q[5]];
    const spans = $(`bar-${i}`).children;
    act.forEach((x, k) => { spans[k].style.width = `${(x * 100).toFixed(1)}%`; });
    const toward = p.side === "left" ? act[0] : act[1];
    const away = p.side === "left" ? act[1] : act[0];
    const v = $(`verdict-${i}`);
    const margin = toward - away;
    v.className = "verdict" + (margin > 0.15 ? " toward" : margin < -0.15 ? " away" : "");
    v.textContent = margin > 0.15 ? "turns toward" : margin < -0.15 ? "turns away" : "no preference";
  });
}

// -- curve

const curveCanvas = $("curve"), cctx = curveCanvas.getContext("2d");
function drawCurve() {
  const W = curveCanvas.width, H = curveCanvas.height, pad = { l: 56, r: 12, t: 12, b: 30 };
  cctx.clearRect(0, 0, W, H);
  cctx.font = "22px 'IBM Plex Mono', ui-monospace, monospace";
  cctx.fillStyle = COLORS.muted; cctx.textAlign = "right"; cctx.textBaseline = "middle";
  [0, 0.5, 1].forEach((y) => {
    const py = pad.t + (1 - y) * (H - pad.t - pad.b);
    cctx.strokeStyle = hexA(COLORS.line, y === 0 ? 1 : 0.6); cctx.lineWidth = 2;
    cctx.beginPath(); cctx.moveTo(pad.l, py); cctx.lineTo(W - pad.r, py); cctx.stroke();
    cctx.fillText(`${y * 100}%`, pad.l - 10, py);
  });
  const chance = pad.t + (1 - 0.23) * (H - pad.t - pad.b);
  cctx.setLineDash([4, 6]); cctx.strokeStyle = hexA(COLORS.muted, 0.5);
  cctx.beginPath(); cctx.moveTo(pad.l, chance); cctx.lineTo(W - pad.r, chance); cctx.stroke(); cctx.setLineDash([]);
  cctx.textAlign = "left"; cctx.textBaseline = "bottom"; cctx.fillStyle = COLORS.faint;
  cctx.fillText("random walk", pad.l + 8, chance - 4);
  const pts = state.curve, n = Math.max(pts.length, 60);
  const x = (i) => pad.l + (i / (n - 1)) * (W - pad.l - pad.r);
  const y = (v) => pad.t + (1 - v) * (H - pad.t - pad.b);
  for (const s of state.swaps) {
    cctx.strokeStyle = hexA(COLORS.butanone, 0.7); cctx.lineWidth = 2;
    cctx.beginPath(); cctx.moveTo(x(s), pad.t); cctx.lineTo(x(s), H - pad.b); cctx.stroke();
    cctx.fillStyle = COLORS.butanone; cctx.textAlign = "left"; cctx.textBaseline = "top";
    cctx.fillText("swap", x(s) + 6, pad.t);
  }
  for (const [key, color] of [["empty", COLORS.warn], ["food", COLORS.food]]) {
    if (pts.length < 2) break;
    cctx.strokeStyle = color; cctx.lineWidth = 4; cctx.lineJoin = "round";
    cctx.beginPath(); pts.forEach((p, i) => (i ? cctx.lineTo(x(i), y(p[key])) : cctx.moveTo(x(i), y(p[key])))); cctx.stroke();
    const last = pts[pts.length - 1];
    cctx.fillStyle = color; cctx.beginPath(); cctx.arc(x(pts.length - 1), y(last[key]), 6, 0, Math.PI * 2); cctx.fill();
  }
  cctx.fillStyle = COLORS.muted; cctx.textAlign = "right"; cctx.textBaseline = "top";
  cctx.fillText(`${state.searches} searches`, W - pad.r, H - pad.b + 6);
}

// -- circuit

const circuitCanvas = $("circuit"), kctx = circuitCanvas.getContext("2d");
const layout = {};
PAGE.columns.forEach((col, c) => col.forEach((name, r) => { layout[name] = { c, r, rows: col.length }; }));
const shown = [];
for (let e = 0; e < PAGE.net.pre.length; e++) {
  const a = PAGE.names[PAGE.net.pre[e]], b = PAGE.names[PAGE.net.post[e]];
  if (layout[a] && layout[b] && layout[a].c < layout[b].c) shown.push(e);
}
function drawCircuit() {
  const W = circuitCanvas.width, H = circuitCanvas.height, padX = 90, padY = 40;
  const pos = (name) => {
    const { c, r, rows } = layout[name];
    return [padX + (c / (PAGE.columns.length - 1)) * (W - 2 * padX), padY + ((r + 0.5) / rows) * (H - 2 * padY)];
  };
  kctx.clearRect(0, 0, W, H);
  for (const e of shown) {
    const delta = brain.scale[e] - state.initialScale[e];
    const mag = Math.min(1, Math.abs(delta) / 1.5);
    const [x1, y1] = pos(PAGE.names[PAGE.net.pre[e]]), [x2, y2] = pos(PAGE.names[PAGE.net.post[e]]);
    kctx.strokeStyle = mag < 0.04 ? hexA(COLORS.line, 0.9) : hexA(delta > 0 ? COLORS.food : COLORS.diacetyl, 0.25 + 0.75 * mag);
    kctx.lineWidth = 1.2 + 5 * mag;
    kctx.beginPath(); kctx.moveTo(x1, y1); kctx.bezierCurveTo((x1 + x2) / 2, y1, (x1 + x2) / 2, y2, x2, y2); kctx.stroke();
  }
  const s = brain.free?.s;
  kctx.font = "500 19px 'IBM Plex Mono', ui-monospace, monospace";
  kctx.textBaseline = "middle";
  for (const name of Object.keys(layout)) {
    const [x, y] = pos(name), i = index[name];
    const act = s ? Math.max(0, Math.min(1, Math.abs(s[i]) * 4)) : 0;
    const removed = brain.mask[i] === 0;
    kctx.fillStyle = removed ? "#2a1f22" : `rgba(237, 244, 239, ${0.12 + 0.88 * act})`;
    kctx.strokeStyle = removed ? COLORS.warn : COLORS.muted; kctx.lineWidth = 2;
    kctx.beginPath(); kctx.arc(x, y, 9, 0, Math.PI * 2); kctx.fill(); kctx.stroke();
    const c = layout[name].c, last = PAGE.columns.length - 1;
    // outer columns label beside their node; middle columns above it, on a halo over the synapses
    const [lx, ly] = c === 0 ? [x - 16, y] : c === last ? [x + 16, y] : [x, y - 20];
    kctx.textAlign = c === 0 ? "right" : c === last ? "left" : "center";
    kctx.lineWidth = 6; kctx.lineJoin = "round"; kctx.strokeStyle = COLORS.panel;
    kctx.strokeText(name, lx, ly);
    kctx.fillStyle = name.startsWith("AWA") ? COLORS.diacetyl : name.startsWith("AWC") ? COLORS.butanone : COLORS.muted;
    kctx.fillText(name, lx, ly);
  }
}

// -- text

function updateMeaning() {
  const k = dish.meaning;
  $("meaning").innerHTML = `Food smells of <b style="color:${k === 0 ? "var(--diacetyl)" : "var(--butanone)"}">${ODOURS[k]}</b>`;
}
function updateTally() {
  $("t-decisions").textContent = state.decisions.toLocaleString("en-US");
  $("t-searches").textContent = state.searches.toLocaleString("en-US");
  const last = state.outcomes.slice(-30);
  $("t-food").textContent = last.length ? `${Math.round(share(last, "food") * 100)}%` : "–";
  $("t-empty").textContent = last.length ? `${Math.round(share(last, "empty") * 100)}%` : "–";
}

function fillResults() {
  const body = $("results").querySelector("tbody");
  if (!PAGE.results?.length) { $("measured").hidden = true; return; }
  const groups = {};
  for (const r of PAGE.results) (groups[r.kind] ??= []).push(r);
  const labels = { connectome: "Cook 2019 connectome", shuffled: "Shuffled wiring", mlp: "MLP, backprop" };
  const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
  body.innerHTML = Object.entries(groups).map(([kind, rs]) => {
    const held = rs.map((r) => r.held_out.none.drawn);
    const learnedBy = rs.map((r) => (r.curve.find((w) => w.food >= 0.9)?.decision ?? null));
    const reached = learnedBy.filter((x) => x !== null);
    const food = mean(held.map((h) => h.food));
    return `<tr><td>${labels[kind] ?? kind}</td><td>${rs.length}</td><td class="${food > 0.8 ? "good" : ""}">${Math.round(food * 100)}%</td>` +
      `<td>${Math.round(mean(held.map((h) => h.empty)) * 100)}%</td><td>${mean(held.map((h) => h.steps_to_food)).toFixed(0)}</td>` +
      `<td>${reached.length === rs.length ? `${Math.round(mean(reached)).toLocaleString("en-US")} decisions` : reached.length === 0 ? "not reached" : `${reached.length} of ${rs.length} seeds`}</td>` +
      `<td>${rs[0].parameters.toLocaleString("en-US")}</td></tr>`;
  }).join("");
}

// -- controls

$("play").addEventListener("click", () => {
  state.running = !state.running;
  $("play").textContent = state.running ? "Pause" : "Play";
  $("play").setAttribute("aria-pressed", String(state.running));
});
document.querySelectorAll("[data-speed]").forEach((b) => b.addEventListener("click", () => {
  state.speed = Number(b.dataset.speed);
  document.querySelectorAll("[data-speed]").forEach((o) => o.setAttribute("aria-pressed", String(o === b)));
}));
$("swap").addEventListener("click", () => {
  dish.meaning = 1 - dish.meaning;
  state.swaps.push(state.curve.length);
  updateMeaning();
});
$("lesion").addEventListener("change", () => applyLesion(brain));
$("learning").addEventListener("change", (e) => { state.learning = e.target.checked; brain.pending = null; });
$("naive").addEventListener("click", () => reset());
const trainedButton = $("trained");
if (PAGE.trained) trainedButton.addEventListener("click", () => reset(PAGE.trained));
else trainedButton.hidden = true;

function loop() {
  if (state.running) {
    const budget = performance.now() + 14;
    if (state.speed === 1) { if (frame % 4 === 0) decide(); }
    else for (let i = 0; i < state.speed && performance.now() < budget; i++) decide();
  }
  frame += 1;
  drawDish();
  if (frame % 12 === 0) { drawProbe(); drawCurve(); drawCircuit(); updateTally(); }
  requestAnimationFrame(loop);
}

reset();
fillResults();
drawProbe(); drawCurve(); drawCircuit(); updateTally();
requestAnimationFrame(loop);
