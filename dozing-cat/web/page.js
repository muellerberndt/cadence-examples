// The page: the sill on the left, the whole brain on the right, the brain selector and four
// tiles under them, the account and the instruments in the card. Everything the page shows
// comes from the brain that runs here, in this tab, with the arithmetic of the library: the
// belief patch (belief.js), the governor patch (governor.js), the life (life.js) and the sill
// (cat.js). Nothing is stored: a reload starts from the brain as it was exported.
import * as viewer from "./brain_scan.js";
import { BeliefPatch } from "./belief.js";
import { Life, ARMS } from "./life.js";
import { World, SCHEDULE, RETINA, READINGS, GRID, CATCH_RADIUS, MODES, Rng, setConstants, drawEvents } from "./cat.js";
import { ROLES, buildConnectome, weightsOf, activityOf } from "./connectome.js";

const $ = (id) => document.getElementById(id);
const RATE = 25; // decisions per second, the experiment's demo rate
const MODE_COLOR = { habit: "#4cc38a", imagine: "#59b7ff", learn: "#ff5c8a" };
const ARM_LABEL = { patch: "governor patch", threshold: "hand-set thresholds", never_wakes: "never wakes", always_awake: "always awake" };
const ARM_TITLE = { patch: "the governor patch: a settling patch reads the brain's own surprise and returns its mode", threshold: "the hand-designed control: a threshold rule over the same readback", never_wakes: "no governor: the habit alone, nothing reads the surprise", always_awake: "no governor, the other way: imagination every decision" };
const WHICH = { patch: "patch", threshold: "threshold", always_awake: "threshold", never_wakes: null };
const GENOME_LABEL = { hand_set: "hand-set", evolved: "evolved (the held-out winner)", random_search: "best of random search" };
const params = new URLSearchParams(location.search);

const S = {
  brain: null, patch: null, world: null, life: null, arm: null, genome: {}, scan: null, connectome: null, previousWeights: null, change: null,
  laser: false, pointer: null, mouse: null, frame: null, catches: 0, misses: 0, lastCatchLatency: null, lastWakeLatency: null,
  recentMoments: [], recentAwake: [], recentMs: [], curves: {}, marks: [], learns: [], learnFlash: 0, catchFlash: null,
  paused: false, mode: "activity", synapses: "weights", style: "brain", decisions: 0, clock: [], perSecond: 0, press: null, rng: new Rng(7),
};
window.S = S;

// ------------------------------------------------------------------ the brain
const spec = await (await fetch("data/brain.json")).json();
S.brain = spec;
setConstants(spec.world);
S.patch = new BeliefPatch(spec.belief);
for (const which of ["patch", "threshold"]) S.genome[which] = spec.default_genome[which];
const schedule = { ...SCHEDULE, ...spec.world.schedule, decisions: 1e9, change: "none" };
S.world = new World(schedule, drawEvents(new Rng(1), { ...schedule, decisions: 200000 }), new Rng(2), "mouse");

function genomeFor(arm) { const which = WHICH[arm] || "threshold"; return { ...spec.genomes[which][S.genome[which]] }; }
function buildLife(arm) {
  S.arm = arm;
  S.world.mouse = S.mouse;
  S.life = new Life(S.patch, S.world, genomeFor(arm), spec.scale, spec.floors, { arm, rng: S.rng, machinery: spec.machinery });
  S.connectome = buildConnectome(S.patch, S.life.habit, S.life.governor);
  const atlas = viewer.layoutAtlas({ n: S.connectome.n, pre: S.connectome.pre, post: S.connectome.post, weight: S.connectome.weight, groups: S.connectome.groups, roles: ROLES, seed: 0 });
  if (S.scan) S.scan.setAtlas(atlas); else makeScan(atlas); // framed once at load; on a switch the view stays where the user left it
  S.previousWeights = weightsOf(S.connectome, S.patch, S.life.habit, S.life.governor);
  S.change = null;
  loadWeights();
  S.activity = new Float32Array(S.connectome.n);
  S.heat = new Float32Array(S.connectome.n);
  S.marks.push([S.decisions, ARM_LABEL[arm]]);
  markArm(arm); fillGenomes(arm); showGenes(); legend();
}
function makeScan(atlas) {
  const canvas = $("scan");
  const wanted = "brainLayout" in viewer ? params.get("style") || "brain" : "scan";
  const options = { labels: $("labels"), strip: $("strip"), spin: false }; // framed once; no self-rotation, no camera animation
  try { S.scan = new viewer.BrainScan(canvas, atlas, { ...options, style: wanted }); }
  catch (error) { console.warn("the brain style could not start here, falling back to the scan", error); S.scan = new viewer.BrainScan(canvas, atlas, { ...options, style: "scan" }); }
  S.style = S.scan.brain ? "brain" : "scan";
  $("style").value = S.style;
  if (!("brainLayout" in viewer)) $("style").disabled = true;
  S.scan.fit();
}
function loadWeights() {
  const current = weightsOf(S.connectome, S.patch, S.life.habit, S.life.governor);
  if (S.synapses === "change" && S.change) S.scan.setWeights(S.change); else S.scan.setWeights(current);
}
function bumpWeights() {
  const current = weightsOf(S.connectome, S.patch, S.life.habit, S.life.governor);
  S.change = Float32Array.from(current, (w, i) => Math.abs(w - S.previousWeights[i]));
  S.previousWeights = current;
  loadWeights();
}
function legend() {
  $("counts").textContent = `${S.scan.n.toLocaleString()} neurons · ${S.scan.edges.toLocaleString()} synapses`;
  $("legend").innerHTML = S.scan.atlas.regions.map((r) => `<span><i style="background:rgb(${(r.color || [143, 163, 184]).join(",")})"></i>${r.name}</span>`).join("") + `<span><i style="background:var(--hot)"></i>distance from equilibrium, when selected</span>`;
}

// ------------------------------------------------------------------ the selector, the genes
const order = ["patch", "threshold", "never_wakes", "always_awake"];
order.forEach((name, i) => {
  if (i) { const sep = document.createElement("i"); sep.textContent = "·"; $("brainline").appendChild(sep); }
  const b = document.createElement("button"); b.dataset.arm = name; b.textContent = ARM_LABEL[name]; b.title = ARM_TITLE[name];
  b.onclick = () => { buildLife(name); };
  $("brainline").appendChild(b);
});
const hint = document.createElement("small"); hint.textContent = "the governor · the hand-designed control · no governor, two ways"; $("brainline").appendChild(hint);
function markArm(arm) { for (const b of $("brainline").querySelectorAll("button")) b.classList.toggle("on", b.dataset.arm === arm); }
function fillGenomes(arm) {
  const which = WHICH[arm], sel = $("genome"); sel.innerHTML = "";
  if (!which) { sel.disabled = true; const o = document.createElement("option"); o.textContent = "no governor on this brain"; sel.appendChild(o); return; }
  sel.disabled = false;
  for (const name of Object.keys(spec.genomes[which])) { const o = document.createElement("option"); o.value = name; o.textContent = `${GENOME_LABEL[name] || name}, ${which}`; sel.appendChild(o); }
  sel.value = S.genome[which];
}
const fmt = (v) => (typeof v === "number" ? +Number(v).toPrecision(3) : v);
function genesText(which, name) { const g = spec.genomes[which][name]; return Object.entries(g).filter(([k, v]) => !(typeof v === "number" && v === 0 && /^(rc|rm|cm|cb|mb)_/.test(k))).map(([k, v]) => `${k} <code>${fmt(v)}</code>`).join(" · "); }
function showGenes() {
  const which = WHICH[S.arm];
  $("genes").innerHTML = which ? `<b>${which === "patch" ? "The governor patch" : "The thresholds"} (${GENOME_LABEL[S.genome[which]] || S.genome[which]}; zero synapses omitted).</b> ` + genesText(which, S.genome[which]) : "<b>No governor on this brain.</b> The habit's genes: " + Object.entries(S.life.habit).map(([k, v]) => `${k} <code>${fmt(v)}</code>`).join(" · ");
}
$("genome").onchange = (e) => { const which = WHICH[S.arm]; if (!which) return; S.genome[which] = e.target.value; buildLife(S.arm); };

// ------------------------------------------------------------------ the texts
$("task").textContent = "A cat on a sill, eyes half closed. Its brain reads a coarse retina over the sill (six by six bumps), the fovea's readout of it (the dot's centroid, its flow, its mass) and the position of its own paw, and pushes the paw once per decision. Your mouse over the sill is the laser dot; a dot that stops moving gets caught (the paw within reach for three decisions). The brain has to catch dots while running as cheaply as it can: in the routine the habit acts (the paw drifts home, one moment of the belief patch per decision); a surprise spike, a dot where the belief expected nothing, wakes it, and while awake ten candidate pushes are imagined in the belief's private imagination and the best is executed, the chase; a quiet field brings the doze back; when the surprise persists (the dot's law has changed) the belief learns from its executed window and the habit is refitted.";
$("note").textContent = `One belief patch (cadence.BeliefPatch: ${spec.belief.belief} units, ${spec.belief.records.cells} record cells, ${spec.parameters.toLocaleString()} slow parameters): a learned transition carries the belief forward under the executed push, the reading of the moment repairs it through two iterations of one map with the store read inside, and a linear readout says how the dot's centroid, its mass and the paw will change. It was pretrained on random and pursuing pushes over the routine sill; its record store is empty, so its read is zero at every moment. The governor is a settling patch of the same kind as every other cadence brain (cadence.Brain): a readback port of seven units reads the surprise over its baseline, its slow average, the repair residual and the current mode; a cortex of four and a motor of three settle under that drive, and the settled motor state is the mode. Every synapse of the governor is a gene; the page runs the genome that won on the held-out lives. The hand-set thresholds (a hand-designed rule over the same readback) are the control; never wakes and always awake are the same brain without a governor.`;
$("here").innerHTML = `The whole brain runs in this tab, in plain JavaScript, with the arithmetic of the library in the library's order: the belief patch's moment, its imagination, its learning (the adjoint over the executed window, the validity check, the rollback and the habit's refit in imagination) and the governor's settle. A parity test replays recorded decisions of the Python brain through this code and holds beliefs, imagination, modes and actions to a relative difference under ${spec.parity ? spec.parity.tolerance : "1e-9"}. What differs: the decisions run at ${RATE} per second on your clock; the random starts of a habit refit and the scripted dots of the driven check are drawn by the page's own generator, so a refit here lands on genes of its own; and the wall-clock milliseconds are this machine's. Every number on the page is computed here as the cat lives. Library: cadence-net ${spec.library.version}${spec.library.commit ? ", commit " + spec.library.commit.slice(0, 8) : ""}; viewer: brain_scan.js${spec.viewer.commit ? " at " + spec.viewer.commit.slice(0, 8) : ""}.`;
$("facts").textContent = `${spec.parameters.toLocaleString()} slow parameters · ${spec.belief.records.cells} record cells · ${spec.moment_macs.toLocaleString()} multiply-accumulates per moment\n${RATE} decisions per second · runs in this tab`;
$("cardFacts").textContent = `cadence BeliefPatch and Brain · cadence-net ${spec.library.version}`;

function whatText() {
  const life = S.life, out = S.frame && S.frame.out, arm = S.arm;
  if (!out) return "";
  const gov = arm === "patch" ? "the governor patch" : arm === "threshold" ? "the hand-set thresholds" : arm === "always_awake" ? "always awake" : "never wakes";
  const learnNote = " Learning fires when the dot's behaviour changes for good: under your mouse it fires rarely; move the dot in a new way for a while (fast, or in circles) and the surprise persists until the belief has learned it.";
  if (arm === "never_wakes") return "<b>Never wakes (no governor).</b> The habit alone: one moment of the belief patch per decision and the drift home. Nothing reads the surprise, so a dot is never chased and is caught only when it walks into the paw. The surprise is still measured, and still spikes at every dot; nothing acts on it.";
  if (arm === "always_awake" && out.mode !== "learn") return "<b>Always awake (no governor, the other way).</b> Imagination every decision: ten candidate pushes are imagined for six decisions each, sixty moments of the patch per decision, dot or no dot. It catches, and it pays for every quiet decision as if it were a chase. Learning still follows the hand-set persistence rule." + learnNote;
  if (out.mode === "imagine") return `<b>Awake, chasing (${gov}).</b> The surprise spiked, a dot where the belief expected nothing, and ${arm === "patch" ? "the governor patch settled on its imagine unit" : "the surprise crossed the spike threshold"}. Each decision ten candidate pushes (rest, the eight directions and the habit's own) are held for six decisions in the belief's private imagination; the imagined cost is the paw's distance from the imagined dot, and the best first push is executed. Imagination writes nothing. A quiet field brings the doze back.`;
  if (out.mode === "learn") return "<b>Learning.</b> The surprise stayed up: the dot no longer moves the way the belief expects. The brain observes its executed window from the belief that was live at its start, moves the weights, and keeps the update only when the window's loss fell by a tenth; then the habit is refitted in imagination. An invalid update is undone.";
  const g = genomeFor(arm);
  const patch = arm === "patch" ? ` The governor patch reads the surprise over its baseline (${(out.surprise / Math.max(1e-9, out.baseline)).toFixed(1)} times), its slow average (log-ratio ${life.slow.toFixed(2)}), the repair residual and its own last mode, settles (${out.governorSteps} steps this decision), and its habit unit won.` : arm === "threshold" ? ` The thresholds: a spike above ${(+g.k_imagine).toPrecision(2)} baselines wakes it for ${Math.round(+g.imagine_budget)} decisions; ${life.above} recent decisions were above the learning threshold.` : "";
  return `<b>Dozing (${gov}).</b> The habit holds the paw: one moment of the belief patch per decision (the transition, two repair iterations, the store read, the readout) and the drift home. The surprise is the belief's own one-step read against what it then read; while it stays near its baseline nothing else runs.${patch}${learnNote}`;
}

// ------------------------------------------------------------------ the tiles and the instruments
const DIGITS = { awake_share: 2, moments: 1, total_moments: 1, ms_per_decision: 2 };
function tile(k, label, v, cls = "") {
  const txt = typeof v === "number" ? (k in DIGITS ? v.toFixed(DIGITS[k]) : Number.isInteger(v) ? v : (Math.abs(v) < 1e-3 && v !== 0 ? v.toExponential(2) : v.toFixed(4))) : (v === "" || v === undefined || v === null ? "–" : v);
  return `<div class="stat${cls}"><b>${txt}</b><span>${label}</span></div>`;
}
const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);
function tiles() {
  const out = S.frame && S.frame.out, mode = out ? out.mode : "habit";
  const awake = mean(S.recentAwake), moments = mean(S.recentMoments);
  $("tiles").innerHTML = tile("mode", "mode", mode, ` ${mode}`) + tile("awake_share", "awake share, last 600 decisions", awake) + tile("catches", "catches", S.catches) + tile("moments", "moments per decision, last 300", moments) + tile("per_second", "decisions per second, measured", S.perSecond);
  const st = $("state"); st.textContent = mode === "habit" ? "dozing" : mode === "imagine" ? "awake, chasing" : "learning"; st.className = `state ${mode}`;
  if (!$("card").open || !out) return;
  const life = S.life, t = life.totals, c = life.compute();
  const spec_ = [["decision", "decision", S.decisions], ["law", "the dot's law", S.world.law], ["residual", "repair residual", out.residual], ["surprise", "surprise", out.surprise], ["baseline", "surprise baseline", out.baseline], ["slow", "slow surprise", life.slow], ["governor_steps", "governor settle steps", out.governorSteps],
    ["misses", "misses", S.misses], ["catch_latency", "last catch latency", S.lastCatchLatency], ["wake_latency", "last wake latency", S.lastWakeLatency], ["learning_calls", "learning calls", t.learn_calls], ["kept_updates", "kept updates", t.kept], ["undone_updates", "undone updates", t.undone],
    ["total_moments", "moments per decision, this life, learning included", c.moments_per_decision], ["ms_per_decision", "ms per decision", mean(S.recentMs)], ["habit_home_x", "habit home x", life.habit.home_x], ["habit_home_y", "habit home y", life.habit.home_y], ["habit_drift", "habit drift", life.habit.drift]];
  $("stats").innerHTML = spec_.map(([k, l, v]) => tile(k, l, v)).join("");
}

// ------------------------------------------------------------------ the sill
function boxOf(W, H) { const side = Math.max(60, Math.min(H - 12, W - 12)); return { x: (W - side) / 2, y: (H - side) / 2, s: side }; }
function drawCat(g, cx, cy, r, mode, look, learning) {
  g.save();
  g.fillStyle = "#3a4656"; g.strokeStyle = "#5b6b7d"; g.lineWidth = 1.5;
  for (const sgn of [-1, 1]) { g.beginPath(); g.moveTo(cx + sgn * r * 0.35, cy - r * 0.75); g.lineTo(cx + sgn * r * 0.95, cy - r * 1.35); g.lineTo(cx + sgn * r * 0.95, cy - r * 0.45); g.closePath(); g.fill(); g.stroke(); }
  g.beginPath(); g.arc(cx, cy, r, 0, Math.PI * 2); g.fillStyle = "#3a4656"; g.fill(); g.stroke();
  const ex = r * 0.42, ey = -r * 0.12, er = r * 0.2;
  for (const sgn of [-1, 1]) {
    const x = cx + sgn * ex, y = cy + ey;
    if (mode === "habit") {
      g.strokeStyle = "#121a24"; g.lineWidth = 3; g.beginPath(); g.arc(x, y - er * 0.1, er, Math.PI * 0.15, Math.PI * 0.85); g.stroke();
      g.fillStyle = "#c9d5e2"; g.beginPath(); g.ellipse(x, y + er * 0.35, er * 0.9, er * 0.25, 0, 0, Math.PI * 2); g.fill();
      g.fillStyle = "#121a24"; g.beginPath(); g.ellipse(x, y + er * 0.35, er * 0.25, er * 0.22, 0, 0, Math.PI * 2); g.fill();
    } else {
      g.fillStyle = "#e8f0f8"; g.beginPath(); g.arc(x, y, er * 1.15, 0, Math.PI * 2); g.fill();
      const dx = look ? look[0] - x : 0, dy = look ? look[1] - y : 0, d = Math.hypot(dx, dy) || 1, k = Math.min(er * 0.5, d);
      g.fillStyle = mode === "imagine" ? "#0a1119" : "#3b2a44"; g.beginPath(); g.ellipse(x + dx / d * k, y + dy / d * k, er * 0.36, er * 0.62, 0, 0, Math.PI * 2); g.fill();
    }
  }
  g.fillStyle = "#ff9bb3"; g.beginPath(); g.moveTo(cx - r * 0.1, cy + r * 0.28); g.lineTo(cx + r * 0.1, cy + r * 0.28); g.lineTo(cx, cy + r * 0.4); g.closePath(); g.fill();
  g.strokeStyle = "#8fa3b8"; g.lineWidth = 1;
  for (const sgn of [-1, 1]) for (const dy of [-0.05, 0.05, 0.15]) { g.beginPath(); g.moveTo(cx + sgn * r * 0.2, cy + r * (0.38 + dy)); g.lineTo(cx + sgn * r * 1.15, cy + r * (0.3 + dy * 2.5)); g.stroke(); }
  if (learning) {
    g.fillStyle = "#e8f0f8"; g.strokeStyle = "#8fa3b8";
    for (const [k, rr] of [[0, 3], [1, 5], [2, 8]]) { g.beginPath(); g.arc(cx - r * (0.8 + 0.25 * k), cy - r * (1.35 + 0.35 * k), rr, 0, Math.PI * 2); g.fill(); }
    const bx = cx - r * 1.75, by = cy - r * 2.85;
    g.beginPath(); g.ellipse(bx, by, r * 1.15, r * 0.62, 0, 0, Math.PI * 2); g.fill(); g.stroke();
    g.fillStyle = "#3b2a44"; g.font = "12px sans-serif"; g.textAlign = "center"; g.textBaseline = "middle"; g.fillText("learning", bx, by - 6); g.fillText("the new law", bx, by + 8);
  }
  g.restore();
}
function sizeView() {
  const c = $("view"), dpr = devicePixelRatio || 1;
  const r = c.getBoundingClientRect(), w = Math.round(r.width * dpr), h = Math.round(r.height * dpr);
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  return { c, dpr, W: r.width, H: r.height };
}
function drawView() {
  const { c, dpr, W, H } = sizeView();
  const g = c.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
  g.font = "12px sans-serif"; g.textBaseline = "middle";
  const f = S.frame, live = !!f;
  const v = live ? f.view : { paw: S.world ? S.world.paw : [0.5, 0.5], dot: null, retina: [], mode: "habit", surprise: 0, baseline: 0, residual: 0, hold: 0, law: "routine", learn: false, motor: null };
  const B = boxOf(W, H);
  const X = (x) => B.x + x * B.s, Y = (y) => B.y + (1 - y) * B.s;
  g.fillStyle = "#101a26"; g.fillRect(B.x, B.y, B.s, B.s);
  const cell = B.s / GRID;
  for (let i = 0; i < GRID * GRID; i++) { const val = v.retina[i] || 0; if (val > 0.02) { const col = i % GRID, row = Math.floor(i / GRID); g.fillStyle = `rgba(255,120,120,${(0.06 + 0.4 * val).toFixed(3)})`; g.fillRect(B.x + col * cell + 1, B.y + B.s - (row + 1) * cell + 1, cell - 2, cell - 2); } }
  g.strokeStyle = "#1c2a3a"; g.lineWidth = 1; for (let k = 0; k <= GRID; k++) { g.beginPath(); g.moveTo(B.x + k * cell, B.y); g.lineTo(B.x + k * cell, B.y + B.s); g.stroke(); g.beginPath(); g.moveTo(B.x, B.y + k * cell); g.lineTo(B.x + B.s, B.y + k * cell); g.stroke(); }
  g.strokeStyle = "#3a4656"; g.strokeRect(B.x, B.y, B.s, B.s);
  // the paw: its reach, the pad and three toes, in the mode's colour
  const scale = B.s / 400;
  const px = X(v.paw[0]), py = Y(v.paw[1]), reach = CATCH_RADIUS * B.s;
  g.strokeStyle = MODE_COLOR[v.mode]; g.globalAlpha = 0.5; g.setLineDash([3, 4]); g.beginPath(); g.arc(px, py, reach, 0, Math.PI * 2); g.stroke(); g.setLineDash([]); g.globalAlpha = 1;
  g.fillStyle = MODE_COLOR[v.mode]; g.beginPath(); g.arc(px, py, 8 * scale, 0, Math.PI * 2); g.fill();
  for (const a of [-0.6, 0, 0.6]) { g.beginPath(); g.arc(px + Math.sin(a) * 12 * scale, py - Math.cos(a) * 12 * scale, 3.5 * scale, 0, Math.PI * 2); g.fill(); }
  if (v.hold > 0) { g.fillStyle = "#ffe680"; for (let k = 0; k < v.hold; k++) { g.beginPath(); g.arc(px - 8 + 8 * k, py + 16 * scale, 2, 0, Math.PI * 2); g.fill(); } }
  // the laser dot: a red point with a faint glow, drawn at the pointer's place now (the brain reads it once per decision)
  const shown = S.laser && S.pointer && !(S.world && S.world.mouseCaught && S.world.dot === null) ? S.pointer : v.dot;
  if (shown) { const dx = X(shown[0]), dy = Y(shown[1]); const R = 18 * Math.max(0.7, scale); const grad = g.createRadialGradient(dx, dy, 0, dx, dy, R); grad.addColorStop(0, "rgba(255,80,80,0.9)"); grad.addColorStop(0.3, "rgba(255,40,40,0.35)"); grad.addColorStop(1, "rgba(255,40,40,0)"); g.fillStyle = grad; g.beginPath(); g.arc(dx, dy, R, 0, Math.PI * 2); g.fill(); g.fillStyle = "#fff0f0"; g.beginPath(); g.arc(dx, dy, 3, 0, Math.PI * 2); g.fill(); }
  else if (S.laser && S.pointer) { g.fillStyle = "rgba(255,80,80,0.35)"; g.beginPath(); g.arc(X(S.pointer[0]), Y(S.pointer[1]), 4, 0, Math.PI * 2); g.fill(); } // the laser is on and the dot was caught: it returns when the mouse moves
  // the dot the cat saw at its last decision, as a faint ring when it is not yet where the pointer is
  if (v.dot && shown && shown !== v.dot && Math.hypot(v.dot[0] - shown[0], v.dot[1] - shown[1]) > 0.01) { g.strokeStyle = "rgba(255,120,120,0.45)"; g.lineWidth = 1; g.beginPath(); g.arc(X(v.dot[0]), Y(v.dot[1]), 5, 0, Math.PI * 2); g.stroke(); }
  if (S.catchFlash && performance.now() - S.catchFlash.t < 700 && S.catchFlash.at) { const k = (performance.now() - S.catchFlash.t) / 700; g.strokeStyle = `rgba(255,230,128,${(1 - k).toFixed(2)})`; g.lineWidth = 2; g.beginPath(); g.arc(X(S.catchFlash.at[0]), Y(S.catchFlash.at[1]), 8 + 30 * k, 0, Math.PI * 2); g.stroke(); g.lineWidth = 1; }
  // the cat sits at the sill's lower right corner, looking at the dot or at its paw
  const r = Math.max(26, Math.min(56, B.s * 0.12)), cx = B.x + B.s - r * 1.4, cy = B.y + B.s - r * 0.35;
  const look = v.dot ? [X(v.dot[0]), Y(v.dot[1])] : [px, py];
  const learning = v.mode === "learn" || v.learn || (S.learnFlash && performance.now() - S.learnFlash < 2000);
  drawCat(g, cx, cy, r, learning ? "learn" : v.mode, look, learning);
  // the account, top left of the sill
  g.font = "12px sans-serif"; g.textAlign = "left";
  const lines = [[live ? (v.mode === "habit" ? "dozing" : v.mode === "imagine" ? "awake, chasing" : "learning") : "dozing", MODE_COLOR[v.mode]], [S.laser ? (v.dot ? "the laser is on" : "the laser is on, the dot was caught: move to bring it back") : "click the sill to switch the laser on", S.laser ? "#ff8a8a" : "#8fa3b8"], [`surprise ${v.surprise.toFixed(3)} · baseline ${v.baseline.toFixed(3)}`, "#8fa3b8"]];
  if (v.law && v.law !== "routine") lines.push([`the dot's law: ${v.law}`, "#ffbf70"]);
  if (v.motor) lines.push([`governor motor ${v.motor.map((m) => m.toFixed(2)).join(" ")}`, "#ed81b6"]);
  g.fillStyle = "rgba(10,17,25,0.75)"; g.fillRect(B.x + 4, B.y + 4, Math.min(B.s - 8, 250), 8 + 16 * lines.length);
  lines.forEach(([t, col], i) => { g.fillStyle = col; g.fillText(t, B.x + 10, B.y + 14 + 16 * i); });
}
window.boxOf = boxOf;

// the laser: a click on the sill switches it on, it follows the pointer while on, a click switches it off
function sillPoint(ev) { const c = $("view"); const rect = c.getBoundingClientRect(); const B = boxOf(rect.width, rect.height); return [(ev.clientX - rect.left - B.x) / B.s, 1 - (ev.clientY - rect.top - B.y) / B.s]; }
// The pointer's place on the sill is kept at every move and drawn at every animation frame; the
// brain reads the latest place once per decision. A press and release without a drag toggles
// the laser exactly once; a drag never does.
function setDot(ev) {
  const [x, y] = sillPoint(ev);
  const inside = x >= 0 && x <= 1 && y >= 0 && y <= 1;
  S.pointer = inside ? [x, y] : null;
  S.mouse = S.laser && inside ? [Math.min(1, Math.max(0, x)), Math.min(1, Math.max(0, y))] : null;
}
function laserOff() { S.laser = false; S.mouse = null; }
$("view").addEventListener("pointerdown", (ev) => { S.press = { x: ev.clientX, y: ev.clientY, t: performance.now(), id: ev.pointerId }; setDot(ev); ev.preventDefault(); });
$("view").addEventListener("pointermove", setDot);
$("view").addEventListener("pointerup", (ev) => {
  const press = S.press; S.press = null;
  if (!press || press.id !== ev.pointerId) return;
  const moved = Math.hypot(ev.clientX - press.x, ev.clientY - press.y);
  if (moved > 6 || performance.now() - press.t > 600) return; // a drag, or a long press: no toggle
  if (S.laser) laserOff(); else S.laser = true;
  setDot(ev);
});
$("view").addEventListener("pointercancel", () => { S.press = null; });
$("view").addEventListener("pointerleave", () => { S.press = null; S.pointer = null; S.mouse = null; });
window.laserOn = () => { S.laser = true; };

// ------------------------------------------------------------------ the charts
function point(curve, x, y) { const c = (S.curves[curve] ||= []); c.push([x, y]); if (c.length > 1200) c.splice(0, c.length - 1200); }
function chart(id, series, opts) {
  const c = $(id), dpr = devicePixelRatio || 1;
  if (!c.clientWidth) return;
  if (c.width !== Math.round(c.clientWidth * dpr)) { c.width = Math.round(c.clientWidth * dpr); c.height = Math.round(c.clientHeight * dpr); }
  const g = c.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = c.clientWidth, H = c.clientHeight, m = { l: 48, r: 10, t: 10, b: 24 }; g.clearRect(0, 0, W, H);
  const all = series.flatMap((s) => s.points); if (!all.length) { g.fillStyle = "#8fa3b8"; g.font = "12px sans-serif"; g.fillText("waiting for the life", m.l, H / 2); return; }
  const tr = (y) => (opts.log ? Math.log10(Math.max(y, 1e-9)) : y);
  const xs = all.map((p) => p[0]), ys = all.map((p) => tr(p[1]));
  const x0 = Math.min(...xs), x1 = Math.max(...xs, x0 + 1), y0 = opts.zero && !opts.log ? 0 : Math.min(...ys), y1 = Math.max(...ys, y0 + 1e-9, opts.one ? 1 : -Infinity);
  const X = (x) => m.l + (x - x0) / (x1 - x0) * (W - m.l - m.r), Y = (y) => H - m.b - (y - y0) / (y1 - y0) * (H - m.t - m.b);
  if (opts.bands) { for (let i = 0; i + 1 < opts.bands.length; i++) { const [x, mode] = opts.bands[i]; const xn = opts.bands[i + 1][0]; g.fillStyle = MODE_COLOR[["habit", "imagine", "learn"][mode]]; g.globalAlpha = mode === 0 ? 0.12 : 0.35; g.fillRect(X(x), m.t, Math.max(1, X(xn) - X(x)), H - m.t - m.b); } g.globalAlpha = 1; }
  g.strokeStyle = "#182430"; g.lineWidth = 1; g.font = "11px sans-serif"; g.fillStyle = "#8fa3b8"; g.textAlign = "right";
  for (let i = 0; i <= 3; i++) { const v = y0 + (y1 - y0) * i / 3, y = Y(v); g.beginPath(); g.moveTo(m.l, y); g.lineTo(W - m.r, y); g.stroke(); g.fillText(opts.log ? Math.pow(10, v).toPrecision(2) : (opts.fixed !== undefined ? v.toFixed(opts.fixed) : v.toPrecision(2)), m.l - 6, y + 4); }
  g.textAlign = "center"; g.fillText(opts.xlabel, (m.l + W - m.r) / 2, H - 6);
  if (opts.marks) for (const [x, label] of opts.marks) { if (x < x0 || x > x1) continue; g.strokeStyle = "#ff5c8a"; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(X(x), m.t); g.lineTo(X(x), H - m.b); g.stroke(); g.setLineDash([]); g.fillStyle = "#ff5c8a"; g.textAlign = "left"; g.fillText(label, X(x) + 3, m.t + 10); }
  series.forEach((s, k) => {
    g.strokeStyle = s.color; g.lineWidth = s.width || 2; g.lineJoin = "round"; g.beginPath();
    if (s.dots) { s.points.forEach((p) => { g.fillStyle = s.color; g.beginPath(); g.arc(X(p[0]), Y(tr(p[1])), 3, 0, Math.PI * 2); g.fill(); }); }
    else { s.points.forEach((p, i) => { i ? g.lineTo(X(p[0]), Y(tr(p[1]))) : g.moveTo(X(p[0]), Y(tr(p[1]))); }); g.stroke(); }
    const last = s.points[s.points.length - 1]; g.fillStyle = s.color; g.textAlign = "left"; g.fillText(`${s.name} ${Number(last[1]).toPrecision(3)}`, m.l + 4, m.t + 12 + 14 * k);
  });
}
function drawCharts() {
  if (!$("card").open) return;
  const C = S.curves, window_ = 600;
  const awake = (C.awake || []).slice(-window_), sur = (C.surprise || []).slice(-window_), base = (C.baseline || []).slice(-window_), modes = (C.mode || []).slice(-window_);
  const catches = (C.catches || []).slice(-window_), misses = (C.misses || []).slice(-window_);
  $("chartANote").textContent = "The share of the last 600 decisions spent awake (imagining or learning), with the mode as a band (dozing dim, awake blue, learning pink). Brain switches are marked.";
  chart("chartA", [{ name: "awake share", points: awake, color: "#59b7ff" }], { xlabel: "decisions", bands: modes, marks: S.marks, fixed: 2, zero: true, one: true });
  $("chartBNote").textContent = "The belief's own one-step read against what it then read (the change of the dot's centroid and mass and of the paw, in units of their motion), per decision, and the running baseline the governor compares with. A dot appearing is a spike; a dot that moves in a new way keeps the surprise up during every chase until the brain has learned. Learning calls are dots: the window's loss after learning.";
  const learnPts = S.learns.filter((l) => l.t >= (sur[0] || [0])[0]).map((l) => [l.t, l.after]);
  chart("chartB", [{ name: "surprise", points: sur, color: "#ff5c8a", width: 1 }, { name: "baseline", points: base, color: "#59e5cb" }].concat(learnPts.length ? [{ name: "loss after learning", points: learnPts, color: "#ffbf70", dots: true }] : []), { log: true, xlabel: "decisions", bands: modes });
  $("chartCNote").textContent = "Cumulative catches (the paw within reach of the dot for three decisions) and misses (a scripted dot that timed out). Under your laser a dot that stops moving gets caught.";
  chart("chartC", [{ name: "catches", points: catches, color: "#ffe680" }, { name: "misses", points: misses, color: "#8fa3b8" }], { xlabel: "decisions", zero: true, fixed: 0, marks: S.learns.filter((l) => l.t >= (catches[0] || [0])[0]).map((l) => [l.t, l.kept ? "learned" : "undone"]) });
}

// ------------------------------------------------------------------ one decision, and its frame
function decide() {
  const life = S.life, world = S.world;
  world.mouse = S.mouse === null ? null : S.mouse.slice(); // the latest dot, read once per decision
  const out = life.decide({ wantCode: true });
  S.decisions += 1;
  const flags = out.flags;
  if (flags.catch) {
    S.catches += 1; S.catchFlash = { t: performance.now(), at: (world.mouseCaught || world.paw).slice() };
    const d = life.dots[life.dots.length - 1];
    if (d && d.end !== null) { S.lastCatchLatency = d.end - d.appear; S.lastWakeLatency = d.wake !== null ? d.wake - d.appear : null; }
  }
  if (flags.miss) S.misses += 1;
  if (out.learn) {
    const e = out.learn;
    S.learns.push({ t: S.decisions, before: e.loss_before, after: e.loss_after, kept: e.kept, refit: e.refit });
    S.marks.push([S.decisions, "learn" + (e.kept ? " (kept)" : " (undone)")]);
    S.learnFlash = performance.now();
    bumpWeights(); // the synapses drawn keep their places; their weights follow the update and the refitted habit
  }
  S.recentMs.push(out.ms); if (S.recentMs.length > 100) S.recentMs.shift();
  S.recentMoments.push(out.moments); if (S.recentMoments.length > 300) S.recentMoments.shift();
  S.recentAwake.push(out.mode !== "habit" ? 1 : 0); if (S.recentAwake.length > 600) S.recentAwake.shift();
  point("surprise", S.decisions, out.surprise); point("baseline", S.decisions, out.baseline); point("mode", S.decisions, MODES[out.mode]);
  point("awake", S.decisions, mean(S.recentAwake)); point("catches", S.decisions, S.catches); point("misses", S.decisions, S.misses);
  // the frame: every neuron's activity this decision, from the moment's own computation
  let gov = null;
  if (life.governor) {
    const rb = out.readback, act = out.governorActivation;
    gov = { readback: [Math.min(1, rb[0] / 4), Math.min(1, rb[1] / 4), Math.min(1, rb[2] / 2), rb[3], rb[4], rb[5], 1.0], cortex: Array.from(life.governor.cortex, (i) => act[i]), motor: Array.from(life.governor.motor, (i) => act[i]) };
  }
  const path = out.path, e = path.e[0][0], z = path.belief[0][0], code = S.patch.records.dense(path.code[0][0]), y = out.expected;
  activityOf(S.connectome, S.activity, out.reading, out.action, e, z, code, y, out.mode === "habit" ? out.action : [0, 0], gov);
  S.heat.fill(0); const step = path.step[0][0], ix = S.connectome.ix.belief; for (let i = 0; i < step.length; i++) S.heat[ix.start + i] = Math.min(1, Math.abs(step[i]) * 4);
  S.frame = { out, view: { paw: world.paw.slice(), dot: world.dot ? world.dot.slice() : null, retina: Array.from(out.next.subarray(0, RETINA)), mode: out.mode, surprise: out.surprise, baseline: out.baseline, residual: out.residual, hold: world.hold, law: world.law, learn: !!out.learn, motor: gov ? gov.motor : null } };
  if (!S.paused) {
    if (S.mode === "heat") S.scan.show(S.activity, S.heat, { draw: false });
    else if (S.decisions === 1 || S.reset) { S.scan.reset(S.activity); S.reset = false; }
    else S.scan.step(S.activity, { draw: false });
  }
}

// ------------------------------------------------------------------ the loop
function step() {
  decide();
  const now = performance.now();
  S.clock.push(now); while (S.clock.length && S.clock[0] < now - 1000) S.clock.shift();
  S.perSecond = S.clock.length;
  tiles(); $("what").innerHTML = whatText(); if ($("card").open && S.decisions % 5 === 0) drawCharts();
}
function tick(now) {
  drawView(); // the sill at every animation frame: the dot is where the pointer is now
  S.scan.draw(now);
  requestAnimationFrame(tick);
}
$("mode").onchange = (e) => { S.mode = e.target.value; S.reset = true; };
$("synapses").onchange = (e) => { S.synapses = e.target.value; loadWeights(); };
$("style").onchange = (e) => { if (S.scan.setStyle) { S.scan.setStyle(e.target.value); S.style = S.scan.brain ? "brain" : "scan"; e.target.value = S.style; } };
$("pause").onclick = () => { S.paused = !S.paused; $("pause").textContent = S.paused ? "Play the viewer" : "Pause the viewer"; };
$("fit").onclick = () => S.scan.fit();
let framed = [0, 0];
window.addEventListener("resize", () => { const r = $("scan").getBoundingClientRect(); if (r.width !== framed[0] || r.height !== framed[1]) { framed = [r.width, r.height]; S.scan.fit(); } drawCharts(); });
$("card").addEventListener("toggle", () => { drawCharts(); tiles(); });

// the law change and the scripted dots stay reachable for the driven check: ?law=gravity|faster|wrap and ?script=1
const startArm = ARMS.includes(params.get("arm")) ? params.get("arm") : "patch";
buildLife(startArm);
if (params.get("law")) S.world.fire(params.get("law"));
if (params.get("script")) S.world.fire("script");
window.fire = (name) => S.world.fire(name);
drawView(); tiles();
$("what").innerHTML = "<b>Dozing.</b> The cat waits for the laser.";
setInterval(step, 1000 / RATE); // the decisions on a fixed timer, RATE per second
requestAnimationFrame(tick);
