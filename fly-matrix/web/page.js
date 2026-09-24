// A fly in a room: the page. The body flies in the three.js room (body.js, room.js, fly.js), the
// life layer decides (life.js), the brain settles in a worker (worker.js over brain.js) on the
// flight-and-instinct sub-net of the BANC connectome and learns there (learner.js, the library's
// actor-critic on the Kenyon cell to mushroom body output synapses), and the whole nervous system
// is drawn at its measured positions (brain_scan.js on data/atlas.json) with the sub-net's
// activity live. A second view keeps the fly in close-up at all times.
import * as THREE from "three";
import { Flight, DT, cos_, sin_ } from "./body.js";
import * as ROOM from "./room.js";
import { createRoom, createRenderer, createComposer, CameraRig, toThree, fromThree } from "./room.js";
import { createFly } from "./fly.js";
import { createFlyEye } from "./eye.js";
import { Life, DECISION_S } from "./life.js";
import { BrainScan, decodeAtlas } from "./brain_scan.js";
import { GlitchPass } from "three/addons/postprocessing/GlitchPass.js";

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const BRAIN_MS = 2.0;                      // ms of simulated time per settling step (declared timescale)
const STEPS_PER_MESSAGE = 12;
// The lessons: the constants of tools/learn_odour.py; data/lessons.json (written with the receipt) overrides them.
let LESSONS = { outputs: ["mbon:MBON11:right", "mbon:MBON05:left"], actions: [0, 1], plastic: "kc>mbon", critic: "kc", beta: 0.1, temperature: 0.3, nudgedSteps: 10, tolerance: 1e-3, gamma: 0.95, lam: 0.9, eta: 10, etaCritic: 0.5, cap: 3, dopamineCap: 1, tonic: {} };

// ---- the room ------------------------------------------------------------------------------
const canvas = $("world");
const renderer = createRenderer(canvas);
// the pixel ratio decides the cost of the bloom pass and of every extra view: 1.25 keeps a Retina laptop at a high frame rate (?dpr=2 for the full density)
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, Number(params.get("dpr") || 1.25)));
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x000000);
const world = new THREE.Group(); world.rotation.x = -Math.PI / 2; scene.add(world);
const room = createRoom(world);
const fly = createFly(); world.add(fly.group, fly.shadow, fly.locator);
const rig = new CameraRig(innerWidth / innerHeight);
const { composer, bloom } = createComposer(renderer, scene, rig.camera);
const eye = createFlyEye(renderer, scene, { hide: [fly.group, fly.shadow, fly.locator] });
const renderPass = composer.passes && composer.passes[0];
// a rare glitch in the Matrix: the screen tears for a moment every so often, and the room's own elements jitter
const glitch = new GlitchPass(); glitch.goWild = false; glitch.enabled = false; composer.addPass(glitch);
// debug switches for the frame budget: ?nobloom=1 ?noscan=1 ?noviews=1 ?lines=60000 ?particles=40000 ?scanrate=2 (draw the brain every n-th frame)
const DEBUG = { nobloom: !!params.get("nobloom"), noscan: !!params.get("noscan"), noviews: !!params.get("noviews"), scanrate: Number(params.get("scanrate") || 1) };
if (DEBUG.nobloom) bloom.enabled = false;
let nextGlitch = 12 + 30 * Math.random(), glitchLeft = 0;
function glitches(dt) {
  nextGlitch -= dt;
  if (nextGlitch <= 0) { glitchLeft = 0.12 + 0.25 * Math.random(); nextGlitch = 15 + 40 * Math.random(); glitch.enabled = true; document.body.classList.add("glitch"); if (room.glitch) room.glitch(); }
  if (glitchLeft > 0) { glitchLeft -= dt; if (glitchLeft <= 0) { glitch.enabled = false; document.body.classList.remove("glitch"); } }
}

// ---- the fly and its life --------------------------------------------------------------------
const S = { speed: 0.5, pilot: "brain", paused: false, ready: false, brainReady: false, atlasReady: false, learnReady: false, simTime: 0, brainDue: 0, pending: false, brainSteps: 0, brainMs: 0, active: 0, frames: 0, fps: 0, fpsClock: 0, lastScan: 0, seed: Number(params.get("seed") || 1), swapViews: false, lessons: [], lessonStats: null, lastLesson: null, decisions: 0 };
let flight = new Flight([1.2, 1.0, 1.2], 0.3);
let life = new Life(flight, room, S.seed);
if (room.setSugar) room.setSugar(life.sugar);

// ---- the brain in its worker ---------------------------------------------------------------
const worker = new Worker("./worker.js", { type: "module" });
let members = null, whole = 0, activation = null, payloadInfo = null;
worker.onmessage = (e) => {
  const m = e.data;
  if (m.type === "ready") {
    members = m.members; whole = m.whole.neurons; payloadInfo = m;
    S.brainReady = true; loadingStep("the brain is awake", 0.9);
    worker.postMessage({ type: "run", stimuli: life.senses(null, 0), steps: 200, baseline: true });  // the level-flight rest
    S.pending = true;
    worker.postMessage({ type: "learn:init", config: LESSONS, seed: S.seed });
    fillCard();
    return;
  }
  if (m.type === "learn:ready") { S.learnReady = true; S.lessonStats = m; lessonsFoot(); return; }
  if (m.type === "lesson") { S.lessonStats = m; if (m.delta !== undefined) { S.lastLesson = m; S.lessons.push(m.delta); if (S.lessons.length > 60) S.lessons.shift(); life.log(`dopamine ${m.delta >= 0 ? "+" : ""}${m.delta.toFixed(2)}: ${m.why || "a lesson"}`); } lessonsFoot(); drawDopamine(); return; }
  if (m.type === "state") {
    S.pending = false; S.brainSteps = m.steps; S.brainMs = m.ms / Math.max(1, m.ran || 1); S.active = m.active;
    life.readouts = m.readouts;
    if (m.baseline) { life.setBaseline(m.readouts); finishLoading(); }
    if (m.decision) { life.applyDecision(m.decision); S.decisions++; drawValence(); }
    if (activation && members) { for (let i = 0; i < members.length; i++) activation[members[i]] = m.s[i]; S.newState = true; }
  }
};
fetch("./data/lessons.json").then((r) => (r.ok ? r.json() : null)).then((j) => { if (j && j.config) LESSONS = { ...LESSONS, ...j.config }; }).catch(() => {}).finally(() => worker.postMessage({ type: "init", url: "./data/brain.json" }));

// ---- the whole nervous system ---------------------------------------------------------------
let scan = null;
fetch("./data/atlas.json").then((r) => r.json()).then((raw) => {
  const atlas = decodeAtlas(raw);
  // the Matrix palette: every region its own shade of green (hue 105 to 160, the mushroom body the brightest)
  atlas.regions.forEach((r, k) => {
    const name = (r.name || "").toLowerCase();
    const hue = name.includes("mushroom") ? 130 : 105 + (k * 37) % 56, light = name.includes("mushroom") ? 0.72 : 0.42 + ((k * 13) % 5) * 0.06;
    const c = new THREE.Color().setHSL(hue / 360, 0.95, light);
    r.color = [Math.round(255 * c.r), Math.round(255 * c.g), Math.round(255 * c.b)];
  });
  activation = new Float32Array(atlas.n);
  if (DEBUG.noscan) { S.atlasReady = true; loadingStep("the nervous system is drawn", 0.6); return; }
  scan = new BrainScan($("scan"), atlas, { style: params.get("style") === "scan" ? "scan" : "brain", labels: $("labels"), strip: $("strip"), shell: false, spin: true, spinRate: 0.06, montageRows: 9, restAlpha: 0.06, lineBudget: Number(params.get("lines") || 60000), particleBudget: Number(params.get("particles") || 40000), background: [0, 0.012, 0.004], hot: [0.85, 1.0, 0.88], cool: [0.2, 0.75, 0.45], exposure: 0.62, glow: 2.4, bloom: 0.9, labelTop: 4, labelCount: 5, heatDecay: 0.93, dpr: Number(params.get("dpr") || 1.25) });
  scan.set(activation);
  $("legend").innerHTML = atlas.regions.map((r) => `<span><i style="background:rgb(${r.color.join(",")})"></i>${r.name}</span>`).join("");
  S.atlasReady = true; loadingStep("the nervous system is drawn", 0.6);
  $("style-brain").onclick = () => { scan.setStyle("brain"); styleButtons("brain"); };
  $("style-scan").onclick = () => { scan.setStyle("scan"); styleButtons("scan"); };
  $("brain-fit").onclick = () => scan.fit();
  $("brain-big").onclick = () => { const big = $("brain").classList.toggle("big"); $("brain-big").textContent = big ? "shrink" : "expand"; scan.options.labelTop = big ? 12 : 4; scan.options.labelCount = big ? 0 : 5; scan.options.exposure = big ? 0.85 : 0.62; setTimeout(() => { scan.resize && scan.resize(); scan.fit(); }, 300); };
  setTimeout(() => { scan.resize && scan.resize(); scan.fit(); }, 50);
  addEventListener("resize", () => { scan.resize && scan.resize(); scan.fit(); });
  styleButtons(scan.brain ? "brain" : "scan");
});
function styleButtons(which) { $("style-brain").classList.toggle("on", which === "brain"); $("style-scan").classList.toggle("on", which === "scan"); }
S.speed = 0.5;

// ---- loading -------------------------------------------------------------------------------
function loadingStep(text, frac) { $("loading-text").textContent = text; $("loading-bar").style.width = (100 * frac).toFixed(0) + "%"; }
function finishLoading() { if (S.ready) return; S.ready = true; loadingStep("ready", 1); setTimeout(() => $("loading").classList.add("gone"), 400); }
loadingStep("loading the nervous system (36 MB)", 0.1);

// ---- the pointer: pick a fruit up, put it down on the table, strike the fly on a fruit ---------
const mouse = new THREE.Vector2(), ray = new THREE.Raycaster();
let carrying = null, hover = null;
function mainCamera() { return S.swapViews && rig.closeup ? rig.closeup : rig.camera; }
function insetCamera() { return S.swapViews || !rig.closeup ? rig.camera : rig.closeup; }
function fruitUnder(e) {
  if (!room.fruits) return null;
  mouse.set(e.clientX / innerWidth * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
  ray.setFromCamera(mouse, mainCamera());
  let best = null, bestD = Infinity;
  for (const [name, f] of Object.entries(room.fruits)) {
    if (!f.group) continue;
    const hits = ray.intersectObject(f.group, true);
    if (hits.length && hits[0].distance < bestD) { best = name; bestD = hits[0].distance; }
  }
  if (best) return best;
  // a generous pick: the ray passing within 6 cm of a fruit's position
  for (const [name, f] of Object.entries(room.fruits)) { const p = toThree(...f.pos); const d = ray.ray.distanceToPoint(p); if (d < 0.06 && d < bestD) { best = name; bestD = d; } }
  return best;
}
function tableUnder(e) {
  mouse.set(e.clientX / innerWidth * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
  ray.setFromCamera(mouse, mainCamera());
  if (room.tableMesh) { const hits = ray.intersectObject(room.tableMesh, true); if (hits.length) return fromThree(hits[0].point); }
  // the table top as a plane, when no mesh is exposed
  const t = room.table, plane = new THREE.Plane().setFromNormalAndCoplanarPoint(new THREE.Vector3(0, 1, 0), toThree(0, 0, t.z)), hit = new THREE.Vector3();
  if (ray.ray.intersectPlane(plane, hit)) { const p = fromThree(hit); if (p[0] > t.x0 && p[0] < t.x1 && p[1] > t.y0 && p[1] < t.y1) return p; }
  return null;
}
function flyUnder(e) {
  mouse.set(e.clientX / innerWidth * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
  ray.setFromCamera(mouse, mainCamera());
  return ray.ray.distanceToPoint(toThree(...flight.p)) < (life.mode === "flying" ? 0.03 : 0.05);
}
canvas.addEventListener("pointermove", (e) => { const f = fruitUnder(e); if (f !== hover) { hover = f; if (room.highlightFruit) room.highlightFruit(carrying || hover); canvas.style.cursor = f || carrying ? "pointer" : "default"; } });
canvas.addEventListener("pointerdown", (e) => {
  if (carrying) { // put the fruit down where the table was clicked
    const p = tableUnder(e);
    if (p && room.placeFruit) { const t = room.table, x = Math.min(t.x1 - 0.08, Math.max(t.x0 + 0.08, p[0])), y = Math.min(t.y1 - 0.08, Math.max(t.y0 + 0.08, p[1])); room.placeFruit(carrying, [x, y]); life.log(`the ${carrying} is moved`); }
    carrying = null; if (room.highlightFruit) room.highlightFruit(null); return;
  }
  if (flyUnder(e) && life.mode !== "flying") { strike(); return; }
  const f = fruitUnder(e);
  if (f) { carrying = f; if (room.highlightFruit) room.highlightFruit(f); life.log(`the ${f} is picked up`); }
});
canvas.addEventListener("wheel", (e) => { rig.wheel(e.deltaY); e.preventDefault(); }, { passive: false });
function strike() { // a blow at the sitting fly: a punishment for the smell it sits in, and a startle
  const punished = life.punished();
  life.log(punished ? "struck" : "startled");
  if (punished) queueReward();
  life.takeoff("startled off the surface");
  flight.swat([0.3 * (Math.random() - 0.5), 0.3 * (Math.random() - 0.5), 0.6], [0, 0, 2], [20 * (Math.random() - 0.5), 20 * (Math.random() - 0.5), 10]);
}
function queueReward() { // the outcome of the open search goes to the worker before its next step
  const r = life.pendingReward; if (!r) return;
  life.pendingReward = null;
  if (S.learnReady && S.pilot !== "instincts") worker.postMessage({ type: "reward", reward: r.reward, done: r.done, why: r.why });
}

// ---- controls on screen ----------------------------------------------------------------------
for (const b of document.querySelectorAll("[data-cam]")) b.addEventListener("click", () => setCam(b.dataset.cam));
function setCam(mode) { rig.setMode(mode); for (const b of document.querySelectorAll("[data-cam]")) b.classList.toggle("on", b.dataset.cam === mode); }
for (const b of document.querySelectorAll("[data-speed]")) b.addEventListener("click", () => { S.speed = Number(b.dataset.speed); for (const c of document.querySelectorAll("[data-speed]")) c.classList.toggle("on", c === b); });
for (const b of document.querySelectorAll("[data-speed]")) b.classList.toggle("on", Number(b.dataset.speed) === S.speed);
for (const b of document.querySelectorAll("[data-pilot]")) b.addEventListener("click", () => setPilot(b.dataset.pilot));
function setPilot(which) {
  S.pilot = which; for (const b of document.querySelectorAll("[data-pilot]")) b.classList.toggle("on", b.dataset.pilot === which);
  worker.postMessage({ type: "shuffle", on: which === "shuffled", seed: S.seed });
  life.valence = null; life.search = null; life.pendingReward = null;
  life.log(which === "brain" ? "the measured wiring steers and learns" : which === "shuffled" ? "the wiring is shuffled; the same lessons run on it" : "the hand-written instincts alone: every smell is approached, nothing is learned");
  lessonsFoot();
}
for (const b of document.querySelectorAll("[data-sugar]")) b.addEventListener("click", () => setSugar(b.dataset.sugar || null));
function setSugar(name) {
  for (const b of document.querySelectorAll("[data-sugar]")) b.classList.toggle("on", (b.dataset.sugar || null) === name);
  life.setSugar(name); if (room.setSugar) room.setSugar(name); drawValence();
}
$("forget").onclick = () => { worker.postMessage({ type: "learn:reset" }); life.lastP = { banana: null, bread: null }; life.visits = { banana: 0, bread: 0, sweet: 0, punished: 0 }; S.lessons = []; S.lastLesson = null; S.decisions = 0; life.log("the synapses are as measured again"); drawValence(); drawDopamine(); };
// the arena's lessons: learned efficacies mapped onto this sub-net, when the campaign has written them
let learned = null;
fetch("./data/learned.json").then((r) => (r.ok ? r.json() : null)).then((j) => { if (!j || !j.edges || !j.edges.length) return; learned = j; $("lessons").style.display = ""; $("lessons").title = `${j.mapped.toLocaleString()} synapses changed in the odour arena (${j.checkpoint}); click to switch them on and off`; }).catch(() => {});
$("lessons").onclick = () => { if (!learned) return; const on = !$("lessons").classList.contains("on"); $("lessons").classList.toggle("on", on); worker.postMessage({ type: "learned", on, edges: learned.edges, efficacy: learned.efficacy }); life.log(on ? "the arena's lessons are in the synapses" : "the synapses are as measured"); };
function togglePause() { S.paused = !S.paused; life.log(S.paused ? "paused" : "resumed"); }
$("about").onclick = () => { $("card").classList.toggle("open"); };
$("instruments").onclick = () => { const on = document.body.classList.toggle("instruments"); $("instruments").classList.toggle("on", on); };

$("inset").addEventListener("click", () => { S.swapViews = !S.swapViews; if (renderPass) renderPass.camera = mainCamera(); $("inset-mode").textContent = S.swapViews ? "the room" : ""; });
$("inset").addEventListener("wheel", (e) => { if (rig.closeupZoom) rig.closeupZoom(e.deltaY); e.preventDefault(); }, { passive: false });
addEventListener("keydown", (e) => { if (e.key === "1") setCam("follow"); if (e.key === "2") setCam("room"); if (e.key === "3") setCam("eye"); if (e.key === " ") { e.preventDefault(); togglePause(); } if (e.key === "i") $("instruments").onclick(); if (e.key === "e") setEyeMain(!S.eyeMain); if (e.key === "g") { const [w, x, y, z] = flight.q, c = cos_(-0.175), sn = sin_(-0.175); flight.q = [w * c - y * sn, x * c + z * sn, y * c + w * sn, z * c - x * sn]; flight.kick(0, -25, 0); if (life.mode !== "flying") life.takeoff("a gust"); life.log("a gust"); } });

function resize() { const w = innerWidth, h = innerHeight; renderer.setSize(w, h); composer.setSize(w, h); bloom.resolution.set(w / 2, h / 2); rig.camera.aspect = w / h; rig.camera.updateProjectionMatrix(); }
addEventListener("resize", resize); resize();

// ---- the HUD --------------------------------------------------------------------------------
const SENSE_ROWS = [["haltere tone", (s) => s["haltere:left"]], ["ocellus L", (s) => s["ocelli:left"]], ["ocellus R", (s) => s["ocelli:right"]], ["banana odour L", (s) => s["orn:decaying_fruit:left"]], ["banana odour R", (s) => s["orn:decaying_fruit:right"]], ["bread odour L", (s) => s["orn:yeasty:left"]], ["bread odour R", (s) => s["orn:yeasty:right"]], ["sugar", (s) => s["grn:sugar:labellum"]], ["leg touch", (s) => s["leg_touch"]], ["wind", (s) => s["jo:C:left"]]];
const MOTOR_ROWS = [["MBON11 approach", "mbon:MBON11:right"], ["MBON05 avoid", "mbon:MBON05:left"], ["PAM dopamine", "dan:pam"], ["PPL1 dopamine", "dan:ppl1"], ["Kenyon cells", "kc"], ["DNa02 left", "dn:DNa02:left"], ["DNa02 right", "dn:DNa02:right"], ["DNp09 forward", "dn:DNp09"], ["landing DNs", "dn:landing"], ["giant fibre", "gf"], ["MN9 feeding", "mn9"], ["power L", "power:left"], ["power R", "power:right"]];
function row(parent, label, cls) { const d = document.createElement("div"); d.className = "row"; d.innerHTML = `<span>${label}</span><div class="bar ${cls}"><i></i></div><span class="num">0.00</span>`; parent.appendChild(d); return { fill: d.querySelector("i"), num: d.querySelector(".num"), label: d.querySelector("span") }; }
const senseBars = SENSE_ROWS.map(([label]) => row($("senses"), label, "")), motorBars = MOTOR_ROWS.map(([label]) => row($("motors"), label, "motor"));
let lastSenses = {};
function hud() {
  if (document.body.classList.contains("instruments")) {
    SENSE_ROWS.forEach(([_, f], k) => { const v = Math.max(0, Math.min(1, f(lastSenses) || 0)); senseBars[k].fill.style.width = (100 * v).toFixed(1) + "%"; senseBars[k].num.textContent = v.toFixed(2); });
    MOTOR_ROWS.forEach(([_, name], k) => { const v = Math.max(0, Math.min(1, life.readouts[name] || 0)); motorBars[k].fill.style.width = (100 * v).toFixed(1) + "%"; motorBars[k].num.textContent = v.toFixed(2); });
  }
  const smell = life.smelled ? `smells the <b>${life.smelled}</b>` : "smells nothing";
  const who = S.pilot === "brain" ? "measured wiring" : S.pilot === "shuffled" ? "shuffled wiring" : "instincts only";
  const paused = S.paused ? ' <span class="sep">·</span> <b>paused</b>' : "";
  $("statusline").innerHTML = `<b>${life.mode}</b> <span class="sep">·</span> ${flight.speed().toFixed(2)} m/s <span class="sep">·</span> ${(100 * flight.p[2]).toFixed(0)} cm up <span class="sep">·</span> ${life.controls.f.toFixed(0)} Hz <span class="sep">·</span> hunger ${life.hunger.toFixed(2)} <span class="sep">·</span> ${smell} <span class="sep">·</span> ${who}${paused}`;
  const ev = life.events.slice(-3).map(([t, x]) => `<b>${t.toFixed(0)} s</b> ${x}`).join(' <span class="sep">·</span> ');
  $("events").innerHTML = ev;
  $("stats").innerHTML = "";
  const info = payloadInfo ? `<b>${payloadInfo.n.toLocaleString()}</b> of ${whole.toLocaleString()} neurons settle live (${(payloadInfo.edges / 1e6).toFixed(2)} M synapse classes) · ${S.brainMs.toFixed(1)} ms per step · ${S.active} active` : "loading the brain";
  $("brainfoot").innerHTML = info + (scan ? ` · ${S.fps.toFixed(0)} fps` : "");
}

// ---- the lessons, in the status strip ----------------------------------------------------------------
const valenceRows = {};
for (const name of ["banana", "bread"]) {
  const span = document.createElement("span"); span.className = "val";
  span.innerHTML = `<span class="who"></span><div class="bar valence"><i></i></div><span class="num" style="color:var(--dim)"></span>`;
  $("valence").appendChild(span);
  valenceRows[name] = { label: span.querySelector(".who"), fill: span.querySelector("i"), num: span.querySelector(".num") };
}
const dopamineBox = document.createElement("span"); dopamineBox.id = "dopamine"; dopamineBox.innerHTML = '<canvas width="240" height="32"></canvas>'; dopamineBox.title = "dopamine per lesson: the temporal-difference error at each search's end";
const lessonsFootEl = document.createElement("span"); lessonsFootEl.id = "lessons-foot"; lessonsFootEl.style.color = "var(--dim)";
$("valence").appendChild(dopamineBox); $("valence").appendChild(lessonsFootEl);
function drawValence() {
  for (const [name, r] of Object.entries(valenceRows)) {
    const p = life.lastP[name];
    r.label.innerHTML = `${name}${life.sugar === name ? ' <span class="sugar">●</span>' : ""}`;
    if (p === null || p === undefined) { r.fill.style.width = "0%"; r.fill.style.left = "50%"; r.num.textContent = "?"; continue; }
    const v = p - 0.5;  // approach to the right of the midline, avoidance to the left
    r.fill.style.left = (50 + 100 * Math.min(0, v)).toFixed(1) + "%"; r.fill.style.width = (100 * Math.abs(v)).toFixed(1) + "%";
    r.fill.style.background = v >= 0 ? "var(--brain)" : "var(--hot)";
    r.num.textContent = (v >= 0 ? "approach " : "avoid ") + (100 * Math.abs(v) * 2).toFixed(0) + "%";
  }
}
function lessonsFoot() {
  const st = S.lessonStats;
  if (!st) { lessonsFootEl.textContent = ""; return; }
  const last = S.lastLesson;
  const who = S.pilot === "instincts" ? "no lessons" : `${(st.plastic || 0).toLocaleString()} plastic${st.changed ? `, <b>${st.changed.toLocaleString()}</b> changed` : ""}`;
  const tail = last ? ` · dopamine <b>${last.delta >= 0 ? "+" : ""}${last.delta.toFixed(2)}</b>` : "";
  lessonsFootEl.innerHTML = who + tail + ` · ${life.visits.sweet} sugar · ${life.visits.punished} blows`;
}
function drawDopamine() {
  const c = dopamineBox.querySelector("canvas"), g = c.getContext("2d"), W = c.width, H = c.height;
  g.clearRect(0, 0, W, H); g.strokeStyle = "rgba(200,255,216,.3)"; g.beginPath(); g.moveTo(0, H / 2); g.lineTo(W, H / 2); g.stroke();
  const n = 40, w = W / n;
  S.lessons.slice(-n).forEach((d, k, arr) => { const x = W - (arr.length - k) * w; g.fillStyle = d >= 0 ? "#7cffa0" : "#ff3b5c"; const h = Math.abs(d) * (H / 2 - 1); g.fillRect(x + 1, d >= 0 ? H / 2 - h : H / 2, w - 2, h); });
}
drawValence(); drawDopamine();

// ---- the model card -------------------------------------------------------------------------
function fillCard() {
  const m = payloadInfo;
  $("card-brain").innerHTML = `<b>The brain.</b> The nervous system of an adult female <i>Drosophila melanogaster</i>, brain and nerve cord wired as measured (BANC release 888: ${whole.toLocaleString()} neurons, 1,861,418 synapse classes at five or more synapses), as one Cadence brain with the library's graded rate model at one global gain selected on two physiology facts. The page draws every neuron at its soma position and settles the ${m.n.toLocaleString()}-neuron sub-net recruited from the flight, looming, odour, taste and mushroom-body populations, one step per ${BRAIN_MS} ms of simulated time; the sub-net's settled populations agree with the whole brain's under the page's stimuli to a readout deviation below 1e-3 (receipts/subnet_closure.json).`;
  $("card-body").innerHTML = `<b>The body.</b> A rigid body with stroke-averaged aerodynamics, unstable in pitch open loop as the animal is; its wingbeat-timescale equilibrium reflex is a hand-written inner loop, supplied like the worm's undulation, because haltere afferents encode rotation in spike timing that a rate model cannot carry. The browser body is bit-identical to the Python reference over 71,000 steps.`;
  $("card-senses").innerHTML = `<b>Senses and muscles.</b> Halteres as a flight tone, the two ocelli by attitude, HS and VS by rotational flow, the antennae by airspeed, the banana's and the bread's odours on the receptor neurons of their classes (decaying fruit, yeast) with a 5 cm bilateral baseline, sugar on the labellar and leg receptors when standing on the sweet fruit, leg touch when landed, the PAM and PPL1 dopaminergic neurons at a reward and at a blow. The descending neurons DNa02 (turns), DNp09 (forward flight), DNp07 and DNp10 (landing), the giant fibre (escape), MN9 (feeding) and aDN1/2 (grooming) are read as deviations from their level-flight rest and override a hand-written instinct layer (bouts, saccades at 0.4 per second, landings, sitting, grooming); the switch below the room replaces the measured wiring by a shuffled one, or removes the brain.`;
  $("card-claim").innerHTML = `<b>The lessons.</b> Every 0.3 s in a smell the mushroom body decides whether to approach it or to avoid it: the choice is a softmax over two output neurons of declared valence, MBON11 (γ1pedc>α/β, GABAergic, approach) and MBON05 (γ4>γ1γ2, glutamatergic, avoidance), by the transmitter rule of Aso et al. 2014. The turn toward or away from the smell is supplied, as the worm's undulation was. Sugar on the fruit it lands on is a reward of one, an empty fruit nothing, a blow from the hand minus one; the library's actor-critic (eligibility traces on the nudged contrast, dopamine as the temporal-difference error, a critic on the Kenyon cells) moves the ${(S.lessonStats && S.lessonStats.plastic) || "Kenyon-cell-to-MBON"} synapses onto the MBONs, the site of the animal's olfactory memory. The same rule, wiring and constants run the receipted experiment (tools/learn_odour.py) against shuffled, frozen and MLP controls. Gate 2: 7 of 13 held-out physiology facts about the wing steering circuit pass on the measured wiring against 4, 3 and 0 on shuffled wirings. Gate 3: 7 of 14 instinct facts against 2 on each shuffled wiring.`;
  $("card-sources").innerHTML = ["BANC: the Lee lab and the BANC community, release 888 (CC BY).", "Aso et al. 2014 eLife 3:e04580 (MBON valence by transmitter); Perisse et al. 2016; Owald et al. 2015 (the Kenyon-cell-to-MBON synapse as the memory site).", "Burke et al. 2012; Liu et al. 2012 (sugar through PAM neurons); Claridge-Chang et al. 2009; Aso et al. 2010 (punishment through PPL1 neurons).", "Fayyazuddin and Dickinson 1996, 1999; Dickinson 1999 (the haltere and wing steering circuit).", "von Reyn et al. 2014; Klapoetke et al. 2017; Ache et al. 2019 (looming, escape, landing).", "Censi et al. 2013; Muijres et al. 2015; van Breugel and Dickinson 2012, 2014 (the ethogram)."].map((s) => `<li>${s}</li>`).join("");
}

// ---- the fly's view ----------------------------------------------------------------------------------
function drawEye(now) { // the rect in CSS pixels from the bottom-left: three.js applies the pixel ratio itself
  if (S.eyeMain) { eye.render(flight, 0, 0, innerWidth, innerHeight, now); return; }  // the compound eye as the whole screen (key e)
  const box = $("eye"); if (!box) return;
  const r = box.getBoundingClientRect();
  if (r.width < 8 || r.height < 8) return;
  eye.render(flight, Math.round(r.left), Math.round(innerHeight - r.bottom), Math.round(r.width), Math.round(r.height), now);
}
function setEyeMain(on) { S.eyeMain = on; $("eye").style.visibility = on ? "hidden" : ""; $("inset").style.visibility = on ? "hidden" : ""; }

// ---- the second view -----------------------------------------------------------------------------
function drawInset() {
  if (!ROOM.renderInset || !rig.closeup) return;
  const box = $("inset"), r = box.getBoundingClientRect();
  const x = Math.round(r.left), y = Math.round(innerHeight - r.bottom), w = Math.round(r.width), h = Math.round(r.height);  // CSS pixels: three.js applies the pixel ratio
  if (w < 8 || h < 8) return;
  const cam = insetCamera();
  if (cam.aspect !== r.width / r.height) { cam.aspect = r.width / r.height; cam.updateProjectionMatrix(); }
  ROOM.renderInset(renderer, scene, cam, x, y, w, h);
}

// ---- the loop --------------------------------------------------------------------------------
let last = performance.now();
function frame(now) {
  requestAnimationFrame(frame);
  const dtWall = Math.min(0.05, (now - last) / 1000); last = now;
  const useBrain = S.pilot !== "instincts" && !!life.baseline;
  if (S.freeze) { life.clock += dtWall * S.speed; }
  else if (S.ready && !S.paused) {
    const simDt = dtWall * S.speed;
    let n = Math.round(simDt / DT), k = 0;
    while (k < n && k < 120) {
      if (life.step(DT)) { life.decide(useBrain); if (life.odourTick() && useBrain && S.learnReady) life.wantDecision = true; }
      k++;
    }
    S.simTime += k * DT;
    S.brainDue += k * DT * 1000;
    lastSenses = life.senses(null, k * DT || dtWall);
    if (life.pendingReward) queueReward();
    if (!S.pending && (S.brainDue >= BRAIN_MS || life.wantDecision)) {
      const steps = Math.max(1, Math.min(STEPS_PER_MESSAGE, Math.floor(S.brainDue / BRAIN_MS)));
      S.brainDue -= steps * BRAIN_MS; S.pending = true;
      if (life.wantDecision) { life.wantDecision = false; worker.postMessage({ type: "decide", stimuli: lastSenses, steps }); }
      else worker.postMessage({ type: "run", stimuli: lastSenses, steps });
    }
    if (S.brainDue > 10 * BRAIN_MS) S.brainDue = 10 * BRAIN_MS; // a slow worker never builds a backlog
  }
  const cam = mainCamera();
  fly.update(flight, life.controls, dtWall, room.surfaceZ, cam === rig.camera && rig.mode === "room", life.pose());
  rig.update(flight, dtWall);
  if (cam.aspect !== innerWidth / innerHeight) { cam.aspect = innerWidth / innerHeight; cam.updateProjectionMatrix(); }
  glitches(dtWall);
  composer.render();
  if (!DEBUG.noviews) { drawInset(); drawEye(now); }
  if (scan) {
    if (S.newState && now - S.lastScan > 33) { scan.step(activation, { draw: false }); S.newState = false; S.lastScan = now; }
    if (DEBUG.scanrate <= 1 || (S.frames % DEBUG.scanrate) === 0) scan.draw(now);
  }
  S.frames++; S.fpsClock += dtWall;
  if (S.fpsClock >= 1) { S.fps = S.frames / S.fpsClock; S.frames = 0; S.fpsClock = 0; window.__fps = S.fps; }
  if ((S.frames & 3) === 0) { hud(); if ((S.frames & 31) === 0) lessonsFoot(); }
  window.__frames = (window.__frames || 0) + 1;
}
requestAnimationFrame(frame);
// the social card: nothing but the fly in close-up (index.html?card=1), shot by tools/card.py
if (params.get("card")) {
  for (const id of ["panel", "status", "toolbar", "hint", "brain", "eye", "inset", "title", "credit", "card"]) { const el = $(id); if (el) el.style.display = "none"; }
  document.body.classList.add("card");
  S.swapViews = true; if (renderPass) renderPass.camera = mainCamera();
  if (rig.cu) rig.cu.d = Number(params.get("d") || 0.0065);
  if (params.get("card") === "sit") { // the fly standing on the banana, the clock running for its fidgets, the physics frozen
    S.freeze = true;
    const B = life.fruits.banana.pos; flight.p = [B[0], B[1], B[2] + 0.0012]; flight.v = [0, 0, 0]; flight.w = [0, 0, 0];
    const yaw = Number(params.get("yaw") || 0.6); flight.q = [Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)];
    life.mode = params.get("pose") || "grooming"; life.groomTarget = "head"; life.episode = 1e9; life.controls = { aL: 0, aR: 0, betaL: 0, betaR: 0, sL: 0, sR: 0, f: 0 };
  }
}
window.__app = { S, life: () => life, flight: () => flight, scan: () => scan, setCam, setPilot, setSugar, setEyeMain, worker, eye, renderer, rig, room };
